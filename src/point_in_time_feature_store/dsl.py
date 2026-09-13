"""Row-level feature transformations as a closed, declarative expression tree.

Feature authors never write SQL or Python callables. A transformation is a JSON tree whose node
types come from a fixed allowlist (:data:`ALLOWED_OPS`); it compiles to a Polars expression. The
tree has no aggregation, window, shift or user-function nodes, so a feature value can depend only
on the source row it was computed from and can never look at other (possibly future) rows.

Limits: at most :data:`MAX_NODES` nodes and :data:`MAX_DEPTH` levels, finite float literals and
short string literals. Substring matching is literal (no regular expressions), so a definition
cannot introduce catastrophic backtracking.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal, get_args

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

__all__ = [
    "ALLOWED_OPS",
    "MAX_DEPTH",
    "MAX_NODES",
    "POLARS_DTYPES",
    "Binary",
    "Cast",
    "Clip",
    "Col",
    "DType",
    "E",
    "Expr",
    "ExpressionError",
    "FillNull",
    "Lit",
    "Round",
    "StrContains",
    "StrUnary",
    "Unary",
    "When",
    "col",
    "compile_expression",
    "infer_dtype",
    "iter_nodes",
    "lit",
    "parse_expression",
    "referenced_columns",
    "validate_expression",
    "when",
]

MAX_NODES = 256
MAX_DEPTH = 32
MAX_STRING_LITERAL = 256
_IDENT = r"^[a-z][a-z0-9_]{0,62}$"


class ExpressionError(ValueError):
    """A transformation is malformed, uses a disallowed node or does not type-check."""


class DType(StrEnum):
    INT64 = "int64"
    FLOAT64 = "float64"
    STRING = "string"
    BOOL = "bool"


POLARS_DTYPES: Mapping[DType, pl.DataType] = {
    DType.INT64: pl.Int64(),
    DType.FLOAT64: pl.Float64(),
    DType.STRING: pl.String(),
    DType.BOOL: pl.Boolean(),
}


class _Node(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Col(_Node):
    op: Literal["col"]
    name: str = Field(pattern=_IDENT)


class Lit(_Node):
    op: Literal["lit"]
    value: bool | int | float | str | None

    @field_validator("value")
    @classmethod
    def _bounded(cls, value: bool | int | float | str | None) -> bool | int | float | str | None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("float literals must be finite")
        if isinstance(value, str) and len(value) > MAX_STRING_LITERAL:
            raise ValueError("string literal too long")
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 2**63:
            raise ValueError("integer literal out of int64 range")
        return value


class Binary(_Node):
    op: Literal["add", "sub", "mul", "div", "eq", "ne", "lt", "le", "gt", "ge", "and", "or"]
    left: Expr
    right: Expr


class Unary(_Node):
    op: Literal["neg", "abs", "not", "is_null", "is_not_null", "log1p", "sqrt", "floor", "ceil"]
    arg: Expr


class StrUnary(_Node):
    op: Literal["str_len", "str_lower", "str_upper"]
    arg: Expr


class StrContains(_Node):
    op: Literal["str_contains"]
    arg: Expr
    literal: str = Field(min_length=1, max_length=128)


class Cast(_Node):
    op: Literal["cast"]
    arg: Expr
    dtype: DType


class FillNull(_Node):
    op: Literal["fill_null"]
    arg: Expr
    value: Expr


class Clip(_Node):
    op: Literal["clip"]
    arg: Expr
    lower: float | None = None
    upper: float | None = None

    @field_validator("lower", "upper")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("clip bounds must be finite")
        return value


class Round(_Node):
    op: Literal["round"]
    arg: Expr
    decimals: int = Field(ge=0, le=12)


class When(_Node):
    op: Literal["when"]
    condition: Expr
    then: Expr
    otherwise: Expr


Expr = Annotated[
    Col | Lit | Binary | Unary | StrUnary | StrContains | Cast | FillNull | Clip | Round | When,
    Field(discriminator="op"),
]

_NODE_TYPES: tuple[type[_Node], ...] = (
    Col,
    Lit,
    Binary,
    Unary,
    StrUnary,
    StrContains,
    Cast,
    FillNull,
    Clip,
    Round,
    When,
)
for _model in _NODE_TYPES:
    _model.model_rebuild()


def _ops_of(model: type[_Node]) -> tuple[str, ...]:
    annotation = model.model_fields["op"].annotation
    return tuple(str(item) for item in get_args(annotation))


ALLOWED_OPS: frozenset[str] = frozenset(op for model in _NODE_TYPES for op in _ops_of(model))
_ADAPTER: TypeAdapter[Expr] = TypeAdapter(Expr)


def parse_expression(data: Any) -> Expr:
    """Validate a JSON-like tree. Error messages name node types and paths, never literal values."""
    try:
        expr = _ADAPTER.validate_python(data)
    except ValidationError as exc:
        locations = sorted(
            {".".join(str(part) for part in err["loc"]) or "$" for err in exc.errors()}
        )
        raise ExpressionError("invalid transformation at: " + ", ".join(locations[:5])) from None
    check_limits(expr)
    return expr


def _children(node: _Node) -> tuple[Expr, ...]:
    if isinstance(node, Binary):
        return (node.left, node.right)
    if isinstance(node, Unary | StrUnary | StrContains | Cast | Clip | Round):
        return (node.arg,)
    if isinstance(node, FillNull):
        return (node.arg, node.value)
    if isinstance(node, When):
        return (node.condition, node.then, node.otherwise)
    return ()


def iter_nodes(expr: Expr) -> Iterator[tuple[_Node, int]]:
    """Depth-first (node, depth) pairs."""
    stack: list[tuple[_Node, int]] = [(expr, 1)]
    while stack:
        node, depth = stack.pop()
        yield node, depth
        stack.extend((child, depth + 1) for child in reversed(_children(node)))


def check_limits(expr: Expr) -> None:
    for count, (node, depth) in enumerate(iter_nodes(expr), start=1):
        if count > MAX_NODES:
            raise ExpressionError(f"transformation has more than {MAX_NODES} nodes")
        if depth > MAX_DEPTH:
            raise ExpressionError(f"transformation is deeper than {MAX_DEPTH} levels")
        op = str(getattr(node, "op", ""))
        if op not in ALLOWED_OPS:  # unreachable through pydantic; kept as a second gate
            raise ExpressionError(f"node type {op!r} is not allowed")


def referenced_columns(expr: Expr) -> frozenset[str]:
    return frozenset(node.name for node, _ in iter_nodes(expr) if isinstance(node, Col))


_BINARY: dict[str, Callable[[pl.Expr, pl.Expr], pl.Expr]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
}
_UNARY: dict[str, Callable[[pl.Expr], pl.Expr]] = {
    "neg": lambda a: -a,
    "abs": lambda a: a.abs(),
    "not": lambda a: ~a,
    "is_null": lambda a: a.is_null(),
    "is_not_null": lambda a: a.is_not_null(),
    "log1p": lambda a: a.log1p(),
    "sqrt": lambda a: a.sqrt(),
    "floor": lambda a: a.floor(),
    "ceil": lambda a: a.ceil(),
}


def _literal(value: bool | int | float | str | None) -> pl.Expr:
    if value is None:
        return pl.lit(None)
    if isinstance(value, bool):
        return pl.lit(value, dtype=pl.Boolean)
    if isinstance(value, int):
        return pl.lit(value, dtype=pl.Int64)
    if isinstance(value, float):
        return pl.lit(value, dtype=pl.Float64)
    return pl.lit(value, dtype=pl.String)


def compile_expression(expr: Expr) -> pl.Expr:  # noqa: PLR0911 - one branch per node type
    """Compile a validated tree to a Polars expression."""
    if isinstance(expr, Col):
        return pl.col(expr.name)
    if isinstance(expr, Lit):
        return _literal(expr.value)
    if isinstance(expr, Binary):
        return _BINARY[expr.op](compile_expression(expr.left), compile_expression(expr.right))
    if isinstance(expr, Unary):
        return _UNARY[expr.op](compile_expression(expr.arg))
    if isinstance(expr, StrUnary):
        inner = compile_expression(expr.arg)
        if expr.op == "str_len":
            return inner.str.len_chars().cast(pl.Int64)
        if expr.op == "str_lower":
            return inner.str.to_lowercase()
        return inner.str.to_uppercase()
    if isinstance(expr, StrContains):
        return compile_expression(expr.arg).str.contains(expr.literal, literal=True)
    if isinstance(expr, Cast):
        return compile_expression(expr.arg).cast(POLARS_DTYPES[expr.dtype], strict=True)
    if isinstance(expr, FillNull):
        return compile_expression(expr.arg).fill_null(compile_expression(expr.value))
    if isinstance(expr, Clip):
        return compile_expression(expr.arg).clip(expr.lower, expr.upper)
    if isinstance(expr, Round):
        return compile_expression(expr.arg).round(expr.decimals)
    return (
        pl.when(compile_expression(expr.condition))
        .then(compile_expression(expr.then))
        .otherwise(compile_expression(expr.otherwise))
    )


def _dtype_of(polars_dtype: pl.DataType) -> DType | None:
    for dtype, candidate in POLARS_DTYPES.items():
        if polars_dtype == candidate:
            return dtype
    return None


def infer_dtype(expr: Expr, schema: Mapping[str, DType]) -> DType:
    """Output type of ``expr`` over a source with ``schema`` (evaluated on an empty frame)."""
    check_limits(expr)
    missing = sorted(referenced_columns(expr) - set(schema))
    if missing:
        raise ExpressionError(f"transformation references unknown columns: {', '.join(missing)}")
    empty = pl.DataFrame(schema={name: POLARS_DTYPES[dtype] for name, dtype in schema.items()})
    try:
        output = empty.select(compile_expression(expr).alias("__out")).schema["__out"]
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise ExpressionError(
            f"transformation does not type-check ({type(exc).__name__})"
        ) from None
    dtype = _dtype_of(output)
    if dtype is None:
        raise ExpressionError(f"transformation produces unsupported type {output}")
    return dtype


def validate_expression(expr: Expr, schema: Mapping[str, DType], declared: DType) -> None:
    inferred = infer_dtype(expr, schema)
    if inferred != declared:
        raise ExpressionError(
            f"transformation produces {inferred.value}, declared {declared.value}"
        )


class E:
    """Small builder so Python definitions read like expressions: ``col("amount").log1p()``."""

    __slots__ = ("node",)
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, node: Expr) -> None:
        self.node = node

    @staticmethod
    def _wrap(value: E | bool | int | float | str | None) -> Expr:
        return value.node if isinstance(value, E) else Lit(op="lit", value=value)

    def _binary(self, op: str, other: E | bool | int | float | str | None) -> E:
        return E(Binary(op=op, left=self.node, right=self._wrap(other)))  # type: ignore[arg-type]

    def __add__(self, other: E | int | float) -> E:
        return self._binary("add", other)

    def __sub__(self, other: E | int | float) -> E:
        return self._binary("sub", other)

    def __mul__(self, other: E | int | float) -> E:
        return self._binary("mul", other)

    def __truediv__(self, other: E | int | float) -> E:
        return self._binary("div", other)

    def __eq__(self, other: object) -> E:  # type: ignore[override]
        return self._binary("eq", other)  # type: ignore[arg-type]

    def __ne__(self, other: object) -> E:  # type: ignore[override]
        return self._binary("ne", other)  # type: ignore[arg-type]

    def __lt__(self, other: E | int | float | str) -> E:
        return self._binary("lt", other)

    def __le__(self, other: E | int | float | str) -> E:
        return self._binary("le", other)

    def __gt__(self, other: E | int | float | str) -> E:
        return self._binary("gt", other)

    def __ge__(self, other: E | int | float | str) -> E:
        return self._binary("ge", other)

    def __and__(self, other: E | bool) -> E:
        return self._binary("and", other)

    def __or__(self, other: E | bool) -> E:
        return self._binary("or", other)

    def _unary(self, op: str) -> E:
        return E(Unary(op=op, arg=self.node))  # type: ignore[arg-type]

    def abs(self) -> E:
        return self._unary("abs")

    def log1p(self) -> E:
        return self._unary("log1p")

    def sqrt(self) -> E:
        return self._unary("sqrt")

    def is_null(self) -> E:
        return self._unary("is_null")

    def is_not_null(self) -> E:
        return self._unary("is_not_null")

    def str_len(self) -> E:
        return E(StrUnary(op="str_len", arg=self.node))

    def str_lower(self) -> E:
        return E(StrUnary(op="str_lower", arg=self.node))

    def contains(self, literal: str) -> E:
        return E(StrContains(op="str_contains", arg=self.node, literal=literal))

    def cast(self, dtype: DType) -> E:
        return E(Cast(op="cast", arg=self.node, dtype=dtype))

    def fill_null(self, value: E | bool | int | float | str) -> E:
        return E(FillNull(op="fill_null", arg=self.node, value=self._wrap(value)))

    def clip(self, lower: float | None = None, upper: float | None = None) -> E:
        return E(Clip(op="clip", arg=self.node, lower=lower, upper=upper))

    def round(self, decimals: int) -> E:
        return E(Round(op="round", arg=self.node, decimals=decimals))


def col(name: str) -> E:
    return E(Col(op="col", name=name))


def lit(value: bool | int | float | str | None) -> E:
    return E(Lit(op="lit", value=value))


def when(
    condition: E, then: E | bool | int | float | str, otherwise: E | bool | int | float | str
) -> E:
    return E(
        When(op="when", condition=condition.node, then=E._wrap(then), otherwise=E._wrap(otherwise))
    )
