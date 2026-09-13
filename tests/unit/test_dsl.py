from __future__ import annotations

import math
from typing import Any

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from point_in_time_feature_store.dsl import (
    ALLOWED_OPS,
    MAX_DEPTH,
    MAX_NODES,
    DType,
    ExpressionError,
    col,
    compile_expression,
    infer_dtype,
    lit,
    parse_expression,
    referenced_columns,
    validate_expression,
    when,
)

pytestmark = pytest.mark.unit

SCHEMA = {"amount": DType.FLOAT64, "count": DType.INT64, "name": DType.STRING, "flag": DType.BOOL}
FRAME = pl.DataFrame(
    {
        "amount": [1.5, None, -2.0],
        "count": [3, 0, None],
        "name": ["Ab", "cD", None],
        "flag": [True, False, None],
    }
)


def evaluate(expr: Any) -> list[Any]:
    return FRAME.select(compile_expression(expr).alias("out"))["out"].to_list()


def test_the_node_set_is_closed() -> None:
    expected = {
        "col", "lit", "add", "sub", "mul", "div", "eq", "ne", "lt", "le", "gt", "ge",
        "and", "or", "neg", "abs", "not", "is_null", "is_not_null", "log1p", "sqrt",
        "floor", "ceil", "str_len", "str_lower", "str_upper", "str_contains", "cast",
        "fill_null", "clip", "round", "when",
    }  # fmt: skip
    assert expected == ALLOWED_OPS


@pytest.mark.parametrize(
    "tree",
    [
        {"op": "map_elements", "arg": {"op": "col", "name": "amount"}, "function": "os.system"},
        {"op": "sql", "query": "SELECT secret_value FROM users"},
        {"op": "col", "name": "amount; DROP TABLE users"},
        {"op": "col", "name": "amount", "alias": "x"},
        {"op": "lit", "value": {"nested": "secret_value"}},
        {"op": "shift", "arg": {"op": "col", "name": "amount"}, "n": -1},
        {"op": "rolling_mean", "arg": {"op": "col", "name": "amount"}, "window": 3},
    ],
)
def test_disallowed_nodes_are_rejected_without_echoing_values(tree: dict[str, Any]) -> None:
    with pytest.raises(ExpressionError) as info:
        parse_expression(tree)
    assert "secret_value" not in str(info.value)
    assert "DROP" not in str(info.value)


def test_trees_deeper_or_larger_than_the_limits_are_rejected() -> None:
    deep = col("amount")
    for _ in range(MAX_DEPTH):
        deep = deep.abs()
    with pytest.raises(ExpressionError, match="deeper"):
        parse_expression(deep.node.model_dump(mode="json"))
    leaves = [col("amount") for _ in range(MAX_NODES // 2 + 2)]
    while len(leaves) > 1:
        paired = [left + right for left, right in zip(leaves[::2], leaves[1::2], strict=False)]
        leaves = paired + ([leaves[-1]] if len(leaves) % 2 else [])
    with pytest.raises(ExpressionError, match="more than"):
        parse_expression(leaves[0].node.model_dump(mode="json"))


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_non_finite_literals_are_rejected(value: float) -> None:
    with pytest.raises(ExpressionError):
        parse_expression({"op": "lit", "value": value})


def test_compiled_expressions_compute_expected_values() -> None:
    assert evaluate((col("amount") * 2 + 1).node) == [4.0, None, -3.0]
    assert evaluate(col("amount").fill_null(0.0).clip(0.0, 1.0).node) == [1.0, 0.0, 0.0]
    assert evaluate(col("name").str_lower().node) == ["ab", "cd", None]
    assert evaluate(col("name").str_len().node) == [2, 2, None]
    assert evaluate(col("name").contains("D").node) == [False, True, None]
    assert evaluate(when(col("count") > 1, "many", "few").node) == ["many", "few", "few"]
    assert evaluate(col("count").cast(DType.FLOAT64).node) == [3.0, 0.0, None]
    assert evaluate((col("flag") & lit(True)).node) == [True, False, None]
    assert evaluate(col("amount").is_null().node) == [False, True, False]
    assert evaluate(col("amount").round(0).node) == [2.0, None, -2.0]


@pytest.mark.parametrize(
    ("expr", "dtype"),
    [
        (col("amount").log1p(), DType.FLOAT64),
        (col("count") + 1, DType.INT64),
        (col("count") / 2, DType.FLOAT64),
        (col("count") >= 2, DType.BOOL),
        (col("name").str_len(), DType.INT64),
        (when(col("flag"), col("name"), "none"), DType.STRING),
    ],
)
def test_output_types_are_inferred(expr: Any, dtype: DType) -> None:
    assert infer_dtype(expr.node, SCHEMA) is dtype


def test_declared_type_and_column_references_are_checked() -> None:
    with pytest.raises(ExpressionError, match="declared"):
        validate_expression(col("amount").node, SCHEMA, DType.INT64)
    with pytest.raises(ExpressionError, match="unknown columns"):
        validate_expression(col("missing").node, SCHEMA, DType.FLOAT64)
    with pytest.raises(ExpressionError, match=r"type-check|unsupported"):
        infer_dtype((col("name") + 1).node, SCHEMA)
    with pytest.raises(ExpressionError, match="unsupported"):
        infer_dtype(lit(None).node, SCHEMA)


_leaves = st.one_of(
    st.sampled_from(["amount", "count"]).map(col),
    st.integers(-5, 5).map(lit),
    st.floats(-10, 10, allow_nan=False).map(lit),
)


def _extend(children: st.SearchStrategy[Any]) -> st.SearchStrategy[Any]:
    return st.one_of(
        st.tuples(children, children).map(lambda pair: pair[0] + pair[1]),
        st.tuples(children, children).map(lambda pair: pair[0] * pair[1]),
        children.map(lambda child: child.abs()),
        st.tuples(children, children).map(lambda pair: when(pair[0] > 0, pair[0], pair[1])),
    )


@settings(max_examples=40)
@given(expr=st.recursive(_leaves, _extend, max_leaves=12))
def test_trees_round_trip_through_json_and_compile_identically(expr: Any) -> None:
    dumped = expr.node.model_dump(mode="json")
    parsed = parse_expression(dumped)
    assert parsed == expr.node
    assert referenced_columns(parsed) <= {"amount", "count"}
    frame = FRAME.select("amount", "count")
    left = frame.select(compile_expression(expr.node).alias("out"))
    right = frame.select(compile_expression(parsed).alias("out"))
    assert left.equals(right)
