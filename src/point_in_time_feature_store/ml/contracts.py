"""Data contracts for bitemporal sources (pandera, Polars backend) with PII-safe failure reports.

Every source frame must carry ``entity_id`` (matching the entity's key preset), ``event_time`` and
``ingested_at`` (``Datetime("us", "UTC")``, no nulls), plus exactly the declared value columns
with their types, nullability, bounds and allowed values. Frame-level rules:

- ``clock_skew``: ``event_time <= ingested_at + max_clock_skew`` (rows known before they happen
  beyond the tolerated skew are clock errors);
- ``max_backfill``: ``event_time >= ingested_at - max_backfill_days``;
- ``knowledge_time_append_only`` (optional): ``ingested_at > min_ingested_at``. Once a view has
  been materialized up to a watermark, rows claiming to have been known earlier would be invisible
  to the online store but visible offline, i.e. silent skew; they are rejected instead.

Failure summaries carry source, column, check and count only; failing values are never copied
into the exception because they are often exactly the personal data being validated.
"""

from __future__ import annotations

from typing import Any

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError, SchemaErrors

from point_in_time_feature_store.definitions import EntityDefinition, SourceDefinition
from point_in_time_feature_store.dsl import POLARS_DTYPES, DType
from point_in_time_feature_store.timeutil import MAX_US, MIN_US, US_PER_DAY, US_PER_SECOND

__all__ = [
    "TIMESTAMP_DTYPE",
    "DataContractError",
    "build_source_schema",
    "validate_source_frame",
]

TIMESTAMP_DTYPE = pl.Datetime("us", "UTC")


class DataContractError(ValueError):
    """A frame violated its contract. ``failures`` holds (source, column, check, count)."""

    def __init__(self, source: str, failures: list[dict[str, str | int]]) -> None:
        total = sum(int(item["count"]) for item in failures)
        super().__init__(
            f"data contract for source {source!r} violated: "
            f"{total} failure(s) across {len(failures)} check(s)"
        )
        self.source = source
        self.failures = failures


def _us(column: str) -> pl.Expr:
    return pl.col(column).dt.epoch("us")


def _frame_check(expression: pl.Expr, name: str) -> pa.Check:
    def check(data: Any) -> pl.LazyFrame:
        return data.lazyframe.select(expression.fill_null(False).alias(name))  # type: ignore[no-any-return]

    return pa.Check(check, name=name, error=name)


def _not_nan(column: str) -> pa.Check:
    def check(data: Any) -> pl.LazyFrame:
        return data.lazyframe.select(pl.col(column).is_nan().fill_null(False).not_())  # type: ignore[no-any-return]

    return pa.Check(check, name="not_nan", error="not_nan")


def build_source_schema(
    source: SourceDefinition,
    entity: EntityDefinition,
    *,
    min_ingested_at_us: int | None = None,
) -> pa.DataFrameSchema:
    columns: dict[str, pa.Column] = {
        "entity_id": pa.Column(
            pl.String,
            checks=[pa.Check.str_matches(entity.key_regex)],
            nullable=False,
        ),
        "event_time": pa.Column(TIMESTAMP_DTYPE, nullable=False),
        "ingested_at": pa.Column(TIMESTAMP_DTYPE, nullable=False),
    }
    for column in source.columns:
        checks: list[pa.Check] = []
        if column.ge is not None:
            checks.append(pa.Check.ge(column.ge))
        if column.le is not None:
            checks.append(pa.Check.le(column.le))
        if column.allowed_values is not None:
            checks.append(pa.Check.isin(list(column.allowed_values)))
        if column.dtype is DType.FLOAT64:
            checks.append(_not_nan(column.name))
        columns[column.name] = pa.Column(
            POLARS_DTYPES[column.dtype], checks=checks, nullable=column.nullable
        )

    skew_us = source.max_clock_skew_seconds * US_PER_SECOND
    backfill_us = source.max_backfill_days * US_PER_DAY
    frame_checks = [
        _frame_check(
            _us("event_time").is_between(MIN_US, MAX_US)
            & _us("ingested_at").is_between(MIN_US, MAX_US),
            "timestamp_range",
        ),
        _frame_check(_us("event_time") <= _us("ingested_at") + skew_us, "clock_skew"),
        _frame_check(_us("event_time") >= _us("ingested_at") - backfill_us, "max_backfill"),
    ]
    if min_ingested_at_us is not None:
        frame_checks.append(
            _frame_check(_us("ingested_at") > min_ingested_at_us, "knowledge_time_append_only")
        )
    return pa.DataFrameSchema(columns, checks=frame_checks, strict=True, name=source.name)


def _summarize(source: str, failure_cases: pl.DataFrame) -> list[dict[str, str | int]]:
    frame = failure_cases.select(
        pl.when(pl.col("column").cast(pl.String) == source)
        .then(pl.lit("*"))
        .otherwise(pl.col("column").cast(pl.String))
        .fill_null("*")
        .alias("column"),
        pl.col("check").cast(pl.String).fill_null("unknown"),
    )
    counts = frame.group_by(["column", "check"]).len().sort(["column", "check"])
    return [
        {"source": source, "column": str(row[0]), "check": str(row[1]), "count": int(row[2])}
        for row in counts.iter_rows()
    ]


def validate_source_frame(
    source: SourceDefinition,
    entity: EntityDefinition,
    frame: pl.DataFrame,
    *,
    min_ingested_at_us: int | None = None,
) -> pl.DataFrame:
    """Validate lazily (all failures at once) and raise a summarized :class:`DataContractError`."""
    schema = build_source_schema(source, entity, min_ingested_at_us=min_ingested_at_us)
    try:
        return schema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        raise DataContractError(source.name, _summarize(source.name, exc.failure_cases)) from None
    except SchemaError as exc:  # pragma: no cover - lazy=True reports SchemaErrors
        column = str(getattr(exc.schema, "name", "*"))
        raise DataContractError(
            source.name, [{"source": source.name, "column": column, "check": "schema", "count": 1}]
        ) from None
