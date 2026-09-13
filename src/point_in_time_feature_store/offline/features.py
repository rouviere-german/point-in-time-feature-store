"""Compute a feature view's rows from its source: one output row per source row.

Output columns: ``entity_id`` (string), ``event_time`` and ``ingested_at`` (int64 microseconds)
and one column per feature in canonical (sorted) order. Float features are normalized so every
store compares them the same way: NaN becomes null and -0.0 becomes 0.0.
"""

from __future__ import annotations

import polars as pl

from point_in_time_feature_store.definitions import FeatureView
from point_in_time_feature_store.dsl import DType, compile_expression

__all__ = ["TransformationError", "compute_feature_rows", "feature_rows_schema"]


class TransformationError(RuntimeError):
    """A transformation failed on real data (for example a strict cast of a malformed value)."""


def _normalized(view: FeatureView, name: str) -> pl.Expr:
    feature = view.feature(name)
    expr = compile_expression(feature.expression)
    if feature.dtype is DType.FLOAT64:
        expr = (
            pl.when(expr.is_nan())
            .then(pl.lit(None, dtype=pl.Float64))
            .when(expr == 0.0)
            .then(pl.lit(0.0, dtype=pl.Float64))
            .otherwise(expr)
        )
    return expr.alias(name)


def compute_feature_rows(view: FeatureView, source_rows: pl.LazyFrame) -> pl.LazyFrame:
    return source_rows.select(
        pl.col("entity_id"),
        pl.col("event_time").dt.epoch("us").alias("event_time"),
        pl.col("ingested_at").dt.epoch("us").alias("ingested_at"),
        *[_normalized(view, name) for name in view.feature_names],
    )


def feature_rows_schema(view: FeatureView) -> dict[str, pl.DataType]:
    from point_in_time_feature_store.dsl import POLARS_DTYPES  # noqa: PLC0415

    schema: dict[str, pl.DataType] = {
        "entity_id": pl.String(),
        "event_time": pl.Int64(),
        "ingested_at": pl.Int64(),
    }
    for name in view.feature_names:
        schema[name] = POLARS_DTYPES[view.feature(name).dtype]
    return schema


def collect_feature_rows(view: FeatureView, source_rows: pl.LazyFrame) -> pl.DataFrame:
    try:
        return compute_feature_rows(view, source_rows).collect()
    except pl.exceptions.PolarsError as exc:
        raise TransformationError(
            f"feature view {view.name}: transformation failed ({type(exc).__name__})"
        ) from None
