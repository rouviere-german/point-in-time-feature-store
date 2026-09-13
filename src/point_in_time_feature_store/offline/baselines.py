"""Deliberately wrong joins, kept to demonstrate and test the failures the store prevents.

Never use these for training data. They exist for the leakage experiment and the mutation test:

- :func:`naive_latest_join` takes each entity's latest row in a data extract (everything ingested
  by ``snapshot_us``) and attaches it to every label row, ignoring label time. This is the classic
  "join the current feature table onto historical labels" bug.
- :func:`event_time_only_asof` is a uni-temporal ASOF join on ``event_time``: correct for event
  time and TTL but blind to ingestion time, so a value that arrived after label time still leaks.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from point_in_time_feature_store.offline.retrieval import SPINE_ROW, DuckDBEngine
from point_in_time_feature_store.sqlbuild import quote_ident

__all__ = ["event_time_only_asof", "naive_latest_join"]


def _order(feature_names: Sequence[str]) -> str:
    return ", ".join(
        ['"event_time" DESC', '"ingested_at" DESC']
        + [quote_ident(name) + " DESC NULLS LAST" for name in feature_names]
    )


def _selected(feature_names: Sequence[str], prefix: str) -> str:
    columns = [
        prefix + '."event_time" AS "__event_time"',
        prefix + '."ingested_at" AS "__ingested_at"',
    ] + [prefix + "." + quote_ident(name) + " AS " + quote_ident(name) for name in feature_names]
    return ", ".join(columns)


def naive_latest_join(
    engine: DuckDBEngine,
    spine: pl.DataFrame,
    rows: pl.DataFrame,
    feature_names: Sequence[str],
    snapshot_us: int,
) -> pl.DataFrame:
    sql = (
        'WITH latest AS (SELECT * FROM feature_rows WHERE "ingested_at" <= $1'  # noqa: S608  # nosec B608
        ' AND "event_time" <= $1 QUALIFY row_number() OVER (PARTITION BY "entity_id" ORDER BY '
        + _order(feature_names)
        + ')= 1) SELECT s."__row" AS "__row", '
        + _selected(feature_names, "l")
        + ' FROM spine_input AS s LEFT JOIN latest AS l ON s."entity_id" = l."entity_id"'
    )
    matched = engine.run_readonly(sql, [snapshot_us], {"spine_input": spine, "feature_rows": rows})
    return spine.select(SPINE_ROW).join(matched, on=SPINE_ROW, how="left").sort(SPINE_ROW)


def event_time_only_asof(
    engine: DuckDBEngine,
    spine: pl.DataFrame,
    rows: pl.DataFrame,
    feature_names: Sequence[str],
    ttl_us: int | None,
) -> pl.DataFrame:
    ttl_clause = ' AND f."event_time" > s."label_time" - $1' if ttl_us is not None else ""
    order = ", ".join(
        ['f."event_time" DESC', 'f."ingested_at" DESC']
        + ["f." + quote_ident(name) + " DESC NULLS LAST" for name in feature_names]
    )
    sql = (
        'SELECT s."__row" AS "__row", '  # noqa: S608  # nosec B608
        + _selected(feature_names, "f")
        + ' FROM spine_input AS s JOIN feature_rows AS f ON f."entity_id" = s."entity_id"'
        ' AND f."event_time" <= s."label_time"'
        + ttl_clause
        + ' QUALIFY row_number() OVER (PARTITION BY s."__row" ORDER BY '
        + order
        + ") = 1"
    )
    params = [ttl_us] if ttl_us is not None else []
    matched = engine.run_readonly(sql, params, {"spine_input": spine, "feature_rows": rows})
    return spine.select(SPINE_ROW).join(matched, on=SPINE_ROW, how="left").sort(SPINE_ROW)
