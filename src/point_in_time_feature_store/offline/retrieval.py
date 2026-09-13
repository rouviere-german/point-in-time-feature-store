"""Point-in-time-correct historical retrieval on DuckDB.

Two physical paths implement the semantics documented in :mod:`offline.oracle` and
``docs/PIT_SEMANTICS.md``:

``range_join`` (always correct)
    Join every spine row to the rows of its entity with ``event_time <= as_of``,
    ``ingested_at <= as_of`` and ``event_time > label_time - ttl``, then keep one row per spine row
    with ``QUALIFY row_number() OVER (PARTITION BY spine row ORDER BY event_time DESC,
    ingested_at DESC, v1 DESC NULLS LAST, ...) = 1``. DuckDB plans the equality on ``entity_id`` as
    a hash join and evaluates the range predicates as join conditions.

``asof`` (fast path, only for views declaring zero ingestion lag)
    When no row was ingested after its event time, "ingested by ``as_of``" is implied by "happened
    by ``as_of``", so a uni-temporal ``ASOF LEFT JOIN`` on ``event_time`` is exact after
    de-duplicating rows that share ``(entity_id, event_time)`` with the same ordering rule. The
    zero-lag declaration is verified against the data first; a violation raises
    :class:`DataContractError` instead of silently leaking late-ingested values.

The engine disables DuckDB's external access (no file, HTTP or extension loading from SQL) and
locks its configuration; inputs are registered as in-memory Arrow tables.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb
import polars as pl

from point_in_time_feature_store.definitions import (
    FeatureRepo,
    FeatureView,
    feature_column,
    parse_feature_ref,
)
from point_in_time_feature_store.ml.contracts import DataContractError
from point_in_time_feature_store.offline.features import collect_feature_rows
from point_in_time_feature_store.offline.store import OfflineStore
from point_in_time_feature_store.sqlbuild import join_idents, quote_ident
from point_in_time_feature_store.timeutil import seconds_to_us

__all__ = [
    "SPINE_ROW",
    "DuckDBEngine",
    "HistoricalFeatures",
    "HistoricalRetriever",
    "PathChoice",
    "RetrievalError",
    "ViewRetrieval",
    "normalize_spine",
]

SPINE_ROW = "__row"
PathChoice = Literal["auto", "range_join", "asof"]
# SQL below is concatenated only from constants and identifiers validated by quote_ident;
# every value is a bound parameter (hence the S608/B608 suppressions).
_SPINE_TABLE = "spine_input"
_ROWS_TABLE = "feature_rows"


class RetrievalError(ValueError):
    """The spine or the request is invalid."""


@dataclass(frozen=True, slots=True)
class ViewRetrieval:
    view: str
    version: str
    path: Literal["range_join", "asof"]
    feature_rows: int


@dataclass(frozen=True, slots=True)
class HistoricalFeatures:
    frame: pl.DataFrame
    views: tuple[ViewRetrieval, ...]


def _to_us(frame: pl.DataFrame, column: str) -> pl.Expr:
    dtype = frame.schema[column]
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is None:
            raise RetrievalError(f"spine column {column} must be timezone-aware")
        return pl.col(column).dt.convert_time_zone("UTC").dt.epoch("us")
    if dtype == pl.Int64:
        return pl.col(column)
    raise RetrievalError(f"spine column {column} must be a UTC datetime or int64 microseconds")


def normalize_spine(spine: pl.DataFrame) -> pl.DataFrame:
    """Return ``__row, entity_id, label_time, as_of`` with int64 microsecond times."""
    for column in ("entity_id", "label_time"):
        if column not in spine.columns:
            raise RetrievalError(f"spine is missing column {column}")
    if spine.schema["entity_id"] != pl.String:
        raise RetrievalError("spine column entity_id must be a string")
    has_as_of = "as_of" in spine.columns
    normalized = spine.select(
        pl.int_range(0, pl.len(), dtype=pl.Int64).alias(SPINE_ROW),
        pl.col("entity_id"),
        _to_us(spine, "label_time").alias("label_time"),
        (_to_us(spine, "as_of") if has_as_of else _to_us(spine, "label_time")).alias("as_of"),
    )
    if normalized.select(pl.any_horizontal(pl.all().is_null().any())).item():
        raise RetrievalError("spine must not contain nulls")
    if normalized.filter(pl.col("as_of") > pl.col("label_time")).height:
        raise RetrievalError("as_of must not be later than label_time")
    return normalized


class DuckDBEngine:
    """A locked-down in-memory DuckDB connection. Calls are serialized."""

    def __init__(
        self,
        *,
        threads: int = 2,
        memory_limit: str = "1GB",
        temp_directory: Path | None = None,
    ) -> None:
        config: dict[str, Any] = {"threads": threads, "memory_limit": memory_limit}
        if temp_directory is not None:
            temp_directory.mkdir(parents=True, exist_ok=True)
            config["temp_directory"] = str(temp_directory)
        self._con = duckdb.connect(":memory:", config=config)
        self._con.execute("SET enable_external_access = false")
        self._con.execute("SET autoload_known_extensions = false")
        # Registered Arrow tables carry no cardinality estimate, and DuckDB then falls back from
        # ASOF_JOIN to a nested loop (measured: 122 s instead of 0.14 s for 200k x 190k rows).
        self._con.execute("SET asof_loop_join_threshold = 0")
        self._con.execute("SET lock_configuration = true")
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._con.close()

    def _query(
        self,
        sql: str,
        params: Sequence[object],
        tables: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        with self._lock:
            for name, frame in tables.items():
                self._con.register(name, frame.to_arrow())
            try:
                return self._con.execute(sql, list(params)).pl()
            finally:
                for name in tables:
                    self._con.unregister(name)

    @staticmethod
    def _order_by(feature_names: Sequence[str], prefix: str) -> str:
        parts = [
            quote_ident(prefix) + '."event_time" DESC',
            quote_ident(prefix) + '."ingested_at" DESC',
        ]
        parts += [
            quote_ident(prefix) + "." + quote_ident(name) + " DESC NULLS LAST"
            for name in feature_names
        ]
        return ", ".join(parts)

    @staticmethod
    def _left_join_spine(spine: pl.DataFrame, matched: pl.DataFrame) -> pl.DataFrame:
        return spine.select(SPINE_ROW).join(matched, on=SPINE_ROW, how="left").sort(SPINE_ROW)

    def pit_range_join(
        self,
        spine: pl.DataFrame,
        rows: pl.DataFrame,
        feature_names: Sequence[str],
        ttl_us: int | None,
    ) -> pl.DataFrame:
        """General bitemporal path. ``spine`` comes from :func:`normalize_spine`."""
        ttl_clause = ' AND f."event_time" > s."label_time" - $1' if ttl_us is not None else ""
        sql = (
            'SELECT s."__row" AS "__row", f."event_time" AS "__event_time", '  # noqa: S608  # nosec B608
            'f."ingested_at" AS "__ingested_at"'
            + ("" if not feature_names else ", " + join_idents(feature_names, prefix="f"))
            + " FROM "
            + _SPINE_TABLE
            + " AS s JOIN "
            + _ROWS_TABLE
            + " AS f"
            ' ON f."entity_id" = s."entity_id"'
            ' AND f."event_time" <= s."as_of" AND f."ingested_at" <= s."as_of"'
            + ttl_clause
            + ' QUALIFY row_number() OVER (PARTITION BY s."__row" ORDER BY '
            + self._order_by(feature_names, "f")
            + ") = 1"
        )
        params = [ttl_us] if ttl_us is not None else []
        matched = self._query(sql, params, {_SPINE_TABLE: spine, _ROWS_TABLE: rows})
        return self._left_join_spine(spine, matched)

    def max_ingestion_lag_rows(self, rows: pl.DataFrame) -> int:
        """Number of rows ingested after their event time (they forbid the ASOF path)."""
        if rows.height == 0:
            return 0
        return int(rows.filter(pl.col("ingested_at") > pl.col("event_time")).height)

    def pit_asof(
        self,
        spine: pl.DataFrame,
        rows: pl.DataFrame,
        feature_names: Sequence[str],
        ttl_us: int | None,
        *,
        source: str = "unknown",
    ) -> pl.DataFrame:
        """Uni-temporal fast path; refuses data that violates the zero-lag precondition."""
        late = self.max_ingestion_lag_rows(rows)
        if late:
            raise DataContractError(
                source,
                [
                    {
                        "source": source,
                        "column": "ingested_at",
                        "check": "zero_ingestion_lag",
                        "count": late,
                    }
                ],
            )
        dedup_order = ['"ingested_at" DESC'] + [
            quote_ident(n) + " DESC NULLS LAST" for n in feature_names
        ]
        ttl_ok = '(d."event_time" > s."label_time" - $1)' if ttl_us is not None else "TRUE"
        selected = ['d."event_time"', 'd."ingested_at"'] + [
            "d." + quote_ident(name) for name in feature_names
        ]
        aliases = ['"__event_time"', '"__ingested_at"'] + [
            quote_ident(name) for name in feature_names
        ]
        sql = (
            "WITH dedup AS (SELECT * FROM "  # noqa: S608  # nosec B608
            + _ROWS_TABLE
            + ' QUALIFY row_number() OVER (PARTITION BY "entity_id", "event_time" ORDER BY '
            + ", ".join(dedup_order)
            + ') = 1) SELECT s."__row" AS "__row", '
            + ", ".join(
                "CASE WHEN " + ttl_ok + " THEN " + expr + " END AS " + alias
                for expr, alias in zip(selected, aliases, strict=True)
            )
            + " FROM "
            + _SPINE_TABLE
            + " AS s ASOF LEFT JOIN dedup AS d"
            ' ON s."entity_id" = d."entity_id" AND s."as_of" >= d."event_time"'
        )
        params = [ttl_us] if ttl_us is not None else []
        matched = self._query(sql, params, {_SPINE_TABLE: spine, _ROWS_TABLE: rows})
        return self._left_join_spine(spine, matched)

    def latest_per_entity(self, rows: pl.DataFrame, feature_names: Sequence[str]) -> pl.DataFrame:
        """The greatest row per entity under the retrieval ordering (used by materialization)."""
        sql = (
            "SELECT * FROM " + _ROWS_TABLE + " AS f QUALIFY row_number() OVER "  # noqa: S608  # nosec B608
            '(PARTITION BY f."entity_id" ORDER BY ' + self._order_by(feature_names, "f") + ") = 1"
        )
        return self._query(sql, [], {_ROWS_TABLE: rows}).sort("entity_id")

    def explain_range_join(
        self, spine: pl.DataFrame, rows: pl.DataFrame, feature_names: Sequence[str]
    ) -> list[str]:
        """Join operators DuckDB chose for the range-join path (for docs and the benchmark)."""
        sql = (
            'EXPLAIN SELECT s."__row" FROM ' + _SPINE_TABLE + " AS s JOIN " + _ROWS_TABLE + " AS f"  # noqa: S608  # nosec B608
            ' ON f."entity_id" = s."entity_id" AND f."event_time" <= s."as_of"'
            ' AND f."ingested_at" <= s."as_of" AND f."event_time" > s."label_time" - $1'
            ' QUALIFY row_number() OVER (PARTITION BY s."__row" ORDER BY '
            + self._order_by(feature_names, "f")
            + ") = 1"
        )
        plan = self._query(sql, [1], {_SPINE_TABLE: spine, _ROWS_TABLE: rows})
        text = "\n".join(str(value) for value in plan.row(0))
        return sorted({line.strip(" │┌┐└┘─") for line in text.splitlines() if "JOIN" in line})

    def run_readonly(
        self, sql: str, params: Sequence[object], tables: dict[str, pl.DataFrame]
    ) -> pl.DataFrame:
        """Internal hook for baselines; ``sql`` must be assembled from constants and quote_ident."""
        return self._query(sql, params, tables)


class HistoricalRetriever:
    """``get_historical_features`` over the offline store and the active definitions."""

    def __init__(self, store: OfflineStore, repo: FeatureRepo, engine: DuckDBEngine) -> None:
        self.store = store
        self.repo = repo
        self.engine = engine

    def feature_rows(self, view: FeatureView) -> pl.DataFrame:
        source = self.repo.source(view.source)
        return collect_feature_rows(view, self.store.scan(source))

    def _requested(self, features: Sequence[str]) -> dict[str, list[str]]:
        requested: dict[str, list[str]] = {}
        for ref in features:
            view_name, feature = parse_feature_ref(ref)
            view = self.repo.view(view_name)
            names = list(view.feature_names) if feature == "*" else [view.feature(feature).name]
            bucket = requested.setdefault(view_name, [])
            bucket.extend(name for name in names if name not in bucket)
        if not requested:
            raise RetrievalError("request at least one feature")
        return requested

    def join_view(
        self,
        view: FeatureView,
        spine: pl.DataFrame,
        rows: pl.DataFrame,
        *,
        path: PathChoice = "auto",
    ) -> tuple[pl.DataFrame, Literal["range_join", "asof"]]:
        ttl_us = seconds_to_us(view.ttl_seconds)
        chosen: Literal["range_join", "asof"] = (
            ("asof" if view.zero_ingestion_lag else "range_join") if path == "auto" else path
        )
        if chosen == "asof":
            joined = self.engine.pit_asof(
                spine, rows, view.feature_names, ttl_us, source=view.source
            )
        else:
            joined = self.engine.pit_range_join(spine, rows, view.feature_names, ttl_us)
        return joined, chosen

    def get_historical_features(
        self,
        spine: pl.DataFrame,
        features: Sequence[str],
        *,
        path: PathChoice = "auto",
    ) -> HistoricalFeatures:
        normalized = normalize_spine(spine)
        output = spine
        reports: list[ViewRetrieval] = []
        for view_name, names in self._requested(features).items():
            view = self.repo.view(view_name)
            rows = self.feature_rows(view)
            joined, chosen = self.join_view(view, normalized, rows, path=path)
            output = attach_view_columns(output, view.name, joined, names)
            reports.append(ViewRetrieval(view.name, view.version_hash(), chosen, rows.height))
        return HistoricalFeatures(frame=output, views=tuple(reports))


def attach_view_columns(
    output: pl.DataFrame, view_name: str, joined: pl.DataFrame, names: Sequence[str]
) -> pl.DataFrame:
    """Append ``<view>__event_time``, ``<view>__ingested_at`` and ``<view>__<feature>`` columns."""
    columns = [
        pl.from_epoch(pl.col("__event_time"), time_unit="us")
        .dt.replace_time_zone("UTC")
        .alias(feature_column(view_name, "event_time")),
        pl.from_epoch(pl.col("__ingested_at"), time_unit="us")
        .dt.replace_time_zone("UTC")
        .alias(feature_column(view_name, "ingested_at")),
        *[pl.col(name).alias(feature_column(view_name, name)) for name in names],
    ]
    return output.hstack(joined.select(columns))
