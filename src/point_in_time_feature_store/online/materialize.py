"""Watermarked, idempotent materialization from the offline store into Redis.

A row becomes visible to point-in-time queries at ``visible_at = max(event_time, ingested_at)``.
Run *k* reads the rows with ``visible_at`` in ``(watermark_{k-1} - overlap, up_to]``, validates
them against the source contract (a failure blocks the run and leaves the watermark unchanged),
drops tombstoned entities, computes the view's features, keeps the greatest row per entity under
the retrieval ordering and writes it with the conditional-write script. By induction the online
row of every entity equals the point-in-time answer as of the watermark, which is what the skew
checker verifies.

Two-phase watermark: ``inflight`` is set before the writes and ``watermark`` committed after, so a
reader that overlaps a run knows each entity is either at the old or at the new watermark.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl
from opentelemetry.trace import Tracer

from point_in_time_feature_store.hashing import EntityHasher
from point_in_time_feature_store.ml.contracts import DataContractError, validate_source_frame
from point_in_time_feature_store.monitoring import FeatureStoreMetrics
from point_in_time_feature_store.offline.features import collect_feature_rows
from point_in_time_feature_store.offline.retrieval import DuckDBEngine
from point_in_time_feature_store.offline.store import OfflineStore
from point_in_time_feature_store.online.store import OnlineRow, OnlineStore
from point_in_time_feature_store.registry import FeatureRegistry
from point_in_time_feature_store.timeutil import US_PER_SECOND, Clock, system_clock_us

__all__ = ["MaterializationReport", "Materializer"]


@dataclass(frozen=True, slots=True)
class MaterializationReport:
    view: str
    version: str
    previous_watermark_us: int | None
    watermark_us: int
    rows_scanned: int
    entities_dropped_tombstoned: int
    written: int
    skipped: int
    ties_resolved: int

    @property
    def noop(self) -> bool:
        return (
            self.previous_watermark_us is not None
            and self.previous_watermark_us >= self.watermark_us
        )


class Materializer:
    def __init__(
        self,
        registry: FeatureRegistry,
        store: OfflineStore,
        online: OnlineStore,
        engine: DuckDBEngine,
        hasher: EntityHasher,
        *,
        metrics: FeatureStoreMetrics | None = None,
        clock: Clock = system_clock_us,
        tracer: Tracer | None = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self.online = online
        self.engine = engine
        self.hasher = hasher
        self.metrics = metrics
        self.clock = clock
        self.tracer = tracer

    def materialize(
        self,
        view_name: str,
        *,
        up_to_us: int | None = None,
        overlap_us: int = 0,
        before_commit: Callable[[], None] | None = None,
    ) -> MaterializationReport:
        """Run one materialization; traced as ``feature_store.materialize`` when a tracer is set."""
        if self.tracer is None:
            return self._materialize(
                view_name, up_to_us=up_to_us, overlap_us=overlap_us, before_commit=before_commit
            )
        with self.tracer.start_as_current_span("feature_store.materialize") as span:
            report = self._materialize(
                view_name, up_to_us=up_to_us, overlap_us=overlap_us, before_commit=before_commit
            )
            span.set_attribute("feature_store.view", report.view)
            span.set_attribute("feature_store.version", report.version[:16])
            span.set_attribute("feature_store.watermark_us", report.watermark_us)
            span.set_attribute("feature_store.rows_scanned", report.rows_scanned)
            span.set_attribute("feature_store.written", report.written)
            return report

    def _materialize(
        self,
        view_name: str,
        *,
        up_to_us: int | None,
        overlap_us: int,
        before_commit: Callable[[], None] | None,
    ) -> MaterializationReport:
        now = self.clock()
        target = now if up_to_us is None else up_to_us
        if target > now:
            raise ValueError("cannot materialize beyond the current time")
        if overlap_us < 0:
            raise ValueError("overlap must not be negative")
        repo = self.registry.active_repo()
        view = repo.view(view_name)
        version = view.version_hash()
        source = repo.source(view.source)
        entity = repo.entity(view.entity)
        previous = self.registry.get_watermark(view.name, version)
        if previous is not None and target <= previous:
            return MaterializationReport(view.name, version, previous, previous, 0, 0, 0, 0, 0)

        visible = pl.max_horizontal(
            pl.col("event_time").dt.epoch("us"), pl.col("ingested_at").dt.epoch("us")
        )
        window = visible <= target
        if previous is not None:
            window = window & (visible > previous - overlap_us)
        batch = self.store.scan(source).filter(window).collect()
        try:
            validate_source_frame(source, entity, batch)
        except DataContractError as exc:
            if self.metrics is not None:
                for failure in exc.failures:
                    self.metrics.contract_failures_total.labels(
                        source.name, str(failure["check"])
                    ).inc(int(failure["count"]))
            raise

        tombstones = self.registry.tombstones()
        dropped = 0
        if tombstones and batch.height:
            erased = [
                entity_id
                for entity_id in batch["entity_id"].unique().to_list()
                if self.hasher.entity(entity_id) in tombstones
            ]
            if erased:
                before = batch.height
                batch = batch.filter(~pl.col("entity_id").is_in(erased))
                dropped = before - batch.height

        rows = collect_feature_rows(view, batch.lazy())
        winners = self.engine.latest_per_entity(rows, view.feature_names) if rows.height else rows
        online_rows = [
            OnlineRow(
                entity_id=str(record[0]),
                event_time_us=int(record[1]),
                ingested_at_us=int(record[2]),
                values=tuple(record[3:]),
            )
            for record in winners.select(
                "entity_id", "event_time", "ingested_at", *view.feature_names
            ).iter_rows()
        ]
        self.online.begin_materialization(view.name, version, target)
        stats = self.online.write_rows(view, version, online_rows)
        if before_commit is not None:
            before_commit()
        self.online.commit_materialization(view.name, version, target)
        self.registry.set_watermark(view.name, version, target)
        if self.metrics is not None:
            self.metrics.materialization_lag_seconds.labels(view.name).set(
                max(0.0, (self.clock() - target) / US_PER_SECOND)
            )
        return MaterializationReport(
            view=view.name,
            version=version,
            previous_watermark_us=previous,
            watermark_us=target,
            rows_scanned=batch.height + dropped,
            entities_dropped_tombstoned=dropped,
            written=stats.written,
            skipped=stats.skipped,
            ties_resolved=stats.ties_resolved,
        )
