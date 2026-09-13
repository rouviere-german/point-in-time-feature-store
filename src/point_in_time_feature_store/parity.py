"""Training-serving skew detection from logged online reads.

For each sampled read ``(entity, request time t, watermark snapshot [low, high])`` the checker
recomputes the view offline with ``label_time = t`` and ``as_of = min(low, t)`` (and ``high`` when a
materialization was in flight), using the same retrieval engine and definitions as training. The
online value matches when the served event time and every feature value equal one of those
recomputations exactly. "Within documented staleness" is therefore precise: online values may lag
the offline store by up to the materialization watermark, and by nothing else.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import polars as pl
from pydantic import JsonValue

from point_in_time_feature_store.hashing import EntityHasher
from point_in_time_feature_store.monitoring import FeatureStoreMetrics
from point_in_time_feature_store.offline.retrieval import (
    SPINE_ROW,
    HistoricalRetriever,
    normalize_spine,
)
from point_in_time_feature_store.serving.request_log import LoggedRead, RequestLog, logged_values

__all__ = ["Mismatch", "SkewChecker", "SkewReport", "ViewParity"]

_NEVER = -(2**62)


@dataclass(frozen=True, slots=True)
class Mismatch:
    request_id: str
    view: str
    fields: tuple[str, ...]


@dataclass(slots=True)
class ViewParity:
    view: str
    checked: int = 0
    mismatches: int = 0
    skipped_version_changed: int = 0

    @property
    def ratio(self) -> float:
        return self.mismatches / self.checked if self.checked else 0.0


@dataclass(slots=True)
class SkewReport:
    views: dict[str, ViewParity] = field(default_factory=dict)
    examples: list[Mismatch] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return sum(item.checked for item in self.views.values())

    @property
    def mismatches(self) -> int:
        return sum(item.mismatches for item in self.views.values())


class SkewChecker:
    def __init__(
        self,
        request_log: RequestLog,
        retriever: HistoricalRetriever,
        hasher: EntityHasher,
        *,
        metrics: FeatureStoreMetrics | None = None,
        max_examples: int = 20,
    ) -> None:
        self.request_log = request_log
        self.retriever = retriever
        self.hasher = hasher
        self.metrics = metrics
        self.max_examples = max_examples

    def _expected(
        self, reads: list[LoggedRead], view_name: str, watermarks: list[int | None]
    ) -> list[tuple[int | None, dict[str, JsonValue]]]:
        view = self.retriever.repo.view(view_name)
        spine = normalize_spine(
            pl.DataFrame(
                {
                    "entity_id": [read.entity_id for read in reads],
                    "label_time": [read.ts_us for read in reads],
                    "as_of": [
                        _NEVER if mark is None else min(mark, read.ts_us)
                        for read, mark in zip(reads, watermarks, strict=True)
                    ],
                },
                schema={"entity_id": pl.String, "label_time": pl.Int64, "as_of": pl.Int64},
            )
        )
        rows = self._rows_cache.get(view_name)
        if rows is None:
            rows = self.retriever.feature_rows(view)
            self._rows_cache[view_name] = rows
        joined, _ = self.retriever.join_view(view, spine, rows)
        expected: list[tuple[int | None, dict[str, JsonValue]]] = []
        for record in joined.drop(SPINE_ROW).iter_rows(named=True):
            event_time = record["__event_time"]
            values = {name: record[name] for name in view.feature_names}
            expected.append((event_time, logged_values(view, values, self.hasher)))
        return expected

    @staticmethod
    def _diff(
        read: LoggedRead, expected: tuple[int | None, dict[str, JsonValue]]
    ) -> tuple[str, ...]:
        served_time = read.event_time_us if read.status == "ok" else None
        fields = [] if served_time == expected[0] else ["__event_time"]
        served = read.values if read.status == "ok" else dict.fromkeys(read.values)
        fields += [name for name in sorted(expected[1]) if served.get(name) != expected[1][name]]
        return tuple(fields)

    def check(self, *, sample_size: int, seed: int = 0) -> SkewReport:
        self._rows_cache: dict[str, pl.DataFrame] = {}
        report = SkewReport()
        by_view: dict[str, list[LoggedRead]] = defaultdict(list)
        active = {view.name: view.version_hash() for view in self.retriever.repo.feature_views}
        for read in self.request_log.sample(sample_size, seed=seed):
            parity = report.views.setdefault(read.view, ViewParity(read.view))
            if active.get(read.view) != read.version:
                parity.skipped_version_changed += 1
                continue
            by_view[read.view].append(read)
        for view_name, reads in sorted(by_view.items()):
            parity = report.views[view_name]
            low = self._expected(reads, view_name, [read.watermark_low_us for read in reads])
            inflight = [
                i for i, read in enumerate(reads) if read.watermark_high_us != read.watermark_low_us
            ]
            high: dict[int, tuple[int | None, dict[str, JsonValue]]] = {}
            if inflight:
                subset = [reads[i] for i in inflight]
                recomputed = self._expected(
                    subset, view_name, [read.watermark_high_us for read in subset]
                )
                high = dict(zip(inflight, recomputed, strict=True))
            for index, read in enumerate(reads):
                parity.checked += 1
                fields = self._diff(read, low[index])
                if fields and index in high and not self._diff(read, high[index]):
                    fields = ()
                if fields:
                    parity.mismatches += 1
                    if len(report.examples) < self.max_examples:
                        report.examples.append(Mismatch(read.request_id, view_name, fields))
        if self.metrics is not None:
            for parity in report.views.values():
                self.metrics.parity_mismatch_ratio.labels(parity.view).set(parity.ratio)
        return report
