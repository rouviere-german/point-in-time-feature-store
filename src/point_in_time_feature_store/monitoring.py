"""Feature-store metrics (Prometheus) and feature drift monitoring (PSI/KS).

Metric names follow the project specification: ``online_get_latency_seconds``,
``feature_freshness_seconds{view}``, ``parity_mismatch_ratio{view}``,
``materialization_lag_seconds{view}``, ``contract_failures_total{source,check}`` and
``feature_psi{view,feature}``. They live on the app's own registry, never the global one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from point_in_time_feature_store.ml.drift import DriftMonitor, DriftReport

__all__ = [
    "FeatureDriftMonitor",
    "FeatureDriftResult",
    "FeatureStoreMetrics",
    "create_feature_store_metrics",
]


@dataclass(frozen=True, slots=True)
class FeatureStoreMetrics:
    registry: CollectorRegistry
    online_get_latency_seconds: Histogram
    feature_freshness_seconds: Histogram
    parity_mismatch_ratio: Gauge
    materialization_lag_seconds: Gauge
    contract_failures_total: Counter
    feature_psi: Gauge


def create_feature_store_metrics(registry: CollectorRegistry | None = None) -> FeatureStoreMetrics:
    reg = registry if registry is not None else CollectorRegistry()
    return FeatureStoreMetrics(
        registry=reg,
        online_get_latency_seconds=Histogram(
            "online_get_latency_seconds",
            "Latency of online feature reads (store round trip and decoding).",
            registry=reg,
            buckets=(0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0),
        ),
        feature_freshness_seconds=Histogram(
            "feature_freshness_seconds",
            "Age of served feature values (request time minus value event time).",
            ["view"],
            registry=reg,
            buckets=(60, 600, 3_600, 6 * 3_600, 86_400, 3 * 86_400, 7 * 86_400, 30 * 86_400),
        ),
        parity_mismatch_ratio=Gauge(
            "parity_mismatch_ratio",
            "Share of sampled online reads that differ from the offline recomputation.",
            ["view"],
            registry=reg,
        ),
        materialization_lag_seconds=Gauge(
            "materialization_lag_seconds",
            "Wall-clock time minus the view's committed materialization watermark.",
            ["view"],
            registry=reg,
        ),
        contract_failures_total=Counter(
            "contract_failures_total",
            "Rows rejected by source data contracts, by source and check.",
            ["source", "check"],
            registry=reg,
        ),
        feature_psi=Gauge(
            "feature_psi",
            "Population Stability Index of a feature against its training reference.",
            ["view", "feature"],
            registry=reg,
        ),
    )


@dataclass(frozen=True, slots=True)
class FeatureDriftResult:
    report: DriftReport
    skipped: tuple[str, ...]


def _numeric(frame: pl.DataFrame, column: str) -> npt.NDArray[np.float64]:
    series = frame[column].drop_nulls()
    return np.asarray(series.cast(pl.Float64).to_numpy(), dtype=np.float64)


class FeatureDriftMonitor:
    """PSI and KS drift of numeric and boolean retrieval columns (``<view>__<feature>``)."""

    def __init__(
        self,
        reference: pl.DataFrame,
        columns: Sequence[str],
        *,
        psi_threshold: float = 0.2,
        ks_alpha: float = 0.01,
        ks_min_statistic: float = 0.05,
    ) -> None:
        self.columns = tuple(column for column in columns if _numeric(reference, column).size)
        if not self.columns:
            raise ValueError("no reference column has any non-null value")
        self._monitor = DriftMonitor(
            {column: _numeric(reference, column) for column in self.columns},
            psi_threshold=psi_threshold,
            ks_alpha=ks_alpha,
            ks_min_statistic=ks_min_statistic,
        )

    def check(
        self, current: pl.DataFrame, metrics: FeatureStoreMetrics | None = None
    ) -> FeatureDriftResult:
        samples = {column: _numeric(current, column) for column in self.columns}
        skipped = tuple(sorted(column for column, values in samples.items() if values.size == 0))
        present = {column: values for column, values in samples.items() if values.size}
        placeholder = dict.fromkeys(skipped, _PLACEHOLDER)
        report = self._monitor.check({**present, **placeholder})
        features = tuple(item for item in report.features if item.feature not in skipped)
        result = DriftReport(features)
        if metrics is not None:
            for item in features:
                view, _, feature = item.feature.partition("__")
                metrics.feature_psi.labels(view, feature or item.feature).set(item.psi)
        return FeatureDriftResult(result, skipped)


_PLACEHOLDER: npt.NDArray[np.float64] = np.zeros(1)
