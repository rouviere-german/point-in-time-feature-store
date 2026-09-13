"""Leakage experiment: a LightGBM churn model trained on naive joins versus point-in-time joins.

Customers are split by a hash of their id into train (60%), validation (20%) and holdout (20%).
Label rows are taken every 14 days for active customers; the label is "churns within 30 days".

- *Offline AUC*: train on the train customers' label rows in the training period, score the
  validation customers' label rows in the same period, with features built by the arm's join.
- *Holdout AUC*: score the holdout customers' label rows in a later period with point-in-time
  features, i.e. exactly what serving would have returned at those times.
- *Period control AUC*: score the validation customers' label rows in the holdout period with
  point-in-time features. Offline and holdout AUC differ in both the customers and the period;
  this control shares the customers with the offline AUC and the period with the holdout AUC, so
  it separates a difference between periods from a difference between customer splits.
- *Gap CI*: a 95% percentile interval for the gap from a cluster bootstrap that resamples
  customers (all label rows of a customer together, since they are correlated) in the validation
  and holdout sets independently; every arm uses the same resamples.

Arms: ``pit`` (the store's retrieval), ``naive_latest`` (latest row per entity in a data extract
taken after the training period) and ``event_time_asof`` (ASOF on event time, blind to ingestion
time). A leaking arm scores well offline and worse on the holdout; the gap is the leakage.

Lineage: every run records the seed, feature-view version hashes, Parquet file hashes and model
hashes in MLflow (file backend) and in the registry's ``training_runs`` table.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import JsonValue
from sklearn.metrics import roc_auc_score

from point_in_time_feature_store.definitions import FeatureRepo, feature_column
from point_in_time_feature_store.offline.baselines import event_time_only_asof, naive_latest_join
from point_in_time_feature_store.offline.retrieval import (
    HistoricalRetriever,
    attach_view_columns,
    normalize_spine,
)
from point_in_time_feature_store.offline.synthetic import TIERS, SyntheticDataset, to_timestamp
from point_in_time_feature_store.registry import FeatureRegistry
from point_in_time_feature_store.timeutil import US_PER_DAY, seconds_to_us

__all__ = [
    "ARMS",
    "ArmResult",
    "LeakageConfig",
    "LeakageResult",
    "build_label_spine",
    "model_matrix",
    "run_leakage_experiment",
    "split_of",
]

Arm = Literal["pit", "naive_latest", "event_time_asof"]
ARMS: tuple[Arm, ...] = ("pit", "naive_latest", "event_time_asof")
FloatMatrix = npt.NDArray[np.float64]

MODEL_FEATURES = (
    "customer_profile:plan_tier",
    "customer_profile:monthly_fee",
    "customer_profile:support_tickets_90d",
    "customer_profile:is_cancelled",
    "transaction_stats:last_amount",
    "transaction_stats:txn_count_30d",
    "transaction_stats:is_large",
    "session_stats:last_session_minutes",
    "session_stats:sessions_7d",
    "session_stats:engaged",
)


@dataclass(frozen=True, slots=True)
class LeakageConfig:
    label_every_days: int = 14
    label_horizon_days: int = 30
    train_start_day: int = 60
    train_end_day: int = 240
    snapshot_day: int = 275
    holdout_start_day: int = 280
    holdout_end_day: int = 380
    num_boost_round: int = 120
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_data_in_leaf: int = 40
    threads: int = 2
    bootstrap_resamples: int = 200


@dataclass(frozen=True, slots=True)
class ArmResult:
    arm: str
    offline_auc: float
    holdout_auc: float
    period_control_auc: float
    gap_ci95: tuple[float, float]
    model_sha256: str

    @property
    def gap(self) -> float:
        return self.offline_auc - self.holdout_auc


@dataclass(frozen=True, slots=True)
class LeakageResult:
    seed: int
    arms: dict[str, ArmResult]
    rows: dict[str, int]
    positive_rate: float
    view_versions: dict[str, str]
    dataset_digest: str
    mlflow_run_id: str | None = None
    extra: dict[str, JsonValue] = field(default_factory=dict)

    def summary(self) -> dict[str, JsonValue]:
        return {
            "seed": self.seed,
            "rows": dict(self.rows),
            "positive_rate": round(self.positive_rate, 4),
            "arms": {
                name: {
                    "offline_auc": round(arm.offline_auc, 4),
                    "holdout_auc": round(arm.holdout_auc, 4),
                    "period_control_auc": round(arm.period_control_auc, 4),
                    "gap": round(arm.gap, 4),
                    "gap_ci95": [round(arm.gap_ci95[0], 4), round(arm.gap_ci95[1], 4)],
                }
                for name, arm in self.arms.items()
            },
            "mlflow_run_id": self.mlflow_run_id,
        }


def split_of(entity_id: str) -> Literal["train", "val", "holdout"]:
    bucket = hashlib.sha256(b"split\x00" + entity_id.encode("utf-8")).digest()[0] % 10
    return "train" if bucket < 6 else "val" if bucket < 8 else "holdout"


def build_label_spine(
    dataset: SyntheticDataset,
    *,
    start_day: int,
    end_day: int,
    every_days: int,
    horizon_days: int,
    split: str,
) -> pl.DataFrame:
    config = dataset.config
    times = [config.day_us(day) for day in range(start_day, end_day + 1, every_days)]
    truth = dataset.truth.filter(
        pl.col("entity_id").map_elements(split_of, return_dtype=pl.String) == split
    )
    grid = truth.join(pl.DataFrame({"label_us": times}, schema={"label_us": pl.Int64}), how="cross")
    active = grid.filter(
        (pl.col("signup_us") <= pl.col("label_us"))
        & (pl.col("churn_us").is_null() | (pl.col("churn_us") > pl.col("label_us")))
    )
    labelled = active.with_columns(
        (
            pl.col("churn_us").is_not_null()
            & (pl.col("churn_us") <= pl.col("label_us") + horizon_days * US_PER_DAY)
        )
        .cast(pl.Int64)
        .alias("label")
    ).sort(["label_us", "entity_id"])
    return labelled.select(
        "entity_id", to_timestamp(labelled["label_us"]).alias("label_time"), "label"
    )


def model_matrix(frame: pl.DataFrame) -> tuple[FloatMatrix, list[str]]:
    """Numeric design matrix; value ages are measured against label time (negative ages leak)."""
    tier_codes = {tier: float(i) for i, tier in enumerate(TIERS)}
    label_us = frame["label_time"].dt.epoch("us")
    columns: dict[str, pl.Series] = {}
    for ref in MODEL_FEATURES:
        view, feature = ref.split(":")
        series = frame[feature_column(view, feature)]
        if feature == "plan_tier":
            series = series.replace_strict(tier_codes, default=None, return_dtype=pl.Float64)
        columns[f"{view}__{feature}"] = series.cast(pl.Float64)
    for view in ("customer_profile", "transaction_stats", "session_stats"):
        event_us = frame[feature_column(view, "event_time")].dt.epoch("us")
        columns[f"{view}__age_days"] = ((label_us - event_us) / US_PER_DAY).cast(pl.Float64)
    names = list(columns)
    matrix = np.column_stack([columns[name].fill_null(np.nan).to_numpy() for name in names])
    return matrix.astype(np.float64), names


def _train(
    x: FloatMatrix, y: npt.NDArray[np.int64], config: LeakageConfig, seed: int
) -> lgb.Booster:
    params = {
        "objective": "binary",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": config.threads,
        "verbose": -1,
    }
    return lgb.train(
        params, lgb.Dataset(x, label=y, free_raw_data=True), num_boost_round=config.num_boost_round
    )


FloatVector = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _Scored:
    """Labels, model scores and the customer (cluster) index of every label row."""

    labels: IntVector
    scores: FloatVector
    groups: IntVector


def _scores(model: lgb.Booster, x: FloatMatrix) -> FloatVector:
    return np.asarray(model.predict(x), dtype=np.float64)


def _auc(labels: IntVector, scores: FloatVector, weights: FloatVector | None = None) -> float:
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def _groups(spine: pl.DataFrame) -> IntVector:
    _, inverse = np.unique(spine["entity_id"].to_numpy(), return_inverse=True)
    return np.asarray(inverse, dtype=np.int64)


def _cluster_weights(groups: IntVector, rng: np.random.Generator) -> FloatVector:
    clusters = int(groups.max()) + 1 if groups.size else 0
    counts = np.bincount(rng.integers(0, clusters, size=clusters), minlength=clusters)
    return np.asarray(counts[groups], dtype=np.float64)


def _has_both_classes(labels: IntVector, weights: FloatVector) -> bool:
    return bool(weights[labels == 1].sum() > 0 and weights[labels == 0].sum() > 0)


def gap_ci95(
    offline: _Scored, holdout: _Scored, *, resamples: int, seed: int
) -> tuple[float, float]:
    """95% percentile interval of ``AUC(offline) - AUC(holdout)`` under a customer bootstrap."""
    rng = np.random.default_rng(seed)
    gaps: list[float] = []
    for _ in range(resamples):
        w_offline = _cluster_weights(offline.groups, rng)
        w_holdout = _cluster_weights(holdout.groups, rng)
        if not (
            _has_both_classes(offline.labels, w_offline)
            and _has_both_classes(holdout.labels, w_holdout)
        ):
            continue
        gaps.append(
            _auc(offline.labels, offline.scores, w_offline)
            - _auc(holdout.labels, holdout.scores, w_holdout)
        )
    if not gaps:
        return (float("nan"), float("nan"))
    low, high = np.percentile(np.asarray(gaps, dtype=np.float64), [2.5, 97.5])
    return (float(low), float(high))


def _arm_features(
    arm: Arm,
    retriever: HistoricalRetriever,
    spine: pl.DataFrame,
    snapshot_us: int,
    rows_cache: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    if arm == "pit":
        return retriever.get_historical_features(spine, MODEL_FEATURES).frame
    repo = retriever.repo
    normalized = normalize_spine(spine)
    output = spine
    views: dict[str, list[str]] = {}
    for ref in MODEL_FEATURES:
        view_name, feature = ref.split(":")
        views.setdefault(view_name, []).append(feature)
    for view_name, names in views.items():
        view = repo.view(view_name)
        rows = rows_cache.get(view_name)
        if rows is None:
            rows = retriever.feature_rows(view)
            rows_cache[view_name] = rows
        if arm == "naive_latest":
            joined = naive_latest_join(
                retriever.engine, normalized, rows, view.feature_names, snapshot_us
            )
        else:
            joined = event_time_only_asof(
                retriever.engine,
                normalized,
                rows,
                view.feature_names,
                seconds_to_us(view.ttl_seconds),
            )
        output = attach_view_columns(output, view_name, joined, names)
    return output


def _labels(frame: pl.DataFrame) -> npt.NDArray[np.int64]:
    return frame["label"].to_numpy().astype(np.int64)


def run_leakage_experiment(
    dataset: SyntheticDataset,
    retriever: HistoricalRetriever,
    *,
    seed: int,
    config: LeakageConfig | None = None,
    arms: Sequence[Arm] = ARMS,
    registry: FeatureRegistry | None = None,
    mlflow_dir: Path | None = None,
) -> LeakageResult:
    cfg = config or LeakageConfig()
    common = {"every_days": cfg.label_every_days, "horizon_days": cfg.label_horizon_days}
    train = build_label_spine(
        dataset, start_day=cfg.train_start_day, end_day=cfg.train_end_day, split="train", **common
    )
    val = build_label_spine(
        dataset, start_day=cfg.train_start_day, end_day=cfg.train_end_day, split="val", **common
    )
    holdout = build_label_spine(
        dataset,
        start_day=cfg.holdout_start_day,
        end_day=cfg.holdout_end_day,
        split="holdout",
        **common,
    )
    control = build_label_spine(
        dataset,
        start_day=cfg.holdout_start_day,
        end_day=cfg.holdout_end_day,
        split="val",
        **common,
    )
    snapshot_us = dataset.config.day_us(cfg.snapshot_day)
    rows_cache: dict[str, pl.DataFrame] = {}
    x_hold, _ = model_matrix(retriever.get_historical_features(holdout, MODEL_FEATURES).frame)
    y_hold = _labels(holdout)
    x_control, _ = model_matrix(retriever.get_historical_features(control, MODEL_FEATURES).frame)
    y_control = _labels(control)
    y_val = _labels(val)
    groups_val, groups_hold = _groups(val), _groups(holdout)
    results: dict[str, ArmResult] = {}
    models: dict[str, str] = {}
    for arm in arms:
        x_train, _names = model_matrix(
            _arm_features(arm, retriever, train, snapshot_us, rows_cache)
        )
        x_val, _ = model_matrix(_arm_features(arm, retriever, val, snapshot_us, rows_cache))
        model = _train(x_train, _labels(train), cfg, seed)
        text = model.model_to_string()
        models[arm] = text
        scored_val = _Scored(y_val, _scores(model, x_val), groups_val)
        scored_hold = _Scored(y_hold, _scores(model, x_hold), groups_hold)
        results[arm] = ArmResult(
            arm=arm,
            offline_auc=_auc(scored_val.labels, scored_val.scores),
            holdout_auc=_auc(scored_hold.labels, scored_hold.scores),
            period_control_auc=_auc(y_control, _scores(model, x_control)),
            gap_ci95=gap_ci95(
                scored_val, scored_hold, resamples=cfg.bootstrap_resamples, seed=seed
            ),
            model_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
    repo: FeatureRepo = retriever.repo
    view_versions = {view.name: view.version_hash() for view in repo.feature_views}
    sources = [source.name for source in repo.sources]
    digest = retriever.store.dataset_digest(sources)
    positives = int(_labels(train).sum() + _labels(val).sum() + y_hold.sum())
    total = train.height + val.height + holdout.height
    result = LeakageResult(
        seed=seed,
        arms=results,
        rows={
            "train": train.height,
            "val": val.height,
            "holdout": holdout.height,
            "period_control": control.height,
        },
        positive_rate=positives / total if total else 0.0,
        view_versions=view_versions,
        dataset_digest=digest,
    )
    run_id = None
    if mlflow_dir is not None:
        run_id = log_lineage(mlflow_dir, result, models, retriever, dataset)
    if registry is not None:
        registry.record_training_run(
            run_id or f"local-{seed}-{digest[:12]}", "churn-lightgbm", view_versions, digest
        )
    return LeakageResult(
        seed=result.seed,
        arms=result.arms,
        rows=result.rows,
        positive_rate=result.positive_rate,
        view_versions=result.view_versions,
        dataset_digest=result.dataset_digest,
        mlflow_run_id=run_id,
    )


def log_lineage(
    mlflow_dir: Path,
    result: LeakageResult,
    models: dict[str, str],
    retriever: HistoricalRetriever,
    dataset: SyntheticDataset,
) -> str:
    """Record the run in MLflow's file store and return the run id."""
    # MLflow 3 keeps the file store in maintenance mode behind an explicit opt-in (see ADR 0004).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow.entities import Metric, Param  # noqa: PLC0415 - heavy import, only for lineage
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    from point_in_time_feature_store.ml.manifest import create_manifest  # noqa: PLC0415

    mlflow_dir.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=mlflow_dir.resolve().as_uri())
    experiment = client.get_experiment_by_name("churn-leakage")
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment("churn-leakage")
    )
    run = client.create_run(experiment_id, run_name=f"seed-{result.seed}")
    run_id = str(run.info.run_id)
    sources = [source.name for source in retriever.repo.sources]
    params = {
        "seed": str(result.seed),
        "n_customers": str(dataset.config.n_customers),
        "generator_seed": str(dataset.config.seed),
        "dataset_digest": result.dataset_digest,
        **{
            f"view_version.{name}": version
            for name, version in sorted(result.view_versions.items())
        },
        **{f"model_sha256.{arm}": item.model_sha256 for arm, item in result.arms.items()},
    }
    for source in sources:
        params[f"source_digest.{source}"] = retriever.store.dataset_digest([source])
    metrics = []
    for arm, item in result.arms.items():
        metrics += [
            Metric(f"{arm}.offline_auc", item.offline_auc, 0, 0),
            Metric(f"{arm}.holdout_auc", item.holdout_auc, 0, 0),
            Metric(f"{arm}.period_control_auc", item.period_control_auc, 0, 0),
            Metric(f"{arm}.gap", item.gap, 0, 0),
            Metric(f"{arm}.gap_ci95_low", item.gap_ci95[0], 0, 0),
            Metric(f"{arm}.gap_ci95_high", item.gap_ci95[1], 0, 0),
        ]
    client.log_batch(
        run_id, metrics=metrics, params=[Param(key, value) for key, value in params.items()]
    )
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for arm, text in models.items():
            path = folder / f"model_{arm}.txt"
            path.write_text(text, encoding="utf-8")
            client.log_artifact(run_id, str(path), artifact_path="models")
        manifest = create_manifest(
            seed=result.seed,
            datasets=[
                retriever.store.root / relative
                for relative in sorted(retriever.store.fingerprints(sources))
            ],
            params=dict(params),
            metrics={f"{arm}.holdout_auc": item.holdout_auc for arm, item in result.arms.items()},
            packages=["duckdb", "polars", "lightgbm", "scikit-learn", "numpy", "pyarrow"],
            run_id=run_id,
        )
        manifest_path = folder / "manifest.json"
        manifest.write(manifest_path)
        client.log_artifact(run_id, str(manifest_path))
    client.set_terminated(run_id)
    return run_id
