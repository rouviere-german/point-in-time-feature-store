from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from point_in_time_feature_store.offline.retrieval import DuckDBEngine, HistoricalRetriever
from point_in_time_feature_store.offline.store import OfflineStore
from point_in_time_feature_store.offline.synthetic import (
    SyntheticConfig,
    churn_feature_repo,
    generate_dataset,
    to_timestamp,
)
from point_in_time_feature_store.registry import FeatureRegistry
from point_in_time_feature_store.timeutil import US_PER_DAY
from point_in_time_feature_store.training import (
    build_label_spine,
    model_matrix,
    run_leakage_experiment,
    split_of,
)

pytestmark = pytest.mark.integration

REPO = churn_feature_repo()
DATASET = generate_dataset(SyntheticConfig(n_customers=1_500, seed=31))


@pytest.fixture(scope="module")
def retriever(tmp_path_factory: pytest.TempPathFactory) -> Iterator[HistoricalRetriever]:
    store = OfflineStore(tmp_path_factory.mktemp("training") / "offline")
    for name, frame in DATASET.sources.items():
        store.append(REPO.source(name), REPO.entity("customer"), frame)
    engine = DuckDBEngine(threads=2, memory_limit="512MB")
    yield HistoricalRetriever(store, REPO, engine)
    engine.close()


def test_naive_joins_leak_and_point_in_time_joins_do_not(retriever: HistoricalRetriever) -> None:
    result = run_leakage_experiment(DATASET, retriever, seed=1, arms=("pit", "naive_latest"))
    pit, naive = result.arms["pit"], result.arms["naive_latest"]
    assert naive.offline_auc > pit.offline_auc
    assert naive.gap > pit.gap + 0.05
    assert 0.5 < pit.holdout_auc < 1.0
    assert min(result.rows.values()) > 100
    assert 0.02 < result.positive_rate < 0.3
    summary = result.summary()
    assert set(summary["arms"]) == {"pit", "naive_latest"}  # type: ignore[arg-type]


def test_training_is_deterministic_for_a_seed(retriever: HistoricalRetriever) -> None:
    first = run_leakage_experiment(DATASET, retriever, seed=4, arms=("pit",))
    second = run_leakage_experiment(DATASET, retriever, seed=4, arms=("pit",))
    assert first.arms["pit"] == second.arms["pit"]
    other = run_leakage_experiment(DATASET, retriever, seed=5, arms=("pit",))
    assert other.arms["pit"].model_sha256 != first.arms["pit"].model_sha256


def test_lineage_is_recorded_in_mlflow_and_the_registry(
    tmp_path: Path, retriever: HistoricalRetriever
) -> None:
    registry = FeatureRegistry(tmp_path / "registry.sqlite")
    try:
        result = run_leakage_experiment(
            DATASET,
            retriever,
            seed=2,
            arms=("pit",),
            registry=registry,
            mlflow_dir=tmp_path / "mlruns",
        )
        assert result.mlflow_run_id is not None
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=(tmp_path / "mlruns").resolve().as_uri())
        run = client.get_run(result.mlflow_run_id)
        versions = {view.name: view.version_hash() for view in REPO.feature_views}
        for name, version in versions.items():
            assert run.data.params[f"view_version.{name}"] == version
        assert run.data.params["dataset_digest"] == retriever.store.dataset_digest(
            ["customers", "sessions", "transactions"]
        )
        assert run.data.params["model_sha256.pit"] == result.arms["pit"].model_sha256
        assert run.data.metrics["pit.holdout_auc"] == pytest.approx(result.arms["pit"].holdout_auc)
        artifacts = {item.path for item in client.list_artifacts(result.mlflow_run_id)}
        assert {"manifest.json", "models"} <= artifacts
        (training_run,) = registry.training_runs()
        assert training_run.run_id == result.mlflow_run_id
        assert training_run.view_versions == versions
    finally:
        registry.close()


def test_splits_and_label_spines_are_consistent() -> None:
    ids = DATASET.truth["entity_id"].to_list()
    assert [split_of(item) for item in ids[:20]] == [split_of(item) for item in ids[:20]]
    shares = {
        name: sum(split_of(item) == name for item in ids) / len(ids)
        for name in ("train", "val", "holdout")
    }
    assert shares["train"] == pytest.approx(0.6, abs=0.06)
    assert shares["holdout"] == pytest.approx(0.2, abs=0.05)
    spine = build_label_spine(
        DATASET, start_day=60, end_day=120, every_days=30, horizon_days=30, split="val"
    )
    joined = spine.with_columns(pl.col("label_time").dt.epoch("us").alias("t")).join(
        DATASET.truth, on="entity_id"
    )
    assert joined.filter(pl.col("signup_us") > pl.col("t")).height == 0
    assert joined.filter(pl.col("churn_us") <= pl.col("t")).height == 0
    positives = joined.filter(pl.col("label") == 1)
    assert positives.filter(pl.col("churn_us") > pl.col("t") + 30 * US_PER_DAY).height == 0
    assert {split_of(item) for item in spine["entity_id"].to_list()} == {"val"}


def test_value_ages_are_measured_against_label_time() -> None:
    label = DATASET.config.day_us(100)
    columns: dict[str, pl.Series] = {"label_time": to_timestamp(pl.Series([label]))}
    for view in ("customer_profile", "transaction_stats", "session_stats"):
        columns[f"{view}__event_time"] = to_timestamp(pl.Series([label - 2 * US_PER_DAY]))
    for ref, value in (
        ("customer_profile__plan_tier", "premium"),
        ("customer_profile__monthly_fee", 39.99),
        ("customer_profile__support_tickets_90d", 1),
        ("customer_profile__is_cancelled", False),
        ("transaction_stats__last_amount", None),
        ("transaction_stats__txn_count_30d", 3),
        ("transaction_stats__is_large", True),
        ("session_stats__last_session_minutes", 12.5),
        ("session_stats__sessions_7d", 2),
        ("session_stats__engaged", False),
    ):
        columns[ref] = pl.Series([value])
    matrix, names = model_matrix(pl.DataFrame(columns))
    row = dict(zip(names, matrix[0], strict=True))
    assert row["customer_profile__plan_tier"] == 2.0
    assert np.isnan(row["transaction_stats__last_amount"])
    assert row["session_stats__age_days"] == pytest.approx(2.0)
