from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from point_in_time_feature_store.core.tracing import build_tracer_provider
from point_in_time_feature_store.definitions import FeatureDefinition
from point_in_time_feature_store.devtools import LocalStack, build_local_stack
from point_in_time_feature_store.dsl import DType, col
from point_in_time_feature_store.ml.contracts import DataContractError
from point_in_time_feature_store.offline.synthetic import (
    SyntheticConfig,
    SyntheticDataset,
    churn_feature_repo,
    generate_dataset,
)
from point_in_time_feature_store.online.store import row_key
from point_in_time_feature_store.services import FeatureStoreServices
from point_in_time_feature_store.serving.auth import Caller
from point_in_time_feature_store.timeutil import US_PER_DAY, US_PER_SECOND

pytestmark = pytest.mark.integration

ALL = ["customer_profile:*", "transaction_stats:*", "session_stats:*"]
READER = Caller("svc-tests", frozenset({"features:read:*", "pii:read"}))
DATASET = generate_dataset(SyntheticConfig(n_customers=250, seed=5))
T1 = DATASET.config.day_us(300)
T2 = DATASET.config.day_us(DATASET.config.horizon_days)


@dataclass
class Env:
    stack: LocalStack
    dataset: SyntheticDataset

    @property
    def services(self) -> FeatureStoreServices:
        return self.stack.services

    def ingest(self, *, up_to: int | None = None, after: int | None = None) -> None:
        for name, frame in self.dataset.sources.items():
            ingested = pl.col("ingested_at").dt.epoch("us")
            selected = frame
            if up_to is not None:
                selected = selected.filter(ingested <= up_to)
            if after is not None:
                selected = selected.filter(ingested > after)
            self.services.ingest(name, selected)

    def materialize(self, up_to: int, **kwargs: object) -> dict[str, int]:
        written = {}
        for view in self.services.registry.active_repo().feature_views:
            report = self.services.materializer.materialize(view.name, up_to_us=up_to, **kwargs)  # type: ignore[arg-type]
            written[view.name] = report.written
        return written

    def serve(self, entities: list[str], times: list[int]) -> None:
        for index, now in enumerate(times):
            batch = entities[index % len(entities) : index % len(entities) + 3]
            self.services.online_service.get(
                READER, batch, ALL, request_id=uuid.uuid4().hex, now_us=now
            )


def make_env(root: Path) -> Env:
    stack = build_local_stack(root, now_us=T2)
    stack.services.registry.apply(churn_feature_repo(), actor="tests")
    stack.services.online_service.refresh()
    return Env(stack, DATASET)


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    environment = make_env(tmp_path / "a")
    yield environment
    environment.stack.close()


def entity_ids(count: int) -> list[str]:
    return DATASET.truth.filter(pl.col("churn_us").is_null())["entity_id"].head(count).to_list()


def test_incremental_runs_with_late_data_equal_a_one_shot_run(env: Env, tmp_path: Path) -> None:
    env.ingest(up_to=T1)
    assert all(count > 0 for count in env.materialize(T1).values())
    env.ingest(after=T1)
    env.materialize(T2)
    incremental = env.services.online.state_digest()
    assert env.materialize(T2) == {
        "customer_profile": 0,
        "session_stats": 0,
        "transaction_stats": 0,
    }
    assert env.services.online.state_digest() == incremental

    other = make_env(tmp_path / "b")
    try:
        other.ingest()
        other.materialize(T1)
        other.materialize(T2, overlap_us=400 * US_PER_DAY)  # re-scans everything: idempotent
        assert other.services.online.state_digest() == incremental
    finally:
        other.stack.close()


def test_materialization_arguments_are_checked(env: Env) -> None:
    with pytest.raises(ValueError, match="beyond the current time"):
        env.services.materializer.materialize("session_stats", up_to_us=T2 + 1)
    with pytest.raises(ValueError, match="overlap"):
        env.services.materializer.materialize("session_stats", up_to_us=T2, overlap_us=-1)
    report = env.services.materializer.materialize("session_stats")
    assert report.watermark_us == T2
    assert not report.noop
    assert env.services.materializer.materialize("session_stats", up_to_us=T1).noop


def test_contract_failures_block_materialization_and_keep_the_watermark(env: Env) -> None:
    env.ingest(up_to=T1)
    source = env.services.registry.active_repo().source("transactions")
    visible = pl.max_horizontal(pl.col("event_time"), pl.col("ingested_at")).dt.epoch("us")
    corrupt = (
        DATASET.sources["transactions"]
        .filter(visible <= T1)
        .head(3)
        .with_columns(pl.lit(-5.0).alias("amount"))
    )
    bucket = env.services.store.root / "sources" / "transactions" / "bucket=00"
    bucket.mkdir(parents=True, exist_ok=True)
    corrupt.write_parquet(
        bucket / f"part-{'9' * 20}-{'a' * 16}.parquet"
    )  # bypasses append's contract
    assert source.name == "transactions"
    with pytest.raises(DataContractError):
        env.services.materializer.materialize("transaction_stats", up_to_us=T1)
    version = env.services.registry.view_version("transaction_stats")
    assert env.services.registry.get_watermark("transaction_stats", version) is None
    failures = env.services.metrics.registry.get_sample_value(
        "contract_failures_total",
        {"source": "transactions", "check": "greater_than_or_equal_to(0.0)"},
    )
    assert failures == 3.0


def test_rows_behind_the_watermark_are_rejected_on_ingest(env: Env) -> None:
    env.ingest(up_to=T1)
    env.materialize(T1)
    stale = (
        DATASET.sources["transactions"].filter(pl.col("ingested_at").dt.epoch("us") <= T1).head(2)
    )
    with pytest.raises(DataContractError) as info:
        env.services.ingest("transactions", stale)
    assert {item["check"] for item in info.value.failures} == {"knowledge_time_append_only"}
    value = env.services.metrics.registry.get_sample_value(
        "contract_failures_total", {"source": "transactions", "check": "knowledge_time_append_only"}
    )
    assert value == 2.0


def _transaction_at(entity: str, event_us: int, ingested_us: int) -> pl.DataFrame:
    def stamp(value: int) -> pl.Expr:
        return pl.from_epoch(pl.lit(value), time_unit="us").dt.replace_time_zone("UTC")

    template = DATASET.sources["transactions"].filter(pl.col("entity_id") == entity).head(1)
    return template.with_columns(
        stamp(event_us).alias("event_time"),
        stamp(ingested_us).alias("ingested_at"),
        pl.lit(4321.5).alias("amount"),
    )


def test_appends_during_an_inflight_materialization_are_refused_not_lost(env: Env) -> None:
    """A row appended mid-run, behind the run's target, would never reach the online store."""
    env.ingest(up_to=T1)
    env.materialize(T1)
    env.ingest(after=T1)
    victim = next(
        entity
        for entity in entity_ids(200)
        if DATASET.sources["transactions"].filter(pl.col("entity_id") == entity).height
    )
    outcome: dict[str, object] = {}

    def append_mid_run() -> None:
        late = _transaction_at(victim, T2 - 2 * US_PER_SECOND, T2 - US_PER_SECOND)
        try:
            env.services.ingest("transactions", late)
            outcome["accepted"] = True
        except DataContractError as exc:
            outcome["checks"] = {item["check"] for item in exc.failures}

    env.services.materializer.materialize(
        "transaction_stats", up_to_us=T2, before_commit=append_mid_run
    )
    assert outcome == {"checks": {"knowledge_time_append_only"}}

    # After the commit, appends beyond the new watermark are accepted and reach the online store.
    later = T2 + US_PER_DAY
    env.stack.clock.now_us = later
    fresh = _transaction_at(victim, later - 2 * US_PER_SECOND, later - US_PER_SECOND)
    assert env.services.ingest("transactions", fresh).rows_written == 1
    env.services.materializer.materialize("transaction_stats", up_to_us=later)
    env.serve([victim], [later])
    report = env.services.skew_checker().check(sample_size=100, seed=0)
    assert report.checked > 0
    assert report.mismatches == 0
    served = env.services.request_log.sample(100, seed=0)
    assert any(read.event_time_us == later - 2 * US_PER_SECOND for read in served)


def test_a_failed_materialization_releases_its_knowledge_floor(env: Env) -> None:
    env.ingest(up_to=T1)
    env.materialize(T1)
    assert env.services.registry.knowledge_floor_for_source("transactions") == T1

    def fail() -> None:
        assert env.services.registry.knowledge_floor_for_source("transactions") == T2
        raise RuntimeError("simulated crash before the commit")

    with pytest.raises(RuntimeError, match="simulated crash"):
        env.services.materializer.materialize("transaction_stats", up_to_us=T2, before_commit=fail)
    assert env.services.registry.knowledge_floor_for_source("transactions") == T1
    env.ingest(after=T1)
    env.materialize(T2)
    env.serve(entity_ids(20), [T2] * 10)
    assert env.services.skew_checker().check(sample_size=1_000, seed=0).mismatches == 0


def test_tombstoned_entities_are_not_materialized(env: Env) -> None:
    env.ingest()
    victim = entity_ids(1)[0]
    env.services.registry.add_tombstone(env.services.hasher.entity(victim), "receipt")
    report = env.services.materializer.materialize("transaction_stats", up_to_us=T2)
    assert report.entities_dropped_tombstoned > 0
    versions = env.services.registry.view_versions()
    assert env.services.online.count_entity_keys(victim, versions) == 0


def test_served_values_match_offline_recomputation(env: Env) -> None:
    env.ingest()
    env.materialize(T2)
    entities = entity_ids(60)
    env.serve(entities, [T2 + (i * 7_919_000_000) % (3 * US_PER_DAY) for i in range(40)])
    report = env.services.skew_checker().check(sample_size=10_000, seed=1)
    assert report.checked == env.services.request_log.count() > 0
    assert report.mismatches == 0
    assert set(report.views) == {"customer_profile", "session_stats", "transaction_stats"}
    ratio = env.services.metrics.registry.get_sample_value(
        "parity_mismatch_ratio", {"view": "session_stats"}
    )
    assert ratio == 0.0


def test_tampered_online_values_are_reported_without_values(env: Env) -> None:
    env.ingest()
    env.materialize(T2)
    entities = entity_ids(5)
    version = env.services.registry.view_version("transaction_stats")
    key = row_key("transaction_stats", version, entities[0])
    assert env.services.redis_client.exists(key)
    env.services.redis_client.hset(key, "last_amount", "123456.5")
    env.serve(entities[:1], [T2])
    report = env.services.skew_checker().check(sample_size=100, seed=0)
    parity = report.views["transaction_stats"]
    assert parity.mismatches == 1
    assert parity.ratio == 1.0
    (example,) = report.examples
    assert example.fields == ("last_amount",)
    assert "123456" not in repr(report)
    assert report.views["session_stats"].mismatches == 0


def test_reads_of_retired_versions_are_skipped(env: Env) -> None:
    env.ingest()
    env.materialize(T2)
    env.serve(entity_ids(3), [T2])
    repo = env.services.registry.active_repo()
    view = repo.view("session_stats")
    extra = FeatureDefinition(
        name="pages_x2", dtype=DType.INT64, expression=(col("pages") * 2).node
    )
    changed = repo.model_copy(
        update={
            "feature_views": tuple(
                v.model_copy(update={"features": (*v.features, extra)})
                if v.name == view.name
                else v
                for v in repo.feature_views
            )
        }
    )
    env.services.registry.apply(changed, actor="tests")
    report = env.services.skew_checker().check(sample_size=100, seed=0)
    assert report.views["session_stats"].skipped_version_changed == 3
    assert report.views["session_stats"].checked == 0


def test_reads_during_an_inflight_materialization_are_not_false_mismatches(env: Env) -> None:
    env.ingest(up_to=T1)
    env.materialize(T1)
    env.ingest(after=T1)
    entities = entity_ids(40)

    def read_mid_run() -> None:
        env.serve(entities, [T2] * 20)

    env.services.materializer.materialize(
        "transaction_stats", up_to_us=T2, before_commit=read_mid_run
    )
    logged = env.services.request_log.sample(1_000, seed=0)
    inflight = [r for r in logged if r.view == "transaction_stats"]
    assert inflight
    assert all(r.watermark_low_us == T1 and r.watermark_high_us == T2 for r in inflight)
    report = env.services.skew_checker().check(sample_size=1_000, seed=0)
    assert report.mismatches == 0


def test_materialization_runs_are_traced(env: Env) -> None:
    env.ingest()
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("tests", extra_exporter=exporter)
    env.services.materializer.tracer = provider.get_tracer("tests")
    report = env.services.materializer.materialize("session_stats", up_to_us=T2)
    (span,) = [s for s in exporter.get_finished_spans() if s.name == "feature_store.materialize"]
    attributes = dict(span.attributes or {})
    assert attributes["feature_store.view"] == "session_stats"
    assert attributes["feature_store.written"] == report.written > 0
    assert attributes["feature_store.watermark_us"] == T2
