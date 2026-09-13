from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from point_in_time_feature_store.definitions import FeatureDefinition, FeatureRepo
from point_in_time_feature_store.dsl import DType, col
from point_in_time_feature_store.offline.synthetic import churn_feature_repo
from point_in_time_feature_store.registry import (
    ApprovalRequiredError,
    FeatureRegistry,
    PlanMismatchError,
)

pytestmark = pytest.mark.unit


class Ticker:
    def __init__(self) -> None:
        self.now = 1_000

    def __call__(self) -> int:
        self.now += 1
        return self.now


@pytest.fixture
def registry(tmp_path: Path) -> FeatureRegistry:
    return FeatureRegistry(tmp_path / "registry.sqlite", clock=Ticker())


def with_view(repo: FeatureRepo, name: str, **updates: object) -> FeatureRepo:
    views = tuple(
        view.model_copy(update=updates) if view.name == name else view
        for view in repo.feature_views
    )
    return repo.model_copy(update={"feature_views": views})


def test_create_then_no_changes(registry: FeatureRegistry) -> None:
    repo = churn_feature_repo()
    plan = registry.plan(repo)
    assert {change.action for change in plan.changes} == {"create"}
    assert not plan.destructive
    registry.apply(repo, actor="alice")
    assert registry.active_repo() == repo
    assert registry.plan(repo).is_empty
    assert registry.plan(repo).render() == "No changes."
    assert registry.apply(repo, actor="alice").is_empty
    assert registry.view_version("customer_profile") == repo.view("customer_profile").version_hash()
    with pytest.raises(KeyError):
        registry.view_version("missing")


def test_adding_a_feature_is_compatible_and_creates_a_new_version(
    registry: FeatureRegistry,
) -> None:
    repo = churn_feature_repo()
    registry.apply(repo, actor="alice")
    view = repo.view("transaction_stats")
    extra = FeatureDefinition(
        name="amount_x2", dtype=DType.FLOAT64, expression=(col("amount") * 2).node
    )
    changed = with_view(repo, "transaction_stats", features=(*view.features, extra))
    plan = registry.apply(changed, actor="alice")
    assert not plan.destructive
    assert plan.changes[0].reasons == ("feature added: amount_x2",)
    versions = registry.view_versions()["transaction_stats"]
    assert versions == [view.version_hash(), changed.view("transaction_stats").version_hash()]


def test_metadata_changes_keep_the_version(registry: FeatureRegistry) -> None:
    repo = churn_feature_repo()
    registry.apply(repo, actor="alice")
    changed = with_view(repo, "session_stats", owner="growth-team")
    plan = registry.apply(changed, actor="alice")
    assert plan.changes[0].reasons == ("metadata changed",)
    assert plan.changes[0].old_version == plan.changes[0].new_version
    assert registry.view("session_stats").owner == "growth-team"


def _pii_removed(repo: FeatureRepo) -> FeatureRepo:
    source = repo.source("customers")
    columns = tuple(column.model_copy(update={"pii": False}) for column in source.columns)
    view = repo.view("customer_profile")
    features = tuple(feature.model_copy(update={"pii": False}) for feature in view.features)
    return FeatureRepo(
        entities=repo.entities,
        sources=tuple(
            source.model_copy(update={"columns": columns}) if s.name == "customers" else s
            for s in repo.sources
        ),
        feature_views=tuple(
            view.model_copy(update={"features": features}) if v.name == view.name else v
            for v in repo.feature_views
        ),
    )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda repo: with_view(repo, "transaction_stats", ttl_seconds=86_400),
            "ttl_seconds changed",
        ),
        (
            lambda repo: with_view(
                repo, "session_stats", features=repo.view("session_stats").features[1:]
            ),
            "feature removed: last_session_minutes",
        ),
        (
            lambda repo: repo.model_copy(update={"feature_views": repo.feature_views[:2]}),
            "definition deleted",
        ),
        (_pii_removed, "pii tag removed: email"),
        (
            lambda repo: repo.model_copy(
                update={
                    "sources": tuple(
                        s.model_copy(update={"max_clock_skew_seconds": 600})
                        if s.name == "transactions"
                        else s
                        for s in repo.sources
                    )
                }
            ),
            "contract loosened: max_clock_skew_seconds",
        ),
    ],
)
def test_destructive_changes_need_approval_of_the_exact_plan(
    registry: FeatureRegistry, mutate: object, reason: str
) -> None:
    repo = churn_feature_repo()
    registry.apply(repo, actor="alice")
    changed = mutate(repo)  # type: ignore[operator]
    plan = registry.plan(changed)
    assert plan.destructive
    assert any(reason in change.reasons for change in plan.changes)
    assert "destructive changes" in plan.render()
    with pytest.raises(ApprovalRequiredError):
        registry.apply(changed, actor="mallory")
    with pytest.raises(PlanMismatchError):
        registry.apply(changed, actor="mallory", approve=True, plan_hash="0" * 64)
    assert registry.active_repo() == repo
    registry.apply(changed, actor="bob", approve=True, plan_hash=plan.plan_hash)
    assert registry.active_repo() == changed.revalidated()
    actions = [(entry.action, entry.actor) for entry in registry.change_log.entries()]
    assert actions == [
        ("apply", "alice"),
        ("apply_rejected", "mallory"),
        ("apply_rejected", "mallory"),
        ("apply", "bob"),
    ]
    assert registry.change_log.entries()[-1].payload["approved"] is True
    assert registry.verify_chains()["change_log"].ok


def test_source_and_entity_diffs(registry: FeatureRegistry) -> None:
    repo = churn_feature_repo()
    registry.apply(repo, actor="alice")
    source = repo.source("sessions")
    from point_in_time_feature_store.definitions import SourceColumn

    nullable = source.model_copy(
        update={"columns": (*source.columns, SourceColumn(name="device", dtype=DType.STRING))}
    )
    required = source.model_copy(
        update={
            "columns": (
                *source.columns,
                SourceColumn(name="device", dtype=DType.STRING, nullable=False),
            )
        }
    )
    tightened = source.model_copy(update={"max_backfill_days": 30})
    for candidate, destructive in ((nullable, False), (required, True), (tightened, False)):
        plan = registry.plan(
            repo.model_copy(
                update={
                    "sources": tuple(candidate if s.name == "sessions" else s for s in repo.sources)
                }
            )
        )
        assert plan.destructive is destructive
    entity = repo.entities[0]
    plan = registry.plan(
        repo.model_copy(update={"entities": (entity.model_copy(update={"key_max_length": 12}),)})
    )
    assert plan.destructive
    assert plan.changes[0].reasons == ("entity key format changed",)
    described = registry.plan(
        repo.model_copy(update={"entities": (entity.model_copy(update={"description": "x"}),)})
    )
    assert not described.destructive


def test_watermarks_tombstones_and_training_runs(registry: FeatureRegistry) -> None:
    repo = churn_feature_repo()
    registry.apply(repo, actor="alice")
    version = repo.view("transaction_stats").version_hash()
    assert registry.get_watermark("transaction_stats", version) is None
    assert registry.max_watermark_for_source("transactions") is None
    registry.set_watermark("transaction_stats", version, 50)
    registry.set_watermark("transaction_stats", version, 50)
    with pytest.raises(ValueError, match="backwards"):
        registry.set_watermark("transaction_stats", version, 49)
    assert registry.max_watermark_for_source("transactions") == 50
    registry.add_tombstone("a" * 64, "receipt-1")
    registry.add_tombstone("a" * 64, "receipt-2")
    assert registry.tombstones() == frozenset({"a" * 64})
    registry.record_training_run("run-1", "churn", {"transaction_stats": version}, "d" * 64)
    (run,) = registry.training_runs()
    assert run.view_versions == {"transaction_stats": version}


def test_tampered_or_truncated_chains_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    registry = FeatureRegistry(path, clock=Ticker())
    repo = churn_feature_repo()
    registry.apply(repo, actor="alice")
    for index in range(3):
        registry.record_access("svc", "online_read", {"entity_count": index})
    head = registry.access_audit.head()
    assert registry.access_audit.verify(expected_head=head).ok
    registry.close()

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE access_audit SET payload = '{\"entity_count\":99}' WHERE seq = 2"
        )
    reopened = FeatureRegistry(path, clock=Ticker())
    result = reopened.verify_chains()["access_audit"]
    assert not result.ok
    assert result.first_bad_seq == 2
    reopened.close()

    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE access_audit SET payload = '{\"entity_count\":1}' WHERE seq = 2")
        connection.execute("DELETE FROM access_audit WHERE seq = 3")
    truncated = FeatureRegistry(path, clock=Ticker())
    assert truncated.access_audit.verify().ok
    assert not truncated.access_audit.verify(expected_head=head).ok
    truncated.close()
