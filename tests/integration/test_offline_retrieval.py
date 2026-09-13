from __future__ import annotations

import random
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl
import pytest

from point_in_time_feature_store.definitions import (
    EntityDefinition,
    FeatureDefinition,
    FeatureRepo,
    FeatureView,
    SourceColumn,
    SourceDefinition,
)
from point_in_time_feature_store.dsl import DType, col
from point_in_time_feature_store.hashing import EntityHasher
from point_in_time_feature_store.ml.contracts import DataContractError
from point_in_time_feature_store.offline.features import TransformationError
from point_in_time_feature_store.offline.oracle import HistoryRow, SpineRow, oracle_pit
from point_in_time_feature_store.offline.retrieval import (
    SPINE_ROW,
    DuckDBEngine,
    HistoricalRetriever,
    RetrievalError,
    normalize_spine,
)
from point_in_time_feature_store.offline.store import OfflineStore
from point_in_time_feature_store.offline.synthetic import (
    SyntheticConfig,
    SyntheticDataset,
    churn_feature_repo,
    generate_dataset,
    to_timestamp,
)
from point_in_time_feature_store.timeutil import US_PER_DAY

pytestmark = pytest.mark.integration

REPO = churn_feature_repo()
ENTITY = REPO.entity("customer")


@pytest.fixture(scope="module")
def dataset() -> SyntheticDataset:
    return generate_dataset(SyntheticConfig(n_customers=300, seed=11))


@pytest.fixture(scope="module")
def engine() -> Iterator[DuckDBEngine]:
    shared = DuckDBEngine(threads=2, memory_limit="512MB")
    yield shared
    shared.close()


@pytest.fixture
def store(tmp_path: Path, dataset: SyntheticDataset) -> OfflineStore:
    offline = OfflineStore(tmp_path / "offline", buckets=8)
    for name, frame in dataset.sources.items():
        offline.append(REPO.source(name), ENTITY, frame)
    return offline


def test_generated_data_contains_the_defects_under_test(dataset: SyntheticDataset) -> None:
    tx = dataset.sources["transactions"]
    lag = (tx["ingested_at"] - tx["event_time"]).dt.total_seconds()
    assert (lag > 86_400).sum() > 0  # late arrivals
    keys = ["entity_id", "event_time"]
    per_event = tx.group_by(keys).agg(pl.len().alias("n"), pl.col("amount").n_unique().alias("v"))
    assert per_event.filter(pl.col("n") > 1).height > 0  # duplicates and re-ingestions
    assert per_event.filter(pl.col("v") > 1).height > 0  # restatements
    assert not tx["event_time"].is_sorted()  # out of order
    sessions = dataset.sources["sessions"]
    assert (sessions["ingested_at"] == sessions["event_time"]).all()
    cancelled = dataset.sources["customers"].filter(pl.col("status") == "cancelled")
    assert cancelled.height == dataset.truth["churn_us"].is_not_null().sum()


def test_generation_is_deterministic(dataset: SyntheticDataset) -> None:
    again = generate_dataset(SyntheticConfig(n_customers=300, seed=11))
    for name, frame in dataset.sources.items():
        assert frame.equals(again.sources[name])
    other = generate_dataset(SyntheticConfig(n_customers=300, seed=12))
    assert not other.sources["transactions"].equals(dataset.sources["transactions"])


def test_store_layout_fingerprints_and_reopen(tmp_path: Path, store: OfflineStore) -> None:
    files = store.files("transactions")
    assert 1 <= len(files) <= 8
    assert all(path.parent.name.startswith("bucket=") for path in files)
    assert store.bucket_of("c000001") == store.bucket_of("c000001")
    digest = store.dataset_digest(["transactions", "sessions"])
    assert digest == store.dataset_digest(["sessions", "transactions"])
    assert all(not key.startswith("/") for key in store.fingerprints(["customers"]))
    with pytest.raises(ValueError, match="bucket count"):
        OfflineStore(store.root, buckets=4)
    with pytest.raises(ValueError, match="between"):
        OfflineStore(tmp_path / "x", buckets=0)
    with pytest.raises(ValueError, match="invalid source"):
        store.files("../etc")
    empty = OfflineStore(tmp_path / "empty")
    assert empty.read(REPO.source("sessions")).height == 0
    assert empty.entity_row_count("sessions", "c1") == 0
    assert empty.erase_entity("sessions", "c1") == []


def test_append_validates_and_honours_tombstones(tmp_path: Path, dataset: SyntheticDataset) -> None:
    offline = OfflineStore(tmp_path / "offline")
    frame = dataset.sources["sessions"]
    with pytest.raises(DataContractError):
        offline.append(
            REPO.source("sessions"), ENTITY, frame.with_columns(pl.lit(-1).alias("pages"))
        )
    assert offline.files("sessions") == []
    victim = frame["entity_id"][0]
    hasher = EntityHasher.ephemeral()
    with pytest.raises(ValueError, match="hasher"):
        offline.append(REPO.source("sessions"), ENTITY, frame, tombstones=frozenset({"x"}))
    report = offline.append(
        REPO.source("sessions"),
        ENTITY,
        frame,
        hasher=hasher,
        tombstones=frozenset({hasher.entity(victim)}),
    )
    assert report.rows_dropped_tombstoned == frame.filter(pl.col("entity_id") == victim).height
    assert offline.entity_row_count("sessions", victim) == 0
    only_victim = frame.filter(pl.col("entity_id") == victim)
    nothing = offline.append(
        REPO.source("sessions"),
        ENTITY,
        only_victim,
        hasher=hasher,
        tombstones=frozenset({hasher.entity(victim)}),
    )
    assert (nothing.rows_written, nothing.files_written) == (0, 0)


def _history(rows: pl.DataFrame, names: tuple[str, ...]) -> list[HistoryRow]:
    return [
        HistoryRow(r["entity_id"], r["event_time"], r["ingested_at"], tuple(r[n] for n in names))
        for r in rows.iter_rows(named=True)
    ]


@pytest.mark.parametrize("view_name", ["customer_profile", "transaction_stats", "session_stats"])
def test_retrieval_on_generated_data_matches_the_oracle(
    store: OfflineStore, dataset: SyntheticDataset, engine: DuckDBEngine, view_name: str
) -> None:
    retriever = HistoricalRetriever(store, REPO, engine)
    view = REPO.view(view_name)
    rows = retriever.feature_rows(view)
    rng = random.Random(view_name)
    entities = rng.sample(sorted(set(rows["entity_id"].to_list())), 25)
    config = dataset.config
    spine = [
        SpineRow(entity, label, label - rng.choice([0, 0, US_PER_DAY]))
        for entity in entities
        for label in (config.day_us(rng.uniform(-20, 425)) for _ in range(6))
    ]
    history = _history(rows.filter(pl.col("entity_id").is_in(entities)), view.feature_names)
    ttl = None if view.ttl_seconds is None else view.ttl_seconds * 1_000_000
    frame = normalize_spine(
        pl.DataFrame(
            {
                "entity_id": [s.entity_id for s in spine],
                "label_time": [s.label_time for s in spine],
                "as_of": [s.as_of for s in spine],
            }
        )
    )
    got = engine.pit_range_join(frame, rows, view.feature_names, ttl)
    expected = oracle_pit(spine, history, ttl)
    actual = [
        None
        if r["__event_time"] is None
        else (r["__event_time"], r["__ingested_at"], tuple(r[n] for n in view.feature_names))
        for r in got.iter_rows(named=True)
    ]
    assert actual == [
        None if h is None else (h.event_time, h.ingested_at, h.values) for h in expected
    ]
    assert sum(item is not None for item in actual) > len(actual) // 4


def test_zero_lag_view_uses_asof_and_agrees_with_the_range_join(
    store: OfflineStore, engine: DuckDBEngine
) -> None:
    retriever = HistoricalRetriever(store, REPO, engine)
    view = REPO.view("session_stats")
    rows = retriever.feature_rows(view)
    spine = normalize_spine(
        rows.select(
            "entity_id",
            (pl.col("event_time") + pl.col("event_time") % 97_000_000).alias("label_time"),
        )
    )
    fast, path = retriever.join_view(view, spine, rows)
    slow, _ = retriever.join_view(view, spine, rows, path="range_join")
    assert path == "asof"
    assert fast.equals(slow)


def test_a_false_zero_lag_declaration_is_rejected(
    store: OfflineStore, engine: DuckDBEngine
) -> None:
    lagged = REPO.view("transaction_stats").model_copy(update={"zero_ingestion_lag": True})
    repo = REPO.model_copy(
        update={
            "feature_views": tuple(
                lagged if v.name == lagged.name else v for v in REPO.feature_views
            )
        }
    )
    retriever = HistoricalRetriever(store, repo, engine)
    spine = pl.DataFrame(
        {"entity_id": ["c000001"], "label_time": [datetime(2025, 6, 1, tzinfo=UTC)]}
    )
    with pytest.raises(DataContractError, match="transactions"):
        retriever.get_historical_features(spine, ["transaction_stats:last_amount"])


def test_historical_features_frame_contract(
    store: OfflineStore, dataset: SyntheticDataset, engine: DuckDBEngine
) -> None:
    retriever = HistoricalRetriever(store, REPO, engine)
    ids = dataset.truth["entity_id"].head(40).to_list()
    labels = [dataset.config.day_us(200 + i) for i in range(40)]
    spine = pl.DataFrame(
        {"entity_id": ids + ids[:5], "label_time": to_timestamp(pl.Series(labels + labels[:5]))}
    )
    spine = spine.with_columns(pl.int_range(0, pl.len()).alias("row_tag"))
    result = retriever.get_historical_features(
        spine, ["transaction_stats:*", "customer_profile:plan_tier", "customer_profile:plan_tier"]
    )
    frame = result.frame
    assert frame["row_tag"].to_list() == list(range(45))
    assert frame.height == 45
    assert "customer_profile__plan_tier" in frame.columns
    assert frame.schema["transaction_stats__event_time"] == pl.Datetime("us", "UTC")
    assert frame.filter(pl.col("transaction_stats__event_time") > pl.col("label_time")).height == 0
    assert frame.filter(pl.col("customer_profile__ingested_at") > pl.col("label_time")).height == 0
    assert [item.path for item in result.views] == ["range_join", "range_join"]
    with pytest.raises(RetrievalError, match="at least one"):
        retriever.get_historical_features(spine, [])
    with pytest.raises(KeyError):
        retriever.get_historical_features(spine, ["nope:x"])


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pl.DataFrame({"label_time": [1]}), "missing column entity_id"),
        (pl.DataFrame({"entity_id": [1], "label_time": [1]}), "must be a string"),
        (
            pl.DataFrame(
                {"entity_id": ["a"], "label_time": pl.Series([0]).cast(pl.Datetime("us"))}
            ),
            "timezone-aware",
        ),
        (pl.DataFrame({"entity_id": ["a"], "label_time": [1.5]}), "UTC datetime or int64"),
        (pl.DataFrame({"entity_id": ["a", None], "label_time": [1, 2]}), "nulls"),
        (
            pl.DataFrame({"entity_id": ["a"], "label_time": [5], "as_of": [6]}),
            "later than label_time",
        ),
    ],
)
def test_spines_are_validated(frame: pl.DataFrame, message: str) -> None:
    with pytest.raises(RetrievalError, match=message):
        normalize_spine(frame)


def test_the_engine_has_no_external_access(engine: DuckDBEngine, tmp_path: Path) -> None:
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(duckdb.Error):
        engine.run_readonly("SELECT * FROM read_csv(?)", [str(secret)], {})
    with pytest.raises(duckdb.Error):
        engine.run_readonly("SET enable_external_access = true", [], {})
    spine = normalize_spine(pl.DataFrame({"entity_id": ["a"], "label_time": [5]}))
    rows = pl.DataFrame({"entity_id": ["a"], "event_time": [1], "ingested_at": [1], "v": [1]})
    assert any("JOIN" in operator for operator in engine.explain_range_join(spine, rows, ["v"]))


def test_erasure_rewrites_only_the_entitys_bucket(
    store: OfflineStore, dataset: SyntheticDataset
) -> None:
    victim = dataset.sources["transactions"]["entity_id"][0]
    before = store.fingerprints(["transactions"])
    rewrites = store.erase_entity("transactions", victim)
    assert rewrites
    assert {Path(item.partition).parent.name for item in rewrites} == {
        f"bucket={store.bucket_of(victim):02d}"
    }
    assert store.entity_row_count("transactions", victim) == 0
    after = store.fingerprints(["transactions"])
    changed = {key for key in before if before[key] != after.get(key)}
    assert changed == {item.partition for item in rewrites}
    assert all(item.before_sha256 != item.after_sha256 for item in rewrites)


def test_a_part_left_empty_is_deleted(tmp_path: Path, dataset: SyntheticDataset) -> None:
    offline = OfflineStore(tmp_path / "offline")
    frame = dataset.sources["sessions"]
    victim = frame["entity_id"][0]
    offline.append(REPO.source("sessions"), ENTITY, frame.filter(pl.col("entity_id") == victim))
    (rewrite,) = offline.erase_entity("sessions", victim)
    assert rewrite.after_sha256 is None
    assert offline.files("sessions") == []


def test_transformation_failures_on_real_data_are_wrapped(
    tmp_path: Path, engine: DuckDBEngine
) -> None:
    source = SourceDefinition(
        name="codes", entity="customer", columns=(SourceColumn(name="code", dtype=DType.STRING),)
    )
    view = FeatureView(
        name="code_view",
        entity="customer",
        source="codes",
        owner="tests",
        features=(
            FeatureDefinition(
                name="code_int", dtype=DType.INT64, expression=col("code").cast(DType.INT64).node
            ),
        ),
    )
    repo = FeatureRepo(
        entities=(EntityDefinition(name="customer"),), sources=(source,), feature_views=(view,)
    )
    offline = OfflineStore(tmp_path / "offline")
    now = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "entity_id": ["c1", "c2"],
            "event_time": [now, now],
            "ingested_at": [now, now],
            "code": ["12", "x"],
        }
    ).with_columns(pl.col("event_time", "ingested_at").cast(pl.Datetime("us", "UTC")))
    offline.append(source, repo.entity("customer"), frame)
    with pytest.raises(TransformationError, match="code_view"):
        HistoricalRetriever(offline, repo, engine).feature_rows(view)
    assert SPINE_ROW == "__row"


def test_the_asof_fast_path_is_planned_as_an_asof_join(engine: DuckDBEngine) -> None:
    spine = normalize_spine(pl.DataFrame({"entity_id": ["a"], "label_time": [5]}))
    rows = pl.DataFrame({"entity_id": ["a"], "event_time": [1], "ingested_at": [1], "v": [1]})
    plan = engine.run_readonly(
        'EXPLAIN SELECT s."__row" FROM spine_input AS s ASOF LEFT JOIN feature_rows AS f'
        ' ON s."entity_id" = f."entity_id" AND s."as_of" >= f."event_time"',
        [],
        {"spine_input": spine, "feature_rows": rows},
    )
    text = "\n".join(str(value) for value in plan.row(0))
    assert "ASOF_JOIN" in text
    assert "NESTED_LOOP_JOIN" not in text
