from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from point_in_time_feature_store.ml.contracts import DataContractError, validate_source_frame
from point_in_time_feature_store.offline.synthetic import (
    SyntheticConfig,
    churn_feature_repo,
    generate_dataset,
)
from point_in_time_feature_store.timeutil import to_us

pytestmark = pytest.mark.unit

REPO = churn_feature_repo()
ENTITY = REPO.entity("customer")
BASE = datetime(2025, 3, 1, tzinfo=UTC)


def transactions(**columns: object) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "entity_id": ["c000001", "c000002"],
            "event_time": [BASE, BASE + timedelta(hours=1)],
            "ingested_at": [BASE + timedelta(minutes=5), BASE + timedelta(hours=2)],
            "amount": [10.5, 20.0],
            "merchant_category": ["grocery", "travel"],
            "txn_count_30d": [1, 2],
        }
    ).with_columns(pl.col("event_time", "ingested_at").cast(pl.Datetime("us", "UTC")))
    return frame.with_columns([pl.Series(name, values) for name, values in columns.items()])  # type: ignore[arg-type]


def failures(
    frame: pl.DataFrame, source: str = "transactions", min_ingested_at_us: int | None = None
) -> list[dict[str, str | int]]:
    with pytest.raises(DataContractError) as info:
        validate_source_frame(
            REPO.source(source), ENTITY, frame, min_ingested_at_us=min_ingested_at_us
        )
    return info.value.failures


def checks(frame: pl.DataFrame, min_ingested_at_us: int | None = None) -> set[str]:
    failed = failures(frame, min_ingested_at_us=min_ingested_at_us)
    return {str(item["check"]) for item in failed}


def test_generated_sources_satisfy_their_contracts() -> None:
    dataset = generate_dataset(SyntheticConfig(n_customers=150, seed=4))
    for name, frame in dataset.sources.items():
        assert validate_source_frame(REPO.source(name), ENTITY, frame).height == frame.height


def test_violations_are_summarized_without_the_offending_values() -> None:
    frame = transactions(amount=[-123456.789, 5.0], entity_id=["jane.doe@example.com", "c000002"])
    error = failures(frame)
    assert {"amount", "entity_id"} <= {str(item["column"]) for item in error}
    rendered = repr(error)
    assert "123456" not in rendered
    assert "jane.doe" not in rendered


def test_schema_drift_is_rejected() -> None:
    assert failures(transactions().with_columns(pl.lit("x").alias("unexpected")))
    assert failures(transactions().drop("txn_count_30d"))
    assert failures(transactions().with_columns(pl.col("txn_count_30d").cast(pl.Float64)))
    naive = transactions().with_columns(pl.col("event_time").dt.replace_time_zone(None))
    assert failures(naive)
    nanoseconds = transactions().with_columns(pl.col("ingested_at").cast(pl.Datetime("ns", "UTC")))
    assert failures(nanoseconds)


def test_value_rules_are_enforced() -> None:
    assert "not_nan" in checks(transactions(amount=[float("nan"), 1.0]))
    assert failures(transactions(merchant_category=["casino", "travel"]))
    assert failures(transactions(txn_count_30d=[0, 1]))
    nulls = transactions().with_columns(pl.Series("amount", [None, 1.0], dtype=pl.Float64))
    assert failures(nulls)


def test_clock_skew_beyond_the_tolerance_is_rejected() -> None:
    within = transactions(ingested_at=[BASE - timedelta(seconds=120), BASE + timedelta(hours=2)])
    within = within.with_columns(pl.col("ingested_at").cast(pl.Datetime("us", "UTC")))
    assert validate_source_frame(REPO.source("transactions"), ENTITY, within).height == 2
    beyond = transactions(ingested_at=[BASE - timedelta(seconds=121), BASE + timedelta(hours=2)])
    beyond = beyond.with_columns(pl.col("ingested_at").cast(pl.Datetime("us", "UTC")))
    assert "clock_skew" in checks(beyond)
    sessions_skewed = pl.DataFrame(
        {
            "entity_id": ["c1"],
            "event_time": [BASE + timedelta(seconds=1)],
            "ingested_at": [BASE],
            "duration_minutes": [3.0],
            "pages": [2],
            "sessions_7d": [1],
        }
    ).with_columns(pl.col("event_time", "ingested_at").cast(pl.Datetime("us", "UTC")))
    assert "clock_skew" in {
        str(item["check"]) for item in failures(sessions_skewed, source="sessions")
    }


def test_backfills_beyond_the_limit_are_rejected() -> None:
    old = transactions(event_time=[BASE - timedelta(days=401), BASE])
    old = old.with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    assert "max_backfill" in checks(old)


def test_rows_behind_the_materialization_watermark_are_rejected() -> None:
    frame = transactions()
    watermark = to_us(BASE + timedelta(hours=1))
    assert "knowledge_time_append_only" in checks(frame, min_ingested_at_us=watermark)
    later = to_us(BASE)
    assert (
        validate_source_frame(
            REPO.source("transactions"), ENTITY, frame, min_ingested_at_us=later
        ).height
        == 2
    )
