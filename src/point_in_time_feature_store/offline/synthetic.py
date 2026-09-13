"""Deterministic synthetic churn data with realistic bitemporal defects.

Three sources for one entity (``customer``):

- ``customers``: CRM profile snapshots (plan, fee, support tickets, status, e-mail, country),
  exported daily, so ``ingested_at`` trails ``event_time`` by 2 to 26 hours. A cancellation is a
  profile row with ``status = "cancelled"`` at the churn time, ingested 0.5 to 3 days later.
- ``transactions``: payments with an upstream-computed ``txn_count_30d`` counter. Most rows land
  within two hours; some arrive days or weeks late, some are ingested twice, and some are restated
  (same event time, corrected amount, later ingestion).
- ``sessions``: clickstream sessions whose event time *is* the ingestion time (zero lag), with a
  small share of exact duplicate deliveries.

The population is generated in steady state (customers active at the start are drawn with the
length-biased engagement distribution; new sign-ups arrive at the rate customers churn), so churn
base rates and feature distributions are stationary over the timeline. Activity declines during the
45 days before churn, which gives point-in-time features a legitimate signal; the cancellation row
itself is the leak a naive join picks up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
import polars as pl

from point_in_time_feature_store.definitions import (
    EntityDefinition,
    FeatureDefinition,
    FeatureRepo,
    FeatureView,
    SourceColumn,
    SourceDefinition,
)
from point_in_time_feature_store.dsl import DType, col, when
from point_in_time_feature_store.timeutil import US_PER_DAY, US_PER_SECOND, to_us

__all__ = [
    "SyntheticConfig",
    "SyntheticDataset",
    "churn_feature_repo",
    "generate_dataset",
    "to_timestamp",
]

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

TIERS = ("basic", "standard", "premium")
FEES = np.array([9.99, 19.99, 39.99])
COUNTRIES = ("DE", "FR", "ES", "IT", "NL", "PL")
MERCHANTS = ("grocery", "travel", "electronics", "fashion", "dining", "utilities")
_SECONDS_PER_DAY = 86_400


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    n_customers: int = 14_000  # about 1.0M source rows
    seed: int = 0
    start: datetime = field(default_factory=lambda: datetime(2025, 1, 1, tzinfo=UTC))
    horizon_days: int = 420
    mean_lifetime_days: float = 300.0
    txn_rate_per_day: float = 0.16
    session_rate_per_day: float = 0.05
    decline_days: int = 45
    late_fraction: float = 0.08
    very_late_fraction: float = 0.01
    duplicate_fraction: float = 0.02
    correction_fraction: float = 0.005

    @property
    def start_us(self) -> int:
        return to_us(self.start)

    def day_us(self, day: float) -> int:
        return self.start_us + round(day * US_PER_DAY)


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    config: SyntheticConfig
    sources: dict[str, pl.DataFrame]
    # entity_id, signup_us, churn_us (null when the customer never churns within the horizon)
    truth: pl.DataFrame

    @property
    def total_rows(self) -> int:
        return sum(frame.height for frame in self.sources.values())


def to_timestamp(values: IntArray | pl.Series) -> pl.Series:
    series = values if isinstance(values, pl.Series) else pl.Series(values, dtype=pl.Int64)
    return pl.from_epoch(series, time_unit="us").dt.replace_time_zone("UTC")


def churn_feature_repo(owner: str = "ml-platform") -> FeatureRepo:
    """Definitions used by the demo, the benchmarks and the integration tests."""
    customer = EntityDefinition(name="customer", description="A paying customer account.")
    customers = SourceDefinition(
        name="customers",
        entity="customer",
        columns=(
            SourceColumn(
                name="plan_tier", dtype=DType.STRING, nullable=False, allowed_values=TIERS
            ),
            SourceColumn(name="monthly_fee", dtype=DType.FLOAT64, nullable=False, ge=0, le=1_000),
            SourceColumn(
                name="support_tickets_90d", dtype=DType.INT64, nullable=False, ge=0, le=10_000
            ),
            SourceColumn(
                name="status",
                dtype=DType.STRING,
                nullable=False,
                allowed_values=("active", "cancelled"),
            ),
            SourceColumn(name="email", dtype=DType.STRING, nullable=False, pii=True),
            SourceColumn(
                name="country", dtype=DType.STRING, nullable=False, allowed_values=COUNTRIES
            ),
        ),
        description="CRM profile snapshots, exported daily.",
    )
    transactions = SourceDefinition(
        name="transactions",
        entity="customer",
        columns=(
            SourceColumn(name="amount", dtype=DType.FLOAT64, nullable=False, ge=0, le=100_000),
            SourceColumn(
                name="merchant_category",
                dtype=DType.STRING,
                nullable=False,
                allowed_values=MERCHANTS,
            ),
            SourceColumn(name="txn_count_30d", dtype=DType.INT64, nullable=False, ge=1, le=100_000),
        ),
        max_clock_skew_seconds=120,
        description="Card payments with late, duplicate and restated deliveries.",
    )
    sessions = SourceDefinition(
        name="sessions",
        entity="customer",
        columns=(
            SourceColumn(
                name="duration_minutes", dtype=DType.FLOAT64, nullable=False, ge=0, le=1_440
            ),
            SourceColumn(name="pages", dtype=DType.INT64, nullable=False, ge=1, le=10_000),
            SourceColumn(name="sessions_7d", dtype=DType.INT64, nullable=False, ge=1, le=100_000),
        ),
        description="Clickstream sessions; event time is assigned at ingestion.",
    )
    views = (
        FeatureView(
            name="customer_profile",
            entity="customer",
            source="customers",
            owner=owner,
            description="Latest known CRM profile.",
            features=(
                FeatureDefinition(
                    name="plan_tier", dtype=DType.STRING, expression=col("plan_tier").node
                ),
                FeatureDefinition(
                    name="monthly_fee", dtype=DType.FLOAT64, expression=col("monthly_fee").node
                ),
                FeatureDefinition(
                    name="support_tickets_90d",
                    dtype=DType.INT64,
                    expression=col("support_tickets_90d").node,
                ),
                FeatureDefinition(
                    name="is_cancelled",
                    dtype=DType.BOOL,
                    expression=(col("status") == "cancelled").node,
                ),
                FeatureDefinition(
                    name="country", dtype=DType.STRING, expression=col("country").node
                ),
                FeatureDefinition(
                    name="email",
                    dtype=DType.STRING,
                    expression=col("email").str_lower().node,
                    pii=True,
                    description="Contact address; PII.",
                ),
            ),
        ),
        FeatureView(
            name="transaction_stats",
            entity="customer",
            source="transactions",
            owner=owner,
            ttl_seconds=30 * _SECONDS_PER_DAY,
            description="Most recent payment within 30 days.",
            features=(
                FeatureDefinition(
                    name="last_amount", dtype=DType.FLOAT64, expression=col("amount").node
                ),
                FeatureDefinition(
                    name="amount_log1p",
                    dtype=DType.FLOAT64,
                    expression=col("amount").log1p().round(6).node,
                ),
                FeatureDefinition(
                    name="txn_count_30d", dtype=DType.INT64, expression=col("txn_count_30d").node
                ),
                FeatureDefinition(
                    name="is_large", dtype=DType.BOOL, expression=(col("amount") > 200.0).node
                ),
                FeatureDefinition(
                    name="merchant_category",
                    dtype=DType.STRING,
                    expression=col("merchant_category").node,
                ),
            ),
        ),
        FeatureView(
            name="session_stats",
            entity="customer",
            source="sessions",
            owner=owner,
            ttl_seconds=14 * _SECONDS_PER_DAY,
            zero_ingestion_lag=True,
            description="Most recent session within 14 days.",
            features=(
                FeatureDefinition(
                    name="last_session_minutes",
                    dtype=DType.FLOAT64,
                    expression=col("duration_minutes").node,
                ),
                FeatureDefinition(
                    name="sessions_7d", dtype=DType.INT64, expression=col("sessions_7d").node
                ),
                FeatureDefinition(name="pages", dtype=DType.INT64, expression=col("pages").node),
                FeatureDefinition(
                    name="engaged",
                    dtype=DType.BOOL,
                    expression=when(col("duration_minutes") > 20.0, True, False).node,
                ),
            ),
        ),
    )
    return FeatureRepo(
        entities=(customer,), sources=(customers, transactions, sessions), feature_views=views
    )


def _population(
    config: SyntheticConfig, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Engagement z, sign-up day and churn day (may exceed the horizon) per customer."""

    def mean_lifetime(z: FloatArray) -> FloatArray:
        return np.asarray(config.mean_lifetime_days * np.exp(0.6 * z), dtype=np.float64)

    base = rng.normal(size=config.n_customers * 3)
    weights = mean_lifetime(base)
    active_share = 1.0 / (1.0 + config.horizon_days / float(np.mean(weights)))
    n_active = max(1, round(config.n_customers * active_share))
    n_new = max(0, config.n_customers - n_active)
    z_active = rng.choice(base, size=n_active, p=weights / weights.sum())
    age = rng.exponential(mean_lifetime(z_active))
    signup_active = -age
    churn_active = rng.exponential(mean_lifetime(z_active))
    z_new = rng.normal(size=n_new)
    signup_new = rng.uniform(0.0, config.horizon_days, size=n_new)
    churn_new = signup_new + rng.exponential(mean_lifetime(z_new))
    return (
        np.concatenate([z_active, z_new]),
        np.concatenate([signup_active, signup_new]),
        np.concatenate([churn_active, churn_new]),
    )


def _activity(times: FloatArray, churn: FloatArray, decline_days: int) -> FloatArray:
    """Activity multiplier: 1 until ``decline_days`` before churn, then linearly down to 0.25."""
    remaining = churn - times
    return np.where(
        remaining >= decline_days, 1.0, 0.25 + 0.75 * np.clip(remaining, 0, None) / decline_days
    )


def _events(
    rng: np.random.Generator,
    owner: IntArray,
    window_start: FloatArray,
    window_end: FloatArray,
    rate: FloatArray,
    churn: FloatArray,
    decline_days: int,
) -> tuple[IntArray, FloatArray]:
    """Thinned Poisson events per customer within [start, end): (customer index, day)."""
    length = np.clip(window_end - window_start, 0, None)
    counts = rng.poisson(rate * length)
    who = np.repeat(owner, counts)
    days = np.repeat(window_start, counts) + rng.random(who.size) * np.repeat(length, counts)
    keep = rng.random(who.size) < _activity(days, churn[who], decline_days)
    return who[keep], days[keep]


def _rolling_count(who: IntArray, seconds: IntArray, window_seconds: int) -> IntArray:
    """Per event, the owner's events in (t - window, t], using a composite sorted key."""
    key = who.astype(np.int64) * (1 << 36) + seconds
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    counts_sorted = (
        np.arange(sorted_key.size)
        - np.searchsorted(sorted_key, sorted_key - window_seconds, side="right")
        + 1
    )
    counts = np.empty_like(counts_sorted)
    counts[order] = counts_sorted
    return counts.astype(np.int64)


def _lag_seconds(rng: np.random.Generator, n: int, config: SyntheticConfig) -> IntArray:
    lag = rng.uniform(60, 7_200, size=n)
    roll = rng.random(n)
    late = roll < config.late_fraction
    very_late = roll < config.very_late_fraction
    lag = np.where(late, rng.uniform(1, 4, size=n) * _SECONDS_PER_DAY, lag)
    lag = np.where(very_late, rng.uniform(10, 30, size=n) * _SECONDS_PER_DAY, lag)
    return lag.astype(np.int64)


def generate_dataset(config: SyntheticConfig) -> SyntheticDataset:
    rng = np.random.default_rng(config.seed)
    horizon = float(config.horizon_days)
    z, signup, churn = _population(config, rng)
    keep = churn > -30.0
    z, signup, churn = z[keep], signup[keep], churn[keep]
    n = z.size
    idx = np.arange(n, dtype=np.int64)
    ids = np.array([f"c{i:06d}" for i in range(n)])
    tier = np.digitize(z + rng.normal(scale=0.6, size=n), [-0.3, 0.9]).astype(np.int64)
    country = rng.choice(len(COUNTRIES), size=n, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
    end = np.minimum(churn, horizon)
    base_us = config.start_us

    def day_seconds(days: FloatArray) -> IntArray:
        return np.floor(days * _SECONDS_PER_DAY).astype(np.int64)

    # --- customers: sign-up row, 30-day snapshots, cancellation row -----------------------------
    snapshot_start = np.maximum(signup, -60.0)
    n_snapshots = np.floor(np.clip(end - snapshot_start, 0, None) / 30.0).astype(np.int64)
    snap_who = np.repeat(idx, n_snapshots)
    within = np.arange(snap_who.size) - np.repeat(np.cumsum(n_snapshots) - n_snapshots, n_snapshots)
    snap_day = np.repeat(snapshot_start, n_snapshots) + 30.0 * (within + 1)
    decline = 1.0 - _activity(snap_day, churn[snap_who], config.decline_days)
    tickets = rng.poisson(0.4 + 0.3 * np.exp(-0.5 * z[snap_who]) + 3.0 * decline)
    downgrade = (decline > 0.3) & (rng.random(snap_who.size) < 0.35)
    snap_tier = np.where(downgrade, np.maximum(tier[snap_who] - 1, 0), tier[snap_who])
    signup_rows = signup <= horizon
    cancel_rows = churn <= horizon
    prof_who = np.concatenate([idx[signup_rows], snap_who, idx[cancel_rows]])
    prof_day = np.concatenate([signup[signup_rows], snap_day, churn[cancel_rows]])
    prof_tier = np.concatenate([tier[signup_rows], snap_tier, tier[cancel_rows]])
    prof_tickets = np.concatenate(
        [
            np.zeros(int(signup_rows.sum()), dtype=np.int64),
            tickets,
            rng.poisson(2.0, size=int(cancel_rows.sum())),
        ]
    )
    prof_status = np.concatenate(
        [
            np.zeros(int(signup_rows.sum()) + snap_who.size, dtype=np.int64),
            np.ones(int(cancel_rows.sum()), dtype=np.int64),
        ]
    )
    prof_event_s = day_seconds(prof_day)
    prof_lag_s = rng.uniform(2 * 3_600, 26 * 3_600, size=prof_who.size).astype(np.int64)
    prof_lag_s = np.where(
        prof_status == 1, rng.uniform(0.5, 3.0, size=prof_who.size) * _SECONDS_PER_DAY, prof_lag_s
    ).astype(np.int64)
    customers = pl.DataFrame(
        {
            "entity_id": ids[prof_who],
            "event_time": to_timestamp(base_us + prof_event_s * US_PER_SECOND),
            "ingested_at": to_timestamp(base_us + (prof_event_s + prof_lag_s) * US_PER_SECOND),
            "plan_tier": np.array(TIERS)[prof_tier],
            "monthly_fee": np.round(FEES[prof_tier] * rng.uniform(0.9, 1.0, size=prof_who.size), 2),
            "support_tickets_90d": prof_tickets,
            "status": np.array(["active", "cancelled"])[prof_status],
            "email": np.char.add(np.char.add("User.", ids[prof_who]), "@Example.com"),
            "country": np.array(COUNTRIES)[country[prof_who]],
        }
    )

    # --- transactions -------------------------------------------------------------------------
    txn_rate = config.txn_rate_per_day * np.exp(0.5 * z)
    t_who, t_day = _events(
        rng, idx, np.maximum(signup, -45.0), end, txn_rate, churn, config.decline_days
    )
    t_event_s = day_seconds(t_day)
    t_count = _rolling_count(t_who, t_event_s, 30 * _SECONDS_PER_DAY)
    t_decline = 1.0 - _activity(t_day, churn[t_who], config.decline_days)
    amount = np.round(rng.gamma(2.0, FEES[tier[t_who]] + 15.0) * (1.0 - 0.3 * t_decline), 2)
    merchant = rng.choice(len(MERCHANTS), size=t_who.size)
    t_ingest_s = t_event_s + _lag_seconds(rng, t_who.size, config)
    dup = rng.random(t_who.size) < config.duplicate_fraction
    fix = rng.random(t_who.size) < config.correction_fraction
    tx_who = np.concatenate([t_who, t_who[dup], t_who[fix]])
    tx_event_s = np.concatenate([t_event_s, t_event_s[dup], t_event_s[fix]])
    tx_ingest_s = np.concatenate(
        [
            t_ingest_s,
            t_ingest_s[dup]
            + rng.uniform(3_600, 3 * _SECONDS_PER_DAY, size=int(dup.sum())).astype(np.int64),
            t_ingest_s[fix]
            + rng.uniform(1, 10, size=int(fix.sum())).astype(np.int64) * _SECONDS_PER_DAY,
        ]
    )
    tx_amount = np.concatenate(
        [amount, amount[dup], np.round(amount[fix] * rng.uniform(0.5, 1.5, size=int(fix.sum())), 2)]
    )
    transactions = pl.DataFrame(
        {
            "entity_id": ids[tx_who],
            "event_time": to_timestamp(base_us + tx_event_s * US_PER_SECOND),
            "ingested_at": to_timestamp(base_us + tx_ingest_s * US_PER_SECOND),
            "amount": tx_amount,
            "merchant_category": np.array(MERCHANTS)[
                np.concatenate([merchant, merchant[dup], merchant[fix]])
            ],
            "txn_count_30d": np.concatenate([t_count, t_count[dup], t_count[fix]]),
        }
    )

    # --- sessions (zero ingestion lag) ----------------------------------------------------------
    session_rate = config.session_rate_per_day * np.exp(0.4 * z)
    s_who, s_day = _events(
        rng, idx, np.maximum(signup, -20.0), end, session_rate, churn, config.decline_days
    )
    s_event_s = day_seconds(s_day)
    s_count = _rolling_count(s_who, s_event_s, 7 * _SECONDS_PER_DAY)
    s_decline = 1.0 - _activity(s_day, churn[s_who], config.decline_days)
    minutes = np.round(
        np.clip(rng.gamma(2.0, 7.0, size=s_who.size) * (1.0 - 0.5 * s_decline), 0.5, 1_440), 1
    )
    pages = rng.poisson(4.0, size=s_who.size) + 1
    s_dup = rng.random(s_who.size) < 0.01
    ss_who = np.concatenate([s_who, s_who[s_dup]])
    ss_event = base_us + np.concatenate([s_event_s, s_event_s[s_dup]]) * US_PER_SECOND
    sessions = pl.DataFrame(
        {
            "entity_id": ids[ss_who],
            "event_time": to_timestamp(ss_event),
            "ingested_at": to_timestamp(ss_event),
            "duration_minutes": np.concatenate([minutes, minutes[s_dup]]),
            "pages": np.concatenate([pages, pages[s_dup]]).astype(np.int64),
            "sessions_7d": np.concatenate([s_count, s_count[s_dup]]),
        }
    )

    def shuffled(frame: pl.DataFrame) -> pl.DataFrame:
        return frame[rng.permutation(frame.height).tolist()] if frame.height else frame

    truth = pl.DataFrame(
        {
            "entity_id": ids,
            "signup_us": base_us + day_seconds(signup) * US_PER_SECOND,
            "churn_us": pl.Series(
                np.where(churn <= horizon, base_us + day_seconds(churn) * US_PER_SECOND, 0),
                dtype=pl.Int64,
            ),
        }
    ).with_columns(
        pl.when(pl.col("churn_us") == 0).then(None).otherwise(pl.col("churn_us")).alias("churn_us")
    )
    return SyntheticDataset(
        config=config,
        sources={
            "customers": shuffled(customers),
            "transactions": shuffled(transactions),
            "sessions": shuffled(sessions),
        },
        truth=truth,
    )
