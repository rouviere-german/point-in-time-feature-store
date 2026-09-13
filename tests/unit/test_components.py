"""Small components: SQL identifiers, hashing, metrics and drift, catalog, time, request log."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from point_in_time_feature_store import erasure
from point_in_time_feature_store.catalog import render_catalog
from point_in_time_feature_store.hashing import EntityHasher
from point_in_time_feature_store.monitoring import FeatureDriftMonitor, create_feature_store_metrics
from point_in_time_feature_store.offline.synthetic import churn_feature_repo
from point_in_time_feature_store.serving.request_log import LoggedRead, RequestLog, logged_values
from point_in_time_feature_store.sqlbuild import SqlIdentifierError, join_idents, quote_ident
from point_in_time_feature_store.timeutil import from_us, to_us

pytestmark = pytest.mark.unit


@given(st.text(max_size=40))
def test_only_plain_identifiers_can_enter_sql_text(name: str) -> None:
    if re.fullmatch(r"[a-z_][a-z0-9_]{0,127}", name):
        assert quote_ident(name) == f'"{name}"'
    else:
        with pytest.raises(SqlIdentifierError):
            quote_ident(name)


def test_injection_shaped_identifiers_are_rejected() -> None:
    for name in ('a"; DROP TABLE x; --', "a b", "A", "", "x" * 200, "read_csv('x')"):
        with pytest.raises(SqlIdentifierError):
            quote_ident(name)
    assert join_idents(["a", "b"], prefix="f", suffix=" DESC") == '"f"."a" DESC, "f"."b" DESC'


def test_source_never_interpolates_values_into_execute_calls(repo_root: Path) -> None:
    pattern = re.compile(
        r"\.execute(many)?\(\s*(f[\"']|[\"'][^\"']*[\"']\s*%|[\"'][^\"']*[\"']\.format)"
    )
    offenders = [
        path.name
        for path in (repo_root / "src").rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_entity_hashing_is_keyed_and_domain_separated() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        EntityHasher(b"short")
    hasher = EntityHasher(b"k" * 32)
    assert hasher.entity("c1") == EntityHasher(b"k" * 32).entity("c1")
    assert hasher.entity("c1") != EntityHasher(b"j" * 32).entity("c1")
    assert hasher.entity("c1") != hasher.value("c1")
    assert hasher.batch(["a", "b"]) == hasher.batch(["b", "a"])
    assert len(EntityHasher.ephemeral().entity("c1")) == 64


def test_metric_families_and_drift_monitoring() -> None:
    metrics = create_feature_store_metrics()
    rng = np.random.default_rng(0)
    reference = pl.DataFrame(
        {"v__a": rng.normal(size=2_000), "v__b": rng.integers(0, 2, 2_000).astype(bool)}
    )
    current = pl.DataFrame({"v__a": rng.normal(loc=1.0, size=2_000), "v__b": [None] * 2_000})
    monitor = FeatureDriftMonitor(reference, ["v__a", "v__b"])
    result = monitor.check(current.with_columns(pl.col("v__b").cast(pl.Boolean)), metrics)
    assert result.skipped == ("v__b",)
    assert result.report.drifted_features == ("v__a",)
    psi = metrics.registry.get_sample_value("feature_psi", {"view": "v", "feature": "a"})
    assert psi is not None
    assert psi > 0.2
    stable = monitor.check(
        pl.DataFrame({"v__a": rng.normal(size=2_000), "v__b": [True, False] * 1_000})
    )
    assert not stable.report.drifted
    with pytest.raises(ValueError, match="non-null"):
        FeatureDriftMonitor(pl.DataFrame({"x": pl.Series([None], dtype=pl.Float64)}), ["x"])


def test_committed_feature_catalog_is_up_to_date(repo_root: Path) -> None:
    committed = (repo_root / "FEATURE_CATALOG.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == render_catalog(churn_feature_repo())


def test_time_conversions() -> None:
    moment = datetime(2025, 3, 4, 5, 6, 7, 891_011, tzinfo=UTC)
    assert from_us(to_us(moment)) == moment
    assert to_us(moment.astimezone(UTC) + timedelta(hours=0)) == to_us(moment)
    with pytest.raises(ValueError, match="timezone-aware"):
        to_us(datetime(2025, 1, 1))  # noqa: DTZ001


def test_request_log_sampling_storage_and_erasure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        RequestLog(tmp_path / "bad.sqlite", sample_rate=1.5)
    log = RequestLog(tmp_path / "log.sqlite", sample_rate=0.25)
    decisions = [log.should_sample(f"req-{i}") for i in range(4_000)]
    assert decisions == [log.should_sample(f"req-{i}") for i in range(4_000)]
    assert 0.2 < sum(decisions) / len(decisions) < 0.3
    assert RequestLog(tmp_path / "all.sqlite", sample_rate=1.0).should_sample("x")
    view = churn_feature_repo().view("customer_profile")
    hasher = EntityHasher(b"k" * 32)
    values = logged_values(
        view, {"email": "a@example.com", "plan_tier": "basic", "monthly_fee": None}, hasher
    )
    assert values["plan_tier"] == "basic"
    assert values["monthly_fee"] is None
    assert values["email"] == {"hmac": hasher.value("str:'a@example.com'")}
    log.record([])
    reads = [
        LoggedRead(
            f"r{i}", i, "c1" if i % 2 else "c2", view.name, "v" * 64, None, None, "ok", i, values
        )
        for i in range(10)
    ]
    log.record(reads)
    assert log.count() == 10
    sample = log.sample(4, seed=1)
    assert len(sample) == 4
    assert all(item.values == values for item in sample)
    assert log.count_entity("c1") == 5
    assert log.delete_entity("c1") == 5
    assert log.count_entity("c1") == 0
    assert "a@example.com" not in (tmp_path / "log.sqlite").read_bytes().decode("latin-1")
    log.close()


def test_raw_occurrence_scan_counts_across_chunk_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(erasure, "_SCAN_CHUNK_BYTES", 7)
    data = tmp_path / "blob.bin"
    data.write_bytes(b"xxxxxc000123yyc000123c000123zz" * 3)
    assert erasure.count_raw_occurrences([data, tmp_path / "missing"], b"c000123") == 9
    assert erasure.count_raw_occurrences([data], b"") == 0
    assert erasure.count_raw_occurrences([data], b"x") == 15


def _raw_bytes(directory: Path, stem: str) -> bytes:
    return b"".join(path.read_bytes() for path in sorted(directory.glob(f"{stem}*")))


def test_request_log_erasure_leaves_no_raw_bytes_on_disk(tmp_path: Path) -> None:
    """A SQL DELETE leaves deleted cells in free pages and the WAL; erasure must scrub them."""
    log = RequestLog(tmp_path / "log.sqlite", sample_rate=1.0)
    view = churn_feature_repo().view("session_stats")
    victim = "victimEntity0042"
    reads = [
        LoggedRead(
            f"r{i}",
            i,
            victim if i % 3 == 0 else f"other{i:05d}",
            view.name,
            "v" * 64,
            None,
            None,
            "ok",
            i,
            {"pages": i},
        )
        for i in range(600)
    ]
    log.record(reads)
    assert victim.encode("utf-8") in _raw_bytes(tmp_path, "log.sqlite")
    assert log.delete_entity(victim) == 200
    assert log.count_entity(victim) == 0
    assert log.count() == 400
    assert _raw_bytes(tmp_path, "log.sqlite").count(victim.encode("utf-8")) == 0
    assert b"other00001" in _raw_bytes(tmp_path, "log.sqlite")
    log.close()
