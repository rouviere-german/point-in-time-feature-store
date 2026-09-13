from __future__ import annotations

import re
from pathlib import Path

import fakeredis
import pytest
from hypothesis import given
from hypothesis import strategies as st

from point_in_time_feature_store.dsl import DType
from point_in_time_feature_store.offline.synthetic import churn_feature_repo
from point_in_time_feature_store.online.store import (
    SCRIPTS,
    OnlineRow,
    OnlineStore,
    ScriptNotRegisteredError,
    ScriptsNotLoadedError,
    decode_value,
    encode_value,
    meta_key,
    row_key,
)

pytestmark = pytest.mark.unit

VIEW = churn_feature_repo().view(
    "session_stats"
)  # features: engaged, last_session_minutes, pages, sessions_7d
VERSION = VIEW.version_hash()
US = 1_000_000
DAY = 86_400 * US


def store() -> OnlineStore:
    online = OnlineStore(fakeredis.FakeRedis())
    online.load_scripts()
    return online


def row(entity: str, event: int, ingested: int, minutes: float, pages: int = 1) -> OnlineRow:
    return OnlineRow(entity, event, ingested, (minutes > 20, minutes, pages, 1))


def read(online: OnlineStore, entity: str, now: int) -> tuple[str, tuple[object, ...]]:
    value = online.read([entity], [(VIEW, VERSION)], now).values[(entity, VIEW.name)]
    return value.status, tuple(value.values[name] for name in VIEW.feature_names)


def test_newer_rows_replace_older_ones_and_reruns_are_noops() -> None:
    online = store()
    first = online.write_rows(VIEW, VERSION, [row("c1", 10, 10, 5.0)])
    assert (first.written, first.skipped) == (1, 0)
    assert online.write_rows(VIEW, VERSION, [row("c1", 10, 10, 5.0)]).skipped == 1
    assert online.write_rows(VIEW, VERSION, [row("c1", 9, 50, 99.0)]).skipped == 1  # older event
    assert online.write_rows(VIEW, VERSION, [row("c1", 10, 11, 7.0)]).written == 1  # restatement
    assert online.write_rows(VIEW, VERSION, [row("c1", 10, 9, 8.0)]).skipped == 1
    assert read(online, "c1", 20)[1] == (False, 7.0, 1, 1)
    assert online.write_rows(VIEW, VERSION, [row("c1", 12, 12, 30.0)]).written == 1
    assert read(online, "c1", 20) == ("ok", (True, 30.0, 1, 1))


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 2, 0)])
def test_timestamp_ties_resolve_to_the_greatest_values_in_any_order(
    order: tuple[int, int, int],
) -> None:
    candidates = [
        row("c1", 10, 10, 5.0, pages=3),
        row("c1", 10, 10, 5.0, pages=9),
        row("c1", 10, 10, 4.0, pages=99),
    ]
    online = store()
    for index in order:
        online.write_rows(VIEW, VERSION, [candidates[index]])
    # Ordering is (engaged, last_session_minutes, pages, sessions_7d): 5.0/9 beats 5.0/3 and 4.0/99.
    assert read(online, "c1", 20)[1] == (False, 5.0, 9, 1)


def test_only_registered_scripts_can_run_and_missing_scripts_fail_closed() -> None:
    client = fakeredis.FakeRedis()
    online = OnlineStore(client)
    assert not online.scripts_loaded()
    with pytest.raises(ScriptsNotLoadedError):
        online.write_rows(VIEW, VERSION, [row("c1", 1, 1, 1.0)])
    online.load_scripts()
    assert online.scripts_loaded()
    with pytest.raises(ScriptNotRegisteredError):
        OnlineStore._evalsha(client.pipeline(), "flush_everything", [], [])
    assert set(SCRIPTS) == {"conditional_write"}
    with pytest.raises(ValueError, match="do not match"):
        online.write_rows(VIEW, VERSION, [OnlineRow("c1", 1, 1, (True,))])


def test_reads_report_missing_expired_and_not_yet_valid_rows() -> None:
    online = store()
    online.write_rows(VIEW, VERSION, [row("c1", 10 * DAY, 10 * DAY, 25.0)])
    assert read(online, "c1", 11 * DAY)[0] == "ok"
    expired = read(online, "c1", 24 * DAY)  # session_stats TTL is 14 days
    assert expired == ("expired", (None, None, None, None))
    assert read(online, "c1", 9 * DAY)[0] == "missing"  # materialized ahead of the reader's clock
    assert read(online, "nobody", 11 * DAY) == ("missing", (None, None, None, None))


def test_watermark_snapshot_reports_inflight_runs() -> None:
    online = store()
    result = online.read(["c1"], [(VIEW, VERSION)], 0)
    assert result.watermarks[VIEW.name] == (None, None)
    online.begin_materialization(VIEW.name, VERSION, 100)
    assert online.read(["c1"], [(VIEW, VERSION)], 0).watermarks[VIEW.name] == (None, 100)
    online.commit_materialization(VIEW.name, VERSION, 100)
    online.begin_materialization(VIEW.name, VERSION, 200)
    assert online.read(["c1"], [(VIEW, VERSION)], 0).watermarks[VIEW.name] == (100, 200)
    online.commit_materialization(VIEW.name, VERSION, 200)
    assert online.read(["c1"], [(VIEW, VERSION)], 0).watermarks[VIEW.name] == (200, 200)


def test_erasure_reaches_every_version_and_digest_ignores_metadata() -> None:
    online = store()
    old_version = "0" * 64
    online.write_rows(VIEW, VERSION, [row("c1", 1, 1, 1.0), row("c2", 1, 1, 1.0)])
    online.write_rows(VIEW, old_version, [row("c1", 1, 1, 1.0)])
    digest = online.state_digest()
    online.begin_materialization(VIEW.name, VERSION, 5)
    assert online.state_digest() == digest
    versions = {VIEW.name: [old_version, VERSION]}
    assert online.count_entity_keys("c1", versions) == 2
    assert online.delete_entity("c1", versions) == 2
    assert online.count_entity_keys("c1", versions) == 0
    assert online.delete_entity("c1", {}) == 0
    assert online.count_entity_keys("c1", {}) == 0
    assert online.state_digest() != digest
    assert read(online, "c2", 2)[0] == "ok"
    assert row_key("v", VERSION, "c1") == f"fs:v1:v:{VERSION[:16]}:c1"
    assert meta_key("v", VERSION).endswith(":__meta__")


@given(
    st.one_of(
        st.tuples(st.just(DType.FLOAT64), st.floats(allow_nan=False)),
        st.tuples(st.just(DType.INT64), st.integers(-(2**63), 2**63 - 1)),
        st.tuples(st.just(DType.STRING), st.text()),
        st.tuples(st.just(DType.BOOL), st.booleans()),
        st.tuples(st.sampled_from(list(DType)), st.none()),
    )
)
def test_values_round_trip_through_the_redis_encoding(item: tuple[DType, object]) -> None:
    dtype, value = item
    encoded = encode_value(dtype, value)  # type: ignore[arg-type]
    raw = None if encoded is None else encoded.encode("utf-8", errors="surrogatepass")
    if isinstance(value, str) and re.search(r"[\ud800-\udfff]", value):
        return  # lone surrogates are not valid UTF-8 and never reach the store
    assert decode_value(dtype, raw) == value


def test_source_never_calls_eval_or_script_load_outside_the_loader(repo_root: Path) -> None:
    offenders = []
    for path in (repo_root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"\.eval(_ro)?\(", text) or (
            "script_load(" in text and path.name != "store.py"
        ):
            offenders.append(path.name)
    assert offenders == []
