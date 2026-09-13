"""Property-based differential tests of point-in-time retrieval and online materialization.

Histories contain ties on both timestamps, TTL expiry, late and out-of-order arrivals, exact
duplicate deliveries, re-ingestions and restatements, across several entities. Both DuckDB paths
must equal the pure-Python oracle exactly, and the known-wrong joins must fail the same property.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import fakeredis
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from point_in_time_feature_store.definitions import FeatureDefinition, FeatureView
from point_in_time_feature_store.dsl import DType, col
from point_in_time_feature_store.ml.contracts import DataContractError
from point_in_time_feature_store.offline.baselines import event_time_only_asof, naive_latest_join
from point_in_time_feature_store.offline.oracle import (
    HistoryRow,
    SpineRow,
    Value,
    is_candidate,
    oracle_pit,
)
from point_in_time_feature_store.offline.retrieval import DuckDBEngine, normalize_spine
from point_in_time_feature_store.online.store import OnlineRow, OnlineStore

pytestmark = pytest.mark.unit

FEATURES = ("f_bool", "f_float", "f_int", "f_str")
ROW_SCHEMA = {
    "entity_id": pl.String,
    "event_time": pl.Int64,
    "ingested_at": pl.Int64,
    "f_bool": pl.Boolean,
    "f_float": pl.Float64,
    "f_int": pl.Int64,
    "f_str": pl.String,
}
Result = tuple[int, int, tuple[Value, ...]] | None


@cache
def engine() -> DuckDBEngine:
    return DuckDBEngine(threads=1, memory_limit="256MB")


values_strategy = st.tuples(
    st.none() | st.booleans(),
    st.none() | st.sampled_from([-1.5, 0.0, 2.25, 1e300, float("inf")]),
    st.none() | st.integers(-3, 3),
    st.none() | st.sampled_from(["", "a", "b", "é", "日本"]),
)
entities = st.sampled_from(["a", "b", "c"])
ttls = st.none() | st.integers(1, 15)


@st.composite
def histories(draw: st.DrawFn, *, zero_lag: bool = False) -> list[HistoryRow]:
    lags = st.integers(-3, 0) if zero_lag else st.integers(0, 12)
    base = draw(
        st.lists(
            st.builds(
                lambda e, t, lag, v: HistoryRow(e, t, t + lag, v),
                entities,
                st.integers(0, 30),
                lags,
                values_strategy,
            ),
            max_size=20,
        )
    )
    rows = list(base)

    def ingest(row: HistoryRow, extra: int) -> int:
        shifted = row.ingested_at + extra
        return min(shifted, row.event_time) if zero_lag else shifted

    if base:
        index = st.integers(0, len(base) - 1)
        rows += [base[i] for i in draw(st.lists(index, max_size=3))]  # duplicate deliveries
        for i, extra in draw(
            st.lists(st.tuples(index, st.integers(1, 6)), max_size=3)
        ):  # re-ingestions
            row = base[i]
            rows.append(HistoryRow(row.entity_id, row.event_time, ingest(row, extra), row.values))
        for i, extra, new_values in draw(
            st.lists(st.tuples(index, st.integers(0, 6), values_strategy), max_size=3)
        ):
            row = base[i]  # restatements; extra == 0 ties on both timestamps with different values
            rows.append(HistoryRow(row.entity_id, row.event_time, ingest(row, extra), new_values))
    return list(draw(st.permutations(rows)))


@st.composite
def spines(draw: st.DrawFn) -> list[SpineRow]:
    items = draw(
        st.lists(
            st.tuples(
                entities | st.just("zz"),
                st.integers(-2, 45),
                st.one_of(st.just(0), st.integers(0, 8)),
            ),
            min_size=1,
            max_size=10,
        )
    )
    return [SpineRow(entity, label, label - back) for entity, label, back in items]


def rows_frame(history: Sequence[HistoryRow]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [row.entity_id for row in history],
            "event_time": [row.event_time for row in history],
            "ingested_at": [row.ingested_at for row in history],
            **{name: [row.values[i] for row in history] for i, name in enumerate(FEATURES)},
        },
        schema=ROW_SCHEMA,
    )


def spine_frame(spine: Sequence[SpineRow]) -> pl.DataFrame:
    return normalize_spine(
        pl.DataFrame(
            {
                "entity_id": [row.entity_id for row in spine],
                "label_time": [row.label_time for row in spine],
                "as_of": [row.as_of for row in spine],
            },
            schema={"entity_id": pl.String, "label_time": pl.Int64, "as_of": pl.Int64},
        )
    )


def as_results(frame: pl.DataFrame) -> list[Result]:
    return [
        None
        if record["__event_time"] is None
        else (
            record["__event_time"],
            record["__ingested_at"],
            tuple(record[name] for name in FEATURES),
        )
        for record in frame.iter_rows(named=True)
    ]


def expected(
    spine: Sequence[SpineRow], history: Sequence[HistoryRow], ttl: int | None
) -> list[Result]:
    return [
        None if row is None else (row.event_time, row.ingested_at, row.values)
        for row in oracle_pit(spine, history, ttl)
    ]


@given(history=histories(), spine=spines(), ttl=ttls)
def test_range_join_equals_the_oracle(
    history: list[HistoryRow], spine: list[SpineRow], ttl: int | None
) -> None:
    got = engine().pit_range_join(spine_frame(spine), rows_frame(history), FEATURES, ttl)
    assert as_results(got) == expected(spine, history, ttl)


@given(history=histories(zero_lag=True), spine=spines(), ttl=ttls)
def test_asof_fast_path_equals_the_oracle_for_zero_lag_histories(
    history: list[HistoryRow], spine: list[SpineRow], ttl: int | None
) -> None:
    got = engine().pit_asof(spine_frame(spine), rows_frame(history), FEATURES, ttl)
    assert as_results(got) == expected(spine, history, ttl)


@given(history=histories(), spine=spines(), ttl=ttls)
def test_asof_path_rejects_lagged_data_instead_of_leaking(
    history: list[HistoryRow], spine: list[SpineRow], ttl: int | None
) -> None:
    lagged = any(row.ingested_at > row.event_time for row in history)
    if lagged:
        with pytest.raises(DataContractError) as info:
            engine().pit_asof(spine_frame(spine), rows_frame(history), FEATURES, ttl, source="p")
        assert info.value.failures[0]["check"] == "zero_ingestion_lag"
        return
    got = engine().pit_asof(spine_frame(spine), rows_frame(history), FEATURES, ttl, source="p")
    assert as_results(got) == expected(spine, history, ttl)


@given(history=histories(), spine=spines(), ttl=ttls)
def test_no_value_derives_from_a_row_unknown_at_label_time(
    history: list[HistoryRow], spine: list[SpineRow], ttl: int | None
) -> None:
    known = {(row.entity_id, row.event_time, row.ingested_at, row.values) for row in history}
    got = as_results(
        engine().pit_range_join(spine_frame(spine), rows_frame(history), FEATURES, ttl)
    )
    for spine_row, result in zip(spine, got, strict=True):
        if result is None:
            assert not any(is_candidate(row, spine_row, ttl) for row in history)
            continue
        event_time, ingested_at, values = result
        assert event_time <= spine_row.as_of <= spine_row.label_time
        assert ingested_at <= spine_row.as_of
        assert ttl is None or event_time > spine_row.label_time - ttl
        assert (spine_row.entity_id, event_time, ingested_at, values) in known


@given(
    history=histories(),
    spine=spines(),
    ttl=ttls,
    late=st.lists(
        st.tuples(entities, st.integers(-5, 45), st.integers(1, 5), values_strategy), max_size=6
    ),
)
def test_rows_ingested_after_as_of_never_change_past_answers(
    history: list[HistoryRow],
    spine: list[SpineRow],
    ttl: int | None,
    late: list[tuple[str, int, int, tuple[Value, ...]]],
) -> None:
    horizon = max(row.as_of for row in spine)
    backfills = [
        HistoryRow(entity, event, horizon + delay, vals) for entity, event, delay, vals in late
    ]
    before = engine().pit_range_join(spine_frame(spine), rows_frame(history), FEATURES, ttl)
    after = engine().pit_range_join(
        spine_frame(spine), rows_frame(history + backfills), FEATURES, ttl
    )
    assert as_results(before) == as_results(after)


@given(history=histories(), spine=spines(), ttl=ttls, data=st.data())
def test_results_do_not_depend_on_row_order_or_duplicate_deliveries(
    history: list[HistoryRow], spine: list[SpineRow], ttl: int | None, data: st.DataObject
) -> None:
    reordered = data.draw(st.permutations(history)) + history
    first = engine().pit_range_join(spine_frame(spine), rows_frame(history), FEATURES, ttl)
    second = engine().pit_range_join(spine_frame(spine), rows_frame(reordered), FEATURES, ttl)
    assert as_results(first) == as_results(second)


def _mutant_property(join: str) -> None:
    @settings(max_examples=300, derandomize=True, database=None, report_multiple_bugs=False)
    @given(history=histories(), spine=spines(), ttl=ttls)
    def prop(history: list[HistoryRow], spine: list[SpineRow], ttl: int | None) -> None:
        rows = rows_frame(history)
        if join == "event_time_only":
            got = event_time_only_asof(engine(), spine_frame(spine), rows, FEATURES, ttl)
        else:
            got = naive_latest_join(engine(), spine_frame(spine), rows, FEATURES, snapshot_us=100)
        assert as_results(got) == expected(spine, history, ttl)

    prop()


@pytest.mark.parametrize("join", ["event_time_only", "naive_latest"])
def test_known_wrong_joins_fail_the_oracle_property(join: str) -> None:
    """Mutation test: the property must be strong enough to reject leaking joins."""
    with pytest.raises(AssertionError):
        _mutant_property(join)


def test_event_time_only_asof_leaks_a_late_arriving_value() -> None:
    history = [HistoryRow("a", 5, 20, (True, 1.0, 1, "late"))]
    spine = [SpineRow("a", 10, 10)]
    leaked = event_time_only_asof(engine(), spine_frame(spine), rows_frame(history), FEATURES, None)
    assert as_results(leaked) == [(5, 20, (True, 1.0, 1, "late"))]
    assert expected(spine, history, None) == [None]


# --------------------------------------------------------------------------- online parity

US = 1_000_000
VERSION = "f" * 64


def prop_view(ttl_seconds: int | None) -> FeatureView:
    dtypes = {
        "f_bool": DType.BOOL,
        "f_float": DType.FLOAT64,
        "f_int": DType.INT64,
        "f_str": DType.STRING,
    }
    return FeatureView(
        name="prop_view",
        entity="customer",
        source="prop_source",
        owner="tests",
        ttl_seconds=ttl_seconds,
        features=tuple(
            FeatureDefinition(name=name, dtype=dtype, expression=col(name).node)
            for name, dtype in dtypes.items()
        ),
    )


def _write(store: OnlineStore, view: FeatureView, rows: Sequence[HistoryRow]) -> None:
    store.write_rows(
        view,
        VERSION,
        [
            OnlineRow(row.entity_id, row.event_time * US, row.ingested_at * US, row.values)
            for row in rows
        ],
    )


def _visible(row: HistoryRow) -> int:
    return max(row.event_time, row.ingested_at)


@settings(max_examples=30)
@given(
    history=histories(),
    cuts=st.lists(st.integers(0, 45), min_size=1, max_size=4),
    ttl=ttls,
    reads=st.lists(st.tuples(entities, st.integers(0, 10)), min_size=1, max_size=5),
    data=st.data(),
)
def test_incremental_out_of_order_materialization_matches_offline_as_of_the_watermark(
    history: list[HistoryRow],
    cuts: list[int],
    ttl: int | None,
    reads: list[tuple[str, int]],
    data: st.DataObject,
) -> None:
    view = prop_view(ttl)
    scaled = [
        HistoryRow(r.entity_id, r.event_time * US, r.ingested_at * US, r.values) for r in history
    ]
    store = OnlineStore(fakeredis.FakeRedis())
    store.load_scripts()
    previous: int | None = None
    for watermark in sorted(set(cuts)):
        batch = [
            row
            for row in history
            if _visible(row) <= watermark and (previous is None or _visible(row) > previous)
        ]
        _write(store, view, data.draw(st.permutations(batch)))  # every row, in any order
        previous = watermark
        for entity, offset in reads:
            now = watermark + offset
            got = store.read([entity], [(view, VERSION)], now * US).values[(entity, view.name)]
            (want,) = oracle_pit(
                [SpineRow(entity, now * US, watermark * US)],
                scaled,
                None if ttl is None else ttl * US,
            )
            if want is None:
                assert got.status in {"missing", "expired"}
                assert all(value is None for value in got.values.values())
            else:
                assert got.status == "ok"
                assert (got.event_time_us, got.ingested_at_us) == (
                    want.event_time,
                    want.ingested_at,
                )
                assert tuple(got.values[name] for name in FEATURES) == want.values
    one_shot = OnlineStore(fakeredis.FakeRedis())
    one_shot.load_scripts()
    assert previous is not None
    _write(one_shot, view, [row for row in history if _visible(row) <= previous])
    assert one_shot.state_digest() == store.state_digest()
    digest = store.state_digest()
    _write(store, view, [row for row in history if _visible(row) <= previous])  # rerun
    assert store.state_digest() == digest
