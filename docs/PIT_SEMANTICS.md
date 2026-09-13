# Point-in-time semantics

This document is the specification that `offline/oracle.py` implements literally and that both
DuckDB paths, the online store and the skew checker are tested against
(`tests/unit/test_pit_properties.py`).

## Time model

- Every timestamp is an int64 count of microseconds since 1970-01-01 UTC. Sources store
  `Datetime("us", "UTC")`; naive or nanosecond timestamps are contract failures.
- A source row carries two times:
  - `event_time`: when the fact became true in the world;
  - `ingested_at`: when the store learned about it.
- A row is **visible** from `visible_at = max(event_time, ingested_at)` onwards.

## Retrieval request

A spine row is `(entity_id, label_time, as_of)`:

- `label_time` is the prediction time and anchors the TTL;
- `as_of` is the knowledge cutoff. It defaults to `label_time` and may not be later
  (a spine with `as_of > label_time` is rejected: that would be knowing the future).

## Candidate rule

For a feature view with TTL `ttl` (or no TTL), a row `r` is a candidate for spine row `s` when:

```text
r.entity_id   == s.entity_id
r.event_time  <= s.as_of
r.ingested_at <= s.as_of
r.event_time  >  s.label_time - ttl        (omitted when the view has no TTL)
```

With the default `as_of = label_time` this is exactly "the row's event had happened, and the row
had been ingested, by label time, and it is younger than the TTL".

Boundaries:

- Both upper bounds are inclusive: a row ingested at exactly `as_of` is visible.
- The TTL bound is strict: a value whose age is exactly `ttl` has expired.

## Selection and ties

The selected row is the candidate with the greatest key

```text
(event_time, ingested_at, v1, v2, ..., vn)
```

where `v1..vn` are the view's feature values in sorted feature-name order and each value compares
with null smaller than any non-null value. Consequences:

- **Restatements**: a correction with the same `event_time` and a later `ingested_at` wins as soon
  as it has been ingested, and not before.
- **Duplicates**: rows with identical keys have identical feature values, so it does not matter
  which one is chosen. Results never depend on file order or on duplicate deliveries.
- **Conflicting duplicates** (same event time and ingestion time, different values) resolve
  deterministically by value, independent of file order.
- The ordering uses all features of the view, not only the requested ones, so asking for a subset
  of features never changes which row is selected.

Value normalization makes the ordering identical in every store: float `NaN` becomes null, `-0.0`
becomes `0.0`, strings compare by code point (which equals UTF-8 byte order in DuckDB).

## Late, backfilled and skewed data

- **Late arrivals** (`ingested_at` well after `event_time`) are invisible to label times before
  their ingestion, even though their event time is earlier.
- **Backfills never change past answers**: adding rows ingested after `as_of` leaves every answer
  for that `as_of` unchanged, so a training set built as of a cutoff can be rebuilt later
  (`test_rows_ingested_after_as_of_never_change_past_answers`). Erasure is the one deliberate
  exception (see below).
- **Clock skew**: a source declares `max_clock_skew_seconds`; rows with
  `event_time > ingested_at + skew` are rejected by the contract. Tolerated skew is harmless
  because visibility uses the later of the two times.
- **Old backfills** beyond `max_backfill_days` are rejected by the contract.

## The ASOF fast path

A view may declare `zero_ingestion_lag`, meaning no row is ingested after its event time
(`ingested_at <= event_time`). Then `event_time <= as_of` implies `ingested_at <= as_of`, so a
uni-temporal `ASOF JOIN` on `event_time` is exact once rows sharing `(entity_id, event_time)` are
reduced to the greatest one under the ordering above. The declaration is verified against the
data on every retrieval; a single lagged row raises `DataContractError` instead of leaking
(`test_asof_path_rejects_lagged_data_instead_of_leaking`).

## Transformations

Feature transformations are row-level expression trees with no aggregation, window or shift
nodes, so a feature value depends only on its own source row. Time-window aggregates must be
computed upstream and arrive as rows with their own event and ingestion times (the synthetic
sources carry `txn_count_30d` and `sessions_7d` this way).

## Online semantics and staleness

Materialization run *k* reads the rows with `visible_at` in `(W[k-1], W[k]]`, writes the greatest
row per entity through a Lua script that only replaces a stored row with a greater key, and then
commits watermark `W[k]`. By induction, after the commit the online row of every entity is the
point-in-time answer as of `W[k]`.

An online read at request time `t` returns

```text
PIT(entity, label_time = t, as_of = min(W, t))
```

where `W` is the committed watermark, with the TTL checked against `t`. The documented staleness
is therefore exactly `t - W`: values can miss rows that became visible after the last
materialization, and nothing else. While a run is in flight, the reader snapshot (MULTI/EXEC)
carries both the committed and the in-flight watermark, and each entity is at one of the two;
the skew checker accepts either and nothing in between.

Once a view is materialized up to `W`, or while a run up to `W` is in flight, the offline store
refuses rows with `ingested_at <= W` for its source (`knowledge_time_append_only`), because such
rows would be visible offline but never reach the online store. The in-flight case matters because
a run scans the store before it commits `W`. A run therefore records its target in the registry
while holding the offline store lock, before scanning, and appends read this knowledge floor (the
highest committed watermark or in-flight target of the source's views) under the same lock. An
append either completed before the scan, which then includes it, or is refused. A run that fails
before committing releases its target. At most one materializer per view runs at a time
(`test_appends_during_an_inflight_materialization_are_refused_not_lost`,
`test_a_failed_materialization_releases_its_knowledge_floor`).

## Erasure

Erasing an entity rewrites history: its rows are removed from every Parquet partition, so
retrieval as of any past time returns nulls for it. This is the intended trade-off of
right-to-erasure; training sets that included the entity cannot be rebuilt bit-for-bit
afterwards, and the signed receipt records the before and after hashes of every rewritten file.
