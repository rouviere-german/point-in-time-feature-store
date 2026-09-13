# 2. Bitemporal visibility and a content-based tie-break

- Status: accepted
- Date: 2026-09-13

## Context

Point-in-time retrieval has to answer "what did we know about this entity at prediction time".
Two timestamps are needed for that: when a fact happened (`event_time`) and when the store learned
it (`ingested_at`). Many feature stores join on event time only; that is correct for data that is
never late, and leaks for data that is (a cancellation effective yesterday but processed
tomorrow). The spec also requires online serving to agree with offline recomputation, and online
stores lag the offline store by their materialization watermark.

Ties are the second problem. Real feeds deliver duplicates and restatements with identical
timestamps. If ties are broken by file order, row position or an ingestion sequence number, the
same query can return different values after a compaction, an erasure rewrite or a replay.

## Decision

- A row is visible at `max(event_time, ingested_at)`. A request carries `label_time` (anchors the
  TTL) and `as_of <= label_time` (the knowledge cutoff, default `label_time`). The candidate rule
  and selection order are specified in `docs/PIT_SEMANTICS.md`.
- The selected row maximizes `(event_time, ingested_at, feature values...)` with nulls smallest.
  The tie-break is a function of row content only.
- The same ordering is implemented three times: DuckDB `ORDER BY ... DESC NULLS LAST`, the Python
  oracle's sort key, and the online store's tie resolution. The oracle is the reference, and
  property tests compare all of them against it.
- `as_of` lets the skew checker ask the exact question the online store answers:
  `as_of = watermark`, `label_time = request time`.

## Consequences

- Answers are independent of file order, duplicate deliveries and partition rewrites, and
  backfills never change answers as of an earlier cutoff (property-tested).
- Online/offline parity can be checked exactly rather than "within tolerance".
- The tie-break needs all feature values of a view, so every store keeps whole rows. Selecting a
  subset of features never changes which row is used.
- Floats must be normalized (`NaN` to null, `-0.0` to `0.0`) or the three implementations would
  order them differently. Strings compare by code point.
- Alternatives considered: an ingestion sequence number (not stable under rewrites and replays),
  last-writer-wins by arrival (non-deterministic under concurrent loads), and event-time-only
  joins (leak late data; kept as `offline/baselines.py` to show the failure).
