# 4. Online store: Lua conditional writes, registered scripts and two-phase watermarks

- Status: accepted
- Date: 2026-09-13

## Context

Materialization must be idempotent (reruns produce the same store), safe under out-of-order and
overlapping batches, and verifiably consistent with offline retrieval. Readers can overlap a run.
The Redis user the service runs as should not be able to flush data, change configuration or run
arbitrary scripts.

Options considered for writes:

1. `HSET` last-writer-wins: order-dependent; a late batch overwrites newer values.
2. `WATCH`/`MULTI` compare-and-set from the client: correct, but a round trip per row and retry
   storms under contention.
3. A Lua script that compares timestamps server-side and writes atomically.
4. Redis Functions (`FUNCTION LOAD`/`FCALL`): similar to 3, but fakeredis does not implement them,
   so tests would need a real server.

## Decision

- Option 3. `conditional_write` replaces a row only if the incoming `(event_time, ingested_at)` is
  greater. Identical rows are no-ops. If both timestamps tie and values differ, it returns a
  conflict and the client resolves it with the offline value ordering, then overwrites with a
  compare-and-set on the stored digest.
- Scripts live in a code allowlist (`online/store.py: SCRIPTS`) and are called only by `EVALSHA`.
  The client refuses unregistered names and never falls back to `EVAL`. An administrative
  connection loads scripts at deploy time. The generated ACL (`pitfs redis-acl`) gives the
  application user only the commands it needs on `fs:v1:*` keys, with no `EVAL`, `SCRIPT LOAD`,
  `FLUSHALL`, `CONFIG` or `KEYS`. Passwords come from the environment and appear in the ACL file
  only as SHA-256 digests.
- Two-phase watermark: a run sets `inflight` before writing and commits `watermark` after. Reads
  use `MULTI`/`EXEC`, so rows and both watermarks come from one snapshot; the request log keeps
  both, and the skew checker accepts either answer.
- The registry watermark is written after the Redis commit, so a crash between the two causes a
  harmless re-run rather than a gap.
- Concurrent appends: a run records its target (`inflight_watermarks`) in the registry while
  holding the offline store lock, before scanning. Appends read the knowledge floor, the highest
  committed watermark or in-flight target of the source's views, under the same lock and refuse
  rows at or below it. Without this (the first version checked committed watermarks only), a row
  appended between a run's scan and its watermark commit, with `ingested_at` between the old
  watermark and the target, passed the append-only check, was never scanned, and every later
  window started above it. A run that fails before committing releases its target, because the
  next run rescans from the committed watermark.

## Consequences

- Order-independence and idempotency are properties of the script, tested with random
  permutations and random watermark cuts against the oracle.
- After a Redis restart the script cache is empty; writes fail closed (`ScriptsNotLoadedError`)
  and `/readyz` reports `scripts: fail` until an operator reloads them.
- fakeredis does not enforce ACLs, so ACL enforcement is tested against a real Redis in CI only
  (`tests/unit/test_acl.py::test_real_redis_enforces_the_acl`). Locally, the rendered rules are
  validated statically.
- Expired rows are hidden at read time (TTL checked against request time) rather than removed by
  Redis expiry, because synthetic timelines are not wall-clock time. A production deployment
  would add a compaction job.
- Appends wait only while a run records its target, not for the whole run. An append refused
  because a run is in flight can be retried with a later `ingested_at`. A process that crashes
  mid-run leaves its target behind, which keeps refusing rows below it (fail closed) until the
  view is materialized to that target or beyond.
- One materializer per view at a time is assumed and not enforced; runs of different views may
  overlap.
