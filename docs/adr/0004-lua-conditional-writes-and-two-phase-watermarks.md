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
