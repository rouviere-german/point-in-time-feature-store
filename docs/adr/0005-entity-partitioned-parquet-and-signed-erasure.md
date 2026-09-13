# 5. Entity-hash partitioned Parquet, no derived copies, signed erasure receipts

- Status: accepted
- Date: 2026-09-13

## Context

Right-to-erasure is hard in feature stores because a person's data spreads into raw sources,
derived feature tables, online caches, request logs, training sets and logs. The spec requires
`DELETE /entities/{id}` to reach every store and to return a signed receipt with before and after
file hashes.

Offline data is usually partitioned by date, which suits time-range scans. Erasing one entity
from date partitions, though, rewrites every partition the entity ever touched.

## Decision

- Sources are partitioned by a stable hash of `entity_id` (`bucket=NN`). Appends add new part
  files; erasure rewrites only the part files of one bucket that contain the entity, deleting a
  part left empty.
- Feature rows are computed from sources on read. No derived Parquet copies exist, so there is
  nothing else to rewrite offline.
- Order of erasure: tombstone (HMAC of the id) in the registry; rewrite Parquet parts (appends and
  rewrites share a store lock, and appends read tombstones while holding it, so an append that
  raced the erasure cannot write the rows back); delete online hashes for every view version ever
  applied; delete request-log rows and scrub the file (`secure_delete`, WAL checkpoint with
  truncation, `VACUUM`); append `erase_entity` to the hash-chained change log; sign a receipt.
  Later appends and materializations drop tombstoned entities.
- Receipts are canonical JSON signed with Ed25519. They contain the entity HMAC, never the id, and
  a key id so verifiers can pick the right public key.
- Application logs carry only HMACs of entity ids, and URL-logging client libraries are held at
  WARNING, so there is nothing to purge from logs. Purge verification queries Parquet, Redis and
  the request log, scans the raw bytes of the request-log and registry SQLite files (with their
  WAL) for the id, and scans captured log text. The raw scan exists because the first version's
  plain SQL `DELETE` left the id in free pages and the WAL while row counts reported clean.

## Consequences

- Erasure cost is proportional to one bucket, not to the entity's lifetime.
- Time-range scans read every bucket; for a store of this size that is cheap, and DuckDB and
  Polars push predicates into Parquet statistics.
- History is rewritten on purpose: training sets that contained the erased entity cannot be
  rebuilt bit-for-bit. Receipts record which files changed.
- Models trained before erasure are not retrained (machine unlearning is a non-goal), and
  external copies (backups, exported training sets, MLflow artifacts written by users) are out of
  scope. The demo writes no training data to disk.
- The tombstone table keeps HMACs forever; rotating the HMAC key would require re-hashing it.
- Each erasure rebuilds the request log with `VACUUM`, which takes time proportional to the log.
- Below SQLite and Parquet nothing is overwritten: replaced part files are unlinked, and
  journaling filesystems, SSD remapping and snapshots may keep old blocks. Redis persistence files
  and replicas are outside the store; the compose Redis runs without RDB or AOF files.
