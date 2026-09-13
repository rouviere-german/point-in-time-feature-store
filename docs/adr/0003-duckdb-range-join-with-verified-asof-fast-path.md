# 3. DuckDB range join as the general path, ASOF only after a data check

- Status: accepted
- Date: 2026-09-13

## Context

Historical retrieval must be exact for arbitrary late, duplicated and restated histories, and
fast enough for about a million source rows on a laptop. The spec names a DuckDB range join with
`QUALIFY row_number()` and a uni-temporal ASOF fast path for views that declare zero ingestion
lag.

Options considered:

1. **Polars `join_asof`**: one "on" column plus a `by` key. It cannot express two independent
   upper bounds (`event_time` and `ingested_at`), so ties and late data would need extra passes.
2. **A pure-Python or NumPy sweep per entity**: simple to reason about, too slow at a million rows.
3. **DuckDB range join + window**: expresses the candidate rule directly as join predicates and
   the selection rule as `QUALIFY row_number() OVER (PARTITION BY spine row ORDER BY ...) = 1`.
4. **DuckDB `ASOF JOIN`**: very fast, but it picks a single greatest `event_time` and knows nothing
   about ingestion time, and its choice among equal `event_time` rows is unspecified.

## Decision

- The general path is option 3. Inputs are registered as in-memory Arrow tables; only
  identifiers validated by `sqlbuild.quote_ident` are spliced into SQL, and TTLs are bound
  parameters.
- The fast path is option 4, used only for views declaring `zero_ingestion_lag`, after
  de-duplicating `(entity_id, event_time)` with the full ordering. Before using it, the engine
  counts rows with `ingested_at > event_time`; any such row raises `DataContractError` rather
  than silently leaking.
- Both paths are differential-tested against the pure-Python oracle with hypothesis histories,
  and a mutation test checks that an event-time-only ASOF fails the same property.
- The connection runs with `enable_external_access = false` and a locked configuration, so SQL
  cannot read files or load extensions.

## Consequences

- DuckDB's planner turns the equality on `entity_id` plus the range predicates into a
  `HASH_JOIN` with the inequalities as join conditions (reported by the benchmark from `EXPLAIN`).
  It does not choose IEJoin for this shape; IEJoin applies to joins made only of inequalities.
  The README states the operator actually used.
- DuckDB chooses between `ASOF_JOIN` and a nested loop with a cardinality heuristic
  (`asof_loop_join_threshold`). Registered Arrow tables have no row-count estimate, and the first
  benchmark run planned a `NESTED_LOOP_JOIN`: the "fast" path took 38.75 s against 0.10 s for the
  range join. The engine now sets the threshold to 0; a test asserts the `ASOF_JOIN` plan and CI
  bounds the benchmark time of the fast path.
- With the plan fixed, the "fast" path is still not faster on the synthetic data: for
  `session_stats` (191k rows, 14-day TTL) and a 200k-row spine it took 0.242 s against 0.104 s for
  the range join, because the lag check and the de-duplication window cost more than the small,
  TTL-bounded candidate sets. It is kept because its cost does not grow with the number of rows
  inside the TTL window, which is what dominates views with long or no TTL, and because both paths
  are held to the same oracle. `auto` still selects it for zero-lag views; callers can force
  `range_join`.
- Candidate rows grow with (spine rows per entity) x (rows per entity within TTL). Views with
  long or no TTL over dense histories cost more; DuckDB spills to its temp directory under the
  configured memory limit.
- Single node only: scale-out is a non-goal.
