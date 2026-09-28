# Point-in-Time Feature Store

A feature store for machine learning that will not let a model see data from the future, and
proves it with tests rather than asking you to trust it.

## The problem this solves

To train a model you need a table of "what we knew about this customer at this exact moment in
the past". Building that table is harder than it sounds, because facts arrive late. A
cancellation effective on Monday gets processed on Wednesday. A payment is restated a week later.
If the training table quietly uses the Wednesday version of Monday's row, the model is being shown
the answer.

Nothing fails when that happens. There is no error and no crash. The model scores beautifully in
testing and then behaves like a coin flip on live traffic. This repository measures exactly that:
a churn model trained with the naive "just take the latest value" join scores a perfect **1.0**
offline and **0.4859** on genuinely unseen future data. Trained through this store, the same model
scores **0.8001** offline and **0.8080** on the holdout - honest, and roughly the same number
twice. (AUC: 1.0 is perfect, 0.5 is a coin flip.)

Who that hurts depends on the model. A credit model approves people it should decline. A fraud
model misses fraud that its test score promised it would catch. A churn model sends retention
budget to the wrong customers. In every case the team finds out weeks later, from the business
metric, not from the test suite.

Two related failures come with it. **Training-serving skew**: the live service returns slightly
different values than the ones the model was trained on, because the cache is stale or the code
paths diverged. And **the right to erasure**: once a person's data is copied into raw files, a
cache, a request log and a training set, deleting them everywhere - and proving you did - becomes
a real engineering problem.

This repository treats all three as correctness properties with tests attached, not as
conventions people are asked to follow.

## What this does

- **Answers "what did we know then?" correctly.** Every fact carries two timestamps: when it
  happened, and when this system found out. A lookup as of last Tuesday ignores anything learned
  after last Tuesday, even if the event itself is older.
- **Checks itself against a slow, obviously-correct reference.** A deliberately simple Python
  implementation (an *oracle* - a plain version written for clarity, not speed) recomputes the
  same answers. Randomly generated awkward histories - duplicates, corrections, late deliveries,
  clock skew - are run through both, and they must agree exactly.
- **Proves the fast cache matches the slow truth.** Served values are logged, then re-derived
  offline as of the exact moment they were served, and compared. The benchmark checks 30,000
  served reads and finds 0 disagreements. The logging is sampled, so this detects a systematic
  problem, not necessarily a single tampered row.
- **Erases a person from every store it owns, and signs a receipt.** One delete call reaches the
  raw files, the cache and the request log - down to the raw bytes of the log database - and
  returns a signed document listing exactly what changed. Models already trained on that person
  are out of scope, as listed under Non-goals.
- **Refuses a risky definition change without a human.** Removing a feature, redefining one,
  changing how long a value stays valid or dropping a privacy tag all require someone to approve
  that specific change before it applies.

## Architecture

```mermaid
flowchart LR
    subgraph definitions["Definitions"]
        repo["FeatureRepo JSON<br/>entities, sources, views"] --> plan["plan / apply<br/>destructive = approve + plan hash"]
        plan --> registry[("SQLite registry<br/>versions, watermarks,<br/>tombstones, hash chains")]
    end
    subgraph offline["Offline"]
        feeds["Synthetic feeds<br/>late, duplicate, restated rows"] --> contracts["pandera contracts"]
        contracts --> parquet[("Parquet<br/>entity-hash buckets")]
        parquet --> features["Row-level Polars<br/>expression trees"]
        features --> join{{"DuckDB<br/>range join + QUALIFY<br/>or verified ASOF"}}
        spine["Label spine<br/>entity, label_time, as_of"] --> join
        oracle["Pure-Python oracle"] -. "hypothesis differential tests" .- join
        join --> training["LightGBM churn model<br/>naive vs PIT, MLflow lineage"]
    end
    subgraph online["Online"]
        parquet --> materializer["Materializer<br/>visible_at watermark windows"]
        materializer -- "EVALSHA conditional_write" --> redis[("Redis hashes<br/>ACL user")]
        api["FastAPI<br/>JWT scopes, PII omission"] -- "MULTI/EXEC snapshot" --> redis
        api --> reqlog[("Request log<br/>PII as HMAC")]
        reqlog --> skew["Skew checker<br/>recompute as of watermark"]
        skew --> join
    end
    subgraph erasure["Erasure"]
        delete["DELETE /v1/entities/{id}"] --> tomb["Tombstone"] --> rewrite["Rewrite partitions"]
        rewrite --> dropkeys["Delete keys of every view version"] --> droplog["Delete request-log rows<br/>scrub the SQLite file"]
        droplog --> receipt["Ed25519 receipt<br/>before/after file hashes"]
    end
    registry --> join
    registry --> materializer
    registry --> api
```

### How a live read flows

A scoring service needs features for three customers, right now:

1. **The token is checked.** The caller presents a short-lived signed token. The signature
   algorithm is pinned, and the token must name its issuer, audience and subject and be at most an
   hour old.
2. **Permissions are turned into a plan.** Each feature group needs its own read scope. Features
   tagged as personal data need an extra scope; without it they are dropped from the plan and
   named in an `omitted` list, so the caller knows something was withheld rather than missing.
3. **Entity ids are validated** against a fixed format, and the batch is capped in both customers
   and features.
4. **One snapshot read.** The cache returns the rows and both watermarks - how far materialization
   has committed, and how far an in-progress run intends to go - inside a single transaction, so
   the values and the "how current is this" markers cannot disagree.
5. **Expiry is applied at read time.** A value older than its view's time-to-live is reported as
   expired rather than returned.
6. **The response carries its own provenance.** Each group comes back with a status, the event
   time of the value, and its freshness in seconds.
7. **The read is logged with the ids hashed.** A sampled copy goes to the request log with
   personal values stored only as keyed hashes, and the watermarks recorded alongside.
8. **The attempt is appended to a tamper-evident audit chain** - granted or refused, with the
   reason, never the values.

Later, the skew checker replays step 7's log entries: it recomputes each logged read offline as of
the watermark that was in force at the time, and every value must match exactly.

### How a training table is built

1. You supply a **label spine**: one row per `(customer, prediction time)` you want to train on.
2. For each spine row the store finds every source row where the event had happened, the store had
   already ingested it, and it is younger than the view's time-to-live.
3. Among those candidates it picks one by a fixed ordering - newest event, then newest ingestion,
   then the feature values themselves as the final tie-break.
4. That ordering depends only on row content, never on file order, so compaction, replays and
   duplicate deliveries cannot change the answer.
5. The same query is run by the Python oracle. In tests, the two must be identical.
6. The registry records which feature-view versions and which source file hashes went into the
   table, so the run can be traced later.

## Results

Measured with `uv run python bench/bench_baseline.py --check` on 2026-09-13: Windows 10
(10.0.19045), 6 logical CPUs, Python 3.12.14, DuckDB limited to 2 threads and 1 GB. The run
printed "all benchmarks within thresholds"; CI runs the same command against
`bench/thresholds.json`. These are single-machine numbers for catching regressions, not
performance claims about other hardware.

| What was measured | Result | What it means | Command |
|---|---|---|---|
| Leakage gap, naive join | **0.5141** | Offline 1.0, real future 0.4859. The offline score carried no information about future performance at all. | `uv run python bench/bench_baseline.py --only leakage_5_seeds --check` |
| Leakage gap, this store | **-0.0079** | Offline 0.8001, real future 0.8080. The score you see in testing is the score you get. | same command |
| Online/offline parity | **0 mismatches** in 30,000 checked reads | Nothing the live service served differed from a fresh offline recomputation. | `uv run python bench/bench_baseline.py --only parity_10k_requests --check` |
| Online read latency, p99 | **4.135 ms** | The slowest 1 read in 100 - 50 features each - still answered in 4.135 ms. Against an in-process fake cache, so no network time is included. | `uv run python bench/bench_baseline.py --only online_get_50_features --check` |
| Historical retrieval, 1M rows | **1.892 s** | A 200,000-row training table over 1,036,748 source rows, built in under two seconds on a laptop. | `uv run python bench/bench_baseline.py --only retrieval_1m_rows` |
| Materialization idempotency | identical store digests | Loading the cache in four steps, in one step, or a second time with overlap all leave the store in the same state. | `uv run python bench/bench_baseline.py --only materialization_idempotency` |

**Correctness is established by tests, not by timing.** Both retrieval paths equal the oracle on
generated histories (60 examples per property with the default profile, 200 with
`HYPOTHESIS_PROFILE=ci`; the online materialization property is capped at 30) and on the synthetic
data, and the mutation test fails for event-time-only and naive joins. Every claim maps to a named
test in [CLAIMS.md](CLAIMS.md).

### Leakage in detail

`leakage_5_seeds`: 5 seeds, 10,000 customers each, mean holdout 7,064 rows. Gap = offline AUC
minus holdout AUC; a large positive gap means the offline score was a lie. Details in
[MODEL_CARD.md](MODEL_CARD.md).

| Join used for training features | Offline AUC | Holdout AUC | Period control AUC | Gap (mean of 5) | Gap range | 95% CI (mean bounds) |
|---|---:|---:|---:|---:|---|---|
| naive latest value | 1.0 | 0.4859 | 0.474 | **0.5141** | 0.4355 to 0.6595 | 0.4865 to 0.5411 |
| ASOF on event time only | 0.8072 | 0.8089 | 0.8025 | -0.0017 | -0.0026 to -0.0001 | -0.0311 to 0.0243 |
| point-in-time (this store) | 0.8001 | 0.8080 | 0.8026 | **-0.0079** | -0.0144 to -0.0058 | -0.0379 to 0.0192 |

The CI is a 95% percentile interval from a bootstrap that resamples customers (all of a customer's
label rows together) in the validation and holdout sets, 200 resamples per seed. The period control
scores the validation customers in the holdout period with point-in-time features, so it shares its
customers with the offline AUC and its period with the holdout AUC.

The spec's targets (naive gap >= 0.05, point-in-time gap <= 0.01) hold. The naive gap is large by
construction: under a latest-value join `customer_profile:is_cancelled` encodes the label, so
offline AUC is 1.0; its lowest per-seed CI bound is 0.4076. The point-in-time gap is negative for
every seed, but every seed's interval contains 0 (`pit_gap_ci95_contains_zero_seeds: 5`), and the
period control (0.8026) sits between offline (0.8001) and holdout (0.8080): about 0.0025 of the
difference is between periods and 0.0054 between the validation and holdout customers, both within
resampling noise. The event-time-only join leaks nothing measurable on this data because most
ingestion lag is hours while labels look 30 days ahead; the property and mutation tests, not this
experiment, are what show it is wrong.

### Parity, idempotency and throughput in detail

**Parity** (`parity_10k_requests`): 10,000 online requests over 3,000 customers, served across four
incremental materializations with late data in between (1,000 of the logged reads taken while a
materialization was in flight); 30,000 logged reads rechecked offline in 1.37 s: **0 mismatches**.

**Idempotency** (`materialization_idempotency`). The point of this table is the right-hand column:
three different loading strategies, one identical store.

| Run over 1,023,768 source rows, 3 views, into fakeredis | Time | Store digest |
|---|---:|---|
| 4 incremental runs (day 100, 200, 300, horizon) | 40.603 s | reference |
| 1 one-shot run to the horizon | 16.895 s | identical to reference |
| 4 runs again with a 500-day overlap over the same store | 41.448 s | identical to reference |

**Online latency** (`online_get_50_features`): one read of 50 features across 5 views through the
service layer (permission planning, `MULTI/EXEC` snapshot read, decoding, freshness), 5,000 reads
against **fakeredis in-process** (no network): p50 1.759 ms, p95 3.346 ms, **p99 4.135 ms**
(target < 10 ms). The access-audit write and HTTP handling are not included. CI repeats the
benchmark against a Redis service container.

**Historical retrieval on 1M rows** (`retrieval_1m_rows`): 1,036,748 source rows (generated in
1.107 s, validated and written in 1.582 s); every feature of the three views for a random
200,000-row spine in **1.892 s** (105,708 spine rows/s, 21 feature columns). DuckDB planned the
range join as `HASH_JOIN`. For `session_stats` the ASOF fast path took 0.323 s and the range join
0.111 s, with identical results. The first benchmark run measured 38.75 s for the ASOF path because
DuckDB had planned a nested loop; the engine now forces the `ASOF_JOIN` plan (see
[ADR 0003](docs/adr/0003-duckdb-range-join-with-verified-asof-fast-path.md)).

## How it works

### Components

| Component | Responsibility |
|---|---|
| Definitions (`dsl.py`, `definitions.py`) | Entities, sources and feature views as pydantic models. A transformation is a JSON expression tree over a closed node set, compiled to Polars. No user SQL, no aggregation: a value can only depend on its own row. |
| Registry (`registry.py`, `hashchain.py`) | SQLite. View versions, watermarks, tombstones, training runs, and two hash-chained logs (definition changes, feature access). |
| Offline store (`offline/`) | Parquet partitioned by a hash of the entity id. Retrieval, the oracle, and the deliberately wrong baselines kept for the mutation test. |
| Online store (`online/`) | Redis hashes written through a registered Lua script; ACL rendering; watermark bookkeeping. |
| Serving (`serving/`) | FastAPI app, JWT verification, scope planning, snapshot reads, request log. |
| Parity, erasure, training, monitoring | The skew checker, the erasure pipeline with signed receipts, the LightGBM leakage experiment with MLflow lineage, and drift metrics. |

### Point-in-time retrieval

For a spine row `(entity, label_time, as_of)` (by default `as_of = label_time`) a source row is a
candidate when `event_time <= as_of`, `ingested_at <= as_of` and `event_time > label_time - TTL`.
The chosen row maximizes `(event_time, ingested_at, feature values)` with nulls smallest, a
content-only tie-break that does not depend on file order. The rules for ties, TTL boundaries, late
data, clock skew and staleness are in [docs/PIT_SEMANTICS.md](docs/PIT_SEMANTICS.md).

The general path is a DuckDB range join with `QUALIFY row_number() ... = 1`. Views that declare
zero ingestion lag can use an `ASOF JOIN`, but only after the data is checked for lagged rows; if
any exist, retrieval raises a contract error instead of leaking.

### The key trade-off

**Two timestamps instead of one.** The obvious approach - and what most as-of joins do - is to join
on event time alone. It is simpler, and every mainstream engine has a fast primitive for it
(`ASOF JOIN`). It is also wrong for any feed that can be late, which is most of them: a
cancellation with Monday's event time that arrived on Wednesday will be joined onto Monday's
training row even though nobody knew about it on Monday.

Tracking ingestion time as well costs you that primitive. Two independent upper bounds do not fit
an `ASOF JOIN`, so the general path becomes a range join plus a window function, which is more work
for the engine. On this data that cost did not materialize: for `session_stats` the range join took
0.111 s against 0.323 s for the ASOF fast path, because the lag check and the de-duplication window
cost more than the small, TTL-bounded candidate sets.

The second trade-off is the **content-based tie-break**. Ordering ties by an ingestion sequence
number would be cheaper, but sequence numbers do not survive a partition rewrite or a replay, so
the same query could return different values after an erasure. Ordering by the feature values
themselves means the answer is a function of the data alone. The cost: every store has to keep
whole rows, because the ordering needs all of a view's features even when you ask for one.

### Why the online path looks the way it does

Materialization reads rows whose `max(event_time, ingested_at)` lies in
`(previous watermark, new watermark]`, keeps the greatest row per entity, and writes it through a
registered Lua script that only replaces a stored row with a greater key. Reruns, overlaps and
out-of-order batches therefore converge to the same state - which is what the idempotency benchmark
measures.

Reads use `MULTI/EXEC` and log the committed and in-flight watermarks with the served values, so
the skew checker can recompute each logged read offline as of exactly the right cutoff and demand
an exact match, rather than allowing a tolerance. A run records its target under the offline store
lock before it scans, and appends check their `ingested_at` against committed and in-flight targets
under the same lock, so a row cannot land between a scan and its watermark commit and be skipped
for good.

### Erasure

Sources are partitioned by a hash of the entity id, so erasure rewrites one bucket per source
rather than every date partition the person ever touched. Keys of every view version ever applied
are deleted, as are request-log rows, and the request-log SQLite file is scrubbed
(`secure_delete`, WAL checkpoint, `VACUUM`) so the id is gone from the bytes on disk, not only from
the table. Tombstones are read under the store lock and keep the entity out of later and racing
appends, materializations and request logging. The receipt is signed with Ed25519, and purge
verification scans the raw SQLite bytes as well as every store.

### Design decisions

- [ADR 0002](docs/adr/0002-bitemporal-visibility-and-value-tie-break.md) - two timestamps and the
  content-based tie-break, and the alternatives rejected.
- [ADR 0003](docs/adr/0003-duckdb-range-join-with-verified-asof-fast-path.md) - range join as the
  general path, ASOF only after a data check, and why the planner had to be pinned.
- [ADR 0004](docs/adr/0004-lua-conditional-writes-and-two-phase-watermarks.md) - server-side
  conditional writes, the script allowlist, and the two-phase watermark that closed a lost-append
  window.
- [ADR 0005](docs/adr/0005-entity-partitioned-parquet-and-signed-erasure.md) - entity-hash
  partitioning, no derived copies, and signed receipts.

### Layout

```text
src/point_in_time_feature_store/
  dsl.py definitions.py registry.py hashchain.py sqlbuild.py        definitions and registry
  offline/ store.py features.py retrieval.py oracle.py baselines.py synthetic.py
  online/  store.py materialize.py acl.py
  serving/ auth.py service.py api.py request_log.py
  parity.py erasure.py training.py monitoring.py catalog.py demo.py cli.py
  core/ (settings, redacting logs, tracing, metrics, hardened app)   ml/ (contracts, drift, manifest)
tests/unit  tests/integration  bench/bench_baseline.py  docs/  FEATURE_CATALOG.md
```

## Security

In plain terms: a caller has to prove who they are, and is only shown the feature groups they were
granted. Personal fields need a second, separate grant, and are silently left out - but explicitly
listed as withheld - when the caller does not have it. Nobody can send code or SQL into the system;
transformations are a fixed vocabulary of operations, not a language. Every read attempt and every
definition change is written to a log that cannot be edited without leaving evidence. And an
entity id never appears in a log file in readable form.

Threats, controls and tests are in [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md); every claim below
links to evidence in [CLAIMS.md](CLAIMS.md).

- **Authentication and scopes.** Callers present EdDSA JWTs verified with the issuer's public key
  (algorithm pinned; `exp`, `iat`, `iss`, `aud`, `sub` required; `exp - iat` at most one hour by
  default, because tokens cannot be revoked). Reading a view needs `features:read:<view>`;
  PII-tagged features need `pii:read` and are otherwise omitted and listed; erasure needs
  `entities:erase`. `/metrics` uses a separate API key.
- **No user SQL.** Transformations are a bounded, closed expression tree; SQL identifiers pass a
  strict validator and all values are bound parameters; DuckDB runs with external access disabled
  and a locked configuration; semgrep bans formatted SQL. Features derived from PII columns must
  carry the PII tag.
- **Redis least privilege.** The application ACL user has only the commands the store uses on
  `fs:v1:*` keys (no `FLUSHALL`, `CONFIG`, `EVAL`, `SCRIPT LOAD`, `KEYS`). Scripts are loaded by an
  admin connection at deploy time and called by SHA from a code allowlist; writes fail closed when
  scripts are missing. Passwords come from the environment.
- **Input limits.** Entity keys are validated against fixed presets, batch and feature counts are
  capped, bodies are size-limited, and rate limits apply per verified JWT subject.
- **Privacy.** Entity ids appear in logs only as HMAC-SHA256; PII values are stored in the request
  log only as keyed HMACs; erasure receipts carry the HMAC, never the id. Erasure scrubs the
  request-log SQLite file, and purge verification checks Parquet, Redis, the request log, the raw
  bytes of the SQLite files and their WAL, and captured logs.
- **Integrity and audit.** Registry changes and every authenticated online read attempt, granted or
  refused, are appended to hash-chained tables, as are refused erasures. The chains detect
  in-place edits; truncation and rebuilt chains are detected against Ed25519-signed anchors kept
  outside the database (`pitfs anchor-chains`, `pitfs verify-chains --anchor`). Erasure receipts
  are Ed25519-signed; contract failures block materialization.
- **Supply chain and runtime.** uv.lock with hashes, pip-audit, bandit, semgrep, gitleaks,
  hadolint, trivy and a CycloneDX SBOM in CI; digest-pinned base images; non-root, read-only
  containers with all capabilities dropped (the compose Redis also starts as its non-root user and
  keeps no persistence files). See [SECURITY.md](SECURITY.md) for reporting.

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12. Everything here runs offline on a
CPU-only laptop. No paid API key, no network, no GPU.

```bash
uv sync --frozen
uv run python tasks.py demo    # end-to-end offline demo, JSON summary on stdout
```

On Windows with a small repository drive, keep the environment elsewhere first:
`$env:UV_PROJECT_ENVIRONMENT = "C:/Users/<you>/.venvs/point-in-time-feature-store"`.

**What you will see.** The demo prints one JSON summary, computed during the run, covering seven
steps in order:

1. `registry` - definitions applied, the version hash of each feature view, and a destructive
   change (shortening a TTL) being refused: `destructive_apply_blocked_without_approval: true`.
2. `ingest_and_materialize` - synthetic feeds ingested in two knowledge-time phases; a batch with
   negative amounts rejected by the contract; late-stamped rows rejected as a knowledge-time
   violation; then `rerun_is_noop: true` and `incremental_equals_one_shot: true`.
3. `leakage` - the naive and point-in-time churn models trained side by side, with MLflow lineage.
4. `serving` - the same batch read with three different tokens: PII omitted without the scope,
   served with it, `403` for a view outside the caller's scopes, `401` unauthenticated.
5. `parity` - the skew checker recomputing every logged read; `mismatches: 0`.
6. `erasure` - one customer deleted; partitions rewritten, online keys and request-log rows
   removed, `receipt_signature_valid: true`, `tampered_receipt_rejected: true`, `purge_clean: true`
   and `raw_entity_id_in_logs: 0`.
7. `drift` and `audit` - PSI between the training and holdout periods, and both hash chains
   verifying.

Other entry points:

```bash
uv run python tasks.py check   # ruff, mypy --strict, pytest + coverage, bandit, pip-audit, baseline
uv run python tasks.py bench   # the benchmarks above, with regression thresholds
```

Definitions workflow:

```bash
uv run pitfs export-definitions --out var/definitions.json
uv run pitfs plan  --definitions var/definitions.json --registry var/registry.sqlite
uv run pitfs apply --definitions var/definitions.json --registry var/registry.sqlite --actor you
uv run pitfs catalog --out FEATURE_CATALOG.md
uv run pitfs generate-keys --out-dir var/keys
uv run pitfs anchor-chains --registry var/registry.sqlite \
    --signing-key var/keys/receipt_private.pem --out anchors/registry-anchor.json
uv run pitfs verify-chains --registry var/registry.sqlite \
    --anchor anchors/registry-anchor.json --public-key var/keys/receipt_public.pem
```

Keep anchors away from the registry host and take them regularly. Without `--anchor`,
`verify-chains` detects edited or reordered entries but not a truncated tail or a chain rebuilt by
someone who can write the SQLite file, and says so.

Run the full stack with Docker Compose (see `docker-compose.yml`):

```bash
uv run pitfs generate-keys --out-dir var/keys
sudo chown 10001:10001 var/keys/receipt_private.pem   # the app container user; mode stays 0600
export APP_REDIS_PASSWORD=... REDIS_ADMIN_PASSWORD=... APP_LOG_HMAC_KEY=...
uv run pitfs redis-acl --out var/redis/users.acl      # passwords appear only as SHA-256
docker compose up --build    # Redis (ACL), script loader, bootstrap, API
TOKEN="$(uv run pitfs mint-token --private-key var/keys/jwt_private.pem --subject svc-scoring \
    --scope 'features:read:customer_profile')"
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"entity_ids": ["c000001"], "features": ["customer_profile:plan_tier"]}' \
     http://127.0.0.1:8000/v1/features/online
```

The one-shot `bootstrap` service (`pitfs bootstrap`) applies the bundled definitions, generates
synthetic sources into the empty data volume and materializes every view up to now, so `/readyz`
becomes healthy and the API has data to serve; re-running it is a no-op apart from materializing
newly visible rows. The synthetic timeline ends in February 2026, so views with a TTL
(`transaction_stats`, `session_stats`) hold no unexpired values at wall-clock time; the example
reads `customer_profile`, which has no TTL.

Containers were not run while building this repository. CI builds, smoke-tests, scans and SBOMs
the image (job `docker`) and brings this compose stack up, waits for `/readyz` and checks an
authenticated online read (job `compose`).

Stack: Python 3.12, DuckDB, Polars, pandera, Redis (fakeredis in tests), FastAPI, LightGBM, MLflow.

## Non-goals

What this repository deliberately does not do. These are limits, not a to-do list.

- **Scale.** Single-node DuckDB and one Redis; no distributed query engine, sharding or
  multi-region replication. Correctness first.
- **Windowed aggregations, request-time transformations and streaming.** Transformations are
  row-level by design (they cannot look at other rows). Point-in-time-correct windowed
  aggregations (a rolling count as of label time and as of the materialization watermark), the
  hardest part of a production feature store, are not implemented: rolling counts arrive
  precomputed from upstream, as in the synthetic data. There is no on-demand transformation
  executed separately at request time and no streaming ingestion.
- **Skew between two transformation implementations.** The skew checker recomputes logged reads
  with the same transformation code that materialization used. It catches watermark, write-path,
  serving and direct-edit skew, but with a single implementation it cannot exercise the classic
  case of offline and online code diverging.
- **IEJoin, and a faster ASOF path.** The spec asked for a DuckDB IEJoin range join. With an
  equality on `entity_id` plus range predicates, DuckDB plans a `HASH_JOIN` with the inequalities
  as join conditions; the benchmark reports the operator it actually used. The ASOF fast path is
  kept because it is oracle-checked, but on this data it is slower than the range join (see
  Results); a spine size or entity-cardinality regime where it pays off was not benchmarked.
- **MLflow tracking server.** Lineage uses MLflow's file store, which MLflow 3 keeps in maintenance
  mode behind `MLFLOW_ALLOW_FILE_STORE=true` (set by the training code); lineage is duplicated in
  the registry's `training_runs` table.
- **Machine unlearning and external copies.** Erasure removes data from the store's own sources,
  online keys and request log. Models trained before erasure are not retrained, and backups or
  datasets exported by users are out of scope. Erasure rewrites history on purpose, so a training
  set that contained the entity cannot be rebuilt bit-for-bit.
- **Token lifecycle.** No token revocation or key rotation workflow; the service trusts one issuer
  public key. Token lifetime is capped instead (one hour by default).
- **Redis key expiry.** TTLs are enforced when reading (against request time), not by Redis
  expiry, so expired rows stay in memory until overwritten or erased.
- **ACL enforcement in local tests.** fakeredis does not enforce ACLs; the ACL is validated
  statically locally and enforced against a real Redis only in CI. The online latency benchmark
  uses fakeredis locally and a Redis service container in CI.
- **Scheduling.** Materialization, skew checks and drift checks are commands and library calls;
  there is no scheduler or alerting.
- **Metric naming.** The spec's `materialization_lag` is exported as
  `materialization_lag_seconds` (Prometheus unit suffix).
- **Local containers.** Neither the image nor `docker compose` was run while building this
  repository; both are exercised only in CI (jobs `docker` and `compose`).
- **Concurrent materializers of one view.** Appends and materialization runs are serialized through
  the offline store lock and in-flight targets, but at most one materializer per view is assumed
  and not enforced.
- **Erasure below the storage engines.** Erasure scrubs the request-log SQLite file and rewrites
  Parquet parts. It does not overwrite filesystem blocks (journaling, SSD remapping, snapshots),
  Redis replicas or persistence files, or backups.
- **Audit of unauthenticated requests.** Requests without a valid token have no verified subject
  and are counted in `auth_failures_total` only, not in the hash-chained access audit.
- **A leak the event-time-only join visibly exploits.** The synthetic data has hours of ingestion
  lag against a 30-day label horizon, so the event-time-only arm shows no measurable AUC leak; its
  wrongness is shown by the property and mutation tests. A regime with label-correlated,
  horizon-scale ingestion lag was not added.
- **Online latency over a network.** Locally the latency benchmark runs in-process on fakeredis;
  no TCP round-trip number against a real Redis was measured here (CI runs it against a Redis
  service container, but that output is not copied into this README).

## Documentation

- [docs/PIT_SEMANTICS.md](docs/PIT_SEMANTICS.md): the retrieval specification (TTL, ties, late data, staleness)
- [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md), [SECURITY.md](SECURITY.md), [CLAIMS.md](CLAIMS.md)
- [MODEL_CARD.md](MODEL_CARD.md): the churn model used in the leakage experiment
- [FEATURE_CATALOG.md](FEATURE_CATALOG.md): generated from the definitions
- [docs/adr/](docs/adr/): design decisions (bitemporal tie-break, range join vs ASOF, Lua writes and
  watermarks, partitioning and erasure)
