# Contributing

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv can install it).

```bash
uv sync --frozen
uv run python tasks.py check      # lint, typecheck, test, audit, baseline
```

If the repository sits on a small or slow drive, keep the virtual environment elsewhere:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "C:/Users/<you>/.venvs/<repo>"
uv sync --frozen
```

## Tasks

`tasks.py` is the cross-platform task runner; the Makefile only delegates to it.

| Target | What it runs |
|---|---|
| `lint` | `ruff check` and `ruff format --check` |
| `format` | `ruff format` and `ruff check --fix` |
| `typecheck` | `mypy --strict` over src, tests, scripts, bench and tasks.py |
| `test` | pytest with branch coverage; fails below 85% |
| `bench` | evaluation benchmarks with regression thresholds from `bench/thresholds.json` |
| `demo` | the offline end-to-end demo (fakeredis, synthetic data) |
| `audit` | bandit and pip-audit |
| `baseline` | `scripts/check_baseline.py` |

`HYPOTHESIS_PROFILE=ci` runs the property tests with 200 examples each (CI does this).

## Rules for changes

- **The oracle is the specification.** Any change to retrieval, materialization or tie handling
  starts in `docs/PIT_SEMANTICS.md` and `offline/oracle.py`, and the property tests in
  `tests/unit/test_pit_properties.py` must pass for both DuckDB paths and the online store.
- **Tests first for behavior.** Mark tests `unit` or `integration` (and `redis` when they need a
  real server). Prefer hypothesis properties (invariants, round trips, oracles) over hand-picked
  examples alone.
- **Offline by default.** Tests, demo and bench must not touch the network; `tests/conftest.py`
  blocks non-loopback sockets. Use fakeredis and the synthetic generator.
- **No user SQL, no new expression nodes without review.** Transformations stay row-level; a new
  node type needs an ADR explaining why it cannot leak future rows.
- **Evidence for claims.** A new README claim needs a row in `CLAIMS.md` pointing to a test,
  bench script or CI job. Performance numbers must be copied from real `bench/` output together
  with the command that produced them.
- **Threat model.** New entry points, stores or copies of data need a STRIDE row in
  `docs/THREAT_MODEL.md` with a control and a test, and erasure must reach any new copy.
- **Feature catalog.** After changing the bundled definitions, regenerate it:
  `uv run pitfs catalog --out FEATURE_CATALOG.md` (a test checks it is current).
- **Decisions.** Record significant trade-offs as ADRs in `docs/adr/`.
- **No pickle.** Use Parquet, JSON, LightGBM text models or npz with `allow_pickle=False`.
- **No secrets in git.** Configuration goes in `.env` (ignored); keys and ACL files stay under
  `var/` (ignored); document new settings in `.env.example` (a test enforces that it lists every
  setting).
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`).
