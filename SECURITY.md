# Security policy

## Supported versions

Only the latest commit on `main` is supported. This is a portfolio project, not a product with
release branches.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting ("Report a vulnerability" under the repository's Security tab). Include:

- the affected commit and component;
- steps to reproduce, ideally as a failing test;
- the impact you expect.

You can expect an acknowledgement within 7 days and a fix or a written assessment within 30 days.
Credit is given in the fix commit unless you ask otherwise.

## Scope

In scope: the code in this repository, its container image, the compose file and the CI
configuration. Out of scope: denial of service by traffic volume, findings that need a
compromised host or a stolen signing key, and issues in third-party dependencies that are already
publicly tracked (please report those upstream).

Especially interesting: any history for which point-in-time retrieval returns a value that was not
known at label time, any online value that differs from offline recomputation beyond the
documented watermark staleness, any personal data left behind after erasure, and any way to read
PII-tagged features without `pii:read`.

## How this repository is hardened

- Threats, controls and the tests that prove them: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
- Every security claim is linked to evidence in [CLAIMS.md](CLAIMS.md).
- Callers authenticate with EdDSA JWTs (public key only in the service); per-view read scopes,
  `pii:read` for PII-tagged features, `entities:erase` for erasure.
- No user-supplied SQL or code: transformations are a closed expression tree; DuckDB runs with
  external access disabled; SQL identifiers are validated and values are bound.
- Redis runs with an ACL user that cannot flush, reconfigure, `EVAL` or `SCRIPT LOAD`; passwords
  come from the environment and appear in the ACL file only as SHA-256 digests.
- Entity ids are HMAC-hashed in logs and receipts; PII values are HMAC-hashed in the request log.
- Erasure receipts are signed with Ed25519; registry changes and feature reads are recorded in
  hash-chained tables.
- CI runs ruff, mypy --strict, pytest with a coverage floor and a Redis service, bandit, semgrep
  (registry packs plus `.semgrep/rules.yml`), pip-audit, gitleaks, hadolint, a trivy image scan and
  a syft CycloneDX SBOM.
- Secrets come only from the environment; only `.env.example` is committed.
- Model artifacts use LightGBM's text format; pickle-family serialization is banned.
