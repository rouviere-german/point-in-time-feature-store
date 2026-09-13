# 1. Record architecture decisions

- Status: accepted
- Date: 2026-09-13

## Context

Each portfolio repo makes trade-offs that a reviewer will question: why SQLite instead of Postgres,
why a scripted fake model instead of live calls, why one approval per call. Without a record, the
reasoning is lost or has to be reconstructed from commit messages, and README claims drift away
from the decisions that justify them.

## Decision

We record architecturally significant decisions as Architecture Decision Records in `docs/adr/`,
following the format described by Michael Nygard ("Documenting Architecture Decisions", 2011):

- one Markdown file per decision, named `NNNN-short-title.md` with a zero-padded sequence number;
- sections: Status, Date, Context, Decision, Consequences;
- status is one of proposed, accepted, deprecated, or superseded by ADR-NNNN;
- accepted ADRs are not rewritten. A changed decision gets a new ADR that supersedes the old one.

A decision is significant when it affects a security control, a correctness guarantee, a claim in
README.md or CLAIMS.md, a non-goal, or the cost of changing course later.

## Consequences

- Reviewers can trace README claims and non-goals back to explicit trade-offs.
- Writing an ADR costs a few minutes per decision; small, reversible choices do not need one.
- `scripts/check_baseline.py` requires at least one ADR, so the directory cannot silently vanish.
