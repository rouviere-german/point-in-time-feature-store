"""Shared engineering baseline for the portfolio repos.

Sub-packages:
- ``core``: settings, logging with PII redaction, tracing, metrics, hardened FastAPI app factory.
- ``agentic``: LLM provider protocol, untrusted-data envelopes, tool registry, approvals,
  budgets and a hash-chained audit log.
- ``ml``: data contracts, run manifests, content-addressed registry, drift monitors and
  pickle-free serialization.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
