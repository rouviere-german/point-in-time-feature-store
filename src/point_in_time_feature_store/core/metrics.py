"""Prometheus metrics on a per-app registry (no global registry, so tests stay isolated)."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

__all__ = ["HttpMetrics", "create_http_metrics", "render_metrics"]


@dataclass(frozen=True, slots=True)
class HttpMetrics:
    registry: CollectorRegistry
    requests_total: Counter
    request_duration_seconds: Histogram
    rate_limited_total: Counter
    auth_failures_total: Counter
    body_too_large_total: Counter


def create_http_metrics(
    registry: CollectorRegistry | None = None, namespace: str = "app"
) -> HttpMetrics:
    """Create the HTTP metric families. Route labels use templates, never raw paths."""
    reg = registry if registry is not None else CollectorRegistry()
    return HttpMetrics(
        registry=reg,
        requests_total=Counter(
            "http_requests_total",
            "HTTP requests by method, route template and status code.",
            ["method", "route", "status"],
            namespace=namespace,
            registry=reg,
        ),
        request_duration_seconds=Histogram(
            "http_request_duration_seconds",
            "HTTP request latency in seconds.",
            ["method", "route"],
            namespace=namespace,
            registry=reg,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        ),
        rate_limited_total=Counter(
            "http_rate_limited_total",
            "Requests rejected by the token-bucket rate limiter.",
            namespace=namespace,
            registry=reg,
        ),
        auth_failures_total=Counter(
            "http_auth_failures_total",
            "Requests rejected for a missing or invalid API key.",
            namespace=namespace,
            registry=reg,
        ),
        body_too_large_total=Counter(
            "http_body_too_large_total",
            "Requests rejected because the body exceeded the size limit.",
            namespace=namespace,
            registry=reg,
        ),
    )


def render_metrics(registry: CollectorRegistry) -> tuple[bytes, str]:
    """Return the exposition payload and its content type."""
    return generate_latest(registry), CONTENT_TYPE_LATEST
