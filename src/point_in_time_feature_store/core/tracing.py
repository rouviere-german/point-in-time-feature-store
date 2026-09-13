"""OpenTelemetry tracing setup that works fully offline.

``none`` records spans in-process without exporting them; ``console`` prints them to stdout.
Repos that ship to a collector add ``opentelemetry-exporter-otlp`` and pass the exporter in via
``extra_exporter``; the template keeps that dependency out on purpose.
"""

from __future__ import annotations

from typing import Literal

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

__all__ = ["build_tracer_provider", "install_global_tracer_provider"]

_installed: list[TracerProvider] = []


def build_tracer_provider(
    service_name: str,
    exporter: Literal["none", "console"] = "none",
    *,
    extra_exporter: SpanExporter | None = None,
) -> TracerProvider:
    """Create a tracer provider without touching global state (safe to call in tests)."""
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    if extra_exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(extra_exporter))
    return provider


def install_global_tracer_provider(provider: TracerProvider) -> bool:
    """Install ``provider`` globally once per process. Returns False if one was already set."""
    if _installed:
        return False
    trace.set_tracer_provider(provider)
    _installed.append(provider)
    return True
