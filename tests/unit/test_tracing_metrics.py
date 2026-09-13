from __future__ import annotations

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from point_in_time_feature_store.core import tracing
from point_in_time_feature_store.core.metrics import create_http_metrics, render_metrics
from point_in_time_feature_store.core.tracing import (
    build_tracer_provider,
    install_global_tracer_provider,
)

pytestmark = pytest.mark.unit


def test_provider_records_spans_without_global_state() -> None:
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("svc", extra_exporter=exporter)
    with provider.get_tracer("t").start_as_current_span("work"):
        pass
    assert [span.name for span in exporter.get_finished_spans()] == ["work"]
    resource = exporter.get_finished_spans()[0].resource
    assert resource.attributes["service.name"] == "svc"


def test_console_exporter_can_be_selected() -> None:
    provider = build_tracer_provider("svc", "console")
    with provider.get_tracer("t").start_as_current_span("work"):
        pass
    provider.shutdown()


def test_global_provider_is_installed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    installed: list[TracerProvider] = []
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", installed.append)
    first, second = build_tracer_provider("a"), build_tracer_provider("b")
    assert install_global_tracer_provider(first) is True
    assert install_global_tracer_provider(second) is False
    assert installed == [first]


def test_metrics_render_in_prometheus_format() -> None:
    metrics = create_http_metrics()
    metrics.requests_total.labels("GET", "/x", "200").inc()
    metrics.request_duration_seconds.labels("GET", "/x").observe(0.01)
    payload, content_type = render_metrics(metrics.registry)
    assert b'app_http_requests_total{method="GET",route="/x",status="200"} 1.0' in payload
    assert content_type.startswith("text/plain")
