from __future__ import annotations

import http.server
import re
import socket
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, ConfigDict, Field

from point_in_time_feature_store.core.app import create_app as create_core_app
from point_in_time_feature_store.core.config import Settings
from point_in_time_feature_store.core.healthcheck import main as healthcheck_main
from point_in_time_feature_store.core.healthcheck import probe
from point_in_time_feature_store.core.middleware import SECURITY_HEADERS
from point_in_time_feature_store.core.redaction import redact_text
from point_in_time_feature_store.core.security import AuthenticatedPrincipal, hash_api_key
from point_in_time_feature_store.core.tracing import build_tracer_provider

pytestmark = pytest.mark.integration

API_KEY = "integration-test-key-" + "x" * 24
AUTH = {"X-API-Key": API_KEY}


class EchoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=10_000)


def create_app(settings: Settings | None = None, **kwargs: Any) -> FastAPI:
    """The core app plus one protected test route (the service's real routes live in serving/)."""
    app = create_core_app(settings, **kwargs)

    @app.post("/v1/echo")
    async def echo(body: EchoRequest, principal: AuthenticatedPrincipal) -> dict[str, str]:
        return {"text": redact_text(body.text)}

    return app


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"api_key_sha256": [hash_api_key(API_KEY)], "environment": "test"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with make_client(create_app(make_settings())) as test_client:
        yield test_client


def test_healthz_is_public_and_carries_security_headers(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    for name, value in SECURITY_HEADERS.items():
        assert response.headers[name] == value
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])


def test_request_ids_are_propagated_only_when_well_formed(client: TestClient) -> None:
    kept = client.get("/healthz", headers={"X-Request-ID": "trace-12345678"})
    assert kept.headers["x-request-id"] == "trace-12345678"
    replaced = client.get("/healthz", headers={"X-Request-ID": "short"})
    assert replaced.headers["x-request-id"] != "short"


def test_readyz_reports_check_names_without_error_details() -> None:
    async def cache_ok() -> bool:
        return True

    def database_down() -> bool:
        raise RuntimeError("password=hunter2 at db.internal:5432")

    app = create_app(make_settings(), readiness_checks={"cache": cache_ok, "db": database_down})
    with make_client(app) as test_client:
        response = test_client.get("/readyz")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "checks": {"cache": "ok", "db": "fail"}}
    assert "hunter2" not in response.text

    healthy = create_app(make_settings(), readiness_checks={"cache": cache_ok})
    with make_client(healthy) as test_client:
        assert test_client.get("/readyz").status_code == 200


def test_protected_endpoint_requires_a_valid_api_key(client: TestClient) -> None:
    missing = client.post("/v1/echo", json={"text": "hi"})
    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "unauthorized"
    assert missing.headers["www-authenticate"] == "ApiKey"

    wrong = client.post("/v1/echo", json={"text": "hi"}, headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401

    ok = client.post("/v1/echo", json={"text": "mail jane@example.com"}, headers=AUTH)
    assert ok.status_code == 200
    assert ok.json() == {"text": "mail [REDACTED:EMAIL]"}

    bearer = client.post(
        "/v1/echo", json={"text": "hi"}, headers={"Authorization": f"Bearer {API_KEY}"}
    )
    assert bearer.status_code == 200


def test_validation_errors_never_echo_submitted_values(client: TestClient) -> None:
    secret = "hunter2-" + "s" * 10
    extra = client.post("/v1/echo", json={"text": "ok", "password": secret}, headers=AUTH)
    assert extra.status_code == 422
    assert extra.json()["error"]["code"] == "validation_error"
    assert secret not in extra.text

    too_long = client.post("/v1/echo", json={"text": "z" * 10_001}, headers=AUTH)
    assert too_long.status_code == 422
    assert "zzzzzzzz" not in too_long.text


def test_unknown_routes_and_methods_get_generic_errors(client: TestClient) -> None:
    missing = client.get("/nope")
    assert missing.status_code == 404
    assert missing.json()["error"] == {
        "code": "not_found",
        "message": "Not Found",
        "request_id": missing.headers["x-request-id"],
    }
    wrong_method = client.get("/v1/echo")
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["code"] == "method_not_allowed"


def test_unhandled_exceptions_become_generic_500s() -> None:
    app = create_app(make_settings())

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        raise RuntimeError("password=hunter2 in /srv/app/secret_module.py")

    with make_client(app) as test_client:
        response = test_client.get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["request_id"] == response.headers["x-request-id"]
    for leak in ("hunter2", "RuntimeError", "secret_module"):
        assert leak not in response.text
    assert response.headers["x-content-type-options"] == "nosniff"


def test_body_size_limit_applies_to_declared_and_streamed_bodies() -> None:
    app = create_app(make_settings(max_body_bytes=64))
    headers = {**AUTH, "Content-Type": "application/json"}
    with make_client(app) as test_client:
        declared = test_client.post("/v1/echo", content=b"x" * 200, headers=headers)
        assert declared.status_code == 413
        assert declared.json()["error"]["code"] == "payload_too_large"

        def chunks() -> Iterator[bytes]:
            for _ in range(10):
                yield b"x" * 20

        streamed = test_client.post("/v1/echo", content=chunks(), headers=headers)
        assert streamed.status_code == 413


def test_rate_limit_rejects_bursts_and_exempts_probes() -> None:
    app = create_app(make_settings(rate_limit_burst=2, rate_limit_per_second=0.001))
    with make_client(app) as test_client:
        codes = [
            test_client.post("/v1/echo", json={"text": "a"}, headers=AUTH).status_code
            for _ in range(3)
        ]
        assert codes == [200, 200, 429]
        limited = test_client.post("/v1/echo", json={"text": "a"}, headers=AUTH)
        assert limited.headers["retry-after"].isdigit()
        assert limited.json()["error"]["code"] == "rate_limited"
        assert all(test_client.get("/healthz").status_code == 200 for _ in range(10))


def test_invalid_keys_cannot_mint_fresh_rate_limit_buckets() -> None:
    app = create_app(make_settings(rate_limit_burst=2, rate_limit_per_second=0.001))
    with make_client(app) as test_client:
        codes = [
            test_client.post(
                "/v1/echo", json={"text": "a"}, headers={"X-API-Key": f"random-{i}"}
            ).status_code
            for i in range(3)
        ]
    assert codes == [401, 401, 429]


def test_cors_allows_only_configured_origins() -> None:
    app = create_app(make_settings(cors_allow_origins=["https://app.example.com"]))
    preflight = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-API-Key",
    }
    with make_client(app) as test_client:
        allowed = test_client.options(
            "/v1/echo", headers={"Origin": "https://app.example.com", **preflight}
        )
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == "https://app.example.com"
        assert "access-control-allow-credentials" not in allowed.headers

        denied = test_client.options(
            "/v1/echo", headers={"Origin": "https://evil.example", **preflight}
        )
        assert "access-control-allow-origin" not in denied.headers
        simple = test_client.get("/healthz", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in simple.headers


def test_metrics_require_auth_and_label_route_templates() -> None:
    app = create_app(make_settings())

    @app.get("/items/{item_id}")
    async def item(item_id: int) -> dict[str, int]:
        return {"id": item_id}

    with make_client(app) as test_client:
        test_client.get("/items/42")
        assert test_client.get("/metrics").status_code == 401
        body = test_client.get("/metrics", headers=AUTH).text
    assert 'route="/items/{item_id}"' in body
    assert "/items/42" not in body
    assert "app_http_auth_failures_total 1.0" in body


def test_metrics_can_be_made_public() -> None:
    app = create_app(make_settings(metrics_require_auth=False))
    with make_client(app) as test_client:
        assert test_client.get("/metrics").status_code == 200


def test_requests_produce_server_spans() -> None:
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("test", extra_exporter=exporter)
    app = create_app(make_settings(), tracer_provider=provider)
    with make_client(app) as test_client:
        test_client.get("/healthz")
    spans = [s for s in exporter.get_finished_spans() if s.name == "http.request"]
    assert spans
    attributes = dict(spans[0].attributes or {})
    assert attributes["http.route"] == "/healthz"
    assert attributes["http.response.status_code"] == 200


def test_docs_are_off_by_default_and_csp_is_relaxed_only_for_docs(client: TestClient) -> None:
    assert client.get("/docs").status_code == 404
    app = create_app(make_settings(enable_docs=True))
    with make_client(app) as docs_client:
        docs = docs_client.get("/docs")
        assert docs.status_code == 200
        assert "content-security-policy" not in docs.headers
        assert "content-security-policy" in docs_client.get("/healthz").headers


class _Handler(http.server.BaseHTTPRequestHandler):
    status = 200

    def do_GET(self) -> None:
        self.send_response(type(self).status)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


class _FailingHandler(_Handler):
    status = 500


def _serve(handler: type[http.server.BaseHTTPRequestHandler]) -> tuple[http.server.HTTPServer, int]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, int(server.server_address[1])


def test_healthcheck_probe_and_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    server, port = _serve(_Handler)
    try:
        assert probe("127.0.0.1", port)
        monkeypatch.setenv("HEALTHCHECK_HOST", "127.0.0.1")
        monkeypatch.setenv("APP_PORT", str(port))
        assert healthcheck_main() == 0
    finally:
        server.shutdown()


def test_healthcheck_fails_on_error_status_or_unreachable_server() -> None:
    server, port = _serve(_FailingHandler)
    try:
        assert not probe("127.0.0.1", port)
    finally:
        server.shutdown()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        free_port = int(sock.getsockname()[1])
    assert not probe("127.0.0.1", free_port, timeout=0.5)
