"""FastAPI application factory with the baseline hardening applied.

The factory only wires the cross-cutting pieces (probes, metrics, error handling, middleware).
Feature routes are added by :mod:`point_in_time_feature_store.serving.api`.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CollectorRegistry
from starlette.datastructures import Headers

from point_in_time_feature_store import __version__
from point_in_time_feature_store.core.config import Settings, get_settings
from point_in_time_feature_store.core.errors import install_error_handlers
from point_in_time_feature_store.core.logging import get_logger
from point_in_time_feature_store.core.metrics import create_http_metrics, render_metrics
from point_in_time_feature_store.core.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
)
from point_in_time_feature_store.core.ratelimit import RateLimiter
from point_in_time_feature_store.core.security import ApiKeyVerifier, require_api_key
from point_in_time_feature_store.core.tracing import build_tracer_provider

__all__ = ["ReadinessCheck", "create_app"]

ReadinessCheck = Callable[[], bool | Awaitable[bool]]
_log = get_logger("point_in_time_feature_store.app")


async def _run_check(name: str, check: ReadinessCheck) -> bool:
    try:
        result = check()
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
    except Exception as exc:  # noqa: BLE001 - a failing probe must report not-ready, not crash
        _log.warning("readiness_check_failed", check=name, error_type=type(exc).__name__)
        return False


def create_app(
    settings: Settings | None = None,
    *,
    readiness_checks: Mapping[str, ReadinessCheck] | None = None,
    registry: CollectorRegistry | None = None,
    tracer_provider: TracerProvider | None = None,
    rate_limit_identity: Callable[[Headers], str | None] | None = None,
) -> FastAPI:
    """Build the app. All state hangs off ``app.state``; nothing global is mutated.

    ``rate_limit_identity`` maps request headers to a verified caller id (for example a JWT
    subject). It must return None for unverified credentials, so they share the address bucket.
    """
    cfg = settings if settings is not None else get_settings()
    checks: dict[str, ReadinessCheck] = dict(readiness_checks or {})
    provider = tracer_provider or build_tracer_provider(cfg.service_name, cfg.tracing_exporter)
    metrics = create_http_metrics(registry)
    verifier = ApiKeyVerifier(cfg.api_key_sha256)
    limiter = RateLimiter(
        cfg.rate_limit_per_second, cfg.rate_limit_burst, max_keys=cfg.rate_limit_max_clients
    )

    app = FastAPI(
        title=cfg.service_name,
        version=__version__,
        docs_url="/docs" if cfg.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if cfg.enable_docs else None,
    )
    app.state.settings = cfg
    app.state.metrics = metrics
    app.state.api_key_verifier = verifier
    app.state.rate_limiter = limiter
    app.state.tracer_provider = provider
    install_error_handlers(app)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        """Liveness: the process is serving requests. Checks no dependencies."""
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        """Readiness: every registered dependency check passes. Reports names, not errors."""
        results = {name: await _run_check(name, check) for name, check in checks.items()}
        ready = all(results.values())
        body: dict[str, Any] = {
            "status": "ready" if ready else "not_ready",
            "checks": {name: "ok" if ok else "fail" for name, ok in results.items()},
        }
        return JSONResponse(body, status_code=200 if ready else 503)

    metrics_deps = [Depends(require_api_key)] if cfg.metrics_require_auth else []

    @app.get("/metrics", include_in_schema=False, dependencies=metrics_deps)
    async def metrics_endpoint(request: Request) -> Response:
        payload, content_type = render_metrics(request.app.state.metrics.registry)
        return Response(content=payload, media_type=content_type)

    # add_middleware prepends, so the last one added is the outermost.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=cfg.max_body_bytes, metrics=metrics)
    app.add_middleware(
        RateLimitMiddleware,
        limiter=limiter,
        verifier=verifier,
        metrics=metrics,
        identity=rate_limit_identity,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_allow_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization", "X-Request-ID"],
        max_age=600,
    )
    app.add_middleware(
        RequestContextMiddleware,
        tracer=provider.get_tracer("point_in_time_feature_store.http"),
        metrics=metrics,
        docs_enabled=cfg.enable_docs,
    )
    return app
