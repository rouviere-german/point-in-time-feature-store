"""Pure ASGI middlewares (no BaseHTTPMiddleware, so streaming bodies stay streaming).

Order, outermost first: RequestContext (request id, span, metrics, access log, security headers,
catch-all 500) -> CORS -> RateLimit -> BodySizeLimit -> application.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import structlog
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from point_in_time_feature_store.core.errors import BodyTooLargeError, error_payload
from point_in_time_feature_store.core.metrics import HttpMetrics
from point_in_time_feature_store.core.ratelimit import RateLimiter
from point_in_time_feature_store.core.security import ApiKeyVerifier, extract_api_key

__all__ = [
    "SECURITY_HEADERS",
    "BodySizeLimitMiddleware",
    "RateLimitMiddleware",
    "RequestContextMiddleware",
]

SECURITY_HEADERS: Mapping[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "Cache-Control": "no-store",
}
_DOCS_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/openapi.json"})
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_log = structlog.get_logger("point_in_time_feature_store.http")


async def _send_json(
    send: Send, status: int, body: dict[str, Any], headers: Mapping[str, str]
) -> None:
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    raw_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode("ascii")),
        *((k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()),
    ]
    await send({"type": "http.response.start", "status": status, "headers": raw_headers})
    await send({"type": "http.response.body", "body": payload})


def client_bucket(host: str) -> str:
    """Rate-limit identity of a client address: IPv4 per address, IPv6 per /64 prefix.

    A single IPv6 host is normally assigned a whole /64, so per-address buckets would let it rotate
    through 2**64 fresh bursts. IPv4-mapped IPv6 addresses count as their IPv4 address.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return str(ipaddress.IPv6Network((address, 64), strict=False))
    return str(address)


def _request_id(scope: Scope) -> str | None:
    state = scope.get("state")
    if isinstance(state, dict):
        value = state.get("request_id")
        return value if isinstance(value, str) else None
    return None


class RequestContextMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        tracer: Tracer,
        metrics: HttpMetrics,
        docs_enabled: bool = False,
    ) -> None:
        self.app = app
        self.tracer = tracer
        self.metrics = metrics
        self.docs_enabled = docs_enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        method = str(scope.get("method", "GET"))
        path = str(scope.get("path", ""))
        skip_csp = self.docs_enabled and path in _DOCS_PATHS
        status_holder = {"status": 500, "started": 0}
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message["status"])
                status_holder["started"] = 1
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    if skip_csp and name == "Content-Security-Policy":
                        continue
                    headers[name] = value
                headers["X-Request-ID"] = request_id
            await send(message)

        with (
            structlog.contextvars.bound_contextvars(request_id=request_id),
            self.tracer.start_as_current_span("http.request", kind=SpanKind.SERVER) as span,
        ):
            try:
                await self.app(scope, receive, send_wrapper)
            except Exception as exc:  # noqa: BLE001 - last-resort boundary; never leak internals
                _log.exception("unhandled_exception", error_type=type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR))
                if not status_holder["started"]:
                    await _send_json(
                        send_wrapper,
                        500,
                        error_payload("internal_error", "Internal server error", request_id),
                        {},
                    )
            finally:
                route_obj = scope.get("route")
                route = str(getattr(route_obj, "path", "unmatched"))
                status = status_holder["status"]
                duration = time.perf_counter() - start
                span.set_attribute("http.request.method", method)
                span.set_attribute("http.route", route)
                span.set_attribute("http.response.status_code", status)
                span.set_attribute("request.id", request_id)
                self.metrics.requests_total.labels(method, route, str(status)).inc()
                self.metrics.request_duration_seconds.labels(method, route).observe(duration)
                _log.info(
                    "http_request",
                    method=method,
                    route=route,
                    status=status,
                    duration_ms=round(duration * 1000, 3),
                )


class RateLimitMiddleware:
    """Token-bucket limit per authenticated key, or per client address when unauthenticated.

    The key is verified here (a hash plus constant-time compares) so random invalid keys cannot be
    used to mint fresh buckets; invalid keys share the caller's address bucket. IPv6 addresses are
    grouped by /64 (see :func:`client_bucket`).
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: RateLimiter,
        verifier: ApiKeyVerifier,
        metrics: HttpMetrics,
        exempt_paths: frozenset[str] = frozenset({"/healthz", "/readyz"}),
        identity: Callable[[Headers], str | None] | None = None,
    ) -> None:
        self.app = app
        self.limiter = limiter
        self.verifier = verifier
        self.metrics = metrics
        self.exempt_paths = exempt_paths
        self.identity = identity

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in self.exempt_paths:
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        verified = self.identity(headers) if self.identity is not None else None
        principal = self.verifier.verify(extract_api_key(headers))
        if verified is not None:
            key = "sub:" + hashlib.sha256(verified.encode("utf-8")).hexdigest()[:32]
        elif principal is not None:
            key = f"key:{principal.key_id}"
        else:
            client = scope.get("client")
            host = client[0] if client else "unknown"
            bucket = client_bucket(str(host))
            key = "ip:" + hashlib.sha256(bucket.encode("utf-8")).hexdigest()[:32]
        decision = self.limiter.check(key)
        if not decision.allowed:
            self.metrics.rate_limited_total.inc()
            await _send_json(
                send,
                429,
                error_payload("rate_limited", "Too many requests", _request_id(scope)),
                {"Retry-After": str(decision.retry_after_seconds)},
            )
            return
        await self.app(scope, receive, send)


class BodySizeLimitMiddleware:
    """Reject bodies over ``max_bytes``: early via Content-Length, and while streaming."""

    def __init__(self, app: ASGIApp, *, max_bytes: int, metrics: HttpMetrics) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        # isascii(): str.isdigit() also accepts characters such as "²" that int() rejects.
        malformed = declared is not None and not (declared.isascii() and declared.isdigit())
        if declared is not None and (malformed or int(declared) > self.max_bytes):
            await self._reject(scope, send)
            return

        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise BodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BodyTooLargeError:
            if started:
                raise
            await self._reject(scope, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        self.metrics.body_too_large_total.inc()
        await _send_json(
            send,
            413,
            error_payload("payload_too_large", "Request body too large", _request_id(scope)),
            {},
        )
