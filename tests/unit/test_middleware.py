"""Direct ASGI tests for header edge cases an HTTP client library will not send."""

from __future__ import annotations

import asyncio

import pytest
from prometheus_client import CollectorRegistry
from starlette.types import Message, Receive, Scope, Send

from point_in_time_feature_store.core.metrics import create_http_metrics
from point_in_time_feature_store.core.middleware import BodySizeLimitMiddleware, RateLimitMiddleware
from point_in_time_feature_store.core.ratelimit import RateLimiter
from point_in_time_feature_store.core.security import ApiKeyVerifier

pytestmark = pytest.mark.unit


async def _unreachable_app(scope: Scope, receive: Receive, send: Send) -> None:
    raise AssertionError("the application must not be reached")


@pytest.mark.parametrize("declared", [b"\xb2", b"\xb9\xb2", b"12x", b"-1"])
def test_malformed_content_length_is_rejected_instead_of_crashing(declared: bytes) -> None:
    # Starlette decodes header bytes as latin-1, where b"\xb2" is "²": str.isdigit() accepts
    # it but int() does not.
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = BodySizeLimitMiddleware(
        _unreachable_app, max_bytes=10, metrics=create_http_metrics(CollectorRegistry())
    )
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/redact",
        "headers": [(b"content-length", declared)],
    }
    asyncio.run(middleware(scope, receive, send))
    assert sent[0]["status"] == 413


async def _ok_app(scope: Scope, receive: Receive, send: Send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b""})


def _statuses_for_clients(hosts: list[str]) -> list[int]:
    """Send one unauthenticated request per client address through a burst-1 rate limiter."""
    middleware = RateLimitMiddleware(
        _ok_app,
        limiter=RateLimiter(rate=0.001, burst=1),
        verifier=ApiKeyVerifier([]),
        metrics=create_http_metrics(CollectorRegistry()),
    )
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    statuses: list[int] = []
    for host in hosts:
        sent.clear()
        scope: Scope = {
            "type": "http",
            "method": "GET",
            "path": "/v1/items",
            "headers": [],
            "client": (host, 50000),
        }
        asyncio.run(middleware(scope, receive, send))
        statuses.append(int(sent[0]["status"]))
    return statuses


def test_ipv6_clients_are_limited_per_64_prefix() -> None:
    # One IPv6 host usually controls a whole /64; per-address buckets would give it 2**64 bursts.
    hosts = ["2001:db8:1:2::1", "2001:db8:1:2:ffff:ffff:ffff:9", "2001:db8:1:3::1"]
    assert _statuses_for_clients(hosts) == [200, 429, 200]


def test_ipv4_mapped_ipv6_clients_share_the_ipv4_bucket() -> None:
    assert _statuses_for_clients(["192.0.2.7", "::ffff:192.0.2.7", "192.0.2.8"]) == [200, 429, 200]
