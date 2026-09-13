"""Local, offline stack for the demo, the benchmarks and the tests.

Builds every service on a temporary data directory with fakeredis (a development dependency),
freshly generated Ed25519 keys for tokens and receipts, a random HMAC key and a settable clock.
Nothing here is used by the production entry points.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import redis

from point_in_time_feature_store.core.config import Settings
from point_in_time_feature_store.erasure import ReceiptSigner
from point_in_time_feature_store.hashing import EntityHasher
from point_in_time_feature_store.services import FeatureStoreServices, build_services
from point_in_time_feature_store.serving.auth import (
    TokenVerifier,
    generate_keypair_pem,
    load_private_key,
    load_public_key,
    mint_token,
)

__all__ = ["AUDIENCE", "ISSUER", "LocalStack", "MutableClock", "build_local_stack", "fake_redis"]

ISSUER = "pitfs-local-issuer"
AUDIENCE = "point-in-time-feature-store"


class MutableClock:
    """A settable clock in microseconds, so a timeline can be replayed deterministically."""

    def __init__(self, now_us: int) -> None:
        self.now_us = now_us

    def __call__(self) -> int:
        return self.now_us


def fake_redis() -> redis.Redis:
    try:
        import fakeredis  # noqa: PLC0415 - development dependency
    except ImportError as exc:  # pragma: no cover - dev dependencies are installed by uv sync
        raise RuntimeError("fakeredis is not installed; run `uv sync` or use a real Redis") from exc
    client: redis.Redis = fakeredis.FakeRedis()
    return client


@dataclass(slots=True)
class LocalStack:
    services: FeatureStoreServices
    clock: MutableClock
    jwt_private_pem: bytes
    receipt_public_pem: bytes

    def token(
        self, scopes: Iterable[str], *, subject: str = "svc-local", ttl_seconds: int = 3_600
    ) -> str:
        return mint_token(
            load_private_key(self.jwt_private_pem),
            subject=subject,
            scopes=scopes,
            issuer=ISSUER,
            audience=AUDIENCE,
            ttl_seconds=ttl_seconds,
        )

    def close(self) -> None:
        self.services.close()


def build_local_stack(
    root: Path,
    *,
    now_us: int,
    request_log_sample_rate: float = 1.0,
    redis_client: redis.Redis | None = None,
    settings_overrides: Mapping[str, Any] | None = None,
) -> LocalStack:
    jwt_private, jwt_public = generate_keypair_pem()
    receipt_private, receipt_public = generate_keypair_pem()
    values: dict[str, Any] = {
        "environment": "dev",
        "data_dir": root,
        "request_log_sample_rate": request_log_sample_rate,
        "jwt_issuer": ISSUER,
        "jwt_audience": AUDIENCE,
        "rate_limit_per_second": 10_000.0,
        "rate_limit_burst": 100_000,
    }
    values.update(settings_overrides or {})
    settings = Settings(_env_file=None, **values)
    clock = MutableClock(now_us)
    services = build_services(
        settings,
        redis_client=redis_client if redis_client is not None else fake_redis(),
        clock=clock,
        hasher=EntityHasher.ephemeral(),
        signer=ReceiptSigner(load_private_key(receipt_private)),
        token_verifier=TokenVerifier(load_public_key(jwt_public), issuer=ISSUER, audience=AUDIENCE),
    )
    services.online.load_scripts()
    return LocalStack(services, clock, jwt_private, receipt_public)
