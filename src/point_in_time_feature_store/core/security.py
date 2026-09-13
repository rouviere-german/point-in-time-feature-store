"""API-key authentication.

Keys are high-entropy random tokens, so a plain SHA-256 digest is an adequate at-rest form (a
slow password hash buys nothing against a 256-bit secret). The presented key is hashed first and
the digest is compared against *every* configured digest with ``hmac.compare_digest`` and no
early exit, so response timing reveals neither key prefixes nor which configured key matched.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from point_in_time_feature_store.core.errors import AppError

__all__ = [
    "API_KEY_HEADER",
    "ApiKeyVerifier",
    "AuthenticatedPrincipal",
    "Principal",
    "extract_api_key",
    "generate_api_key",
    "hash_api_key",
    "require_api_key",
]

API_KEY_HEADER = "X-API-Key"
_MAX_KEY_LENGTH = 512


def hash_api_key(raw_key: str) -> str:
    """Lowercase hex SHA-256 of an API key, as stored in ``APP_API_KEY_SHA256``."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Return a new (raw_key, sha256_hex) pair. Show the raw key once; store only the digest."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_api_key(raw)


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller, identified by a non-secret prefix of its key digest."""

    key_id: str


class ApiKeyVerifier:
    def __init__(self, sha256_hex_digests: Iterable[str]) -> None:
        self._digests = tuple(bytes.fromhex(item) for item in sha256_hex_digests)

    @property
    def configured(self) -> bool:
        return bool(self._digests)

    def verify(self, presented: str | None) -> Principal | None:
        """Return the principal for a valid key, else None. Fails closed with no keys set."""
        if not presented or len(presented) > _MAX_KEY_LENGTH:
            return None
        digest = hashlib.sha256(presented.encode("utf-8")).digest()
        matched: bytes | None = None
        for candidate in self._digests:
            if hmac.compare_digest(digest, candidate):
                matched = candidate
        return Principal(key_id=matched.hex()[:12]) if matched is not None else None


def extract_api_key(headers: Mapping[str, str]) -> str | None:
    """Read the key from ``X-API-Key`` or ``Authorization: Bearer <key>``.

    Starlette's ``Headers`` is case-insensitive; plain mappings must use lowercase names.
    """
    value = headers.get("x-api-key")
    if value:
        return value
    auth = headers.get("authorization")
    if auth and auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    return None


def require_api_key(request: Request) -> Principal:
    """FastAPI dependency: authenticate the caller or raise a uniform 401."""
    verifier: ApiKeyVerifier = request.app.state.api_key_verifier
    principal = verifier.verify(extract_api_key(request.headers))
    if principal is None:
        request.app.state.metrics.auth_failures_total.inc()
        raise AppError(
            401,
            "unauthorized",
            "A valid API key is required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return principal


AuthenticatedPrincipal = Annotated[Principal, Depends(require_api_key)]
