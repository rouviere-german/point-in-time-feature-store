"""Caller authentication with EdDSA (Ed25519) JWTs and per-view scopes.

The service holds only the issuer's public key, so a compromised service cannot mint tokens.
Tokens must carry ``iss``, ``aud``, ``sub``, ``iat`` and ``exp``; the algorithm is pinned to
``EdDSA`` (``alg: none`` and HMAC confusion are refused by construction).

Scopes (space-separated ``scope`` claim):

- ``features:read:<view>`` or ``features:read:*``: read a feature view online;
- ``pii:read``: receive PII-tagged features (without it they are omitted, not refused);
- ``entities:erase``: erase an entity from every store;
- ``registry:read``: list definitions.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

__all__ = [
    "SCOPE_ERASE",
    "SCOPE_PII",
    "SCOPE_REGISTRY",
    "Caller",
    "TokenVerifier",
    "bearer_token",
    "generate_keypair_pem",
    "load_private_key",
    "load_public_key",
    "mint_token",
]

SCOPE_PII = "pii:read"
SCOPE_ERASE = "entities:erase"
SCOPE_REGISTRY = "registry:read"
_SCOPE = re.compile(r"^[a-z][a-z0-9_:*-]{0,127}$")
_SUBJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@:-]{0,127}$")
_MAX_TOKEN_CHARS = 4_096
_MAX_SCOPES = 64


@dataclass(frozen=True, slots=True)
class Caller:
    subject: str
    scopes: frozenset[str]

    def can_read_view(self, view: str) -> bool:
        return "features:read:*" in self.scopes or f"features:read:{view}" in self.scopes

    @property
    def can_read_pii(self) -> bool:
        return SCOPE_PII in self.scopes

    def has(self, scope: str) -> bool:
        return scope in self.scopes


def generate_keypair_pem() -> tuple[bytes, bytes]:
    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private_pem, public_pem


def load_private_key(pem: bytes) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("expected an Ed25519 private key")
    return key


def load_public_key(pem: bytes) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("expected an Ed25519 public key")
    return key


def load_public_key_file(path: Path) -> Ed25519PublicKey:
    return load_public_key(path.read_bytes())


def mint_token(
    private_key: Ed25519PrivateKey,
    *,
    subject: str,
    scopes: Iterable[str],
    issuer: str,
    audience: str,
    ttl_seconds: int = 900,
    now: float | None = None,
) -> str:
    issued = int(time.time() if now is None else now)
    claims = {
        "iss": issuer,
        "aud": audience,
        "sub": subject,
        "iat": issued,
        "exp": issued + ttl_seconds,
        "scope": " ".join(sorted(set(scopes))),
    }
    return jwt.encode(claims, private_key, algorithm="EdDSA")


class TokenVerifier:
    def __init__(
        self,
        public_key: Ed25519PublicKey,
        *,
        issuer: str,
        audience: str,
        leeway_seconds: int = 30,
    ) -> None:
        self._key = public_key
        self._issuer = issuer
        self._audience = audience
        self._leeway = leeway_seconds

    def verify(self, token: str | None) -> Caller | None:
        """Return the caller for a valid token, else None (no reason is exposed)."""
        if not token or len(token) > _MAX_TOKEN_CHARS:
            return None
        try:
            claims = jwt.decode(
                token,
                self._key,
                algorithms=["EdDSA"],
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError:
            return None
        subject = claims.get("sub")
        scope = claims.get("scope", "")
        if (
            not isinstance(subject, str)
            or not _SUBJECT.fullmatch(subject)
            or not isinstance(scope, str)
        ):
            return None
        scopes = scope.split()
        if len(scopes) > _MAX_SCOPES or not all(_SCOPE.fullmatch(item) for item in scopes):
            return None
        return Caller(subject=subject, scopes=frozenset(scopes))


def bearer_token(headers: Mapping[str, str]) -> str | None:
    auth = headers.get("authorization")
    if auth and auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    return None


def identity_from(verifier: TokenVerifier | None) -> Callable[[Mapping[str, str]], str | None]:
    """Rate-limit identity: the verified JWT subject, or None."""

    def identify(headers: Mapping[str, str]) -> str | None:
        if verifier is None:
            return None
        caller = verifier.verify(bearer_token(headers))
        return None if caller is None else caller.subject

    return identify
