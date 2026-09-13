from __future__ import annotations

import base64
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from point_in_time_feature_store.serving.auth import (
    Caller,
    TokenVerifier,
    bearer_token,
    generate_keypair_pem,
    identity_from,
    load_private_key,
    load_public_key,
    mint_token,
)

pytestmark = pytest.mark.unit

ISSUER, AUDIENCE = "issuer-test", "audience-test"
PRIVATE_PEM, PUBLIC_PEM = generate_keypair_pem()
PRIVATE = load_private_key(PRIVATE_PEM)
VERIFIER = TokenVerifier(
    load_public_key(PUBLIC_PEM), issuer=ISSUER, audience=AUDIENCE, leeway_seconds=0
)


def token(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "subject": "svc-a",
        "scopes": ["features:read:customer_profile", "pii:read"],
        "issuer": ISSUER,
        "audience": AUDIENCE,
    }
    kwargs.update(overrides)
    return mint_token(PRIVATE, **kwargs)  # type: ignore[arg-type]


def test_valid_tokens_yield_scoped_callers() -> None:
    caller = VERIFIER.verify(token())
    assert caller == Caller("svc-a", frozenset({"features:read:customer_profile", "pii:read"}))
    assert caller.can_read_view("customer_profile")
    assert not caller.can_read_view("transaction_stats")
    assert caller.can_read_pii
    wildcard = VERIFIER.verify(token(scopes=["features:read:*"]))
    assert wildcard is not None
    assert wildcard.can_read_view("anything")
    assert not wildcard.can_read_pii
    assert not wildcard.has("entities:erase")


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "",
        "not.a.jwt",
        "x" * 5_000,
        token(issuer="someone-else"),
        token(audience="another-service"),
        token(ttl_seconds=-120),
        token(subject="bad subject with spaces"),
        token(scopes=["UPPER:case"]),
        token(scopes=[f"s{i}" for i in range(70)]),
    ],
)
def test_invalid_tokens_are_rejected(bad: str | None) -> None:
    assert VERIFIER.verify(bad) is None


def _segments(value: dict[str, object]) -> str:
    return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()


def test_algorithm_confusion_and_tampering_are_rejected() -> None:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "svc-a",
        "iat": now,
        "exp": now + 60,
        "scope": "pii:read",
    }
    unsigned = _segments({"alg": "none", "typ": "JWT"}) + "." + _segments(claims) + "."
    assert VERIFIER.verify(unsigned) is None
    header, _payload, signature = token().split(".")
    forged = _segments({**claims, "scope": "features:read:* pii:read entities:erase"})
    assert VERIFIER.verify(f"{header}.{forged}.{signature}") is None
    other_private, _ = generate_keypair_pem()
    assert (
        VERIFIER.verify(
            mint_token(
                load_private_key(other_private),
                subject="svc-a",
                scopes=[],
                issuer=ISSUER,
                audience=AUDIENCE,
            )
        )
        is None
    )
    missing_exp = jwt.encode(
        {k: v for k, v in claims.items() if k != "exp"}, PRIVATE, algorithm="EdDSA"
    )
    assert VERIFIER.verify(missing_exp) is None
    non_string_scope = jwt.encode({**claims, "scope": ["pii:read"]}, PRIVATE, algorithm="EdDSA")
    assert VERIFIER.verify(non_string_scope) is None


def test_key_loading_refuses_other_key_types() -> None:
    ec_key = ec.generate_private_key(ec.SECP256R1())
    ec_private = ec_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    ec_public = ec_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    with pytest.raises(TypeError, match="Ed25519"):
        load_private_key(ec_private)
    with pytest.raises(TypeError, match="Ed25519"):
        load_public_key(ec_public)


def test_bearer_parsing_and_rate_limit_identity() -> None:
    assert bearer_token({"authorization": "Bearer abc"}) == "abc"
    assert bearer_token({"authorization": "bearer  abc "}) == "abc"
    assert bearer_token({"authorization": "Basic abc"}) is None
    assert bearer_token({"authorization": "Bearer "}) is None
    assert bearer_token({}) is None
    identify = identity_from(VERIFIER)
    assert identify({"authorization": f"Bearer {token()}"}) == "svc-a"
    assert identify({"authorization": "Bearer forged"}) is None
    assert identity_from(None)({"authorization": f"Bearer {token()}"}) is None
