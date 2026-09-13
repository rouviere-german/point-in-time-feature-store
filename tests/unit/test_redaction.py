from __future__ import annotations

import time

import pytest
from hypothesis import given
from hypothesis import strategies as st

from point_in_time_feature_store.core.redaction import (
    REDACTED,
    is_sensitive_key,
    luhn_valid,
    redact_processor,
    redact_text,
    redact_value,
)

pytestmark = pytest.mark.unit

# Secret-shaped fixtures are assembled at runtime so secret scanners do not flag this file.
ANTHROPIC_KEY = "sk-" + "ant-api03-" + "Q" * 30
GITHUB_TOKEN = "ghp" + "_" + "b" * 36
AWS_KEY_ID = "AKIA" + "Z" * 16
JWT = "eyJ" + "hbGciOiJIUzI1NiJ9" + "." + "eyJzdWIiOiIxMjMifQ" + "." + "c2lnbmF0dXJlX2hlcmU"
PEM = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END " + "RSA PRIVATE KEY-----"


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("contact jane.doe+ops@example.co.uk today", "jane.doe+ops@example.co.uk"),
        (f"key {ANTHROPIC_KEY} used", ANTHROPIC_KEY),
        (f"token {GITHUB_TOKEN}", GITHUB_TOKEN),
        (f"aws {AWS_KEY_ID}", AWS_KEY_ID),
        ("Authorization: Bearer abcdefghijklmnop123", "abcdefghijklmnop123"),
        (f"jwt={JWT}", JWT),
        ("login password=hunter2hunter2 ok", "hunter2hunter2"),
        ("pan 4111 1111 1111 1111 exp", "4111 1111 1111 1111"),
        ("pan 4111111111111111", "4111111111111111"),
        ("grouped 1234-5678-9012-3456", "1234-5678-9012-3456"),
        ("call +49 30 1234 5678", "+49 30 1234 5678"),
        ("call (415) 555-0134 now", "(415) 555-0134"),
        ("call 415.555.0134 now", "415.555.0134"),
        (f"key material {PEM}", "MIIEowIBAAKCAQEA"),
    ],
)
def test_known_secret_and_pii_formats_are_redacted(text: str, secret: str) -> None:
    assert secret not in redact_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "request took 12 ms",
        "at 2026-09-13T10:00:00Z",
        "version 1.2.3",
        "order 12345 shipped",
        "pi is 3.14159",
        "user_id=42",
        "input_tokens 1200",
    ],
)
def test_benign_text_is_left_alone(text: str) -> None:
    assert redact_text(text) == text


@given(st.emails())
def test_any_email_address_is_redacted(email: str) -> None:
    out = redact_text(f"user {email} signed in")
    assert email not in out
    assert "@" not in out


@given(
    country=st.integers(min_value=1, max_value=999),
    groups=st.lists(st.integers(min_value=100, max_value=9999), min_size=2, max_size=3),
)
def test_international_phone_numbers_are_redacted(country: int, groups: list[int]) -> None:
    number = f"+{country} " + " ".join(str(group) for group in groups)
    assert number not in redact_text(f"call {number} now")


@given(
    st.integers(min_value=200, max_value=999),
    st.integers(min_value=200, max_value=999),
    st.integers(min_value=0, max_value=9999),
)
def test_north_american_phone_numbers_are_redacted(area: int, exchange: int, line: int) -> None:
    number = f"({area}) {exchange}-{line:04d}"
    assert number not in redact_text(f"call {number} now")


def _complete_luhn(prefix: str) -> str:
    return next(prefix + str(d) for d in range(10) if luhn_valid(prefix + str(d)))


@given(st.text(alphabet="0123456789", min_size=12, max_size=18))
def test_luhn_valid_card_numbers_are_redacted(prefix: str) -> None:
    number = _complete_luhn(prefix)
    assert number not in redact_text(f"pan {number} end")


@given(st.text(max_size=300))
def test_redaction_is_idempotent(text: str) -> None:
    once = redact_text(text)
    assert redact_text(once) == once


@pytest.mark.parametrize(
    ("key", "sensitive"),
    [
        ("password", True),
        ("X-API-Key", True),
        ("Authorization", True),
        ("session_id", True),
        ("refresh_token", True),
        ("card_number", True),
        ("input_tokens", False),
        ("max_tokens", False),
        ("token_count", False),
        ("username", False),
    ],
)
def test_sensitive_key_detection(key: str, sensitive: bool) -> None:
    assert is_sensitive_key(key) is sensitive


def test_luhn_checksum() -> None:
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("41x1")


def test_redact_value_replaces_sensitive_keys_and_scrubs_nested_text() -> None:
    data = {
        "password": "pw",
        "token": None,
        "nested": {"api_key": 123, "note": "mail jane@example.com"},
        "items": [{"Authorization": "Bearer abcdefghijk"}],
        "input_tokens": 42,
        "pair": ("call 415-555-0134",),
    }
    out = redact_value(data)
    assert out["password"] == REDACTED
    assert out["token"] is None
    assert out["nested"]["api_key"] == REDACTED
    assert out["nested"]["note"] == "mail [REDACTED:EMAIL]"
    assert out["items"][0]["Authorization"] == REDACTED
    assert out["input_tokens"] == 42
    assert out["pair"] == ("call [REDACTED:PHONE]",)


def test_structures_beyond_the_depth_limit_are_replaced() -> None:
    nested: object = "jane@example.com"
    for _ in range(20):
        nested = [nested]
    current = redact_value(nested)
    while isinstance(current, list):
        current = current[0]
    assert current == REDACTED


def test_processor_redacts_fields_and_rendered_exceptions() -> None:
    event = {
        "event": "login for jane@example.com",
        "api_key": "abc",
        "exception": "Traceback (most recent call last): ValueError: password=supersecret1",
        "attempt": 2,
    }
    out = redact_processor(None, "info", event)
    assert out["event"] == "login for [REDACTED:EMAIL]"
    assert out["api_key"] == REDACTED
    assert "supersecret1" not in out["exception"]
    assert out["attempt"] == 2


_json_like = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=20),
    lambda children: (
        st.lists(children, max_size=3)
        | st.dictionaries(st.sampled_from(["a", "b", "note", "count"]), children, max_size=3)
    ),
    max_leaves=10,
)


def _shape(value: object) -> object:
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return type(value).__name__


@given(_json_like)
def test_redact_value_preserves_structure_and_types(value: object) -> None:
    assert _shape(redact_value(value)) == _shape(value)


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ('{"password": "hunter2hunter2"}', "hunter2hunter2"),
        ("{'api_key': 'k-live-0123456789'}", "k-live-0123456789"),
        ("DB_PASSWORD=hunter2hunter2", "hunter2hunter2"),
        ("GET /callback?token=abcdef1234567890&state=x", "abcdef1234567890"),
        ("csrf_token: abcdef1234567890", "abcdef1234567890"),
        ("password=[REDACTEDhunter2hunter2", "hunter2hunter2"),
        ("ref 12 4111 1111 1111 1111", "4111 1111 1111 1111"),
        ("order 7-4111-1111-1111-1111 paid", "4111-1111-1111-1111"),
    ],
)
def test_quoted_prefixed_and_embedded_secrets_are_redacted(text: str, secret: str) -> None:
    assert secret not in redact_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "call_hash=5d8f6cce532a7aeb57196be62344095936793400b3aeb3580d248b17d5518a86",
        "request 0f1e2d3c-4b5a-6978-1234-567890123452 done",
    ],
)
def test_hex_digests_and_uuids_are_not_mistaken_for_card_numbers(text: str) -> None:
    assert redact_text(text) == text


@pytest.mark.parametrize("key", ["privateKey", "sessionId", "cardNumber", "apiKey", "dbPassword"])
def test_camel_case_and_unseparated_sensitive_keys_are_detected(key: str) -> None:
    assert is_sensitive_key(key)


def test_mapping_keys_are_redacted_too() -> None:
    assert redact_value({"jane@example.com": 3, "count": 1}) == {"[REDACTED:EMAIL]": 3, "count": 1}


@pytest.mark.parametrize(
    ("text", "leaks"),
    [
        ('login {"password": "correct horse battery staple"}', ("correct", "horse", "staple")),
        ("password='my very secret'", ("my very", "secret'")),
        ('{"password": "ab\\"cd efgh ijkl"}', ("cd", "efgh", "ijkl")),
        ("password=abc ok", ("abc",)),
    ],
)
def test_quoted_short_and_escaped_secret_values_are_redacted_completely(
    text: str, leaks: tuple[str, ...]
) -> None:
    out = redact_text(text)
    assert [leak for leak in leaks if leak in out] == [], out


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "connect failed: postgresql://app:" + "s3cr3t" + "passw0rd@localhost:5432/app",
            "connect failed: postgresql://app:[REDACTED:SECRET]@localhost:5432/app",
        ),
        ("redis://:" + "hunter2" * 2 + "@cache:6379/0", "redis://:[REDACTED:SECRET]@cache:6379/0"),
        ("see https://example.com/docs?page=2", "see https://example.com/docs?page=2"),
    ],
)
def test_credentials_embedded_in_urls_are_redacted(text: str, expected: str) -> None:
    assert redact_text(text) == expected
    assert redact_text(expected) == expected


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("aws_secret_access_key=" + "wJalr" + "Q" * 30, "wJalr"),
        ("SECRET_KEY=" + "django-insecure-" + "7" * 12, "django-insecure"),
        ("passphrase: correct-horse-battery", "correct-horse"),
        ("private_key=" + "abcdef" * 3, "abcdef"),
        ("signing_key=" + "abcdef" * 3, "abcdef"),
    ],
)
def test_key_style_secret_names_are_redacted_in_free_text(text: str, secret: str) -> None:
    assert secret not in redact_text(text)


@pytest.mark.parametrize(
    ("key", "sensitive"),
    [
        ("secret_key", True),
        ("SECRET_KEY", True),
        ("aws_secret_access_key", True),
        ("access_key", True),
        ("signing_key", True),
        ("encryptionKey", True),
        ("password_hash", True),
        ("client_secret_value", True),
        ("credentials_json", True),
        ("cache_key", False),
        ("sort_key", False),
        ("primary_key", False),
        ("idempotency_key", False),
        ("output_tokens", False),
        ("request_sha256", False),
    ],
)
def test_secret_words_make_a_key_sensitive_wherever_they_appear(key: str, sensitive: bool) -> None:
    assert is_sensitive_key(key) is sensitive


@pytest.mark.parametrize(
    "text",
    [
        "a." * 15_000,  # a local-part run with no "@": rescanned from every start position
        "a." * 15_000 + "@://",  # the same run as a URL scheme, with the prefilters defeated
        "a@" + "b-" * 15_000,  # a domain run with no TLD
        "postgres://u:" + "p" * 30_000,  # a URL password with no "@"
        "eyJ-" * 7_500,  # JWT-shaped starts inside one token run
    ],
)
def test_redaction_stays_fast_on_adversarial_input(text: str) -> None:
    # Unbounded runs made these quadratic (about 6 s for 30K characters); logs and the redact
    # endpoint carry attacker-controlled text, so that is a CPU denial of service.
    start = time.perf_counter()
    redact_text(text)
    assert time.perf_counter() - start < 1.5


@pytest.mark.parametrize("email", ["jöhn.döe@example.com", "john@exämple.de", "用户@例子.广告"])
def test_non_ascii_email_addresses_are_redacted(email: str) -> None:
    assert redact_text(f"contact {email} today") == "contact [REDACTED:EMAIL] today"
