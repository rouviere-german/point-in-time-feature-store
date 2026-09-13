"""PII and secret redaction for log events, audit records and exports.

Two layers:
1. Key-based: values under sensitive keys (``password``, ``apiKey``, ``Authorization`` ...) are
   replaced wholesale, whatever they contain. Key matching ignores case and separators, so
   ``private_key``, ``privateKey`` and ``PRIVATE-KEY`` are treated alike. Strong words (password,
   secret, credential, api key, private key) mark a key wherever they appear (``password_hash``,
   ``aws_secret_access_key``); generic ones (token, cookie, access/signing key) only at the end,
   so ``input_tokens`` or ``cache_key`` stay readable.
2. Pattern-based: free text and mapping keys are scanned for private-key blocks, JWTs, well-known
   API-key/token formats, bearer credentials, passwords in URLs (``scheme://user:pass@host``),
   ``secret=...`` pairs (quoted values are replaced up to the closing quote, spaces and escaped
   quotes included), e-mail addresses (including non-ASCII ones), payment-card-like numbers and
   phone numbers.

Log events can carry values that are not JSON (sets, bytes, arbitrary objects). A renderer would
print their ``repr`` without redaction, so the structlog processor turns them into redacted text
first (:func:`redact_log_value`).

Known limits (documented in docs/THREAT_MODEL.md): names, postal addresses and free-form
identifiers are not detected; digit runs are only treated as cards when they pass the Luhn check
(or use 4-4-4-4 grouping), so some card-like numbers survive and some Luhn-valid IDs are redacted.
Hex digests (32 or more hex characters) and UUIDs are never treated as card numbers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping
from typing import Any

__all__ = [
    "REDACTED",
    "is_sensitive_key",
    "luhn_valid",
    "redact_log_value",
    "redact_processor",
    "redact_text",
    "redact_value",
]

REDACTED = "[REDACTED]"
_EMAIL = "[REDACTED:EMAIL]"
_PHONE = "[REDACTED:PHONE]"
_CARD = "[REDACTED:CARD]"
# Placeholder labels, not credentials.
_TOKEN = "[REDACTED:TOKEN]"  # noqa: S105  # nosec B105
_SECRET = "[REDACTED:SECRET]"  # noqa: S105  # nosec B105
_PRIVATE_KEY = "[REDACTED:PRIVATE_KEY]"
_PLACEHOLDERS = frozenset({REDACTED, _EMAIL, _PHONE, _CARD, _TOKEN, _SECRET, _PRIVATE_KEY})

_MAX_DEPTH = 12
_CARD_PASSES = 4
# structlog's ProcessorFormatter bookkeeping keys; removed again before rendering.
_PROCESSOR_META_KEYS = frozenset({"_record", "_from_structlog"})

# Compared against the key lower-cased with every non-alphanumeric character removed. Words mark a
# key wherever they appear; suffixes only at the end (``token`` would otherwise hit input_tokens).
_SENSITIVE_WORDS = (
    "password",
    "passwd",
    "passphrase",
    "secret",
    "credential",
    "apikey",
    "privatekey",
)
_SENSITIVE_SUFFIXES = (
    "token",
    "authorization",
    "cookie",
    "sessionid",
    "cardnumber",
    "cvv",
    "ssn",
    "accesskey",
    "signingkey",
    "encryptionkey",
)
_NON_ALNUM = re.compile(r"[^a-z0-9]")

_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
    re.DOTALL,
)
# Linear-time rule for every pattern below: a match may start only at the beginning of a run of
# the characters it consumes (a lookbehind), and runs are consumed possessively (*+, ++) or with
# RFC bounds. Otherwise a long run is rescanned from every position inside it, which is quadratic
# (seconds of CPU for a 30K-character log line or /v1/redact body).
_JWT = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{5,}+\.[A-Za-z0-9_-]{5,}+\.[A-Za-z0-9_-]{5,}+"
)
_KNOWN_TOKENS = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"sk-ant-[A-Za-z0-9_-]{16,}"
    r"|sk-(?:proj-)?[A-Za-z0-9_-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|github_pat_[A-Za-z0-9_]{30,}"
    r"|glpat-[A-Za-z0-9_-]{20,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_-]{35}"
    r")"
)
_AUTH_SCHEME = re.compile(r"(?i)\b(bearer|basic|token)\s+[A-Za-z0-9._~+/=-]{8,}")
# scheme://user:password@host (database DSNs, broker URLs): only the password is replaced.
_URL_CREDENTIALS = re.compile(
    r"(?i)(?<![a-z0-9+.-])(?P<prefix>[a-z][a-z0-9+.-]*+://[^\s:/?#@\[\]]*+:)"
    r"(?P<password>[^\s/?#@]++)@"
)
# The name may follow "_" or "-" (DB_PASSWORD, csrf_token, aws_secret_access_key), unlike a \b
# boundary. The separator may carry a closing quote, as in JSON. A quoted value is replaced up to
# its closing quote (spaces and backslash escapes included); a bare value runs to whitespace, a
# quote or one of & , ; (query strings and cookie lists).
_SECRET_PAIR = re.compile(
    r"(?i)(?<![a-z0-9])"
    r"(?P<name>(?:api|access|secret|private|signing|encryption)[_-]?key"
    r"|(?:access|refresh)[_-]?token|client[_-]?secret|secret|password|passwd|passphrase|token)"
    r"(?P<sep>[\"']?\s*[=:]\s*)"
    r"(?:\"(?P<dq>(?:[^\"\\\n]|\\.)+)\"|'(?P<sq>(?:[^'\\\n]|\\.)+)'"
    r"|(?P<open>[\"']?)(?P<bare>[^\s\"'&,;]+))"
)
# \w is Unicode-aware, so internationalized addresses (jöhn@exämple.de) are matched too. Domain
# labels are at most 63 characters and so is the TLD (RFC 1035).
_EMAIL_ADDR = re.compile(
    r"(?<![\w!#$%&'*+/=?^`{|}~.-])[\w!#$%&'*+/=?^`{|}~.-]++@"
    r"[^\W_](?:[\w-]{0,61}[^\W_])?(?:\.[^\W_](?:[\w-]{0,61}[^\W_])?)*\.[^\W\d_]{2,63}"
)
# Hex identifiers are matched first and left alone, so digit runs inside SHA-256 digests, request
# ids and UUIDs (which audit records are full of) are never rewritten as card numbers.
_CARD_OR_HEX_ID = re.compile(
    r"(?P<hex>(?<![0-9A-Fa-f])"
    r"(?:[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}|[0-9A-Fa-f]{32,})"
    r"(?![0-9A-Fa-f]))"
    r"|(?P<card>(?<!\d)\d(?:[ -]?\d){12,18}(?!\d))"
)
_GROUPED_CARD = re.compile(r"^\d{4}([ -])\d{4}\1\d{4}\1\d{4}$")
_CARD_SEPARATOR = re.compile(r"([ -])")
_PHONE_INTL = re.compile(r"(?<![\w+])\+\d{1,3}(?:[ .-]?\(?\d{1,5}\)?){1,6}(?![\d])")
_PHONE_NATIONAL = re.compile(r"(?<![\w(])(?:\(\d{3}\)\s?|\d{3}[ .-])\d{3}[ .-]\d{4}(?![\d])")


def is_sensitive_key(key: str) -> bool:
    """True when a mapping key names a credential or regulated identifier.

    Case and separators are ignored: ``session_id``, ``sessionId`` and ``X-Session-ID`` all match.
    """
    normalized = _NON_ALNUM.sub("", key.lower())
    return normalized.endswith(_SENSITIVE_SUFFIXES) or any(
        word in normalized for word in _SENSITIVE_WORDS
    )


def luhn_valid(digits: str) -> bool:
    """Luhn (mod 10) checksum used by payment card numbers."""
    if not digits.isdigit():
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _looks_like_card(candidate: str) -> bool:
    digits = candidate.replace(" ", "").replace("-", "")
    return 13 <= len(digits) <= 19 and (
        luhn_valid(digits) or _GROUPED_CARD.fullmatch(candidate) is not None
    )


def _join_groups(groups: list[str], separators: list[str], start: int, stop: int) -> str:
    text = groups[start]
    for index in range(start + 1, stop):
        text += separators[index - 1] + groups[index]
    return text


def _redact_card_candidate(candidate: str) -> str:
    """Redact the whole candidate, or the longest card-shaped runs of its digit groups.

    Checking only the whole candidate would let a short number in front of a card (for example
    ``ref 12 4111 1111 1111 1111``) hide the card from the Luhn check.
    """
    if _looks_like_card(candidate):
        return _CARD
    parts = _CARD_SEPARATOR.split(candidate)
    groups, separators = parts[0::2], parts[1::2]
    pieces: list[str] = []
    start = 0
    while start < len(groups):
        stop = next(
            (
                end
                for end in range(len(groups), start, -1)
                if _looks_like_card(_join_groups(groups, separators, start, end))
            ),
            None,
        )
        if stop is None:
            pieces.append(groups[start])
            stop = start + 1
        else:
            pieces.append(_CARD)
        if stop < len(groups):
            pieces.append(separators[stop - 1])
        start = stop
    return "".join(pieces)


def _card_or_hex_id(match: re.Match[str]) -> str:
    hex_id = match.group("hex")
    if hex_id is not None:
        return hex_id
    return _redact_card_candidate(match.group("card"))


def _secret_pair(match: re.Match[str]) -> str:
    if match.group("dq") is not None:
        value, opening, closing = match.group("dq"), '"', '"'
    elif match.group("sq") is not None:
        value, opening, closing = match.group("sq"), "'", "'"
    else:
        value, opening, closing = match.group("bare"), match.group("open"), ""
    if value in _PLACEHOLDERS:
        return match.group(0)
    return f"{match.group('name')}{match.group('sep')}{opening}{_SECRET}{closing}"


def _url_credentials(match: re.Match[str]) -> str:
    if match.group("password") in _PLACEHOLDERS:
        return match.group(0)
    return f"{match.group('prefix')}{_SECRET}@"


def _intl_phone(match: re.Match[str]) -> str:
    digit_count = sum(char.isdigit() for char in match.group(0))
    return _PHONE if 7 <= digit_count <= 15 else match.group(0)


def redact_text(text: str) -> str:
    """Return ``text`` with secrets and PII replaced by typed placeholders. Idempotent."""
    text = _PRIVATE_KEY_BLOCK.sub(_PRIVATE_KEY, text)
    # The substring checks are only fast paths; the patterns are linear without them.
    if "eyJ" in text:
        text = _JWT.sub(_TOKEN, text)
    text = _KNOWN_TOKENS.sub(_TOKEN, text)
    text = _AUTH_SCHEME.sub(lambda m: f"{m.group(1)} {_TOKEN}", text)
    if "://" in text:
        text = _URL_CREDENTIALS.sub(_url_credentials, text)
    text = _SECRET_PAIR.sub(_secret_pair, text)
    if "@" in text:
        text = _EMAIL_ADDR.sub(_EMAIL, text)
    for _ in range(_CARD_PASSES):
        updated = _CARD_OR_HEX_ID.sub(_card_or_hex_id, text)
        if updated == text:
            break
        text = updated
    text = _PHONE_INTL.sub(_intl_phone, text)
    return _PHONE_NATIONAL.sub(_PHONE, text)


def _redact_key(key: Any, *, stringify: bool) -> Any:
    if isinstance(key, str):
        return redact_text(key)
    if not stringify or key is None or isinstance(key, (bool, int, float)):
        return key
    return redact_text(repr(key))


def _redact(value: Any, *, depth: int, stringify: bool) -> Any:  # noqa: PLR0911
    if depth > _MAX_DEPTH:
        return REDACTED
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            sensitive = isinstance(key, str) and is_sensitive_key(key) and item is not None
            redacted[_redact_key(key, stringify=stringify)] = (
                REDACTED if sensitive else _redact(item, depth=depth + 1, stringify=stringify)
            )
        return redacted
    if isinstance(value, list):
        return [_redact(item, depth=depth + 1, stringify=stringify) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, depth=depth + 1, stringify=stringify) for item in value)
    if not stringify:
        return value
    if isinstance(value, (set, frozenset)):
        return [_redact(item, depth=depth + 1, stringify=True) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return redact_text(bytes(value).decode("utf-8", errors="replace"))
    return redact_text(repr(value))


def redact_value(value: Any) -> Any:
    """Recursively redact strings (values and string keys) inside mappings, lists and tuples.

    Values under sensitive keys are replaced entirely. Structures deeper than the depth limit
    are replaced with ``[REDACTED]`` rather than passed through unchecked. Values of other types
    are returned unchanged; use :func:`redact_log_value` for data that is about to be rendered.
    """
    return _redact(value, depth=0, stringify=False)


def redact_log_value(value: Any) -> Any:
    """Like :func:`redact_value`, but values a JSON renderer would ``repr`` become redacted text.

    Sets become lists, bytes are decoded as UTF-8 (invalid bytes replaced), and any other object
    is replaced by its redacted ``repr``.
    """
    return _redact(value, depth=0, stringify=True)


def redact_processor(
    logger: object, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: redact every field (including rendered exceptions) in place."""
    del logger, method_name
    for key in list(event_dict.keys()):
        if key in _PROCESSOR_META_KEYS:
            continue
        item = event_dict[key]
        if is_sensitive_key(key) and item is not None:
            event_dict[key] = REDACTED
        else:
            event_dict[key] = redact_log_value(item)
    return event_dict
