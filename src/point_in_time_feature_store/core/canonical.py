"""Canonical JSON encoding and hashing.

The encoding is deterministic for the JSON data model: object keys are sorted, separators carry
no whitespace, non-ASCII text is kept as UTF-8, and NaN/Infinity are rejected. Only ``str`` keys
are accepted, so ``{1: x}`` and ``{"1": x}`` can never collide. Floats use Python's shortest
round-trip ``repr``; this is stable across CPython builds but is not byte-identical to RFC 8785
for every exponent form, so hashes are only compared between Python processes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["CanonicalJSONError", "canonical_json", "sha256_hex"]


class CanonicalJSONError(TypeError):
    """Raised when a value is outside the JSON data model."""


def _check(value: Any, path: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJSONError(f"non-finite float at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJSONError(f"non-string key {type(key).__name__} at {path}")
            _check(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if not isinstance(value, list):
            raise CanonicalJSONError(f"use a list, not {type(value).__name__}, at {path}")
        for index, item in enumerate(value):
            _check(item, f"{path}[{index}]")
        return
    raise CanonicalJSONError(f"unsupported type {type(value).__name__} at {path}")


def canonical_json(value: Any) -> str:
    """Return the canonical JSON text for ``value``."""
    _check(value, "$")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_hex(data: str | bytes) -> str:
    """Return the lowercase hex SHA-256 of ``data`` (text is encoded as UTF-8)."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()
