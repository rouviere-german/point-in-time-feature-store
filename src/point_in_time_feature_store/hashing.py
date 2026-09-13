"""Keyed hashing of entity identifiers for logs, audit records and receipts.

Raw entity ids never appear in application logs. An HMAC-SHA256 under a service key lets an
operator correlate records for one entity (given the key) without storing the id itself; a plain
hash would not do, because entity ids are short and enumerable.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Iterable

__all__ = ["EntityHasher"]


class EntityHasher:
    def __init__(self, key: bytes) -> None:
        if len(key) < 32:
            raise ValueError("the HMAC key must be at least 32 bytes")
        self._key = key

    @classmethod
    def ephemeral(cls) -> EntityHasher:
        """A random per-process key (dev and tests); hashes do not survive restarts."""
        return cls(secrets.token_bytes(32))

    def entity(self, entity_id: str) -> str:
        return hmac.new(
            self._key, b"entity\x00" + entity_id.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def value(self, value: str) -> str:
        """Keyed digest of a PII value, so parity checks compare values without storing them."""
        return hmac.new(self._key, b"value\x00" + value.encode("utf-8"), hashlib.sha256).hexdigest()

    def batch(self, entity_ids: Iterable[str]) -> str:
        """One digest for a batch of entity ids (order-independent)."""
        digest = hashlib.sha256()
        for item in sorted(self.entity(entity_id) for entity_id in entity_ids):
            digest.update(bytes.fromhex(item))
        return digest.hexdigest()
