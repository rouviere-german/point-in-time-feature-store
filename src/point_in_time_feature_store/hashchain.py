"""Hash-chained, append-only tables in SQLite (registry change log and feature access audit).

Each entry stores ``entry_hash = SHA-256(canonical JSON of seq, ts_us, actor, action, payload,
prev_hash)``. Editing, deleting or reordering an entry in place breaks every later hash. The chain
is unkeyed, so a truncated tail, or a chain rebuilt from scratch by someone who can write the
SQLite file, still verifies on its own; that is detected only against a head kept elsewhere
(``expected_head``) or a signed anchor (:mod:`point_in_time_feature_store.anchors`).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from point_in_time_feature_store.core.canonical import canonical_json, sha256_hex

__all__ = ["GENESIS", "ChainEntry", "ChainName", "ChainVerification", "HashChain", "entry_hash"]

GENESIS = "0" * 64
ChainName = Literal["change_log", "access_audit"]

# Table names come from this fixed map; statements are constants, values are always bound.
_STATEMENTS: dict[ChainName, dict[str, str]] = {
    "change_log": {
        "create": (
            "CREATE TABLE IF NOT EXISTS change_log (seq INTEGER PRIMARY KEY, "
            "ts_us INTEGER NOT NULL, "
            "actor TEXT NOT NULL, action TEXT NOT NULL, payload TEXT NOT NULL, "
            "prev_hash TEXT NOT NULL, entry_hash TEXT NOT NULL)"
        ),
        "last": "SELECT seq, entry_hash FROM change_log ORDER BY seq DESC LIMIT 1",
        "insert": (
            "INSERT INTO change_log (seq, ts_us, actor, action, payload, prev_hash, entry_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        ),
        "all": (
            "SELECT seq, ts_us, actor, action, payload, prev_hash, entry_hash FROM change_log "
            "ORDER BY seq"
        ),
    },
    "access_audit": {
        "create": (
            "CREATE TABLE IF NOT EXISTS access_audit (seq INTEGER PRIMARY KEY, "
            "ts_us INTEGER NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, "
            "payload TEXT NOT NULL, prev_hash TEXT NOT NULL, entry_hash TEXT NOT NULL)"
        ),
        "last": "SELECT seq, entry_hash FROM access_audit ORDER BY seq DESC LIMIT 1",
        "insert": (
            "INSERT INTO access_audit (seq, ts_us, actor, action, payload, prev_hash, entry_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        ),
        "all": (
            "SELECT seq, ts_us, actor, action, payload, prev_hash, entry_hash FROM access_audit "
            "ORDER BY seq"
        ),
    },
}


def entry_hash(
    seq: int, ts_us: int, actor: str, action: str, payload: dict[str, JsonValue], prev_hash: str
) -> str:
    return sha256_hex(
        canonical_json(
            {
                "seq": seq,
                "ts_us": ts_us,
                "actor": actor,
                "action": action,
                "payload": payload,
                "prev_hash": prev_hash,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class ChainEntry:
    seq: int
    ts_us: int
    actor: str
    action: str
    payload: dict[str, JsonValue]
    prev_hash: str
    entry_hash: str


@dataclass(frozen=True, slots=True)
class ChainVerification:
    ok: bool
    entries: int
    head: str
    first_bad_seq: int | None = None


class HashChain:
    """One hash-chained table. Callers serialize access (the registry holds a lock)."""

    def __init__(self, connection: sqlite3.Connection, name: ChainName) -> None:
        self._conn = connection
        self._sql = _STATEMENTS[name]
        self.name = name
        self._conn.execute(self._sql["create"])

    def append_in_transaction(
        self, ts_us: int, actor: str, action: str, payload: dict[str, JsonValue]
    ) -> ChainEntry:
        """Append inside a transaction the caller already opened (``BEGIN IMMEDIATE``)."""
        row = self._conn.execute(self._sql["last"]).fetchone()
        seq, prev = (int(row[0]) + 1, str(row[1])) if row else (1, GENESIS)
        digest = entry_hash(seq, ts_us, actor, action, payload, prev)
        self._conn.execute(
            self._sql["insert"],
            (seq, ts_us, actor, action, canonical_json(payload), prev, digest),
        )
        return ChainEntry(seq, ts_us, actor, action, payload, prev, digest)

    def entries(self) -> list[ChainEntry]:
        return [
            ChainEntry(
                int(row[0]),
                int(row[1]),
                str(row[2]),
                str(row[3]),
                json.loads(row[4]),
                str(row[5]),
                str(row[6]),
            )
            for row in self._conn.execute(self._sql["all"]).fetchall()
        ]

    def head(self) -> str:
        row = self._conn.execute(self._sql["last"]).fetchone()
        return str(row[1]) if row else GENESIS

    def verify(self, expected_head: str | None = None) -> ChainVerification:
        prev = GENESIS
        count = 0
        for expected_seq, entry in enumerate(self.entries(), start=1):
            count += 1
            recomputed = entry_hash(
                entry.seq, entry.ts_us, entry.actor, entry.action, entry.payload, prev
            )
            if (
                entry.seq != expected_seq
                or entry.prev_hash != prev
                or entry.entry_hash != recomputed
            ):
                return ChainVerification(False, count, prev, first_bad_seq=expected_seq)
            prev = entry.entry_hash
        if expected_head is not None and expected_head != prev:
            return ChainVerification(False, count, prev, first_bad_seq=count + 1)
        return ChainVerification(True, count, prev)
