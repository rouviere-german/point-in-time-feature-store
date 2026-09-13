"""Sampled log of served online values, the input of the skew checker.

Each sampled read stores the request time, the entity, the view version, the watermark snapshot
read atomically with the values, the served status, the value event time and the values. Values
of PII-tagged features are stored only as keyed HMAC digests, so parity can be checked without
keeping the personal data. Entity ids are stored because recomputation needs them; the log is an
access-controlled store and entity erasure deletes an entity's rows.

Sampling is deterministic in the request id (SHA-256 below ``rate * 2**64``).
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from point_in_time_feature_store.definitions import FeatureView
from point_in_time_feature_store.hashing import EntityHasher
from point_in_time_feature_store.offline.oracle import Value

__all__ = ["LoggedRead", "RequestLog", "logged_values"]

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS request_log (id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "request_id TEXT NOT NULL, ts_us INTEGER NOT NULL, entity_id TEXT NOT NULL, "
    "view TEXT NOT NULL, "
    "version_hash TEXT NOT NULL, watermark_low_us INTEGER, watermark_high_us INTEGER, "
    "status TEXT NOT NULL, event_time_us INTEGER, vals TEXT NOT NULL)"
)
_INDEX = "CREATE INDEX IF NOT EXISTS request_log_entity ON request_log (entity_id)"
_SIDE_FILES = ("-wal", "-journal")


def _hmac_input(value: Value) -> str:
    kind = type(value).__name__
    return f"{kind}:{value!r}"


def logged_values(
    view: FeatureView, values: Mapping[str, Value], hasher: EntityHasher
) -> dict[str, JsonValue]:
    """Values as stored in the log: PII features become ``{"hmac": ...}``."""
    encoded: dict[str, JsonValue] = {}
    for name in view.feature_names:
        value = values.get(name)
        if value is None:
            encoded[name] = None
        elif view.feature(name).pii:
            encoded[name] = {"hmac": hasher.value(_hmac_input(value))}
        else:
            encoded[name] = value
    return encoded


@dataclass(frozen=True, slots=True)
class LoggedRead:
    request_id: str
    ts_us: int
    entity_id: str
    view: str
    version: str
    watermark_low_us: int | None
    watermark_high_us: int | None
    status: str
    event_time_us: int | None
    values: dict[str, JsonValue]


class RequestLog:
    def __init__(self, path: Path, *, sample_rate: float) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be within [0, 1]")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # Deleted cells and freed pages are overwritten with zeros instead of lingering on disk.
        self._conn.execute("PRAGMA secure_delete=ON")
        self._conn.execute(_SCHEMA)
        self._conn.execute(_INDEX)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def should_sample(self, request_id: str) -> bool:
        if self.sample_rate >= 1.0:
            return True
        draw = int.from_bytes(hashlib.sha256(request_id.encode("utf-8")).digest()[:8], "big")
        return draw < int(self.sample_rate * 2**64)

    def record(self, reads: Sequence[LoggedRead]) -> None:
        if not reads:
            return
        rows = [
            (
                read.request_id,
                read.ts_us,
                read.entity_id,
                read.view,
                read.version,
                read.watermark_low_us,
                read.watermark_high_us,
                read.status,
                read.event_time_us,
                json.dumps(read.values, sort_keys=True, allow_nan=True),
            )
            for read in reads
        ]
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.executemany(
                    "INSERT INTO request_log (request_id, ts_us, entity_id, view, version_hash, "
                    "watermark_low_us, watermark_high_us, status, event_time_us, vals) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM request_log").fetchone()[0])

    def sample(self, limit: int, *, seed: int) -> list[LoggedRead]:
        with self._lock:
            ids = [
                int(row[0]) for row in self._conn.execute("SELECT id FROM request_log ORDER BY id")
            ]
        chosen = sorted(random.Random(seed).sample(ids, min(limit, len(ids))))  # noqa: S311  # nosec B311
        reads: list[LoggedRead] = []
        with self._lock:
            for start in range(0, len(chosen), 500):
                batch = chosen[start : start + 500]
                placeholders = ",".join("?" * len(batch))
                query = (
                    "SELECT request_id, ts_us, entity_id, view, version_hash, watermark_low_us, "  # noqa: S608  # nosec B608
                    "watermark_high_us, status, event_time_us, vals FROM request_log WHERE id IN ("
                    + placeholders
                    + ") ORDER BY id"
                )
                for row in self._conn.execute(query, batch):
                    reads.append(
                        LoggedRead(
                            request_id=str(row[0]),
                            ts_us=int(row[1]),
                            entity_id=str(row[2]),
                            view=str(row[3]),
                            version=str(row[4]),
                            watermark_low_us=None if row[5] is None else int(row[5]),
                            watermark_high_us=None if row[6] is None else int(row[6]),
                            status=str(row[7]),
                            event_time_us=None if row[8] is None else int(row[8]),
                            values=json.loads(row[9]),
                        )
                    )
        return reads

    def count_entity(self, entity_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT count(*) FROM request_log WHERE entity_id = ?", (entity_id,)
            ).fetchone()
        return int(row[0])

    def storage_files(self) -> list[Path]:
        """The database file and its WAL/journal side files that currently exist on disk."""
        candidates = [self.path, *(self.path.with_name(self.path.name + s) for s in _SIDE_FILES)]
        return [path for path in candidates if path.exists()]

    def delete_entity(self, entity_id: str) -> int:
        """Delete an entity's rows and scrub them from the files, not only from the table.

        ``secure_delete`` zeroes the deleted cells; the WAL still holds the frames that inserted
        them, so it is checkpointed and truncated, and ``VACUUM`` rebuilds the file without free
        pages. Filesystem-level remnants (journaling, SSD wear levelling, snapshots) are out of
        reach of SQLite and are documented in the threat model.
        """
        with self._lock:
            cursor = self._conn.execute("DELETE FROM request_log WHERE entity_id = ?", (entity_id,))
            deleted = int(cursor.rowcount)
            self._checkpoint()
            self._conn.execute("VACUUM")
            self._checkpoint()
            return deleted

    def _checkpoint(self) -> None:
        busy, _, _ = self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if int(busy):
            raise RuntimeError("request log checkpoint was blocked by another connection")
