"""Entity erasure across every store, with Ed25519-signed receipts and purge verification.

Order of operations for ``erase(entity_id)``:

1. tombstone the entity's HMAC in the registry, so appends and materializations that start later
   drop its rows (appends and rewrites share the offline store lock);
2. rewrite every Parquet part file of every source that contains the entity (before/after SHA-256
   recorded; a part left empty is deleted);
3. delete the entity's online hashes for every version of every view ever applied;
4. delete its request-log rows and scrub the SQLite files (secure delete, WAL checkpoint and
   truncation, ``VACUUM``), so the id is gone from the bytes on disk, not only from the table;
5. append an ``erase_entity`` entry to the hash-chained change log;
6. sign a receipt over the canonical JSON of all of the above.

The receipt carries the entity's HMAC, never the id. Application logs only ever carry HMACs, so
there is nothing to purge there; :meth:`ErasureService.verify_purged` checks all stores through
their query interfaces, scans the raw bytes of the request-log and registry SQLite files (including
WAL and journal files) for the id, and scans any captured log text.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field

from point_in_time_feature_store.core.canonical import canonical_json, sha256_hex
from point_in_time_feature_store.core.logging import get_logger
from point_in_time_feature_store.hashing import EntityHasher
from point_in_time_feature_store.offline.store import OfflineStore
from point_in_time_feature_store.online.store import OnlineStore
from point_in_time_feature_store.registry import FeatureRegistry
from point_in_time_feature_store.serving.request_log import RequestLog
from point_in_time_feature_store.timeutil import Clock, from_us, system_clock_us

__all__ = [
    "ErasureReceipt",
    "ErasureService",
    "OfflineRewrite",
    "PurgeReport",
    "ReceiptSigner",
    "count_raw_occurrences",
    "public_key_id",
    "verify_receipt",
]

_log = get_logger("point_in_time_feature_store.erasure")


def public_key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return sha256_hex(raw)[:16]


class OfflineRewrite(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    partition: str
    rows_removed: int
    before_sha256: str
    after_sha256: str | None


class ErasureReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    receipt_version: int = 1
    receipt_id: str
    entity_hmac: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_by: str
    erased_at: str
    offline_rewrites: tuple[OfflineRewrite, ...]
    online_keys_deleted: int
    request_log_rows_deleted: int
    tombstoned: bool
    change_log_entry_hash: str
    key_id: str
    signature: str = ""

    def signing_payload(self) -> bytes:
        return canonical_json(self.model_dump(mode="json", exclude={"signature"})).encode("utf-8")


class ReceiptSigner:
    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._key = private_key
        self.key_id = public_key_id(private_key.public_key())

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._key.public_key()

    def sign(self, receipt: ErasureReceipt) -> ErasureReceipt:
        unsigned = receipt.model_copy(update={"key_id": self.key_id, "signature": ""})
        signature = self._key.sign(unsigned.signing_payload())
        return unsigned.model_copy(
            update={"signature": base64.urlsafe_b64encode(signature).decode("ascii")}
        )


def verify_receipt(receipt: ErasureReceipt, public_key: Ed25519PublicKey) -> bool:
    if receipt.key_id != public_key_id(public_key) or not receipt.signature:
        return False
    try:
        public_key.verify(base64.urlsafe_b64decode(receipt.signature), receipt.signing_payload())
    except (InvalidSignature, ValueError):
        return False
    return True


_SCAN_CHUNK_BYTES = 1 << 20


def count_raw_occurrences(paths: Iterable[Path], needle: bytes) -> int:
    """Occurrences of ``needle`` in the raw bytes of ``paths``, read in overlapping chunks."""
    if not needle:
        return 0
    keep = len(needle) - 1
    total = 0
    for path in paths:
        try:
            handle = path.open("rb")
        except FileNotFoundError:
            continue
        with handle:
            tail = b""
            while chunk := handle.read(_SCAN_CHUNK_BYTES):
                window = tail + chunk
                total += window.count(needle)
                # The tail is shorter than the needle, so no occurrence is counted twice.
                tail = window[-keep:] if keep else b""
    return total


@dataclass(slots=True)
class PurgeReport:
    offline_rows: dict[str, int] = field(default_factory=dict)
    online_keys: int = 0
    request_log_rows: int = 0
    log_occurrences: int = 0
    storage_occurrences: int = 0

    @property
    def clean(self) -> bool:
        return (
            not any(self.offline_rows.values())
            and self.online_keys == 0
            and self.request_log_rows == 0
            and self.log_occurrences == 0
            and self.storage_occurrences == 0
        )


class ErasureService:
    def __init__(
        self,
        registry: FeatureRegistry,
        store: OfflineStore,
        online: OnlineStore,
        request_log: RequestLog,
        hasher: EntityHasher,
        signer: ReceiptSigner,
        *,
        clock: Clock = system_clock_us,
    ) -> None:
        self.registry = registry
        self.store = store
        self.online = online
        self.request_log = request_log
        self.hasher = hasher
        self.signer = signer
        self.clock = clock

    def erase(self, entity_id: str, *, requested_by: str) -> ErasureReceipt:
        repo = self.registry.active_repo()
        receipt_id = uuid.uuid4().hex
        entity_hmac = self.hasher.entity(entity_id)
        self.registry.add_tombstone(entity_hmac, receipt_id)
        rewrites = [
            OfflineRewrite(
                source=item.source,
                partition=item.partition,
                rows_removed=item.rows_removed,
                before_sha256=item.before_sha256,
                after_sha256=item.after_sha256,
            )
            for source in repo.sources
            for item in self.store.erase_entity(source.name, entity_id)
        ]
        online_keys = self.online.delete_entity(entity_id, self.registry.view_versions())
        log_rows = self.request_log.delete_entity(entity_id)
        entry = self.registry.record_change(
            requested_by,
            "erase_entity",
            {
                "receipt_id": receipt_id,
                "entity_hmac": entity_hmac,
                "partitions_rewritten": len(rewrites),
                "rows_removed": sum(item.rows_removed for item in rewrites),
                "online_keys_deleted": online_keys,
                "request_log_rows_deleted": log_rows,
            },
        )
        receipt = self.signer.sign(
            ErasureReceipt(
                receipt_id=receipt_id,
                entity_hmac=entity_hmac,
                requested_by=requested_by,
                erased_at=from_us(self.clock()).astimezone(UTC).isoformat(),
                offline_rewrites=tuple(rewrites),
                online_keys_deleted=online_keys,
                request_log_rows_deleted=log_rows,
                tombstoned=True,
                change_log_entry_hash=entry.entry_hash,
                key_id=self.signer.key_id,
            )
        )
        _log.info(
            "entity_erased",
            entity_hmac=entity_hmac,
            receipt_id=receipt_id,
            partitions=len(rewrites),
            online_keys=online_keys,
        )
        return receipt

    def verify_purged(self, entity_id: str, *, log_texts: Iterable[str] = ()) -> PurgeReport:
        repo = self.registry.active_repo()
        return PurgeReport(
            offline_rows={
                source.name: self.store.entity_row_count(source.name, entity_id)
                for source in repo.sources
            },
            online_keys=self.online.count_entity_keys(entity_id, self.registry.view_versions()),
            request_log_rows=self.request_log.count_entity(entity_id),
            log_occurrences=sum(text.count(entity_id) for text in log_texts),
            storage_occurrences=count_raw_occurrences(
                [*self.request_log.storage_files(), *self.registry.storage_files()],
                entity_id.encode("utf-8"),
            ),
        )


def utc_now_iso() -> str:  # pragma: no cover - convenience for callers without a clock
    return datetime.now(UTC).isoformat()
