"""SQLite feature registry: plan/apply of definitions, watermarks, tombstones and audit chains.

``plan`` diffs a :class:`FeatureRepo` against the active definitions. Changes that alter the values
or the access rules of existing features (removing or redefining a feature, changing a view's
entity, source, TTL or lag declaration, loosening a source contract, removing a PII tag, deleting
anything) are *destructive*. ``apply`` refuses destructive plans unless the caller approves the
exact plan it reviewed (``plan_hash``); every apply, approved or rejected, is appended to the
hash-chained change log in the same transaction as the change itself.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from point_in_time_feature_store.core.canonical import canonical_json, sha256_hex
from point_in_time_feature_store.definitions import (
    EntityDefinition,
    FeatureRepo,
    FeatureView,
    SourceDefinition,
)
from point_in_time_feature_store.hashchain import ChainEntry, ChainVerification, HashChain
from point_in_time_feature_store.timeutil import Clock, system_clock_us

__all__ = [
    "ApprovalRequiredError",
    "FeatureRegistry",
    "Plan",
    "PlanMismatchError",
    "PlannedChange",
    "TrainingRun",
]

ObjectKind = Literal["entity", "source", "feature_view"]
_KINDS: tuple[ObjectKind, ...] = ("entity", "source", "feature_view")

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS objects (kind TEXT NOT NULL, name TEXT NOT NULL, "
    "version_hash TEXT NOT NULL, definition TEXT NOT NULL, first_applied_us INTEGER NOT NULL, "
    "PRIMARY KEY (kind, name, version_hash))",
    "CREATE TABLE IF NOT EXISTS active (kind TEXT NOT NULL, name TEXT NOT NULL, "
    "version_hash TEXT NOT NULL, metadata_hash TEXT NOT NULL, definition TEXT NOT NULL, "
    "PRIMARY KEY (kind, name))",
    "CREATE TABLE IF NOT EXISTS watermarks (view TEXT NOT NULL, version_hash TEXT NOT NULL, "
    "watermark_us INTEGER NOT NULL, updated_us INTEGER NOT NULL, PRIMARY KEY (view, version_hash))",
    "CREATE TABLE IF NOT EXISTS tombstones (entity_hmac TEXT PRIMARY KEY, "
    "erased_us INTEGER NOT NULL, receipt_id TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS training_runs (run_id TEXT PRIMARY KEY, model_name TEXT NOT NULL, "
    "created_us INTEGER NOT NULL, view_versions TEXT NOT NULL, dataset_digest TEXT NOT NULL)",
)


class ApprovalRequiredError(PermissionError):
    """A destructive plan was applied without approval."""


class PlanMismatchError(ValueError):
    """The approved plan hash does not match the plan computed at apply time."""


class PlannedChange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ObjectKind
    name: str
    action: Literal["create", "update", "delete"]
    old_version: str | None
    new_version: str | None
    destructive: bool
    reasons: tuple[str, ...]


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    changes: tuple[PlannedChange, ...]

    @property
    def destructive(self) -> bool:
        return any(change.destructive for change in self.changes)

    @property
    def is_empty(self) -> bool:
        return not self.changes

    @property
    def plan_hash(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))

    def render(self) -> str:
        if self.is_empty:
            return "No changes."
        lines = []
        for change in self.changes:
            marker = "!" if change.destructive else "+" if change.action == "create" else "~"
            if change.action == "delete":
                marker = "-"
            reasons = "; ".join(change.reasons)
            lines.append(f"{marker} {change.kind} {change.name} ({change.action}): {reasons}")
        lines.append(f"plan hash: {self.plan_hash}")
        if self.destructive:
            lines.append("destructive changes: re-run apply with --approve --plan-hash <hash>")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class TrainingRun:
    run_id: str
    model_name: str
    created_us: int
    view_versions: dict[str, str]
    dataset_digest: str


def _diff_entity(old: EntityDefinition, new: EntityDefinition) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    destructive = old.version_hash() != new.version_hash()
    if destructive:
        reasons.append("entity key format changed")
    if old.metadata_hash() != new.metadata_hash():
        reasons.append("metadata changed")
    return destructive, reasons


def _diff_source(old: SourceDefinition, new: SourceDefinition) -> tuple[bool, list[str]]:
    destructive = False
    reasons: list[str] = []
    old_columns = {column.name: column for column in old.columns}
    new_columns = {column.name: column for column in new.columns}
    for name in sorted(old_columns.keys() - new_columns.keys()):
        destructive = True
        reasons.append(f"column removed: {name}")
    for name in sorted(old_columns.keys() & new_columns.keys()):
        if old_columns[name] != new_columns[name]:
            destructive = True
            reasons.append(f"column redefined: {name}")
    for name in sorted(new_columns.keys() - old_columns.keys()):
        if new_columns[name].nullable:
            reasons.append(f"nullable column added: {name}")
        else:
            destructive = True
            reasons.append(f"non-nullable column added: {name}")
    if new.max_clock_skew_seconds > old.max_clock_skew_seconds:
        destructive = True
        reasons.append("contract loosened: max_clock_skew_seconds")
    if new.max_backfill_days > old.max_backfill_days:
        destructive = True
        reasons.append("contract loosened: max_backfill_days")
    if old.entity != new.entity:
        destructive = True
        reasons.append("entity changed")
    if old.metadata_hash() != new.metadata_hash():
        reasons.append("metadata changed")
    if not reasons and old.version_hash() != new.version_hash():
        reasons.append("contract tightened")
    return destructive, reasons


def _diff_view(old: FeatureView, new: FeatureView) -> tuple[bool, list[str]]:
    destructive = False
    reasons: list[str] = []
    for attribute in ("entity", "source", "ttl_seconds", "zero_ingestion_lag"):
        if getattr(old, attribute) != getattr(new, attribute):
            destructive = True
            reasons.append(f"{attribute} changed")
    old_features = {feature.name: feature for feature in old.features}
    new_features = {feature.name: feature for feature in new.features}
    for name in sorted(old_features.keys() - new_features.keys()):
        destructive = True
        reasons.append(f"feature removed: {name}")
    for name in sorted(old_features.keys() & new_features.keys()):
        before, after = old_features[name], new_features[name]
        if before.dtype != after.dtype or before.expression != after.expression:
            destructive = True
            reasons.append(f"feature redefined: {name}")
        if before.pii and not after.pii:
            destructive = True
            reasons.append(f"pii tag removed: {name}")
    for name in sorted(new_features.keys() - old_features.keys()):
        reasons.append(f"feature added: {name}")
    shared = old_features.keys() & new_features.keys()
    tagged = any(new_features[name].pii and not old_features[name].pii for name in shared)
    described = any(
        new_features[name].description != old_features[name].description for name in shared
    )
    labelled = (old.owner, old.description, old.tags) != (new.owner, new.description, new.tags)
    if tagged or described or labelled:
        reasons.append("metadata changed")
    return destructive, reasons


def _objects(
    repo: FeatureRepo, kind: ObjectKind
) -> dict[str, EntityDefinition | SourceDefinition | FeatureView]:
    items: Iterable[EntityDefinition | SourceDefinition | FeatureView]
    if kind == "entity":
        items = repo.entities
    elif kind == "source":
        items = repo.sources
    else:
        items = repo.feature_views
    return {item.name: item for item in items}


class FeatureRegistry:
    """Thread-safe registry over one SQLite file (WAL mode)."""

    def __init__(self, path: Path, *, clock: Clock = system_clock_us) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._clock = clock
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            for statement in _SCHEMA:
                self._conn.execute(statement)
            self.change_log = HashChain(self._conn, "change_log")
            self.access_audit = HashChain(self._conn, "access_audit")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----------------------------------------------------------------- definitions
    def active_repo(self) -> FeatureRepo:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, definition FROM active ORDER BY kind, name"
            ).fetchall()
        grouped: dict[str, list[dict[str, JsonValue]]] = {kind: [] for kind in _KINDS}
        for kind, definition in rows:
            grouped[str(kind)].append(json.loads(definition))
        return FeatureRepo.model_validate(
            {
                "entities": grouped["entity"],
                "sources": grouped["source"],
                "feature_views": grouped["feature_view"],
            }
        )

    def plan(self, repo: FeatureRepo) -> Plan:
        repo = repo.revalidated()
        active = self.active_repo()
        changes: list[PlannedChange] = []
        for kind in _KINDS:
            old_items, new_items = _objects(active, kind), _objects(repo, kind)
            for name in sorted(new_items.keys() - old_items.keys()):
                changes.append(
                    PlannedChange(
                        kind=kind,
                        name=name,
                        action="create",
                        old_version=None,
                        new_version=new_items[name].version_hash(),
                        destructive=False,
                        reasons=("new definition",),
                    )
                )
            for name in sorted(old_items.keys() & new_items.keys()):
                old, new = old_items[name], new_items[name]
                if (
                    old.version_hash() == new.version_hash()
                    and old.metadata_hash() == new.metadata_hash()
                ):
                    continue
                if isinstance(old, FeatureView) and isinstance(new, FeatureView):
                    destructive, reasons = _diff_view(old, new)
                elif isinstance(old, SourceDefinition) and isinstance(new, SourceDefinition):
                    destructive, reasons = _diff_source(old, new)
                elif isinstance(old, EntityDefinition) and isinstance(new, EntityDefinition):
                    destructive, reasons = _diff_entity(old, new)
                else:  # pragma: no cover - kinds are keyed separately
                    raise TypeError("definition kinds do not match")
                changes.append(
                    PlannedChange(
                        kind=kind,
                        name=name,
                        action="update",
                        old_version=old.version_hash(),
                        new_version=new.version_hash(),
                        destructive=destructive,
                        reasons=tuple(reasons),
                    )
                )
            for name in sorted(old_items.keys() - new_items.keys()):
                changes.append(
                    PlannedChange(
                        kind=kind,
                        name=name,
                        action="delete",
                        old_version=old_items[name].version_hash(),
                        new_version=None,
                        destructive=True,
                        reasons=("definition deleted",),
                    )
                )
        return Plan(changes=tuple(changes))

    def apply(
        self,
        repo: FeatureRepo,
        *,
        actor: str,
        approve: bool = False,
        plan_hash: str | None = None,
    ) -> Plan:
        """Apply ``repo``. Destructive plans need ``approve=True`` and the reviewed plan hash."""
        repo = repo.revalidated()
        with self._lock:
            plan = self.plan(repo)
            if plan.is_empty:
                return plan
            now = self._clock()
            summary: dict[str, JsonValue] = {
                "plan_hash": plan.plan_hash,
                "changes": [change.model_dump(mode="json") for change in plan.changes],
            }
            if plan.destructive and not approve:
                self._append_change(
                    now, actor, "apply_rejected", {**summary, "reason": "approval required"}
                )
                raise ApprovalRequiredError(
                    "plan contains destructive changes; review it and apply with approval"
                )
            if plan.destructive and plan_hash != plan.plan_hash:
                self._append_change(
                    now, actor, "apply_rejected", {**summary, "reason": "plan hash mismatch"}
                )
                raise PlanMismatchError("the approved plan hash does not match the current plan")
            new_objects = {kind: _objects(repo, kind) for kind in _KINDS}
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for change in plan.changes:
                    if change.action == "delete":
                        self._conn.execute(
                            "DELETE FROM active WHERE kind = ? AND name = ?",
                            (change.kind, change.name),
                        )
                        continue
                    item = new_objects[change.kind][change.name]
                    definition = item.model_dump_json()
                    self._conn.execute(
                        "INSERT OR IGNORE INTO objects (kind, name, version_hash, definition, "
                        "first_applied_us) VALUES (?, ?, ?, ?, ?)",
                        (change.kind, change.name, item.version_hash(), definition, now),
                    )
                    self._conn.execute(
                        "INSERT OR REPLACE INTO active (kind, name, version_hash, metadata_hash, "
                        "definition) VALUES (?, ?, ?, ?, ?)",
                        (
                            change.kind,
                            change.name,
                            item.version_hash(),
                            item.metadata_hash(),
                            definition,
                        ),
                    )
                self.change_log.append_in_transaction(
                    now, actor, "apply", {**summary, "approved": bool(plan.destructive and approve)}
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            return plan

    def _append_change(
        self, now: int, actor: str, action: str, payload: dict[str, JsonValue]
    ) -> ChainEntry:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            entry = self.change_log.append_in_transaction(now, actor, action, payload)
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        return entry

    def view(self, name: str) -> FeatureView:
        return self.active_repo().view(name)

    def view_version(self, name: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT version_hash FROM active WHERE kind = 'feature_view' AND name = ?", (name,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown feature view: {name}")
        return str(row[0])

    def view_versions(self) -> dict[str, list[str]]:
        """Every version ever applied, per view (erasure must reach keys of old versions too)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, version_hash FROM objects WHERE kind = 'feature_view' "
                "ORDER BY name, first_applied_us, version_hash"
            ).fetchall()
        versions: dict[str, list[str]] = {}
        for name, version in rows:
            versions.setdefault(str(name), []).append(str(version))
        return versions

    # ----------------------------------------------------------------- watermarks
    def get_watermark(self, view: str, version: str) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT watermark_us FROM watermarks WHERE view = ? AND version_hash = ?",
                (view, version),
            ).fetchone()
        return None if row is None else int(row[0])

    def set_watermark(self, view: str, version: str, watermark_us: int) -> None:
        with self._lock:
            current = self.get_watermark(view, version)
            if current is not None and watermark_us < current:
                raise ValueError("watermarks never move backwards")
            self._conn.execute(
                "INSERT OR REPLACE INTO watermarks (view, version_hash, watermark_us, updated_us) "
                "VALUES (?, ?, ?, ?)",
                (view, version, watermark_us, self._clock()),
            )

    def max_watermark_for_source(self, source: str) -> int | None:
        """Highest watermark of any active view over ``source`` (for the append-only check)."""
        repo = self.active_repo()
        marks = [
            self.get_watermark(view.name, view.version_hash())
            for view in repo.feature_views
            if view.source == source
        ]
        present = [mark for mark in marks if mark is not None]
        return max(present) if present else None

    # ----------------------------------------------------------------- erasure
    def add_tombstone(self, entity_hmac: str, receipt_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO tombstones (entity_hmac, erased_us, receipt_id) "
                "VALUES (?, ?, ?)",
                (entity_hmac, self._clock(), receipt_id),
            )

    def tombstones(self) -> frozenset[str]:
        with self._lock:
            rows = self._conn.execute("SELECT entity_hmac FROM tombstones").fetchall()
        return frozenset(str(row[0]) for row in rows)

    # ----------------------------------------------------------------- audit
    def record_access(self, actor: str, action: str, payload: dict[str, JsonValue]) -> ChainEntry:
        with self._lock:
            now = self._clock()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                entry = self.access_audit.append_in_transaction(now, actor, action, payload)
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            return entry

    def record_change(self, actor: str, action: str, payload: dict[str, JsonValue]) -> ChainEntry:
        with self._lock:
            return self._append_change(self._clock(), actor, action, payload)

    def verify_chains(self) -> dict[str, ChainVerification]:
        with self._lock:
            return {
                "change_log": self.change_log.verify(),
                "access_audit": self.access_audit.verify(),
            }

    # ----------------------------------------------------------------- lineage
    def record_training_run(
        self, run_id: str, model_name: str, view_versions: dict[str, str], dataset_digest: str
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO training_runs (run_id, model_name, created_us, "
                "view_versions, "
                "dataset_digest) VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    model_name,
                    self._clock(),
                    canonical_json(dict(view_versions)),
                    dataset_digest,
                ),
            )

    def training_runs(self) -> list[TrainingRun]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, model_name, created_us, view_versions, dataset_digest "
                "FROM training_runs ORDER BY created_us, run_id"
            ).fetchall()
        return [
            TrainingRun(str(r[0]), str(r[1]), int(r[2]), json.loads(r[3]), str(r[4])) for r in rows
        ]
