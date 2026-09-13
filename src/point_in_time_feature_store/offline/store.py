"""Partitioned Parquet offline store.

Layout under ``root``::

    store.json                                   {"format": 1, "buckets": N}
    sources/<source>/bucket=<NN>/part-<seq>-<rand>.parquet

Rows are partitioned by a stable hash of ``entity_id`` (SHA-256, first 4 bytes, modulo N), so all
rows of one entity live in one bucket directory. Appends add new part files and never rewrite
existing ones; erasure rewrites only the part files of that entity's bucket that contain it.
Both take the same store lock, so an append cannot land between a tombstone and the rewrite.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from point_in_time_feature_store.core.fileio import atomic_write_text, file_lock
from point_in_time_feature_store.definitions import EntityDefinition, SourceDefinition
from point_in_time_feature_store.dsl import POLARS_DTYPES
from point_in_time_feature_store.hashing import EntityHasher
from point_in_time_feature_store.ml.contracts import TIMESTAMP_DTYPE, validate_source_frame
from point_in_time_feature_store.ml.manifest import sha256_file

__all__ = ["AppendReport", "OfflineStore", "PartitionRewrite", "source_schema"]

_IDENT = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_PART = re.compile(r"^part-(\d{20})-[0-9a-f]{16}\.parquet$")


@dataclass(frozen=True, slots=True)
class PartitionRewrite:
    source: str
    partition: str
    rows_removed: int
    before_sha256: str
    after_sha256: str | None


@dataclass(frozen=True, slots=True)
class AppendReport:
    source: str
    rows_written: int
    rows_dropped_tombstoned: int
    files_written: int


def source_schema(source: SourceDefinition) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "entity_id": pl.String(),
        "event_time": TIMESTAMP_DTYPE,
        "ingested_at": TIMESTAMP_DTYPE,
    }
    for column in source.columns:
        schema[column.name] = POLARS_DTYPES[column.dtype]
    return schema


class OfflineStore:
    def __init__(self, root: Path, *, buckets: int = 16) -> None:
        if not 1 <= buckets <= 1024:
            raise ValueError("buckets must be between 1 and 1024")
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        meta = root / "store.json"
        if meta.exists():
            stored = json.loads(meta.read_text(encoding="utf-8"))
            if int(stored["buckets"]) != buckets:
                raise ValueError("bucket count differs from the existing store")
        else:
            atomic_write_text(meta, json.dumps({"format": 1, "buckets": buckets}) + "\n")
        self.buckets = buckets
        self._lock_path = root / "store"

    def bucket_of(self, entity_id: str) -> int:
        digest = hashlib.sha256(entity_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.buckets

    def _source_dir(self, source_name: str) -> Path:
        if not _IDENT.fullmatch(source_name):
            raise ValueError("invalid source name")
        return self.root / "sources" / source_name

    def _bucket_dir(self, source_name: str, bucket: int) -> Path:
        return self._source_dir(source_name) / f"bucket={bucket:02d}"

    def files(self, source_name: str) -> list[Path]:
        base = self._source_dir(source_name)
        if not base.exists():
            return []
        return sorted(
            path
            for path in base.glob("bucket=*/part-*.parquet")
            if _PART.fullmatch(path.name) is not None
        )

    def append(
        self,
        source: SourceDefinition,
        entity: EntityDefinition,
        frame: pl.DataFrame,
        *,
        hasher: EntityHasher | None = None,
        tombstones: frozenset[str] = frozenset(),
        min_ingested_at_us: int | None = None,
    ) -> AppendReport:
        """Validate ``frame`` against the source contract, drop erased entities and write parts."""
        validated = validate_source_frame(
            source, entity, frame, min_ingested_at_us=min_ingested_at_us
        )
        with file_lock(self._lock_path):
            dropped = 0
            if tombstones:
                if hasher is None:
                    raise ValueError("a hasher is required to honour tombstones")
                erased = [
                    entity_id
                    for entity_id in validated["entity_id"].unique().to_list()
                    if hasher.entity(entity_id) in tombstones
                ]
                if erased:
                    before = validated.height
                    validated = validated.filter(~pl.col("entity_id").is_in(erased))
                    dropped = before - validated.height
            if validated.height == 0:
                return AppendReport(source.name, 0, dropped, 0)
            ids = validated["entity_id"].unique()
            buckets = pl.DataFrame(
                {"entity_id": ids, "__bucket": [self.bucket_of(item) for item in ids.to_list()]},
                schema={"entity_id": pl.String, "__bucket": pl.Int64},
            )
            keyed = validated.join(buckets, on="entity_id", how="left")
            files = 0
            for (bucket,), part in keyed.group_by(["__bucket"], maintain_order=True):
                rows = part.drop("__bucket").sort(["entity_id", "event_time", "ingested_at"])
                self._write_part(source.name, int(str(bucket)), rows)
                files += 1
            return AppendReport(source.name, validated.height, dropped, files)

    def _write_part(self, source_name: str, bucket: int, rows: pl.DataFrame) -> Path:
        directory = self._bucket_dir(source_name, bucket)
        directory.mkdir(parents=True, exist_ok=True)
        existing = [
            int(match.group(1)) for p in directory.iterdir() if (match := _PART.fullmatch(p.name))
        ]
        seq = max(existing, default=0) + 1
        path = directory / f"part-{seq:020d}-{uuid.uuid4().hex[:16]}.parquet"
        self._atomic_parquet(rows, path)
        return path

    @staticmethod
    def _atomic_parquet(rows: pl.DataFrame, path: Path) -> None:
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            rows.write_parquet(tmp, compression="zstd", statistics=True)
            with tmp.open("rb+") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
            raise

    def scan(self, source: SourceDefinition) -> pl.LazyFrame:
        files = self.files(source.name)
        if not files:
            return pl.DataFrame(schema=source_schema(source)).lazy()
        return pl.scan_parquet([str(path) for path in files])

    def read(self, source: SourceDefinition) -> pl.DataFrame:
        return self.scan(source).collect()

    def fingerprints(self, source_names: Iterable[str]) -> dict[str, str]:
        """``relative path -> sha256`` for every part file of the given sources."""
        result: dict[str, str] = {}
        for name in sorted(set(source_names)):
            for path in self.files(name):
                result[path.relative_to(self.root).as_posix()] = sha256_file(path)
        return result

    def dataset_digest(self, source_names: Iterable[str]) -> str:
        digest = hashlib.sha256()
        for relative, file_hash in sorted(self.fingerprints(source_names).items()):
            digest.update(f"{relative}:{file_hash}\n".encode())
        return digest.hexdigest()

    def entity_row_count(self, source_name: str, entity_id: str) -> int:
        files = self.files(source_name)
        if not files:
            return 0
        frame = pl.scan_parquet([str(path) for path in files])
        return int(frame.filter(pl.col("entity_id") == entity_id).select(pl.len()).collect().item())

    def erase_entity(self, source_name: str, entity_id: str) -> list[PartitionRewrite]:
        """Rewrite the entity's bucket files that contain it; return before/after hashes."""
        rewrites: list[PartitionRewrite] = []
        with file_lock(self._lock_path):
            directory = self._bucket_dir(source_name, self.bucket_of(entity_id))
            if not directory.exists():
                return rewrites
            for path in sorted(directory.glob("part-*.parquet")):
                if _PART.fullmatch(path.name) is None:
                    continue
                frame = pl.read_parquet(path)
                keep = frame.filter(pl.col("entity_id") != entity_id)
                removed = frame.height - keep.height
                if removed == 0:
                    continue
                before = sha256_file(path)
                after: str | None
                if keep.height == 0:
                    path.unlink()
                    after = None
                else:
                    self._atomic_parquet(keep, path)
                    after = sha256_file(path)
                rewrites.append(
                    PartitionRewrite(
                        source=source_name,
                        partition=path.relative_to(self.root).as_posix(),
                        rows_removed=removed,
                        before_sha256=before,
                        after_sha256=after,
                    )
                )
        return rewrites
