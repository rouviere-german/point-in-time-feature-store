"""Content-addressed model registry on the local filesystem.

Layout under ``root``::

    blobs/sha256/<2 hex>/<digest>          artifact bytes, named by their SHA-256
    models/<name>/versions/<digest>.json   version record (file -> blob digest, manifest, metrics)
    models/<name>/aliases.json             alias -> version digest (for example "production")

A version digest is the SHA-256 of the canonical record, so identical content always yields the
same version and any edit to a record or blob is detected on load.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from point_in_time_feature_store.core.canonical import canonical_json, sha256_hex
from point_in_time_feature_store.core.fileio import atomic_write_bytes, atomic_write_text
from point_in_time_feature_store.ml.manifest import RunManifest

__all__ = ["ModelRegistry", "ModelVersion", "RegistryError", "RegistryIntegrityError"]

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class RegistryError(ValueError):
    """Invalid registry request (bad name, unknown version or alias)."""


class RegistryIntegrityError(RegistryError):
    """Stored bytes or records do not match their content address."""


class ModelVersion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: dict[str, str]
    manifest: RunManifest
    metrics: dict[str, float]


def _check_name(kind: str, value: str, pattern: re.Pattern[str]) -> str:
    if not pattern.fullmatch(value) or ".." in value:
        raise RegistryError(f"invalid {kind}: {value!r}")
    return value


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _blob_path(self, digest: str) -> Path:
        _check_name("digest", digest, _DIGEST)
        return self.root / "blobs" / "sha256" / digest[:2] / digest

    def _model_dir(self, name: str) -> Path:
        return self.root / "models" / _check_name("model name", name, _NAME)

    def put_blob(self, data: bytes) -> str:
        digest = sha256_hex(data)
        path = self._blob_path(digest)
        if not path.exists():
            atomic_write_bytes(path, data)
        return digest

    def read_blob(self, digest: str) -> bytes:
        path = self._blob_path(digest)
        if not path.exists():
            raise RegistryIntegrityError(f"missing blob {digest}")
        data = path.read_bytes()
        if sha256_hex(data) != digest:
            raise RegistryIntegrityError(f"blob {digest} does not match its content hash")
        return data

    def register(
        self,
        name: str,
        files: Mapping[str, bytes],
        manifest: RunManifest,
        metrics: Mapping[str, float] | None = None,
    ) -> ModelVersion:
        model_dir = self._model_dir(name)
        if not files:
            raise RegistryError("a model version needs at least one file")
        blob_refs = {
            _check_name("file name", file_name, _FILE): self.put_blob(data)
            for file_name, data in sorted(files.items())
        }
        record: dict[str, JsonValue] = {
            "name": name,
            "files": dict(blob_refs),
            "manifest": manifest.model_dump(mode="json"),
            "metrics": dict(metrics or {}),
        }
        digest = sha256_hex(canonical_json(record))
        version = ModelVersion.model_validate({**record, "digest": digest})
        path = model_dir / "versions" / f"{digest}.json"
        if not path.exists():
            atomic_write_text(path, version.model_dump_json(indent=2) + "\n")
        return version

    def versions(self, name: str) -> list[str]:
        directory = self._model_dir(name) / "versions"
        return sorted(p.stem for p in directory.glob("*.json")) if directory.exists() else []

    def _aliases_path(self, name: str) -> Path:
        return self._model_dir(name) / "aliases.json"

    def _aliases(self, name: str) -> dict[str, str]:
        path = self._aliases_path(name)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()}

    def set_alias(self, name: str, alias: str, digest: str) -> None:
        _check_name("alias", alias, _NAME)
        self.get(name, digest)  # must exist and verify
        aliases = self._aliases(name)
        aliases[alias] = digest
        atomic_write_text(
            self._aliases_path(name), json.dumps(aliases, indent=2, sort_keys=True) + "\n"
        )

    def get(self, name: str, ref: str) -> ModelVersion:
        """Load a version by digest or alias and verify its record hash."""
        digest = ref if _DIGEST.fullmatch(ref) else self._aliases(name).get(ref)
        if digest is None:
            raise RegistryError(f"unknown alias {ref!r} for model {name}")
        path = self._model_dir(name) / "versions" / f"{digest}.json"
        if not path.exists():
            raise RegistryError(f"unknown version {digest} for model {name}")
        version = ModelVersion.model_validate_json(path.read_text(encoding="utf-8"))
        record = version.model_dump(mode="json")
        record.pop("digest")
        if version.digest != digest or sha256_hex(canonical_json(record)) != digest:
            raise RegistryIntegrityError(f"version record {digest} was modified")
        return version

    def load_files(self, version: ModelVersion) -> dict[str, bytes]:
        return {file_name: self.read_blob(blob) for file_name, blob in version.files.items()}
