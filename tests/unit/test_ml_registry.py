from __future__ import annotations

import json
from pathlib import Path

import pytest

from point_in_time_feature_store.ml.manifest import RunManifest, create_manifest
from point_in_time_feature_store.ml.registry import (
    ModelRegistry,
    RegistryError,
    RegistryIntegrityError,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def manifest() -> RunManifest:
    return create_manifest(seed=1, datasets=[], run_id="run", packages=[])


def test_register_resolve_and_load(tmp_path: Path, manifest: RunManifest) -> None:
    registry = ModelRegistry(tmp_path)
    files = {"weights.npz": b"\x00\x01", "config.json": b"{}"}
    version = registry.register("demo-model", files, manifest, metrics={"auc": 0.8})
    assert registry.register("demo-model", files, manifest, metrics={"auc": 0.8}) == version
    assert registry.versions("demo-model") == [version.digest]
    registry.set_alias("demo-model", "production", version.digest)
    assert registry.get("demo-model", "production") == version
    assert registry.load_files(registry.get("demo-model", version.digest)) == files

    other = registry.register("demo-model", files, manifest, metrics={"auc": 0.81})
    assert other.digest != version.digest
    assert registry.versions("missing-model") == []


def test_tampered_blob_is_detected(tmp_path: Path, manifest: RunManifest) -> None:
    registry = ModelRegistry(tmp_path)
    version = registry.register("m", {"w.bin": b"original"}, manifest)
    blob = version.files["w.bin"]
    (tmp_path / "blobs" / "sha256" / blob[:2] / blob).write_bytes(b"tampered")
    with pytest.raises(RegistryIntegrityError, match="content hash"):
        registry.load_files(version)
    (tmp_path / "blobs" / "sha256" / blob[:2] / blob).unlink()
    with pytest.raises(RegistryIntegrityError, match="missing blob"):
        registry.load_files(version)


def test_tampered_version_record_is_detected(tmp_path: Path, manifest: RunManifest) -> None:
    registry = ModelRegistry(tmp_path)
    version = registry.register("m", {"w.bin": b"data"}, manifest, metrics={"auc": 0.5})
    record_path = tmp_path / "models" / "m" / "versions" / f"{version.digest}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["metrics"]["auc"] = 0.99
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RegistryIntegrityError, match="modified"):
        registry.get("m", version.digest)


@pytest.mark.parametrize(
    ("name", "file_name"),
    [("../evil", "w.bin"), ("Upper", "w.bin"), ("m", "../w.bin"), ("m", "a..b"), ("m", "")],
)
def test_path_traversal_and_invalid_names_are_rejected(
    tmp_path: Path, manifest: RunManifest, name: str, file_name: str
) -> None:
    with pytest.raises(RegistryError):
        ModelRegistry(tmp_path).register(name, {file_name: b"x"}, manifest)


def test_unknown_references_and_empty_versions(tmp_path: Path, manifest: RunManifest) -> None:
    registry = ModelRegistry(tmp_path)
    with pytest.raises(RegistryError, match="at least one file"):
        registry.register("m", {}, manifest)
    with pytest.raises(RegistryError, match="unknown alias"):
        registry.get("m", "production")
    with pytest.raises(RegistryError, match="unknown version"):
        registry.get("m", "0" * 64)
    with pytest.raises(RegistryError, match="unknown version"):
        registry.set_alias("m", "production", "0" * 64)
