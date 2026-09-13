"""Run manifests: seed, dataset SHA-256 fingerprints, package versions, params and metrics."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import random
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from point_in_time_feature_store.core.canonical import canonical_json, sha256_hex
from point_in_time_feature_store.core.fileio import atomic_write_text

__all__ = [
    "DEFAULT_PACKAGES",
    "DatasetFingerprint",
    "RunManifest",
    "create_manifest",
    "fingerprint_dataset",
    "package_versions",
    "seed_everything",
    "sha256_file",
]

DEFAULT_PACKAGES = ("numpy", "pandas", "pandera", "pydantic")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class DatasetFingerprint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


def fingerprint_dataset(path: Path, name: str | None = None) -> DatasetFingerprint:
    return DatasetFingerprint(
        name=name or path.name, sha256=sha256_file(path), size_bytes=path.stat().st_size
    )


def package_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in sorted(set(names)):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python's and NumPy's global RNGs and return a fresh NumPy Generator.

    PYTHONHASHSEED cannot be changed after start-up; it is recorded in the manifest instead.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))  # legacy global RNG, for libraries that still use it
    return np.random.default_rng(seed)


class RunManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    created_at: str
    seed: int
    python: str
    platform: str
    pythonhashseed: str | None
    git_commit: str | None
    packages: dict[str, str]
    datasets: tuple[DatasetFingerprint, ...]
    params: dict[str, JsonValue]
    metrics: dict[str, float]

    @field_validator("metrics")
    @classmethod
    def _finite_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        for key, metric in value.items():
            if not math.isfinite(metric):
                raise ValueError(f"metric {key} is not finite")
        return value

    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))

    def write(self, path: Path) -> None:
        atomic_write_text(path, self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def create_manifest(
    *,
    seed: int,
    datasets: Iterable[Path],
    params: Mapping[str, JsonValue] | None = None,
    metrics: Mapping[str, float] | None = None,
    packages: Iterable[str] = DEFAULT_PACKAGES,
    run_id: str | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> RunManifest:
    return RunManifest(
        run_id=run_id or uuid.uuid4().hex,
        created_at=clock().isoformat(),
        seed=seed,
        python=sys.version.split()[0],
        platform=platform.platform(),
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        git_commit=os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA"),
        packages=package_versions(packages),
        datasets=tuple(fingerprint_dataset(path) for path in datasets),
        params=dict(params or {}),
        metrics=dict(metrics or {}),
    )
