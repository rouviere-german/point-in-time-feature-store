from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from point_in_time_feature_store.ml.manifest import (
    RunManifest,
    create_manifest,
    fingerprint_dataset,
    package_versions,
    seed_everything,
    sha256_file,
)

pytestmark = pytest.mark.unit

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_file_hashing_and_fingerprints(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_bytes(b"abc")
    assert sha256_file(path, chunk_size=1) == ABC_SHA256
    fingerprint = fingerprint_dataset(path)
    assert (fingerprint.name, fingerprint.sha256, fingerprint.size_bytes) == (
        "data.csv",
        ABC_SHA256,
        3,
    )


def test_manifest_records_environment_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_COMMIT", "abc1234")
    data = tmp_path / "train.csv"
    data.write_bytes(b"abc")
    manifest = create_manifest(
        seed=7,
        datasets=[data],
        params={"lr": 0.1},
        metrics={"auc": 0.9},
        packages=["numpy", "definitely-not-installed-pkg"],
        run_id="run-1",
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert manifest.git_commit == "abc1234"
    assert manifest.packages["numpy"] == np.__version__
    assert manifest.packages["definitely-not-installed-pkg"] == "not-installed"
    assert manifest.datasets[0].sha256 == ABC_SHA256
    path = tmp_path / "manifest.json"
    manifest.write(path)
    loaded = RunManifest.load(path)
    assert loaded == manifest
    assert loaded.sha256() == manifest.sha256()


def test_non_finite_metrics_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not finite"):
        create_manifest(seed=1, datasets=[], metrics={"loss": float("nan")})


def test_seeding_makes_python_and_numpy_reproducible() -> None:
    first = seed_everything(3)
    sample = (random.random(), float(np.random.random()), float(first.random()))
    second = seed_everything(3)
    assert sample == (random.random(), float(np.random.random()), float(second.random()))


def test_package_versions_are_sorted_and_deduplicated() -> None:
    assert list(package_versions(["pydantic", "numpy", "numpy"])) == ["numpy", "pydantic"]
