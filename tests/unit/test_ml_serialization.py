from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from point_in_time_feature_store.ml.serialization import (
    UnsafeArtifactError,
    dumps_npz,
    load_json,
    load_npz,
    loads_npz,
    save_json,
    save_npz,
)

pytestmark = pytest.mark.unit


def test_json_round_trip_and_nan_rejection(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    save_json(path, {"b": [1, 2], "a": "é"})
    assert load_json(path) == {"a": "é", "b": [1, 2]}
    with pytest.raises(ValueError, match="Out of range float"):
        save_json(path, {"loss": float("nan")})


def test_npz_round_trip_for_numeric_and_string_arrays(tmp_path: Path) -> None:
    path = tmp_path / "weights.npz"
    arrays = {"w": np.arange(6, dtype=np.float32).reshape(2, 3), "labels": np.array(["a", "bc"])}
    save_npz(path, arrays)
    loaded = load_npz(path)
    assert set(loaded) == {"w", "labels"}
    np.testing.assert_array_equal(loaded["w"], arrays["w"])
    np.testing.assert_array_equal(loaded["labels"], arrays["labels"])


def test_object_arrays_are_refused_on_save() -> None:
    with pytest.raises(UnsafeArtifactError, match="object dtype"):
        dumps_npz({"bad": np.array([{"x": 1}], dtype=object)})


def test_pickled_npz_files_are_refused_on_load() -> None:
    buffer = io.BytesIO()
    np.savez(buffer, payload=np.array([{"x": 1}], dtype=object))
    with pytest.raises(UnsafeArtifactError, match="pickled"):
        loads_npz(buffer.getvalue())


@pytest.mark.parametrize("name", ["1abc", "has-dash", "", "a" * 65])
def test_invalid_array_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="invalid array name"):
        dumps_npz({name: np.zeros(1)})


def test_suffix_and_empty_input_checks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one array"):
        dumps_npz({})
    with pytest.raises(ValueError, match=r"\.npz suffix"):
        save_npz(tmp_path / "weights.npy", {"w": np.zeros(1)})
    with pytest.raises(ValueError, match=r"\.npz suffix"):
        load_npz(tmp_path / "weights.pkl")
