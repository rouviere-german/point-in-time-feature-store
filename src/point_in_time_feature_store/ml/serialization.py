"""Pickle-free artifact I/O: JSON and NumPy ``.npz`` with ``allow_pickle=False``.

pickle, joblib, dill and cloudpickle execute code on load, so a tampered artifact is remote code
execution. They are banned by a semgrep rule and by tests/unit/test_no_pickle.py.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from point_in_time_feature_store.core.fileio import atomic_write_bytes, atomic_write_text

__all__ = [
    "UnsafeArtifactError",
    "dumps_npz",
    "load_json",
    "load_npz",
    "loads_npz",
    "save_json",
    "save_npz",
]

_ARRAY_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class UnsafeArtifactError(ValueError):
    """The artifact would need pickle (object arrays) or is otherwise unsafe to load."""


def save_json(path: Path, obj: Any) -> None:
    atomic_write_text(
        path, json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dumps_npz(arrays: Mapping[str, npt.ArrayLike]) -> bytes:
    """Serialize numeric/string arrays to compressed npz bytes; object arrays are refused."""
    if not arrays:
        raise ValueError("at least one array is required")
    checked: dict[str, npt.NDArray[Any]] = {}
    for name, values in arrays.items():
        if not _ARRAY_NAME.fullmatch(name):
            raise ValueError(f"invalid array name {name!r}")
        array = np.asarray(values)
        if array.dtype.hasobject:
            raise UnsafeArtifactError(f"array {name!r} has object dtype, which requires pickle")
        checked[name] = array
    buffer = io.BytesIO()
    np.savez_compressed(buffer, allow_pickle=False, **checked)
    return buffer.getvalue()


def loads_npz(data: bytes) -> dict[str, npt.NDArray[Any]]:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except ValueError as exc:
        raise UnsafeArtifactError("npz contains pickled object arrays; refusing to load") from exc


def save_npz(path: Path, arrays: Mapping[str, npt.ArrayLike]) -> None:
    if path.suffix != ".npz":
        raise ValueError("npz artifacts must use the .npz suffix")
    atomic_write_bytes(path, dumps_npz(arrays))


def load_npz(path: Path) -> dict[str, npt.NDArray[Any]]:
    if path.suffix != ".npz":
        raise ValueError("npz artifacts must use the .npz suffix")
    return loads_npz(path.read_bytes())
