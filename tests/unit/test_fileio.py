from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from point_in_time_feature_store.core.fileio import atomic_write_text, file_lock

pytestmark = pytest.mark.unit


def _acquire(manager: contextlib.AbstractContextManager[None]) -> None:
    with manager:
        pass


def test_file_lock_is_exclusive_times_out_and_is_released(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    with file_lock(target), pytest.raises(TimeoutError, match=r"state\.json"):
        _acquire(file_lock(target, timeout_seconds=0.05))
    _acquire(file_lock(target, timeout_seconds=0.05))


def test_the_locked_file_itself_can_still_be_replaced_atomically(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    with file_lock(target):
        atomic_write_text(target, "one\n")
        atomic_write_text(target, "two\n")
    assert target.read_text(encoding="utf-8") == "two\n"
