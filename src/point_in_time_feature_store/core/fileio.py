"""Crash-safe file helpers and a cross-process advisory file lock."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

__all__ = ["atomic_write_bytes", "atomic_write_text", "file_lock"]

if sys.platform == "win32":
    import msvcrt

    def _try_lock(fd: int) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so readers see either the old or the new file, never a mix.

    The bytes go to a temporary file in the same directory, are flushed and fsynced, and the
    temporary file then replaces the target with ``os.replace`` (atomic on POSIX and NTFS).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """UTF-8 text variant of :func:`atomic_write_bytes` (newlines are written as ``\\n``)."""
    atomic_write_bytes(path, text.encode("utf-8"))


@contextlib.contextmanager
def file_lock(
    path: Path, *, timeout_seconds: float = 30.0, poll_seconds: float = 0.01
) -> Iterator[None]:
    """Hold an exclusive advisory lock for ``path`` across threads, instances and processes.

    The lock is taken on a sidecar ``<name>.lock`` file, so ``path`` itself can still be replaced
    atomically. It is advisory: every writer of ``path`` must take it, and readers that need a
    consistent view take it too. Raises ``TimeoutError`` when the lock is not acquired in time.
    """
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + timeout_seconds
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for the lock on {path.name}")
            time.sleep(poll_seconds)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)
