"""Repository guard: no pickle-family serialization anywhere in shipped code."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_BANNED_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(?:pickle|cPickle|_pickle|dill|joblib|cloudpickle|shelve)\b"
    r"|\b(?:read|to)_pickle\s*\(",
    re.MULTILINE,
)
_ALLOW_PICKLE = re.compile(r"allow_pickle\s*=\s*True")


def test_source_tree_has_no_pickle_family_serialization(repo_root: Path) -> None:
    offenders = [
        str(path.relative_to(repo_root))
        for path in sorted((repo_root / "src").rglob("*.py"))
        if _BANNED_IMPORT.search(text := path.read_text(encoding="utf-8"))
        or _ALLOW_PICKLE.search(text)
    ]
    assert offenders == []
