from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from point_in_time_feature_store.core.canonical import (
    CanonicalJSONError,
    canonical_json,
    sha256_hex,
)
from point_in_time_feature_store.core.fileio import atomic_write_text

pytestmark = pytest.mark.unit

_json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=10),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=20,
)


@given(_json_values)
def test_canonical_encoding_is_a_fixed_point(value: Any) -> None:
    text = canonical_json(value)
    assert canonical_json(json.loads(text)) == text


@given(st.dictionaries(st.text(max_size=5), st.integers(), max_size=6))
def test_key_insertion_order_does_not_change_the_encoding(mapping: dict[str, int]) -> None:
    reordered = dict(reversed(list(mapping.items())))
    assert canonical_json(mapping) == canonical_json(reordered)


def test_encoding_is_compact_sorted_and_utf8() -> None:
    assert canonical_json({"b": 1, "a": [True, None, "é"]}) == '{"a":[true,null,"é"],"b":1}'


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), {1: "x"}, ("tuple",), {"a": {1, 2}}, b"bytes"],
)
def test_values_outside_the_json_model_are_rejected(value: Any) -> None:
    with pytest.raises(CanonicalJSONError):
        canonical_json(value)


def test_sha256_known_vector() -> None:
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_hex("abc") == expected
    assert sha256_hex(b"abc") == expected


def test_atomic_write_replaces_content_and_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "file.json"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second")
    assert target.read_text(encoding="utf-8") == "second"
    assert sorted(p.name for p in target.parent.iterdir()) == ["file.json"]


def test_atomic_write_failure_keeps_old_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "file.txt"
    atomic_write_text(target, "original")

    def failing_replace(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["file.txt"]
