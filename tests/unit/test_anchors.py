"""Signed chain anchors: truncation and whole-chain rewrites that the bare hash chain accepts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from point_in_time_feature_store.anchors import (
    ChainAnchor,
    create_anchor,
    sign_anchor,
    verify_anchored,
)
from point_in_time_feature_store.hashchain import GENESIS, entry_hash
from point_in_time_feature_store.offline.synthetic import churn_feature_repo
from point_in_time_feature_store.registry import FeatureRegistry
from point_in_time_feature_store.serving.auth import generate_keypair_pem, load_private_key

pytestmark = pytest.mark.unit

KEY = load_private_key(generate_keypair_pem()[0])


class Ticker:
    def __init__(self) -> None:
        self.now = 1_000

    def __call__(self) -> int:
        self.now += 1
        return self.now


def make_registry(path: Path) -> FeatureRegistry:
    registry = FeatureRegistry(path, clock=Ticker())
    registry.apply(churn_feature_repo(), actor="alice")
    for index in range(4):
        registry.record_access("svc", "online_read", {"entity_count": index})
    return registry


def rechain_access_audit(path: Path) -> None:
    """What anyone with write access to the SQLite file can do: renumber and rehash the chain."""
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT ts_us, actor, action, payload FROM access_audit ORDER BY seq"
        ).fetchall()
        connection.execute("DELETE FROM access_audit")
        prev = GENESIS
        for seq, (ts_us, actor, action, payload) in enumerate(rows, start=1):
            digest = entry_hash(seq, ts_us, actor, action, json.loads(payload), prev)
            connection.execute(
                "INSERT INTO access_audit (seq, ts_us, actor, action, payload, prev_hash, "
                "entry_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (seq, ts_us, actor, action, payload, prev, digest),
            )
            prev = digest


def test_truncation_passes_the_bare_chain_check_but_not_the_signed_anchor(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    registry = make_registry(path)
    anchor = create_anchor(registry, KEY)
    assert anchor.chains["access_audit"].entries == 4
    registry.record_access("svc", "online_read", {"entity_count": 4})  # growth is expected
    assert verify_anchored(registry, anchor, KEY.public_key()).ok
    registry.close()

    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM access_audit WHERE seq IN (3, 5)")
    rechain_access_audit(path)
    truncated = FeatureRegistry(path, clock=Ticker())
    try:
        assert truncated.verify_chains()["access_audit"].ok  # all the bare chain can tell
        result = verify_anchored(truncated, anchor, KEY.public_key())
        assert result.signature_valid
        assert not result.ok
        assert not result.chains["access_audit"].ok
        assert result.chains["access_audit"].first_bad_seq == 4
        assert result.chains["change_log"].ok
    finally:
        truncated.close()


def test_a_same_length_rewrite_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    registry = make_registry(path)
    anchor = create_anchor(registry, KEY)
    registry.close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE access_audit SET payload = '{\"entity_count\":99}' WHERE seq = 2"
        )
    rechain_access_audit(path)
    rewritten = FeatureRegistry(path, clock=Ticker())
    try:
        assert rewritten.verify_chains()["access_audit"].ok
        result = verify_anchored(rewritten, anchor, KEY.public_key())
        assert not result.ok
        assert result.chains["access_audit"].first_bad_seq == 4
    finally:
        rewritten.close()


def test_forged_or_foreign_anchors_are_rejected(tmp_path: Path) -> None:
    registry = make_registry(tmp_path / "registry.sqlite")
    empty = FeatureRegistry(tmp_path / "empty.sqlite", clock=Ticker())
    try:
        anchor = create_anchor(registry, KEY)
        second = registry.access_audit.entries()[1]
        forged_head = anchor.chains["access_audit"].model_copy(
            update={"entries": 2, "head": second.entry_hash}
        )
        forged = anchor.model_copy(
            update={"chains": {**anchor.chains, "access_audit": forged_head}}
        )
        result = verify_anchored(registry, forged, KEY.public_key())
        assert not result.signature_valid
        assert not result.ok
        other = load_private_key(generate_keypair_pem()[0])
        assert not verify_anchored(registry, anchor, other.public_key()).ok
        garbled = anchor.model_copy(update={"signature": "!!not-base64!!"})
        assert not verify_anchored(registry, garbled, KEY.public_key()).ok
        roundtrip = ChainAnchor.model_validate_json(anchor.model_dump_json())
        assert verify_anchored(registry, roundtrip, KEY.public_key()).ok

        blank = create_anchor(empty, KEY)
        assert blank.chains["change_log"].entries == 0
        assert blank.chains["change_log"].head == GENESIS
        assert verify_anchored(empty, blank, KEY.public_key()).ok
        # Validly signed, but silent about one chain: an anchor must cover every chain.
        partial = sign_anchor(
            anchor.model_copy(update={"chains": {"change_log": anchor.chains["change_log"]}}), KEY
        )
        partial_result = verify_anchored(registry, partial, KEY.public_key())
        assert partial_result.signature_valid
        assert not partial_result.ok
        assert not partial_result.chains["access_audit"].ok
    finally:
        registry.close()
        empty.close()
