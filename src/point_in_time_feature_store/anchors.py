"""Signed anchors for the registry's hash chains (change log and access audit).

The hash chain alone detects an edited, reordered or deleted *middle* entry. It cannot detect a
truncated tail, or a chain rebuilt from scratch by someone with write access to the SQLite file:
an unkeyed SHA-256 chain can simply be recomputed. An anchor records, for every chain, the entry
count and the head hash at one point in time, and is signed with an Ed25519 key held outside the
database (the receipt signing key). Stored away from the registry host (a ticket, an object store
with retention, a separate repository), it lets :func:`verify_anchored` detect:

- truncation below the anchored length;
- any rewrite of the anchored prefix: every entry hash covers all earlier entries, so the entry at
  the anchored length no longer hashes to the anchored head.

Entries appended after an anchor are covered only by the next anchor; anchoring regularly (for
example after every apply or erasure) bounds what an undetected rewrite can remove.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field

from point_in_time_feature_store.core.canonical import canonical_json
from point_in_time_feature_store.erasure import public_key_id
from point_in_time_feature_store.hashchain import GENESIS, ChainVerification
from point_in_time_feature_store.registry import FeatureRegistry
from point_in_time_feature_store.timeutil import Clock, system_clock_us

__all__ = [
    "AnchoredVerification",
    "ChainAnchor",
    "ChainHead",
    "create_anchor",
    "sign_anchor",
    "verify_anchor_signature",
    "verify_anchored",
]


class ChainHead(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: int = Field(ge=0)
    head: str = Field(pattern=r"^[0-9a-f]{64}$")


class ChainAnchor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    anchor_version: int = 1
    created_us: int
    chains: dict[str, ChainHead]
    key_id: str = ""
    signature: str = ""

    def signing_payload(self) -> bytes:
        return canonical_json(self.model_dump(mode="json", exclude={"signature"})).encode("utf-8")


@dataclass(frozen=True, slots=True)
class AnchoredVerification:
    signature_valid: bool
    chains: dict[str, ChainVerification]

    @property
    def ok(self) -> bool:
        return self.signature_valid and all(item.ok for item in self.chains.values())


def sign_anchor(anchor: ChainAnchor, private_key: Ed25519PrivateKey) -> ChainAnchor:
    unsigned = anchor.model_copy(
        update={"key_id": public_key_id(private_key.public_key()), "signature": ""}
    )
    signature = private_key.sign(unsigned.signing_payload())
    return unsigned.model_copy(
        update={"signature": base64.urlsafe_b64encode(signature).decode("ascii")}
    )


def verify_anchor_signature(anchor: ChainAnchor, public_key: Ed25519PublicKey) -> bool:
    if anchor.key_id != public_key_id(public_key) or not anchor.signature:
        return False
    try:
        public_key.verify(base64.urlsafe_b64decode(anchor.signature), anchor.signing_payload())
    except (InvalidSignature, ValueError):
        return False
    return True


def create_anchor(
    registry: FeatureRegistry, private_key: Ed25519PrivateKey, *, clock: Clock = system_clock_us
) -> ChainAnchor:
    """Anchor the current length and head of every chain; a broken chain is never anchored."""
    heads: dict[str, ChainHead] = {}
    for name, result in registry.verify_chains().items():
        if not result.ok:
            raise ValueError(f"refusing to anchor a chain that fails verification: {name}")
        heads[name] = ChainHead(entries=result.entries, head=result.head)
    return sign_anchor(ChainAnchor(created_us=clock(), chains=heads), private_key)


def _against(bare: ChainVerification, anchored: ChainHead, hashes: list[str]) -> ChainVerification:
    if not bare.ok:
        return bare
    if len(hashes) < anchored.entries:  # the tail was truncated below the anchored length
        return ChainVerification(False, len(hashes), bare.head, first_bad_seq=len(hashes) + 1)
    expected = GENESIS if anchored.entries == 0 else hashes[anchored.entries - 1]
    if expected != anchored.head:  # the anchored prefix was rewritten and re-chained
        return ChainVerification(False, len(hashes), bare.head, first_bad_seq=anchored.entries)
    return bare


def verify_anchored(
    registry: FeatureRegistry, anchor: ChainAnchor, public_key: Ed25519PublicKey
) -> AnchoredVerification:
    """Verify every chain on its own and against a signed anchor it must extend."""
    bare = registry.verify_chains()
    if not verify_anchor_signature(anchor, public_key):
        # The anchored heads are untrusted; report the bare results, which cannot make ``ok``.
        return AnchoredVerification(signature_valid=False, chains=bare)
    chains = {"change_log": registry.change_log, "access_audit": registry.access_audit}
    results: dict[str, ChainVerification] = {}
    for name, result in bare.items():
        anchored = anchor.chains.get(name)
        if anchored is None:  # a signed anchor that is silent about a chain proves nothing
            results[name] = ChainVerification(False, result.entries, result.head)
            continue
        hashes = [entry.entry_hash for entry in chains[name].entries()]
        results[name] = _against(result, anchored, hashes)
    for name in anchor.chains.keys() - bare.keys():
        results[name] = ChainVerification(False, 0, GENESIS)
    return AnchoredVerification(signature_valid=True, chains=results)
