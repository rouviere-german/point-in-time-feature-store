from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest

from point_in_time_feature_store.definitions import FeatureDefinition
from point_in_time_feature_store.devtools import LocalStack, build_local_stack
from point_in_time_feature_store.dsl import DType, col
from point_in_time_feature_store.erasure import ReceiptSigner, verify_receipt
from point_in_time_feature_store.offline.synthetic import (
    SyntheticConfig,
    churn_feature_repo,
    generate_dataset,
)
from point_in_time_feature_store.serving.auth import (
    Caller,
    generate_keypair_pem,
    load_private_key,
    load_public_key,
)
from point_in_time_feature_store.timeutil import US_PER_DAY

pytestmark = pytest.mark.integration

DATASET = generate_dataset(SyntheticConfig(n_customers=120, seed=21))
T2 = DATASET.config.day_us(DATASET.config.horizon_days)
READER = Caller("svc-tests", frozenset({"features:read:*", "pii:read"}))
ALL = ["customer_profile:*", "transaction_stats:*", "session_stats:*"]


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[LocalStack]:
    local = build_local_stack(tmp_path, now_us=T2)
    services = local.services
    services.registry.apply(churn_feature_repo(), actor="tests")
    services.online_service.refresh()
    for name, frame in DATASET.sources.items():
        services.ingest(name, frame)
    for view in services.registry.active_repo().feature_views:
        services.materializer.materialize(view.name, up_to_us=T2)
    yield local
    local.close()


def victim_with_every_source() -> str:
    present = [set(frame["entity_id"].to_list()) for frame in DATASET.sources.values()]
    return str(sorted(set.intersection(*present))[0])


def add_version(stack: LocalStack) -> None:
    """Apply a compatible change so the victim has online keys under two view versions."""
    services = stack.services
    repo = services.registry.active_repo()
    extra = FeatureDefinition(
        name="pages_x2", dtype=DType.INT64, expression=(col("pages") * 2).node
    )
    services.registry.apply(
        repo.model_copy(
            update={
                "feature_views": tuple(
                    v.model_copy(update={"features": (*v.features, extra)})
                    if v.name == "session_stats"
                    else v
                    for v in repo.feature_views
                )
            }
        ),
        actor="tests",
    )
    services.online_service.refresh()
    services.materializer.materialize("session_stats", up_to_us=T2)


def test_erasure_removes_every_copy_and_signs_a_receipt(stack: LocalStack) -> None:
    services = stack.services
    victim = victim_with_every_source()
    services.online_service.get(READER, [victim], ALL, request_id=uuid.uuid4().hex, now_us=T2)
    add_version(stack)
    versions = services.registry.view_versions()
    assert services.online.count_entity_keys(victim, versions) == 4
    residue = services.erasure.verify_purged(victim)
    assert not residue.clean

    receipt = services.erasure.erase(victim, requested_by="privacy-desk")
    assert {item.source for item in receipt.offline_rewrites} == {
        "customers",
        "transactions",
        "sessions",
    }
    assert sum(item.rows_removed for item in receipt.offline_rewrites) == sum(
        frame.filter(pl.col("entity_id") == victim).height for frame in DATASET.sources.values()
    )
    assert receipt.online_keys_deleted == 4
    assert receipt.request_log_rows_deleted == 3
    assert receipt.entity_hmac == services.hasher.entity(victim)
    assert victim not in receipt.model_dump_json()
    assert verify_receipt(receipt, load_public_key(stack.receipt_public_pem))
    assert services.erasure.verify_purged(victim).clean
    entry = services.registry.change_log.entries()[-1]
    assert entry.action == "erase_entity"
    assert entry.entry_hash == receipt.change_log_entry_hash
    assert services.registry.verify_chains()["change_log"].ok


def test_receipts_detect_tampering_and_foreign_keys(stack: LocalStack) -> None:
    receipt = stack.services.erasure.erase(victim_with_every_source(), requested_by="privacy-desk")
    public = load_public_key(stack.receipt_public_pem)
    assert verify_receipt(receipt, public)
    assert not verify_receipt(receipt.model_copy(update={"online_keys_deleted": 0}), public)
    assert not verify_receipt(receipt.model_copy(update={"signature": ""}), public)
    assert not verify_receipt(receipt.model_copy(update={"signature": "!!not-base64!!"}), public)
    other = ReceiptSigner(load_private_key(generate_keypair_pem()[0]))
    assert not verify_receipt(receipt, other.public_key)
    resigned_by_other = other.sign(receipt)
    assert verify_receipt(resigned_by_other, other.public_key)
    assert not verify_receipt(resigned_by_other, public)


def test_erased_entities_do_not_come_back(stack: LocalStack) -> None:
    services = stack.services
    victim = victim_with_every_source()
    services.erasure.erase(victim, requested_by="privacy-desk")
    later = T2 + US_PER_DAY
    stack.clock.now_us = later
    replay = (
        DATASET.sources["transactions"]
        .filter(pl.col("entity_id") == victim)
        .filter(pl.col("event_time").dt.epoch("us") > later - 399 * US_PER_DAY)
        .with_columns(
            pl.from_epoch(pl.lit(later), time_unit="us")
            .dt.replace_time_zone("UTC")
            .alias("ingested_at")
        )
    )
    report = services.ingest("transactions", replay)
    assert report.rows_written == 0
    assert report.rows_dropped_tombstoned == replay.height
    services.materializer.materialize("transaction_stats", up_to_us=later)
    assert services.online.count_entity_keys(victim, services.registry.view_versions()) == 0
    spine = pl.DataFrame({"entity_id": [victim], "label_time": [later]})
    frame = services.retriever().get_historical_features(spine, ALL).frame
    assert frame.select(pl.all().exclude("entity_id", "label_time").is_null().all()).row(
        0
    ) == tuple([True] * (frame.width - 2))


def test_purge_verification_reports_log_residue(stack: LocalStack) -> None:
    victim = victim_with_every_source()
    stack.services.erasure.erase(victim, requested_by="privacy-desk")
    report = stack.services.erasure.verify_purged(
        victim, log_texts=[f"user {victim} did something"]
    )
    assert report.log_occurrences == 1
    assert not report.clean
