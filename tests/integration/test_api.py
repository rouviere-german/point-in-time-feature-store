from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fakeredis
import polars as pl
import pytest
from fastapi.testclient import TestClient

from point_in_time_feature_store.core.config import Settings
from point_in_time_feature_store.core.logging import configure_logging
from point_in_time_feature_store.core.security import hash_api_key
from point_in_time_feature_store.devtools import LocalStack, build_local_stack
from point_in_time_feature_store.erasure import ErasureReceipt, verify_receipt
from point_in_time_feature_store.offline.synthetic import (
    SyntheticConfig,
    churn_feature_repo,
    generate_dataset,
)
from point_in_time_feature_store.services import ServiceConfigurationError, build_services
from point_in_time_feature_store.serving.api import create_feature_app
from point_in_time_feature_store.serving.auth import (
    generate_keypair_pem,
    load_private_key,
    load_public_key,
    mint_token,
)
from point_in_time_feature_store.timeutil import US_PER_DAY, US_PER_SECOND

pytestmark = pytest.mark.integration

DATASET = generate_dataset(SyntheticConfig(n_customers=150, seed=9))
T2 = DATASET.config.day_us(DATASET.config.horizon_days)
METRICS_KEY = "metrics-scraper-key-" + "k" * 24
ACTIVE = DATASET.truth.filter(pl.col("churn_us").is_null())["entity_id"].to_list()
ALL = ["customer_profile:*", "transaction_stats:*", "session_stats:*"]


def make_stack(root: Path, **overrides: Any) -> LocalStack:
    settings = {
        "api_key_sha256": [hash_api_key(METRICS_KEY)],
        "online_max_batch_entities": 5,
        **overrides,
    }
    stack = build_local_stack(root, now_us=T2, settings_overrides=settings)
    services = stack.services
    services.registry.apply(churn_feature_repo(), actor="tests")
    services.online_service.refresh()
    for name, frame in DATASET.sources.items():
        services.ingest(name, frame)
    for view in services.registry.active_repo().feature_views:
        services.materializer.materialize(view.name, up_to_us=T2)
    return stack


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[LocalStack]:
    local = make_stack(tmp_path)
    yield local
    local.close()


@pytest.fixture
def client(stack: LocalStack) -> Iterator[TestClient]:
    with TestClient(
        create_feature_app(stack.services), raise_server_exceptions=False
    ) as test_client:
        yield test_client


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def read(client: TestClient, token: str | None, entities: list[str], features: list[str]) -> Any:
    headers = {} if token is None else bearer(token)
    return client.post(
        "/v1/features/online", json={"entity_ids": entities, "features": features}, headers=headers
    )


def test_probes_report_registry_redis_and_scripts(client: TestClient, stack: LocalStack) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {"registry": "ok", "redis": "ok", "scripts": "ok"}
    stack.services.redis_client.script_flush()
    not_ready = client.get("/readyz")
    assert not_ready.status_code == 503
    assert not_ready.json()["checks"]["scripts"] == "fail"


def test_feature_routes_require_a_valid_token(client: TestClient, stack: LocalStack) -> None:
    missing = read(client, None, ACTIVE[:1], ALL)
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert read(client, "garbage", ACTIVE[:1], ALL).status_code == 401
    other_private, _ = generate_keypair_pem()
    forged = mint_token(
        load_private_key(other_private),
        subject="x",
        scopes=["features:read:*"],
        issuer="pitfs-local-issuer",
        audience="point-in-time-feature-store",
    )
    assert read(client, forged, ACTIVE[:1], ALL).status_code == 401
    expired = stack.token(["features:read:*"], ttl_seconds=-300)
    assert read(client, expired, ACTIVE[:1], ALL).status_code == 401
    assert client.delete(f"/v1/entities/{ACTIVE[0]}").status_code == 401
    assert client.get("/v1/views").status_code == 401


def test_view_scopes_are_enforced_and_pii_is_omitted_without_its_scope(
    client: TestClient, stack: LocalStack
) -> None:
    narrow = stack.token(["features:read:session_stats"])
    assert read(client, narrow, ACTIVE[:2], ["customer_profile:plan_tier"]).status_code == 403
    assert read(client, narrow, ACTIVE[:2], ["session_stats:*"]).status_code == 200

    plain = read(
        client, stack.token(["features:read:*"]), ACTIVE[:2], ["customer_profile:*"]
    ).json()
    assert plain["omitted"] == ["customer_profile:email"]
    assert all("customer_profile:email" not in item["features"] for item in plain["results"])
    served = read(
        client, stack.token(["features:read:*", "pii:read"]), ACTIVE[:2], ["customer_profile:email"]
    )
    body = served.json()
    assert body["omitted"] == []
    assert all(
        str(item["features"]["customer_profile:email"]).endswith("@example.com")
        for item in body["results"]
    )


def test_served_values_equal_offline_point_in_time_values(
    client: TestClient, stack: LocalStack
) -> None:
    now = T2 + 2 * US_PER_DAY
    stack.clock.now_us = now
    entities = ACTIVE[:5]
    body = read(client, stack.token(["features:read:*", "pii:read"]), entities, ALL).json()
    spine = pl.DataFrame({"entity_id": entities, "label_time": [now] * 5, "as_of": [T2] * 5})
    offline = stack.services.retriever().get_historical_features(spine, ALL).frame
    for item, record in zip(body["results"], offline.iter_rows(named=True), strict=True):
        assert item["entity_id"] == record["entity_id"]
        for ref, value in item["features"].items():
            assert value == record[ref.replace(":", "__")], ref
        meta = item["views"]["transaction_stats"]
        if meta["status"] == "ok":
            assert meta["freshness_seconds"] > 0
            assert meta["event_time"] is not None
    assert body["watermarks"]["session_stats"] is not None


def test_requests_are_bounded_and_validated(client: TestClient, stack: LocalStack) -> None:
    token = stack.token(["features:read:*"])
    too_many = read(client, token, ACTIVE[:6], ALL)
    assert too_many.status_code == 422
    assert too_many.json()["error"]["code"] == "batch_too_large"
    assert read(client, token, ["bad key!"], ALL).json()["error"]["code"] == "invalid_request"
    assert read(client, token, ACTIVE[:1], ["nope:x"]).json()["error"]["code"] == "invalid_request"
    assert read(client, token, ACTIVE[:1], ["customer_profile:nope"]).status_code == 422
    extra = client.post(
        "/v1/features/online",
        json={"entity_ids": ACTIVE[:1], "features": ALL, "sql": "DROP TABLE x"},
        headers=bearer(token),
    )
    assert extra.status_code == 422
    assert "DROP TABLE" not in extra.text


@pytest.mark.usefixtures("restore_logging")
def test_access_is_audited_and_logs_carry_only_entity_hmacs(
    client: TestClient, stack: LocalStack
) -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    entities = ACTIVE[:3]
    assert (
        read(
            client, stack.token(["features:read:*"], subject="svc-audit"), entities, ALL
        ).status_code
        == 200
    )
    output = stream.getvalue()
    for entity in entities:
        assert entity not in output
        assert stack.services.hasher.entity(entity) in output
    entry = stack.services.registry.access_audit.entries()[-1]
    assert entry.actor == "svc-audit"
    assert entry.payload["entity_count"] == 3
    assert entry.payload["entities_digest"] == stack.services.hasher.batch(entities)
    assert stack.services.registry.verify_chains()["access_audit"].ok


def test_denied_reads_and_erasures_are_audited(client: TestClient, stack: LocalStack) -> None:
    audit = stack.services.registry.access_audit
    hasher = stack.services.hasher
    narrow = stack.token(["features:read:session_stats"], subject="svc-narrow")
    assert read(client, narrow, ACTIVE[:2], ["customer_profile:plan_tier"]).status_code == 403
    denied = audit.entries()[-1]
    assert denied.action == "online_read_denied"
    assert denied.actor == "svc-narrow"
    assert denied.payload == {
        "request_id": denied.payload["request_id"],
        "reason": "forbidden",
        "views": ["customer_profile"],
        "entity_count": 2,
        "entities_digest": hasher.batch(ACTIVE[:2]),
    }
    assert read(client, narrow, ACTIVE[:6], ["session_stats:pages"]).status_code == 422
    assert audit.entries()[-1].payload["reason"] == "batch_too_large"
    wide = stack.token(["features:read:*"], subject="svc-wide")
    assert read(client, wide, ACTIVE[:1], ["bogus view!:x"]).status_code == 422
    invalid = audit.entries()[-1]
    assert (invalid.actor, invalid.payload["reason"]) == ("svc-wide", "invalid_request")
    assert invalid.payload["views"] == ["<invalid>"]

    before = len(audit.entries())
    assert client.delete(f"/v1/entities/{ACTIVE[0]}", headers=bearer(narrow)).status_code == 403
    erase_denied = audit.entries()[-1]
    assert len(audit.entries()) == before + 1
    assert erase_denied.action == "erase_denied"
    assert erase_denied.payload["entity_hmac"] == hasher.entity(ACTIVE[0])
    assert read(client, None, ACTIVE[:1], ALL).status_code == 401
    assert len(audit.entries()) == before + 1  # no verified subject: counted in metrics only
    recorded = json.dumps([entry.payload for entry in audit.entries()])
    assert all(entity not in recorded for entity in ACTIVE[:6])
    assert stack.services.registry.verify_chains()["access_audit"].ok


def test_metrics_expose_the_feature_store_families(client: TestClient, stack: LocalStack) -> None:
    read(client, stack.token(["features:read:*"]), ACTIVE[:2], ALL)
    assert client.get("/metrics").status_code == 401
    text = client.get("/metrics", headers={"X-API-Key": METRICS_KEY}).text
    for family in (
        "online_get_latency_seconds_bucket",
        'feature_freshness_seconds_count{view="transaction_stats"}',
        "# HELP parity_mismatch_ratio",
        'materialization_lag_seconds{view="session_stats"}',
        "# HELP contract_failures_total",
        "# HELP feature_psi",
        "app_http_requests_total",
    ):
        assert family in text


def test_erasure_endpoint_requires_its_scope_and_returns_a_signed_receipt(
    client: TestClient, stack: LocalStack
) -> None:
    victim = ACTIVE[0]
    reader = stack.token(["features:read:*"])
    read(client, reader, [victim], ALL)
    assert client.delete(f"/v1/entities/{victim}", headers=bearer(reader)).status_code == 403
    eraser = stack.token(["entities:erase"], subject="privacy-desk")
    assert client.delete("/v1/entities/bad%20key", headers=bearer(eraser)).status_code == 422
    response = client.delete(f"/v1/entities/{victim}", headers=bearer(eraser))
    assert response.status_code == 200
    assert victim not in response.text
    receipt = ErasureReceipt.model_validate(response.json())
    assert receipt.requested_by == "privacy-desk"
    assert verify_receipt(receipt, load_public_key(stack.receipt_public_pem))
    after = read(client, reader, [victim], ALL).json()["results"][0]["views"]
    assert {meta["status"] for meta in after.values()} == {"missing"}
    assert stack.services.erasure.verify_purged(victim).clean


def test_views_listing_needs_registry_scope_and_hides_transformations(
    client: TestClient, stack: LocalStack
) -> None:
    assert (
        client.get("/v1/views", headers=bearer(stack.token(["features:read:*"]))).status_code == 403
    )
    listing = client.get("/v1/views", headers=bearer(stack.token(["registry:read"])))
    views = {view["name"]: view for view in listing.json()["views"]}
    assert (
        views["customer_profile"]["version"]
        == churn_feature_repo().view("customer_profile").version_hash()
    )
    assert {"name": "email", "dtype": "string", "pii": True} in views["customer_profile"][
        "features"
    ]
    assert "expression" not in listing.text


def test_rate_limits_apply_per_verified_subject(tmp_path: Path) -> None:
    stack = make_stack(tmp_path, rate_limit_burst=3, rate_limit_per_second=0.001)
    try:
        with TestClient(
            create_feature_app(stack.services), raise_server_exceptions=False
        ) as client:
            alice = stack.token(["features:read:*"], subject="alice")
            bob = stack.token(["features:read:*"], subject="bob")
            codes = [
                read(client, alice, ACTIVE[:1], ["session_stats:pages"]).status_code
                for _ in range(4)
            ]
            assert codes == [200, 200, 200, 429]
            assert read(client, bob, ACTIVE[:1], ["session_stats:pages"]).status_code == 200
    finally:
        stack.close()


def test_production_configuration_fails_closed(tmp_path: Path) -> None:
    redis_client = fakeredis.FakeRedis()
    with pytest.raises(ServiceConfigurationError, match="LOG_HMAC_KEY"):
        build_services(
            Settings(_env_file=None, environment="prod", data_dir=tmp_path / "a"),
            redis_client=redis_client,
        )
    hmac_key = "h" * 40
    with pytest.raises(ServiceConfigurationError, match="SIGNING_KEY"):
        build_services(
            Settings(
                _env_file=None, environment="prod", data_dir=tmp_path / "b", log_hmac_key=hmac_key
            ),
            redis_client=redis_client,
        )
    receipt_private, _ = generate_keypair_pem()
    _, jwt_public = generate_keypair_pem()
    (tmp_path / "receipt.pem").write_bytes(receipt_private)
    (tmp_path / "jwt.pem").write_bytes(jwt_public)
    with pytest.raises(ServiceConfigurationError, match="JWT_PUBLIC_KEY"):
        build_services(
            Settings(
                _env_file=None,
                environment="prod",
                data_dir=tmp_path / "c",
                receipt_signing_key_path=tmp_path / "receipt.pem",
                log_hmac_key=hmac_key,
            ),
            redis_client=redis_client,
        )
    configured = build_services(
        Settings(
            _env_file=None,
            environment="prod",
            data_dir=tmp_path / "d",
            receipt_signing_key_path=tmp_path / "receipt.pem",
            jwt_public_key_path=tmp_path / "jwt.pem",
            log_hmac_key=hmac_key,
        ),
        redis_client=redis_client,
    )
    assert configured.token_verifier is not None
    configured.close()
    configured.close()

    dev = build_services(
        Settings(_env_file=None, data_dir=tmp_path / "e"), redis_client=redis_client
    )
    try:
        assert dev.token_verifier is None
        with TestClient(create_feature_app(dev), raise_server_exceptions=False) as client:
            assert (
                client.post(
                    "/v1/features/online", json={"entity_ids": ["a"], "features": ALL}
                ).status_code
                == 401
            )
    finally:
        dev.close()
    assert US_PER_SECOND == 1_000_000
