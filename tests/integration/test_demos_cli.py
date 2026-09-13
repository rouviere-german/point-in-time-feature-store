from __future__ import annotations

import json
import re
import runpy
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fakeredis
import pytest
import redis
from opentelemetry import trace as otel_trace

from point_in_time_feature_store import cli
from point_in_time_feature_store.catalog import render_catalog
from point_in_time_feature_store.core import tracing
from point_in_time_feature_store.core.config import get_settings
from point_in_time_feature_store.core.security import hash_api_key
from point_in_time_feature_store.definitions import FeatureRepo
from point_in_time_feature_store.erasure import ErasureReceipt, ReceiptSigner
from point_in_time_feature_store.offline.synthetic import churn_feature_repo
from point_in_time_feature_store.online.acl import validate_acl
from point_in_time_feature_store.online.store import OnlineStore
from point_in_time_feature_store.registry import FeatureRegistry
from point_in_time_feature_store.serving.auth import (
    TokenVerifier,
    load_private_key,
    load_public_key,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("restore_logging")
def test_demo_runs_end_to_end_offline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "demo",
            "--workdir",
            str(tmp_path),
            "--customers",
            "700",
            "--requests",
            "40",
            "--seed",
            "3",
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert code == 0
    assert summary["registry"]["destructive_apply_blocked_without_approval"] is True
    assert summary["ingest_and_materialize"]["late_stamped_rows_rejected"] is True
    assert summary["ingest_and_materialize"]["incremental_equals_one_shot"] is True
    assert summary["ingest_and_materialize"]["rerun_is_noop"] is True
    assert summary["ingest_and_materialize"]["contract_rejection"]
    assert summary["serving"]["forbidden_view_status"] == 403
    assert summary["serving"]["unauthenticated_status"] == 401
    assert summary["serving"]["pii_omitted_without_scope"] == ["customer_profile:email"]
    assert summary["parity"]["mismatches"] == 0
    assert summary["parity"]["checked"] > 0
    assert summary["erasure"]["receipt_signature_valid"] is True
    assert summary["erasure"]["tampered_receipt_rejected"] is True
    assert summary["erasure"]["purge_clean"] is True
    assert summary["erasure"]["raw_entity_id_in_logs"] == 0
    assert all(item["ok"] for item in summary["audit"].values())
    assert summary["leakage"]["mlflow_run_id"]


@pytest.mark.usefixtures("restore_logging")
def test_definition_workflow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    definitions = tmp_path / "definitions.json"
    registry = tmp_path / "registry.sqlite"
    assert cli.main(["export-definitions", "--out", str(definitions)]) == 0
    assert FeatureRepo.from_json(definitions.read_text(encoding="utf-8")) == churn_feature_repo()
    assert cli.main(["plan", "--definitions", str(definitions), "--registry", str(registry)]) == 0
    assert "+ feature_view customer_profile (create)" in capsys.readouterr().out
    assert (
        cli.main(
            [
                "apply",
                "--definitions",
                str(definitions),
                "--registry",
                str(registry),
                "--actor",
                "alice",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert cli.main(["plan", "--definitions", str(definitions), "--registry", str(registry)]) == 0
    assert capsys.readouterr().out.strip() == "No changes."

    changed = json.loads(definitions.read_text(encoding="utf-8"))
    for view in changed["feature_views"]:
        if view["name"] == "transaction_stats":
            view["ttl_seconds"] = 86_400
    definitions.write_text(json.dumps(changed), encoding="utf-8")
    apply = [
        "apply",
        "--definitions",
        str(definitions),
        "--registry",
        str(registry),
        "--actor",
        "bob",
    ]
    assert cli.main(apply) == 2
    err = capsys.readouterr().err
    plan_hash = re.search(r"plan hash: ([0-9a-f]{64})", err)
    assert plan_hash is not None
    assert cli.main([*apply, "--approve", "--plan-hash", "0" * 64]) == 2
    capsys.readouterr()
    assert cli.main([*apply, "--approve", "--plan-hash", plan_hash.group(1)]) == 0
    assert cli.main(["verify-chains", "--registry", str(registry)]) == 0
    chains = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert chains["change_log"]["entries"] == 4

    out = tmp_path / "catalog.md"
    assert cli.main(["catalog", "--definitions", str(definitions), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == render_catalog(
        FeatureRepo.from_json(definitions.read_text(encoding="utf-8"))
    )
    assert cli.main(["catalog"]) == 0
    assert capsys.readouterr().out == render_catalog(churn_feature_repo())


@pytest.mark.usefixtures("restore_logging")
def test_chain_anchor_commands_detect_truncation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    keys = tmp_path / "keys"
    assert cli.main(["generate-keys", "--out-dir", str(keys)]) == 0
    registry_path = tmp_path / "registry.sqlite"
    registry = FeatureRegistry(registry_path)
    registry.apply(churn_feature_repo(), actor="alice")
    for index in range(3):
        registry.record_access("svc", "online_read", {"entity_count": index})
    registry.close()
    base = ["--registry", str(registry_path)]
    anchor = tmp_path / "anchor.json"
    signing = ["--signing-key", str(keys / "receipt_private.pem")]
    assert cli.main(["anchor-chains", *base, *signing, "--out", str(anchor)]) == 0
    capsys.readouterr()
    verify = [
        "verify-chains",
        *base,
        "--anchor",
        str(anchor),
        "--public-key",
        str(keys / "receipt_public.pem"),
    ]
    assert cli.main(verify) == 0
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (report["anchored"], report["anchor_signature_valid"]) == (True, True)
    assert cli.main(["verify-chains", *base, "--anchor", str(anchor)]) == 2
    assert cli.main(["verify-chains", *base]) == 0
    assert "truncated tail" in capsys.readouterr().err

    with sqlite3.connect(registry_path) as connection:
        connection.execute("DELETE FROM access_audit WHERE seq = 3")  # drop the tail
    assert cli.main(["verify-chains", *base]) == 0  # the bare chain cannot tell
    capsys.readouterr()
    assert cli.main(verify) == 1
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["access_audit"]["ok"] is False
    assert report["change_log"]["ok"] is True

    with sqlite3.connect(registry_path) as connection:
        connection.execute("UPDATE access_audit SET payload = '{}' WHERE seq = 1")
    refused = tmp_path / "refused.json"
    assert cli.main(["anchor-chains", *base, *signing, "--out", str(refused)]) == 1
    assert not refused.exists()


@pytest.mark.usefixtures("restore_logging")
def test_key_token_receipt_and_acl_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = tmp_path / "keys"
    assert cli.main(["generate-keys", "--out-dir", str(keys)]) == 0
    token_args = [
        "mint-token",
        "--private-key",
        str(keys / "jwt_private.pem"),
        "--subject",
        "svc-a",
    ]
    capsys.readouterr()
    assert cli.main([*token_args, "--scope", "features:read:*", "--scope", "pii:read"]) == 0
    token = capsys.readouterr().out.strip()
    verifier = TokenVerifier(
        load_public_key((keys / "jwt_public.pem").read_bytes()),
        issuer="pitfs-dev-issuer",
        audience="point-in-time-feature-store",
    )
    caller = verifier.verify(token)
    assert caller is not None
    assert caller.can_read_pii

    signer = ReceiptSigner(load_private_key((keys / "receipt_private.pem").read_bytes()))
    receipt = signer.sign(
        ErasureReceipt(
            receipt_id="r1",
            entity_hmac="a" * 64,
            requested_by="desk",
            erased_at="2026-01-01T00:00:00+00:00",
            offline_rewrites=(),
            online_keys_deleted=1,
            request_log_rows_deleted=0,
            tombstoned=True,
            change_log_entry_hash="b" * 64,
            key_id="",
        )
    )
    path = tmp_path / "receipt.json"
    path.write_text(receipt.model_dump_json(), encoding="utf-8")
    public = str(keys / "receipt_public.pem")
    assert cli.main(["verify-receipt", str(path), "--public-key", public]) == 0
    path.write_text(
        receipt.model_copy(update={"online_keys_deleted": 9}).model_dump_json(), encoding="utf-8"
    )
    assert cli.main(["verify-receipt", str(path), "--public-key", public]) == 1

    acl = tmp_path / "users.acl"
    monkeypatch.delenv("APP_REDIS_PASSWORD", raising=False)
    monkeypatch.delenv("REDIS_ADMIN_PASSWORD", raising=False)
    assert cli.main(["redis-acl", "--out", str(acl)]) == 2
    monkeypatch.setenv("APP_REDIS_PASSWORD", "app-password-0123456789")
    monkeypatch.setenv("REDIS_ADMIN_PASSWORD", "admin-password-0123456789")
    assert cli.main(["redis-acl", "--out", str(acl)]) == 0
    text = acl.read_text(encoding="utf-8")
    assert validate_acl(text, app_user="pitfs_app") == []
    assert "app-password-0123456789" not in text
    capsys.readouterr()


@pytest.mark.usefixtures("fresh_settings", "restore_logging")
def test_data_and_materialize_commands_use_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    server = fakeredis.FakeServer()
    OnlineStore(fakeredis.FakeRedis(server=server)).load_scripts()  # deploy-time admin step
    monkeypatch.setattr(
        redis.Redis, "from_url", lambda *args, **kwargs: fakeredis.FakeRedis(server=server)
    )
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    assert cli.main(["generate-data", "--customers", "60", "--seed", "2"]) == 2
    definitions = tmp_path / "definitions.json"
    cli.main(["export-definitions", "--out", str(definitions)])
    registry = str(tmp_path / "registry.sqlite")
    assert (
        cli.main(
            ["apply", "--definitions", str(definitions), "--registry", registry, "--actor", "ops"]
        )
        == 0
    )
    capsys.readouterr()
    assert cli.main(["generate-data", "--customers", "60", "--seed", "2"]) == 0
    written = json.loads(capsys.readouterr().out)["rows_written"]
    assert written["transactions"] > 0
    assert cli.main(["materialize", "--up-to", "2026-02-01T00:00:00+00:00"]) == 0
    reports = json.loads(capsys.readouterr().out)
    assert set(reports) == {"customer_profile", "session_stats", "transaction_stats"}
    assert reports["transaction_stats"]["written"] > 0
    assert cli.main(["materialize", "--view", "session_stats"]) == 0


@pytest.mark.usefixtures("restore_logging", "fresh_settings")
def test_serve_configures_uvicorn_safely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    monkeypatch.setattr(redis.Redis, "from_url", lambda *args, **kwargs: fakeredis.FakeRedis())
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    assert cli.main(["serve", "--port", "8123"]) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False


@pytest.mark.usefixtures("restore_logging")
def test_generate_api_key_and_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["generate-api-key"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["api_key_sha256"] == hash_api_key(payload["api_key"])
    monkeypatch.setattr(sys, "argv", ["point_in_time_feature_store", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("point_in_time_feature_store", run_name="__main__")
    assert exit_info.value.code == 0
    assert "api_key_sha256" in capsys.readouterr().out


@pytest.mark.usefixtures("restore_logging")
def test_load_scripts_uses_the_admin_connection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = fakeredis.FakeServer()
    monkeypatch.setattr(
        redis.Redis, "from_url", lambda *args, **kwargs: fakeredis.FakeRedis(server=server)
    )
    monkeypatch.delenv("REDIS_ADMIN_URL", raising=False)
    assert cli.main(["load-scripts"]) == 2
    monkeypatch.setenv("REDIS_ADMIN_URL", "redis://admin@localhost:6379/0")
    assert cli.main(["load-scripts"]) == 0
    assert OnlineStore(fakeredis.FakeRedis(server=server)).scripts_loaded()
    assert "conditional_write" in capsys.readouterr().out
