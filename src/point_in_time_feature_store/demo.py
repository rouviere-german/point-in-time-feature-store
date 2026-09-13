"""Offline end-to-end demo: definitions, contracts, retrieval, leakage, serving, parity, erasure.

Runs without network access or API keys. Redis is served by fakeredis (a development dependency)
unless ``redis_url`` points at a real server whose admin connection may load the Lua scripts.
Every number in the returned summary is computed during the run.
"""

from __future__ import annotations

import io
import random
import shutil
from pathlib import Path
from typing import Any, cast

import polars as pl
import redis
from fastapi.testclient import TestClient
from pydantic import JsonValue

from point_in_time_feature_store.core.logging import configure_logging
from point_in_time_feature_store.devtools import build_local_stack, fake_redis
from point_in_time_feature_store.erasure import ErasureReceipt, verify_receipt
from point_in_time_feature_store.ml.contracts import DataContractError
from point_in_time_feature_store.monitoring import FeatureDriftMonitor
from point_in_time_feature_store.offline.synthetic import (
    SyntheticConfig,
    churn_feature_repo,
    generate_dataset,
)
from point_in_time_feature_store.online.materialize import Materializer
from point_in_time_feature_store.online.store import OnlineStore
from point_in_time_feature_store.registry import ApprovalRequiredError, FeatureRegistry
from point_in_time_feature_store.services import FeatureStoreServices
from point_in_time_feature_store.serving.api import create_feature_app
from point_in_time_feature_store.serving.auth import load_public_key
from point_in_time_feature_store.timeutil import US_PER_DAY
from point_in_time_feature_store.training import (
    LeakageConfig,
    build_label_spine,
    run_leakage_experiment,
)

__all__ = ["run_demo"]

ALL_FEATURES = ["customer_profile:*", "transaction_stats:*", "session_stats:*"]


def _materialize_all(services: FeatureStoreServices, up_to_us: int) -> dict[str, JsonValue]:
    reports: dict[str, JsonValue] = {}
    for view in services.registry.active_repo().feature_views:
        report = services.materializer.materialize(view.name, up_to_us=up_to_us)
        reports[view.name] = {
            "written": report.written,
            "skipped": report.skipped,
            "rows": report.rows_scanned,
        }
    return reports


def _clear_demo_keys(client: redis.Redis) -> None:
    """Remove keys of a previous demo run (a dedicated Redis database is assumed)."""
    keys = list(client.scan_iter(match="fs:v1:*", count=1_000))
    if keys:
        client.delete(*keys)


def _same(left: str, right: str | None) -> bool | None:
    return None if right is None else left == right


def _one_shot_digest(services: FeatureStoreServices, root: Path, up_to_us: int) -> str | None:
    """Materialize the whole history in one run into a separate store and hash it."""
    registry = FeatureRegistry(root / "replay_registry.sqlite", clock=services.clock)
    try:
        registry.apply(services.registry.active_repo(), actor="demo-replay")
        try:
            client = fake_redis()
        except RuntimeError:
            return None  # fakeredis is a dev dependency; the container demo skips the replay
        online = OnlineStore(client)
        online.load_scripts()
        replay = Materializer(
            registry, services.store, online, services.engine, services.hasher, clock=services.clock
        )
        for view in registry.active_repo().feature_views:
            replay.materialize(view.name, up_to_us=up_to_us)
        return online.state_digest()
    finally:
        registry.close()


def _post(client: TestClient, token: str | None, entity_ids: list[str], features: list[str]) -> Any:
    headers = {} if token is None else {"Authorization": f"Bearer {token}"}
    return client.post(
        "/v1/features/online",
        json={"entity_ids": entity_ids, "features": features},
        headers=headers,
    )


def run_demo(
    workdir: Path,
    *,
    n_customers: int = 5_000,
    seed: int = 7,
    online_requests: int = 1_000,
    redis_url: str | None = None,
) -> dict[str, JsonValue]:
    root = workdir / "pitfs"
    if root.exists():
        shutil.rmtree(root)
    (root / "keys").mkdir(parents=True)
    log_buffer = io.StringIO()
    configure_logging("INFO", json_output=True, stream=log_buffer)

    dataset = generate_dataset(SyntheticConfig(n_customers=n_customers, seed=seed))
    config = dataset.config
    t1 = config.day_us(300)
    t2 = config.day_us(config.horizon_days)
    stack = build_local_stack(
        root,
        now_us=t1,
        redis_client=None if redis_url is None else redis.Redis.from_url(redis_url),
    )
    services, clock = stack.services, stack.clock
    if redis_url is not None:
        _clear_demo_keys(services.redis_client)
    receipt_public = stack.receipt_public_pem
    (root / "keys" / "receipt_public.pem").write_bytes(receipt_public)
    summary: dict[str, JsonValue] = {
        "source_rows": dataset.total_rows,
        "customers": dataset.truth.height,
    }
    try:
        # 1. Definitions: plan/apply, and a destructive change that is refused without approval.
        repo = churn_feature_repo()
        created = services.registry.apply(repo, actor="demo")
        changed = repo.model_copy(
            update={
                "feature_views": tuple(
                    view.model_copy(update={"ttl_seconds": 7 * 86_400})
                    if view.name == "transaction_stats"
                    else view
                    for view in repo.feature_views
                )
            }
        )
        destructive = services.registry.plan(changed)
        try:
            services.registry.apply(changed, actor="demo")
            blocked = False
        except ApprovalRequiredError:
            blocked = True
        summary["registry"] = {
            "created": len(created.changes),
            "view_versions": {view.name: view.version_hash()[:16] for view in repo.feature_views},
            "destructive_plan_reasons": list(destructive.changes[0].reasons),
            "destructive_apply_blocked_without_approval": blocked,
        }
        services.online_service.refresh()

        # 2. Offline ingest with contracts, in two knowledge-time phases.
        phase_a = {
            name: frame.filter(pl.col("ingested_at").dt.epoch("us") <= t1)
            for name, frame in dataset.sources.items()
        }
        phase_b = {
            name: frame.filter(pl.col("ingested_at").dt.epoch("us") > t1)
            for name, frame in dataset.sources.items()
        }
        for name, frame in phase_a.items():
            services.ingest(name, frame)
        broken = phase_b["transactions"].head(5).with_columns(pl.lit(-1.0).alias("amount"))
        try:
            services.ingest("transactions", broken)
            contract_rejected: JsonValue = []
        except DataContractError as exc:
            contract_rejected = [dict(item) for item in exc.failures]
        materialized_t1 = _materialize_all(services, t1)
        late_stamped = phase_a["transactions"].head(3)
        try:
            services.ingest("transactions", late_stamped)
            append_only_rejected = False
        except DataContractError as exc:
            append_only_rejected = any(
                item["check"] == "knowledge_time_append_only" for item in exc.failures
            )
        for name, frame in phase_b.items():
            services.ingest(name, frame)
        clock.now_us = t2
        materialized_t2 = _materialize_all(services, t2)
        incremental_digest = services.online.state_digest()
        rerun = _materialize_all(services, t2)
        summary["ingest_and_materialize"] = {
            "contract_rejection": contract_rejected,
            "late_stamped_rows_rejected": append_only_rejected,
            "materialized_to_day_300": materialized_t1,
            "materialized_to_horizon": materialized_t2,
            "rerun_is_noop": all(
                isinstance(item, dict) and item["written"] == 0 for item in rerun.values()
            ),
            "incremental_equals_one_shot": _same(
                incremental_digest, _one_shot_digest(services, root, t2)
            ),
            "online_state_digest": incremental_digest[:16],
        }

        # 3. Leakage experiment with MLflow lineage.
        retriever = services.retriever()
        leakage = run_leakage_experiment(
            dataset, retriever, seed=seed, registry=services.registry, mlflow_dir=root / "mlruns"
        )
        summary["leakage"] = leakage.summary()

        # 4. Serving with scopes.
        app = create_feature_app(services)
        entities = dataset.truth.filter(pl.col("churn_us").is_null())["entity_id"].to_list()
        rng = random.Random(seed)  # noqa: S311  # nosec B311
        reader = stack.token(["features:read:*"], subject="svc-scoring")
        pii_reader = stack.token(["features:read:*", "pii:read"], subject="svc-crm")
        narrow = stack.token(["features:read:session_stats"], subject="svc-sessions")
        eraser = stack.token(["entities:erase"], subject="privacy-desk")
        with TestClient(app, raise_server_exceptions=False) as http:
            ready = http.get("/readyz").json()
            sample = entities[:3]
            plain = _post(http, reader, sample, ALL_FEATURES).json()
            with_pii = _post(http, pii_reader, sample, ["customer_profile:email"]).json()
            forbidden = _post(http, narrow, sample, ["customer_profile:plan_tier"]).status_code
            unauthenticated = _post(http, None, sample, ALL_FEATURES).status_code
            statuses: dict[str, int] = {}
            for _ in range(online_requests):
                clock.now_us = t2 + rng.randrange(0, 3 * US_PER_DAY)
                batch = rng.sample(entities, rng.randint(1, 4))
                response = _post(http, reader, batch, ALL_FEATURES)
                statuses[str(response.status_code)] = statuses.get(str(response.status_code), 0) + 1
            summary["serving"] = {
                "readyz": ready,
                "pii_omitted_without_scope": plain["omitted"],
                "pii_served_with_scope": with_pii["results"][0]["features"][
                    "customer_profile:email"
                ]
                is not None,
                "forbidden_view_status": forbidden,
                "unauthenticated_status": unauthenticated,
                "online_requests": cast(JsonValue, statuses),
                "request_log_rows": services.request_log.count(),
            }

            # 5. Parity between online reads and offline recomputation.
            skew = services.skew_checker().check(
                sample_size=services.request_log.count(), seed=seed
            )
            summary["parity"] = {
                "checked": skew.checked,
                "mismatches": skew.mismatches,
                "per_view": {
                    name: {"checked": v.checked, "mismatches": v.mismatches}
                    for name, v in skew.views.items()
                },
            }

            # 6. Erasure with a signed receipt and purge verification.
            victim = sample[0]
            clock.now_us = t2 + 3 * US_PER_DAY
            erased = http.delete(
                f"/v1/entities/{victim}", headers={"Authorization": f"Bearer {eraser}"}
            )
            receipt = ErasureReceipt.model_validate(erased.json())
            tampered = receipt.model_copy(
                update={"online_keys_deleted": receipt.online_keys_deleted + 1}
            )
            services.ingest("transactions", phase_b["transactions"].head(0))
            after = _materialize_all(services, clock.now_us)
            purge = services.erasure.verify_purged(victim, log_texts=[log_buffer.getvalue()])
            summary["erasure"] = {
                "status": erased.status_code,
                "partitions_rewritten": len(receipt.offline_rewrites),
                "rows_removed": sum(item.rows_removed for item in receipt.offline_rewrites),
                "online_keys_deleted": receipt.online_keys_deleted,
                "request_log_rows_deleted": receipt.request_log_rows_deleted,
                "receipt_signature_valid": verify_receipt(receipt, load_public_key(receipt_public)),
                "tampered_receipt_rejected": not verify_receipt(
                    tampered, load_public_key(receipt_public)
                ),
                "purge_clean": purge.clean,
                "rematerialized_after_erasure": after,
                "raw_entity_id_in_logs": log_buffer.getvalue().count(victim),
            }

        # 7. Drift between the training and holdout periods.
        leak_cfg = LeakageConfig()
        common = {
            "every_days": leak_cfg.label_every_days,
            "horizon_days": leak_cfg.label_horizon_days,
        }
        train = build_label_spine(
            dataset,
            start_day=leak_cfg.train_start_day,
            end_day=leak_cfg.train_end_day,
            split="train",
            **common,
        )
        hold = build_label_spine(
            dataset,
            start_day=leak_cfg.holdout_start_day,
            end_day=leak_cfg.holdout_end_day,
            split="holdout",
            **common,
        )
        numeric = [
            "transaction_stats__last_amount",
            "transaction_stats__txn_count_30d",
            "session_stats__last_session_minutes",
            "session_stats__sessions_7d",
            "customer_profile__support_tickets_90d",
        ]
        refs = [ref.replace("__", ":") for ref in numeric]
        monitor = FeatureDriftMonitor(retriever.get_historical_features(train, refs).frame, numeric)
        drift = monitor.check(retriever.get_historical_features(hold, refs).frame, services.metrics)
        summary["drift"] = {
            "max_psi": round(max(item.psi for item in drift.report.features), 4),
            "drifted_features": list(drift.report.drifted_features),
        }

        chains = services.registry.verify_chains()
        summary["audit"] = {
            name: {"ok": item.ok, "entries": item.entries} for name, item in chains.items()
        }
        return summary
    finally:
        services.close()
        configure_logging("WARNING", json_output=True)
