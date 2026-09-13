"""Wiring: build every component of the feature store from :class:`Settings`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import redis
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CollectorRegistry

from point_in_time_feature_store.core.config import Settings
from point_in_time_feature_store.core.logging import get_logger
from point_in_time_feature_store.core.tracing import build_tracer_provider
from point_in_time_feature_store.erasure import ErasureService, ReceiptSigner
from point_in_time_feature_store.hashing import EntityHasher
from point_in_time_feature_store.ml.contracts import DataContractError
from point_in_time_feature_store.monitoring import FeatureStoreMetrics, create_feature_store_metrics
from point_in_time_feature_store.offline.retrieval import DuckDBEngine, HistoricalRetriever
from point_in_time_feature_store.offline.store import AppendReport, OfflineStore
from point_in_time_feature_store.online.materialize import Materializer
from point_in_time_feature_store.online.store import OnlineStore
from point_in_time_feature_store.parity import SkewChecker
from point_in_time_feature_store.registry import FeatureRegistry
from point_in_time_feature_store.serving.auth import (
    TokenVerifier,
    generate_keypair_pem,
    load_private_key,
    load_public_key,
)
from point_in_time_feature_store.serving.request_log import RequestLog
from point_in_time_feature_store.serving.service import OnlineFeatureService
from point_in_time_feature_store.timeutil import Clock, system_clock_us

__all__ = ["FeatureStoreServices", "ServiceConfigurationError", "build_services"]

_log = get_logger("point_in_time_feature_store.services")


class ServiceConfigurationError(RuntimeError):
    """Required production configuration is missing."""


@dataclass(slots=True)
class FeatureStoreServices:
    settings: Settings
    clock: Clock
    registry: FeatureRegistry
    store: OfflineStore
    engine: DuckDBEngine
    redis_client: redis.Redis
    online: OnlineStore
    hasher: EntityHasher
    metrics: FeatureStoreMetrics
    request_log: RequestLog
    materializer: Materializer
    online_service: OnlineFeatureService
    erasure: ErasureService
    signer: ReceiptSigner
    token_verifier: TokenVerifier | None
    tracer_provider: TracerProvider
    closed: bool = field(default=False)

    def retriever(self) -> HistoricalRetriever:
        return HistoricalRetriever(self.store, self.registry.active_repo(), self.engine)

    def ingest(self, source_name: str, frame: pl.DataFrame) -> AppendReport:
        """Append with every guard: contract, tombstones and append-only knowledge time."""
        repo = self.registry.active_repo()
        source = repo.source(source_name)
        try:
            return self.store.append(
                source,
                repo.entity(source.entity),
                frame,
                hasher=self.hasher,
                # Both are evaluated while the store lock is held (see OfflineStore.append).
                tombstones=self.registry.tombstones,
                min_ingested_at_us=lambda: self.registry.knowledge_floor_for_source(source_name),
            )
        except DataContractError as exc:
            for failure in exc.failures:
                self.metrics.contract_failures_total.labels(source_name, str(failure["check"])).inc(
                    int(failure["count"])
                )
            raise

    def skew_checker(self) -> SkewChecker:
        return SkewChecker(self.request_log, self.retriever(), self.hasher, metrics=self.metrics)

    def close(self) -> None:
        if self.closed:
            return
        self.request_log.close()
        self.engine.close()
        self.registry.close()
        self.closed = True


def _hasher(settings: Settings) -> EntityHasher:
    if settings.log_hmac_key is not None:
        return EntityHasher(settings.log_hmac_key.get_secret_value().encode("utf-8"))
    if settings.environment == "prod":
        raise ServiceConfigurationError("APP_LOG_HMAC_KEY is required in prod")
    return EntityHasher.ephemeral()


def _signer(settings: Settings) -> ReceiptSigner:
    if settings.receipt_signing_key_path is not None:
        return ReceiptSigner(load_private_key(settings.receipt_signing_key_path.read_bytes()))
    if settings.environment == "prod":
        raise ServiceConfigurationError("APP_RECEIPT_SIGNING_KEY_PATH is required in prod")
    _log.warning("ephemeral_receipt_signing_key", reason="no signing key configured")
    return ReceiptSigner(load_private_key(generate_keypair_pem()[0]))


def _verifier(settings: Settings) -> TokenVerifier | None:
    if settings.jwt_public_key_path is None:
        if settings.environment == "prod":
            raise ServiceConfigurationError("APP_JWT_PUBLIC_KEY_PATH is required in prod")
        return None  # every feature route then fails closed with 401
    return TokenVerifier(
        load_public_key(settings.jwt_public_key_path.read_bytes()),
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
    )


def build_services(
    settings: Settings,
    *,
    redis_client: redis.Redis | None = None,
    clock: Clock = system_clock_us,
    hasher: EntityHasher | None = None,
    signer: ReceiptSigner | None = None,
    token_verifier: TokenVerifier | None = None,
    metrics_registry: CollectorRegistry | None = None,
    data_dir: Path | None = None,
) -> FeatureStoreServices:
    root = data_dir if data_dir is not None else settings.data_dir
    root.mkdir(parents=True, exist_ok=True)
    registry = FeatureRegistry(root / "registry.sqlite", clock=clock)
    store = OfflineStore(root / "offline")
    engine = DuckDBEngine(
        threads=settings.duckdb_threads,
        memory_limit=settings.duckdb_memory_limit,
        temp_directory=root / "duckdb_tmp",
    )
    client = redis_client
    if client is None:
        password = (
            None if settings.redis_password is None else settings.redis_password.get_secret_value()
        )
        client = redis.Redis.from_url(
            settings.redis_url,
            username=settings.redis_username,
            password=password,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
        )
    online = OnlineStore(client)
    entity_hasher = hasher if hasher is not None else _hasher(settings)
    metrics = create_feature_store_metrics(metrics_registry)
    request_log = RequestLog(
        root / "request_log.sqlite", sample_rate=settings.request_log_sample_rate
    )
    tracer_provider = build_tracer_provider(settings.service_name, settings.tracing_exporter)
    materializer = Materializer(
        registry,
        store,
        online,
        engine,
        entity_hasher,
        metrics=metrics,
        clock=clock,
        tracer=tracer_provider.get_tracer("point_in_time_feature_store.materialize"),
    )
    online_service = OnlineFeatureService(
        registry, online, entity_hasher, request_log=request_log, metrics=metrics, clock=clock
    )
    receipt_signer = signer if signer is not None else _signer(settings)
    erasure = ErasureService(
        registry, store, online, request_log, entity_hasher, receipt_signer, clock=clock
    )
    return FeatureStoreServices(
        settings=settings,
        clock=clock,
        registry=registry,
        store=store,
        engine=engine,
        redis_client=client,
        online=online,
        hasher=entity_hasher,
        metrics=metrics,
        request_log=request_log,
        materializer=materializer,
        online_service=online_service,
        erasure=erasure,
        signer=receipt_signer,
        token_verifier=token_verifier if token_verifier is not None else _verifier(settings),
        tracer_provider=tracer_provider,
    )
