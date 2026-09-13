"""HTTP API of the feature store, on top of the hardened core app.

Routes (all but the probes and ``/metrics`` need a bearer JWT):

- ``POST /v1/features/online``: values, value timestamps and freshness for a batch of entities;
  views need ``features:read:<view>``; PII-tagged features need ``pii:read`` and are otherwise
  omitted (listed under ``omitted``). Every read is appended to the access audit chain.
- ``DELETE /v1/entities/{entity_id}``: erase an entity everywhere (``entities:erase``); returns a
  signed receipt.
- ``GET /v1/views``: active definitions without transformations (``registry:read``).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, Path, Request
from pydantic import BaseModel, ConfigDict, Field

from point_in_time_feature_store.core.app import create_app
from point_in_time_feature_store.core.errors import AppError, request_id_of
from point_in_time_feature_store.core.logging import get_logger
from point_in_time_feature_store.services import FeatureStoreServices
from point_in_time_feature_store.serving.auth import (
    SCOPE_ERASE,
    SCOPE_REGISTRY,
    Caller,
    bearer_token,
    identity_from,
)
from point_in_time_feature_store.serving.service import (
    FeatureAccessDeniedError,
    InvalidFeatureRequestError,
)

__all__ = ["OnlineFeaturesRequest", "create_feature_app"]

_log = get_logger("point_in_time_feature_store.api")


class OnlineFeaturesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_ids: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(min_length=1)
    features: list[Annotated[str, Field(min_length=3, max_length=130)]] = Field(min_length=1)


def _services(request: Request) -> FeatureStoreServices:
    services: FeatureStoreServices = request.app.state.services
    return services


def require_caller(request: Request) -> Caller:
    verifier = _services(request).token_verifier
    caller = None if verifier is None else verifier.verify(bearer_token(request.headers))
    if caller is None:
        request.app.state.metrics.auth_failures_total.inc()
        raise AppError(
            401,
            "unauthorized",
            "A valid bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return caller


CallerDep = Annotated[Caller, Depends(require_caller)]


def create_feature_app(services: FeatureStoreServices) -> FastAPI:
    settings = services.settings

    def registry_ready() -> bool:
        return bool(services.registry.active_repo().feature_views)

    def redis_ready() -> bool:
        return services.online.ping()

    def scripts_ready() -> bool:
        return services.online.scripts_loaded()

    app = create_app(
        settings,
        readiness_checks={
            "registry": registry_ready,
            "redis": redis_ready,
            "scripts": scripts_ready,
        },
        registry=services.metrics.registry,
        tracer_provider=services.tracer_provider,
        rate_limit_identity=identity_from(services.token_verifier),
    )
    app.state.services = services

    @app.post("/v1/features/online")
    def online_features(
        body: OnlineFeaturesRequest, request: Request, caller: CallerDep
    ) -> dict[str, Any]:
        if len(body.entity_ids) > settings.online_max_batch_entities:
            raise AppError(422, "batch_too_large", "Too many entity ids in one request.")
        if len(body.features) > settings.online_max_features:
            raise AppError(422, "too_many_features", "Too many feature references in one request.")
        request_id = request_id_of(request) or "unknown"
        service = services.online_service
        try:
            response = service.get(caller, body.entity_ids, body.features, request_id=request_id)
        except FeatureAccessDeniedError:
            raise AppError(
                403, "forbidden", "The caller may not read a requested feature view."
            ) from None
        except InvalidFeatureRequestError as exc:
            raise AppError(422, "invalid_request", str(exc)) from None
        views = sorted({ref.split(":", 1)[0] for ref in body.features})
        services.registry.record_access(
            caller.subject,
            "online_read",
            {
                "request_id": request_id,
                "views": list(views),
                "entity_count": len(body.entity_ids),
                "entities_digest": services.hasher.batch(body.entity_ids),
                "pii_served": response.pii_served,
            },
        )
        _log.info(
            "online_read",
            subject=caller.subject,
            views=views,
            entity_count=len(body.entity_ids),
            entity_hmacs=[services.hasher.entity(item) for item in body.entity_ids[:5]],
        )
        return {
            "results": response.results,
            "omitted": response.omitted,
            "watermarks": response.watermarks,
        }

    @app.delete("/v1/entities/{entity_id}")
    def erase_entity(
        caller: CallerDep,
        entity_id: Annotated[str, Path(min_length=1, max_length=128)],
    ) -> dict[str, Any]:
        if not caller.has(SCOPE_ERASE):
            raise AppError(403, "forbidden", "The caller may not erase entities.")
        repo = services.registry.active_repo()
        if not any(entity.is_valid_key(entity_id) for entity in repo.entities):
            raise AppError(422, "invalid_request", "The entity key is malformed.")
        receipt = services.erasure.erase(entity_id, requested_by=caller.subject)
        return receipt.model_dump(mode="json")

    @app.get("/v1/views")
    def list_views(caller: CallerDep) -> dict[str, Any]:
        if not caller.has(SCOPE_REGISTRY):
            raise AppError(403, "forbidden", "The caller may not read the registry.")
        repo = services.registry.active_repo()
        return {
            "views": [
                {
                    "name": view.name,
                    "version": view.version_hash(),
                    "entity": view.entity,
                    "ttl_seconds": view.ttl_seconds,
                    "zero_ingestion_lag": view.zero_ingestion_lag,
                    "owner": view.owner,
                    "features": [
                        {
                            "name": name,
                            "dtype": view.feature(name).dtype.value,
                            "pii": view.feature(name).pii,
                        }
                        for name in view.feature_names
                    ],
                }
                for view in repo.feature_views
            ]
        }

    return app
