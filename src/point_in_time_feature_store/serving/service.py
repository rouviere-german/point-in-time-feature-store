"""Online feature reads: permission planning, Redis snapshot reads, freshness, request log."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from point_in_time_feature_store.definitions import FeatureRepo, FeatureView, parse_feature_ref
from point_in_time_feature_store.hashing import EntityHasher
from point_in_time_feature_store.monitoring import FeatureStoreMetrics
from point_in_time_feature_store.offline.oracle import Value
from point_in_time_feature_store.online.store import OnlineStore
from point_in_time_feature_store.registry import FeatureRegistry
from point_in_time_feature_store.serving.auth import Caller
from point_in_time_feature_store.serving.request_log import LoggedRead, RequestLog, logged_values
from point_in_time_feature_store.timeutil import US_PER_SECOND, Clock, from_us, system_clock_us

__all__ = [
    "FeatureAccessDeniedError",
    "InvalidFeatureRequestError",
    "OnlineFeatureService",
    "OnlineResponse",
    "ReadPlan",
]


class FeatureAccessDeniedError(PermissionError):
    """The caller lacks the read scope of at least one requested view."""


class InvalidFeatureRequestError(ValueError):
    """Unknown views or features, or malformed entity keys."""


@dataclass(frozen=True, slots=True)
class ReadPlan:
    views: tuple[tuple[FeatureView, str, tuple[str, ...]], ...]
    omitted: tuple[str, ...]


@dataclass(slots=True)
class OnlineResponse:
    results: list[dict[str, JsonValue]] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)
    watermarks: dict[str, JsonValue] = field(default_factory=dict)
    pii_served: bool = False


def _iso(value: int | None) -> str | None:
    return None if value is None else from_us(value).isoformat()


class OnlineFeatureService:
    def __init__(
        self,
        registry: FeatureRegistry,
        online: OnlineStore,
        hasher: EntityHasher,
        *,
        request_log: RequestLog | None = None,
        metrics: FeatureStoreMetrics | None = None,
        clock: Clock = system_clock_us,
    ) -> None:
        self.registry = registry
        self.online = online
        self.hasher = hasher
        self.request_log = request_log
        self.metrics = metrics
        self.clock = clock
        self.repo: FeatureRepo = registry.active_repo()

    def refresh(self) -> None:
        self.repo = self.registry.active_repo()

    def plan(self, caller: Caller, features: Sequence[str]) -> ReadPlan:
        requested: dict[str, list[str]] = {}
        try:
            for ref in features:
                view_name, feature = parse_feature_ref(ref)
                view = self.repo.view(view_name)
                names = list(view.feature_names) if feature == "*" else [view.feature(feature).name]
                bucket = requested.setdefault(view_name, [])
                bucket.extend(name for name in names if name not in bucket)
        except (KeyError, ValueError):
            raise InvalidFeatureRequestError("unknown or malformed feature reference") from None
        denied = sorted(name for name in requested if not caller.can_read_view(name))
        if denied:
            raise FeatureAccessDeniedError("caller may not read one or more requested views")
        planned: list[tuple[FeatureView, str, tuple[str, ...]]] = []
        omitted: list[str] = []
        for view_name, names in requested.items():
            view = self.repo.view(view_name)
            allowed = [name for name in names if caller.can_read_pii or not view.feature(name).pii]
            omitted += [f"{view_name}:{name}" for name in names if name not in allowed]
            planned.append((view, view.version_hash(), tuple(allowed)))
        return ReadPlan(tuple(planned), tuple(sorted(omitted)))

    def validate_entities(self, plan: ReadPlan, entity_ids: Sequence[str]) -> None:
        for view, _version, _names in plan.views:
            entity = self.repo.entity(view.entity)
            if not all(entity.is_valid_key(entity_id) for entity_id in entity_ids):
                raise InvalidFeatureRequestError("one or more entity keys are malformed")

    def get(
        self,
        caller: Caller,
        entity_ids: Sequence[str],
        features: Sequence[str],
        *,
        request_id: str,
        now_us: int | None = None,
    ) -> OnlineResponse:
        plan = self.plan(caller, features)
        self.validate_entities(plan, entity_ids)
        now = self.clock() if now_us is None else now_us
        started = time.perf_counter()
        read = self.online.read(
            entity_ids, [(view, version) for view, version, _ in plan.views], now
        )
        response = OnlineResponse(omitted=list(plan.omitted))
        logged: list[LoggedRead] = []
        sample = self.request_log is not None and self.request_log.should_sample(request_id)
        erased = self.registry.tombstones() if sample else frozenset()
        for entity_id in entity_ids:
            features_out: dict[str, JsonValue] = {}
            meta: dict[str, JsonValue] = {}
            for view, version, names in plan.views:
                value = read.values[(entity_id, view.name)]
                freshness = (
                    None
                    if value.event_time_us is None
                    else (now - value.event_time_us) / US_PER_SECOND
                )
                meta[view.name] = {
                    "status": value.status,
                    "event_time": _iso(value.event_time_us),
                    "freshness_seconds": freshness,
                }
                for name in names:
                    served: Value = value.values.get(name)
                    features_out[f"{view.name}:{name}"] = served
                    response.pii_served = response.pii_served or (
                        view.feature(name).pii and served is not None
                    )
                if value.status == "ok" and freshness is not None and self.metrics is not None:
                    self.metrics.feature_freshness_seconds.labels(view.name).observe(freshness)
                if sample and self.hasher.entity(entity_id) not in erased:
                    low, high = read.watermarks[view.name]
                    logged.append(
                        LoggedRead(
                            request_id=request_id,
                            ts_us=now,
                            entity_id=entity_id,
                            view=view.name,
                            version=version,
                            watermark_low_us=low,
                            watermark_high_us=high,
                            status=value.status,
                            event_time_us=value.event_time_us,
                            values=logged_values(view, value.values, self.hasher),
                        )
                    )
            response.results.append(
                {"entity_id": entity_id, "features": features_out, "views": meta}
            )
        if self.metrics is not None:
            self.metrics.online_get_latency_seconds.observe(time.perf_counter() - started)
        for view, _version, _names in plan.views:
            low, _high = read.watermarks[view.name]
            response.watermarks[view.name] = _iso(low)
        if logged and self.request_log is not None:
            self.request_log.record(logged)
        return response
