"""Benchmarks for the project's evaluation metrics, with CI regression thresholds.

Usage:
  uv run python bench/bench_baseline.py                 # run all, write bench/results/latest.json
  uv run python bench/bench_baseline.py --check         # also enforce bench/thresholds.json
  uv run python bench/bench_baseline.py --only leakage_5_seeds --check

Everything runs offline on CPU. The online benchmark uses fakeredis (in-process, no network)
unless APP_BENCH_REDIS_URL points at a real Redis, which CI provides as a service container.

Benchmarks:
  retrieval_1m_rows            point-in-time retrieval of every feature over ~1M source rows
  online_get_50_features       latency of one online read of 50 features across 5 views
  parity_10k_requests          online vs offline recomputation for 10k logged requests
  leakage_5_seeds              offline-vs-holdout AUC gap, naive join vs point-in-time join
  materialization_idempotency  incremental == one-shot == rerun store digests
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import polars as pl
import redis

from point_in_time_feature_store.definitions import (
    EntityDefinition,
    FeatureDefinition,
    FeatureRepo,
    FeatureView,
    SourceColumn,
    SourceDefinition,
)
from point_in_time_feature_store.devtools import build_local_stack, fake_redis
from point_in_time_feature_store.dsl import DType, col
from point_in_time_feature_store.hashing import EntityHasher
from point_in_time_feature_store.offline.retrieval import (
    DuckDBEngine,
    HistoricalRetriever,
    normalize_spine,
)
from point_in_time_feature_store.offline.store import OfflineStore
from point_in_time_feature_store.offline.synthetic import (
    SyntheticConfig,
    churn_feature_repo,
    generate_dataset,
    to_timestamp,
)
from point_in_time_feature_store.online.materialize import Materializer
from point_in_time_feature_store.online.store import OnlineRow, OnlineStore
from point_in_time_feature_store.registry import FeatureRegistry
from point_in_time_feature_store.serving.auth import Caller
from point_in_time_feature_store.serving.service import OnlineFeatureService
from point_in_time_feature_store.timeutil import US_PER_DAY
from point_in_time_feature_store.training import run_leakage_experiment

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "latest.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"
ALL_FEATURES = ["customer_profile:*", "transaction_stats:*", "session_stats:*"]
Result = dict[str, float | int | str]


def _seconds(start: float) -> float:
    return round(time.perf_counter() - start, 3)


def bench_retrieval(workdir: Path) -> Result:
    started = time.perf_counter()
    dataset = generate_dataset(SyntheticConfig(n_customers=14_000, seed=0))
    generate_s = _seconds(started)
    repo = churn_feature_repo()
    store = OfflineStore(workdir / "offline")
    started = time.perf_counter()
    for name, frame in dataset.sources.items():
        store.append(repo.source(name), repo.entity("customer"), frame)
    ingest_s = _seconds(started)
    rng = random.Random(0)
    ids = dataset.truth["entity_id"].to_list()
    n_spine = 200_000
    spine = pl.DataFrame(
        {
            "entity_id": [rng.choice(ids) for _ in range(n_spine)],
            "label_time": to_timestamp(
                pl.Series([dataset.config.day_us(rng.uniform(30, 420)) for _ in range(n_spine)])
            ),
        }
    )
    engine = DuckDBEngine(threads=2, memory_limit="1GB", temp_directory=workdir / "duckdb")
    try:
        retriever = HistoricalRetriever(store, repo, engine)
        started = time.perf_counter()
        result = retriever.get_historical_features(spine, ALL_FEATURES)
        retrieval_s = _seconds(started)
        view = repo.view("session_stats")
        rows = retriever.feature_rows(view)
        normalized = normalize_spine(spine)
        started = time.perf_counter()
        fast, _ = retriever.join_view(view, normalized, rows, path="asof")
        asof_s = _seconds(started)
        started = time.perf_counter()
        slow, _ = retriever.join_view(view, normalized, rows, path="range_join")
        range_s = _seconds(started)
        operators = engine.explain_range_join(
            normalized.head(1_000), rows.head(1_000), view.feature_names
        )
        return {
            "source_rows": dataset.total_rows,
            "spine_rows": n_spine,
            "feature_columns": len([c for c in result.frame.columns if "__" in c]),
            "generate_s": generate_s,
            "ingest_s": ingest_s,
            "retrieval_s": retrieval_s,
            "spine_rows_per_s": round(n_spine / retrieval_s),
            "session_asof_s": asof_s,
            "session_range_join_s": range_s,
            "asof_equals_range_join": int(fast.equals(slow)),
            "range_join_operators": ",".join(operators),
        }
    finally:
        engine.close()


def _bench_repo(views: int, features: int) -> FeatureRepo:
    source = SourceDefinition(
        name="bench_events",
        entity="customer",
        columns=tuple(SourceColumn(name=f"v{j}", dtype=DType.FLOAT64) for j in range(features)),
    )
    return FeatureRepo(
        entities=(EntityDefinition(name="customer"),),
        sources=(source,),
        feature_views=tuple(
            FeatureView(
                name=f"bench_view_{i}",
                entity="customer",
                source="bench_events",
                owner="bench",
                ttl_seconds=30 * 86_400,
                features=tuple(
                    FeatureDefinition(
                        name=f"f{j}", dtype=DType.FLOAT64, expression=(col(f"v{j}") * (i + 1)).node
                    )
                    for j in range(features)
                ),
            )
            for i in range(views)
        ),
    )


def bench_online(workdir: Path) -> Result:
    url = os.environ.get("APP_BENCH_REDIS_URL")
    client = redis.Redis.from_url(url) if url else fake_redis()
    backend = "redis" if url else "fakeredis (in-process)"
    registry = FeatureRegistry(workdir / "registry.sqlite")
    try:
        repo = _bench_repo(views=5, features=10)
        registry.apply(repo, actor="bench")
        online = OnlineStore(client)
        online.load_scripts()
        now = 1_760_000_000_000_000
        rng = random.Random(1)
        entities = [f"b{i:06d}" for i in range(2_000)]
        for view in repo.feature_views:
            rows = [
                OnlineRow(
                    entity,
                    now - US_PER_DAY,
                    now - US_PER_DAY,
                    tuple(rng.random() for _ in range(10)),
                )
                for entity in entities
            ]
            online.write_rows(view, view.version_hash(), rows)
        service = OnlineFeatureService(
            registry, online, EntityHasher.ephemeral(), clock=lambda: now
        )
        caller = Caller("bench", frozenset({"features:read:*"}))
        features = [f"{view.name}:*" for view in repo.feature_views]
        for _ in range(300):
            service.get(caller, [rng.choice(entities)], features, request_id="warmup")
        samples = []
        for _ in range(5_000):
            entity = rng.choice(entities)
            started = time.perf_counter_ns()
            response = service.get(caller, [entity], features, request_id="bench")
            samples.append((time.perf_counter_ns() - started) / 1e6)
        features_out = response.results[0]["features"]
        served = (
            len([v for v in features_out.values() if v is not None])
            if isinstance(features_out, dict)
            else 0
        )
        quantiles = statistics.quantiles(samples, n=100)
        if url:
            for view in repo.feature_views:
                online.delete_entity("unused", {view.name: [view.version_hash()]})
        return {
            "backend": backend,
            "features_per_read": served,
            "reads": len(samples),
            "p50_ms": round(quantiles[49], 3),
            "p95_ms": round(quantiles[94], 3),
            "p99_ms": round(quantiles[98], 3),
        }
    finally:
        registry.close()


def bench_parity(workdir: Path) -> Result:
    dataset = generate_dataset(SyntheticConfig(n_customers=3_000, seed=5))
    config = dataset.config
    marks = [config.day_us(day) for day in (330, 360, 390, config.horizon_days)]
    stack = build_local_stack(workdir / "stack", now_us=marks[0])
    services = stack.services
    try:
        services.registry.apply(churn_feature_repo(), actor="bench")
        services.online_service.refresh()
        caller = Caller("bench", frozenset({"features:read:*", "pii:read"}))
        entities = dataset.truth["entity_id"].to_list()
        rng = random.Random(7)
        requests = 0
        previous = None

        def serve(count: int, low: int, high: int) -> None:
            nonlocal requests
            for _ in range(count):
                stack.clock.now_us = rng.randrange(low, high)
                services.online_service.get(
                    caller, [rng.choice(entities)], ALL_FEATURES, request_id=uuid.uuid4().hex
                )
                requests += 1

        for index, mark in enumerate(marks):
            for name, frame in dataset.sources.items():
                ingested = pl.col("ingested_at").dt.epoch("us")
                window = (
                    ingested <= mark
                    if previous is None
                    else (ingested > previous) & (ingested <= mark)
                )
                services.ingest(name, frame.filter(window))
            stack.clock.now_us = mark
            for view in services.registry.active_repo().feature_views:
                hook = (
                    (lambda at=mark: serve(250, at, at + 1))
                    if view.name == "transaction_stats"
                    else None
                )
                services.materializer.materialize(view.name, up_to_us=mark, before_commit=hook)
            end = marks[index + 1] if index + 1 < len(marks) else mark + 3 * US_PER_DAY
            serve(2_250, mark, end)
            previous = mark
        logged = services.request_log.count()
        started = time.perf_counter()
        report = services.skew_checker().check(sample_size=logged, seed=0)
        inflight = sum(
            1
            for read in services.request_log.sample(logged, seed=0)
            if read.watermark_low_us != read.watermark_high_us
        )
        return {
            "requests": requests,
            "logged_reads": logged,
            "inflight_reads": inflight,
            "checked": report.checked,
            "mismatches": report.mismatches,
            "check_s": _seconds(started),
        }
    finally:
        stack.close()


def bench_leakage(workdir: Path) -> Result:
    repo = churn_feature_repo()
    per_seed = []
    for seed in range(5):
        dataset = generate_dataset(SyntheticConfig(n_customers=10_000, seed=100 + seed))
        store = OfflineStore(workdir / f"seed{seed}")
        for name, frame in dataset.sources.items():
            store.append(repo.source(name), repo.entity("customer"), frame)
        engine = DuckDBEngine(threads=2, memory_limit="1GB")
        try:
            result = run_leakage_experiment(
                dataset, HistoricalRetriever(store, repo, engine), seed=seed
            )
        finally:
            engine.close()
        per_seed.append(result)

    def mean(arm: str, field: str) -> float:
        return round(statistics.fmean(getattr(r.arms[arm], field) for r in per_seed), 4)

    output: Result = {
        "seeds": len(per_seed),
        "customers_per_seed": 10_000,
        "holdout_rows_mean": round(statistics.fmean(r.rows["holdout"] for r in per_seed)),
    }

    def ci_mean(arm: str, bound: int) -> float:
        return round(statistics.fmean(r.arms[arm].gap_ci95[bound] for r in per_seed), 4)

    for arm in ("naive_latest", "event_time_asof", "pit"):
        output[f"{arm}_offline_auc_mean"] = mean(arm, "offline_auc")
        output[f"{arm}_holdout_auc_mean"] = mean(arm, "holdout_auc")
        output[f"{arm}_period_control_auc_mean"] = mean(arm, "period_control_auc")
        output[f"{arm}_gap_mean"] = mean(arm, "gap")
        output[f"{arm}_gap_ci95_low_mean"] = ci_mean(arm, 0)
        output[f"{arm}_gap_ci95_high_mean"] = ci_mean(arm, 1)
    output["naive_gap_min"] = round(min(r.arms["naive_latest"].gap for r in per_seed), 4)
    output["naive_gap_ci95_low_min"] = round(
        min(r.arms["naive_latest"].gap_ci95[0] for r in per_seed), 4
    )
    output["pit_gap_max"] = round(max(r.arms["pit"].gap for r in per_seed), 4)
    output["pit_abs_gap_mean"] = round(
        statistics.fmean(abs(r.arms["pit"].gap) for r in per_seed), 4
    )
    # Seeds whose point-in-time gap interval contains 0 (no gap distinguishable from noise).
    output["pit_gap_ci95_contains_zero_seeds"] = sum(
        1 for r in per_seed if r.arms["pit"].gap_ci95[0] <= 0.0 <= r.arms["pit"].gap_ci95[1]
    )
    output["per_seed"] = json.dumps(
        [
            {
                arm: {
                    "gap": round(item.gap, 4),
                    "ci95": [round(item.gap_ci95[0], 4), round(item.gap_ci95[1], 4)],
                    "period_control_auc": round(item.period_control_auc, 4),
                }
                for arm, item in r.arms.items()
            }
            for r in per_seed
        ],
        separators=(",", ":"),
    )
    return output


def bench_materialization(workdir: Path) -> Result:
    dataset = generate_dataset(SyntheticConfig(n_customers=14_000, seed=3))
    config = dataset.config
    repo = churn_feature_repo()
    store = OfflineStore(workdir / "offline")
    for name, frame in dataset.sources.items():
        store.append(repo.source(name), repo.entity("customer"), frame)
    engine = DuckDBEngine(threads=2, memory_limit="1GB")
    hasher = EntityHasher.ephemeral()
    marks = [config.day_us(day) for day in (100, 200, 300, config.horizon_days)]

    def run(tag: str, steps: list[int], online: OnlineStore, overlap_us: int = 0) -> float:
        registry = FeatureRegistry(workdir / f"{tag}.sqlite", clock=lambda: marks[-1])
        try:
            registry.apply(repo, actor="bench")
            materializer = Materializer(
                registry, store, online, engine, hasher, clock=lambda: marks[-1]
            )
            started = time.perf_counter()
            for mark in steps:
                for view in repo.feature_views:
                    materializer.materialize(view.name, up_to_us=mark, overlap_us=overlap_us)
            return _seconds(started)
        finally:
            registry.close()

    try:
        incremental = OnlineStore(fake_redis())
        incremental.load_scripts()
        incremental_s = run("incremental", marks, incremental)
        digest = incremental.state_digest()
        one_shot = OnlineStore(fake_redis())
        one_shot.load_scripts()
        one_shot_s = run("one_shot", marks[-1:], one_shot)
        rerun_s = run("rerun", marks, incremental, overlap_us=500 * US_PER_DAY)
        return {
            "source_rows": dataset.total_rows,
            "incremental_4_runs_s": incremental_s,
            "one_shot_s": one_shot_s,
            "full_rerun_s": rerun_s,
            "incremental_equals_one_shot": int(one_shot.state_digest() == digest),
            "rerun_identical": int(incremental.state_digest() == digest),
        }
    finally:
        engine.close()


BENCHES: dict[str, Callable[[Path], Result]] = {
    "retrieval_1m_rows": bench_retrieval,
    "online_get_50_features": bench_online,
    "parity_10k_requests": bench_parity,
    "leakage_5_seeds": bench_leakage,
    "materialization_idempotency": bench_materialization,
}


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def check_thresholds(results: dict[str, Result]) -> list[str]:
    thresholds: dict[str, dict[str, dict[str, float]]] = json.loads(
        THRESHOLDS.read_text(encoding="utf-8")
    )
    problems = []
    for name, limits in thresholds.items():
        if name not in results:
            continue
        for metric, bound in limits.items():
            value = float(results[name][metric])
            if "max" in bound and value > bound["max"]:
                problems.append(f"{name}.{metric} = {value} exceeds max {bound['max']}")
            if "min" in bound and value < bound["min"]:
                problems.append(f"{name}.{metric} = {value} is below min {bound['min']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="feature store benchmarks")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    parser.add_argument("--only", choices=sorted(BENCHES), action="append", default=None)
    args = parser.parse_args(argv)
    selected = args.only or list(BENCHES)
    results: dict[str, Result] = {}
    for name in selected:
        with tempfile.TemporaryDirectory() as tmp:
            started = time.perf_counter()
            results[name] = BENCHES[name](Path(tmp))
            results[name]["wall_s"] = _seconds(started)
        print(f"## {name}")
        for metric, value in results[name].items():
            print(f"- {metric}: {value}")
        sys.stdout.flush()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps({"machine": machine(), "results": results}, indent=2) + "\n", encoding="utf-8"
    )
    if args.check:
        problems = check_thresholds(results)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
