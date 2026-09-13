"""Command line entry point: ``pitfs`` (alias ``point-in-time-feature-store``) or ``python -m``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

__all__ = ["main"]


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn

    from point_in_time_feature_store.core.config import get_settings
    from point_in_time_feature_store.core.logging import configure_logging
    from point_in_time_feature_store.core.tracing import (
        install_global_tracer_provider,
    )
    from point_in_time_feature_store.services import build_services
    from point_in_time_feature_store.serving.api import create_feature_app

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    services = build_services(settings)
    app = create_feature_app(services)
    install_global_tracer_provider(app.state.tracer_provider)
    try:
        uvicorn.run(
            app,
            host=host or settings.host,
            port=port or settings.port,
            log_config=None,
            access_log=False,
            server_header=False,
        )
    finally:
        services.close()
    return 0


def _demo(args: argparse.Namespace) -> int:
    from point_in_time_feature_store.demo import run_demo

    summary = run_demo(
        args.workdir,
        n_customers=args.customers,
        seed=args.seed,
        online_requests=args.requests,
        redis_url=args.redis_url,
    )
    print(json.dumps({"demo": "point-in-time-feature-store", **summary}, indent=2, sort_keys=True))
    parity = summary.get("parity")
    erasure = summary.get("erasure")
    ok = (
        isinstance(parity, dict)
        and parity.get("mismatches") == 0
        and isinstance(erasure, dict)
        and erasure.get("purge_clean") is True
    )
    return 0 if ok else 1


def _registry(path: Path) -> object:
    from point_in_time_feature_store.registry import FeatureRegistry

    return FeatureRegistry(path)


def _plan_or_apply(args: argparse.Namespace) -> int:
    from point_in_time_feature_store.definitions import FeatureRepo
    from point_in_time_feature_store.registry import (
        ApprovalRequiredError,
        FeatureRegistry,
        PlanMismatchError,
    )

    repo = FeatureRepo.from_json(args.definitions.read_text(encoding="utf-8"))
    registry = FeatureRegistry(args.registry)
    try:
        if args.command == "plan":
            print(registry.plan(repo).render())
            return 0
        try:
            plan = registry.apply(
                repo, actor=args.actor, approve=args.approve, plan_hash=args.plan_hash
            )
        except (ApprovalRequiredError, PlanMismatchError) as exc:
            print(f"refused: {exc}", file=sys.stderr)
            print(registry.plan(repo).render(), file=sys.stderr)
            return 2
        print(plan.render())
        return 0
    finally:
        registry.close()


def _export_definitions(args: argparse.Namespace) -> int:
    from point_in_time_feature_store.offline.synthetic import churn_feature_repo

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(churn_feature_repo().to_json(), encoding="utf-8")
    return 0


def _catalog(args: argparse.Namespace) -> int:
    from point_in_time_feature_store.catalog import render_catalog
    from point_in_time_feature_store.definitions import FeatureRepo
    from point_in_time_feature_store.offline.synthetic import churn_feature_repo

    repo = (
        FeatureRepo.from_json(args.definitions.read_text(encoding="utf-8"))
        if args.definitions is not None
        else churn_feature_repo()
    )
    text = render_catalog(repo)
    if args.out is None:
        print(text, end="")
    else:
        args.out.write_text(text, encoding="utf-8")
    return 0


def _generate_data(args: argparse.Namespace) -> int:
    from point_in_time_feature_store.core.config import get_settings
    from point_in_time_feature_store.offline.synthetic import (
        SyntheticConfig,
        generate_dataset,
    )
    from point_in_time_feature_store.services import build_services

    services = build_services(get_settings())
    try:
        if not services.registry.active_repo().sources:
            print("apply definitions first (pitfs apply)", file=sys.stderr)
            return 2
        dataset = generate_dataset(SyntheticConfig(n_customers=args.customers, seed=args.seed))
        reports = {
            name: services.ingest(name, frame).rows_written
            for name, frame in dataset.sources.items()
        }
        print(json.dumps({"rows_written": reports}, sort_keys=True))
        return 0
    finally:
        services.close()


def _materialize(args: argparse.Namespace) -> int:
    from point_in_time_feature_store.core.config import get_settings
    from point_in_time_feature_store.services import build_services
    from point_in_time_feature_store.timeutil import to_us

    services = build_services(get_settings())
    try:
        up_to = None if args.up_to is None else to_us(datetime.fromisoformat(args.up_to))
        views = (
            [view.name for view in services.registry.active_repo().feature_views]
            if args.view is None
            else [args.view]
        )
        out = {}
        for name in views:
            report = services.materializer.materialize(name, up_to_us=up_to)
            out[name] = {
                "watermark_us": report.watermark_us,
                "written": report.written,
                "skipped": report.skipped,
            }
        print(json.dumps(out, sort_keys=True))
        return 0
    finally:
        services.close()


def _verify_receipt(args: argparse.Namespace) -> int:
    from point_in_time_feature_store.erasure import ErasureReceipt, verify_receipt
    from point_in_time_feature_store.serving.auth import load_public_key

    receipt = ErasureReceipt.model_validate_json(args.receipt.read_text(encoding="utf-8"))
    valid = verify_receipt(receipt, load_public_key(args.public_key.read_bytes()))
    print("receipt signature valid" if valid else "receipt signature INVALID")
    return 0 if valid else 1


def _verify_chains(args: argparse.Namespace) -> int:
    from point_in_time_feature_store.registry import FeatureRegistry

    registry = FeatureRegistry(args.registry)
    try:
        results = registry.verify_chains()
    finally:
        registry.close()
    print(
        json.dumps(
            {
                name: {"ok": item.ok, "entries": item.entries, "head": item.head}
                for name, item in results.items()
            },
            sort_keys=True,
        )
    )
    return 0 if all(item.ok for item in results.values()) else 1


def _generate_keys(args: argparse.Namespace) -> int:
    from point_in_time_feature_store.serving.auth import generate_keypair_pem

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("jwt", "receipt"):
        private, public = generate_keypair_pem()
        private_path = args.out_dir / f"{name}_private.pem"
        private_path.write_bytes(private)
        os.chmod(private_path, 0o600)
        (args.out_dir / f"{name}_public.pem").write_bytes(public)
    print(f"wrote key pairs to {args.out_dir}; keep *_private.pem out of the service container")
    return 0


def _mint_token(args: argparse.Namespace) -> int:
    from point_in_time_feature_store.serving.auth import (
        load_private_key,
        mint_token,
    )

    token = mint_token(
        load_private_key(args.private_key.read_bytes()),
        subject=args.subject,
        scopes=args.scope,
        issuer=args.issuer,
        audience=args.audience,
        ttl_seconds=args.ttl,
    )
    print(token)
    return 0


def _redis_acl(args: argparse.Namespace) -> int:
    from point_in_time_feature_store.online.acl import (
        password_sha256,
        render_acl,
        validate_acl,
    )

    app_password = os.environ.get("APP_REDIS_PASSWORD", "")
    admin_password = os.environ.get("REDIS_ADMIN_PASSWORD", "")
    if not app_password or not admin_password:
        print("set APP_REDIS_PASSWORD and REDIS_ADMIN_PASSWORD", file=sys.stderr)
        return 2
    text = render_acl(
        app_user=args.app_user,
        app_password_sha256=password_sha256(app_password),
        admin_password_sha256=password_sha256(admin_password),
    )
    problems = validate_acl(text, app_user=args.app_user)
    if problems:  # pragma: no cover - render_acl output always validates
        print("\n".join(problems), file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    return 0


def _load_scripts(args: argparse.Namespace) -> int:
    del args
    import redis

    from point_in_time_feature_store.online.store import SCRIPTS, OnlineStore

    url = os.environ.get("REDIS_ADMIN_URL", "")
    if not url:
        print("set REDIS_ADMIN_URL (an administrative Redis connection)", file=sys.stderr)
        return 2
    OnlineStore(redis.Redis.from_url(url)).load_scripts()
    print(json.dumps({name: script.sha1 for name, script in SCRIPTS.items()}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - one subparser per command
    parser = argparse.ArgumentParser(prog="pitfs")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("generate-api-key", help="print a new API key (for /metrics) and its SHA-256")

    demo = sub.add_parser("demo", help="run the offline end-to-end demo")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))
    demo.add_argument("--customers", type=int, default=5_000)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--requests", type=int, default=1_000)
    demo.add_argument("--redis-url", default=None)

    for name in ("plan", "apply"):
        command = sub.add_parser(name, help=f"{name} feature definitions")
        command.add_argument("--definitions", type=Path, required=True)
        command.add_argument("--registry", type=Path, default=Path("var/registry.sqlite"))
        if name == "apply":
            command.add_argument("--actor", required=True)
            command.add_argument("--approve", action="store_true")
            command.add_argument("--plan-hash", default=None)

    export = sub.add_parser(
        "export-definitions", help="write the bundled churn definitions as JSON"
    )
    export.add_argument("--out", type=Path, required=True)

    catalog = sub.add_parser("catalog", help="render FEATURE_CATALOG.md")
    catalog.add_argument("--definitions", type=Path, default=None)
    catalog.add_argument("--out", type=Path, default=None)

    generate = sub.add_parser(
        "generate-data", help="generate synthetic sources into the offline store"
    )
    generate.add_argument("--customers", type=int, default=14_000)
    generate.add_argument("--seed", type=int, default=0)

    materialize = sub.add_parser("materialize", help="materialize views into Redis")
    materialize.add_argument("--view", default=None)
    materialize.add_argument("--up-to", default=None, help="ISO timestamp with timezone")

    receipt = sub.add_parser("verify-receipt", help="verify an erasure receipt signature")
    receipt.add_argument("receipt", type=Path)
    receipt.add_argument("--public-key", type=Path, required=True)

    chains = sub.add_parser("verify-chains", help="verify the registry's hash chains")
    chains.add_argument("--registry", type=Path, default=Path("var/registry.sqlite"))

    keys = sub.add_parser("generate-keys", help="generate JWT and receipt Ed25519 key pairs")
    keys.add_argument("--out-dir", type=Path, required=True)

    token = sub.add_parser("mint-token", help="mint a JWT (development issuer)")
    token.add_argument("--private-key", type=Path, required=True)
    token.add_argument("--subject", required=True)
    token.add_argument("--scope", action="append", default=[])
    token.add_argument("--issuer", default="pitfs-dev-issuer")
    token.add_argument("--audience", default="point-in-time-feature-store")
    token.add_argument("--ttl", type=int, default=900)

    sub.add_parser("load-scripts", help="load the registered Lua scripts (REDIS_ADMIN_URL)")

    acl = sub.add_parser("redis-acl", help="render the Redis ACL file from password env vars")
    acl.add_argument("--out", type=Path, required=True)
    acl.add_argument("--app-user", default="pitfs_app")

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port)
    if args.command == "generate-api-key":
        from point_in_time_feature_store.core.security import generate_api_key

        raw, digest = generate_api_key()
        print(json.dumps({"api_key": raw, "api_key_sha256": digest}, indent=2))
        print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
        return 0
    if args.command != "demo":
        from point_in_time_feature_store.core.logging import configure_logging

        configure_logging("WARNING", json_output=True, stream=sys.stderr)
    handlers = {
        "demo": _demo,
        "plan": _plan_or_apply,
        "apply": _plan_or_apply,
        "export-definitions": _export_definitions,
        "catalog": _catalog,
        "generate-data": _generate_data,
        "materialize": _materialize,
        "verify-receipt": _verify_receipt,
        "verify-chains": _verify_chains,
        "generate-keys": _generate_keys,
        "mint-token": _mint_token,
        "redis-acl": _redis_acl,
        "load-scripts": _load_scripts,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
