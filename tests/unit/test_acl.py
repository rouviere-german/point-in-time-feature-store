from __future__ import annotations

import os

import pytest
import redis
from redis.exceptions import NoPermissionError, ResponseError

from point_in_time_feature_store.online.acl import (
    APP_COMMANDS,
    password_sha256,
    render_acl,
    validate_acl,
)
from point_in_time_feature_store.online.store import SCRIPTS, OnlineStore

pytestmark = pytest.mark.unit

APP_PASSWORD = "app-password-for-tests-0123456789"
ADMIN_PASSWORD = "admin-password-for-tests-0123456789"


def rendered() -> str:
    return render_acl(
        app_user="pitfs_app",
        app_password_sha256=password_sha256(APP_PASSWORD),
        admin_password_sha256=password_sha256(ADMIN_PASSWORD),
    )


def test_rendered_acl_is_least_privilege() -> None:
    text = rendered()
    assert validate_acl(text, app_user="pitfs_app") == []
    assert APP_PASSWORD not in text
    assert ADMIN_PASSWORD not in text


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (lambda t: t.replace("-@all", "+@all"), "@all"),
        (lambda t: t.replace("+evalsha", "+evalsha +eval"), "eval"),
        (lambda t: t.replace("+evalsha", "+evalsha +flushall"), "flushall"),
        (lambda t: t.replace("+evalsha", "+evalsha +config"), "config"),
        (lambda t: t.replace("+evalsha", "+evalsha +script|load"), "script|load"),
        (lambda t: t.replace("+evalsha", "+evalsha +keys"), "keys"),
        (lambda t: t.replace("~fs:v1:*", "~*"), "fs:v1"),
        (lambda t: t.replace("user default off", "user default on nopass"), "default user"),
        (lambda t: t.replace(" +hmget", ""), "missing hmget"),
        (
            lambda t: "\n".join(
                line.replace("#", ">", 1) if "pitfs_app" in line else line
                for line in t.splitlines()
            ),
            "plaintext",
        ),
        (
            lambda t: "\n".join(line for line in t.splitlines() if "pitfs_app" not in line),
            "no rules",
        ),
    ],
)
def test_validation_catches_every_privilege_widening(edit: object, problem: str) -> None:
    problems = validate_acl(edit(rendered()), app_user="pitfs_app")  # type: ignore[operator]
    assert any(problem in item for item in problems), problems


def test_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="16 characters"):
        password_sha256("short")
    with pytest.raises(ValueError, match="user name"):
        render_acl(
            app_user="bad user", app_password_sha256="a" * 64, admin_password_sha256="b" * 64
        )
    with pytest.raises(ValueError, match="SHA-256"):
        render_acl(app_user="app", app_password_sha256="nope", admin_password_sha256="b" * 64)
    with pytest.raises(ValueError, match="differ"):
        render_acl(
            app_user="same",
            admin_user="same",
            app_password_sha256="a" * 64,
            admin_password_sha256="b" * 64,
        )


@pytest.mark.redis
@pytest.mark.skipif(
    "APP_TEST_REDIS_URL" not in os.environ, reason="needs a real Redis (CI job redis)"
)
def test_real_redis_enforces_the_acl() -> None:
    admin = redis.Redis.from_url(os.environ["APP_TEST_REDIS_URL"])
    app_line = next(line for line in rendered().splitlines() if line.startswith("user pitfs_app"))
    rules = app_line.split()[2:]
    admin.execute_command("ACL", "SETUSER", "pitfs_app", "reset", *rules)  # type: ignore[no-untyped-call]
    OnlineStore(admin).load_scripts()
    host = admin.connection_pool.connection_kwargs.get("host", "127.0.0.1")
    port = admin.connection_pool.connection_kwargs.get("port", 6379)
    app = redis.Redis(host=host, port=port, username="pitfs_app", password=APP_PASSWORD)
    try:
        assert app.ping()
        for command in (
            app.flushall,
            lambda: app.config_get("*"),
            lambda: app.eval("return 1", 0),
            lambda: app.script_load("return 1"),
            lambda: app.keys("*"),
            lambda: app.hset("outside:prefix", "a", "b"),
        ):
            with pytest.raises((NoPermissionError, ResponseError)):
                command()
        sha = SCRIPTS["conditional_write"].sha1
        assert app.evalsha(sha, 1, "fs:v1:acl:test:c1", "1", "1", "d", "", "f", "v") == 1
        assert set(APP_COMMANDS) >= {"evalsha", "hgetall"}
    finally:
        admin.delete("fs:v1:acl:test:c1")
        admin.execute_command("ACL", "DELUSER", "pitfs_app")  # type: ignore[no-untyped-call]
