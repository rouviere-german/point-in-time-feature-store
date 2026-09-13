"""Redis ACL for the online store.

Two users replace the passwordless ``default`` user (which is switched off):

- ``pitfs_admin`` (deploy-time only) loads the registered Lua scripts;
- the application user may only run the commands the store needs, only on ``fs:v1:*`` keys. It
  cannot ``FLUSHALL``/``FLUSHDB``, touch ``CONFIG``, run ``EVAL`` or ``SCRIPT LOAD`` (so it can only
  ``EVALSHA`` scripts an administrator loaded), list ``KEYS``, or manage ACLs and modules.

Passwords never appear in the file: users are declared with ``#<sha256>`` of the password, which
the operator computes from environment variables at deploy time.
"""

from __future__ import annotations

import hashlib
import re

__all__ = [
    "APP_COMMANDS",
    "DENIED_APP_COMMANDS",
    "password_sha256",
    "render_acl",
    "validate_acl",
]

APP_COMMANDS = (
    "ping",
    "hello",
    "auth",
    "client|setname",
    "select",
    "hgetall",
    "hmget",
    "hset",
    "hdel",
    "del",
    "exists",
    "evalsha",
    "script|exists",
    "multi",
    "exec",
    "discard",
)
DENIED_APP_COMMANDS = (
    "flushall",
    "flushdb",
    "config",
    "eval",
    "eval_ro",
    "script|load",
    "script|flush",
    "function",
    "fcall",
    "keys",
    "scan",
    "acl",
    "module",
    "debug",
    "shutdown",
    "save",
    "bgsave",
    "replicaof",
    "migrate",
)
_USER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def password_sha256(password: str) -> str:
    if len(password) < 16:
        raise ValueError("Redis passwords must be at least 16 characters")
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def render_acl(
    *,
    app_user: str,
    app_password_sha256: str,
    admin_user: str = "pitfs_admin",
    admin_password_sha256: str,
) -> str:
    for user in (app_user, admin_user):
        if not _USER.fullmatch(user):
            raise ValueError("invalid Redis user name")
    for digest in (app_password_sha256, admin_password_sha256):
        if not _SHA256.fullmatch(digest):
            raise ValueError("password digests must be lowercase hex SHA-256")
    if app_user == admin_user:
        raise ValueError("the application and admin users must differ")
    allowed = " ".join(f"+{command}" for command in APP_COMMANDS)
    return (
        "user default off resetpass nocommands resetkeys\n"
        f"user {admin_user} on #{admin_password_sha256} ~* &* +@all\n"
        f"user {app_user} on #{app_password_sha256} resetkeys ~fs:v1:* "
        f"resetchannels -@all {allowed}\n"
    )


def validate_acl(text: str, *, app_user: str) -> list[str]:
    """Return the problems with the application user's rules (empty when they are safe)."""
    problems: list[str] = []
    lines = [
        line.split() for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]
    default = next((parts for parts in lines if parts[:2] == ["user", "default"]), None)
    if default is None or "off" not in default:
        problems.append("the default user must be switched off")
    app = next((parts for parts in lines if parts[:2] == ["user", app_user]), None)
    if app is None:
        return [*problems, f"no rules for application user {app_user}"]
    rules = app[2:]
    if any(rule.startswith(">") for rule in rules):
        problems.append("plaintext passwords are not allowed; use #<sha256>")
    if not any(rule.startswith("#") for rule in rules):
        problems.append("the application user needs a password digest")
    if "-@all" not in rules or rules.index("-@all") > min(
        (i for i, rule in enumerate(rules) if rule.startswith("+")), default=len(rules)
    ):
        problems.append("the application user must start from -@all")
    granted = {rule[1:].lower() for rule in rules if rule.startswith("+")}
    for broad in ("@all", "@dangerous", "@admin", "@scripting", "@keyspace", "@write"):
        if broad in granted or "allcommands" in rules:
            problems.append(f"the application user must not be granted {broad}")
    for command in DENIED_APP_COMMANDS:
        if command in granted or command.split("|")[0] in granted:
            problems.append(f"the application user must not be granted {command}")
    for command in APP_COMMANDS:
        if command not in granted:
            problems.append(f"the application user is missing {command}")
    key_rules = [rule for rule in rules if rule.startswith("~") or rule in {"allkeys", "%RW~*"}]
    if key_rules != ["~fs:v1:*"]:
        problems.append("the application user must be limited to ~fs:v1:* keys")
    return problems
