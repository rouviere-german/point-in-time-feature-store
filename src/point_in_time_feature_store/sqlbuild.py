"""The one place where dynamic SQL text is assembled.

Only identifiers are ever interpolated, and only after validation against a strict pattern; every
value (TTLs, timestamps, entity ids, paths) is passed as a bound parameter. There is no code path
that accepts SQL from users: transformations are expression trees (:mod:`dsl`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

__all__ = ["SqlIdentifierError", "join_idents", "quote_ident"]

_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,127}$")


class SqlIdentifierError(ValueError):
    """An identifier did not match the allowed pattern."""


def quote_ident(name: str) -> str:
    """Validate ``name`` and return it double-quoted."""
    if not _IDENT.fullmatch(name):
        raise SqlIdentifierError("invalid SQL identifier")
    return '"' + name + '"'


def join_idents(names: Iterable[str], *, prefix: str | None = None, suffix: str = "") -> str:
    """Comma-separated quoted identifiers, optionally qualified (``p."a"``) and suffixed."""
    qualifier = "" if prefix is None else quote_ident(prefix) + "."
    return ", ".join(qualifier + quote_ident(name) + suffix for name in names)
