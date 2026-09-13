"""Redis online store: one hash per (view, version, entity), written only through a Lua script.

Key layout (entity keys are validated against the entity's key preset, so they contain no ``:``)::

    fs:v1:<view>:<version[:16]>:<entity_id>   __event_time, __ingested_at, __digest, <feature>...
    fs:v1:<view>:<version[:16]>:__meta__      watermark, inflight

``conditional_write`` replaces a row only when the incoming row is newer by ``event_time``, then by
``ingested_at``. Identical rows are no-ops, so reruns are idempotent and out-of-order batches are
safe. When both timestamps tie but values differ, the script returns ``2`` and the client resolves
the tie with the same value ordering the offline store uses, then overwrites with a
compare-and-set on the stored digest. The client refuses to call any script that is not in
:data:`SCRIPTS` and never falls back to ``EVAL``; scripts are loaded by an administrative
connection (the application ACL user cannot ``SCRIPT LOAD`` or ``EVAL``).

Reads run in ``MULTI``/``EXEC``, so the rows and the watermark metadata come from one consistent
snapshot (the skew checker relies on that).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import redis
from redis.exceptions import NoScriptError, ResponseError

from point_in_time_feature_store.core.canonical import canonical_json, sha256_hex
from point_in_time_feature_store.definitions import FeatureView
from point_in_time_feature_store.dsl import DType
from point_in_time_feature_store.offline.oracle import Value, value_sort_key
from point_in_time_feature_store.timeutil import seconds_to_us

__all__ = [
    "KEY_PREFIX",
    "SCRIPTS",
    "OnlineReadResult",
    "OnlineRow",
    "OnlineStore",
    "OnlineValue",
    "RegisteredScript",
    "ScriptNotRegisteredError",
    "ScriptsNotLoadedError",
    "WriteStats",
    "decode_value",
    "encode_value",
    "meta_key",
    "row_key",
]

KEY_PREFIX = "fs:v1"
_MAX_TIE_RETRIES = 5

CONDITIONAL_WRITE_LUA = """
local current = redis.call('HMGET', KEYS[1], '__event_time', '__ingested_at', '__digest')
local new_event = tonumber(ARGV[1])
local new_ingest = tonumber(ARGV[2])
if current[1] then
  local cur_event = tonumber(current[1])
  local cur_ingest = tonumber(current[2])
  if cur_event > new_event then return 0 end
  if cur_event == new_event then
    if cur_ingest > new_ingest then return 0 end
    if cur_ingest == new_ingest then
      if current[3] == ARGV[3] then return 0 end
      if ARGV[4] ~= current[3] then return 2 end
    end
  end
end
redis.call('DEL', KEYS[1])
redis.call('HSET', KEYS[1], '__event_time', ARGV[1], '__ingested_at', ARGV[2],
  '__digest', ARGV[3], unpack(ARGV, 5))
return 1
"""


@dataclass(frozen=True, slots=True)
class RegisteredScript:
    name: str
    source: str

    @property
    def sha1(self) -> str:
        # Redis addresses scripts by SHA-1; this is an identifier, not a security boundary.
        return hashlib.sha1(self.source.encode("utf-8"), usedforsecurity=False).hexdigest()


SCRIPTS: Mapping[str, RegisteredScript] = {
    "conditional_write": RegisteredScript("conditional_write", CONDITIONAL_WRITE_LUA),
}


class ScriptNotRegisteredError(RuntimeError):
    """Code asked to run a Lua script that is not in the allowlist."""


class ScriptsNotLoadedError(RuntimeError):
    """Redis does not have the registered scripts (for example after a restart)."""


def row_key(view: str, version: str, entity_id: str) -> str:
    return f"{KEY_PREFIX}:{view}:{version[:16]}:{entity_id}"


def meta_key(view: str, version: str) -> str:
    return f"{KEY_PREFIX}:{view}:{version[:16]}:__meta__"


def encode_value(dtype: DType, value: Value) -> str | None:
    if value is None:
        return None
    if dtype is DType.BOOL:
        return "1" if value else "0"
    if dtype is DType.INT64:
        return str(int(value))
    if dtype is DType.FLOAT64:
        return repr(float(value))
    return str(value)


def decode_value(dtype: DType, raw: bytes | str | None) -> Value:
    if raw is None:
        return None
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    if dtype is DType.BOOL:
        return text == "1"
    if dtype is DType.INT64:
        return int(text)
    if dtype is DType.FLOAT64:
        return float(text)
    return text


@dataclass(frozen=True, slots=True)
class OnlineRow:
    entity_id: str
    event_time_us: int
    ingested_at_us: int
    values: tuple[Value, ...]  # aligned with FeatureView.feature_names


@dataclass(slots=True)
class WriteStats:
    written: int = 0
    skipped: int = 0
    ties_resolved: int = 0


@dataclass(frozen=True, slots=True)
class OnlineValue:
    entity_id: str
    view: str
    status: Literal["ok", "missing", "expired"]
    event_time_us: int | None
    ingested_at_us: int | None
    values: dict[str, Value]


@dataclass(frozen=True, slots=True)
class OnlineReadResult:
    values: dict[tuple[str, str], OnlineValue]
    # view -> (committed watermark, in-flight watermark or the committed one)
    watermarks: dict[str, tuple[int | None, int | None]]


def _bytes_map(value: object) -> dict[bytes, bytes]:
    """Normalize a redis-py hash reply (bytes keys and values) for typed access."""
    if not isinstance(value, dict):
        return {}
    return {
        (k if isinstance(k, bytes) else str(k).encode("utf-8")): (
            v if isinstance(v, bytes) else str(v).encode("utf-8")
        )
        for k, v in value.items()
    }


def _digest(encoded: Sequence[str | None]) -> str:
    return sha256_hex(canonical_json(list(encoded)))


def _chunks[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class OnlineStore:
    def __init__(self, client: redis.Redis, *, pipeline_batch: int = 500) -> None:
        self._client = client
        self._batch = pipeline_batch

    # ----------------------------------------------------------------- scripts
    def load_scripts(self, loader: redis.Redis | None = None) -> None:
        """Load every registered script through ``loader`` (an administrative connection)."""
        connection = loader if loader is not None else self._client
        for script in SCRIPTS.values():
            sha = connection.script_load(script.source)
            if sha != script.sha1:  # pragma: no cover - Redis computes the same SHA-1
                raise ScriptsNotLoadedError(f"script {script.name} loaded with unexpected SHA")

    def scripts_loaded(self) -> bool:
        shas = [script.sha1 for script in SCRIPTS.values()]
        return all(bool(flag) for flag in self._client.script_exists(*shas))

    def ping(self) -> bool:
        return bool(self._client.ping())

    @staticmethod
    def _evalsha(pipe: Any, name: str, keys: Sequence[str], args: Sequence[str]) -> None:
        script = SCRIPTS.get(name)
        if script is None:
            raise ScriptNotRegisteredError(f"script {name!r} is not registered")
        pipe.evalsha(script.sha1, len(keys), *keys, *args)

    @staticmethod
    def _execute(pipe: Any) -> list[Any]:
        try:
            return list(pipe.execute())
        except NoScriptError:
            raise ScriptsNotLoadedError("registered scripts are not loaded in Redis") from None
        except ResponseError as exc:
            if str(exc).startswith("NOSCRIPT"):
                raise ScriptsNotLoadedError("registered scripts are not loaded in Redis") from None
            raise

    # ----------------------------------------------------------------- writes
    def begin_materialization(self, view: str, version: str, high_us: int) -> None:
        self._client.hset(meta_key(view, version), "inflight", str(high_us))

    def commit_materialization(self, view: str, version: str, high_us: int) -> None:
        pipe = self._client.pipeline(transaction=True)
        pipe.hset(meta_key(view, version), "watermark", str(high_us))
        pipe.hdel(meta_key(view, version), "inflight")
        pipe.execute()

    def write_rows(self, view: FeatureView, version: str, rows: Sequence[OnlineRow]) -> WriteStats:
        names = view.feature_names
        dtypes = [view.feature(name).dtype for name in names]
        stats = WriteStats()
        for chunk in _chunks(rows, self._batch):
            pipe = self._client.pipeline(transaction=False)
            prepared: list[tuple[str, OnlineRow, list[str]]] = []
            for row in chunk:
                if len(row.values) != len(names):
                    raise ValueError("row values do not match the view's features")
                encoded = [
                    encode_value(dtype, value)
                    for dtype, value in zip(dtypes, row.values, strict=True)
                ]
                args = [str(row.event_time_us), str(row.ingested_at_us), _digest(encoded), ""]
                for name, value in zip(names, encoded, strict=True):
                    if value is not None:
                        args += [name, value]
                key = row_key(view.name, version, row.entity_id)
                self._evalsha(pipe, "conditional_write", [key], args)
                prepared.append((key, row, args))
            for (key, row, args), result in zip(prepared, self._execute(pipe), strict=True):
                if int(result) == 1:
                    stats.written += 1
                elif int(result) == 0:
                    stats.skipped += 1
                elif self._resolve_tie(view, key, row, args):
                    stats.written += 1
                    stats.ties_resolved += 1
                else:
                    stats.skipped += 1
                    stats.ties_resolved += 1
        return stats

    @staticmethod
    def _stored_values(view: FeatureView, stored: Mapping[bytes, bytes]) -> tuple[Value, ...]:
        return tuple(
            decode_value(view.feature(name).dtype, stored.get(name.encode("utf-8")))
            for name in view.feature_names
        )

    def _resolve_tie(self, view: FeatureView, key: str, row: OnlineRow, args: list[str]) -> bool:
        """Both timestamps tie and values differ: keep the greater value tuple (offline rule)."""
        for _ in range(_MAX_TIE_RETRIES):
            stored = _bytes_map(self._client.hgetall(key))
            expected = stored.get(b"__digest", b"").decode("utf-8") if stored else ""
            if stored:
                same_time = (
                    int(stored[b"__event_time"]) == row.event_time_us
                    and int(stored[b"__ingested_at"]) == row.ingested_at_us
                )
                if same_time and value_sort_key(row.values) <= value_sort_key(
                    self._stored_values(view, stored)
                ):
                    return False
            pipe = self._client.pipeline(transaction=False)
            self._evalsha(pipe, "conditional_write", [key], [*args[:3], expected, *args[4:]])
            result = int(self._execute(pipe)[0])
            if result != 2:
                return result == 1
        raise RuntimeError("could not resolve a concurrent tie on an online row")

    # ----------------------------------------------------------------- reads
    def read(
        self,
        entity_ids: Sequence[str],
        views: Sequence[tuple[FeatureView, str]],
        now_us: int,
    ) -> OnlineReadResult:
        pipe = self._client.pipeline(transaction=True)
        for view, version in views:
            pipe.hmget(meta_key(view.name, version), "watermark", "inflight")
        for entity_id in entity_ids:
            for view, version in views:
                pipe.hgetall(row_key(view.name, version, entity_id))
        results = self._execute(pipe)
        watermarks: dict[str, tuple[int | None, int | None]] = {}
        for (view, _version), meta in zip(views, results[: len(views)], strict=True):
            low = None if meta[0] is None else int(meta[0])
            high = low if meta[1] is None else int(meta[1])
            watermarks[view.name] = (low, high)
        values: dict[tuple[str, str], OnlineValue] = {}
        cursor = len(views)
        for entity_id in entity_ids:
            for view, _version in views:
                stored = _bytes_map(results[cursor])
                cursor += 1
                values[(entity_id, view.name)] = self._decode_row(view, entity_id, stored, now_us)
        return OnlineReadResult(values=values, watermarks=watermarks)

    def _decode_row(
        self, view: FeatureView, entity_id: str, stored: Mapping[bytes, bytes], now_us: int
    ) -> OnlineValue:
        empty = dict.fromkeys(view.feature_names)
        if not stored:
            return OnlineValue(entity_id, view.name, "missing", None, None, empty)
        event_us = int(stored[b"__event_time"])
        ingested_us = int(stored[b"__ingested_at"])
        ttl_us = seconds_to_us(view.ttl_seconds)
        if event_us > now_us or ingested_us > now_us:
            # Materialized beyond the request time (clock skew between writer and reader).
            return OnlineValue(entity_id, view.name, "missing", None, None, empty)
        if ttl_us is not None and event_us <= now_us - ttl_us:
            return OnlineValue(entity_id, view.name, "expired", event_us, ingested_us, empty)
        decoded = dict(zip(view.feature_names, self._stored_values(view, stored), strict=True))
        return OnlineValue(entity_id, view.name, "ok", event_us, ingested_us, decoded)

    # ----------------------------------------------------------------- erasure and audit
    def entity_keys(self, entity_id: str, versions: Mapping[str, Sequence[str]]) -> list[str]:
        return [
            row_key(view, version, entity_id)
            for view, view_versions in sorted(versions.items())
            for version in view_versions
        ]

    def delete_entity(self, entity_id: str, versions: Mapping[str, Sequence[str]]) -> int:
        keys = self.entity_keys(entity_id, versions)
        return int(self._client.delete(*keys)) if keys else 0

    def count_entity_keys(self, entity_id: str, versions: Mapping[str, Sequence[str]]) -> int:
        keys = self.entity_keys(entity_id, versions)
        return int(self._client.exists(*keys)) if keys else 0

    def state_digest(self) -> str:
        """SHA-256 over every row hash (administrative; uses SCAN). Metadata keys are excluded."""
        digest = hashlib.sha256()
        keys = sorted(
            key.decode("utf-8") if isinstance(key, bytes) else str(key)
            for key in self._client.scan_iter(match=f"{KEY_PREFIX}:*", count=1_000)
        )
        for key in keys:
            if key.endswith(":__meta__"):
                continue
            fields = _bytes_map(self._client.hgetall(key))
            digest.update(key.encode("utf-8") + b"\n")
            for name in sorted(fields):
                digest.update(name + b"=" + fields[name] + b"\n")
        return digest.hexdigest()
