from __future__ import annotations

import io
import json
import logging
from typing import Any

import pytest

from point_in_time_feature_store.core.logging import configure_logging, get_logger

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("restore_logging")]


def _records(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_structlog_events_are_json_and_redacted() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    get_logger("test", component="unit").info(
        "user_login", email="jane@example.com", api_key="k-123456", attempts=2
    )
    (record,) = _records(stream)
    assert record["event"] == "user_login"
    assert record["email"] == "[REDACTED:EMAIL]"
    assert record["api_key"] == "[REDACTED]"
    assert record["attempts"] == 2
    assert record["level"] == "info"
    assert record["component"] == "unit"
    assert "timestamp" in record


def test_stdlib_records_pass_through_the_same_redaction() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    logging.getLogger("uvicorn.error").warning("login failed for %s", "bob@example.org")
    (record,) = _records(stream)
    assert record["event"] == "login failed for [REDACTED:EMAIL]"
    assert record["level"] == "warning"
    assert logging.getLogger("uvicorn.access").disabled


def _raise_with_secret() -> None:
    raise ValueError("db password=hunter2hunter2")


def test_exception_tracebacks_are_redacted() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    try:
        _raise_with_secret()
    except ValueError:
        get_logger().exception("query_failed")
    output = stream.getvalue()
    assert "hunter2hunter2" not in output
    assert "ValueError" in output


def test_level_filtering() -> None:
    stream = io.StringIO()
    configure_logging("WARNING", stream=stream)
    log = get_logger()
    log.info("hidden")
    log.warning("shown")
    assert [r["event"] for r in _records(stream)] == ["shown"]


def test_console_renderer_is_also_redacted() -> None:
    stream = io.StringIO()
    configure_logging("DEBUG", json_output=False, stream=stream)
    get_logger().debug("hello", email="jane@example.com")
    output = stream.getvalue()
    assert "[REDACTED:EMAIL]" in output
    assert "jane@example.com" not in output


class _Holder:
    def __repr__(self) -> str:
        return "Holder(owner='jane@example.com')"


def test_sets_bytes_objects_and_mapping_keys_are_redacted_before_rendering() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    get_logger().info(
        "fan_out",
        recipients={"jane@example.com"},
        raw=b"password=hunter2hunter2",
        holder=_Holder(),
        per_user={"bob@example.org": 3},
    )
    output = stream.getvalue()
    for leaked in ("jane@example.com", "hunter2hunter2", "bob@example.org"):
        assert leaked not in output
    (record,) = _records(stream)
    assert record["recipients"] == ["[REDACTED:EMAIL]"]
    assert record["per_user"] == {"[REDACTED:EMAIL]": 3}
