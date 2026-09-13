"""Time representation: every timestamp in the store is int64 microseconds since the epoch (UTC).

Integers avoid timezone and resolution ambiguity in DuckDB, Polars, Redis (Lua doubles represent
every microsecond timestamp before year 2255 exactly) and the pure-Python oracle alike.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

__all__ = [
    "MAX_US",
    "MIN_US",
    "US_PER_DAY",
    "US_PER_SECOND",
    "Clock",
    "from_us",
    "seconds_to_us",
    "system_clock_us",
    "to_us",
]

US_PER_SECOND = 1_000_000
US_PER_DAY = 86_400 * US_PER_SECOND
# Year 1900 .. year 2200: far inside Lua's exact-integer double range (2**53 us is year 2255).
MIN_US = -2_208_988_800 * US_PER_SECOND
MAX_US = 7_258_118_400 * US_PER_SECOND

Clock = Callable[[], int]
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def to_us(value: datetime) -> int:
    """Microseconds since the epoch for a timezone-aware datetime. Naive datetimes are refused."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    delta = value - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * US_PER_SECOND + delta.microseconds


def from_us(value: int) -> datetime:
    """UTC datetime for a microsecond timestamp."""
    return _EPOCH + timedelta(microseconds=value)


def seconds_to_us(seconds: int | None) -> int | None:
    return None if seconds is None else seconds * US_PER_SECOND


def system_clock_us() -> int:
    return to_us(datetime.now(UTC))
