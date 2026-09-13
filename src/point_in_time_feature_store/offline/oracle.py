"""Slow, obviously-correct reference implementation of point-in-time retrieval.

This module is the specification the DuckDB paths are differential-tested against. For a spine row
``(entity, label_time, as_of)`` with ``as_of <= label_time`` and a view with TTL ``ttl``, a history
row ``r`` is a *candidate* when::

    r.entity_id == entity
    and r.event_time  <= as_of
    and r.ingested_at <= as_of
    and (ttl is None or r.event_time > label_time - ttl)

and the selected row is the candidate with the greatest key
``(event_time, ingested_at, v1, ..., vn)`` where feature values compare with null smallest. With
``as_of == label_time`` (the default) this is exactly "latest row whose event had happened and
which had been ingested by label time, within TTL". Rows with identical keys have identical
values, so the choice among them does not matter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["HistoryRow", "SpineRow", "is_candidate", "oracle_pit", "row_sort_key", "value_sort_key"]

Value = bool | int | float | str | None


@dataclass(frozen=True, slots=True)
class HistoryRow:
    entity_id: str
    event_time: int
    ingested_at: int
    values: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class SpineRow:
    entity_id: str
    label_time: int
    as_of: int


def value_sort_key(values: Sequence[Value]) -> tuple[tuple[int, Value], ...]:
    """Null sorts before every non-null value of the same column."""
    return tuple((0, 0) if value is None else (1, value) for value in values)


def row_sort_key(row: HistoryRow) -> tuple[int, int, tuple[tuple[int, Value], ...]]:
    return (row.event_time, row.ingested_at, value_sort_key(row.values))


def is_candidate(row: HistoryRow, spine: SpineRow, ttl_us: int | None) -> bool:
    return (
        row.entity_id == spine.entity_id
        and row.event_time <= spine.as_of
        and row.ingested_at <= spine.as_of
        and (ttl_us is None or row.event_time > spine.label_time - ttl_us)
    )


def oracle_pit(
    spine: Sequence[SpineRow], history: Sequence[HistoryRow], ttl_us: int | None
) -> list[HistoryRow | None]:
    results: list[HistoryRow | None] = []
    for spine_row in spine:
        best: HistoryRow | None = None
        for row in history:
            if not is_candidate(row, spine_row, ttl_us):
                continue
            if best is None or row_sort_key(row) > row_sort_key(best):
                best = row
        results.append(best)
    return results
