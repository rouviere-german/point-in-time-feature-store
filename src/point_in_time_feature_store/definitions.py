"""Declarative definitions: entities, sources and feature views, versioned by definition hash.

A feature view's *version* is the SHA-256 of the canonical JSON of everything that determines its
values (entity, source, TTL, ingestion-lag declaration and each feature's name, type and
transformation). Ownership, descriptions, tags and PII flags form a separate *metadata hash*: they
change who may read a feature, not what it computes. Both are independent of feature order.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from point_in_time_feature_store.core.canonical import canonical_json, sha256_hex
from point_in_time_feature_store.dsl import DType, Expr, referenced_columns, validate_expression

__all__ = [
    "IDENT_PATTERN",
    "RESERVED_COLUMNS",
    "EntityDefinition",
    "FeatureDefinition",
    "FeatureRepo",
    "FeatureView",
    "KeyCharset",
    "SourceColumn",
    "SourceDefinition",
    "feature_column",
    "parse_feature_ref",
]

IDENT_PATTERN = r"^[a-z][a-z0-9_]{0,62}$"
RESERVED_COLUMNS = frozenset({"entity_id", "event_time", "ingested_at", "label_time", "as_of"})
_OWNER = r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$"
_FEATURE_REF = re.compile(r"^([a-z][a-z0-9_]{0,62}):([a-z][a-z0-9_]{0,62}|\*)$")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class KeyCharset(StrEnum):
    """Entity key formats. Keys are validated with a fixed regex per preset, never a user regex."""

    SLUG = "slug"
    DIGITS = "digits"
    HEX = "hex"


_CHARSET_REGEX = {
    KeyCharset.SLUG: ("[A-Za-z0-9]", "[A-Za-z0-9_-]"),
    KeyCharset.DIGITS: ("[0-9]", "[0-9]"),
    KeyCharset.HEX: ("[0-9a-f]", "[0-9a-f]"),
}


class EntityDefinition(_Model):
    name: str = Field(pattern=IDENT_PATTERN)
    description: str = Field(default="", max_length=1_000)
    key_charset: KeyCharset = KeyCharset.SLUG
    key_max_length: int = Field(default=64, ge=1, le=128)

    @property
    def key_regex(self) -> str:
        first, rest = _CHARSET_REGEX[self.key_charset]
        return f"^{first}{rest}{{0,{self.key_max_length - 1}}}$"

    def is_valid_key(self, key: str) -> bool:
        return re.fullmatch(self.key_regex, key) is not None

    def version_hash(self) -> str:
        return sha256_hex(
            canonical_json(
                {"name": self.name, "charset": self.key_charset.value, "max": self.key_max_length}
            )
        )

    def metadata_hash(self) -> str:
        return sha256_hex(canonical_json({"description": self.description}))


class SourceColumn(_Model):
    name: str = Field(pattern=IDENT_PATTERN)
    dtype: DType
    nullable: bool = True
    ge: float | None = None
    le: float | None = None
    allowed_values: tuple[str, ...] | None = Field(default=None, max_length=256)
    pii: bool = False

    @field_validator("name")
    @classmethod
    def _not_reserved(cls, value: str) -> str:
        if value in RESERVED_COLUMNS:
            raise ValueError(f"column name {value!r} is reserved")
        return value


class SourceDefinition(_Model):
    """A bitemporal source: every row carries entity_id, event_time and ingested_at."""

    name: str = Field(pattern=IDENT_PATTERN)
    entity: str = Field(pattern=IDENT_PATTERN)
    columns: tuple[SourceColumn, ...] = Field(min_length=1, max_length=256)
    # Rows may carry an event_time at most this far after their ingested_at (clock skew).
    max_clock_skew_seconds: int = Field(default=0, ge=0, le=3_600)
    # Rows may arrive at most this long after their event_time (backfills beyond are rejected).
    max_backfill_days: int = Field(default=400, ge=1, le=36_500)
    owner: str = Field(default="data-platform", pattern=_OWNER)
    description: str = Field(default="", max_length=1_000)

    @model_validator(mode="after")
    def _unique_columns(self) -> SourceDefinition:
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError(f"source {self.name}: duplicate column names")
        return self

    def column(self, name: str) -> SourceColumn:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    @property
    def column_types(self) -> dict[str, DType]:
        return {column.name: column.dtype for column in self.columns}

    def version_hash(self) -> str:
        payload: dict[str, JsonValue] = {
            "name": self.name,
            "entity": self.entity,
            "columns": [
                column.model_dump(mode="json")
                for column in sorted(self.columns, key=lambda item: item.name)
            ],
            "max_clock_skew_seconds": self.max_clock_skew_seconds,
            "max_backfill_days": self.max_backfill_days,
        }
        return sha256_hex(canonical_json(payload))

    def metadata_hash(self) -> str:
        return sha256_hex(canonical_json({"owner": self.owner, "description": self.description}))


class FeatureDefinition(_Model):
    name: str = Field(pattern=IDENT_PATTERN)
    dtype: DType
    expression: Expr
    pii: bool = False
    description: str = Field(default="", max_length=1_000)

    @field_validator("name")
    @classmethod
    def _not_reserved(cls, value: str) -> str:
        if value in RESERVED_COLUMNS:
            raise ValueError(f"feature name {value!r} is reserved")
        return value


class FeatureView(_Model):
    name: str = Field(pattern=IDENT_PATTERN)
    entity: str = Field(pattern=IDENT_PATTERN)
    source: str = Field(pattern=IDENT_PATTERN)
    features: tuple[FeatureDefinition, ...] = Field(min_length=1, max_length=256)
    ttl_seconds: int | None = Field(default=None, ge=1, le=100 * 365 * 86_400)
    # Declares that rows are never ingested after their event time (ingested_at <= event_time).
    # Enables the uni-temporal ASOF fast path; the data is checked before that path is used.
    zero_ingestion_lag: bool = False
    owner: str = Field(pattern=_OWNER)
    description: str = Field(default="", max_length=1_000)
    tags: dict[str, str] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def _unique_features(self) -> FeatureView:
        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise ValueError(f"feature view {self.name}: duplicate feature names")
        if "__" in self.name:
            raise ValueError("feature view names must not contain '__'")
        return self

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature names in canonical (sorted) order."""
        return tuple(sorted(feature.name for feature in self.features))

    def feature(self, name: str) -> FeatureDefinition:
        for feature in self.features:
            if feature.name == name:
                return feature
        raise KeyError(name)

    @property
    def pii_features(self) -> frozenset[str]:
        return frozenset(feature.name for feature in self.features if feature.pii)

    def semantic_payload(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "entity": self.entity,
            "source": self.source,
            "ttl_seconds": self.ttl_seconds,
            "zero_ingestion_lag": self.zero_ingestion_lag,
            "features": [
                {
                    "name": feature.name,
                    "dtype": feature.dtype.value,
                    "expression": feature.expression.model_dump(mode="json"),
                }
                for feature in sorted(self.features, key=lambda item: item.name)
            ],
        }

    def version_hash(self) -> str:
        return sha256_hex(canonical_json(self.semantic_payload()))

    def metadata_hash(self) -> str:
        return sha256_hex(
            canonical_json(
                {
                    "owner": self.owner,
                    "description": self.description,
                    "tags": dict(self.tags),
                    "pii": sorted(self.pii_features),
                    "feature_descriptions": {f.name: f.description for f in self.features},
                }
            )
        )


class FeatureRepo(_Model):
    """A complete, cross-validated set of definitions (what ``plan``/``apply`` operate on)."""

    entities: tuple[EntityDefinition, ...] = ()
    sources: tuple[SourceDefinition, ...] = ()
    feature_views: tuple[FeatureView, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _canonical_order(cls, data: Any) -> Any:
        """Definitions are sets: store them sorted by name so equal repos compare equal."""
        if not isinstance(data, dict):
            return data

        def name_of(item: Any) -> str:
            if isinstance(item, dict):
                return str(item.get("name", ""))
            return str(getattr(item, "name", ""))

        ordered = dict(data)
        for key in ("entities", "sources", "feature_views"):
            items = ordered.get(key)
            if isinstance(items, list | tuple):
                ordered[key] = sorted(items, key=name_of)
        return ordered

    @model_validator(mode="after")
    def _cross_validate(self) -> FeatureRepo:
        for kind, items in (
            ("entity", self.entities),
            ("source", self.sources),
            ("feature view", self.feature_views),
        ):
            names = [item.name for item in items]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {kind} names")
        entities = {entity.name for entity in self.entities}
        sources = {source.name: source for source in self.sources}
        for declared in self.sources:
            if declared.entity not in entities:
                raise ValueError(f"source {declared.name}: unknown entity {declared.entity}")
        for view in self.feature_views:
            source = sources.get(view.source)
            if source is None:
                raise ValueError(f"feature view {view.name}: unknown source {view.source}")
            if view.entity != source.entity:
                raise ValueError(f"feature view {view.name}: entity differs from its source")
            pii_columns = {column.name for column in source.columns if column.pii}
            for feature in view.features:
                validate_expression(feature.expression, source.column_types, feature.dtype)
                laundered = referenced_columns(feature.expression) & pii_columns
                if laundered and not feature.pii:
                    raise ValueError(
                        f"feature {view.name}:{feature.name} derives from PII columns "
                        f"({', '.join(sorted(laundered))}) and must be tagged pii"
                    )
        return self

    def entity(self, name: str) -> EntityDefinition:
        return _find(self.entities, name, "entity")

    def source(self, name: str) -> SourceDefinition:
        return _find(self.sources, name, "source")

    def view(self, name: str) -> FeatureView:
        return _find(self.feature_views, name, "feature view")

    def to_json(self) -> str:
        return self.model_dump_json(indent=2) + "\n"

    def revalidated(self) -> FeatureRepo:
        """A validated copy; ``model_copy(update=...)`` skips validators, so callers re-check."""
        return FeatureRepo.model_validate(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, text: str) -> FeatureRepo:
        return cls.model_validate_json(text)


def _find[T: Any](items: tuple[T, ...], name: str, kind: str) -> T:
    for item in items:
        if item.name == name:
            return item
    raise KeyError(f"unknown {kind}: {name}")


def feature_column(view: str, feature: str) -> str:
    """Column name of a feature in retrieval output: ``<view>__<feature>``."""
    return f"{view}__{feature}"


def parse_feature_ref(ref: str) -> tuple[str, str]:
    """Split ``view:feature`` (or ``view:*``) into its parts."""
    match = _FEATURE_REF.fullmatch(ref)
    if match is None:
        raise ValueError("feature references look like 'view:feature' or 'view:*'")
    return match.group(1), match.group(2)
