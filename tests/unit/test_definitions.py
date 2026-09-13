from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from point_in_time_feature_store.definitions import (
    EntityDefinition,
    FeatureDefinition,
    FeatureRepo,
    FeatureView,
    KeyCharset,
    SourceColumn,
    SourceDefinition,
    feature_column,
    parse_feature_ref,
)
from point_in_time_feature_store.dsl import DType, col
from point_in_time_feature_store.offline.synthetic import churn_feature_repo

pytestmark = pytest.mark.unit


def _view(**overrides: object) -> FeatureView:
    return churn_feature_repo().view("transaction_stats").model_copy(update=overrides)


def test_version_hash_ignores_feature_order_and_metadata() -> None:
    view = _view()
    reordered = _view(features=tuple(reversed(view.features)))
    assert reordered.version_hash() == view.version_hash()
    relabelled = _view(owner="another-team", description="changed", tags={"tier": "gold"})
    assert relabelled.version_hash() == view.version_hash()
    assert relabelled.metadata_hash() != view.metadata_hash()


def test_version_hash_changes_with_semantics() -> None:
    view = _view()
    assert _view(ttl_seconds=7 * 86_400).version_hash() != view.version_hash()
    assert _view(zero_ingestion_lag=True).version_hash() != view.version_hash()
    redefined = tuple(
        feature.model_copy(update={"expression": (col("amount") * 2).node})
        if feature.name == "last_amount"
        else feature
        for feature in view.features
    )
    assert _view(features=redefined).version_hash() != view.version_hash()


def test_repo_round_trips_through_json() -> None:
    repo = churn_feature_repo()
    assert FeatureRepo.from_json(repo.to_json()) == repo


def test_features_derived_from_pii_columns_must_be_tagged() -> None:
    repo = churn_feature_repo()
    profile = repo.view("customer_profile")
    untagged = tuple(
        feature.model_copy(update={"pii": False}) if feature.name == "email" else feature
        for feature in profile.features
    )
    with pytest.raises(ValidationError, match="must be tagged pii"):
        FeatureRepo(
            entities=repo.entities,
            sources=repo.sources,
            feature_views=(profile.model_copy(update={"features": untagged}),),
        )


def test_cross_references_and_types_are_validated() -> None:
    repo = churn_feature_repo()
    view = repo.view("transaction_stats")
    with pytest.raises(ValidationError, match="unknown source"):
        FeatureRepo(entities=repo.entities, sources=(), feature_views=(view,))
    with pytest.raises(ValidationError, match="unknown entity"):
        FeatureRepo(entities=(), sources=repo.sources)
    bad_type = FeatureDefinition(
        name="last_amount", dtype=DType.INT64, expression=col("amount").node
    )
    with pytest.raises(ValidationError, match="declared"):
        FeatureRepo(
            entities=repo.entities,
            sources=repo.sources,
            feature_views=(view.model_copy(update={"features": (bad_type,)}),),
        )
    with pytest.raises(ValidationError, match="duplicate"):
        FeatureRepo(entities=repo.entities + repo.entities)


def test_names_are_validated() -> None:
    with pytest.raises(ValidationError):
        SourceColumn(name="event_time", dtype=DType.INT64)
    with pytest.raises(ValidationError):
        FeatureDefinition(name="Bad-Name", dtype=DType.INT64, expression=col("x").node)
    with pytest.raises(ValidationError, match="duplicate column"):
        SourceDefinition(
            name="s",
            entity="e",
            columns=(
                SourceColumn(name="a", dtype=DType.INT64),
                SourceColumn(name="a", dtype=DType.INT64),
            ),
        )
    feature = FeatureDefinition(name="a", dtype=DType.INT64, expression=col("a").node)
    with pytest.raises(ValidationError, match="duplicate feature"):
        FeatureView(name="v", entity="e", source="s", owner="o", features=(feature, feature))
    with pytest.raises(ValidationError, match="__"):
        FeatureView(name="v__x", entity="e", source="s", owner="o", features=(feature,))


def test_feature_references() -> None:
    assert parse_feature_ref("customer_profile:plan_tier") == ("customer_profile", "plan_tier")
    assert parse_feature_ref("session_stats:*") == ("session_stats", "*")
    assert feature_column("a", "b") == "a__b"
    for bad in ("nocolon", "a:b:c", "A:b", "a:", ":b", "a:b c"):
        with pytest.raises(ValueError, match="feature references"):
            parse_feature_ref(bad)


def test_lookups_raise_for_unknown_names() -> None:
    repo = churn_feature_repo()
    with pytest.raises(KeyError):
        repo.view("nope")
    with pytest.raises(KeyError):
        repo.source("customers").column("nope")
    with pytest.raises(KeyError):
        repo.view("customer_profile").feature("nope")


@pytest.mark.parametrize(
    ("charset", "good", "bad"),
    [
        (KeyCharset.SLUG, "c000123", "c:1"),
        (KeyCharset.DIGITS, "0042", "42a"),
        (KeyCharset.HEX, "deadbeef", "DEADBEEF"),
    ],
)
def test_entity_key_presets(charset: KeyCharset, good: str, bad: str) -> None:
    entity = EntityDefinition(name="e", key_charset=charset, key_max_length=8)
    assert entity.is_valid_key(good)
    assert not entity.is_valid_key(bad)
    assert not entity.is_valid_key(good + "0" * 8)


@given(st.text(min_size=1, max_size=70))
def test_entity_keys_never_contain_key_separators_or_whitespace(key: str) -> None:
    entity = EntityDefinition(name="customer")
    if entity.is_valid_key(key):
        assert ":" not in key
        assert not any(char.isspace() for char in key)
        assert key.isascii()
        assert len(key) <= 64
