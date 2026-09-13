from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from point_in_time_feature_store.core.config import Settings, get_settings

pytestmark = pytest.mark.unit


def test_defaults_fail_closed() -> None:
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == []
    assert settings.cors_allow_origins == []
    assert settings.enable_docs is False
    assert settings.metrics_require_auth is True
    assert settings.host == "127.0.0.1"
    assert settings.jwt_public_key_path is None
    assert settings.redis_password is None
    assert settings.log_hmac_key is None


def test_environment_variables_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_API_KEY_SHA256", f"{'a' * 64}, {'b' * 64}")
    monkeypatch.setenv("APP_CORS_ALLOW_ORIGINS", "https://app.example.com,http://localhost:3000")
    monkeypatch.setenv("APP_REDIS_PASSWORD", "not-a-real-password-123")
    monkeypatch.setenv("APP_LOG_HMAC_KEY", "k" * 40)
    monkeypatch.setenv("APP_ONLINE_MAX_BATCH_ENTITIES", "25")
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == ["a" * 64, "b" * 64]
    assert settings.cors_allow_origins == ["https://app.example.com", "http://localhost:3000"]
    assert settings.redis_password is not None
    assert settings.redis_password.get_secret_value() == "not-a-real-password-123"
    assert "not-a-real-password-123" not in repr(settings)
    assert "k" * 40 not in repr(settings)
    assert settings.online_max_batch_entities == 25


@pytest.mark.parametrize("origins", [["*"], ["https://example.com/"], ["example.com"]])
def test_wildcard_or_malformed_origins_are_rejected(origins: list[str]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cors_allow_origins=origins)


@pytest.mark.parametrize("digest", ["A" * 64, "abc", "g" * 64])
def test_non_sha256_key_hashes_are_rejected(digest: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, api_key_sha256=[digest])


def test_short_hmac_keys_and_bad_memory_limits_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_hmac_key="short")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, duckdb_memory_limit="1GB; SET x=1")


def test_env_example_documents_every_setting(repo_root: Path) -> None:
    lines = (repo_root / ".env.example").read_text(encoding="utf-8").splitlines()
    documented = {
        line.split("=", 1)[0].strip() for line in lines if line.strip() and not line.startswith("#")
    }
    expected = {f"APP_{name.upper()}" for name in Settings.model_fields}
    assert documented == expected


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
