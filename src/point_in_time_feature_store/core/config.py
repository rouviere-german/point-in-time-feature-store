"""Typed configuration loaded from environment variables (prefix ``APP_``).

Only ``.env.example`` is committed. Secrets are ``SecretStr`` so they never appear in ``repr``
or logs. API keys are configured as SHA-256 digests, so the process never holds raw keys. Signing
and verification keys are read from files whose paths are configured here, never inlined.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = ["Settings", "get_settings"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Service settings. Every field can be set with ``APP_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- service ---
    service_name: str = Field(
        default="point-in-time-feature-store", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$"
    )
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    enable_docs: bool = False

    # --- HTTP security ---
    api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    rate_limit_per_second: float = Field(default=50.0, gt=0)
    rate_limit_burst: int = Field(default=200, ge=1)
    rate_limit_max_clients: int = Field(default=10_000, ge=1)
    max_body_bytes: int = Field(default=65_536, ge=1)
    metrics_require_auth: bool = True

    # --- caller authentication (JWT, EdDSA) ---
    jwt_public_key_path: Path | None = None
    jwt_issuer: str = Field(default="pitfs-dev-issuer", min_length=1, max_length=256)
    jwt_audience: str = Field(default="point-in-time-feature-store", min_length=1, max_length=256)
    # Upper bound on exp - iat: tokens cannot be revoked, so long-lived ones are refused.
    jwt_max_token_lifetime_seconds: int = Field(default=3_600, ge=60, le=86_400)

    # --- observability ---
    tracing_exporter: Literal["none", "console"] = "none"
    # Keyed hash for entity ids in logs and receipts. Required outside dev/test.
    log_hmac_key: SecretStr | None = None

    # --- online store ---
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_username: str = Field(default="pitfs_app", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    redis_password: SecretStr | None = None
    online_max_batch_entities: int = Field(default=100, ge=1, le=10_000)
    online_max_features: int = Field(default=200, ge=1, le=5_000)
    request_log_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)

    # --- offline store and erasure ---
    data_dir: Path = Path("var")
    receipt_signing_key_path: Path | None = None
    duckdb_threads: int = Field(default=2, ge=1, le=64)
    duckdb_memory_limit: str = Field(default="1GB", pattern=r"^[0-9]{1,6}(MB|GB)$")

    @field_validator("api_key_sha256", mode="before")
    @classmethod
    def _parse_key_hashes(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("api_key_sha256")
    @classmethod
    def _check_key_hashes(cls, value: list[str]) -> list[str]:
        for digest in value:
            if not _SHA256_HEX.fullmatch(digest):
                raise ValueError("api_key_sha256 entries must be lowercase hex SHA-256 digests")
        return value

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("cors_allow_origins")
    @classmethod
    def _check_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if not _ORIGIN.fullmatch(origin):
                raise ValueError(
                    "cors_allow_origins must be explicit scheme://host[:port] origins "
                    "('*' is refused)"
                )
        return value

    @field_validator("log_hmac_key")
    @classmethod
    def _check_hmac_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 32:
            raise ValueError("log_hmac_key must be at least 32 characters")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests construct ``Settings(...)`` directly instead."""
    return Settings()
