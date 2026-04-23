"""Dashboard configuration for ATP Dashboard v2.

This module provides centralized configuration management for the dashboard,
supporting environment variables and sensible defaults.
"""

import logging
import warnings
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger("atp.dashboard")

_DEFAULT_SECRET_KEY = "atp-dashboard-dev-secret-key-change-in-prod"


class DashboardConfig(BaseSettings):
    """Configuration settings for ATP Dashboard v2.

    Settings can be configured via environment variables with the ATP_ prefix.
    """

    # Database settings
    database_url: str | None = Field(
        default=None,
        description="Database URL. Defaults to SQLite at ~/.atp/dashboard.db",
    )
    database_echo: bool = Field(
        default=False,
        description="Echo SQL statements for debugging",
    )

    # Authentication settings
    secret_key: str = Field(
        default=_DEFAULT_SECRET_KEY,
        description="Secret key for JWT token signing",
    )
    token_expire_minutes: int = Field(
        default=60,
        ge=1,
        description="JWT token expiration time in minutes",
    )

    # CORS settings
    cors_origins: str = Field(
        default="",
        description="Comma-separated list of allowed CORS origins. Empty = no CORS.",
    )

    # Server settings
    host: str = Field(
        default="127.0.0.1",
        description="Server host address",
    )
    port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="Server port",
    )

    # Feature flags
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    disable_auth: bool = Field(
        default=False,
        description="Disable authentication (WARNING: For development only!)",
    )

    # GitHub OAuth settings (for Device Flow)
    github_client_id: str | None = Field(
        default=None,
        description="GitHub OAuth App client ID for device flow",
    )
    github_client_secret: str | None = Field(
        default=None,
        description="GitHub OAuth App client secret for device flow",
    )

    # Batch settings
    batch_max_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum batch size for next-task endpoint",
    )

    # Upload settings
    upload_max_size_mb: int = Field(
        default=1,
        ge=1,
        le=50,
        description="Maximum YAML upload file size in MB",
    )

    # Rate limiting settings
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable HTTP rate limiting",
    )
    rate_limit_default: str = Field(
        default="60/minute",
        description="Default rate limit for undecorated endpoints",
    )
    rate_limit_auth: str = Field(
        default="5/minute",
        description="Rate limit for auth endpoints (brute-force protection)",
    )
    rate_limit_api: str = Field(
        default="120/minute",
        description="Rate limit for benchmark API endpoints",
    )
    rate_limit_upload: str = Field(
        default="10/minute",
        description="Rate limit for file upload endpoints",
    )
    rate_limit_storage: str = Field(
        default="memory://",
        description=("Rate limit storage URI (memory:// or redis://host:port)"),
    )

    # Token self-service settings
    registration_mode: str = Field(
        default="invite",
        description="Registration mode: 'invite' (code required) or 'open'",
    )
    max_agents_per_user: int = Field(
        default=10,
        ge=1,
        description=(
            "DEPRECATED (LABS-TSA PR-2): use max_benchmark_agents_per_user and "
            "max_tournament_agents_per_user instead. Retained so external tooling "
            "setting ATP_MAX_AGENTS_PER_USER does not fail to load."
        ),
    )
    max_benchmark_agents_per_user: int = Field(
        default=10,
        ge=1,
        description="Max Agent rows per user with purpose='benchmark'",
        validation_alias="ATP_MAX_BENCHMARK_AGENTS_PER_USER",
    )
    max_tournament_agents_per_user: int = Field(
        default=5,
        ge=1,
        description="Max Agent rows per user with purpose='tournament'",
        validation_alias="ATP_MAX_TOURNAMENT_AGENTS_PER_USER",
    )
    max_concurrent_private_tournaments_per_user: int = Field(
        default=3,
        ge=1,
        description="Max pending+active private tournaments per user",
        validation_alias="ATP_MAX_CONCURRENT_PRIVATE_TOURNAMENTS_PER_USER",
    )
    max_tokens_per_agent: int = Field(
        default=3,
        ge=1,
        description="Maximum active API tokens per agent",
    )
    max_user_tokens: int = Field(
        default=5,
        ge=1,
        description="Maximum user-level API tokens",
    )
    default_token_days: int = Field(
        default=30,
        ge=1,
        description="Default token expiry in days",
    )
    max_token_days: int = Field(
        default=365,
        ge=0,
        description="Maximum token expiry in days (0 = allow 'never')",
    )

    # App metadata
    title: str = Field(
        default="ATP Dashboard",
        description="Application title",
    )
    description: str = Field(
        default="Web dashboard for Agent Test Platform results",
        description="Application description",
    )
    version: str = Field(
        default="0.2.0",
        description="Application version",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate security-sensitive defaults."""
        if self.secret_key == _DEFAULT_SECRET_KEY:
            if not self.debug:
                raise ValueError(
                    "ATP_SECRET_KEY must be set in production. "
                    "Set ATP_SECRET_KEY environment variable."
                )
            warnings.warn(
                "ATP_SECRET_KEY is not set. Using a hardcoded dev secret. "
                "Tokens will be shared across restarts but are insecure. "
                "Set ATP_SECRET_KEY for production use.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "ATP_SECRET_KEY is using the insecure default. "
                "Set ATP_SECRET_KEY before deploying to production."
            )

    model_config = {
        "env_prefix": "ATP_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging/debugging."""
        return {
            "database_url": "***" if self.database_url else None,
            "database_echo": self.database_echo,
            "secret_key": "***",
            "token_expire_minutes": self.token_expire_minutes,
            "cors_origins": self.cors_origins,
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "disable_auth": self.disable_auth,
            "github_client_id": self.github_client_id,
            "github_client_secret": "***" if self.github_client_secret else None,
            "batch_max_size": self.batch_max_size,
            "upload_max_size_mb": self.upload_max_size_mb,
            "title": self.title,
            "version": self.version,
        }


@lru_cache
def get_config() -> DashboardConfig:
    """Get the dashboard configuration (cached).

    Returns:
        DashboardConfig instance with settings loaded from environment.
    """
    return DashboardConfig()
