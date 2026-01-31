"""Dashboard configuration for ATP Dashboard v2.

This module provides centralized configuration management for the dashboard,
supporting environment variables and sensible defaults.
"""

import os
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


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
        default="atp-dashboard-dev-secret-key-change-in-prod",
        description="Secret key for JWT token signing",
    )
    token_expire_minutes: int = Field(
        default=60,
        ge=1,
        description="JWT token expiration time in minutes",
    )

    # CORS settings
    cors_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins",
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


def is_v2_enabled() -> bool:
    """Check if Dashboard v2 is enabled via feature flag.

    Returns:
        True if ATP_DASHBOARD_V2 environment variable is set to 'true'.
    """
    return os.getenv("ATP_DASHBOARD_V2", "false").lower() == "true"
