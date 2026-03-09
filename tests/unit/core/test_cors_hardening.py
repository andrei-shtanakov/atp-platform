"""Tests for CORS production hardening."""

import os
from unittest.mock import patch

import pytest

from atp.core.settings import DashboardSettings


class TestCORSHardening:
    def test_wildcard_allowed_in_debug(self) -> None:
        settings = DashboardSettings(debug=True, cors_origins="*")
        assert settings.cors_origins == "*"

    def test_wildcard_warns_in_dev(self) -> None:
        """Wildcard logs a warning but doesn't raise in dev."""
        settings = DashboardSettings(cors_origins="*")
        assert settings.cors_origins_list == ["*"]

    def test_wildcard_rejected_in_production(self) -> None:
        with patch.dict(os.environ, {"ATP_ENV": "production"}):
            with pytest.raises(ValueError, match="not allowed in production"):
                DashboardSettings(
                    cors_origins="*",
                    secret_key="a-real-secret-key-for-production-use",
                )

    def test_wildcard_rejected_in_staging(self) -> None:
        with patch.dict(os.environ, {"ATP_ENV": "staging"}):
            with pytest.raises(ValueError, match="not allowed in production"):
                DashboardSettings(
                    cors_origins="*",
                    secret_key="a-real-secret-key-for-production-use",
                )

    def test_explicit_origins_ok_in_production(self) -> None:
        with patch.dict(os.environ, {"ATP_ENV": "production"}):
            settings = DashboardSettings(
                cors_origins="https://app.example.com,https://admin.example.com",
                secret_key="a-real-secret-key-for-production-use",
            )
            assert len(settings.cors_origins_list) == 2

    def test_cors_origins_list_parsing(self) -> None:
        settings = DashboardSettings(
            cors_origins="http://localhost:3000, http://localhost:8080"
        )
        assert settings.cors_origins_list == [
            "http://localhost:3000",
            "http://localhost:8080",
        ]
