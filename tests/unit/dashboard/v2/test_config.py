"""Tests for ATP Dashboard v2 configuration module."""

import os
from unittest.mock import patch

import pytest

from atp.dashboard.v2.config import DashboardConfig, get_config, is_v2_enabled


class TestDashboardConfig:
    """Tests for DashboardConfig class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = DashboardConfig(_env_file=None)
        assert config.database_url is None
        assert config.database_echo is False
        assert config.secret_key == "atp-dashboard-dev-secret-key-change-in-prod"
        assert config.token_expire_minutes == 60
        assert config.cors_origins == "*"
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.debug is False
        assert config.title == "ATP Dashboard"
        assert config.version == "0.2.0"

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = DashboardConfig(
            database_url="postgresql+asyncpg://localhost/test",
            database_echo=True,
            secret_key="custom-secret",
            token_expire_minutes=120,
            cors_origins="http://localhost:3000,http://localhost:8080",
            host="0.0.0.0",
            port=9000,
            debug=True,
        )
        assert config.database_url == "postgresql+asyncpg://localhost/test"
        assert config.database_echo is True
        assert config.secret_key == "custom-secret"
        assert config.token_expire_minutes == 120
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.debug is True

    def test_cors_origins_list_single(self) -> None:
        """Test cors_origins_list with single origin."""
        config = DashboardConfig(cors_origins="http://localhost:3000")
        assert config.cors_origins_list == ["http://localhost:3000"]

    def test_cors_origins_list_multiple(self) -> None:
        """Test cors_origins_list with multiple origins."""
        config = DashboardConfig(
            cors_origins="http://localhost:3000, http://localhost:8080"
        )
        assert config.cors_origins_list == [
            "http://localhost:3000",
            "http://localhost:8080",
        ]

    def test_cors_origins_list_wildcard(self) -> None:
        """Test cors_origins_list with wildcard."""
        config = DashboardConfig(cors_origins="*")
        assert config.cors_origins_list == ["*"]

    def test_cors_origins_list_empty_entries(self) -> None:
        """Test cors_origins_list filters empty entries."""
        config = DashboardConfig(
            cors_origins="http://localhost:3000,  , http://localhost:8080"
        )
        assert config.cors_origins_list == [
            "http://localhost:3000",
            "http://localhost:8080",
        ]

    def test_to_dict(self) -> None:
        """Test to_dict method masks sensitive values."""
        config = DashboardConfig(
            database_url="postgresql://user:password@localhost/db",
            secret_key="super-secret-key",
        )
        result = config.to_dict()
        assert result["database_url"] == "***"
        assert result["secret_key"] == "***"
        assert result["debug"] is False
        assert result["port"] == 8080

    def test_to_dict_no_database_url(self) -> None:
        """Test to_dict when database_url is None."""
        with patch.dict(os.environ, {}, clear=True):
            config = DashboardConfig(_env_file=None)
        result = config.to_dict()
        assert result["database_url"] is None

    def test_port_validation_min(self) -> None:
        """Test port validation minimum value."""
        with pytest.raises(ValueError):
            DashboardConfig(port=0)

    def test_port_validation_max(self) -> None:
        """Test port validation maximum value."""
        with pytest.raises(ValueError):
            DashboardConfig(port=70000)

    def test_token_expire_minutes_min(self) -> None:
        """Test token_expire_minutes validation minimum value."""
        with pytest.raises(ValueError):
            DashboardConfig(token_expire_minutes=0)


class TestGetConfig:
    """Tests for get_config function."""

    def test_returns_dashboard_config(self) -> None:
        """Test that get_config returns DashboardConfig instance."""
        # Clear the cache first
        get_config.cache_clear()
        config = get_config()
        assert isinstance(config, DashboardConfig)

    def test_cached_config(self) -> None:
        """Test that get_config returns cached instance."""
        get_config.cache_clear()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


class TestIsV2Enabled:
    """Tests for is_v2_enabled function."""

    def test_disabled_by_default(self) -> None:
        """Test that v2 is disabled when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ATP_DASHBOARD_V2 if present
            os.environ.pop("ATP_DASHBOARD_V2", None)
            assert is_v2_enabled() is False

    def test_enabled_when_true(self) -> None:
        """Test that v2 is enabled when env var is 'true'."""
        with patch.dict(os.environ, {"ATP_DASHBOARD_V2": "true"}):
            assert is_v2_enabled() is True

    def test_enabled_when_true_uppercase(self) -> None:
        """Test that v2 is enabled when env var is 'TRUE'."""
        with patch.dict(os.environ, {"ATP_DASHBOARD_V2": "TRUE"}):
            assert is_v2_enabled() is True

    def test_disabled_when_false(self) -> None:
        """Test that v2 is disabled when env var is 'false'."""
        with patch.dict(os.environ, {"ATP_DASHBOARD_V2": "false"}):
            assert is_v2_enabled() is False

    def test_disabled_when_other_value(self) -> None:
        """Test that v2 is disabled when env var is other value."""
        with patch.dict(os.environ, {"ATP_DASHBOARD_V2": "yes"}):
            assert is_v2_enabled() is False
