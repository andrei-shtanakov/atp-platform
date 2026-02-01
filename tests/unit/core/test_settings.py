"""Tests for ATP Platform configuration management."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from atp.core.settings import (
    ATPSettings,
    DashboardSettings,
    LLMSettings,
    LoggingSettings,
    RunnerSettings,
    _find_config_file,
    _flatten_dict,
    _load_yaml_config,
    generate_example_config,
    generate_json_schema,
    get_cached_settings,
    get_settings,
)


class TestATPSettings:
    """Tests for ATPSettings class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = ATPSettings(_skip_file_loading=True)
        assert settings.log_level == "INFO"
        assert settings.parallel_workers == 4
        assert settings.default_timeout == 300
        assert settings.anthropic_api_key is None
        assert settings.openai_api_key is None
        assert settings.default_llm_model == "claude-sonnet-4-20250514"
        assert settings.dashboard_host == "127.0.0.1"
        assert settings.dashboard_port == 8080
        assert settings.dashboard_debug is False
        assert settings.database_url is None
        assert settings.fail_fast is False
        assert settings.sandbox_enabled is False
        assert settings.runs_per_test == 1

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        settings = ATPSettings(
            _skip_file_loading=True,
            log_level="DEBUG",
            parallel_workers=8,
            default_timeout=600,
            dashboard_host="0.0.0.0",
            dashboard_port=9000,
            dashboard_debug=True,
            fail_fast=True,
            runs_per_test=5,
        )
        assert settings.log_level == "DEBUG"
        assert settings.parallel_workers == 8
        assert settings.default_timeout == 600
        assert settings.dashboard_host == "0.0.0.0"
        assert settings.dashboard_port == 9000
        assert settings.dashboard_debug is True
        assert settings.fail_fast is True
        assert settings.runs_per_test == 5

    def test_log_level_validation(self) -> None:
        """Test log level validation."""
        # Valid levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = ATPSettings(_skip_file_loading=True, log_level=level)
            assert settings.log_level == level

        # Case insensitive
        settings = ATPSettings(_skip_file_loading=True, log_level="debug")
        assert settings.log_level == "DEBUG"

        # Invalid level
        with pytest.raises(ValueError):
            ATPSettings(_skip_file_loading=True, log_level="INVALID")

    def test_parallel_workers_validation(self) -> None:
        """Test parallel workers validation."""
        # Valid range
        settings = ATPSettings(_skip_file_loading=True, parallel_workers=1)
        assert settings.parallel_workers == 1

        settings = ATPSettings(_skip_file_loading=True, parallel_workers=32)
        assert settings.parallel_workers == 32

        # Invalid: below minimum
        with pytest.raises(ValueError):
            ATPSettings(_skip_file_loading=True, parallel_workers=0)

        # Invalid: above maximum
        with pytest.raises(ValueError):
            ATPSettings(_skip_file_loading=True, parallel_workers=33)

    def test_port_validation(self) -> None:
        """Test port validation."""
        # Valid range
        settings = ATPSettings(_skip_file_loading=True, dashboard_port=1)
        assert settings.dashboard_port == 1

        settings = ATPSettings(_skip_file_loading=True, dashboard_port=65535)
        assert settings.dashboard_port == 65535

        # Invalid: below minimum
        with pytest.raises(ValueError):
            ATPSettings(_skip_file_loading=True, dashboard_port=0)

        # Invalid: above maximum
        with pytest.raises(ValueError):
            ATPSettings(_skip_file_loading=True, dashboard_port=70000)

    def test_timeout_validation(self) -> None:
        """Test timeout validation."""
        # Valid range
        settings = ATPSettings(_skip_file_loading=True, default_timeout=1)
        assert settings.default_timeout == 1

        settings = ATPSettings(_skip_file_loading=True, default_timeout=3600)
        assert settings.default_timeout == 3600

        # Invalid: below minimum
        with pytest.raises(ValueError):
            ATPSettings(_skip_file_loading=True, default_timeout=0)

        # Invalid: above maximum
        with pytest.raises(ValueError):
            ATPSettings(_skip_file_loading=True, default_timeout=3601)

    def test_secret_str_for_api_keys(self) -> None:
        """Test that API keys are stored as SecretStr."""
        from pydantic import SecretStr

        settings = ATPSettings(
            _skip_file_loading=True,
            anthropic_api_key="test-anthropic-key",
            openai_api_key="test-openai-key",
        )

        # Keys should be SecretStr instances
        assert isinstance(settings.anthropic_api_key, SecretStr)
        assert isinstance(settings.openai_api_key, SecretStr)

        # Values should be retrievable
        assert settings.anthropic_api_key.get_secret_value() == "test-anthropic-key"
        assert settings.openai_api_key.get_secret_value() == "test-openai-key"

        # String representation should be masked
        assert "test-anthropic-key" not in str(settings.anthropic_api_key)

    def test_to_dict_masks_secrets(self) -> None:
        """Test to_dict masks sensitive values by default."""
        settings = ATPSettings(
            _skip_file_loading=True,
            anthropic_api_key="secret-key",
            dashboard_secret_key="dashboard-secret",
        )

        result = settings.to_dict(mask_secrets=True)
        assert result["anthropic_api_key"] == "***"
        assert result["dashboard_secret_key"] == "***"

    def test_to_dict_reveals_secrets_when_disabled(self) -> None:
        """Test to_dict reveals secrets when mask_secrets=False."""
        settings = ATPSettings(
            _skip_file_loading=True,
            anthropic_api_key="secret-key",
        )

        result = settings.to_dict(mask_secrets=False)
        assert result["anthropic_api_key"] == "secret-key"

    def test_get_llm_api_key_anthropic(self) -> None:
        """Test get_llm_api_key for Anthropic."""
        settings = ATPSettings(
            _skip_file_loading=True,
            anthropic_api_key="anthropic-key",
        )

        assert settings.get_llm_api_key("anthropic") == "anthropic-key"
        assert settings.get_llm_api_key("ANTHROPIC") == "anthropic-key"

    def test_get_llm_api_key_openai(self) -> None:
        """Test get_llm_api_key for OpenAI."""
        settings = ATPSettings(
            _skip_file_loading=True,
            openai_api_key="openai-key",
        )

        assert settings.get_llm_api_key("openai") == "openai-key"
        assert settings.get_llm_api_key("OPENAI") == "openai-key"

    def test_get_llm_api_key_default_provider(self) -> None:
        """Test get_llm_api_key uses default provider when not specified."""
        settings = ATPSettings(
            _skip_file_loading=True,
            anthropic_api_key="anthropic-key",
        )

        # Default provider is anthropic
        assert settings.get_llm_api_key() == "anthropic-key"

    def test_get_llm_api_key_unknown_provider(self) -> None:
        """Test get_llm_api_key returns None for unknown provider."""
        settings = ATPSettings(_skip_file_loading=True)
        assert settings.get_llm_api_key("unknown") is None


class TestEnvironmentVariables:
    """Tests for environment variable loading."""

    def test_env_var_loading(self) -> None:
        """Test that environment variables are loaded with ATP_ prefix."""
        with patch.dict(
            os.environ,
            {
                "ATP_LOG_LEVEL": "DEBUG",
                "ATP_PARALLEL_WORKERS": "16",
                "ATP_DASHBOARD_PORT": "9090",
            },
        ):
            settings = ATPSettings(_skip_file_loading=True)
            assert settings.log_level == "DEBUG"
            assert settings.parallel_workers == 16
            assert settings.dashboard_port == 9090

    def test_env_var_api_keys(self) -> None:
        """Test loading API keys from environment variables."""
        with patch.dict(
            os.environ,
            {
                "ATP_ANTHROPIC_API_KEY": "env-anthropic-key",
                "ATP_OPENAI_API_KEY": "env-openai-key",
            },
        ):
            settings = ATPSettings(_skip_file_loading=True)
            assert (
                settings.anthropic_api_key is not None
                and settings.anthropic_api_key.get_secret_value() == "env-anthropic-key"
            )
            assert (
                settings.openai_api_key is not None
                and settings.openai_api_key.get_secret_value() == "env-openai-key"
            )

    def test_env_var_nested_delimiter(self) -> None:
        """Test nested environment variable delimiter (__)."""
        with patch.dict(
            os.environ,
            {
                "ATP_DASHBOARD__HOST": "0.0.0.0",
                "ATP_DASHBOARD__PORT": "3000",
            },
        ):
            settings = ATPSettings(_skip_file_loading=True)
            assert settings.dashboard.host == "0.0.0.0"
            assert settings.dashboard.port == 3000


class TestConfigFileLoading:
    """Tests for YAML configuration file loading."""

    def test_load_yaml_config_valid(self, tmp_path: Path) -> None:
        """Test loading valid YAML config file."""
        config_file = tmp_path / "atp.config.yaml"
        config_file.write_text("""
log_level: DEBUG
parallel_workers: 8
dashboard_port: 9000
""")

        config = _load_yaml_config(config_file)
        assert config["log_level"] == "DEBUG"
        assert config["parallel_workers"] == 8
        assert config["dashboard_port"] == 9000

    def test_load_yaml_config_invalid(self, tmp_path: Path) -> None:
        """Test loading invalid YAML returns empty dict."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content:")

        config = _load_yaml_config(config_file)
        assert config == {}

    def test_load_yaml_config_missing(self, tmp_path: Path) -> None:
        """Test loading missing file returns empty dict."""
        config = _load_yaml_config(tmp_path / "nonexistent.yaml")
        assert config == {}

    def test_find_config_file_current_dir(self, tmp_path: Path) -> None:
        """Test finding config file in current directory."""
        config_file = tmp_path / "atp.config.yaml"
        config_file.write_text("log_level: INFO")

        found = _find_config_file(tmp_path)
        assert found == config_file

    def test_find_config_file_parent_dir(self, tmp_path: Path) -> None:
        """Test finding config file in parent directory."""
        config_file = tmp_path / "atp.config.yaml"
        config_file.write_text("log_level: INFO")

        subdir = tmp_path / "subdir"
        subdir.mkdir()

        found = _find_config_file(subdir)
        assert found == config_file

    def test_find_config_file_yml_extension(self, tmp_path: Path) -> None:
        """Test finding config file with .yml extension."""
        config_file = tmp_path / "atp.config.yml"
        config_file.write_text("log_level: INFO")

        found = _find_config_file(tmp_path)
        assert found == config_file

    def test_find_config_file_not_found(self, tmp_path: Path) -> None:
        """Test returns None when no config file found."""
        found = _find_config_file(tmp_path)
        assert found is None

    def test_settings_loads_from_config_file(self, tmp_path: Path) -> None:
        """Test ATPSettings loads values from config file."""
        config_file = tmp_path / "atp.config.yaml"
        config_file.write_text("""
log_level: WARNING
parallel_workers: 12
dashboard_port: 7000
""")

        with patch("atp.core.settings._find_config_file", return_value=config_file):
            settings = ATPSettings()
            assert settings.log_level == "WARNING"
            assert settings.parallel_workers == 12
            assert settings.dashboard_port == 7000


class TestConfigHierarchy:
    """Tests for configuration hierarchy (defaults -> file -> env)."""

    def test_env_overrides_file(self, tmp_path: Path) -> None:
        """Test that environment variables override file values."""
        config_file = tmp_path / "atp.config.yaml"
        config_file.write_text("""
log_level: WARNING
parallel_workers: 8
""")

        with patch("atp.core.settings._find_config_file", return_value=config_file):
            with patch.dict(os.environ, {"ATP_LOG_LEVEL": "ERROR"}):
                settings = ATPSettings()
                # Env var overrides file
                assert settings.log_level == "ERROR"
                # File value used when no env var
                assert settings.parallel_workers == 8

    def test_explicit_overrides_all(self, tmp_path: Path) -> None:
        """Test that explicit values override both file and env."""
        config_file = tmp_path / "atp.config.yaml"
        config_file.write_text("""
log_level: WARNING
parallel_workers: 8
""")

        with patch("atp.core.settings._find_config_file", return_value=config_file):
            with patch.dict(os.environ, {"ATP_PARALLEL_WORKERS": "16"}):
                settings = ATPSettings(
                    log_level="CRITICAL",
                    parallel_workers=24,
                )
                # Explicit values override everything
                assert settings.log_level == "CRITICAL"
                assert settings.parallel_workers == 24


class TestNestedSettings:
    """Tests for nested settings groups."""

    def test_dashboard_settings_defaults(self) -> None:
        """Test DashboardSettings default values."""
        settings = DashboardSettings()
        assert settings.host == "127.0.0.1"
        assert settings.port == 8080
        assert settings.debug is False
        assert settings.database_url is None
        assert settings.database_echo is False
        assert settings.token_expire_minutes == 60
        assert settings.cors_origins == "*"

    def test_dashboard_cors_origins_list(self) -> None:
        """Test cors_origins_list property."""
        settings = DashboardSettings(
            cors_origins="http://localhost:3000, http://localhost:8080"
        )
        assert settings.cors_origins_list == [
            "http://localhost:3000",
            "http://localhost:8080",
        ]

    def test_runner_settings_defaults(self) -> None:
        """Test RunnerSettings default values."""
        settings = RunnerSettings()
        assert settings.default_timeout == 300
        assert settings.parallel_workers == 4
        assert settings.runs_per_test == 1
        assert settings.fail_fast is False
        assert settings.sandbox_enabled is False

    def test_llm_settings_defaults(self) -> None:
        """Test LLMSettings default values."""
        # Clear env vars that might affect defaults
        with patch.dict(os.environ, {}, clear=True):
            settings = LLMSettings()
            assert settings.anthropic_api_key is None
            assert settings.openai_api_key is None
            assert settings.default_model == "claude-sonnet-4-20250514"
            assert settings.default_provider == "anthropic"
            assert settings.max_retries == 3
            assert settings.request_timeout == 60

    def test_logging_settings_defaults(self) -> None:
        """Test LoggingSettings default values."""
        settings = LoggingSettings()
        assert settings.level == "INFO"
        assert settings.json_output is False
        assert settings.file is None

    def test_logging_level_validation(self) -> None:
        """Test logging level validation."""
        with pytest.raises(ValueError):
            LoggingSettings(level="INVALID")

    def test_nested_settings_in_atp_settings(self) -> None:
        """Test nested settings are available in ATPSettings."""
        settings = ATPSettings(_skip_file_loading=True)
        assert isinstance(settings.dashboard, DashboardSettings)
        assert isinstance(settings.runner, RunnerSettings)
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.logging, LoggingSettings)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_flatten_dict_simple(self) -> None:
        """Test flattening a simple nested dict."""
        d = {"a": {"b": 1, "c": 2}, "d": 3}
        result = _flatten_dict(d)
        assert result == {"a__b": 1, "a__c": 2, "d": 3}

    def test_flatten_dict_deep(self) -> None:
        """Test flattening a deeply nested dict."""
        d = {"a": {"b": {"c": 1}}}
        result = _flatten_dict(d)
        assert result == {"a__b__c": 1}

    def test_get_settings_factory(self) -> None:
        """Test get_settings factory function."""
        settings = get_settings(
            _skip_file_loading=True,
            log_level="DEBUG",
        )
        assert isinstance(settings, ATPSettings)
        assert settings.log_level == "DEBUG"

    def test_get_settings_with_config_file(self, tmp_path: Path) -> None:
        """Test get_settings with explicit config file."""
        config_file = tmp_path / "custom.yaml"
        config_file.write_text("""
log_level: WARNING
parallel_workers: 10
""")

        settings = get_settings(config_file=config_file)
        assert settings.log_level == "WARNING"
        assert settings.parallel_workers == 10

    def test_get_cached_settings(self) -> None:
        """Test cached settings singleton."""
        # Clear cache first
        get_cached_settings.cache_clear()

        settings1 = get_cached_settings()
        settings2 = get_cached_settings()
        assert settings1 is settings2


class TestSchemaGeneration:
    """Tests for JSON Schema generation."""

    def test_generate_json_schema(self) -> None:
        """Test JSON Schema generation."""
        schema = generate_json_schema()

        assert "$schema" in schema
        assert schema["title"] == "ATP Configuration Schema"
        assert "properties" in schema
        assert "log_level" in schema["properties"]
        assert "parallel_workers" in schema["properties"]
        assert "dashboard" in schema["properties"]

    def test_generate_json_schema_to_file(self, tmp_path: Path) -> None:
        """Test JSON Schema generation to file."""
        output_path = tmp_path / "schema.json"
        schema = generate_json_schema(output_path)

        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded == schema

    def test_generate_example_config(self) -> None:
        """Test example config generation."""
        example = generate_example_config()

        assert "# ATP Platform Configuration" in example
        assert "log_level:" in example
        assert "parallel_workers:" in example
        assert "dashboard_host:" in example
        assert "ATP_ANTHROPIC_API_KEY" in example

    def test_generate_example_config_to_file(self, tmp_path: Path) -> None:
        """Test example config generation to file."""
        output_path = tmp_path / "example.yaml"
        example = generate_example_config(output_path)

        assert output_path.exists()
        with open(output_path) as f:
            content = f.read()
        assert content == example
