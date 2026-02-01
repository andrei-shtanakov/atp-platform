"""ATP Platform Configuration Management.

This module provides centralized, hierarchical configuration management for
ATP Platform. Configuration is loaded from multiple sources with the following
priority (highest to lowest):
1. CLI arguments (when applicable)
2. Environment variables (with ATP_ prefix)
3. Configuration files (atp.config.yaml, atp.config.{env}.yaml)
4. Default values

Example usage:
    from atp.core.settings import get_settings

    settings = get_settings()
    print(settings.log_level)

Environment variable support:
    ATP_LOG_LEVEL=DEBUG
    ATP_PARALLEL_WORKERS=8
    ATP_ANTHROPIC_API_KEY=sk-...
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Default config file names to search for
CONFIG_FILE_NAMES = ["atp.config.yaml", "atp.config.yml"]


def _find_config_file(start_dir: Path | None = None) -> Path | None:
    """Find configuration file by searching current directory and parents.

    Args:
        start_dir: Directory to start search from.
            Defaults to current working directory.

    Returns:
        Path to config file if found, None otherwise.
    """
    search_dir = start_dir or Path.cwd()

    # Limit search depth to prevent infinite loops
    for _ in range(10):
        for filename in CONFIG_FILE_NAMES:
            config_path = search_dir / filename
            if config_path.exists():
                return config_path

        parent = search_dir.parent
        if parent == search_dir:
            break
        search_dir = parent

    return None


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration values.
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
            return config if isinstance(config, dict) else {}
    except yaml.YAMLError as e:
        logger.warning("Failed to parse config file %s: %s", config_path, e)
        return {}
    except OSError as e:
        logger.warning("Failed to read config file %s: %s", config_path, e)
        return {}


def _flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "__"
) -> dict[str, Any]:
    """Flatten a nested dictionary for environment variable mapping.

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for nested keys.
        sep: Separator between key levels.

    Returns:
        Flattened dictionary.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class DashboardSettings(BaseSettings):
    """Dashboard-specific settings.

    These settings control the ATP Dashboard web interface.
    """

    host: str = Field(
        default="127.0.0.1",
        description="Dashboard server host address",
    )
    port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="Dashboard server port",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode for development",
    )
    database_url: str | None = Field(
        default=None,
        description="Database URL. Defaults to SQLite at ~/.atp/dashboard.db",
    )
    database_echo: bool = Field(
        default=False,
        description="Echo SQL statements for debugging",
    )
    secret_key: SecretStr = Field(
        default=SecretStr("atp-dashboard-dev-secret-key-change-in-prod"),
        description="Secret key for JWT token signing. Change in production!",
    )
    token_expire_minutes: int = Field(
        default=60,
        ge=1,
        description="JWT token expiration time in minutes",
    )
    cors_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]


class RunnerSettings(BaseSettings):
    """Test runner settings.

    These settings control test execution behavior.
    """

    default_timeout: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Default test timeout in seconds (1-3600)",
    )
    parallel_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of parallel test workers (1-32)",
    )
    runs_per_test: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Default number of runs per test (1-100)",
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop test suite on first failure",
    )
    sandbox_enabled: bool = Field(
        default=False,
        description="Enable sandbox isolation by default",
    )


class LLMSettings(BaseSettings):
    """LLM provider settings.

    These settings configure LLM API access for evaluators and agents.
    All API keys are stored as SecretStr for security.
    """

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key for Claude models",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key for GPT models",
    )
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default LLM model for evaluators",
    )
    default_provider: str = Field(
        default="anthropic",
        description="Default LLM provider (anthropic, openai)",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum API call retries (0-10)",
    )
    request_timeout: int = Field(
        default=60,
        ge=10,
        le=600,
        description="API request timeout in seconds (10-600)",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format string",
    )
    json_output: bool = Field(
        default=False,
        description="Output logs in JSON format",
    )
    file: str | None = Field(
        default=None,
        description="Log file path (optional)",
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper_v


class ATPSettings(BaseSettings):
    """Main ATP Platform configuration settings.

    This class provides centralized configuration management with:
    - Environment variable support (ATP_ prefix)
    - .env file loading
    - YAML config file loading (atp.config.yaml)
    - Hierarchical merging: defaults → file → env → CLI
    - Type validation via Pydantic
    - Secret value protection (API keys masked in logs)

    Example:
        # Basic usage
        settings = ATPSettings()
        print(settings.log_level)

        # With custom values
        settings = ATPSettings(log_level="DEBUG", parallel_workers=8)

        # Access nested settings
        print(settings.dashboard.port)
        print(settings.llm.default_model)
    """

    model_config = SettingsConfigDict(
        env_prefix="ATP_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        validate_default=True,
    )

    # Core settings
    log_level: str = Field(
        default="INFO",
        description="Application log level",
    )
    parallel_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of parallel workers for test execution",
    )
    default_timeout: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Default timeout for test execution in seconds",
    )

    # LLM API keys (using SecretStr for security)
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key for Claude models",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key for GPT models",
    )
    default_llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default LLM model for evaluators and agents",
    )

    # Dashboard settings
    dashboard_host: str = Field(
        default="127.0.0.1",
        description="Dashboard server host",
    )
    dashboard_port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="Dashboard server port",
    )
    dashboard_debug: bool = Field(
        default=False,
        description="Enable dashboard debug mode",
    )
    dashboard_secret_key: SecretStr = Field(
        default=SecretStr("atp-dashboard-dev-secret-key-change-in-prod"),
        description="Dashboard JWT secret key",
    )

    # Database settings
    database_url: str | None = Field(
        default=None,
        description="Database URL. Defaults to SQLite at ~/.atp/dashboard.db",
    )

    # Runner settings
    fail_fast: bool = Field(
        default=False,
        description="Stop test suite execution on first failure",
    )
    sandbox_enabled: bool = Field(
        default=False,
        description="Enable sandbox isolation for test execution",
    )
    runs_per_test: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Default number of runs per test",
    )

    # Nested settings groups
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    runner: RunnerSettings = Field(default_factory=RunnerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Internal: path to loaded config file
    _config_file_path: Path | None = None

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper_v

    @model_validator(mode="before")
    @classmethod
    def load_from_config_file(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Load configuration from YAML file and merge with provided data.

        This validator runs before field validation and merges config file
        values with any explicitly provided values.
        """
        # Skip file loading if explicitly disabled
        if data.get("_skip_file_loading"):
            data.pop("_skip_file_loading", None)
            return data

        # Find and load config file
        config_path = _find_config_file()
        if config_path:
            file_config = _load_yaml_config(config_path)
            if file_config:
                logger.debug("Loaded configuration from %s", config_path)

                # Flatten nested config for easier merging
                flattened = _flatten_dict(file_config)

                # Merge: file config provides defaults, data overrides
                merged = {**flattened, **data}

                # Also merge nested objects directly
                for section in ["dashboard", "runner", "llm", "logging"]:
                    if section in file_config and isinstance(
                        file_config[section], dict
                    ):
                        if section not in merged or not isinstance(
                            merged[section], dict
                        ):
                            merged[section] = {}
                        merged[section] = {
                            **file_config[section],
                            **(
                                data.get(section, {})
                                if isinstance(data.get(section), dict)
                                else {}
                            ),
                        }

                return merged

        return data

    def to_dict(self, mask_secrets: bool = True) -> dict[str, Any]:
        """Convert settings to dictionary.

        Args:
            mask_secrets: If True, mask sensitive values like API keys.

        Returns:
            Dictionary representation of settings.
        """
        result: dict[str, Any] = {}

        for field_name, field_info in type(self).model_fields.items():
            value = getattr(self, field_name)

            if isinstance(value, SecretStr):
                result[field_name] = "***" if mask_secrets else value.get_secret_value()
            elif isinstance(value, BaseSettings):
                # Handle nested settings
                nested: dict[str, Any] = {}
                for nested_field in type(value).model_fields:
                    nested_value = getattr(value, nested_field)
                    if isinstance(nested_value, SecretStr):
                        nested[nested_field] = (
                            "***" if mask_secrets else nested_value.get_secret_value()
                        )
                    else:
                        nested[nested_field] = nested_value
                result[field_name] = nested
            else:
                result[field_name] = value

        return result

    def get_llm_api_key(self, provider: str | None = None) -> str | None:
        """Get API key for the specified LLM provider.

        Args:
            provider: LLM provider name (anthropic, openai). If None, uses default.

        Returns:
            API key string or None if not configured.
        """
        provider = provider or self.llm.default_provider

        if provider.lower() == "anthropic":
            return (
                self.anthropic_api_key.get_secret_value()
                if self.anthropic_api_key
                else None
            )
        elif provider.lower() == "openai":
            return (
                self.openai_api_key.get_secret_value() if self.openai_api_key else None
            )
        return None


def get_settings(
    config_file: Path | None = None,
    **overrides: Any,
) -> ATPSettings:
    """Get ATP settings instance.

    This function creates and returns an ATPSettings instance with configuration
    loaded from multiple sources in order of precedence:
    1. Explicit overrides (passed as kwargs)
    2. Environment variables (ATP_ prefix)
    3. Configuration file (atp.config.yaml)
    4. Default values

    Args:
        config_file: Optional explicit path to configuration file.
        **overrides: Explicit configuration overrides.

    Returns:
        Configured ATPSettings instance.

    Example:
        # Use defaults + env vars + config file
        settings = get_settings()

        # Override specific values
        settings = get_settings(log_level="DEBUG", parallel_workers=8)

        # Use specific config file
        settings = get_settings(config_file=Path("custom.yaml"))
    """
    if config_file and config_file.exists():
        file_config = _load_yaml_config(config_file)
        merged = {**file_config, **overrides}
        return ATPSettings(**merged)

    return ATPSettings(**overrides)


@lru_cache
def get_cached_settings() -> ATPSettings:
    """Get cached settings instance.

    Returns a cached singleton instance of ATPSettings. Use this when you need
    consistent settings throughout the application and don't need to override values.

    Returns:
        Cached ATPSettings instance.

    Note:
        The cache can be cleared with get_cached_settings.cache_clear() if needed.
    """
    return get_settings()


def generate_json_schema(output_path: Path | None = None) -> dict[str, Any]:
    """Generate JSON Schema for ATP configuration.

    This schema can be used for IDE autocompletion and validation
    of atp.config.yaml files.

    Args:
        output_path: Optional path to write schema file.

    Returns:
        JSON Schema dictionary.
    """
    schema = ATPSettings.model_json_schema()

    # Add schema metadata
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "ATP Configuration Schema"
    schema["description"] = (
        "JSON Schema for ATP Platform configuration files (atp.config.yaml)"
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)
        logger.info("Generated JSON Schema at %s", output_path)

    return schema


def generate_example_config(output_path: Path | None = None) -> str:
    """Generate example configuration file.

    Creates a commented YAML configuration file with all available options
    and their default values.

    Args:
        output_path: Optional path to write example config file.

    Returns:
        Example configuration as YAML string.
    """
    example = """\
# ATP Platform Configuration
# See documentation for complete configuration reference
# Environment variables can override these values with ATP_ prefix
# Example: ATP_LOG_LEVEL=DEBUG

# Core settings
log_level: INFO              # DEBUG, INFO, WARNING, ERROR, CRITICAL
parallel_workers: 4          # Number of parallel test workers (1-32)
default_timeout: 300         # Default test timeout in seconds

# LLM settings
# anthropic_api_key: sk-...  # Set via ATP_ANTHROPIC_API_KEY env var
# openai_api_key: sk-...     # Set via ATP_OPENAI_API_KEY env var
default_llm_model: claude-sonnet-4-20250514

# Dashboard settings
dashboard_host: 127.0.0.1    # Dashboard bind address
dashboard_port: 8080         # Dashboard port
dashboard_debug: false       # Enable debug mode

# Database settings
# database_url: postgresql+asyncpg://user:pass@localhost/atp

# Runner settings
fail_fast: false             # Stop on first test failure
sandbox_enabled: false       # Enable sandbox isolation
runs_per_test: 1             # Default runs per test

# Nested settings (alternative format)
# dashboard:
#   host: 127.0.0.1
#   port: 8080
#   debug: false
#   cors_origins: "*"
#   secret_key: change-me-in-production
#   token_expire_minutes: 60

# runner:
#   default_timeout: 300
#   parallel_workers: 4
#   fail_fast: false
#   sandbox_enabled: false

# llm:
#   default_provider: anthropic
#   default_model: claude-sonnet-4-20250514
#   max_retries: 3
#   request_timeout: 60

# logging:
#   level: INFO
#   json_output: false
#   file: null  # Optional log file path
"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(example)
        logger.info("Generated example config at %s", output_path)

    return example
