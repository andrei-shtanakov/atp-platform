"""Unit tests for plugin configuration schema system."""

import os
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from pydantic import Field, ValidationError

from atp.plugins.config import (
    ConfigMetadata,
    PluginConfig,
    PluginConfigError,
    validate_plugin_config,
)
from atp.plugins.interfaces import (
    get_plugin_config_metadata,
    get_plugin_config_schema,
)


class TestPluginConfig:
    """Tests for PluginConfig base class."""

    def test_basic_config_creation(self) -> None:
        """Test creating a basic plugin config."""

        class BasicConfig(PluginConfig):
            timeout: int = 60
            enabled: bool = True

        config = BasicConfig()
        assert config.timeout == 60
        assert config.enabled is True

    def test_config_with_custom_values(self) -> None:
        """Test creating config with custom values."""

        class BasicConfig(PluginConfig):
            timeout: int = 60
            enabled: bool = True

        config = BasicConfig(timeout=120, enabled=False)
        assert config.timeout == 120
        assert config.enabled is False

    def test_config_validation_with_constraints(self) -> None:
        """Test that field constraints are validated."""

        class ConstrainedConfig(PluginConfig):
            timeout: int = Field(default=60, ge=1, le=300)
            retries: int = Field(default=3, ge=0)

        # Valid values
        config = ConstrainedConfig(timeout=100, retries=5)
        assert config.timeout == 100
        assert config.retries == 5

        # Invalid timeout (too low)
        with pytest.raises(ValidationError):
            ConstrainedConfig(timeout=0)

        # Invalid timeout (too high)
        with pytest.raises(ValidationError):
            ConstrainedConfig(timeout=500)

        # Invalid retries (negative)
        with pytest.raises(ValidationError):
            ConstrainedConfig(retries=-1)

    def test_config_forbids_extra_fields(self) -> None:
        """Test that extra fields are rejected."""

        class StrictConfig(PluginConfig):
            timeout: int = 60

        with pytest.raises(ValidationError):
            StrictConfig(timeout=60, unknown_field="value")

    def test_config_strips_whitespace(self) -> None:
        """Test that string whitespace is stripped."""

        class StringConfig(PluginConfig):
            name: str = "default"

        config = StringConfig(name="  test  ")
        assert config.name == "test"

    def test_config_with_optional_fields(self) -> None:
        """Test configuration with optional fields."""

        class OptionalConfig(PluginConfig):
            api_key: str | None = None
            endpoint: str = "https://api.example.com"

        config = OptionalConfig()
        assert config.api_key is None
        assert config.endpoint == "https://api.example.com"

        config_with_key = OptionalConfig(api_key="secret123")
        assert config_with_key.api_key == "secret123"


class TestPluginConfigEnvSupport:
    """Tests for environment variable support in PluginConfig."""

    def test_from_env_with_prefix(self) -> None:
        """Test loading config from environment variables with prefix."""

        class EnvConfig(PluginConfig):
            env_prefix: ClassVar[str] = "TEST_PLUGIN_"
            timeout: int = 60
            enabled: bool = True
            name: str = "default"

        with patch.dict(
            os.environ,
            {
                "TEST_PLUGIN_TIMEOUT": "120",
                "TEST_PLUGIN_ENABLED": "false",
                "TEST_PLUGIN_NAME": "custom",
            },
        ):
            config = EnvConfig.from_env()
            assert config.timeout == 120
            assert config.enabled is False
            assert config.name == "custom"

    def test_from_env_with_overrides(self) -> None:
        """Test that overrides take precedence over env vars."""

        class EnvConfig(PluginConfig):
            env_prefix: ClassVar[str] = "TEST_PLUGIN_"
            timeout: int = 60
            enabled: bool = True

        with patch.dict(os.environ, {"TEST_PLUGIN_TIMEOUT": "120"}):
            config = EnvConfig.from_env(timeout=180)
            assert config.timeout == 180  # Override wins

    def test_from_env_without_prefix(self) -> None:
        """Test from_env when no prefix is defined."""

        class NoEnvPrefixConfig(PluginConfig):
            timeout: int = 60

        # Should use defaults since no env prefix
        config = NoEnvPrefixConfig.from_env()
        assert config.timeout == 60

    def test_from_env_bool_conversion(self) -> None:
        """Test boolean conversion from environment variables."""

        class BoolConfig(PluginConfig):
            env_prefix: ClassVar[str] = "BOOL_"
            flag1: bool = False
            flag2: bool = False
            flag3: bool = False
            flag4: bool = True

        with patch.dict(
            os.environ,
            {
                "BOOL_FLAG1": "true",
                "BOOL_FLAG2": "1",
                "BOOL_FLAG3": "yes",
                "BOOL_FLAG4": "false",
            },
        ):
            config = BoolConfig.from_env()
            assert config.flag1 is True
            assert config.flag2 is True
            assert config.flag3 is True
            assert config.flag4 is False

    def test_from_env_list_conversion(self) -> None:
        """Test list conversion from comma-separated env var."""

        class ListConfig(PluginConfig):
            env_prefix: ClassVar[str] = "LIST_"
            items: list[str] = Field(default_factory=list)

        with patch.dict(os.environ, {"LIST_ITEMS": "a, b, c"}):
            config = ListConfig.from_env()
            assert config.items == ["a", "b", "c"]

    def test_from_env_optional_field(self) -> None:
        """Test optional field handling with env vars."""

        class OptionalEnvConfig(PluginConfig):
            env_prefix: ClassVar[str] = "OPT_"
            api_key: str | None = None

        # Without env var
        config = OptionalEnvConfig.from_env()
        assert config.api_key is None

        # With env var
        with patch.dict(os.environ, {"OPT_API_KEY": "secret"}):
            config = OptionalEnvConfig.from_env()
            assert config.api_key == "secret"


class TestPluginConfigJsonSchema:
    """Tests for JSON Schema generation from PluginConfig."""

    def test_json_schema_generation(self) -> None:
        """Test JSON Schema is generated correctly."""

        class SchemaConfig(PluginConfig):
            timeout: int = Field(default=60, description="Timeout in seconds")
            enabled: bool = Field(default=True, description="Enable feature")

        schema = SchemaConfig.json_schema()

        assert "properties" in schema
        assert "timeout" in schema["properties"]
        assert "enabled" in schema["properties"]

        # Check descriptions
        assert schema["properties"]["timeout"]["description"] == "Timeout in seconds"
        assert schema["properties"]["enabled"]["description"] == "Enable feature"

    def test_json_schema_with_constraints(self) -> None:
        """Test JSON Schema includes validation constraints."""

        class ConstrainedSchemaConfig(PluginConfig):
            value: int = Field(default=10, ge=0, le=100)

        schema = ConstrainedSchemaConfig.json_schema()

        assert "properties" in schema
        value_schema = schema["properties"]["value"]
        assert value_schema.get("minimum") == 0
        assert value_schema.get("maximum") == 100

    def test_json_schema_string(self) -> None:
        """Test JSON Schema string generation."""

        class SimpleConfig(PluginConfig):
            name: str = "test"

        schema_str = SimpleConfig.json_schema_string()

        assert '"name"' in schema_str
        assert '"string"' in schema_str


class TestPluginConfigExamples:
    """Tests for config examples functionality."""

    def test_get_examples(self) -> None:
        """Test getting configuration examples."""

        class ExampleConfig(PluginConfig):
            config_examples: ClassVar[list[dict[str, Any]]] = [
                {"timeout": 30, "retries": 3},
                {"timeout": 120, "retries": 0},
            ]
            timeout: int = 60
            retries: int = 3

        examples = ExampleConfig.get_examples()

        assert len(examples) == 2
        assert examples[0] == {"timeout": 30, "retries": 3}
        assert examples[1] == {"timeout": 120, "retries": 0}

    def test_get_examples_empty(self) -> None:
        """Test get_examples returns empty list by default."""

        class NoExamplesConfig(PluginConfig):
            timeout: int = 60

        examples = NoExamplesConfig.get_examples()
        assert examples == []

    def test_examples_are_copied(self) -> None:
        """Test that returned examples are copies (not references)."""

        class ExampleConfig(PluginConfig):
            config_examples: ClassVar[list[dict[str, Any]]] = [{"value": 1}]
            value: int = 1

        examples = ExampleConfig.get_examples()
        examples[0]["value"] = 999

        # Original should be unchanged
        assert ExampleConfig.config_examples[0]["value"] == 1


class TestPluginConfigFieldDescriptions:
    """Tests for field description extraction."""

    def test_get_field_descriptions(self) -> None:
        """Test getting field descriptions."""

        class DescribedConfig(PluginConfig):
            timeout: int = Field(default=60, description="Request timeout")
            retries: int = Field(default=3, description="Number of retries")
            name: str = "default"  # No description

        descriptions = DescribedConfig.get_field_descriptions()

        assert descriptions["timeout"] == "Request timeout"
        assert descriptions["retries"] == "Number of retries"
        assert descriptions["name"] == ""  # Empty string for no description


class TestPluginConfigDefaults:
    """Tests for default value extraction."""

    def test_get_defaults(self) -> None:
        """Test getting default values."""

        class DefaultsConfig(PluginConfig):
            timeout: int = 60
            enabled: bool = True
            name: str = "default"

        defaults = DefaultsConfig.get_defaults()

        assert defaults["timeout"] == 60
        assert defaults["enabled"] is True
        assert defaults["name"] == "default"

    def test_get_defaults_with_factory(self) -> None:
        """Test getting defaults with factory functions."""

        class FactoryDefaultsConfig(PluginConfig):
            items: list[str] = Field(default_factory=list)
            tags: dict[str, str] = Field(default_factory=dict)

        defaults = FactoryDefaultsConfig.get_defaults()

        assert defaults["items"] == []
        assert defaults["tags"] == {}


class TestPluginConfigToDict:
    """Tests for config to dictionary conversion."""

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""

        class DictConfig(PluginConfig):
            timeout: int = 60
            enabled: bool = True

        config = DictConfig(timeout=120)
        result = config.to_dict()

        assert result == {"timeout": 120, "enabled": True}


class TestValidatePluginConfig:
    """Tests for validate_plugin_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid configuration."""

        class ValidConfig(PluginConfig):
            timeout: int = 60

        result = validate_plugin_config(ValidConfig, {"timeout": 120}, "test_plugin")

        assert isinstance(result, ValidConfig)
        assert result.timeout == 120

    def test_invalid_config_raises_error(self) -> None:
        """Test that invalid config raises PluginConfigError."""

        class StrictConfig(PluginConfig):
            timeout: int = Field(default=60, ge=1)

        with pytest.raises(PluginConfigError) as exc_info:
            validate_plugin_config(StrictConfig, {"timeout": -1}, "test_plugin")

        error = exc_info.value
        assert error.plugin_name == "test_plugin"
        assert len(error.errors) > 0

    def test_extra_field_raises_error(self) -> None:
        """Test that extra fields raise PluginConfigError."""

        class SimpleConfig(PluginConfig):
            timeout: int = 60

        with pytest.raises(PluginConfigError) as exc_info:
            validate_plugin_config(
                SimpleConfig, {"timeout": 60, "extra": "field"}, "test_plugin"
            )

        assert "test_plugin" in str(exc_info.value)


class TestPluginConfigError:
    """Tests for PluginConfigError exception."""

    def test_error_message_formatting(self) -> None:
        """Test error message is properly formatted."""
        error = PluginConfigError(
            plugin_name="my_plugin",
            errors=["Invalid timeout", "Missing required field"],
        )

        assert "my_plugin" in str(error)
        assert "Invalid timeout" in str(error)
        assert "Missing required field" in str(error)

    def test_error_attributes(self) -> None:
        """Test error has correct attributes."""
        error = PluginConfigError(
            plugin_name="test",
            errors=["Error 1", "Error 2"],
        )

        assert error.plugin_name == "test"
        assert error.errors == ["Error 1", "Error 2"]


class TestConfigMetadata:
    """Tests for ConfigMetadata model."""

    def test_from_config_class(self) -> None:
        """Test creating metadata from config class."""

        class MetadataTestConfig(PluginConfig):
            env_prefix: ClassVar[str] = "META_"
            config_examples: ClassVar[list[dict[str, Any]]] = [{"timeout": 30}]

            timeout: int = Field(default=60, description="Timeout seconds")

        metadata = ConfigMetadata.from_config_class(MetadataTestConfig)

        assert metadata.env_prefix == "META_"
        assert metadata.examples == [{"timeout": 30}]
        assert metadata.field_descriptions["timeout"] == "Timeout seconds"
        assert metadata.defaults["timeout"] == 60
        assert "properties" in metadata.schema_

    def test_metadata_schema_alias(self) -> None:
        """Test that schema field uses alias 'schema' in serialization."""

        class SimpleTestConfig(PluginConfig):
            value: int = 1

        metadata = ConfigMetadata.from_config_class(SimpleTestConfig)
        dumped = metadata.model_dump(by_alias=True)

        assert "schema" in dumped
        assert "schema_" not in dumped


class TestGetPluginConfigSchema:
    """Tests for get_plugin_config_schema function."""

    def test_plugin_with_config_schema(self) -> None:
        """Test getting config schema from plugin with one defined."""

        class TestPluginConfig(PluginConfig):
            timeout: int = 60

        class PluginWithConfig:
            config_schema: ClassVar[type[PluginConfig]] = TestPluginConfig

        result = get_plugin_config_schema(PluginWithConfig)
        assert result is TestPluginConfig

    def test_plugin_without_config_schema(self) -> None:
        """Test getting config schema from plugin without one."""

        class PluginWithoutConfig:
            pass

        result = get_plugin_config_schema(PluginWithoutConfig)
        assert result is None

    def test_plugin_with_none_config_schema(self) -> None:
        """Test plugin with config_schema set to None."""

        class PluginWithNoneConfig:
            config_schema = None

        result = get_plugin_config_schema(PluginWithNoneConfig)
        assert result is None

    def test_plugin_with_invalid_config_schema(self) -> None:
        """Test plugin with invalid config_schema type."""

        class PluginWithInvalidConfig:
            config_schema = "not a class"

        result = get_plugin_config_schema(PluginWithInvalidConfig)
        assert result is None


class TestGetPluginConfigMetadata:
    """Tests for get_plugin_config_metadata function."""

    def test_plugin_with_config_returns_metadata(self) -> None:
        """Test getting config metadata from plugin with config schema."""

        class TestPluginConfig(PluginConfig):
            env_prefix: ClassVar[str] = "TEST_"
            timeout: int = Field(default=60, description="Timeout")

        class PluginWithConfig:
            config_schema: ClassVar[type[PluginConfig]] = TestPluginConfig

        result = get_plugin_config_metadata(PluginWithConfig)

        assert result is not None
        assert "schema" in result
        assert result["env_prefix"] == "TEST_"

    def test_plugin_without_config_returns_none(self) -> None:
        """Test getting config metadata from plugin without config schema."""

        class PluginWithoutConfig:
            pass

        result = get_plugin_config_metadata(PluginWithoutConfig)
        assert result is None


class TestPluginConfigIntegration:
    """Integration tests for plugin config system."""

    def test_full_config_workflow(self) -> None:
        """Test complete workflow: define, validate, generate docs."""

        # Define a config class
        class MyAdapterConfig(PluginConfig):
            env_prefix: ClassVar[str] = "MY_ADAPTER_"
            config_examples: ClassVar[list[dict[str, Any]]] = [
                {"timeout": 30, "retries": 3, "debug": False},
                {"timeout": 120, "retries": 0, "debug": True},
            ]

            timeout: int = Field(
                default=60, ge=1, le=300, description="Request timeout in seconds"
            )
            retries: int = Field(
                default=3, ge=0, le=10, description="Number of retry attempts"
            )
            debug: bool = Field(default=False, description="Enable debug logging")

        # Test creating with defaults
        config = MyAdapterConfig()
        assert config.timeout == 60
        assert config.retries == 3
        assert config.debug is False

        # Test creating with custom values
        config = MyAdapterConfig(timeout=120, retries=5, debug=True)
        assert config.timeout == 120
        assert config.retries == 5
        assert config.debug is True

        # Test validation
        with pytest.raises(ValidationError):
            MyAdapterConfig(timeout=500)  # Too high

        # Test JSON Schema generation
        schema = MyAdapterConfig.json_schema()
        assert "properties" in schema
        assert len(schema["properties"]) == 3

        # Test examples
        examples = MyAdapterConfig.get_examples()
        assert len(examples) == 2

        # Test environment loading
        with patch.dict(
            os.environ,
            {"MY_ADAPTER_TIMEOUT": "90", "MY_ADAPTER_DEBUG": "true"},
        ):
            config = MyAdapterConfig.from_env()
            assert config.timeout == 90
            assert config.debug is True

        # Test config metadata
        metadata = ConfigMetadata.from_config_class(MyAdapterConfig)
        assert metadata.env_prefix == "MY_ADAPTER_"
        assert len(metadata.examples) == 2
        assert "timeout" in metadata.field_descriptions
