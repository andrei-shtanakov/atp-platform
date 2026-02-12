"""Plugin configuration schema system.

This module provides a base configuration model for plugins with validation,
default values, environment variable support, and auto-generated JSON Schema
documentation.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class PluginConfigError(Exception):
    """Raised when plugin configuration validation fails."""

    def __init__(
        self,
        plugin_name: str,
        errors: list[str],
    ) -> None:
        """Initialize plugin configuration error.

        Args:
            plugin_name: Name of the plugin with config issues.
            errors: List of validation error messages.
        """
        self.plugin_name = plugin_name
        self.errors = errors
        error_list = "\n  - ".join(errors)
        message = (
            f"Plugin '{plugin_name}' configuration validation failed:\n  - {error_list}"
        )
        super().__init__(message)


class PluginConfig(BaseModel):
    """Base configuration model for ATP plugins.

    This class provides a foundation for plugin-specific configuration with:
    - Pydantic validation
    - Environment variable support via env_prefix
    - JSON Schema generation
    - Default values
    - Configuration examples

    Plugins should subclass this and define their configuration fields.

    Example:
        class MyAdapterConfig(PluginConfig):
            '''Configuration for MyAdapter plugin.'''

            # Class-level metadata
            env_prefix: ClassVar[str] = "MY_ADAPTER_"
            config_examples: ClassVar[list[dict[str, Any]]] = [
                {"timeout": 30, "retries": 3},
            ]

            # Configuration fields
            timeout: int = Field(
                default=60,
                description="Request timeout in seconds",
                ge=1,
                le=300,
            )
            retries: int = Field(
                default=3,
                description="Number of retry attempts",
                ge=0,
            )
            api_key: str | None = Field(
                default=None,
                description="API key for authentication",
            )

    Attributes:
        env_prefix: Prefix for environment variable overrides.
        config_examples: List of example configuration dictionaries.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    # Class-level constants for subclasses
    env_prefix: ClassVar[str] = ""
    config_examples: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def from_env(cls, **overrides: Any) -> PluginConfig:
        """Create configuration from environment variables.

        Reads environment variables with the class's env_prefix and
        merges them with any explicit overrides.

        Args:
            **overrides: Explicit configuration values that take precedence
                over environment variables.

        Returns:
            PluginConfig instance with values from environment and overrides.

        Example:
            # With MY_ADAPTER_TIMEOUT=120 in environment
            config = MyAdapterConfig.from_env(retries=5)
            # config.timeout = 120 (from env)
            # config.retries = 5 (from override)
        """
        env_values: dict[str, Any] = {}

        if cls.env_prefix:
            for field_name, field_info in cls.model_fields.items():
                env_key = f"{cls.env_prefix}{field_name.upper()}"
                env_value = os.environ.get(env_key)
                if env_value is not None:
                    # Convert string to appropriate type
                    env_values[field_name] = cls._convert_env_value(
                        env_value, field_info.annotation
                    )

        # Overrides take precedence over environment variables
        merged = {**env_values, **overrides}
        return cls(**merged)

    @classmethod
    def _convert_env_value(cls, value: str, annotation: Any) -> Any:
        """Convert environment variable string to appropriate type.

        Args:
            value: String value from environment.
            annotation: Type annotation for the field.

        Returns:
            Converted value appropriate for the field type.
        """
        # Handle Optional types
        if hasattr(annotation, "__origin__"):
            args = getattr(annotation, "__args__", ())
            # For Optional[X], get X
            if type(None) in args:
                non_none_args = [a for a in args if a is not type(None)]
                if non_none_args:
                    annotation = non_none_args[0]

        # Convert based on type
        if annotation is bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif annotation is int:
            return int(value)
        elif annotation is float:
            return float(value)
        elif annotation is list or (
            hasattr(annotation, "__origin__") and annotation.__origin__ is list
        ):
            # Comma-separated list
            return [v.strip() for v in value.split(",") if v.strip()]

        return value

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Generate JSON Schema for this configuration.

        Returns:
            JSON Schema dictionary for the configuration model.
        """
        return cls.model_json_schema()

    @classmethod
    def json_schema_string(cls) -> str:
        """Generate JSON Schema as formatted string.

        Returns:
            JSON Schema as formatted JSON string.
        """
        import json

        return json.dumps(cls.json_schema(), indent=2)

    @classmethod
    def get_examples(cls) -> list[dict[str, Any]]:
        """Get configuration examples.

        Returns:
            List of example configuration dictionaries (deep copy).
        """
        import copy

        return copy.deepcopy(cls.config_examples)

    @classmethod
    def get_field_descriptions(cls) -> dict[str, str]:
        """Get descriptions for all configuration fields.

        Returns:
            Dictionary mapping field names to their descriptions.
        """
        descriptions: dict[str, str] = {}
        for field_name, field_info in cls.model_fields.items():
            description = field_info.description or ""
            descriptions[field_name] = description
        return descriptions

    @classmethod
    def get_defaults(cls) -> dict[str, Any]:
        """Get default values for all configuration fields.

        Returns:
            Dictionary mapping field names to their default values.
        """
        from pydantic_core import PydanticUndefined

        defaults: dict[str, Any] = {}
        for field_name, field_info in cls.model_fields.items():
            # Check for default_factory first (returns PydanticUndefined for default)
            if field_info.default_factory is not None:
                factory = field_info.default_factory
                defaults[field_name] = factory()  # pyrefly: ignore[bad-argument-count]
            elif field_info.default is not PydanticUndefined:
                defaults[field_name] = field_info.default
        return defaults

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return self.model_dump()


def validate_plugin_config(
    config_class: type[PluginConfig],
    config_data: dict[str, Any],
    plugin_name: str,
) -> PluginConfig:
    """Validate plugin configuration data against its config schema.

    Args:
        config_class: The PluginConfig subclass to validate against.
        config_data: Configuration data dictionary to validate.
        plugin_name: Name of the plugin (for error messages).

    Returns:
        Validated PluginConfig instance.

    Raises:
        PluginConfigError: If validation fails.
    """
    try:
        return config_class(**config_data)
    except Exception as e:
        errors = _extract_validation_errors(e)
        raise PluginConfigError(plugin_name, errors) from e


def _extract_validation_errors(error: Exception) -> list[str]:
    """Extract validation error messages from a Pydantic ValidationError.

    Args:
        error: The exception to extract errors from.

    Returns:
        List of error message strings.
    """
    from pydantic import ValidationError

    errors: list[str] = []

    if isinstance(error, ValidationError):
        for err in error.errors():
            loc = ".".join(str(part) for part in err.get("loc", []))
            msg = err.get("msg", "Unknown error")
            if loc:
                errors.append(f"{loc}: {msg}")
            else:
                errors.append(msg)
    else:
        errors.append(str(error))

    return errors


class ConfigMetadata(BaseModel):
    """Metadata about a plugin's configuration schema.

    This model provides documentation-friendly metadata about
    a plugin's configuration options.

    Attributes:
        schema: JSON Schema for the configuration.
        examples: List of example configuration dictionaries.
        field_descriptions: Map of field names to descriptions.
        defaults: Map of field names to default values.
        env_prefix: Environment variable prefix for this config.
    """

    schema_: dict[str, Any] = Field(..., alias="schema")
    examples: list[dict[str, Any]] = Field(default_factory=list)
    field_descriptions: dict[str, str] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    env_prefix: str = ""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @classmethod
    def from_config_class(cls, config_class: type[PluginConfig]) -> ConfigMetadata:
        """Create ConfigMetadata from a PluginConfig subclass.

        Args:
            config_class: The PluginConfig subclass to extract metadata from.

        Returns:
            ConfigMetadata instance with all documentation.
        """
        return cls(
            schema=config_class.json_schema(),
            examples=config_class.get_examples(),
            field_descriptions=config_class.get_field_descriptions(),
            defaults=config_class.get_defaults(),
            env_prefix=config_class.env_prefix,
        )
