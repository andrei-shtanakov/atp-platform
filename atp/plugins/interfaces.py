"""Plugin interface protocols and validation.

This module defines the protocol interfaces that plugins must implement
to be compatible with the ATP plugin system. It also provides validation
utilities for checking plugin compliance.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from atp.evaluators.base import EvalResult
    from atp.loader.models import Assertion, TestDefinition
    from atp.plugins.config import PluginConfig
    from atp.protocol import ATPEvent, ATPRequest, ATPResponse
    from atp.reporters.base import SuiteReport


# Minimum ATP version required for plugin compatibility
MIN_ATP_VERSION = "0.1.0"


class PluginValidationError(Exception):
    """Raised when a plugin fails validation."""

    def __init__(
        self,
        plugin_name: str,
        plugin_type: str,
        errors: list[str],
    ) -> None:
        """Initialize plugin validation error.

        Args:
            plugin_name: Name of the plugin that failed validation.
            plugin_type: Type of plugin (adapter, evaluator, reporter).
            errors: List of validation error messages.
        """
        self.plugin_name = plugin_name
        self.plugin_type = plugin_type
        self.errors = errors
        error_list = "\n  - ".join(errors)
        message = (
            f"Plugin '{plugin_name}' ({plugin_type}) failed validation:\n"
            f"  - {error_list}"
        )
        super().__init__(message)


class PluginVersionError(Exception):
    """Raised when a plugin has incompatible version requirements."""

    def __init__(
        self,
        plugin_name: str,
        required_version: str,
        current_version: str,
    ) -> None:
        """Initialize plugin version error.

        Args:
            plugin_name: Name of the plugin with version issues.
            required_version: The version required by the plugin.
            current_version: The current ATP version.
        """
        self.plugin_name = plugin_name
        self.required_version = required_version
        self.current_version = current_version
        message = (
            f"Plugin '{plugin_name}' requires ATP version {required_version}, "
            f"but current version is {current_version}"
        )
        super().__init__(message)


@runtime_checkable
class AdapterPlugin(Protocol):
    """Protocol defining the interface for adapter plugins.

    Adapters translate between the ATP Protocol and agent-specific
    communication mechanisms (HTTP, Docker, CLI, etc.).

    Required class/instance attributes:
        adapter_type: String identifier for the adapter type.

    Required methods:
        execute: Execute a task synchronously.
        stream_events: Execute a task with event streaming.

    Optional class attributes:
        atp_version: Minimum ATP version required (default: "0.1.0").
        config_schema: PluginConfig subclass defining configuration options.

    Optional methods:
        health_check: Check if the agent is available.
        cleanup: Release resources held by the adapter.

    Example:
        class MyAdapterConfig(PluginConfig):
            '''Configuration for MyAdapter.'''
            env_prefix: ClassVar[str] = "MY_ADAPTER_"
            timeout: int = Field(default=60, description="Timeout in seconds")

        class MyAdapter:
            adapter_type = "my_adapter"
            atp_version = "0.1.0"  # Optional
            config_schema = MyAdapterConfig  # Optional

            async def execute(self, request: ATPRequest) -> ATPResponse:
                ...

            async def stream_events(
                self, request: ATPRequest
            ) -> AsyncIterator[ATPEvent | ATPResponse]:
                ...
    """

    # Class-level constants
    atp_version: ClassVar[str]
    config_schema: ClassVar[type[PluginConfig] | None]

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""
        ...

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """Execute a task synchronously.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse with execution results.
        """
        ...

    def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """Execute a task with event streaming.

        Args:
            request: ATP Request with task specification.

        Yields:
            ATPEvent objects during execution.
            Final ATPResponse when complete.
        """
        ...


@runtime_checkable
class EvaluatorPlugin(Protocol):
    """Protocol defining the interface for evaluator plugins.

    Evaluators assess agent results against assertions defined in tests.

    Required class/instance attributes:
        name: String identifier for the evaluator.

    Required methods:
        evaluate: Evaluate agent results against an assertion.

    Optional class attributes:
        atp_version: Minimum ATP version required (default: "0.1.0").
        config_schema: PluginConfig subclass defining configuration options.

    Example:
        class MyEvaluatorConfig(PluginConfig):
            '''Configuration for MyEvaluator.'''
            threshold: float = Field(default=0.8, description="Score threshold")

        class MyEvaluator:
            name = "my_evaluator"
            atp_version = "0.1.0"  # Optional
            config_schema = MyEvaluatorConfig  # Optional

            async def evaluate(
                self,
                task: TestDefinition,
                response: ATPResponse,
                trace: list[ATPEvent],
                assertion: Assertion,
            ) -> EvalResult:
                ...
    """

    # Class-level constants
    atp_version: ClassVar[str]
    config_schema: ClassVar[type[PluginConfig] | None]

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        ...

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Evaluate agent results against an assertion.

        Args:
            task: Test definition containing task details.
            response: ATP Response from the agent.
            trace: List of ATP Events from execution.
            assertion: Assertion to evaluate against.

        Returns:
            EvalResult containing check results.
        """
        ...


@runtime_checkable
class ReporterPlugin(Protocol):
    """Protocol defining the interface for reporter plugins.

    Reporters format and output test results in various formats.

    Required class/instance attributes:
        name: String identifier for the reporter.

    Required methods:
        report: Generate and output the report.

    Optional class attributes:
        atp_version: Minimum ATP version required (default: "0.1.0").
        supports_streaming: Whether the reporter supports streaming output.
        config_schema: PluginConfig subclass defining configuration options.

    Example:
        class MyReporterConfig(PluginConfig):
            '''Configuration for MyReporter.'''
            output_path: str = Field(default="report.html", description="Output path")

        class MyReporter:
            name = "my_reporter"
            atp_version = "0.1.0"  # Optional
            supports_streaming = False  # Optional
            config_schema = MyReporterConfig  # Optional

            def report(self, report: SuiteReport) -> None:
                ...
    """

    # Class-level constants
    atp_version: ClassVar[str]
    config_schema: ClassVar[type[PluginConfig] | None]

    @property
    def name(self) -> str:
        """Return the reporter name."""
        ...

    def report(self, report: SuiteReport) -> None:
        """Generate and output the report.

        Args:
            report: Suite report data to output.
        """
        ...


# Type alias for any plugin protocol
PluginProtocol = AdapterPlugin | EvaluatorPlugin | ReporterPlugin


def get_plugin_protocol_for_group(group: str) -> type[PluginProtocol] | None:
    """Get the appropriate protocol for a plugin group.

    Args:
        group: Entry point group name (e.g., 'atp.adapters').

    Returns:
        The protocol class for the group, or None if unknown.
    """
    from atp.plugins.discovery import ADAPTER_GROUP, EVALUATOR_GROUP, REPORTER_GROUP

    protocol_map: dict[str, type[PluginProtocol]] = {
        ADAPTER_GROUP: AdapterPlugin,
        EVALUATOR_GROUP: EvaluatorPlugin,
        REPORTER_GROUP: ReporterPlugin,
    }
    return protocol_map.get(group)


def get_required_attributes(group: str) -> list[str]:
    """Get the required attributes for a plugin group.

    Args:
        group: Entry point group name.

    Returns:
        List of required attribute names.
    """
    from atp.plugins.discovery import ADAPTER_GROUP, EVALUATOR_GROUP, REPORTER_GROUP

    required_attrs: dict[str, list[str]] = {
        ADAPTER_GROUP: ["adapter_type", "execute", "stream_events"],
        EVALUATOR_GROUP: ["name", "evaluate"],
        REPORTER_GROUP: ["name", "report"],
    }
    return required_attrs.get(group, [])


def get_required_methods(group: str) -> list[str]:
    """Get the required methods for a plugin group.

    Args:
        group: Entry point group name.

    Returns:
        List of required method names.
    """
    from atp.plugins.discovery import ADAPTER_GROUP, EVALUATOR_GROUP, REPORTER_GROUP

    required_methods: dict[str, list[str]] = {
        ADAPTER_GROUP: ["execute", "stream_events"],
        EVALUATOR_GROUP: ["evaluate"],
        REPORTER_GROUP: ["report"],
    }
    return required_methods.get(group, [])


def validate_plugin(
    plugin_class: type[Any],
    group: str,
    plugin_name: str | None = None,
) -> list[str]:
    """Validate that a plugin class implements the required interface.

    This function checks that the plugin class has all required attributes
    and methods for the given plugin group.

    Args:
        plugin_class: The plugin class to validate.
        group: Entry point group name (e.g., 'atp.adapters').
        plugin_name: Optional name for error messages.

    Returns:
        List of validation error messages (empty if valid).
    """
    from atp.plugins.discovery import ADAPTER_GROUP, EVALUATOR_GROUP, REPORTER_GROUP

    errors: list[str] = []

    # Check if it's a known group
    if group not in (ADAPTER_GROUP, EVALUATOR_GROUP, REPORTER_GROUP):
        errors.append(f"Unknown plugin group: {group}")
        return errors

    # Get required attributes and methods
    required_attrs = get_required_attributes(group)

    # Check each required attribute/method
    for attr in required_attrs:
        if not hasattr(plugin_class, attr):
            errors.append(f"Missing required attribute or method: {attr}")
            continue

        # For methods, check they are callable
        attr_value = getattr(plugin_class, attr)
        if attr in get_required_methods(group):
            if not callable(attr_value):
                attr_type = type(attr_value).__name__
                errors.append(f"'{attr}' must be a callable method, not {attr_type}")

    # Check for protocol compliance using isinstance with Protocol
    protocol = get_plugin_protocol_for_group(group)
    if protocol is not None:
        try:
            # Create a dummy instance to check protocol compliance
            # We can't use isinstance directly on classes
            if not _check_protocol_compliance(plugin_class, protocol, group):
                # Only add this if we don't already have specific errors
                if not errors:
                    errors.append(
                        f"Plugin does not fully implement {protocol.__name__} protocol"
                    )
        except Exception:
            # Protocol checking failed, rely on attribute checks
            pass

    return errors


def _check_protocol_compliance(
    plugin_class: type[Any],
    protocol: type[PluginProtocol],
    group: str,
) -> bool:
    """Check if a plugin class is compliant with a protocol.

    Args:
        plugin_class: The plugin class to check.
        protocol: The protocol to check against.
        group: The plugin group name.

    Returns:
        True if the plugin is compliant, False otherwise.
    """
    # Check all required methods exist and are callable
    required_methods = get_required_methods(group)
    for method_name in required_methods:
        if not hasattr(plugin_class, method_name):
            return False
        method = getattr(plugin_class, method_name)
        if not callable(method):
            return False

    return True


def check_version_compatibility(
    plugin_class: type[Any],
    current_version: str,
    plugin_name: str | None = None,
) -> str | None:
    """Check if a plugin is compatible with the current ATP version.

    Args:
        plugin_class: The plugin class to check.
        current_version: Current ATP version string.
        plugin_name: Optional plugin name for error messages.

    Returns:
        Error message if incompatible, None if compatible.
    """
    name = plugin_name or getattr(plugin_class, "__name__", str(plugin_class))

    # Get the plugin's required ATP version
    required_version = getattr(plugin_class, "atp_version", MIN_ATP_VERSION)

    if not isinstance(required_version, str):
        return f"Plugin '{name}' has invalid atp_version: must be a string"

    # Parse versions for comparison
    try:
        required_parts = _parse_version(required_version)
        current_parts = _parse_version(current_version)
    except ValueError as e:
        return f"Plugin '{name}' version parsing error: {e}"

    # Check if current version meets the requirement
    # We use semver-style comparison: current must be >= required
    if not _version_satisfies(current_parts, required_parts):
        return (
            f"Plugin '{name}' requires ATP version >= {required_version}, "
            f"but current version is {current_version}"
        )

    return None


def _parse_version(version: str) -> tuple[int, int, int]:
    """Parse a semver-style version string.

    Args:
        version: Version string like "1.2.3".

    Returns:
        Tuple of (major, minor, patch).

    Raises:
        ValueError: If version format is invalid.
    """
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version} (expected X.Y.Z)")

    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as e:
        raise ValueError(f"Invalid version number in {version}: {e}") from e


def _version_satisfies(
    current: tuple[int, int, int],
    required: tuple[int, int, int],
) -> bool:
    """Check if current version satisfies the required version.

    Uses semver comparison: current >= required.

    Args:
        current: Current version tuple.
        required: Required version tuple.

    Returns:
        True if current >= required.
    """
    # Compare major.minor.patch
    return current >= required


def validate_plugin_full(
    plugin_class: type[Any],
    group: str,
    current_version: str,
    plugin_name: str | None = None,
) -> tuple[list[str], str | None]:
    """Perform full validation of a plugin including version check.

    Args:
        plugin_class: The plugin class to validate.
        group: Entry point group name.
        current_version: Current ATP version string.
        plugin_name: Optional plugin name for error messages.

    Returns:
        Tuple of (interface_errors, version_error).
    """
    interface_errors = validate_plugin(plugin_class, group, plugin_name)
    version_error = check_version_compatibility(
        plugin_class, current_version, plugin_name
    )
    return interface_errors, version_error


def get_plugin_config_schema(
    plugin_class: type[Any],
) -> type[PluginConfig] | None:
    """Get the configuration schema class from a plugin.

    Args:
        plugin_class: The plugin class to get config schema from.

    Returns:
        The PluginConfig subclass if defined, None otherwise.
    """
    config_schema = getattr(plugin_class, "config_schema", None)
    if config_schema is None:
        return None

    # Validate it's a proper PluginConfig subclass
    from atp.plugins.config import PluginConfig

    if isinstance(config_schema, type) and issubclass(config_schema, PluginConfig):
        return config_schema

    return None


def validate_plugin_config(
    plugin_class: type[Any],
    config_data: dict[str, Any],
    plugin_name: str | None = None,
) -> PluginConfig | None:
    """Validate configuration data against a plugin's config schema.

    Args:
        plugin_class: The plugin class to validate config for.
        config_data: Configuration data dictionary to validate.
        plugin_name: Optional plugin name for error messages.

    Returns:
        Validated PluginConfig instance if schema exists, None otherwise.

    Raises:
        PluginConfigError: If validation fails.
    """
    from atp.plugins.config import validate_plugin_config

    config_schema = get_plugin_config_schema(plugin_class)
    if config_schema is None:
        return None

    name = plugin_name or getattr(plugin_class, "__name__", str(plugin_class))
    return validate_plugin_config(config_schema, config_data, name)


def get_plugin_config_metadata(
    plugin_class: type[Any],
) -> dict[str, Any] | None:
    """Get configuration metadata from a plugin for documentation.

    Args:
        plugin_class: The plugin class to get config metadata from.

    Returns:
        Dictionary with config metadata (schema, examples, defaults, etc.)
        or None if no config schema is defined.
    """
    from atp.plugins.config import ConfigMetadata

    config_schema = get_plugin_config_schema(plugin_class)
    if config_schema is None:
        return None

    metadata = ConfigMetadata.from_config_class(config_schema)
    return metadata.model_dump(by_alias=True)
