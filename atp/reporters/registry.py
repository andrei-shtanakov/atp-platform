"""Reporter registry for managing and creating reporters."""

from pathlib import Path
from typing import Any

from .base import Reporter
from .console import ConsoleReporter
from .html_reporter import HTMLReporter
from .json_reporter import JSONReporter
from .junit_reporter import JUnitReporter


class ReporterNotFoundError(Exception):
    """Raised when a reporter type is not registered."""

    def __init__(self, reporter_type: str) -> None:
        self.reporter_type = reporter_type
        super().__init__(f"Reporter type not found: {reporter_type}")


class ReporterRegistry:
    """Registry for reporter types.

    Provides factory methods for creating reporters from configuration.
    """

    def __init__(self) -> None:
        """Initialize the registry with built-in reporters."""
        self._reporters: dict[str, type[Reporter]] = {}

        self.register("console", ConsoleReporter)
        self.register("html", HTMLReporter)
        self.register("json", JSONReporter)
        self.register("junit", JUnitReporter)

    def register(
        self,
        reporter_type: str,
        reporter_class: type[Reporter],
    ) -> None:
        """Register a reporter type.

        Args:
            reporter_type: Unique identifier for the reporter type.
            reporter_class: Reporter class to instantiate.
        """
        self._reporters[reporter_type] = reporter_class

    def unregister(self, reporter_type: str) -> bool:
        """Unregister a reporter type.

        Args:
            reporter_type: Identifier of the reporter to remove.

        Returns:
            True if reporter was removed, False if it didn't exist.
        """
        if reporter_type in self._reporters:
            del self._reporters[reporter_type]
            return True
        return False

    def get_reporter_class(self, reporter_type: str) -> type[Reporter]:
        """Get the reporter class for a type.

        Args:
            reporter_type: Reporter type identifier.

        Returns:
            Reporter class.

        Raises:
            ReporterNotFoundError: If reporter type is not registered.
        """
        if reporter_type not in self._reporters:
            raise ReporterNotFoundError(reporter_type)
        return self._reporters[reporter_type]

    def create(
        self, reporter_type: str, config: dict[str, Any] | None = None
    ) -> Reporter:
        """Create a reporter instance.

        Args:
            reporter_type: Reporter type identifier.
            config: Optional configuration for the reporter.

        Returns:
            Configured reporter instance.

        Raises:
            ReporterNotFoundError: If reporter type is not registered.
        """
        reporter_class = self.get_reporter_class(reporter_type)
        config = config or {}

        if reporter_type == "console":
            return ConsoleReporter(
                use_colors=config.get("use_colors", True),
                verbose=config.get("verbose", False),
            )
        elif reporter_type == "html":
            output_file = config.get("output_file")
            if output_file:
                output_file = Path(output_file)
            return HTMLReporter(
                output_file=output_file,
                title=config.get("title", "ATP Test Results"),
                include_trace=config.get("include_trace", True),
                auto_expand_failed=config.get("auto_expand_failed", True),
            )
        elif reporter_type == "json":
            output_file = config.get("output_file")
            if output_file:
                output_file = Path(output_file)
            return JSONReporter(
                output_file=output_file,
                indent=config.get("indent", 2),
                include_details=config.get("include_details", True),
            )
        elif reporter_type == "junit":
            output_file = config.get("output_file")
            if output_file:
                output_file = Path(output_file)
            return JUnitReporter(
                output_file=output_file,
                include_properties=config.get("include_properties", True),
                include_system_out=config.get("include_system_out", True),
            )
        else:
            return reporter_class()

    def list_reporters(self) -> list[str]:
        """List all registered reporter types.

        Returns:
            List of reporter type identifiers.
        """
        return list(self._reporters.keys())

    def is_registered(self, reporter_type: str) -> bool:
        """Check if a reporter type is registered.

        Args:
            reporter_type: Reporter type identifier.

        Returns:
            True if reporter is registered, False otherwise.
        """
        return reporter_type in self._reporters


_default_registry: ReporterRegistry | None = None


def get_registry() -> ReporterRegistry:
    """Get the global reporter registry.

    Returns:
        Global ReporterRegistry instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ReporterRegistry()
    return _default_registry


def create_reporter(
    reporter_type: str, config: dict[str, Any] | None = None
) -> Reporter:
    """Create a reporter using the global registry.

    Args:
        reporter_type: Reporter type identifier.
        config: Optional configuration for the reporter.

    Returns:
        Configured reporter instance.
    """
    return get_registry().create(reporter_type, config)
