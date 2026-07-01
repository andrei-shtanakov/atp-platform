"""Tests for the reporter registry."""

from pathlib import Path

import pytest

from atp.reporters import (
    ConsoleReporter,
    HTMLReporter,
    JSONReporter,
    Reporter,
    ReporterNotFoundError,
    ReporterRegistry,
    SummaryReporter,
    create_reporter,
    get_registry,
)
from atp.reporters.base import SuiteReport


class CustomReporter(Reporter):
    """Custom reporter for testing registration."""

    @property
    def name(self) -> str:
        return "custom"

    def report(self, report: SuiteReport) -> None:
        pass


class TestReporterRegistry:
    """Tests for ReporterRegistry."""

    @pytest.fixture
    def registry(self) -> ReporterRegistry:
        """Create a fresh registry."""
        return ReporterRegistry()

    def test_builtin_reporters_registered(self, registry: ReporterRegistry) -> None:
        """Verify built-in reporters are registered."""
        assert registry.is_registered("console")
        assert registry.is_registered("html")
        assert registry.is_registered("json")
        assert registry.is_registered("summary")

    def test_list_reporters(self, registry: ReporterRegistry) -> None:
        """Verify list_reporters returns all registered reporters."""
        reporters = registry.list_reporters()
        assert "console" in reporters
        assert "html" in reporters
        assert "json" in reporters
        assert "summary" in reporters

    def test_get_reporter_class(self, registry: ReporterRegistry) -> None:
        """Verify get_reporter_class returns correct class."""
        assert registry.get_reporter_class("console") == ConsoleReporter
        assert registry.get_reporter_class("html") == HTMLReporter
        assert registry.get_reporter_class("json") == JSONReporter
        assert registry.get_reporter_class("summary") == SummaryReporter

    def test_get_reporter_class_not_found(self, registry: ReporterRegistry) -> None:
        """Verify get_reporter_class raises for unknown reporter."""
        with pytest.raises(ReporterNotFoundError) as exc_info:
            registry.get_reporter_class("unknown")

        assert exc_info.value.reporter_type == "unknown"

    def test_create_console_reporter(self, registry: ReporterRegistry) -> None:
        """Verify create returns ConsoleReporter."""
        reporter = registry.create("console")
        assert isinstance(reporter, ConsoleReporter)
        assert reporter.name == "console"

    def test_create_json_reporter(self, registry: ReporterRegistry) -> None:
        """Verify create returns JSONReporter."""
        reporter = registry.create("json")
        assert isinstance(reporter, JSONReporter)
        assert reporter.name == "json"

    def test_create_summary_reporter(self, registry: ReporterRegistry) -> None:
        """Verify create returns SummaryReporter."""
        reporter = registry.create("summary")
        assert isinstance(reporter, SummaryReporter)
        assert reporter.name == "summary"

    def test_create_html_reporter(self, registry: ReporterRegistry) -> None:
        """Verify create returns HTMLReporter."""
        reporter = registry.create("html")
        assert isinstance(reporter, HTMLReporter)
        assert reporter.name == "html"

    def test_create_console_with_config(self, registry: ReporterRegistry) -> None:
        """Verify create passes config to ConsoleReporter."""
        reporter = registry.create(
            "console",
            config={"verbose": True, "use_colors": False},
        )

        assert isinstance(reporter, ConsoleReporter)
        assert reporter._verbose is True
        assert reporter._use_colors is False

    def test_create_json_with_config(
        self, registry: ReporterRegistry, tmp_path: Path
    ) -> None:
        """Verify create passes config to JSONReporter."""
        output_file = tmp_path / "results.json"
        reporter = registry.create(
            "json",
            config={"output_file": str(output_file), "indent": 4},
        )

        assert isinstance(reporter, JSONReporter)
        assert reporter._output_file == output_file
        assert reporter._indent == 4

    def test_create_summary_with_config(
        self, registry: ReporterRegistry, tmp_path: Path
    ) -> None:
        """Verify create passes config to SummaryReporter."""
        output_file = tmp_path / "summary.json"
        reporter = registry.create(
            "summary",
            config={
                "output_file": str(output_file),
                "format": "json",
                "indent": None,
                "include_passed": True,
                "max_failures": 3,
                "use_colors": False,
            },
        )

        assert isinstance(reporter, SummaryReporter)
        assert reporter._output_file == output_file
        assert reporter._format == "json"
        assert reporter._indent is None
        assert reporter._include_passed is True
        assert reporter._max_failures == 3
        assert reporter._use_colors is False

    def test_create_html_with_config(
        self, registry: ReporterRegistry, tmp_path: Path
    ) -> None:
        """Verify create passes config to HTMLReporter."""
        output_file = tmp_path / "report.html"
        reporter = registry.create(
            "html",
            config={
                "output_file": str(output_file),
                "title": "Custom Title",
                "include_trace": False,
                "auto_expand_failed": False,
            },
        )

        assert isinstance(reporter, HTMLReporter)
        assert reporter._output_file == output_file
        assert reporter._title == "Custom Title"
        assert reporter._include_trace is False
        assert reporter._auto_expand_failed is False

    def test_create_not_found(self, registry: ReporterRegistry) -> None:
        """Verify create raises for unknown reporter."""
        with pytest.raises(ReporterNotFoundError):
            registry.create("unknown")

    def test_register_custom_reporter(self, registry: ReporterRegistry) -> None:
        """Verify custom reporter registration."""
        registry.register("custom", CustomReporter)

        assert registry.is_registered("custom")
        assert registry.get_reporter_class("custom") == CustomReporter

    def test_unregister_reporter(self, registry: ReporterRegistry) -> None:
        """Verify reporter unregistration."""
        registry.register("custom", CustomReporter)
        assert registry.is_registered("custom")

        result = registry.unregister("custom")
        assert result is True
        assert not registry.is_registered("custom")

    def test_unregister_nonexistent(self, registry: ReporterRegistry) -> None:
        """Verify unregister returns False for nonexistent reporter."""
        result = registry.unregister("nonexistent")
        assert result is False


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry_returns_same_instance(self) -> None:
        """Verify get_registry returns singleton."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_create_reporter_function(self) -> None:
        """Verify create_reporter function works."""
        reporter = create_reporter("console")
        assert isinstance(reporter, ConsoleReporter)

    def test_create_reporter_with_config(self) -> None:
        """Verify create_reporter passes config."""
        reporter = create_reporter("console", config={"verbose": True})
        assert isinstance(reporter, ConsoleReporter)
        assert reporter._verbose is True
