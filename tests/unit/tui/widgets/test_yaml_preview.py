"""Tests for YAMLPreviewWidget."""

from __future__ import annotations

import pytest

# Check if TUI dependencies are available
try:
    from textual.pilot import Pilot  # noqa: F401

    HAS_TUI_DEPS = True
except ImportError:
    HAS_TUI_DEPS = False

# Skip all tests if TUI dependencies are not installed
pytestmark = pytest.mark.skipif(
    not HAS_TUI_DEPS,
    reason="TUI dependencies not installed (install with: uv add atp-platform[tui])",
)

if HAS_TUI_DEPS:
    from atp.generator.core import TestGenerator
    from atp.tui.widgets.yaml_preview import YAMLPreviewWidget


class TestYAMLPreviewWidget:
    """Test YAMLPreviewWidget class."""

    def test_instantiation(self) -> None:
        """Test that YAMLPreviewWidget can be instantiated."""
        widget = YAMLPreviewWidget()
        assert widget is not None
        assert widget.suite is None

    def test_instantiation_with_suite(self) -> None:
        """Test instantiation with a test suite."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        widget = YAMLPreviewWidget(suite)
        assert widget.suite is not None
        assert widget.suite.name == "test_suite"

    def test_set_suite(self) -> None:
        """Test setting a suite after instantiation."""
        widget = YAMLPreviewWidget()
        assert widget.suite is None

        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        widget.set_suite(suite)
        assert widget.suite is not None
        assert widget.suite.name == "test_suite"

    def test_set_suite_to_none(self) -> None:
        """Test clearing the suite."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        widget = YAMLPreviewWidget(suite)
        assert widget.suite is not None

        widget.set_suite(None)
        assert widget.suite is None

    def test_show_test(self) -> None:
        """Test showing a specific test."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        widget = YAMLPreviewWidget(suite)
        widget.show_test("test-001")
        assert widget.selected_test_id == "test-001"

    def test_show_full_suite(self) -> None:
        """Test showing full suite after showing a test."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        widget = YAMLPreviewWidget(suite)
        widget.show_test("test-001")
        assert widget.selected_test_id == "test-001"

        widget.show_full_suite()
        assert widget.selected_test_id is None

    def test_content_changed_message(self) -> None:
        """Test that ContentChanged message is created correctly."""
        message = YAMLPreviewWidget.ContentChanged()
        assert message is not None


@pytest.mark.anyio
class TestYAMLPreviewWidgetIntegration:
    """Integration tests for YAMLPreviewWidget."""

    async def test_preview_displays_suite(self) -> None:
        """Test that the preview displays suite content."""
        from textual.app import App, ComposeResult

        generator = TestGenerator()
        suite = generator.create_suite("my_suite", "My Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield YAMLPreviewWidget(suite, id="preview")

        app = TestApp()
        async with app.run_test() as pilot:
            preview = pilot.app.query_one("#preview", YAMLPreviewWidget)
            assert preview.suite is not None
            assert preview.suite.name == "my_suite"

    async def test_preview_empty_state(self) -> None:
        """Test that the preview shows empty state when no suite."""
        from textual.app import App, ComposeResult

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield YAMLPreviewWidget(id="preview")

        app = TestApp()
        async with app.run_test() as pilot:
            preview = pilot.app.query_one("#preview", YAMLPreviewWidget)
            assert preview.suite is None

    async def test_preview_shows_test_details(self) -> None:
        """Test that the preview shows test details when test selected."""
        from textual.app import App, ComposeResult

        generator = TestGenerator()
        suite = generator.create_suite("my_suite", "My Test Suite")
        test = generator.create_custom_test(
            "test-001",
            "Test One",
            "Do something important",
            tags=["smoke", "fast"],
        )
        suite = generator.add_test(suite, test)

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield YAMLPreviewWidget(suite, id="preview")

        app = TestApp()
        async with app.run_test() as pilot:
            preview = pilot.app.query_one("#preview", YAMLPreviewWidget)
            preview.show_test("test-001")
            assert preview.selected_test_id == "test-001"
