"""Tests for MainScreen."""

from __future__ import annotations

from pathlib import Path

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
    from atp.tui import ATPTUI
    from atp.tui.screens.main_menu import MainScreen
    from atp.tui.widgets.test_tree import TestTreeWidget
    from atp.tui.widgets.yaml_preview import YAMLPreviewWidget


class TestMainScreen:
    """Test MainScreen class."""

    def test_instantiation(self) -> None:
        """Test that MainScreen can be instantiated."""
        screen = MainScreen()
        assert screen is not None
        assert screen.suite is None
        assert screen.suite_path is None

    def test_instantiation_with_suite(self) -> None:
        """Test instantiation with a test suite."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        screen = MainScreen(suite=suite)
        assert screen.suite is not None
        assert screen.suite.name == "test_suite"

    def test_instantiation_with_path(self) -> None:
        """Test instantiation with a path."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        path = Path("/tmp/test_suite.yaml")
        screen = MainScreen(suite=suite, suite_path=path)
        assert screen.suite_path == path

    def test_screen_name(self) -> None:
        """Test that screen has correct name."""
        assert MainScreen.SCREEN_NAME == "main"

    def test_bindings_defined(self) -> None:
        """Test that bindings are defined."""
        screen = MainScreen()
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "n" in binding_keys  # new
        assert "o" in binding_keys  # open
        assert "s" in binding_keys  # save
        assert "a" in binding_keys  # add test
        assert "h" in binding_keys  # home
        assert "question_mark" in binding_keys  # help
        assert "escape" in binding_keys  # back

    def test_is_modified_default_false(self) -> None:
        """Test that is_modified is false by default."""
        screen = MainScreen()
        assert screen.is_modified is False

    def test_request_messages(self) -> None:
        """Test that request messages are created correctly."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        new_msg = MainScreen.RequestNewSuite()
        assert new_msg is not None

        open_msg = MainScreen.RequestOpenSuite()
        assert open_msg is not None

        save_msg = MainScreen.RequestSaveSuite()
        assert save_msg is not None

        add_msg = MainScreen.RequestAddTest()
        assert add_msg is not None

        changed_msg = MainScreen.SuiteChanged(suite)
        assert changed_msg.suite is suite


@pytest.mark.anyio
class TestMainScreenIntegration:
    """Integration tests for MainScreen using the full TUI app."""

    async def test_navigate_to_main_screen(self) -> None:
        """Test navigation to main screen."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("m")
            assert isinstance(pilot.app.screen, MainScreen)

    async def test_main_screen_has_tree_widget(self) -> None:
        """Test that main screen has tree widget."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("m")
            screen = pilot.app.screen
            tree = screen.query_one("#test-tree", TestTreeWidget)
            assert tree is not None

    async def test_main_screen_has_preview_widget(self) -> None:
        """Test that main screen has preview widget."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("m")
            screen = pilot.app.screen
            preview = screen.query_one("#yaml-preview", YAMLPreviewWidget)
            assert preview is not None

    async def test_main_screen_navigate_back(self) -> None:
        """Test navigation back to home from main screen."""
        from atp.tui.app import HomeScreen

        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("m")
            assert isinstance(pilot.app.screen, MainScreen)
            await pilot.press("escape")
            assert isinstance(pilot.app.screen, HomeScreen)

    async def test_main_screen_navigate_to_help(self) -> None:
        """Test navigation to help from main screen."""
        from atp.tui.app import HelpScreen

        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("m")
            assert isinstance(pilot.app.screen, MainScreen)
            await pilot.press("?")
            assert isinstance(pilot.app.screen, HelpScreen)

    async def test_main_screen_set_suite(self) -> None:
        """Test setting a suite on the main screen."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite", "My Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("m")
            screen = pilot.app.screen
            assert isinstance(screen, MainScreen)

            screen.set_suite(suite)
            assert screen.suite is not None
            assert screen.suite.name == "my_suite"

            # Verify widgets are updated
            tree = screen.query_one("#test-tree", TestTreeWidget)
            assert tree.suite is not None

            preview = screen.query_one("#yaml-preview", YAMLPreviewWidget)
            assert preview.suite is not None

    async def test_main_screen_mark_modified(self) -> None:
        """Test marking the screen as modified."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite", "My Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("m")
            screen = pilot.app.screen
            assert isinstance(screen, MainScreen)

            screen.set_suite(suite)
            assert screen.is_modified is False

            screen.mark_modified()
            assert screen.is_modified is True

    async def test_main_screen_mark_saved(self) -> None:
        """Test marking the screen as saved."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite", "My Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("m")
            screen = pilot.app.screen
            assert isinstance(screen, MainScreen)

            screen.set_suite(suite)
            screen.mark_modified()
            assert screen.is_modified is True

            screen.mark_saved()
            assert screen.is_modified is False
