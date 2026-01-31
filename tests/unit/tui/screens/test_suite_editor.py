"""Tests for NewSuiteScreen."""

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
    from textual.widgets import Button, Input, TextArea

    from atp.tui import ATPTUI
    from atp.tui.screens.main_menu import MainScreen
    from atp.tui.screens.suite_editor import NewSuiteScreen


class TestNewSuiteScreen:
    """Test NewSuiteScreen class."""

    def test_instantiation(self) -> None:
        """Test that NewSuiteScreen can be instantiated."""
        screen = NewSuiteScreen()
        assert screen is not None

    def test_screen_name(self) -> None:
        """Test that screen has correct name."""
        assert NewSuiteScreen.SCREEN_NAME == "new_suite"

    def test_bindings_defined(self) -> None:
        """Test that bindings are defined."""
        screen = NewSuiteScreen()
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "escape" in binding_keys  # cancel
        assert "ctrl+s" in binding_keys  # submit

    def test_suite_created_message(self) -> None:
        """Test that SuiteCreated message is created correctly."""
        from atp.generator.core import TestSuiteData

        suite = TestSuiteData(name="test_suite")
        msg = NewSuiteScreen.SuiteCreated(suite)
        assert msg.suite is suite
        assert msg.suite.name == "test_suite"

    def test_cancelled_message(self) -> None:
        """Test that Cancelled message is created correctly."""
        msg = NewSuiteScreen.Cancelled()
        assert msg is not None


@pytest.mark.anyio
class TestNewSuiteScreenIntegration:
    """Integration tests for NewSuiteScreen using the full TUI app."""

    async def test_navigate_to_new_suite_screen(self) -> None:
        """Test navigation to new suite screen from main screen."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            # Go to main screen
            await pilot.press("m")
            assert isinstance(pilot.app.screen, MainScreen)

            # Press 'n' for new suite
            await pilot.press("n")

            # Should now be on NewSuiteScreen
            assert isinstance(pilot.app.screen, NewSuiteScreen)

    async def test_new_suite_screen_has_form_fields(self) -> None:
        """Test that new suite screen has all required form fields."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("m")
            await pilot.press("n")

            screen = pilot.app.screen
            assert isinstance(screen, NewSuiteScreen)

            # Check for required inputs
            name_input = screen.query_one("#name-input", Input)
            assert name_input is not None

            description_input = screen.query_one("#description-input", TextArea)
            assert description_input is not None

            version_input = screen.query_one("#version-input", Input)
            assert version_input is not None
            assert version_input.value == "1.0"

            runs_input = screen.query_one("#runs-input", Input)
            assert runs_input is not None
            assert runs_input.value == "1"

            timeout_input = screen.query_one("#timeout-input", Input)
            assert timeout_input is not None
            assert timeout_input.value == "300"

    async def test_new_suite_screen_has_buttons(self) -> None:
        """Test that new suite screen has Create and Cancel buttons."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("m")
            await pilot.press("n")

            screen = pilot.app.screen
            assert isinstance(screen, NewSuiteScreen)

            create_btn = screen.query_one("#create-btn", Button)
            assert create_btn is not None

            cancel_btn = screen.query_one("#cancel-btn", Button)
            assert cancel_btn is not None

    async def test_cancel_button_dismisses_screen(self) -> None:
        """Test that Cancel button dismisses the screen."""
        app = ATPTUI()
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.press("m")
            await pilot.press("n")
            assert isinstance(pilot.app.screen, NewSuiteScreen)

            # Click cancel
            await pilot.click("#cancel-btn")

            # Should be back on main screen
            assert isinstance(pilot.app.screen, MainScreen)

    async def test_escape_key_dismisses_screen(self) -> None:
        """Test that Escape key dismisses the screen."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("m")
            await pilot.press("n")
            assert isinstance(pilot.app.screen, NewSuiteScreen)

            # Press escape
            await pilot.press("escape")

            # Should be back on main screen
            assert isinstance(pilot.app.screen, MainScreen)

    async def test_create_suite_with_valid_data(self) -> None:
        """Test creating a suite with valid data."""
        app = ATPTUI()
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.press("m")
            await pilot.press("n")

            screen = pilot.app.screen
            assert isinstance(screen, NewSuiteScreen)

            # Fill in the form
            name_input = screen.query_one("#name-input", Input)
            name_input.value = "my-test-suite"

            description_area = screen.query_one("#description-input", TextArea)
            description_area.text = "A test suite for testing"

            # Click create
            await pilot.click("#create-btn")

            # Should be on main screen with suite loaded
            assert isinstance(pilot.app.screen, MainScreen)
            main_screen = pilot.app.screen
            assert main_screen.suite is not None
            assert main_screen.suite.name == "my-test-suite"
            assert main_screen.suite.description == "A test suite for testing"
            assert main_screen.is_modified is True

    async def test_create_suite_validation_fails_without_name(self) -> None:
        """Test that validation fails when name is empty."""
        app = ATPTUI()
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.press("m")
            await pilot.press("n")

            screen = pilot.app.screen
            assert isinstance(screen, NewSuiteScreen)

            # Try to create without filling name
            await pilot.click("#create-btn")

            # Should still be on NewSuiteScreen (validation failed)
            assert isinstance(pilot.app.screen, NewSuiteScreen)

    async def test_ctrl_s_submits_form(self) -> None:
        """Test that Ctrl+S submits the form."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("m")
            await pilot.press("n")

            screen = pilot.app.screen
            assert isinstance(screen, NewSuiteScreen)

            # Fill in required field
            name_input = screen.query_one("#name-input", Input)
            name_input.value = "ctrl-s-test-suite"

            # Press Ctrl+S
            await pilot.press("ctrl+s")

            # Should be on main screen with suite loaded
            assert isinstance(pilot.app.screen, MainScreen)
            assert pilot.app.screen.suite is not None
            assert pilot.app.screen.suite.name == "ctrl-s-test-suite"
