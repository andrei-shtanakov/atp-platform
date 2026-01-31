"""Tests for ATP TUI application."""

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
    from atp import __version__
    from atp.tui import ATPTUI
    from atp.tui.app import (
        AgentsPanel,
        AgentsScreen,
        HelpPanel,
        HelpScreen,
        HomeScreen,
        ResultsPanel,
        ResultsScreen,
        SuitesPanel,
        SuitesScreen,
        WelcomePanel,
        run_tui,
    )


class TestATPTUI:
    """Test ATPTUI application class."""

    def test_atptui_instantiation(self) -> None:
        """Test that ATPTUI can be instantiated."""
        app = ATPTUI()
        assert app.TITLE == "ATP - Agent Test Platform"
        assert app.SUB_TITLE == f"v{__version__}"

    def test_atptui_screens_defined(self) -> None:
        """Test that all screens are defined."""
        app = ATPTUI()
        expected_screens = {"home", "suites", "results", "agents", "help"}
        assert set(app.SCREENS.keys()) == expected_screens

    def test_atptui_bindings_defined(self) -> None:
        """Test that quit bindings are defined."""
        app = ATPTUI()
        binding_keys = [b.key for b in app.BINDINGS]
        assert "q" in binding_keys
        assert "ctrl+c" in binding_keys

    def test_atptui_css_not_empty(self) -> None:
        """Test that CSS is defined."""
        app = ATPTUI()
        assert app.CSS is not None
        assert len(app.CSS) > 0
        assert "Screen" in app.CSS
        assert "Header" in app.CSS
        assert "Footer" in app.CSS


class TestScreens:
    """Test screen classes."""

    def test_home_screen_bindings(self) -> None:
        """Test HomeScreen has expected bindings."""
        screen = HomeScreen()
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "s" in binding_keys
        assert "r" in binding_keys
        assert "a" in binding_keys
        assert "question_mark" in binding_keys

    def test_suites_screen_bindings(self) -> None:
        """Test SuitesScreen has expected bindings."""
        screen = SuitesScreen()
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "h" in binding_keys
        assert "r" in binding_keys
        assert "a" in binding_keys
        assert "escape" in binding_keys

    def test_results_screen_bindings(self) -> None:
        """Test ResultsScreen has expected bindings."""
        screen = ResultsScreen()
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "h" in binding_keys
        assert "s" in binding_keys
        assert "a" in binding_keys
        assert "escape" in binding_keys

    def test_agents_screen_bindings(self) -> None:
        """Test AgentsScreen has expected bindings."""
        screen = AgentsScreen()
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "h" in binding_keys
        assert "s" in binding_keys
        assert "r" in binding_keys
        assert "escape" in binding_keys

    def test_help_screen_bindings(self) -> None:
        """Test HelpScreen has expected bindings."""
        screen = HelpScreen()
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "h" in binding_keys
        assert "s" in binding_keys
        assert "r" in binding_keys
        assert "a" in binding_keys
        assert "escape" in binding_keys


class TestPanels:
    """Test panel classes."""

    def test_welcome_panel_instantiation(self) -> None:
        """Test WelcomePanel can be instantiated."""
        panel = WelcomePanel()
        assert panel is not None

    def test_suites_panel_instantiation(self) -> None:
        """Test SuitesPanel can be instantiated."""
        panel = SuitesPanel()
        assert panel is not None

    def test_results_panel_instantiation(self) -> None:
        """Test ResultsPanel can be instantiated."""
        panel = ResultsPanel()
        assert panel is not None

    def test_agents_panel_instantiation(self) -> None:
        """Test AgentsPanel can be instantiated."""
        panel = AgentsPanel()
        assert panel is not None

    def test_help_panel_instantiation(self) -> None:
        """Test HelpPanel can be instantiated."""
        panel = HelpPanel()
        assert panel is not None


class TestRunTui:
    """Test run_tui function."""

    def test_run_tui_function_exists(self) -> None:
        """Test that run_tui function exists and is callable."""
        assert callable(run_tui)


@pytest.mark.anyio
class TestATTPUIIntegration:
    """Integration tests for ATPTUI using Textual's test framework."""

    async def test_app_starts_on_home_screen(self) -> None:
        """Test that the app starts on the home screen."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            # The app should start on the home screen
            assert isinstance(pilot.app.screen, HomeScreen)

    async def test_navigate_to_suites(self) -> None:
        """Test navigation to suites screen."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("s")
            assert isinstance(pilot.app.screen, SuitesScreen)

    async def test_navigate_to_results(self) -> None:
        """Test navigation to results screen."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("r")
            assert isinstance(pilot.app.screen, ResultsScreen)

    async def test_navigate_to_agents(self) -> None:
        """Test navigation to agents screen."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("a")
            assert isinstance(pilot.app.screen, AgentsScreen)

    async def test_navigate_to_help(self) -> None:
        """Test navigation to help screen."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("?")
            assert isinstance(pilot.app.screen, HelpScreen)

    async def test_navigate_back_with_escape(self) -> None:
        """Test navigation back to home with escape."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            await pilot.press("s")
            assert isinstance(pilot.app.screen, SuitesScreen)
            await pilot.press("escape")
            assert isinstance(pilot.app.screen, HomeScreen)

    async def test_navigate_between_screens(self) -> None:
        """Test navigation between different screens."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            # Navigate through all screens
            await pilot.press("s")
            assert isinstance(pilot.app.screen, SuitesScreen)

            await pilot.press("r")
            assert isinstance(pilot.app.screen, ResultsScreen)

            await pilot.press("a")
            assert isinstance(pilot.app.screen, AgentsScreen)

            await pilot.press("h")
            assert isinstance(pilot.app.screen, HomeScreen)

    async def test_quit_app(self) -> None:
        """Test that app can be quit with 'q'."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            # The app should exit when 'q' is pressed
            await pilot.press("q")
            # If we get here without exception, the quit action was triggered
            # The app exits cleanly

    async def test_home_screen_active(self) -> None:
        """Test that home screen is the active screen on start."""
        app = ATPTUI()
        async with app.run_test() as pilot:
            # Verify the home screen is active
            assert isinstance(pilot.app.screen, HomeScreen)
            # Verify the screen has the expected bindings
            screen = pilot.app.screen
            binding_keys = [b.key for b in screen.BINDINGS]
            assert "s" in binding_keys
            assert "r" in binding_keys
            assert "a" in binding_keys
