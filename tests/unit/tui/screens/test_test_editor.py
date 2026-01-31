"""Tests for AddTestScreen."""

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

    from atp.generator.core import TestGenerator
    from atp.tui import ATPTUI
    from atp.tui.screens.main_menu import MainScreen
    from atp.tui.screens.test_editor import AddTestScreen


class TestAddTestScreen:
    """Test AddTestScreen class."""

    def test_instantiation(self) -> None:
        """Test that AddTestScreen can be instantiated."""
        screen = AddTestScreen()
        assert screen is not None

    def test_instantiation_with_suite(self) -> None:
        """Test instantiation with a test suite."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")

        screen = AddTestScreen(suite=suite)
        assert screen._suite is suite

    def test_screen_name(self) -> None:
        """Test that screen has correct name."""
        assert AddTestScreen.SCREEN_NAME == "add_test"

    def test_bindings_defined(self) -> None:
        """Test that bindings are defined."""
        screen = AddTestScreen()
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "escape" in binding_keys  # cancel
        assert "ctrl+s" in binding_keys  # submit

    def test_test_created_message(self) -> None:
        """Test that TestCreated message is created correctly."""
        from atp.loader.models import Constraints, TaskDefinition, TestDefinition

        test = TestDefinition(
            id="test-001",
            name="Test One",
            task=TaskDefinition(description="Do something"),
            constraints=Constraints(),
        )
        msg = AddTestScreen.TestCreated(test)
        assert msg.test is test
        assert msg.test.id == "test-001"

    def test_cancelled_message(self) -> None:
        """Test that Cancelled message is created correctly."""
        msg = AddTestScreen.Cancelled()
        assert msg is not None

    def test_generate_test_id_empty_suite(self) -> None:
        """Test ID generation with empty suite."""
        screen = AddTestScreen()
        test_id = screen._generate_test_id()
        assert test_id == "test-001"

    def test_generate_test_id_with_existing_tests(self) -> None:
        """Test ID generation with existing tests."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        screen = AddTestScreen(suite=suite)
        test_id = screen._generate_test_id()
        assert test_id == "test-002"

    def test_parse_comma_list(self) -> None:
        """Test comma-separated list parsing."""
        screen = AddTestScreen()

        # Empty string
        assert screen._parse_comma_list("") == []
        assert screen._parse_comma_list("   ") == []

        # Single item
        assert screen._parse_comma_list("item") == ["item"]

        # Multiple items
        assert screen._parse_comma_list("a, b, c") == ["a", "b", "c"]

        # Items with extra whitespace
        assert screen._parse_comma_list("  a  ,  b  ,  c  ") == ["a", "b", "c"]

        # Empty items should be filtered
        assert screen._parse_comma_list("a,,b") == ["a", "b"]

    def test_parse_optional_int(self) -> None:
        """Test optional integer parsing."""
        screen = AddTestScreen()

        assert screen._parse_optional_int("") is None
        assert screen._parse_optional_int("   ") is None
        assert screen._parse_optional_int("10") == 10
        assert screen._parse_optional_int("  100  ") == 100

    def test_parse_optional_float(self) -> None:
        """Test optional float parsing."""
        screen = AddTestScreen()

        assert screen._parse_optional_float("") is None
        assert screen._parse_optional_float("   ") is None
        assert screen._parse_optional_float("10.5") == 10.5
        assert screen._parse_optional_float("  100.25  ") == 100.25


@pytest.mark.anyio
class TestAddTestScreenIntegration:
    """Integration tests for AddTestScreen using the full TUI app."""

    async def test_add_test_requires_suite(self) -> None:
        """Test that add test action requires a loaded suite."""
        app = ATPTUI()
        async with app.run_test(size=(140, 80)) as pilot:
            await pilot.press("m")
            assert isinstance(pilot.app.screen, MainScreen)

            # Try to add test without suite (should do nothing)
            await pilot.press("a")

            # Should still be on main screen, not add test screen
            assert isinstance(pilot.app.screen, MainScreen)

    async def test_navigate_to_add_test_screen(self) -> None:
        """Test navigation to add test screen when suite is loaded."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")

        app = ATPTUI()
        async with app.run_test(size=(140, 80)) as pilot:
            await pilot.press("m")
            main_screen = pilot.app.screen
            assert isinstance(main_screen, MainScreen)

            # Load a suite
            main_screen.set_suite(suite)

            # Now press 'a' for add test
            await pilot.press("a")

            # Should be on AddTestScreen
            assert isinstance(pilot.app.screen, AddTestScreen)

    async def test_add_test_screen_has_form_fields(self) -> None:
        """Test that add test screen has all required form fields."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")

        app = ATPTUI()
        async with app.run_test(size=(140, 80)) as pilot:
            await pilot.press("m")
            main_screen = pilot.app.screen
            assert isinstance(main_screen, MainScreen)
            main_screen.set_suite(suite)

            await pilot.press("a")
            screen = pilot.app.screen
            assert isinstance(screen, AddTestScreen)

            # Check for required inputs
            id_input = screen.query_one("#id-input", Input)
            assert id_input is not None

            name_input = screen.query_one("#name-input", Input)
            assert name_input is not None

            description_input = screen.query_one("#description-input", TextArea)
            assert description_input is not None

            tags_input = screen.query_one("#tags-input", Input)
            assert tags_input is not None

            task_input = screen.query_one("#task-input", TextArea)
            assert task_input is not None

            artifacts_input = screen.query_one("#artifacts-input", Input)
            assert artifacts_input is not None

            max_steps_input = screen.query_one("#max-steps-input", Input)
            assert max_steps_input is not None

            max_tokens_input = screen.query_one("#max-tokens-input", Input)
            assert max_tokens_input is not None

            timeout_input = screen.query_one("#timeout-input", Input)
            assert timeout_input is not None
            assert timeout_input.value == "300"

            budget_input = screen.query_one("#budget-input", Input)
            assert budget_input is not None

    async def test_add_test_screen_has_buttons(self) -> None:
        """Test that add test screen has Add and Cancel buttons."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")

        app = ATPTUI()
        async with app.run_test(size=(140, 80)) as pilot:
            await pilot.press("m")
            main_screen = pilot.app.screen
            assert isinstance(main_screen, MainScreen)
            main_screen.set_suite(suite)

            await pilot.press("a")
            screen = pilot.app.screen
            assert isinstance(screen, AddTestScreen)

            add_btn = screen.query_one("#add-btn", Button)
            assert add_btn is not None

            cancel_btn = screen.query_one("#cancel-btn", Button)
            assert cancel_btn is not None

    async def test_cancel_button_dismisses_screen(self) -> None:
        """Test that Cancel button dismisses the screen."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")

        app = ATPTUI()
        async with app.run_test(size=(140, 80)) as pilot:
            await pilot.press("m")
            main_screen = pilot.app.screen
            assert isinstance(main_screen, MainScreen)
            main_screen.set_suite(suite)

            await pilot.press("a")
            screen = pilot.app.screen
            assert isinstance(screen, AddTestScreen)

            # Use button.press() as pilot.click() doesn't work for buttons
            cancel_btn = screen.query_one("#cancel-btn", Button)
            cancel_btn.press()
            await pilot.pause()

            assert isinstance(pilot.app.screen, MainScreen)

    async def test_escape_key_dismisses_screen(self) -> None:
        """Test that Escape key dismisses the screen."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")

        app = ATPTUI()
        async with app.run_test(size=(140, 80)) as pilot:
            await pilot.press("m")
            main_screen = pilot.app.screen
            assert isinstance(main_screen, MainScreen)
            main_screen.set_suite(suite)

            await pilot.press("a")
            assert isinstance(pilot.app.screen, AddTestScreen)

            await pilot.press("escape")
            assert isinstance(pilot.app.screen, MainScreen)

    async def test_add_test_with_valid_data(self) -> None:
        """Test adding a test with valid data."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")

        app = ATPTUI()
        async with app.run_test(size=(140, 80)) as pilot:
            await pilot.press("m")
            main_screen = pilot.app.screen
            assert isinstance(main_screen, MainScreen)
            main_screen.set_suite(suite)

            await pilot.press("a")
            screen = pilot.app.screen
            assert isinstance(screen, AddTestScreen)

            # Fill in the form
            name_input = screen.query_one("#name-input", Input)
            name_input.value = "My Test"

            task_area = screen.query_one("#task-input", TextArea)
            task_area.text = "Do something important"

            tags_input = screen.query_one("#tags-input", Input)
            tags_input.value = "smoke, api"

            # Use button.press() as pilot.click() doesn't work for buttons
            add_btn = screen.query_one("#add-btn", Button)
            add_btn.press()
            await pilot.pause()

            # Should be back on main screen with test added
            assert isinstance(pilot.app.screen, MainScreen)
            assert pilot.app.screen.suite is not None
            assert len(pilot.app.screen.suite.tests) == 1

            added_test = pilot.app.screen.suite.tests[0]
            assert added_test.name == "My Test"
            assert added_test.task.description == "Do something important"
            assert added_test.tags == ["smoke", "api"]
            assert added_test.id == "test-001"  # Auto-generated

    async def test_add_test_with_custom_id(self) -> None:
        """Test adding a test with custom ID."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")

        app = ATPTUI()
        async with app.run_test(size=(140, 80)) as pilot:
            await pilot.press("m")
            main_screen = pilot.app.screen
            assert isinstance(main_screen, MainScreen)
            main_screen.set_suite(suite)

            await pilot.press("a")
            screen = pilot.app.screen
            assert isinstance(screen, AddTestScreen)

            # Fill in the form with custom ID
            id_input = screen.query_one("#id-input", Input)
            id_input.value = "custom-test-id"

            name_input = screen.query_one("#name-input", Input)
            name_input.value = "Custom Test"

            task_area = screen.query_one("#task-input", TextArea)
            task_area.text = "Task description"

            # Use button.press() as pilot.click() doesn't work for buttons
            add_btn = screen.query_one("#add-btn", Button)
            add_btn.press()
            await pilot.pause()

            assert isinstance(pilot.app.screen, MainScreen)
            assert pilot.app.screen.suite is not None
            added_test = pilot.app.screen.suite.tests[0]
            assert added_test.id == "custom-test-id"

    async def test_add_test_validation_fails_without_required_fields(self) -> None:
        """Test that validation fails when required fields are empty."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")

        app = ATPTUI()
        async with app.run_test(size=(140, 80)) as pilot:
            await pilot.press("m")
            main_screen = pilot.app.screen
            assert isinstance(main_screen, MainScreen)
            main_screen.set_suite(suite)

            await pilot.press("a")
            screen = pilot.app.screen
            assert isinstance(screen, AddTestScreen)

            # Try to add without filling required fields
            add_btn = screen.query_one("#add-btn", Button)
            add_btn.press()
            await pilot.pause()

            # Should still be on AddTestScreen
            assert isinstance(pilot.app.screen, AddTestScreen)

    async def test_add_test_with_constraints(self) -> None:
        """Test adding a test with custom constraints."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")

        app = ATPTUI()
        async with app.run_test(size=(140, 80)) as pilot:
            await pilot.press("m")
            main_screen = pilot.app.screen
            assert isinstance(main_screen, MainScreen)
            main_screen.set_suite(suite)

            await pilot.press("a")
            screen = pilot.app.screen
            assert isinstance(screen, AddTestScreen)

            # Fill in required fields
            name_input = screen.query_one("#name-input", Input)
            name_input.value = "Constrained Test"

            task_area = screen.query_one("#task-input", TextArea)
            task_area.text = "Task with constraints"

            # Fill in constraints
            max_steps_input = screen.query_one("#max-steps-input", Input)
            max_steps_input.value = "10"

            max_tokens_input = screen.query_one("#max-tokens-input", Input)
            max_tokens_input.value = "5000"

            timeout_input = screen.query_one("#timeout-input", Input)
            timeout_input.value = "120"

            budget_input = screen.query_one("#budget-input", Input)
            budget_input.value = "1.50"

            # Use button.press() as pilot.click() doesn't work for buttons
            add_btn = screen.query_one("#add-btn", Button)
            add_btn.press()
            await pilot.pause()

            assert isinstance(pilot.app.screen, MainScreen)
            assert pilot.app.screen.suite is not None
            added_test = pilot.app.screen.suite.tests[0]
            assert added_test.constraints.max_steps == 10
            assert added_test.constraints.max_tokens == 5000
            assert added_test.constraints.timeout_seconds == 120
            assert added_test.constraints.budget_usd == 1.50

    async def test_ctrl_s_submits_form(self) -> None:
        """Test that Ctrl+S submits the form."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")

        app = ATPTUI()
        async with app.run_test(size=(140, 80)) as pilot:
            await pilot.press("m")
            main_screen = pilot.app.screen
            assert isinstance(main_screen, MainScreen)
            main_screen.set_suite(suite)

            await pilot.press("a")
            screen = pilot.app.screen
            assert isinstance(screen, AddTestScreen)

            # Fill in required fields
            name_input = screen.query_one("#name-input", Input)
            name_input.value = "Ctrl-S Test"

            task_area = screen.query_one("#task-input", TextArea)
            task_area.text = "Submit with ctrl+s"

            # Press Ctrl+S
            await pilot.press("ctrl+s")

            # Should be on main screen with test added
            assert isinstance(pilot.app.screen, MainScreen)
            assert len(pilot.app.screen.suite.tests) == 1
            assert pilot.app.screen.suite.tests[0].name == "Ctrl-S Test"
