"""Tests for TestTreeWidget."""

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
    from atp.tui.widgets.test_tree import TestTreeWidget


class TestTestTreeWidget:
    """Test TestTreeWidget class."""

    def test_instantiation(self) -> None:
        """Test that TestTreeWidget can be instantiated."""
        widget = TestTreeWidget()
        assert widget is not None
        assert widget.suite is None

    def test_instantiation_with_suite(self) -> None:
        """Test instantiation with a test suite."""
        generator = TestGenerator()
        suite = generator.create_suite("test_suite", "Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        widget = TestTreeWidget(suite)
        assert widget.suite is not None
        assert widget.suite.name == "test_suite"

    def test_set_suite(self) -> None:
        """Test setting a suite after instantiation."""
        widget = TestTreeWidget()
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

        widget = TestTreeWidget(suite)
        assert widget.suite is not None

        widget.set_suite(None)
        assert widget.suite is None

    def test_test_selected_message(self) -> None:
        """Test that TestSelected message is created correctly."""
        message = TestTreeWidget.TestSelected("test-001")
        assert message.test_id == "test-001"

    def test_agent_selected_message(self) -> None:
        """Test that AgentSelected message is created correctly."""
        message = TestTreeWidget.AgentSelected("agent-1")
        assert message.agent_name == "agent-1"

    def test_suite_node_selected_message(self) -> None:
        """Test that SuiteNodeSelected message is created correctly."""
        message = TestTreeWidget.SuiteNodeSelected()
        assert message is not None


@pytest.mark.anyio
class TestTestTreeWidgetIntegration:
    """Integration tests for TestTreeWidget."""

    async def test_tree_displays_suite_name(self) -> None:
        """Test that the tree displays the suite name."""
        from textual.app import App, ComposeResult

        generator = TestGenerator()
        suite = generator.create_suite("my_suite", "My Test Suite")
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield TestTreeWidget(suite, id="tree")

        app = TestApp()
        async with app.run_test() as pilot:
            tree = pilot.app.query_one("#tree", TestTreeWidget)
            assert tree.suite is not None
            assert tree.suite.name == "my_suite"

    async def test_tree_shows_tests(self) -> None:
        """Test that the tree shows tests."""
        from textual.app import App, ComposeResult

        generator = TestGenerator()
        suite = generator.create_suite("my_suite", "My Test Suite")
        test1 = generator.create_custom_test("test-001", "Test One", "First test")
        test2 = generator.create_custom_test("test-002", "Test Two", "Second test")
        suite = generator.add_test(suite, test1)
        suite = generator.add_test(suite, test2)

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield TestTreeWidget(suite, id="tree")

        app = TestApp()
        async with app.run_test() as pilot:
            tree = pilot.app.query_one("#tree", TestTreeWidget)
            assert len(tree.suite.tests) == 2

    async def test_tree_shows_agents(self) -> None:
        """Test that the tree shows agents."""
        from textual.app import App, ComposeResult

        generator = TestGenerator()
        suite = generator.create_suite("my_suite", "My Test Suite")
        suite = generator.add_agent(
            suite, "agent1", "http", {"endpoint": "http://localhost"}
        )
        test = generator.create_custom_test("test-001", "Test One", "Do something")
        suite = generator.add_test(suite, test)

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield TestTreeWidget(suite, id="tree")

        app = TestApp()
        async with app.run_test() as pilot:
            tree = pilot.app.query_one("#tree", TestTreeWidget)
            assert len(tree.suite.agents) == 1
            assert tree.suite.agents[0].name == "agent1"
