"""Test tree widget for displaying test suite structure.

Provides a hierarchical view of test suites and their tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.message import Message
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

if TYPE_CHECKING:
    from atp.generator.core import TestSuiteData
    from atp.loader.models import TestDefinition


class TestTreeWidget(Tree[str]):
    """Widget for displaying test suite hierarchy.

    Shows a tree structure with the test suite at the root,
    agents as a branch, and tests as another branch with
    individual test items as leaves.

    Attributes:
        suite: The current test suite data being displayed.
    """

    class TestSelected(Message):
        """Message sent when a test is selected in the tree."""

        def __init__(self, test_id: str) -> None:
            """Initialize the message.

            Args:
                test_id: The ID of the selected test.
            """
            self.test_id = test_id
            super().__init__()

    class SuiteNodeSelected(Message):
        """Message sent when the suite root node is selected."""

        pass

    class AgentSelected(Message):
        """Message sent when an agent is selected in the tree."""

        def __init__(self, agent_name: str) -> None:
            """Initialize the message.

            Args:
                agent_name: The name of the selected agent.
            """
            self.agent_name = agent_name
            super().__init__()

    def __init__(
        self,
        suite: TestSuiteData | None = None,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the test tree widget.

        Args:
            suite: Optional initial test suite data.
            id: Optional widget ID.
            classes: Optional CSS classes.
        """
        super().__init__(
            "[bold]Test Suite[/]",
            id=id,
            classes=classes,
        )
        self._suite: TestSuiteData | None = suite
        self._test_nodes: dict[str, TreeNode[str]] = {}

    @property
    def suite(self) -> TestSuiteData | None:
        """Get the current test suite."""
        return self._suite

    def set_suite(self, suite: TestSuiteData | None) -> None:
        """Set and display a new test suite.

        Args:
            suite: The test suite to display, or None to clear.
        """
        self._suite = suite
        self._refresh_tree()

    def _refresh_tree(self) -> None:
        """Rebuild the tree structure from the current suite."""
        self.clear()
        self._test_nodes.clear()

        if self._suite is None:
            self.root.set_label("[dim]No suite loaded[/]")
            return

        # Set root label to suite name
        suite_label = f"[bold cyan]{self._suite.name}[/]"
        if self._suite.description:
            suite_label += f" [dim]- {self._suite.description[:40]}...[/]"
        self.root.set_label(suite_label)
        self.root.data = "suite"

        # Add agents branch if there are agents
        if self._suite.agents:
            agents_node = self.root.add("[bold]Agents[/]", data="agents_branch")
            agents_node.expand()
            for agent in self._suite.agents:
                agent_label = f"[cyan]{agent.name}[/]"
                if agent.type:
                    agent_label += f" [dim]({agent.type})[/]"
                agents_node.add_leaf(agent_label, data=f"agent:{agent.name}")

        # Add tests branch
        tests_node = self.root.add("[bold]Tests[/]", data="tests_branch")
        tests_node.expand()

        for test in self._suite.tests:
            test_label = self._format_test_label(test)
            node = tests_node.add_leaf(test_label, data=f"test:{test.id}")
            self._test_nodes[test.id] = node

        # Expand root
        self.root.expand()

    def _format_test_label(self, test: TestDefinition) -> str:
        """Format a test definition for display.

        Args:
            test: The test definition.

        Returns:
            Formatted label string with markup.
        """
        tags_str = ""
        if test.tags:
            tags = test.tags[:3]  # Show max 3 tags
            tags_str = " " + " ".join(f"[dim][{t}][/]" for t in tags)
            if len(test.tags) > 3:
                tags_str += f" [dim]+{len(test.tags) - 3}[/]"

        return f"[green]{test.id}[/] {test.name}{tags_str}"

    def add_test(self, test: TestDefinition) -> None:
        """Add a test to the tree.

        Args:
            test: The test definition to add.
        """
        if self._suite is None:
            return

        # Find or create tests branch
        tests_node = None
        for node in self.root.children:
            if node.data == "tests_branch":
                tests_node = node
                break

        if tests_node is None:
            tests_node = self.root.add("[bold]Tests[/]", data="tests_branch")
            tests_node.expand()

        test_label = self._format_test_label(test)
        node = tests_node.add_leaf(test_label, data=f"test:{test.id}")
        self._test_nodes[test.id] = node

    def remove_test(self, test_id: str) -> None:
        """Remove a test from the tree.

        Args:
            test_id: The ID of the test to remove.
        """
        if test_id in self._test_nodes:
            node = self._test_nodes[test_id]
            node.remove()
            del self._test_nodes[test_id]

    def highlight_test(self, test_id: str) -> None:
        """Highlight a specific test in the tree.

        Args:
            test_id: The ID of the test to highlight.
        """
        if test_id in self._test_nodes:
            node = self._test_nodes[test_id]
            self.select_node(node)
            self.scroll_to_node(node)

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        """Handle tree node selection.

        Posts appropriate messages based on what was selected.
        """
        data = event.node.data
        if data is None:
            return

        if data == "suite":
            self.post_message(self.SuiteNodeSelected())
        elif data.startswith("test:"):
            test_id = data[5:]  # Remove "test:" prefix
            self.post_message(self.TestSelected(test_id))
        elif data.startswith("agent:"):
            agent_name = data[6:]  # Remove "agent:" prefix
            self.post_message(self.AgentSelected(agent_name))
