"""Main screen for the ATP TUI application.

Provides the primary interface for viewing and editing test suites
with a tree view on the left and YAML preview on the right.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from atp.tui.widgets.test_tree import TestTreeWidget
from atp.tui.widgets.yaml_preview import YAMLPreviewWidget

if TYPE_CHECKING:
    from atp.generator.core import TestSuiteData


class MainScreen(Screen):
    """Main screen with test tree and YAML preview.

    Features:
    - Left panel (40%): Tree view of test suite structure
    - Right panel (60%): YAML preview of selected item
    - Keyboard shortcuts for common actions

    Attributes:
        suite: The current test suite being edited.
        suite_path: Path to the current suite file, if any.
    """

    SCREEN_NAME = "main"

    BINDINGS = [
        Binding("n", "new_suite", "New Suite"),
        Binding("o", "open_suite", "Open"),
        Binding("s", "save_suite", "Save"),
        Binding("a", "add_test", "Add Test"),
        Binding("h", "switch_home", "Home"),
        Binding("question_mark", "switch_help", "Help"),
        Binding("escape", "switch_home", "Back"),
    ]

    DEFAULT_CSS = """
    MainScreen {
        layout: vertical;
    }

    #main-screen-container {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }

    #main-screen-content {
        width: 100%;
        height: 100%;
    }

    #tree-panel {
        width: 40%;
        height: 100%;
        padding-right: 1;
    }

    #preview-panel {
        width: 60%;
        height: 100%;
        padding-left: 1;
    }

    #test-tree {
        width: 100%;
        height: 100%;
        border: solid $primary;
        background: $surface;
    }

    #yaml-preview {
        width: 100%;
        height: 100%;
        border: solid $primary;
        background: $surface;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $surface-darken-1;
        color: $text-muted;
    }
    """

    class SuiteChanged(Message):
        """Message sent when the suite is modified."""

        def __init__(self, suite: TestSuiteData | None) -> None:
            """Initialize the message.

            Args:
                suite: The updated test suite.
            """
            self.suite = suite
            super().__init__()

    class RequestNewSuite(Message):
        """Message sent when user wants to create a new suite."""

        pass

    class RequestOpenSuite(Message):
        """Message sent when user wants to open a suite."""

        pass

    class RequestSaveSuite(Message):
        """Message sent when user wants to save the suite."""

        pass

    class RequestAddTest(Message):
        """Message sent when user wants to add a test."""

        pass

    def __init__(
        self,
        suite: TestSuiteData | None = None,
        suite_path: Path | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the main screen.

        Args:
            suite: Optional initial test suite data.
            suite_path: Optional path to the suite file.
            name: Optional screen name.
            id: Optional screen ID.
            classes: Optional CSS classes.
        """
        super().__init__(name=name, id=id, classes=classes)
        self._suite: TestSuiteData | None = suite
        self._suite_path: Path | None = suite_path
        self._modified: bool = False

    @property
    def suite(self) -> TestSuiteData | None:
        """Get the current test suite."""
        return self._suite

    @property
    def suite_path(self) -> Path | None:
        """Get the current suite file path."""
        return self._suite_path

    @property
    def is_modified(self) -> bool:
        """Check if the suite has unsaved changes."""
        return self._modified

    def compose(self) -> ComposeResult:
        """Compose the main screen layout."""
        yield Header()
        yield Container(
            Horizontal(
                Vertical(
                    TestTreeWidget(self._suite, id="test-tree"),
                    id="tree-panel",
                ),
                Vertical(
                    YAMLPreviewWidget(self._suite, id="yaml-preview"),
                    id="preview-panel",
                ),
                id="main-screen-content",
            ),
            Static(self._format_status(), id="status-bar"),
            id="main-screen-container",
        )
        yield Footer()

    def _format_status(self) -> str:
        """Format the status bar text.

        Returns:
            Status text showing file path and modification state.
        """
        if self._suite is None:
            return (
                "[dim]No suite loaded - "
                "Press [bold]n[/] to create or [bold]o[/] to open[/]"
            )

        status_parts = []

        if self._suite_path:
            status_parts.append(f"[cyan]{self._suite_path.name}[/]")
        else:
            status_parts.append(f"[cyan]{self._suite.name}[/]")

        status_parts.append(f"[dim]{len(self._suite.tests)} tests[/]")

        if self._modified:
            status_parts.append("[yellow]* Modified[/]")

        return " | ".join(status_parts)

    def _update_status(self) -> None:
        """Update the status bar."""
        status_bar = self.query_one("#status-bar", Static)
        status_bar.update(self._format_status())

    def set_suite(
        self,
        suite: TestSuiteData | None,
        path: Path | None = None,
    ) -> None:
        """Set a new test suite to display.

        Args:
            suite: The test suite to display, or None to clear.
            path: Optional path to the suite file.
        """
        self._suite = suite
        self._suite_path = path
        self._modified = False

        # Update widgets
        tree = self.query_one("#test-tree", TestTreeWidget)
        tree.set_suite(suite)

        preview = self.query_one("#yaml-preview", YAMLPreviewWidget)
        preview.set_suite(suite)

        self._update_status()
        self.post_message(self.SuiteChanged(suite))

    def mark_modified(self) -> None:
        """Mark the suite as having unsaved changes."""
        self._modified = True
        self._update_status()

    def mark_saved(self, path: Path | None = None) -> None:
        """Mark the suite as saved.

        Args:
            path: Optional new path if saved to a different location.
        """
        self._modified = False
        if path is not None:
            self._suite_path = path
        self._update_status()

    def on_test_tree_widget_test_selected(
        self,
        event: TestTreeWidget.TestSelected,
    ) -> None:
        """Handle test selection in the tree.

        Updates the YAML preview to show the selected test.
        """
        preview = self.query_one("#yaml-preview", YAMLPreviewWidget)
        preview.show_test(event.test_id)

    def on_test_tree_widget_suite_node_selected(
        self,
        event: TestTreeWidget.SuiteNodeSelected,
    ) -> None:
        """Handle suite root selection in the tree.

        Updates the YAML preview to show the full suite.
        """
        preview = self.query_one("#yaml-preview", YAMLPreviewWidget)
        preview.show_full_suite()

    def on_test_tree_widget_agent_selected(
        self,
        event: TestTreeWidget.AgentSelected,
    ) -> None:
        """Handle agent selection in the tree.

        Currently shows the full suite when an agent is selected.
        """
        preview = self.query_one("#yaml-preview", YAMLPreviewWidget)
        preview.show_full_suite()

    def action_new_suite(self) -> None:
        """Handle 'n' key - create new suite."""
        self.post_message(self.RequestNewSuite())

    def action_open_suite(self) -> None:
        """Handle 'o' key - open suite."""
        self.post_message(self.RequestOpenSuite())

    def action_save_suite(self) -> None:
        """Handle 's' key - save suite."""
        if self._suite is not None:
            self.post_message(self.RequestSaveSuite())

    def action_add_test(self) -> None:
        """Handle 'a' key - add test."""
        if self._suite is not None:
            self.post_message(self.RequestAddTest())

    def action_switch_home(self) -> None:
        """Switch to home screen."""
        self.app.switch_screen("home")

    def action_switch_help(self) -> None:
        """Switch to help screen."""
        self.app.switch_screen("help")
