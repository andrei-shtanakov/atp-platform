"""ATP TUI Application.

Main application class for the ATP Terminal User Interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from atp import __version__
from atp.tui.screens.main_menu import MainScreen
from atp.tui.screens.suite_editor import NewSuiteScreen
from atp.tui.screens.test_editor import AddTestScreen

if TYPE_CHECKING:
    pass


class WelcomePanel(Static):
    """Welcome panel displayed on the home screen."""

    def compose(self) -> ComposeResult:
        """Compose the welcome panel content."""
        yield Static(
            f"[bold cyan]ATP - Agent Test Platform v{__version__}[/]\n\n"
            "[dim]Framework-agnostic platform for testing "
            "and evaluating AI agents.[/]\n\n"
            "[bold]Quick Start:[/]\n"
            "  [cyan]m[/] - Suite editor (main screen)\n"
            "  [cyan]s[/] - View test suites\n"
            "  [cyan]r[/] - View recent results\n"
            "  [cyan]a[/] - View agents\n"
            "  [cyan]q[/] - Quit\n\n"
            "[dim]Press [bold]?[/] for help[/]",
            id="welcome-content",
        )


class SuitesPanel(Static):
    """Panel for displaying test suites."""

    def compose(self) -> ComposeResult:
        """Compose the suites panel content."""
        yield Static(
            "[bold]Test Suites[/]\n\n"
            "[dim]No suites loaded. Use 'atp test' to run tests.[/]",
            id="suites-content",
        )


class ResultsPanel(Static):
    """Panel for displaying test results."""

    def compose(self) -> ComposeResult:
        """Compose the results panel content."""
        yield Static(
            "[bold]Recent Results[/]\n\n"
            "[dim]No results available. Run tests to see results here.[/]",
            id="results-content",
        )


class AgentsPanel(Static):
    """Panel for displaying agents."""

    def compose(self) -> ComposeResult:
        """Compose the agents panel content."""
        yield Static(
            "[bold]Agents[/]\n\n"
            "[dim]No agents configured. Configure agents in atp.config.yaml.[/]",
            id="agents-content",
        )


class HelpPanel(Static):
    """Panel for displaying help information."""

    def compose(self) -> ComposeResult:
        """Compose the help panel content."""
        yield Static(
            "[bold cyan]ATP TUI Help[/]\n\n"
            "[bold]Navigation:[/]\n"
            "  [cyan]h[/] - Home screen\n"
            "  [cyan]m[/] - Main screen (suite editor)\n"
            "  [cyan]s[/] - Suites screen\n"
            "  [cyan]r[/] - Results screen\n"
            "  [cyan]a[/] - Agents screen\n"
            "  [cyan]?[/] - This help screen\n\n"
            "[bold]Main Screen Actions:[/]\n"
            "  [cyan]n[/] - New suite\n"
            "  [cyan]o[/] - Open suite\n"
            "  [cyan]s[/] - Save suite\n"
            "  [cyan]a[/] - Add test\n\n"
            "[bold]General:[/]\n"
            "  [cyan]q[/] - Quit application\n"
            "  [cyan]Escape[/] - Go back / Cancel\n"
            "  [cyan]Enter[/] - Select item\n"
            "  [cyan]Arrow keys[/] - Navigate lists\n\n"
            "[dim]Press any navigation key to continue...[/]",
            id="help-content",
        )


class HomeScreen(Screen):
    """Home screen of the TUI application."""

    SCREEN_NAME = "home"

    BINDINGS = [
        Binding("m", "switch_main", "Main"),
        Binding("s", "switch_suites", "Suites"),
        Binding("r", "switch_results", "Results"),
        Binding("a", "switch_agents", "Agents"),
        Binding("question_mark", "switch_help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the home screen layout."""
        yield Header()
        yield Container(
            Horizontal(
                Vertical(
                    WelcomePanel(id="welcome-panel"),
                    id="left-column",
                ),
                Vertical(
                    ResultsPanel(id="quick-results"),
                    id="right-column",
                ),
                id="main-content",
            ),
            id="home-container",
        )
        yield Footer()

    def action_switch_main(self) -> None:
        """Switch to main screen."""
        self.app.switch_screen("main")

    def action_switch_suites(self) -> None:
        """Switch to suites screen."""
        self.app.switch_screen("suites")

    def action_switch_results(self) -> None:
        """Switch to results screen."""
        self.app.switch_screen("results")

    def action_switch_agents(self) -> None:
        """Switch to agents screen."""
        self.app.switch_screen("agents")

    def action_switch_help(self) -> None:
        """Switch to help screen."""
        self.app.switch_screen("help")


class SuitesScreen(Screen):
    """Screen for viewing test suites."""

    SCREEN_NAME = "suites"

    BINDINGS = [
        Binding("h", "switch_home", "Home"),
        Binding("r", "switch_results", "Results"),
        Binding("a", "switch_agents", "Agents"),
        Binding("escape", "switch_home", "Back"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the suites screen layout."""
        yield Header()
        yield Container(
            SuitesPanel(id="suites-main"),
            id="suites-container",
        )
        yield Footer()

    def action_switch_home(self) -> None:
        """Switch to home screen."""
        self.app.switch_screen("home")

    def action_switch_results(self) -> None:
        """Switch to results screen."""
        self.app.switch_screen("results")

    def action_switch_agents(self) -> None:
        """Switch to agents screen."""
        self.app.switch_screen("agents")


class ResultsScreen(Screen):
    """Screen for viewing test results."""

    SCREEN_NAME = "results"

    BINDINGS = [
        Binding("h", "switch_home", "Home"),
        Binding("s", "switch_suites", "Suites"),
        Binding("a", "switch_agents", "Agents"),
        Binding("escape", "switch_home", "Back"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the results screen layout."""
        yield Header()
        yield Container(
            ResultsPanel(id="results-main"),
            id="results-container",
        )
        yield Footer()

    def action_switch_home(self) -> None:
        """Switch to home screen."""
        self.app.switch_screen("home")

    def action_switch_suites(self) -> None:
        """Switch to suites screen."""
        self.app.switch_screen("suites")

    def action_switch_agents(self) -> None:
        """Switch to agents screen."""
        self.app.switch_screen("agents")


class AgentsScreen(Screen):
    """Screen for viewing agents."""

    SCREEN_NAME = "agents"

    BINDINGS = [
        Binding("h", "switch_home", "Home"),
        Binding("s", "switch_suites", "Suites"),
        Binding("r", "switch_results", "Results"),
        Binding("escape", "switch_home", "Back"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the agents screen layout."""
        yield Header()
        yield Container(
            AgentsPanel(id="agents-main"),
            id="agents-container",
        )
        yield Footer()

    def action_switch_home(self) -> None:
        """Switch to home screen."""
        self.app.switch_screen("home")

    def action_switch_suites(self) -> None:
        """Switch to suites screen."""
        self.app.switch_screen("suites")

    def action_switch_results(self) -> None:
        """Switch to results screen."""
        self.app.switch_screen("results")


class HelpScreen(Screen):
    """Screen for displaying help information."""

    SCREEN_NAME = "help"

    BINDINGS = [
        Binding("h", "switch_home", "Home"),
        Binding("m", "switch_main", "Main"),
        Binding("s", "switch_suites", "Suites"),
        Binding("r", "switch_results", "Results"),
        Binding("a", "switch_agents", "Agents"),
        Binding("escape", "switch_home", "Back"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the help screen layout."""
        yield Header()
        yield Container(
            HelpPanel(id="help-main"),
            id="help-container",
        )
        yield Footer()

    def action_switch_home(self) -> None:
        """Switch to home screen."""
        self.app.switch_screen("home")

    def action_switch_main(self) -> None:
        """Switch to main screen."""
        self.app.switch_screen("main")

    def action_switch_suites(self) -> None:
        """Switch to suites screen."""
        self.app.switch_screen("suites")

    def action_switch_results(self) -> None:
        """Switch to results screen."""
        self.app.switch_screen("results")

    def action_switch_agents(self) -> None:
        """Switch to agents screen."""
        self.app.switch_screen("agents")


class ATPTUI(App):
    """ATP Terminal User Interface Application.

    A Textual-based TUI for interacting with ATP test results,
    test suites, and agent configurations.

    Handles messages from editor screens to create/modify suites and tests.
    """

    TITLE = "ATP - Agent Test Platform"
    SUB_TITLE = f"v{__version__}"

    CSS = """
    /* Global styles */
    Screen {
        background: $surface;
    }

    /* Header styling */
    Header {
        dock: top;
        height: 3;
        background: $primary;
    }

    /* Footer styling */
    Footer {
        dock: bottom;
        height: 1;
        background: $primary-darken-2;
    }

    /* Main container styles */
    #home-container {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }

    #suites-container,
    #results-container,
    #agents-container,
    #help-container,
    #main-screen-container {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }

    /* Main screen layout */
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
        overflow-y: auto;
        padding: 1 2;
    }

    #status-bar {
        height: 1;
        padding: 0 2;
        background: $surface-darken-1;
        color: $text-muted;
    }

    /* Main content layout */
    #main-content {
        width: 100%;
        height: 100%;
    }

    #left-column {
        width: 50%;
        height: 100%;
        padding-right: 1;
    }

    #right-column {
        width: 50%;
        height: 100%;
        padding-left: 1;
    }

    /* Panel styles */
    #welcome-panel,
    #quick-results,
    #suites-main,
    #results-main,
    #agents-main,
    #help-main {
        width: 100%;
        height: 100%;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }

    #welcome-panel {
        border: double $accent;
    }

    #help-main {
        border: solid $warning;
    }

    /* Content styling */
    #welcome-content,
    #suites-content,
    #results-content,
    #agents-content,
    #help-content {
        width: 100%;
    }

    /* Status indicators */
    .success {
        color: $success;
    }

    .error {
        color: $error;
    }

    .warning {
        color: $warning;
    }

    .info {
        color: $primary;
    }
    """

    SCREENS = {
        "home": HomeScreen,
        "main": MainScreen,
        "suites": SuitesScreen,
        "results": ResultsScreen,
        "agents": AgentsScreen,
        "help": HelpScreen,
    }

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def on_mount(self) -> None:
        """Handle application mount event."""
        self.push_screen("home")

    def on_main_screen_request_new_suite(
        self,
        event: MainScreen.RequestNewSuite,
    ) -> None:
        """Handle request to create a new suite."""
        self.push_screen(NewSuiteScreen())

    def on_main_screen_request_add_test(
        self,
        event: MainScreen.RequestAddTest,
    ) -> None:
        """Handle request to add a new test."""
        screen = self.screen
        if isinstance(screen, MainScreen) and screen.suite is not None:
            self.push_screen(AddTestScreen(suite=screen.suite))

    def on_new_suite_screen_suite_created(
        self,
        event: NewSuiteScreen.SuiteCreated,
    ) -> None:
        """Handle suite creation from the new suite screen."""
        # Switch to main screen and set the new suite
        self.switch_screen("main")
        screen = self.screen
        if isinstance(screen, MainScreen):
            screen.set_suite(event.suite)
            screen.mark_modified()

    def on_add_test_screen_test_created(
        self,
        event: AddTestScreen.TestCreated,
    ) -> None:
        """Handle test creation from the add test screen."""
        # Get the main screen (should be below the modal)
        main_screen = None
        for screen in self.screen_stack:
            if isinstance(screen, MainScreen):
                main_screen = screen
                break

        if main_screen is not None and main_screen.suite is not None:
            # Add the test to the suite
            main_screen.suite.tests.append(event.test)
            # Update the UI
            main_screen.set_suite(main_screen.suite, main_screen.suite_path)
            main_screen.mark_modified()


def run_tui() -> None:
    """Run the ATP TUI application.

    This is the main entry point for running the TUI.
    """
    app = ATPTUI()
    app.run()
