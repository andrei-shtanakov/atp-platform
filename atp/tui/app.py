"""ATP TUI Application.

Main application class for the ATP Terminal User Interface.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from atp import __version__


class WelcomePanel(Static):
    """Welcome panel displayed on the home screen."""

    def compose(self) -> ComposeResult:
        """Compose the welcome panel content."""
        yield Static(
            f"[bold cyan]ATP - Agent Test Platform v{__version__}[/]\n\n"
            "[dim]Framework-agnostic platform for testing "
            "and evaluating AI agents.[/]\n\n"
            "[bold]Quick Start:[/]\n"
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
            "  [cyan]s[/] - Suites screen\n"
            "  [cyan]r[/] - Results screen\n"
            "  [cyan]a[/] - Agents screen\n"
            "  [cyan]?[/] - This help screen\n\n"
            "[bold]Actions:[/]\n"
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
    #help-container {
        width: 100%;
        height: 100%;
        padding: 1 2;
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


def run_tui() -> None:
    """Run the ATP TUI application.

    This is the main entry point for running the TUI.
    """
    app = ATPTUI()
    app.run()
