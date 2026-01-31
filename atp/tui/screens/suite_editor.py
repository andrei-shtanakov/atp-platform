"""Suite editor screen for creating new test suites.

Provides a form-based interface for creating new test suites
with all required and optional properties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.validation import Length, Number
from textual.widgets import Button, Footer, Header, Input, Label, Static, TextArea

if TYPE_CHECKING:
    from atp.generator.core import TestSuiteData


class NewSuiteScreen(ModalScreen[None]):
    """Screen for creating a new test suite.

    Provides input fields for:
    - Suite name (required)
    - Description (optional)
    - Version (default: 1.0)
    - Default runs per test (default: 1)
    - Default timeout in seconds (default: 300)

    Attributes:
        SCREEN_NAME: The name used for screen registration.
    """

    SCREEN_NAME = "new_suite"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "submit", "Create", priority=True),
    ]

    DEFAULT_CSS = """
    NewSuiteScreen {
        align: center middle;
    }

    #new-suite-dialog {
        width: 70;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }

    #dialog-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        padding-bottom: 1;
    }

    .form-row {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    .form-label {
        width: 100%;
        height: 1;
        color: $text;
    }

    .required-marker {
        color: $error;
    }

    .form-input {
        width: 100%;
    }

    .form-hint {
        width: 100%;
        height: 1;
        color: $text-muted;
        text-style: italic;
    }

    #description-input {
        height: 4;
    }

    .section-header {
        width: 100%;
        height: 1;
        color: $primary;
        text-style: bold;
        border-bottom: solid $primary;
        margin-top: 1;
        margin-bottom: 1;
    }

    #button-row {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }

    #button-row Button {
        margin: 0 1;
    }

    .validation-error {
        color: $error;
        height: 1;
        width: 100%;
    }

    #error-message {
        color: $error;
        text-align: center;
        height: auto;
        padding: 1;
    }
    """

    class SuiteCreated(Message):
        """Message sent when a new suite is created successfully."""

        def __init__(self, suite: TestSuiteData) -> None:
            """Initialize the message.

            Args:
                suite: The newly created test suite.
            """
            self.suite = suite
            super().__init__()

    class Cancelled(Message):
        """Message sent when suite creation is cancelled."""

        pass

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the new suite screen.

        Args:
            name: Optional screen name.
            id: Optional screen ID.
            classes: Optional CSS classes.
        """
        super().__init__(name=name, id=id, classes=classes)
        self._validation_errors: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        yield Header()
        yield Container(
            Static("Create New Test Suite", id="dialog-title"),
            # Basic Info Section
            Static("Basic Information", classes="section-header"),
            Vertical(
                Label("[bold]Suite Name[/] [red]*[/]", classes="form-label"),
                Input(
                    placeholder="e.g., my-test-suite",
                    id="name-input",
                    classes="form-input",
                    validators=[
                        Length(minimum=1, failure_description="Name is required")
                    ],
                ),
                Static("", id="name-error", classes="validation-error"),
                classes="form-row",
            ),
            Vertical(
                Label("Description", classes="form-label"),
                TextArea(
                    id="description-input",
                    classes="form-input",
                ),
                Static("Optional description for the test suite", classes="form-hint"),
                classes="form-row",
            ),
            Vertical(
                Label("Version", classes="form-label"),
                Input(
                    value="1.0",
                    placeholder="1.0",
                    id="version-input",
                    classes="form-input",
                ),
                classes="form-row",
            ),
            # Defaults Section
            Static("Default Settings", classes="section-header"),
            Horizontal(
                Vertical(
                    Label("Runs per Test", classes="form-label"),
                    Input(
                        value="1",
                        id="runs-input",
                        classes="form-input",
                        validators=[
                            Number(
                                minimum=1,
                                failure_description="Must be at least 1",
                            )
                        ],
                    ),
                    Static("", id="runs-error", classes="validation-error"),
                    classes="form-row",
                ),
                Vertical(
                    Label("Timeout (seconds)", classes="form-label"),
                    Input(
                        value="300",
                        id="timeout-input",
                        classes="form-input",
                        validators=[
                            Number(
                                minimum=1,
                                failure_description="Must be at least 1",
                            )
                        ],
                    ),
                    Static("", id="timeout-error", classes="validation-error"),
                    classes="form-row",
                ),
            ),
            # Error message area
            Static("", id="error-message"),
            # Buttons
            Horizontal(
                Button("Create", variant="primary", id="create-btn"),
                Button("Cancel", variant="default", id="cancel-btn"),
                id="button-row",
            ),
            id="new-suite-dialog",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Focus the name input on mount."""
        self.query_one("#name-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes for validation feedback."""
        input_widget = event.input
        if input_widget.id is None:
            return

        error_id = f"{input_widget.id.replace('-input', '')}-error"

        try:
            error_widget = self.query_one(f"#{error_id}", Static)
            if not event.validation_result or event.validation_result.is_valid:
                error_widget.update("")
            else:
                failures = event.validation_result.failure_descriptions
                error_widget.update(failures[0] if failures else "Invalid value")
        except Exception:
            # Error widget might not exist for all inputs
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "create-btn":
            self._submit_form()
        elif event.button.id == "cancel-btn":
            self._cancel()

    def _validate_form(self) -> bool:
        """Validate all form inputs.

        Returns:
            True if form is valid, False otherwise.
        """
        is_valid = True
        self._validation_errors.clear()

        # Validate name (required)
        name_input = self.query_one("#name-input", Input)
        if not name_input.value.strip():
            self._validation_errors["name"] = "Suite name is required"
            is_valid = False

        # Validate runs_per_test (must be positive integer)
        runs_input = self.query_one("#runs-input", Input)
        try:
            runs = int(runs_input.value)
            if runs < 1:
                self._validation_errors["runs"] = "Must be at least 1"
                is_valid = False
        except ValueError:
            self._validation_errors["runs"] = "Must be a valid number"
            is_valid = False

        # Validate timeout (must be positive integer)
        timeout_input = self.query_one("#timeout-input", Input)
        try:
            timeout = int(timeout_input.value)
            if timeout < 1:
                self._validation_errors["timeout"] = "Must be at least 1"
                is_valid = False
        except ValueError:
            self._validation_errors["timeout"] = "Must be a valid number"
            is_valid = False

        # Update error displays
        for field_name, error in self._validation_errors.items():
            try:
                error_widget = self.query_one(f"#{field_name}-error", Static)
                error_widget.update(f"[red]{error}[/]")
            except Exception:
                pass

        return is_valid

    def _submit_form(self) -> None:
        """Submit the form and create the suite."""
        if not self._validate_form():
            error_msg = self.query_one("#error-message", Static)
            error_msg.update("[red]Please fix the errors above[/]")
            return

        # Import here to avoid circular imports
        from atp.generator.core import TestSuiteData
        from atp.loader.models import TestDefaults

        # Get values
        name = self.query_one("#name-input", Input).value.strip()
        description_area = self.query_one("#description-input", TextArea)
        description = description_area.text.strip() or None
        version = self.query_one("#version-input", Input).value.strip() or "1.0"
        runs_per_test = int(self.query_one("#runs-input", Input).value)
        timeout_seconds = int(self.query_one("#timeout-input", Input).value)

        # Create the suite
        suite = TestSuiteData(
            name=name,
            version=version,
            description=description,
            defaults=TestDefaults(
                runs_per_test=runs_per_test,
                timeout_seconds=timeout_seconds,
            ),
            agents=[],
            tests=[],
        )

        self.post_message(self.SuiteCreated(suite))
        self.dismiss(None)

    def _cancel(self) -> None:
        """Cancel suite creation."""
        self.post_message(self.Cancelled())
        self.dismiss(None)

    def action_cancel(self) -> None:
        """Handle escape key - cancel creation."""
        self._cancel()

    def action_submit(self) -> None:
        """Handle ctrl+s - submit form."""
        self._submit_form()
