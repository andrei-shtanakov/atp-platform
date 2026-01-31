"""Test editor screen for adding new tests to a suite.

Provides a form-based interface for adding new tests
with all required and optional properties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.validation import Length, Number
from textual.widgets import Button, Footer, Header, Input, Label, Static, TextArea

if TYPE_CHECKING:
    from atp.generator.core import TestSuiteData
    from atp.loader.models import TestDefinition


class AddTestScreen(ModalScreen[None]):
    """Screen for adding a new test to a suite.

    Provides input fields for:
    - Test ID (auto-generated if empty)
    - Test name (required)
    - Description (optional)
    - Tags (comma-separated)
    - Task description (required)
    - Expected artifacts (comma-separated, optional)
    - Constraints:
        - Max steps (optional)
        - Max tokens (optional)
        - Timeout seconds (default: 300)
        - Budget USD (optional)

    Attributes:
        SCREEN_NAME: The name used for screen registration.
    """

    SCREEN_NAME = "add_test"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "submit", "Add Test", priority=True),
    ]

    DEFAULT_CSS = """
    AddTestScreen {
        align: center middle;
    }

    #add-test-dialog {
        width: 80;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }

    #dialog-content {
        width: 100%;
        height: auto;
        max-height: 100%;
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

    #description-input, #task-input {
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

    .half-width {
        width: 50%;
    }

    #constraints-row {
        width: 100%;
        height: auto;
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

    class TestCreated(Message):
        """Message sent when a new test is created successfully."""

        def __init__(self, test: TestDefinition) -> None:
            """Initialize the message.

            Args:
                test: The newly created test definition.
            """
            self.test = test
            super().__init__()

    class Cancelled(Message):
        """Message sent when test creation is cancelled."""

        pass

    def __init__(
        self,
        suite: TestSuiteData | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the add test screen.

        Args:
            suite: The test suite to add the test to (used for ID generation).
            name: Optional screen name.
            id: Optional screen ID.
            classes: Optional CSS classes.
        """
        super().__init__(name=name, id=id, classes=classes)
        self._suite = suite
        self._validation_errors: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        yield Header()
        yield Container(
            Static("Add New Test", id="dialog-title"),
            ScrollableContainer(
                # Basic Info Section
                Static("Test Information", classes="section-header"),
                Horizontal(
                    Vertical(
                        Label("Test ID", classes="form-label"),
                        Input(
                            placeholder="Auto-generated if empty",
                            id="id-input",
                            classes="form-input",
                        ),
                        Static(
                            "Leave empty for auto-generated ID", classes="form-hint"
                        ),
                        classes="form-row half-width",
                    ),
                    Vertical(
                        Label("[bold]Test Name[/] [red]*[/]", classes="form-label"),
                        Input(
                            placeholder="e.g., Create README file",
                            id="name-input",
                            classes="form-input",
                            validators=[
                                Length(
                                    minimum=1, failure_description="Name is required"
                                )
                            ],
                        ),
                        Static("", id="name-error", classes="validation-error"),
                        classes="form-row half-width",
                    ),
                ),
                Vertical(
                    Label("Description", classes="form-label"),
                    TextArea(
                        id="description-input",
                        classes="form-input",
                    ),
                    Static(
                        "Optional description of what the test does",
                        classes="form-hint",
                    ),
                    classes="form-row",
                ),
                Vertical(
                    Label("Tags", classes="form-label"),
                    Input(
                        placeholder="smoke, api, slow (comma-separated)",
                        id="tags-input",
                        classes="form-input",
                    ),
                    Static("Comma-separated list of tags", classes="form-hint"),
                    classes="form-row",
                ),
                # Task Section
                Static("Task Definition", classes="section-header"),
                Vertical(
                    Label("[bold]Task Description[/] [red]*[/]", classes="form-label"),
                    TextArea(
                        id="task-input",
                        classes="form-input",
                    ),
                    Static("", id="task-error", classes="validation-error"),
                    classes="form-row",
                ),
                Vertical(
                    Label("Expected Artifacts", classes="form-label"),
                    Input(
                        placeholder="README.md, output.json (comma-separated)",
                        id="artifacts-input",
                        classes="form-input",
                    ),
                    Static(
                        "Expected files the agent should create", classes="form-hint"
                    ),
                    classes="form-row",
                ),
                # Constraints Section
                Static("Constraints", classes="section-header"),
                Horizontal(
                    Vertical(
                        Label("Max Steps", classes="form-label"),
                        Input(
                            placeholder="No limit",
                            id="max-steps-input",
                            classes="form-input",
                            validators=[
                                Number(
                                    minimum=1,
                                    failure_description="Must be at least 1",
                                )
                            ],
                        ),
                        classes="form-row half-width",
                    ),
                    Vertical(
                        Label("Max Tokens", classes="form-label"),
                        Input(
                            placeholder="No limit",
                            id="max-tokens-input",
                            classes="form-input",
                            validators=[
                                Number(
                                    minimum=1,
                                    failure_description="Must be at least 1",
                                )
                            ],
                        ),
                        classes="form-row half-width",
                    ),
                    id="constraints-row",
                ),
                Horizontal(
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
                        classes="form-row half-width",
                    ),
                    Vertical(
                        Label("Budget (USD)", classes="form-label"),
                        Input(
                            placeholder="No limit",
                            id="budget-input",
                            classes="form-input",
                            validators=[
                                Number(
                                    minimum=0,
                                    failure_description="Must be non-negative",
                                )
                            ],
                        ),
                        classes="form-row half-width",
                    ),
                ),
                id="dialog-content",
            ),
            # Error message area
            Static("", id="error-message"),
            # Buttons
            Horizontal(
                Button("Add Test", variant="primary", id="add-btn"),
                Button("Cancel", variant="default", id="cancel-btn"),
                id="button-row",
            ),
            id="add-test-dialog",
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

        # Map input id to error widget id
        error_id_map = {
            "name-input": "name-error",
            "timeout-input": "timeout-error",
        }

        error_id = error_id_map.get(input_widget.id)
        if error_id is None:
            return

        try:
            error_widget = self.query_one(f"#{error_id}", Static)
            if not event.validation_result or event.validation_result.is_valid:
                error_widget.update("")
            else:
                failures = event.validation_result.failure_descriptions
                error_widget.update(failures[0] if failures else "Invalid value")
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "add-btn":
            self._submit_form()
        elif event.button.id == "cancel-btn":
            self._cancel()

    def _generate_test_id(self) -> str:
        """Generate a unique test ID.

        Returns:
            A unique test ID in the format 'test-001', 'test-002', etc.
        """
        if self._suite is None:
            return "test-001"

        existing_ids = {t.id for t in self._suite.tests}
        for i in range(1, 1000):
            test_id = f"test-{i:03d}"
            if test_id not in existing_ids:
                return test_id

        # Fallback with timestamp
        import time

        return f"test-{int(time.time())}"

    def _parse_comma_list(self, value: str) -> list[str]:
        """Parse a comma-separated string into a list.

        Args:
            value: Comma-separated string.

        Returns:
            List of trimmed non-empty strings.
        """
        if not value.strip():
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    def _parse_optional_int(self, value: str) -> int | None:
        """Parse an optional integer value.

        Args:
            value: String value to parse.

        Returns:
            Integer value or None if empty.
        """
        if not value.strip():
            return None
        return int(value)

    def _parse_optional_float(self, value: str) -> float | None:
        """Parse an optional float value.

        Args:
            value: String value to parse.

        Returns:
            Float value or None if empty.
        """
        if not value.strip():
            return None
        return float(value)

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
            self._validation_errors["name"] = "Test name is required"
            is_valid = False

        # Validate task description (required)
        task_input = self.query_one("#task-input", TextArea)
        if not task_input.text.strip():
            self._validation_errors["task"] = "Task description is required"
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

        # Validate optional numeric fields
        max_steps_input = self.query_one("#max-steps-input", Input)
        if max_steps_input.value.strip():
            try:
                steps = int(max_steps_input.value)
                if steps < 1:
                    self._validation_errors["max-steps"] = "Must be at least 1"
                    is_valid = False
            except ValueError:
                self._validation_errors["max-steps"] = "Must be a valid number"
                is_valid = False

        max_tokens_input = self.query_one("#max-tokens-input", Input)
        if max_tokens_input.value.strip():
            try:
                tokens = int(max_tokens_input.value)
                if tokens < 1:
                    self._validation_errors["max-tokens"] = "Must be at least 1"
                    is_valid = False
            except ValueError:
                self._validation_errors["max-tokens"] = "Must be a valid number"
                is_valid = False

        budget_input = self.query_one("#budget-input", Input)
        if budget_input.value.strip():
            try:
                budget = float(budget_input.value)
                if budget < 0:
                    self._validation_errors["budget"] = "Must be non-negative"
                    is_valid = False
            except ValueError:
                self._validation_errors["budget"] = "Must be a valid number"
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
        """Submit the form and create the test."""
        if not self._validate_form():
            error_msg = self.query_one("#error-message", Static)
            error_msg.update("[red]Please fix the errors above[/]")
            return

        # Import here to avoid circular imports
        from atp.loader.models import Constraints, TaskDefinition, TestDefinition

        # Get values
        test_id = self.query_one("#id-input", Input).value.strip()
        if not test_id:
            test_id = self._generate_test_id()

        test_name = self.query_one("#name-input", Input).value.strip()

        description_area = self.query_one("#description-input", TextArea)
        description = description_area.text.strip() or None

        tags = self._parse_comma_list(self.query_one("#tags-input", Input).value)

        task_area = self.query_one("#task-input", TextArea)
        task_description = task_area.text.strip()

        artifacts = self._parse_comma_list(
            self.query_one("#artifacts-input", Input).value
        )
        expected_artifacts = artifacts if artifacts else None

        # Constraints
        max_steps = self._parse_optional_int(
            self.query_one("#max-steps-input", Input).value
        )
        max_tokens = self._parse_optional_int(
            self.query_one("#max-tokens-input", Input).value
        )
        timeout_seconds = int(self.query_one("#timeout-input", Input).value)
        budget_usd = self._parse_optional_float(
            self.query_one("#budget-input", Input).value
        )

        # Create the test definition
        test = TestDefinition(
            id=test_id,
            name=test_name,
            description=description,
            tags=tags,
            task=TaskDefinition(
                description=task_description,
                expected_artifacts=expected_artifacts,
            ),
            constraints=Constraints(
                max_steps=max_steps,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                budget_usd=budget_usd,
            ),
            assertions=[],
        )

        self.post_message(self.TestCreated(test))
        self.dismiss(None)

    def _cancel(self) -> None:
        """Cancel test creation."""
        self.post_message(self.Cancelled())
        self.dismiss(None)

    def action_cancel(self) -> None:
        """Handle escape key - cancel creation."""
        self._cancel()

    def action_submit(self) -> None:
        """Handle ctrl+s - submit form."""
        self._submit_form()
