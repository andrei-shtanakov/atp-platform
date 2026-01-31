"""YAML preview widget for displaying test suite content.

Provides a read-only preview of test suite YAML with syntax highlighting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.message import Message
from textual.widgets import Static

if TYPE_CHECKING:
    from atp.generator.core import TestSuiteData
    from atp.loader.models import TestDefinition


class YAMLPreviewWidget(Static):
    """Widget for displaying YAML preview of test suite or test.

    Shows formatted YAML content with syntax highlighting using
    Rich markup. Supports displaying full suite or individual tests.

    Attributes:
        suite: The current test suite data being displayed.
        selected_test_id: The currently selected test ID, if any.
    """

    class ContentChanged(Message):
        """Message sent when the preview content changes."""

        pass

    DEFAULT_CSS = """
    YAMLPreviewWidget {
        width: 100%;
        height: 100%;
        overflow-y: auto;
        padding: 1 2;
    }
    """

    def __init__(
        self,
        suite: TestSuiteData | None = None,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the YAML preview widget.

        Args:
            suite: Optional initial test suite data.
            id: Optional widget ID.
            classes: Optional CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._suite: TestSuiteData | None = suite
        self._selected_test_id: str | None = None
        self._show_full_suite: bool = True

    @property
    def suite(self) -> TestSuiteData | None:
        """Get the current test suite."""
        return self._suite

    @property
    def selected_test_id(self) -> str | None:
        """Get the currently selected test ID."""
        return self._selected_test_id

    def set_suite(self, suite: TestSuiteData | None) -> None:
        """Set and display a new test suite.

        Args:
            suite: The test suite to display, or None to clear.
        """
        self._suite = suite
        self._selected_test_id = None
        self._show_full_suite = True
        self._refresh_preview()

    def show_test(self, test_id: str) -> None:
        """Show preview for a specific test.

        Args:
            test_id: The ID of the test to display.
        """
        self._selected_test_id = test_id
        self._show_full_suite = False
        self._refresh_preview()

    def show_full_suite(self) -> None:
        """Show preview for the full test suite."""
        self._selected_test_id = None
        self._show_full_suite = True
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        """Rebuild the YAML preview content."""
        if self._suite is None:
            self.update(self._format_empty_state())
            return

        if self._show_full_suite:
            content = self._format_suite_yaml()
        else:
            content = self._format_test_yaml()

        self.update(content)
        self.post_message(self.ContentChanged())

    def _format_empty_state(self) -> str:
        """Format the empty state message.

        Returns:
            Formatted empty state with instructions.
        """
        return (
            "[bold cyan]YAML Preview[/]\n\n"
            "[dim]No test suite loaded.[/]\n\n"
            "[bold]Quick Actions:[/]\n"
            "  [cyan]n[/] - Create new suite\n"
            "  [cyan]o[/] - Open existing suite\n"
        )

    def _format_suite_yaml(self) -> str:
        """Format the full suite as YAML preview.

        Returns:
            Formatted YAML string with syntax highlighting.
        """
        if self._suite is None:
            return self._format_empty_state()

        lines = [
            "[bold cyan]# Test Suite Preview[/]",
            "",
            f'[yellow]test_suite:[/] [green]"{self._suite.name}"[/]',
            f'[yellow]version:[/] [green]"{self._suite.version}"[/]',
        ]

        if self._suite.description:
            lines.append(
                f'[yellow]description:[/] [green]"{self._suite.description}"[/]'
            )

        # Add defaults section
        runs = self._suite.defaults.runs_per_test
        timeout = self._suite.defaults.timeout_seconds
        lines.extend(
            [
                "",
                "[yellow]defaults:[/]",
                f"  [yellow]runs_per_test:[/] [cyan]{runs}[/]",
                f"  [yellow]timeout_seconds:[/] [cyan]{timeout}[/]",
            ]
        )

        # Add agents section if present
        if self._suite.agents:
            lines.extend(
                [
                    "",
                    "[yellow]agents:[/]",
                ]
            )
            for agent in self._suite.agents:
                lines.append(f'  [dim]-[/] [yellow]name:[/] [green]"{agent.name}"[/]')
                if agent.type:
                    lines.append(f'    [yellow]type:[/] [green]"{agent.type}"[/]')

        # Add tests section
        lines.extend(
            [
                "",
                "[yellow]tests:[/]",
            ]
        )

        for test in self._suite.tests:
            lines.append(f'  [dim]-[/] [yellow]id:[/] [green]"{test.id}"[/]')
            lines.append(f'    [yellow]name:[/] [green]"{test.name}"[/]')
            if test.tags:
                tags_str = ", ".join(f'"{t}"' for t in test.tags)
                lines.append(f"    [yellow]tags:[/] [{tags_str}]")
            lines.append("    [yellow]task:[/]")
            desc = test.task.description
            if len(desc) > 50:
                desc = desc[:47] + "..."
            lines.append(f'      [yellow]description:[/] [green]"{desc}"[/]')
            lines.append("")

        return "\n".join(lines)

    def _format_test_yaml(self) -> str:
        """Format a single test as YAML preview.

        Returns:
            Formatted YAML string for the selected test.
        """
        if self._suite is None or self._selected_test_id is None:
            return self._format_empty_state()

        test = self._find_test(self._selected_test_id)
        if test is None:
            return f"[red]Test not found: {self._selected_test_id}[/]"

        lines = [
            f"[bold cyan]# Test: {test.name}[/]",
            "",
            f'[yellow]id:[/] [green]"{test.id}"[/]',
            f'[yellow]name:[/] [green]"{test.name}"[/]',
        ]

        if test.description:
            lines.append(f'[yellow]description:[/] [green]"{test.description}"[/]')

        if test.tags:
            tags_str = ", ".join(f'"{t}"' for t in test.tags)
            lines.append(f"[yellow]tags:[/] [{tags_str}]")

        # Task section
        lines.extend(
            [
                "",
                "[yellow]task:[/]",
                "  [yellow]description:[/] |",
            ]
        )
        # Wrap description for readability
        desc_lines = self._wrap_text(test.task.description, 60)
        for desc_line in desc_lines:
            lines.append(f"    [green]{desc_line}[/]")

        if test.task.expected_artifacts:
            artifacts_str = ", ".join(f'"{a}"' for a in test.task.expected_artifacts)
            lines.append(f"  [yellow]expected_artifacts:[/] [{artifacts_str}]")

        # Constraints section
        lines.extend(
            [
                "",
                "[yellow]constraints:[/]",
            ]
        )
        if test.constraints.max_steps is not None:
            lines.append(
                f"  [yellow]max_steps:[/] [cyan]{test.constraints.max_steps}[/]"
            )
        if test.constraints.max_tokens is not None:
            lines.append(
                f"  [yellow]max_tokens:[/] [cyan]{test.constraints.max_tokens}[/]"
            )
        lines.append(
            f"  [yellow]timeout_seconds:[/] [cyan]{test.constraints.timeout_seconds}[/]"
        )
        if test.constraints.allowed_tools:
            tools_str = ", ".join(f'"{t}"' for t in test.constraints.allowed_tools)
            lines.append(f"  [yellow]allowed_tools:[/] [{tools_str}]")
        if test.constraints.budget_usd is not None:
            lines.append(
                f"  [yellow]budget_usd:[/] [cyan]{test.constraints.budget_usd}[/]"
            )

        # Assertions section
        if test.assertions:
            lines.extend(
                [
                    "",
                    "[yellow]assertions:[/]",
                ]
            )
            for assertion in test.assertions:
                lines.append(
                    f'  [dim]-[/] [yellow]type:[/] [green]"{assertion.type}"[/]'
                )
                if assertion.config:
                    lines.append("    [yellow]config:[/]")
                    for key, value in assertion.config.items():
                        if isinstance(value, str):
                            lines.append(f'      [yellow]{key}:[/] [green]"{value}"[/]')
                        else:
                            lines.append(f"      [yellow]{key}:[/] [cyan]{value}[/]")

        return "\n".join(lines)

    def _find_test(self, test_id: str) -> TestDefinition | None:
        """Find a test by ID in the current suite.

        Args:
            test_id: The test ID to find.

        Returns:
            The test definition, or None if not found.
        """
        if self._suite is None:
            return None
        for test in self._suite.tests:
            if test.id == test_id:
                return test
        return None

    def _wrap_text(self, text: str, width: int) -> list[str]:
        """Wrap text to specified width.

        Args:
            text: The text to wrap.
            width: Maximum line width.

        Returns:
            List of wrapped lines.
        """
        words = text.split()
        lines: list[str] = []
        current_line: list[str] = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        return lines if lines else [""]
