"""Compact console and JSON reporter."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Literal, TextIO

from atp.core.results import SuiteReport
from atp.reporters.base import Reporter
from atp.reporters.summary_models import CompactSuiteSummary, CompactTestSummary


class SummaryReporter(Reporter):
    """Reporter that renders a compact suite summary."""

    def __init__(
        self,
        output_file: Path | str | None = None,
        output: TextIO | None = None,
        format: Literal["console", "json"] = "console",
        indent: int | None = 2,
        include_passed: bool = False,
        max_failures: int | None = None,
        use_colors: bool = True,
    ) -> None:
        """Initialize the compact summary reporter."""
        self._output_file = Path(output_file) if output_file else None
        self._output = output
        self._format = format
        self._indent = indent
        self._include_passed = include_passed
        self._max_failures = max_failures
        self._use_colors = use_colors and self._supports_color()

    @property
    def name(self) -> str:
        """Return the reporter name."""
        return "summary"

    def report(self, report: SuiteReport) -> None:
        """Generate and output the compact report."""
        summary = CompactSuiteSummary.from_report(
            report,
            include_passed=self._include_passed,
            max_failures=self._max_failures,
        )
        if self._format == "json":
            text = json.dumps(
                summary.model_dump(mode="json"),
                indent=self._indent,
                default=str,
            )
        else:
            text = self._render_console(summary)
        self._write(text)

    def _render_console(self, summary: CompactSuiteSummary) -> str:
        lines = [
            self._bold("ATP Summary"),
            f"Suite: {summary.suite_name}",
            f"Agent: {summary.agent_name}",
            "",
            f"Result: {self._result_text(summary.success)}",
            (
                "Tests: "
                f"{summary.passed_tests} passed, "
                f"{summary.failed_tests} failed, "
                f"{summary.malformed_tests} malformed, "
                f"{summary.errored_tests} error"
            ),
            f"Success rate: {summary.success_rate * 100:.1f}%",
            f"Duration: {self._format_duration(summary.duration_seconds)}",
        ]
        if summary.error:
            lines.append(f"Suite error: {summary.error}")
        if summary.failures:
            lines.extend(["", "Failures:"])
            for test in summary.failures:
                lines.extend(self._render_failure(test))
            if summary.truncated_failures:
                lines.append(f"  ... {summary.truncated_failures} more failures")
        if summary.passed:
            lines.extend(["", "Passed:"])
            for test in summary.passed:
                lines.append(f"  * {test.test_id}")
        return "\n".join(lines)

    def _render_failure(self, test: CompactTestSummary) -> list[str]:
        lines = [f"  x {test.test_id}"]
        lines.append(f"    score: {self._format_score(test.score)}")
        if test.failure is None:
            lines.append(f"    reason: {test.status}")
            return lines
        failure = test.failure
        lines.append(f"    reason: {failure.kind}")
        if failure.evaluator and failure.check:
            lines.append(f"    check: {failure.evaluator}:{failure.check}")
        if failure.path:
            lines.append(f"    path: {failure.path}")
        if failure.expected is not None:
            lines.append(f"    expected: {_format_value(failure.expected)}")
        if failure.received is not None:
            lines.append(f"    received: {_format_value(failure.received)}")
        if failure.message and failure.message != failure.kind:
            lines.append(f"    message: {failure.message}")
        return lines

    def _write(self, text: str) -> None:
        if self._output_file:
            self._output_file.parent.mkdir(parents=True, exist_ok=True)
            self._output_file.write_text(text + "\n")
            return
        output = self._output or sys.stdout
        output.write(text + "\n")

    def _supports_color(self) -> bool:
        output = self._output
        if output is None:
            return False
        if isinstance(output, StringIO):
            return True
        return hasattr(output, "isatty") and output.isatty()

    def _color(self, text: str, color_code: str) -> str:
        if not self._use_colors:
            return text
        return f"\033[{color_code}m{text}\033[0m"

    def _bold(self, text: str) -> str:
        return self._color(text, "1")

    def _result_text(self, success: bool) -> str:
        if success:
            return self._color("PASSED", "32")
        return self._color("FAILED", "31")


def _format_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)
