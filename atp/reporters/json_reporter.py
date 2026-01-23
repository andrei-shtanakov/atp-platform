"""JSON reporter for machine-readable output."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from atp.reporters.base import Reporter, SuiteReport, TestReport


class JSONReporter(Reporter):
    """Reporter that outputs results in JSON format.

    Produces a stable, documented JSON format suitable for CI/CD
    integration and automated processing.
    """

    # JSON format version for compatibility tracking
    FORMAT_VERSION = "1.0"

    def __init__(
        self,
        output_file: Path | str | None = None,
        output: TextIO | None = None,
        indent: int | None = 2,
        include_details: bool = True,
    ) -> None:
        """Initialize the JSON reporter.

        Args:
            output_file: Path to write JSON file (takes precedence over output).
            output: Output stream (defaults to sys.stdout if no file specified).
            indent: JSON indentation level (None for compact output).
            include_details: Whether to include detailed evaluation results.
        """
        self._output_file = Path(output_file) if output_file else None
        self._output = output
        self._indent = indent
        self._include_details = include_details

    @property
    def name(self) -> str:
        """Return the reporter name."""
        return "json"

    def report(self, report: SuiteReport) -> None:
        """Generate and output the JSON report.

        Args:
            report: Suite report data to output.
        """
        json_data = self._build_json(report)
        json_str = json.dumps(json_data, indent=self._indent, default=str)

        if self._output_file:
            self._output_file.parent.mkdir(parents=True, exist_ok=True)
            self._output_file.write_text(json_str + "\n")
        else:
            output = self._output or sys.stdout
            output.write(json_str + "\n")

    def _build_json(self, report: SuiteReport) -> dict[str, Any]:
        """Build the JSON structure from the report.

        Args:
            report: Suite report data.

        Returns:
            Dictionary representing the JSON structure.
        """
        return {
            "version": self.FORMAT_VERSION,
            "generated_at": datetime.now().isoformat(),
            "summary": self._build_summary(report),
            "tests": [self._build_test(test) for test in report.tests],
        }

    def _build_summary(self, report: SuiteReport) -> dict[str, Any]:
        """Build the summary section.

        Args:
            report: Suite report data.

        Returns:
            Summary dictionary.
        """
        return {
            "suite_name": report.suite_name,
            "agent_name": report.agent_name,
            "total_tests": report.total_tests,
            "passed_tests": report.passed_tests,
            "failed_tests": report.failed_tests,
            "success_rate": round(report.success_rate, 4),
            "duration_seconds": (
                round(report.duration_seconds, 3)
                if report.duration_seconds is not None
                else None
            ),
            "runs_per_test": report.runs_per_test,
            "success": report.passed_tests == report.total_tests,
            "error": report.error,
        }

    def _build_test(self, test: TestReport) -> dict[str, Any]:
        """Build a test result entry.

        Args:
            test: Test report data.

        Returns:
            Test dictionary.
        """
        result: dict[str, Any] = {
            "test_id": test.test_id,
            "test_name": test.test_name,
            "success": test.success,
            "score": round(test.score, 2) if test.score is not None else None,
            "duration_seconds": (
                round(test.duration_seconds, 3)
                if test.duration_seconds is not None
                else None
            ),
            "runs": {
                "total": test.total_runs,
                "successful": test.successful_runs,
            },
            "error": test.error,
        }

        # Add score breakdown if available
        if test.scored_result:
            result["score_breakdown"] = test.scored_result.breakdown.to_dict()

        # Add statistics if available
        if test.statistics:
            result["statistics"] = test.statistics.to_dict()

        # Add evaluation details if enabled
        if self._include_details and test.eval_results:
            result["evaluations"] = self._build_evaluations(test)

        return result

    def _build_evaluations(self, test: TestReport) -> list[dict[str, Any]]:
        """Build evaluation results for a test.

        Args:
            test: Test report data.

        Returns:
            List of evaluation dictionaries.
        """
        evaluations = []

        for eval_result in test.eval_results:
            eval_dict: dict[str, Any] = {
                "evaluator": eval_result.evaluator,
                "passed": eval_result.passed,
                "score": round(eval_result.score, 4),
                "checks": [],
            }

            for check in eval_result.checks:
                check_dict: dict[str, Any] = {
                    "name": check.name,
                    "passed": check.passed,
                    "score": round(check.score, 4),
                    "message": check.message,
                }

                if check.details:
                    check_dict["details"] = check.details

                eval_dict["checks"].append(check_dict)

            evaluations.append(eval_dict)

        return evaluations
