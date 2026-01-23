"""Console reporter for terminal output."""

import sys
from io import StringIO
from typing import TextIO

from atp.reporters.base import Reporter, SuiteReport, TestReport


class ConsoleReporter(Reporter):
    """Reporter that outputs results to the terminal.

    Provides human-readable output with optional color support and
    verbosity levels.
    """

    def __init__(
        self,
        output: TextIO | None = None,
        use_colors: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the console reporter.

        Args:
            output: Output stream (defaults to sys.stdout).
            use_colors: Whether to use ANSI color codes.
            verbose: Whether to show detailed output.
        """
        self._output = output or sys.stdout
        self._use_colors = use_colors and self._supports_color()
        self._verbose = verbose

    @property
    def name(self) -> str:
        """Return the reporter name."""
        return "console"

    def _supports_color(self) -> bool:
        """Check if the output stream supports ANSI colors."""
        if self._output is None:
            return False
        if isinstance(self._output, StringIO):
            return True
        if not hasattr(self._output, "isatty"):
            return False
        return self._output.isatty()

    def _color(self, text: str, color_code: str) -> str:
        """Apply ANSI color to text if colors are enabled.

        Args:
            text: Text to colorize.
            color_code: ANSI color code.

        Returns:
            Colorized text or original text if colors disabled.
        """
        if not self._use_colors:
            return text
        return f"\033[{color_code}m{text}\033[0m"

    def _green(self, text: str) -> str:
        """Apply green color."""
        return self._color(text, "32")

    def _red(self, text: str) -> str:
        """Apply red color."""
        return self._color(text, "31")

    def _yellow(self, text: str) -> str:
        """Apply yellow color."""
        return self._color(text, "33")

    def _bold(self, text: str) -> str:
        """Apply bold style."""
        return self._color(text, "1")

    def _dim(self, text: str) -> str:
        """Apply dim style."""
        return self._color(text, "2")

    def _write(self, text: str = "") -> None:
        """Write text to output stream.

        Args:
            text: Text to write.
        """
        self._output.write(text + "\n")

    def report(self, report: SuiteReport) -> None:
        """Generate and output the console report.

        Args:
            report: Suite report data to output.
        """
        self._write_header(report)
        self._write()
        self._write_tests(report)
        self._write()
        self._write_summary(report)

    def _write_header(self, report: SuiteReport) -> None:
        """Write the report header.

        Args:
            report: Suite report data.
        """
        self._write(self._bold("ATP Test Results"))
        self._write("=" * 50)
        self._write()
        self._write(f"Suite: {report.suite_name}")
        self._write(f"Agent: {report.agent_name}")

        if report.runs_per_test > 1:
            self._write(f"Runs per test: {report.runs_per_test}")

    def _write_tests(self, report: SuiteReport) -> None:
        """Write individual test results.

        Args:
            report: Suite report data.
        """
        self._write("Tests:")

        for test in report.tests:
            self._write_test(test)

    def _write_test(self, test: TestReport) -> None:
        """Write a single test result.

        Args:
            test: Test report data.
        """
        # Status indicator
        if test.success:
            indicator = self._green("*")
        else:
            indicator = self._red("x")

        # Build the main line
        score_str = self._format_score(test.score)
        duration_str = f"[{self._format_duration(test.duration_seconds)}]"

        # Add standard deviation if statistics available
        std_str = ""
        if test.statistics and test.statistics.score_stats:
            std = test.statistics.score_stats.std
            std_str = f" (s={std:.1f})"

        name_part = f"{test.test_name:<30}"
        line = f"  {indicator} {name_part} {score_str}{std_str}  {duration_str}"
        self._write(line)

        # In verbose mode or for failed tests, show details
        if not test.success or self._verbose:
            self._write_test_details(test)

    def _write_test_details(self, test: TestReport) -> None:
        """Write detailed information for a test.

        Args:
            test: Test report data.
        """
        # Show error if present
        if test.error:
            self._write(f"      {self._red('Error:')} {test.error}")

        # Show failed checks
        for eval_result in test.eval_results:
            for check in eval_result.checks:
                if not check.passed:
                    msg = check.message or "Check failed"
                    check_line = f"      - {eval_result.evaluator}:{check.name}: {msg}"
                    self._write(self._red(check_line))

        # In verbose mode, show all checks
        if self._verbose:
            for eval_result in test.eval_results:
                for check in eval_result.checks:
                    if check.passed:
                        evaluator = eval_result.evaluator
                        name = check.name
                        score = check.score
                        check_line = f"      + {evaluator}:{name}: {score:.2f}"
                        self._write(self._dim(check_line))

        # Show run statistics if multiple runs
        if test.total_runs > 1 and self._verbose:
            self._write(
                f"      Runs: {test.successful_runs}/{test.total_runs} successful"
            )

    def _write_summary(self, report: SuiteReport) -> None:
        """Write the summary section.

        Args:
            report: Suite report data.
        """
        # Summary line
        passed = report.passed_tests
        failed = report.failed_tests
        rate = report.success_rate * 100

        if failed == 0:
            summary = self._green(f"Summary: {passed} passed")
        elif passed == 0:
            summary = self._red(f"Summary: {failed} failed")
        else:
            passed_part = self._green(f"{passed} passed")
            failed_part = self._red(f"{failed} failed")
            summary = f"Summary: {passed_part}, {failed_part}"

        summary += f" ({rate:.1f}%)"
        self._write(summary)

        # Total time
        duration = self._format_duration(report.duration_seconds)
        self._write(f"Total time: {duration}")

        # Show suite error if present
        if report.error:
            self._write()
            self._write(self._red(f"Suite error: {report.error}"))
