"""JUnit XML reporter for CI/CD integration.

Generates JUnit XML format compatible with most CI systems including
GitHub Actions, GitLab CI, Jenkins, CircleCI, and Azure DevOps.
"""

import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import TextIO

from atp.reporters.base import Reporter, SuiteReport, TestReport


class JUnitReporter(Reporter):
    """Reporter that outputs results in JUnit XML format.

    Produces JUnit XML compatible with CI systems for test result visualization
    and reporting.

    JUnit XML Schema Reference:
    https://llg.cubic.org/docs/junit/

    Example output:
    ```xml
    <?xml version="1.0" encoding="UTF-8"?>
    <testsuites name="ATP Tests" tests="3" failures="1" errors="0" time="5.5">
        <testsuite name="test-suite" tests="3" failures="1" errors="0" time="5.5">
            <testcase name="Test One" classname="test-suite" time="2.5"/>
            <testcase name="Test Two" classname="test-suite" time="1.5">
                <failure message="Test failed" type="AssertionError">
                    Score: 45.0/100
                    Error: Timeout after 300s
                </failure>
            </testcase>
            <testcase name="Test Three" classname="test-suite" time="1.5"/>
        </testsuite>
    </testsuites>
    ```
    """

    def __init__(
        self,
        output_file: Path | str | None = None,
        output: TextIO | None = None,
        include_properties: bool = True,
        include_system_out: bool = True,
    ) -> None:
        """Initialize the JUnit reporter.

        Args:
            output_file: Path to write XML file (takes precedence over output).
            output: Output stream (defaults to sys.stdout if no file specified).
            include_properties: Whether to include test properties section.
            include_system_out: Whether to include evaluation details in system-out.
        """
        self._output_file = Path(output_file) if output_file else None
        self._output = output
        self._include_properties = include_properties
        self._include_system_out = include_system_out

    @property
    def name(self) -> str:
        """Return the reporter name."""
        return "junit"

    def report(self, report: SuiteReport) -> None:
        """Generate and output the JUnit XML report.

        Args:
            report: Suite report data to output.
        """
        root = self._build_xml(report)
        xml_str = self._format_xml(root)

        if self._output_file:
            self._output_file.parent.mkdir(parents=True, exist_ok=True)
            self._output_file.write_text(xml_str)
        else:
            output = self._output or sys.stdout
            output.write(xml_str)

    def _build_xml(self, report: SuiteReport) -> ET.Element:
        """Build the XML structure from the report.

        Args:
            report: Suite report data.

        Returns:
            Root XML element.
        """
        # Root testsuites element
        testsuites = ET.Element("testsuites")
        testsuites.set("name", "ATP Tests")
        testsuites.set("tests", str(report.total_tests))
        testsuites.set("failures", str(report.failed_tests))
        testsuites.set("errors", str(self._count_errors(report)))
        testsuites.set(
            "time",
            f"{report.duration_seconds:.3f}" if report.duration_seconds else "0",
        )
        testsuites.set("timestamp", datetime.now().isoformat())

        # Single testsuite element for the suite
        testsuite = ET.SubElement(testsuites, "testsuite")
        testsuite.set("name", report.suite_name)
        testsuite.set("tests", str(report.total_tests))
        testsuite.set("failures", str(report.failed_tests))
        testsuite.set("errors", str(self._count_errors(report)))
        testsuite.set(
            "time",
            f"{report.duration_seconds:.3f}" if report.duration_seconds else "0",
        )
        testsuite.set("timestamp", datetime.now().isoformat())

        # Add properties section
        if self._include_properties:
            props = ET.SubElement(testsuite, "properties")
            self._add_property(props, "agent_name", report.agent_name)
            self._add_property(props, "runs_per_test", str(report.runs_per_test))
            self._add_property(
                props, "success_rate", f"{report.success_rate * 100:.2f}%"
            )

        # Add test cases
        for test in report.tests:
            self._add_testcase(testsuite, test, report.suite_name)

        # Add suite-level error if present
        if report.error:
            system_err = ET.SubElement(testsuite, "system-err")
            system_err.text = report.error

        return testsuites

    def _add_property(self, parent: ET.Element, name: str, value: str) -> None:
        """Add a property element.

        Args:
            parent: Parent element.
            name: Property name.
            value: Property value.
        """
        prop = ET.SubElement(parent, "property")
        prop.set("name", name)
        prop.set("value", value)

    def _add_testcase(
        self, parent: ET.Element, test: TestReport, suite_name: str
    ) -> None:
        """Add a testcase element.

        Args:
            parent: Parent element.
            test: Test report data.
            suite_name: Name of the test suite (used as classname).
        """
        testcase = ET.SubElement(parent, "testcase")
        testcase.set("name", test.test_name)
        testcase.set("classname", suite_name)
        testcase.set(
            "time",
            f"{test.duration_seconds:.3f}" if test.duration_seconds else "0",
        )

        # Add failure or error element if test failed
        if not test.success:
            if test.error and self._is_error(test.error):
                # Errors are unexpected issues (exceptions, timeouts, etc.)
                error_elem = ET.SubElement(testcase, "error")
                error_elem.set("message", test.error[:200] if test.error else "Error")
                error_elem.set("type", self._get_error_type(test.error))
                error_elem.text = self._build_failure_text(test)
            else:
                # Failures are assertion failures
                failure = ET.SubElement(testcase, "failure")
                failure.set(
                    "message", test.error[:200] if test.error else "Test failed"
                )
                failure.set("type", "AssertionError")
                failure.text = self._build_failure_text(test)

        # Add evaluation details in system-out
        if self._include_system_out and test.eval_results:
            system_out = ET.SubElement(testcase, "system-out")
            system_out.text = self._build_system_out(test)

    def _build_failure_text(self, test: TestReport) -> str:
        """Build detailed failure text.

        Args:
            test: Test report data.

        Returns:
            Formatted failure details.
        """
        lines = []

        if test.score is not None:
            lines.append(f"Score: {test.score:.1f}/100")

        if test.total_runs > 1:
            lines.append(f"Runs: {test.successful_runs}/{test.total_runs} successful")

        if test.error:
            lines.append(f"Error: {test.error}")

        # Add failed evaluation checks
        for eval_result in test.eval_results:
            if not eval_result.passed:
                lines.append(f"\nEvaluator: {eval_result.evaluator}")
                for check in eval_result.checks:
                    if not check.passed:
                        lines.append(f"  - {check.name}: {check.message or 'Failed'}")

        return "\n".join(lines)

    def _build_system_out(self, test: TestReport) -> str:
        """Build system-out content with evaluation details.

        Args:
            test: Test report data.

        Returns:
            Formatted evaluation details.
        """
        lines = []

        lines.append(f"Test ID: {test.test_id}")
        if test.score is not None:
            lines.append(f"Score: {test.score:.1f}/100")
        if test.statistics and test.statistics.score_stats:
            lines.append(f"Mean Score: {test.statistics.score_stats.mean:.1f}")
            lines.append(f"Std Dev: {test.statistics.score_stats.std:.2f}")

        lines.append("\nEvaluation Results:")
        for eval_result in test.eval_results:
            lines.append(f"\n  {eval_result.evaluator}:")
            for check in eval_result.checks:
                status = "✓" if check.passed else "✗"
                msg = check.message or ""
                lines.append(f"    {status} {check.name}: {check.score:.2f} - {msg}")

        return "\n".join(lines)

    def _count_errors(self, report: SuiteReport) -> int:
        """Count tests that had errors (not assertion failures).

        Args:
            report: Suite report data.

        Returns:
            Number of tests with errors.
        """
        count = 0
        for test in report.tests:
            if not test.success and test.error and self._is_error(test.error):
                count += 1
        return count

    def _is_error(self, error_message: str) -> bool:
        """Determine if an error message represents an error vs failure.

        Errors are unexpected issues like timeouts, exceptions, or configuration
        problems. Failures are assertion failures where the test ran but didn't
        pass validation.

        Args:
            error_message: Error message string.

        Returns:
            True if this is an error, False if it's a failure.
        """
        # Keywords that indicate a system/runtime error rather than test failure
        error_keywords = [
            "timeout",
            "exception",
            "failed to connect",
            "failed to initialize",
            "unable to",
            "connection refused",
            "connection error",
            "connectionerror",
            "not found",
            "timed out",
        ]
        lower_message = error_message.lower()
        return any(keyword in lower_message for keyword in error_keywords)

    def _get_error_type(self, error_message: str) -> str:
        """Determine the error type from the error message.

        Args:
            error_message: Error message string.

        Returns:
            Error type classification.
        """
        lower_message = error_message.lower()
        if "timeout" in lower_message or "timed out" in lower_message:
            return "TimeoutError"
        elif (
            "connection" in lower_message
            or "refused" in lower_message
            or "connect" in lower_message
        ):
            return "ConnectionError"
        elif "not found" in lower_message:
            return "NotFoundError"
        else:
            return "RuntimeError"

    def _format_xml(self, root: ET.Element) -> str:
        """Format XML with proper declaration and indentation.

        Args:
            root: Root XML element.

        Returns:
            Formatted XML string.
        """
        self._indent_xml(root)
        xml_str = ET.tostring(root, encoding="unicode")
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_str}\n'

    def _indent_xml(self, elem: ET.Element, level: int = 0) -> None:
        """Add indentation to XML elements for readability.

        Args:
            elem: Element to indent.
            level: Current indentation level.
        """
        indent = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent
