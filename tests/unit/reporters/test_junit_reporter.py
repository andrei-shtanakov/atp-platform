"""Tests for the JUnit XML reporter."""

import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path

import pytest

from atp.evaluators.base import EvalCheck, EvalResult
from atp.reporters.base import SuiteReport, TestReport
from atp.reporters.junit_reporter import JUnitReporter
from atp.statistics.models import (
    StabilityAssessment,
    StabilityLevel,
    StatisticalResult,
    TestRunStatistics,
)


class TestJUnitReporter:
    """Tests for JUnitReporter."""

    @pytest.fixture
    def output(self) -> StringIO:
        """Create a StringIO for capturing output."""
        return StringIO()

    @pytest.fixture
    def reporter(self, output: StringIO) -> JUnitReporter:
        """Create a JUnit reporter with captured output."""
        return JUnitReporter(output=output)

    @pytest.fixture
    def basic_suite_report(self) -> SuiteReport:
        """Create a basic suite report for testing."""
        return SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=2,
            passed_tests=2,
            failed_tests=0,
            success_rate=1.0,
            duration_seconds=5.5,
            runs_per_test=1,
            tests=[
                TestReport(
                    test_id="test-001",
                    test_name="Test One",
                    success=True,
                    score=85.0,
                    duration_seconds=2.5,
                ),
                TestReport(
                    test_id="test-002",
                    test_name="Test Two",
                    success=True,
                    score=90.0,
                    duration_seconds=3.0,
                ),
            ],
        )

    @pytest.fixture
    def failed_suite_report(self) -> SuiteReport:
        """Create a suite report with failures."""
        return SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=3,
            passed_tests=1,
            failed_tests=2,
            success_rate=0.333,
            duration_seconds=10.0,
            runs_per_test=1,
            tests=[
                TestReport(
                    test_id="test-001",
                    test_name="Test One",
                    success=True,
                    score=85.0,
                    duration_seconds=2.5,
                ),
                TestReport(
                    test_id="test-002",
                    test_name="Test Two",
                    success=False,
                    score=45.0,
                    duration_seconds=3.0,
                    error="Assertion failed: expected output did not match",
                ),
                TestReport(
                    test_id="test-003",
                    test_name="Test Three",
                    success=False,
                    duration_seconds=4.5,
                    error="Timeout after 300 seconds",
                ),
            ],
        )

    def test_reporter_name(self, reporter: JUnitReporter) -> None:
        """Verify reporter name."""
        assert reporter.name == "junit"

    def test_supports_streaming_false(self, reporter: JUnitReporter) -> None:
        """Verify streaming is not supported."""
        assert reporter.supports_streaming is False

    def test_valid_xml_output(
        self, reporter: JUnitReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify output is valid XML."""
        reporter.report(basic_suite_report)
        result = output.getvalue()

        # Should start with XML declaration
        assert result.startswith('<?xml version="1.0" encoding="UTF-8"?>')

        # Should be parseable
        root = ET.fromstring(
            result.replace('<?xml version="1.0" encoding="UTF-8"?>\n', "")
        )
        assert root.tag == "testsuites"

    def test_testsuites_element(
        self, reporter: JUnitReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify testsuites element attributes."""
        reporter.report(basic_suite_report)
        root = self._parse_output(output)

        assert root.get("name") == "ATP Tests"
        assert root.get("tests") == "2"
        assert root.get("failures") == "0"
        assert root.get("errors") == "0"
        assert root.get("time") == "5.500"

    def test_testsuite_element(
        self, reporter: JUnitReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify testsuite element attributes."""
        reporter.report(basic_suite_report)
        root = self._parse_output(output)

        testsuite = root.find("testsuite")
        assert testsuite is not None
        assert testsuite.get("name") == "test-suite"
        assert testsuite.get("tests") == "2"
        assert testsuite.get("failures") == "0"
        assert testsuite.get("errors") == "0"

    def test_properties_section(
        self, reporter: JUnitReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify properties section is included."""
        reporter.report(basic_suite_report)
        root = self._parse_output(output)

        testsuite = root.find("testsuite")
        assert testsuite is not None
        properties = testsuite.find("properties")
        assert properties is not None

        # Check for expected properties
        props = {p.get("name"): p.get("value") for p in properties.findall("property")}
        assert props["agent_name"] == "test-agent"
        assert props["runs_per_test"] == "1"
        assert props["success_rate"] == "100.00%"

    def test_properties_disabled(
        self, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify properties can be disabled."""
        reporter = JUnitReporter(output=output, include_properties=False)
        reporter.report(basic_suite_report)
        root = self._parse_output(output)

        testsuite = root.find("testsuite")
        assert testsuite is not None
        properties = testsuite.find("properties")
        assert properties is None

    def test_testcase_elements(
        self, reporter: JUnitReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify testcase elements."""
        reporter.report(basic_suite_report)
        root = self._parse_output(output)

        testsuite = root.find("testsuite")
        assert testsuite is not None
        testcases = testsuite.findall("testcase")
        assert len(testcases) == 2

        tc1 = testcases[0]
        assert tc1.get("name") == "Test One"
        assert tc1.get("classname") == "test-suite"
        assert tc1.get("time") == "2.500"

        tc2 = testcases[1]
        assert tc2.get("name") == "Test Two"
        assert tc2.get("time") == "3.000"

    def test_failure_element(
        self,
        reporter: JUnitReporter,
        failed_suite_report: SuiteReport,
        output: StringIO,
    ) -> None:
        """Verify failure element for failed tests."""
        reporter.report(failed_suite_report)
        root = self._parse_output(output)

        testsuite = root.find("testsuite")
        assert testsuite is not None
        testcases = testsuite.findall("testcase")

        # Test Two should have a failure element
        tc2 = testcases[1]
        failure = tc2.find("failure")
        assert failure is not None
        assert failure.get("type") == "AssertionError"
        assert "Assertion failed" in (failure.get("message") or "")

        # Failure text should contain score and error
        failure_text = failure.text or ""
        assert "45.0" in failure_text
        assert "Assertion failed" in failure_text

    def test_error_element(
        self,
        reporter: JUnitReporter,
        failed_suite_report: SuiteReport,
        output: StringIO,
    ) -> None:
        """Verify error element for timeout/exception errors."""
        reporter.report(failed_suite_report)
        root = self._parse_output(output)

        testsuite = root.find("testsuite")
        assert testsuite is not None
        testcases = testsuite.findall("testcase")

        # Test Three should have an error element (timeout)
        tc3 = testcases[2]
        error = tc3.find("error")
        assert error is not None
        assert error.get("type") == "TimeoutError"
        assert "Timeout" in (error.get("message") or "")

    def test_errors_count(
        self,
        reporter: JUnitReporter,
        failed_suite_report: SuiteReport,
        output: StringIO,
    ) -> None:
        """Verify errors are counted separately from failures."""
        reporter.report(failed_suite_report)
        root = self._parse_output(output)

        # One error (timeout), one failure (assertion)
        assert root.get("errors") == "1"
        assert root.get("failures") == "2"

        testsuite = root.find("testsuite")
        assert testsuite is not None
        assert testsuite.get("errors") == "1"

    def test_system_out_with_eval_results(
        self, reporter: JUnitReporter, output: StringIO
    ) -> None:
        """Verify system-out contains evaluation details."""
        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            success_rate=1.0,
            tests=[
                TestReport(
                    test_id="test-001",
                    test_name="Test One",
                    success=True,
                    score=85.0,
                    eval_results=[
                        EvalResult(
                            evaluator="artifact",
                            checks=[
                                EvalCheck(
                                    name="file_exists",
                                    passed=True,
                                    score=1.0,
                                    message="File found",
                                )
                            ],
                        )
                    ],
                )
            ],
        )

        reporter.report(report)
        root = self._parse_output(output)

        testsuite = root.find("testsuite")
        assert testsuite is not None
        testcase = testsuite.find("testcase")
        assert testcase is not None

        system_out = testcase.find("system-out")
        assert system_out is not None
        assert "artifact" in (system_out.text or "")
        assert "file_exists" in (system_out.text or "")

    def test_system_out_disabled(self, output: StringIO) -> None:
        """Verify system-out can be disabled."""
        reporter = JUnitReporter(output=output, include_system_out=False)

        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            success_rate=1.0,
            tests=[
                TestReport(
                    test_id="test-001",
                    test_name="Test One",
                    success=True,
                    eval_results=[
                        EvalResult(
                            evaluator="artifact",
                            checks=[EvalCheck(name="check", passed=True, score=1.0)],
                        )
                    ],
                )
            ],
        )

        reporter.report(report)
        root = self._parse_output(output)

        testsuite = root.find("testsuite")
        assert testsuite is not None
        testcase = testsuite.find("testcase")
        assert testcase is not None
        assert testcase.find("system-out") is None

    def test_suite_level_error(self, reporter: JUnitReporter, output: StringIO) -> None:
        """Verify suite-level error in system-err."""
        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            success_rate=0.0,
            error="Failed to initialize adapter",
            tests=[],
        )

        reporter.report(report)
        root = self._parse_output(output)

        testsuite = root.find("testsuite")
        assert testsuite is not None
        system_err = testsuite.find("system-err")
        assert system_err is not None
        assert "Failed to initialize adapter" in (system_err.text or "")

    def test_file_output(self, basic_suite_report: SuiteReport, tmp_path: Path) -> None:
        """Verify file output works correctly."""
        output_file = tmp_path / "junit.xml"
        reporter = JUnitReporter(output_file=output_file)

        reporter.report(basic_suite_report)

        assert output_file.exists()
        content = output_file.read_text()
        assert '<?xml version="1.0" encoding="UTF-8"?>' in content
        assert "<testsuites" in content

    def test_file_output_creates_parent_dirs(
        self, basic_suite_report: SuiteReport, tmp_path: Path
    ) -> None:
        """Verify file output creates parent directories."""
        output_file = tmp_path / "subdir" / "junit.xml"
        reporter = JUnitReporter(output_file=output_file)

        reporter.report(basic_suite_report)

        assert output_file.exists()

    def test_null_duration_handled(
        self, reporter: JUnitReporter, output: StringIO
    ) -> None:
        """Verify null durations are handled correctly."""
        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            success_rate=1.0,
            duration_seconds=None,
            tests=[
                TestReport(
                    test_id="test-001",
                    test_name="Test One",
                    success=True,
                    duration_seconds=None,
                )
            ],
        )

        reporter.report(report)
        root = self._parse_output(output)

        assert root.get("time") == "0"
        testsuite = root.find("testsuite")
        assert testsuite is not None
        testcase = testsuite.find("testcase")
        assert testcase is not None
        assert testcase.get("time") == "0"

    def test_multiple_runs_in_failure(
        self, reporter: JUnitReporter, output: StringIO
    ) -> None:
        """Verify multiple runs info is included in failure text."""
        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            passed_tests=0,
            failed_tests=1,
            success_rate=0.0,
            runs_per_test=5,
            tests=[
                TestReport(
                    test_id="test-001",
                    test_name="Test One",
                    success=False,
                    total_runs=5,
                    successful_runs=2,
                    error="Flaky test",
                )
            ],
        )

        reporter.report(report)
        root = self._parse_output(output)

        testsuite = root.find("testsuite")
        assert testsuite is not None
        testcase = testsuite.find("testcase")
        assert testcase is not None
        failure = testcase.find("failure")
        assert failure is not None
        assert "2/5" in (failure.text or "")

    def test_statistics_in_system_out(
        self, reporter: JUnitReporter, output: StringIO
    ) -> None:
        """Verify statistics are included in system-out."""
        stats = TestRunStatistics(
            test_id="test-001",
            n_runs=5,
            successful_runs=5,
            success_rate=1.0,
            score_stats=StatisticalResult(
                mean=85.0,
                std=3.5,
                min=80.0,
                max=90.0,
                median=85.0,
                confidence_interval=(82.0, 88.0),
                n_runs=5,
                coefficient_of_variation=0.04,
            ),
            overall_stability=StabilityAssessment(
                level=StabilityLevel.STABLE,
                cv=0.04,
                message="Stable results",
            ),
        )

        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            success_rate=1.0,
            tests=[
                TestReport(
                    test_id="test-001",
                    test_name="Test One",
                    success=True,
                    score=85.0,
                    statistics=stats,
                    eval_results=[
                        EvalResult(
                            evaluator="test",
                            checks=[EvalCheck(name="check", passed=True, score=1.0)],
                        )
                    ],
                )
            ],
        )

        reporter.report(report)
        root = self._parse_output(output)

        testsuite = root.find("testsuite")
        assert testsuite is not None
        testcase = testsuite.find("testcase")
        assert testcase is not None
        system_out = testcase.find("system-out")
        assert system_out is not None
        text = system_out.text or ""
        assert "Mean Score: 85.0" in text
        assert "Std Dev: 3.50" in text

    def test_error_type_classification(self) -> None:
        """Verify error type classification."""
        reporter = JUnitReporter()

        # Timeout errors
        assert reporter._get_error_type("Timeout after 300s") == "TimeoutError"
        assert reporter._get_error_type("Request timed out") == "TimeoutError"

        # Connection errors
        assert reporter._get_error_type("Connection refused") == "ConnectionError"
        assert reporter._get_error_type("Failed to connect") == "ConnectionError"

        # Not found errors
        assert reporter._get_error_type("File not found") == "NotFoundError"
        assert reporter._get_error_type("Resource not found") == "NotFoundError"

        # Default
        assert reporter._get_error_type("Unknown error") == "RuntimeError"

    def test_is_error_vs_failure(self) -> None:
        """Verify error vs failure classification."""
        reporter = JUnitReporter()

        # These should be classified as errors
        assert reporter._is_error("Timeout after 300s") is True
        assert reporter._is_error("Connection refused to endpoint") is True
        assert reporter._is_error("Exception occurred") is True
        assert reporter._is_error("Failed to initialize adapter") is True
        assert reporter._is_error("Resource not found") is True
        assert reporter._is_error("Request timed out") is True

        # These should be classified as failures (no error keywords)
        assert reporter._is_error("Score below threshold") is False
        assert reporter._is_error("Assertion did not match") is False
        assert reporter._is_error("Check returned false") is False
        assert reporter._is_error("Assertion failed: expected file") is False

    def test_failed_eval_checks_in_failure(
        self, reporter: JUnitReporter, output: StringIO
    ) -> None:
        """Verify failed eval checks appear in failure text."""
        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            passed_tests=0,
            failed_tests=1,
            success_rate=0.0,
            tests=[
                TestReport(
                    test_id="test-001",
                    test_name="Test One",
                    success=False,
                    score=45.0,
                    eval_results=[
                        EvalResult(
                            evaluator="artifact",
                            checks=[
                                EvalCheck(
                                    name="file_exists",
                                    passed=False,
                                    score=0.0,
                                    message="File not found: output.txt",
                                ),
                                EvalCheck(
                                    name="content_check",
                                    passed=False,
                                    score=0.0,
                                    message="Missing required section",
                                ),
                            ],
                        )
                    ],
                )
            ],
        )

        reporter.report(report)
        root = self._parse_output(output)

        testsuite = root.find("testsuite")
        assert testsuite is not None
        testcase = testsuite.find("testcase")
        assert testcase is not None
        failure = testcase.find("failure")
        assert failure is not None

        text = failure.text or ""
        assert "artifact" in text
        assert "file_exists" in text
        assert "File not found" in text
        assert "content_check" in text
        assert "Missing required section" in text

    def _parse_output(self, output: StringIO) -> ET.Element:
        """Parse XML output, removing the declaration."""
        content = output.getvalue()
        # Remove XML declaration for parsing
        xml_content = content.replace('<?xml version="1.0" encoding="UTF-8"?>\n', "")
        return ET.fromstring(xml_content)
