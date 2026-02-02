"""Tests for the console reporter."""

from io import StringIO

import pytest

from atp.evaluators.base import EvalCheck, EvalResult
from atp.reporters.base import SuiteReport, TestReport
from atp.reporters.console import ConsoleReporter
from atp.statistics.models import (
    StabilityAssessment,
    StabilityLevel,
    StatisticalResult,
    TestRunStatistics,
)


class TestConsoleReporter:
    """Tests for ConsoleReporter."""

    @pytest.fixture
    def output(self) -> StringIO:
        """Create a StringIO for capturing output."""
        return StringIO()

    @pytest.fixture
    def reporter(self, output: StringIO) -> ConsoleReporter:
        """Create a console reporter with captured output."""
        return ConsoleReporter(output=output, use_colors=False, verbose=False)

    @pytest.fixture
    def verbose_reporter(self, output: StringIO) -> ConsoleReporter:
        """Create a verbose console reporter."""
        return ConsoleReporter(output=output, use_colors=False, verbose=True)

    @pytest.fixture
    def colored_reporter(self, output: StringIO) -> ConsoleReporter:
        """Create a colored console reporter."""
        return ConsoleReporter(output=output, use_colors=True, verbose=False)

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
    def mixed_suite_report(self) -> SuiteReport:
        """Create a suite report with mixed results."""
        return SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=3,
            passed_tests=2,
            failed_tests=1,
            success_rate=0.667,
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
                    score=40.0,
                    duration_seconds=3.0,
                    eval_results=[
                        EvalResult(
                            evaluator="artifact",
                            checks=[
                                EvalCheck(
                                    name="file_exists",
                                    passed=False,
                                    score=0.0,
                                    message="File not found: output.txt",
                                )
                            ],
                        )
                    ],
                ),
                TestReport(
                    test_id="test-003",
                    test_name="Test Three",
                    success=True,
                    score=95.0,
                    duration_seconds=4.5,
                ),
            ],
        )

    def test_reporter_name(self, reporter: ConsoleReporter) -> None:
        """Verify reporter name."""
        assert reporter.name == "console"

    def test_supports_streaming_false(self, reporter: ConsoleReporter) -> None:
        """Verify streaming is not supported."""
        assert reporter.supports_streaming is False

    def test_basic_report_output(
        self,
        reporter: ConsoleReporter,
        basic_suite_report: SuiteReport,
        output: StringIO,
    ) -> None:
        """Verify basic report output."""
        reporter.report(basic_suite_report)
        result = output.getvalue()

        assert "ATP Test Results" in result
        assert "Suite: test-suite" in result
        assert "Agent: test-agent" in result
        assert "Test One" in result
        assert "Test Two" in result
        assert "85.0/100" in result
        assert "90.0/100" in result
        assert "Summary:" in result
        assert "2 passed" in result
        assert "Total time:" in result

    def test_failed_test_shows_details(
        self,
        reporter: ConsoleReporter,
        mixed_suite_report: SuiteReport,
        output: StringIO,
    ) -> None:
        """Verify failed tests show failure details."""
        reporter.report(mixed_suite_report)
        result = output.getvalue()

        assert "Test Two" in result
        assert "artifact:file_exists" in result
        assert "File not found: output.txt" in result

    def test_summary_shows_mixed_results(
        self,
        reporter: ConsoleReporter,
        mixed_suite_report: SuiteReport,
        output: StringIO,
    ) -> None:
        """Verify summary shows mixed pass/fail."""
        reporter.report(mixed_suite_report)
        result = output.getvalue()

        assert "2 passed" in result
        assert "1 failed" in result
        assert "66.7%" in result

    def test_verbose_shows_all_checks(
        self,
        verbose_reporter: ConsoleReporter,
        basic_suite_report: SuiteReport,
        output: StringIO,
    ) -> None:
        """Verify verbose mode shows passed checks too."""
        # Add eval results to the basic report
        basic_suite_report.tests[0].eval_results = [
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
        ]

        verbose_reporter.report(basic_suite_report)
        result = output.getvalue()

        assert "artifact:file_exists" in result

    def test_multiple_runs_header(
        self, reporter: ConsoleReporter, output: StringIO
    ) -> None:
        """Verify multiple runs are shown in header."""
        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            success_rate=1.0,
            runs_per_test=5,
            tests=[
                TestReport(
                    test_id="test-001",
                    test_name="Test One",
                    success=True,
                    total_runs=5,
                    successful_runs=5,
                )
            ],
        )

        reporter.report(report)
        result = output.getvalue()

        assert "Runs per test: 5" in result

    def test_statistics_shows_std(
        self, reporter: ConsoleReporter, output: StringIO
    ) -> None:
        """Verify statistics standard deviation is shown."""
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
                )
            ],
        )

        reporter.report(report)
        result = output.getvalue()

        assert "(s=3.5)" in result

    def test_colored_output_pass(
        self, colored_reporter: ConsoleReporter, output: StringIO
    ) -> None:
        """Verify colored output for passing tests."""
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
                )
            ],
        )

        colored_reporter.report(report)
        result = output.getvalue()

        # Check for ANSI color codes (green = 32)
        assert "\033[32m" in result

    def test_colored_output_fail(
        self, colored_reporter: ConsoleReporter, output: StringIO
    ) -> None:
        """Verify colored output for failing tests."""
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
                )
            ],
        )

        colored_reporter.report(report)
        result = output.getvalue()

        # Check for ANSI color codes (red = 31)
        assert "\033[31m" in result

    def test_suite_error_shown(
        self, reporter: ConsoleReporter, output: StringIO
    ) -> None:
        """Verify suite-level errors are shown."""
        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            success_rate=0.0,
            error="Connection failed to agent",
        )

        reporter.report(report)
        result = output.getvalue()

        assert "Suite error: Connection failed to agent" in result

    def test_test_error_shown(
        self, reporter: ConsoleReporter, output: StringIO
    ) -> None:
        """Verify test-level errors are shown."""
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
                    error="Timeout after 300s",
                )
            ],
        )

        reporter.report(report)
        result = output.getvalue()

        assert "Error:" in result
        assert "Timeout after 300s" in result

    def test_all_failed_summary(
        self, reporter: ConsoleReporter, output: StringIO
    ) -> None:
        """Verify summary when all tests fail."""
        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=2,
            passed_tests=0,
            failed_tests=2,
            success_rate=0.0,
            tests=[
                TestReport(test_id="test-001", test_name="Test One", success=False),
                TestReport(test_id="test-002", test_name="Test Two", success=False),
            ],
        )

        reporter.report(report)
        result = output.getvalue()

        assert "2 failed" in result
        assert "0.0%" in result

    def test_supports_color_with_none_output(self) -> None:
        """Test that color is not supported when output is None."""
        reporter = ConsoleReporter(output=None, use_colors=True)
        assert reporter._supports_color() is False

    def test_supports_color_without_isatty(self) -> None:
        """Test that color is not supported when output lacks isatty."""

        class NoIsattyOutput:
            def write(self, s: str) -> int:
                return len(s)

        output = NoIsattyOutput()
        reporter = ConsoleReporter(output=output, use_colors=True)  # type: ignore
        assert reporter._supports_color() is False
