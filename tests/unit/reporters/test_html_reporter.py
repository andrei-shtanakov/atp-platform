"""Tests for the HTML reporter."""

from io import StringIO
from pathlib import Path

import pytest

from atp.evaluators.base import EvalCheck, EvalResult
from atp.reporters.base import SuiteReport, TestReport
from atp.reporters.html_reporter import HTMLReporter
from atp.scoring.models import ComponentScore, ScoreBreakdown, ScoredTestResult
from atp.statistics.models import (
    StabilityAssessment,
    StabilityLevel,
    StatisticalResult,
    TestRunStatistics,
)


class TestHTMLReporter:
    """Tests for HTMLReporter."""

    @pytest.fixture
    def output(self) -> StringIO:
        """Create a StringIO for capturing output."""
        return StringIO()

    @pytest.fixture
    def reporter(self, output: StringIO) -> HTMLReporter:
        """Create an HTML reporter with captured output."""
        return HTMLReporter(output=output)

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
    def suite_with_failures(self) -> SuiteReport:
        """Create a suite report with failures."""
        return SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=3,
            passed_tests=1,
            failed_tests=2,
            success_rate=0.333,
            duration_seconds=10.0,
            tests=[
                TestReport(
                    test_id="test-001",
                    test_name="Passing Test",
                    success=True,
                    score=95.0,
                    duration_seconds=2.0,
                ),
                TestReport(
                    test_id="test-002",
                    test_name="Failing Test",
                    success=False,
                    score=30.0,
                    duration_seconds=5.0,
                    error="Assertion failed: expected output not found",
                ),
                TestReport(
                    test_id="test-003",
                    test_name="Another Failure",
                    success=False,
                    score=45.0,
                    duration_seconds=3.0,
                    eval_results=[
                        EvalResult(
                            evaluator="artifact",
                            checks=[
                                EvalCheck(
                                    name="file_exists",
                                    passed=False,
                                    score=0.0,
                                    message="Required file not found",
                                ),
                                EvalCheck(
                                    name="content_check",
                                    passed=True,
                                    score=1.0,
                                    message="Content matches",
                                ),
                            ],
                        )
                    ],
                ),
            ],
        )

    def test_reporter_name(self, reporter: HTMLReporter) -> None:
        """Verify reporter name."""
        assert reporter.name == "html"

    def test_supports_streaming_false(self, reporter: HTMLReporter) -> None:
        """Verify streaming is not supported."""
        assert reporter.supports_streaming is False

    def test_generates_valid_html(
        self, reporter: HTMLReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify generated output is valid HTML."""
        reporter.report(basic_suite_report)
        html_content = output.getvalue()

        assert html_content.startswith("<!DOCTYPE html>")
        assert "<html" in html_content
        assert "</html>" in html_content
        assert "<head>" in html_content
        assert "</head>" in html_content
        assert "<body>" in html_content
        assert "</body>" in html_content

    def test_title_included(
        self, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify custom title is included."""
        reporter = HTMLReporter(output=output, title="Custom Report Title")
        reporter.report(basic_suite_report)
        html_content = output.getvalue()

        assert "<title>Custom Report Title</title>" in html_content
        assert "Custom Report Title" in html_content

    def test_suite_info_included(
        self, reporter: HTMLReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify suite information is included."""
        reporter.report(basic_suite_report)
        html_content = output.getvalue()

        assert "test-suite" in html_content
        assert "test-agent" in html_content

    def test_summary_section_included(
        self, reporter: HTMLReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify summary section with metrics is included."""
        reporter.report(basic_suite_report)
        html_content = output.getvalue()

        # Should include total tests, passed, failed
        assert "Total Tests" in html_content
        assert "Passed" in html_content
        assert "Failed" in html_content
        assert "Success Rate" in html_content
        assert "Duration" in html_content

    def test_test_details_accordion_included(
        self, reporter: HTMLReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify test details accordion is included."""
        reporter.report(basic_suite_report)
        html_content = output.getvalue()

        # Should include test items with expandable details
        assert "test-001" in html_content
        assert "test-002" in html_content
        assert "Test One" in html_content
        assert "Test Two" in html_content
        assert "test-item" in html_content
        assert "toggleTest" in html_content

    def test_scores_displayed(
        self, reporter: HTMLReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify test scores are displayed."""
        reporter.report(basic_suite_report)
        html_content = output.getvalue()

        assert "85.0" in html_content or "85" in html_content
        assert "90.0" in html_content or "90" in html_content

    def test_chartjs_included(
        self, reporter: HTMLReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify Chart.js is included."""
        reporter.report(basic_suite_report)
        html_content = output.getvalue()

        assert "chart.js" in html_content.lower() or "Chart(" in html_content

    def test_embedded_css(
        self, reporter: HTMLReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify CSS is embedded in the output."""
        reporter.report(basic_suite_report)
        html_content = output.getvalue()

        assert "<style>" in html_content
        assert "</style>" in html_content
        # Check for some CSS variables/rules
        assert "--color-success" in html_content
        assert "--color-error" in html_content

    def test_single_file_output(
        self, reporter: HTMLReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify all content is in a single file (no external dependencies)."""
        reporter.report(basic_suite_report)
        html_content = output.getvalue()

        # Should have embedded styles
        assert "<style>" in html_content
        # Should have inline scripts
        assert "<script>" in html_content
        # Should NOT have external stylesheet links (except CDN)
        assert 'rel="stylesheet"' not in html_content

    def test_failed_checks_highlighted(
        self, reporter: HTMLReporter, suite_with_failures: SuiteReport, output: StringIO
    ) -> None:
        """Verify failed checks are highlighted."""
        reporter.report(suite_with_failures)
        html_content = output.getvalue()

        # Should include failed check messages
        assert "Required file not found" in html_content
        # Should have error styling
        assert "check-failed" in html_content or "error" in html_content.lower()

    def test_error_messages_displayed(
        self, reporter: HTMLReporter, suite_with_failures: SuiteReport, output: StringIO
    ) -> None:
        """Verify error messages are displayed."""
        reporter.report(suite_with_failures)
        html_content = output.getvalue()

        assert "Assertion failed: expected output not found" in html_content

    def test_file_output(self, basic_suite_report: SuiteReport, tmp_path: Path) -> None:
        """Verify file output works correctly."""
        output_file = tmp_path / "report.html"
        reporter = HTMLReporter(output_file=output_file)

        reporter.report(basic_suite_report)

        assert output_file.exists()
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content
        assert "test-suite" in content

    def test_file_output_creates_parent_dirs(
        self, basic_suite_report: SuiteReport, tmp_path: Path
    ) -> None:
        """Verify file output creates parent directories."""
        output_file = tmp_path / "subdir" / "report.html"
        reporter = HTMLReporter(output_file=output_file)

        reporter.report(basic_suite_report)

        assert output_file.exists()

    def test_score_breakdown_displayed(self, output: StringIO) -> None:
        """Verify score breakdown is displayed."""
        component = ComponentScore(
            name="quality",
            raw_value=0.85,
            normalized_value=0.85,
            weight=0.4,
            weighted_value=0.34,
        )
        breakdown = ScoreBreakdown(
            quality=component,
            completeness=component,
            efficiency=component,
            cost=component,
        )
        scored = ScoredTestResult(
            test_id="test-001",
            score=85.0,
            breakdown=breakdown,
            passed=True,
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
                    scored_result=scored,
                )
            ],
        )

        reporter = HTMLReporter(output=output)
        reporter.report(report)
        html_content = output.getvalue()

        # Should include score breakdown components
        assert "Score Breakdown" in html_content
        assert "Quality" in html_content or "quality" in html_content

    def test_statistics_displayed(self, output: StringIO) -> None:
        """Verify statistics are displayed for multiple runs."""
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
            score_stability=StabilityAssessment(
                level=StabilityLevel.STABLE,
                cv=0.04,
                message="Stable results",
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
            runs_per_test=5,
            tests=[
                TestReport(
                    test_id="test-001",
                    test_name="Test One",
                    success=True,
                    score=85.0,
                    total_runs=5,
                    successful_runs=5,
                    statistics=stats,
                )
            ],
        )

        reporter = HTMLReporter(output=output)
        reporter.report(report)
        html_content = output.getvalue()

        # Should include run statistics
        assert "Run Statistics" in html_content or "Runs" in html_content
        assert "Mean Score" in html_content or "mean" in html_content.lower()
        # Should include stability badge
        assert "stable" in html_content.lower()

    def test_trace_viewer_included(self, output: StringIO) -> None:
        """Verify trace viewer is included when trace data exists."""
        # Note: Current implementation doesn't populate trace from TestReport
        # This test verifies the structure exists even if empty
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

        reporter = HTMLReporter(output=output, include_trace=True)
        reporter.report(report)
        html_content = output.getvalue()

        # Trace viewer toggle function should be included
        assert "toggleTrace" in html_content

    def test_results_chart_included(
        self, reporter: HTMLReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify results pie chart is included."""
        reporter.report(basic_suite_report)
        html_content = output.getvalue()

        assert "resultsChart" in html_content
        assert "doughnut" in html_content

    def test_score_distribution_chart_included(
        self, reporter: HTMLReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify score distribution bar chart is included when scores exist."""
        reporter.report(basic_suite_report)
        html_content = output.getvalue()

        assert "scoresChart" in html_content

    def test_no_scores_chart_when_no_scores(self, output: StringIO) -> None:
        """Verify score chart is not included when no scores exist."""
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
                    score=None,
                )
            ],
        )

        reporter = HTMLReporter(output=output)
        reporter.report(report)
        html_content = output.getvalue()

        # Results chart should still be there
        assert "resultsChart" in html_content
        # Score distribution chart should not render data
        assert "Score Distribution" not in html_content

    def test_suite_error_displayed(self, output: StringIO) -> None:
        """Verify suite-level errors are displayed."""
        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            success_rate=0.0,
            error="Failed to connect to agent",
            tests=[],
        )

        reporter = HTMLReporter(output=output)
        reporter.report(report)
        html_content = output.getvalue()

        assert "Failed to connect to agent" in html_content
        assert "Suite Error" in html_content or "error" in html_content.lower()

    def test_null_duration_handled(self, output: StringIO) -> None:
        """Verify null duration is handled gracefully."""
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

        reporter = HTMLReporter(output=output)
        reporter.report(report)
        html_content = output.getvalue()

        # Should handle None gracefully (N/A or similar)
        assert "N/A" in html_content or "<!DOCTYPE html>" in html_content

    def test_special_characters_escaped(self, output: StringIO) -> None:
        """Verify special characters in test names are escaped."""
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
                    test_name="Test <script>alert('xss')</script>",
                    success=True,
                )
            ],
        )

        reporter = HTMLReporter(output=output)
        reporter.report(report)
        html_content = output.getvalue()

        # Raw script tag should NOT be in output
        assert "<script>alert('xss')</script>" not in html_content
        # Should be escaped
        assert "&lt;script&gt;" in html_content or "alert" not in html_content

    def test_long_test_name_truncated(
        self, reporter: HTMLReporter, output: StringIO
    ) -> None:
        """Verify long test names are truncated in charts."""
        long_name = "A" * 100
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
                    test_name=long_name,
                    success=True,
                    score=85.0,
                )
            ],
        )

        reporter.report(report)
        # The truncation is for chart labels, not the HTML itself
        # Just verify it doesn't break
        html_content = output.getvalue()
        assert "<!DOCTYPE html>" in html_content


class TestHTMLReporterScoreColors:
    """Tests for score color classification."""

    @pytest.fixture
    def reporter(self) -> HTMLReporter:
        """Create an HTML reporter."""
        return HTMLReporter(output=StringIO())

    def test_high_score_color(self, reporter: HTMLReporter) -> None:
        """Verify high scores get green color."""
        color = reporter._get_score_color(85.0)
        assert color == "#22c55e"  # Green

    def test_medium_score_color(self, reporter: HTMLReporter) -> None:
        """Verify medium scores get yellow/amber color."""
        color = reporter._get_score_color(65.0)
        assert color == "#f59e0b"  # Yellow/amber

    def test_low_score_color(self, reporter: HTMLReporter) -> None:
        """Verify low scores get red color."""
        color = reporter._get_score_color(30.0)
        assert color == "#ef4444"  # Red

    def test_boundary_80_is_high(self, reporter: HTMLReporter) -> None:
        """Verify 80 is classified as high."""
        assert reporter._get_score_class(80.0) == "high"

    def test_boundary_50_is_medium(self, reporter: HTMLReporter) -> None:
        """Verify 50 is classified as medium."""
        assert reporter._get_score_class(50.0) == "medium"

    def test_boundary_49_is_low(self, reporter: HTMLReporter) -> None:
        """Verify 49 is classified as low."""
        assert reporter._get_score_class(49.0) == "low"

    def test_none_score_class(self, reporter: HTMLReporter) -> None:
        """Verify None score returns empty class."""
        assert reporter._get_score_class(None) == ""


class TestHTMLReporterHelpers:
    """Tests for HTML reporter helper methods."""

    @pytest.fixture
    def reporter(self) -> HTMLReporter:
        """Create an HTML reporter."""
        return HTMLReporter(output=StringIO())

    def test_truncate_short_name(self, reporter: HTMLReporter) -> None:
        """Verify short names are not truncated."""
        result = reporter._truncate_name("Short", 30)
        assert result == "Short"

    def test_truncate_long_name(self, reporter: HTMLReporter) -> None:
        """Verify long names are truncated with ellipsis."""
        result = reporter._truncate_name("A" * 50, 30)
        assert len(result) == 30
        assert result.endswith("...")

    def test_truncate_exact_length(self, reporter: HTMLReporter) -> None:
        """Verify exact length names are not truncated."""
        result = reporter._truncate_name("A" * 30, 30)
        assert result == "A" * 30
        assert "..." not in result
