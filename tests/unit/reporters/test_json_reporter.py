"""Tests for the JSON reporter."""

import json
from io import StringIO
from pathlib import Path

import pytest

from atp.evaluators.base import EvalCheck, EvalResult
from atp.reporters.base import SuiteReport, TestReport
from atp.reporters.json_reporter import JSONReporter
from atp.scoring.models import ComponentScore, ScoreBreakdown, ScoredTestResult
from atp.statistics.models import (
    StabilityAssessment,
    StabilityLevel,
    StatisticalResult,
    TestRunStatistics,
)


class TestJSONReporter:
    """Tests for JSONReporter."""

    @pytest.fixture
    def output(self) -> StringIO:
        """Create a StringIO for capturing output."""
        return StringIO()

    @pytest.fixture
    def reporter(self, output: StringIO) -> JSONReporter:
        """Create a JSON reporter with captured output."""
        return JSONReporter(output=output, indent=2)

    @pytest.fixture
    def compact_reporter(self, output: StringIO) -> JSONReporter:
        """Create a compact JSON reporter."""
        return JSONReporter(output=output, indent=None)

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

    def test_reporter_name(self, reporter: JSONReporter) -> None:
        """Verify reporter name."""
        assert reporter.name == "json"

    def test_supports_streaming_false(self, reporter: JSONReporter) -> None:
        """Verify streaming is not supported."""
        assert reporter.supports_streaming is False

    def test_format_version_included(
        self, reporter: JSONReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify format version is included."""
        reporter.report(basic_suite_report)
        result = json.loads(output.getvalue())

        assert "version" in result
        assert result["version"] == JSONReporter.FORMAT_VERSION

    def test_generated_at_included(
        self, reporter: JSONReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify generated_at timestamp is included."""
        reporter.report(basic_suite_report)
        result = json.loads(output.getvalue())

        assert "generated_at" in result
        # Should be valid ISO format
        assert "T" in result["generated_at"]

    def test_summary_section(
        self, reporter: JSONReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify summary section content."""
        reporter.report(basic_suite_report)
        result = json.loads(output.getvalue())

        assert "summary" in result
        summary = result["summary"]
        assert summary["suite_name"] == "test-suite"
        assert summary["agent_name"] == "test-agent"
        assert summary["total_tests"] == 2
        assert summary["passed_tests"] == 2
        assert summary["failed_tests"] == 0
        assert summary["success_rate"] == 1.0
        assert summary["duration_seconds"] == 5.5
        assert summary["runs_per_test"] == 1
        assert summary["success"] is True
        assert summary["error"] is None

    def test_tests_section(
        self, reporter: JSONReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify tests section content."""
        reporter.report(basic_suite_report)
        result = json.loads(output.getvalue())

        assert "tests" in result
        assert len(result["tests"]) == 2

        test1 = result["tests"][0]
        assert test1["test_id"] == "test-001"
        assert test1["test_name"] == "Test One"
        assert test1["success"] is True
        assert test1["score"] == 85.0
        assert test1["duration_seconds"] == 2.5
        assert test1["runs"]["total"] == 1
        assert test1["runs"]["successful"] == 1

    def test_compact_json_output(
        self,
        compact_reporter: JSONReporter,
        basic_suite_report: SuiteReport,
        output: StringIO,
    ) -> None:
        """Verify compact JSON output has no indentation."""
        compact_reporter.report(basic_suite_report)
        result = output.getvalue()

        # Should be single line (plus newline)
        lines = result.strip().split("\n")
        assert len(lines) == 1

    def test_valid_json_output(
        self, reporter: JSONReporter, basic_suite_report: SuiteReport, output: StringIO
    ) -> None:
        """Verify output is valid JSON."""
        reporter.report(basic_suite_report)
        result = output.getvalue()

        # Should not raise
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_file_output(self, basic_suite_report: SuiteReport, tmp_path: Path) -> None:
        """Verify file output works correctly."""
        output_file = tmp_path / "results.json"
        reporter = JSONReporter(output_file=output_file, indent=2)

        reporter.report(basic_suite_report)

        assert output_file.exists()
        content = output_file.read_text()
        result = json.loads(content)
        assert result["summary"]["suite_name"] == "test-suite"

    def test_file_output_creates_parent_dirs(
        self, basic_suite_report: SuiteReport, tmp_path: Path
    ) -> None:
        """Verify file output creates parent directories."""
        output_file = tmp_path / "subdir" / "results.json"
        reporter = JSONReporter(output_file=output_file, indent=2)

        reporter.report(basic_suite_report)

        assert output_file.exists()

    def test_eval_results_included(
        self, reporter: JSONReporter, output: StringIO
    ) -> None:
        """Verify evaluation results are included."""
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
                            checks=[
                                EvalCheck(
                                    name="file_exists",
                                    passed=True,
                                    score=1.0,
                                    message="File found",
                                    details={"path": "/output/result.txt"},
                                )
                            ],
                        )
                    ],
                )
            ],
        )

        reporter.report(report)
        result = json.loads(output.getvalue())

        test = result["tests"][0]
        assert "evaluations" in test
        assert len(test["evaluations"]) == 1

        eval_data = test["evaluations"][0]
        assert eval_data["evaluator"] == "artifact"
        assert eval_data["passed"] is True
        assert eval_data["score"] == 1.0
        assert len(eval_data["checks"]) == 1

        check = eval_data["checks"][0]
        assert check["name"] == "file_exists"
        assert check["passed"] is True
        assert check["score"] == 1.0
        assert check["message"] == "File found"
        assert check["details"]["path"] == "/output/result.txt"

    def test_eval_results_excluded_when_disabled(self, output: StringIO) -> None:
        """Verify evaluation results can be excluded."""
        reporter = JSONReporter(output=output, include_details=False)

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
                            checks=[
                                EvalCheck(
                                    name="file_exists",
                                    passed=True,
                                    score=1.0,
                                )
                            ],
                        )
                    ],
                )
            ],
        )

        reporter.report(report)
        result = json.loads(output.getvalue())

        test = result["tests"][0]
        assert "evaluations" not in test

    def test_score_breakdown_included(
        self, reporter: JSONReporter, output: StringIO
    ) -> None:
        """Verify score breakdown is included."""
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

        reporter.report(report)
        result = json.loads(output.getvalue())

        test = result["tests"][0]
        assert "score_breakdown" in test
        assert "final_score" in test["score_breakdown"]
        assert "components" in test["score_breakdown"]

    def test_statistics_included(
        self, reporter: JSONReporter, output: StringIO
    ) -> None:
        """Verify statistics are included."""
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
        result = json.loads(output.getvalue())

        test = result["tests"][0]
        assert "statistics" in test
        assert test["statistics"]["n_runs"] == 5
        assert test["statistics"]["success_rate"] == 1.0

    def test_error_field_included(
        self, reporter: JSONReporter, output: StringIO
    ) -> None:
        """Verify error fields are included."""
        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            passed_tests=0,
            failed_tests=1,
            success_rate=0.0,
            error="Suite failed to initialize",
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
        result = json.loads(output.getvalue())

        assert result["summary"]["error"] == "Suite failed to initialize"
        assert result["tests"][0]["error"] == "Timeout after 300s"

    def test_null_values_handled(
        self, reporter: JSONReporter, output: StringIO
    ) -> None:
        """Verify null values are handled correctly."""
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
                    score=None,
                    duration_seconds=None,
                )
            ],
        )

        reporter.report(report)
        result = json.loads(output.getvalue())

        assert result["summary"]["duration_seconds"] is None
        assert result["tests"][0]["score"] is None
        assert result["tests"][0]["duration_seconds"] is None
