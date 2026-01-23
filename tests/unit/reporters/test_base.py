"""Tests for the base reporter module."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from atp.loader.models import TestDefinition
    from atp.protocol import ATPResponse

from atp.evaluators.base import EvalCheck, EvalResult
from atp.reporters.base import Reporter, SuiteReport, TestReport
from atp.runner.models import SuiteResult, TestResult
from atp.scoring.models import ComponentScore, ScoreBreakdown, ScoredTestResult


class ConcreteReporter(Reporter):
    """Concrete reporter for testing base class."""

    def __init__(self) -> None:
        self.reported: SuiteReport | None = None

    @property
    def name(self) -> str:
        return "test"

    def report(self, report: SuiteReport) -> None:
        self.reported = report


class TestReporterBase:
    """Tests for the Reporter base class."""

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        """Verify Reporter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Reporter()  # type: ignore[abstract]

    def test_concrete_reporter_name(self) -> None:
        """Verify concrete reporter has a name."""
        reporter = ConcreteReporter()
        assert reporter.name == "test"

    def test_supports_streaming_default_false(self) -> None:
        """Verify default supports_streaming is False."""
        reporter = ConcreteReporter()
        assert reporter.supports_streaming is False

    def test_format_duration_milliseconds(self) -> None:
        """Verify duration formatting for milliseconds."""
        reporter = ConcreteReporter()
        assert reporter._format_duration(0.5) == "500ms"
        assert reporter._format_duration(0.001) == "1ms"

    def test_format_duration_seconds(self) -> None:
        """Verify duration formatting for seconds."""
        reporter = ConcreteReporter()
        assert reporter._format_duration(1.5) == "1.5s"
        assert reporter._format_duration(30.0) == "30.0s"

    def test_format_duration_minutes(self) -> None:
        """Verify duration formatting for minutes."""
        reporter = ConcreteReporter()
        assert reporter._format_duration(90.0) == "1m 30.0s"
        assert reporter._format_duration(120.0) == "2m 0.0s"

    def test_format_duration_hours(self) -> None:
        """Verify duration formatting for hours."""
        reporter = ConcreteReporter()
        assert reporter._format_duration(3660.0) == "1h 1m"
        assert reporter._format_duration(7200.0) == "2h 0m"

    def test_format_duration_none(self) -> None:
        """Verify duration formatting for None."""
        reporter = ConcreteReporter()
        assert reporter._format_duration(None) == "N/A"

    def test_format_score(self) -> None:
        """Verify score formatting."""
        reporter = ConcreteReporter()
        assert reporter._format_score(85.5) == "85.5/100"
        assert reporter._format_score(100.0) == "100.0/100"

    def test_format_score_none(self) -> None:
        """Verify score formatting for None."""
        reporter = ConcreteReporter()
        assert reporter._format_score(None) == "N/A"


class TestTestReport:
    """Tests for the TestReport model."""

    def test_create_basic_test_report(self) -> None:
        """Verify basic test report creation."""
        report = TestReport(
            test_id="test-001",
            test_name="Test One",
            success=True,
        )

        assert report.test_id == "test-001"
        assert report.test_name == "Test One"
        assert report.success is True
        assert report.score is None
        assert report.duration_seconds is None
        assert report.total_runs == 1
        assert report.successful_runs == 1
        assert report.eval_results == []
        assert report.scored_result is None
        assert report.statistics is None
        assert report.error is None

    def test_create_full_test_report(self) -> None:
        """Verify full test report creation with all fields."""
        eval_result = EvalResult(
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

        report = TestReport(
            test_id="test-001",
            test_name="Test One",
            success=True,
            score=85.5,
            duration_seconds=1.5,
            total_runs=3,
            successful_runs=3,
            eval_results=[eval_result],
            error=None,
        )

        assert report.score == 85.5
        assert report.duration_seconds == 1.5
        assert report.total_runs == 3
        assert report.successful_runs == 3
        assert len(report.eval_results) == 1


class TestSuiteReport:
    """Tests for the SuiteReport model."""

    def test_create_basic_suite_report(self) -> None:
        """Verify basic suite report creation."""
        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=3,
            passed_tests=2,
            failed_tests=1,
            success_rate=0.67,
        )

        assert report.suite_name == "test-suite"
        assert report.agent_name == "test-agent"
        assert report.total_tests == 3
        assert report.passed_tests == 2
        assert report.failed_tests == 1
        assert report.success_rate == 0.67
        assert report.duration_seconds is None
        assert report.runs_per_test == 1
        assert report.tests == []
        assert report.error is None

    def test_create_suite_report_with_tests(self) -> None:
        """Verify suite report with test reports."""
        test_reports = [
            TestReport(test_id="test-001", test_name="Test One", success=True),
            TestReport(test_id="test-002", test_name="Test Two", success=False),
        ]

        report = SuiteReport(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=2,
            passed_tests=1,
            failed_tests=1,
            success_rate=0.5,
            tests=test_reports,
        )

        assert len(report.tests) == 2
        assert report.tests[0].test_id == "test-001"
        assert report.tests[1].test_id == "test-002"


class TestSuiteReportFromSuiteResult:
    """Tests for SuiteReport.from_suite_result factory method."""

    @pytest.fixture
    def sample_test_definition(self) -> TestDefinition:
        """Create a sample test definition."""
        from atp.loader.models import TaskDefinition, TestDefinition

        return TestDefinition(
            id="test-001",
            name="Test One",
            task=TaskDefinition(description="Sample task"),
        )

    @pytest.fixture
    def sample_response(self) -> ATPResponse:
        """Create a sample ATP response."""
        from atp.protocol import ATPResponse, ResponseStatus

        return ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
        )

    def test_from_suite_result_basic(
        self, sample_test_definition: TestDefinition, sample_response: ATPResponse
    ) -> None:
        """Verify basic conversion from SuiteResult."""
        from atp.runner.models import RunResult

        run_result = RunResult(
            test_id="test-001",
            run_number=1,
            response=sample_response,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        test_result = TestResult(
            test=sample_test_definition,
            runs=[run_result],
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        suite_result = SuiteResult(
            suite_name="test-suite",
            agent_name="test-agent",
            tests=[test_result],
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        report = SuiteReport.from_suite_result(suite_result)

        assert report.suite_name == "test-suite"
        assert report.agent_name == "test-agent"
        assert report.total_tests == 1
        assert len(report.tests) == 1
        assert report.tests[0].test_id == "test-001"
        assert report.tests[0].test_name == "Test One"

    def test_from_suite_result_with_eval_results(
        self, sample_test_definition: TestDefinition, sample_response: ATPResponse
    ) -> None:
        """Verify conversion with evaluation results."""
        from atp.runner.models import RunResult

        run_result = RunResult(
            test_id="test-001",
            run_number=1,
            response=sample_response,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        test_result = TestResult(
            test=sample_test_definition,
            runs=[run_result],
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        suite_result = SuiteResult(
            suite_name="test-suite",
            agent_name="test-agent",
            tests=[test_result],
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        eval_results = {
            "test-001": [
                EvalResult(
                    evaluator="artifact",
                    checks=[
                        EvalCheck(
                            name="file_exists",
                            passed=True,
                            score=1.0,
                            message="OK",
                        )
                    ],
                )
            ]
        }

        report = SuiteReport.from_suite_result(suite_result, eval_results=eval_results)

        assert len(report.tests[0].eval_results) == 1
        assert report.tests[0].eval_results[0].evaluator == "artifact"

    def test_from_suite_result_with_scored_results(
        self, sample_test_definition: TestDefinition, sample_response: ATPResponse
    ) -> None:
        """Verify conversion with scored results."""
        from atp.runner.models import RunResult

        run_result = RunResult(
            test_id="test-001",
            run_number=1,
            response=sample_response,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        test_result = TestResult(
            test=sample_test_definition,
            runs=[run_result],
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        suite_result = SuiteResult(
            suite_name="test-suite",
            agent_name="test-agent",
            tests=[test_result],
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

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

        scored_results = {"test-001": scored}

        report = SuiteReport.from_suite_result(
            suite_result, scored_results=scored_results
        )

        assert report.tests[0].score == 85.0
        assert report.tests[0].scored_result is not None
        assert report.tests[0].scored_result.passed is True
