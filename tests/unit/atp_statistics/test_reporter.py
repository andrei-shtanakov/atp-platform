"""Unit tests for StatisticalReporter."""

from datetime import datetime

import pytest

from atp.loader.models import Constraints, TaskDefinition, TestDefinition
from atp.protocol import ATPResponse, Metrics, ResponseStatus
from atp.runner.models import RunResult, SuiteResult, TestResult
from atp.scoring.models import ScoreBreakdown, ScoredTestResult
from atp.statistics.models import (
    StabilityAssessment,
    StabilityLevel,
    StatisticalResult,
    TestRunStatistics,
)
from atp.statistics.reporter import StatisticalReporter


@pytest.fixture
def sample_test_definition() -> TestDefinition:
    """Create a sample test definition."""
    return TestDefinition(
        id="test-001",
        name="Sample Test",
        task=TaskDefinition(description="Test task"),
        constraints=Constraints(timeout_seconds=60),
    )


@pytest.fixture
def sample_run_results(sample_test_definition: TestDefinition) -> list[RunResult]:
    """Create sample run results."""
    base_time = datetime.now()
    runs: list[RunResult] = []

    for i in range(5):
        run = RunResult(
            test_id=sample_test_definition.id,
            run_number=i + 1,
            response=ATPResponse(
                task_id=f"task-{i}",
                status=ResponseStatus.COMPLETED,
                metrics=Metrics(
                    total_steps=10 + i,
                    total_tokens=1000 + i * 100,
                    cost_usd=0.01 + i * 0.001,
                    wall_time_seconds=1.5 + i * 0.2,
                ),
            ),
            start_time=base_time,
            end_time=base_time,  # Simplified for testing
        )
        runs.append(run)

    return runs


@pytest.fixture
def sample_test_result(
    sample_test_definition: TestDefinition,
    sample_run_results: list[RunResult],
) -> TestResult:
    """Create a sample test result."""
    return TestResult(
        test=sample_test_definition,
        runs=sample_run_results,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )


@pytest.fixture
def sample_suite_result(sample_test_result: TestResult) -> SuiteResult:
    """Create a sample suite result."""
    return SuiteResult(
        suite_name="Test Suite",
        agent_name="Test Agent",
        tests=[sample_test_result],
        start_time=datetime.now(),
        end_time=datetime.now(),
    )


class TestStatisticalReporter:
    """Tests for StatisticalReporter class."""

    def test_initialization(self) -> None:
        """Test reporter initializes correctly."""
        reporter = StatisticalReporter()
        assert reporter._calculator is not None

    def test_compute_test_statistics_without_scores(
        self,
        sample_test_result: TestResult,
    ) -> None:
        """Test computing stats without scored results."""
        reporter = StatisticalReporter()
        stats = reporter.compute_test_statistics(sample_test_result)

        assert stats.test_id == sample_test_result.test.id
        assert stats.n_runs == sample_test_result.total_runs
        assert stats.successful_runs == sample_test_result.successful_runs
        assert stats.score_stats is None  # No scores provided
        # But duration stats should be computed if durations available

    def test_compute_test_statistics_with_scores(
        self,
        sample_test_result: TestResult,
    ) -> None:
        """Test computing stats with scored results."""
        reporter = StatisticalReporter()

        # Create mock scored results
        scored_results = [
            ScoredTestResult(
                test_id=sample_test_result.test.id,
                score=80.0 + i * 2,
                breakdown=_create_mock_breakdown(),
                passed=True,
            )
            for i in range(5)
        ]

        stats = reporter.compute_test_statistics(
            sample_test_result,
            scored_results=scored_results,
        )

        assert stats.score_stats is not None
        assert stats.score_stats.n_runs == 5
        assert stats.score_stability is not None

    def test_generate_summary(self, sample_suite_result: SuiteResult) -> None:
        """Test generating summary dictionary."""
        reporter = StatisticalReporter()
        summary = reporter.generate_summary(sample_suite_result)

        assert summary["suite_name"] == "Test Suite"
        assert summary["agent_name"] == "Test Agent"
        assert summary["total_tests"] == 1
        assert "success_rate" in summary
        assert "total_runs" in summary

    def test_generate_summary_with_test_statistics(
        self,
        sample_suite_result: SuiteResult,
    ) -> None:
        """Test summary with pre-computed test statistics."""
        reporter = StatisticalReporter()

        test_stats = {
            "test-001": TestRunStatistics(
                test_id="test-001",
                n_runs=5,
                successful_runs=5,
                success_rate=1.0,
                overall_stability=StabilityAssessment(
                    level=StabilityLevel.STABLE,
                    cv=0.03,
                    message="Consistent",
                ),
            )
        }

        summary = reporter.generate_summary(
            sample_suite_result,
            test_statistics=test_stats,
        )

        assert "test_statistics" in summary
        assert "test-001" in summary["test_statistics"]

    def test_format_text_summary(self, sample_suite_result: SuiteResult) -> None:
        """Test formatting text summary."""
        reporter = StatisticalReporter()
        text = reporter.format_text_summary(sample_suite_result)

        assert "Test Suite" in text
        assert "Test Agent" in text
        assert "Tests:" in text
        assert "Success Rate:" in text

    def test_format_text_summary_with_duration(
        self, sample_test_result: TestResult
    ) -> None:
        """Test text summary includes duration when present."""
        from datetime import timedelta

        base_time = datetime.now()
        suite_result = SuiteResult(
            suite_name="Test Suite",
            agent_name="Test Agent",
            tests=[sample_test_result],
            start_time=base_time,
            end_time=base_time + timedelta(seconds=125.5),
        )

        reporter = StatisticalReporter()
        text = reporter.format_text_summary(suite_result)

        assert "Duration:" in text
        assert "125.50s" in text

    def test_format_text_summary_with_statistics(
        self,
        sample_suite_result: SuiteResult,
    ) -> None:
        """Test text summary with statistics."""
        reporter = StatisticalReporter()

        score_stats = StatisticalResult(
            mean=85.0,
            std=5.0,
            min=78.0,
            max=92.0,
            median=85.0,
            confidence_interval=(80.0, 90.0),
            n_runs=5,
            coefficient_of_variation=0.059,
        )

        test_stats = {
            "test-001": TestRunStatistics(
                test_id="test-001",
                n_runs=5,
                successful_runs=5,
                success_rate=1.0,
                score_stats=score_stats,
                score_stability=StabilityAssessment(
                    level=StabilityLevel.MODERATE,
                    cv=0.059,
                    message="Acceptable variance",
                ),
                overall_stability=StabilityAssessment(
                    level=StabilityLevel.MODERATE,
                    cv=0.059,
                    message="Acceptable variance",
                ),
            )
        }

        text = reporter.format_text_summary(
            sample_suite_result,
            test_statistics=test_stats,
        )

        assert "test-001:" in text
        assert "Score:" in text
        assert "95% CI:" in text
        assert "Stability:" in text
        assert "moderate" in text

    def test_format_json_summary(self, sample_suite_result: SuiteResult) -> None:
        """Test JSON summary is dictionary."""
        reporter = StatisticalReporter()
        json_summary = reporter.format_json_summary(sample_suite_result)

        assert isinstance(json_summary, dict)
        assert json_summary["suite_name"] == "Test Suite"

    def test_format_text_summary_with_all_stats(
        self,
        sample_suite_result: SuiteResult,
    ) -> None:
        """Test text summary with all stats (score, duration, steps, tokens, cost)."""
        reporter = StatisticalReporter()

        score_stats = StatisticalResult(
            mean=85.0,
            std=5.0,
            min=78.0,
            max=92.0,
            median=85.0,
            confidence_interval=(80.0, 90.0),
            n_runs=5,
            coefficient_of_variation=0.059,
        )

        duration_stats = StatisticalResult(
            mean=2.5,
            std=0.3,
            min=2.0,
            max=3.0,
            median=2.5,
            confidence_interval=(2.2, 2.8),
            n_runs=5,
            coefficient_of_variation=0.12,
        )

        steps_stats = StatisticalResult(
            mean=10.0,
            std=2.0,
            min=7.0,
            max=13.0,
            median=10.0,
            confidence_interval=(8.0, 12.0),
            n_runs=5,
            coefficient_of_variation=0.2,
        )

        tokens_stats = StatisticalResult(
            mean=1500.0,
            std=200.0,
            min=1200.0,
            max=1800.0,
            median=1500.0,
            confidence_interval=(1300.0, 1700.0),
            n_runs=5,
            coefficient_of_variation=0.133,
        )

        cost_stats = StatisticalResult(
            mean=0.015,
            std=0.002,
            min=0.012,
            max=0.018,
            median=0.015,
            confidence_interval=(0.013, 0.017),
            n_runs=5,
            coefficient_of_variation=0.133,
        )

        test_stats = {
            "test-001": TestRunStatistics(
                test_id="test-001",
                n_runs=5,
                successful_runs=5,
                success_rate=1.0,
                score_stats=score_stats,
                duration_stats=duration_stats,
                steps_stats=steps_stats,
                tokens_stats=tokens_stats,
                cost_stats=cost_stats,
                score_stability=StabilityAssessment(
                    level=StabilityLevel.STABLE,
                    cv=0.059,
                    message="Consistent",
                ),
                overall_stability=StabilityAssessment(
                    level=StabilityLevel.STABLE,
                    cv=0.059,
                    message="Consistent",
                ),
            )
        }

        text = reporter.format_text_summary(
            sample_suite_result,
            test_statistics=test_stats,
        )

        # Verify all stats sections are present
        assert "test-001:" in text
        assert "Score:" in text
        assert "95% CI:" in text
        assert "Duration:" in text
        assert "Steps:" in text
        assert "Tokens:" in text
        assert "Cost:" in text
        assert "Stability:" in text


def _create_mock_breakdown() -> ScoreBreakdown:
    """Create a mock score breakdown for testing."""
    from atp.scoring.models import ComponentScore

    return ScoreBreakdown(
        quality=ComponentScore(
            name="quality",
            normalized_value=0.85,
            weight=0.4,
            weighted_value=0.34,
        ),
        completeness=ComponentScore(
            name="completeness",
            normalized_value=1.0,
            weight=0.3,
            weighted_value=0.3,
        ),
        efficiency=ComponentScore(
            name="efficiency",
            normalized_value=0.8,
            weight=0.2,
            weighted_value=0.16,
        ),
        cost=ComponentScore(
            name="cost",
            normalized_value=0.9,
            weight=0.1,
            weighted_value=0.09,
        ),
    )
