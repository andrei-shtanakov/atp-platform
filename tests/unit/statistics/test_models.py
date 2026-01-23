"""Unit tests for statistics models."""

import pytest

from atp.statistics.models import (
    StabilityAssessment,
    StabilityLevel,
    StatisticalResult,
    TestRunStatistics,
)


class TestStabilityLevel:
    """Tests for StabilityLevel enum."""

    def test_all_levels_exist(self) -> None:
        """Test that all expected stability levels exist."""
        assert StabilityLevel.STABLE == "stable"
        assert StabilityLevel.MODERATE == "moderate"
        assert StabilityLevel.UNSTABLE == "unstable"
        assert StabilityLevel.CRITICAL == "critical"

    def test_values_are_strings(self) -> None:
        """Test that all values are strings."""
        for level in StabilityLevel:
            assert isinstance(level.value, str)


class TestStabilityAssessment:
    """Tests for StabilityAssessment model."""

    def test_create_stable_assessment(self) -> None:
        """Test creating a stable assessment."""
        assessment = StabilityAssessment(
            level=StabilityLevel.STABLE,
            cv=0.03,
            message="Consistent results",
        )
        assert assessment.level == StabilityLevel.STABLE
        assert assessment.cv == 0.03
        assert assessment.message == "Consistent results"

    def test_create_critical_assessment(self) -> None:
        """Test creating a critical assessment."""
        assessment = StabilityAssessment(
            level=StabilityLevel.CRITICAL,
            cv=0.45,
            message="Very high variance",
        )
        assert assessment.level == StabilityLevel.CRITICAL
        assert assessment.cv == 0.45

    def test_cv_must_be_non_negative(self) -> None:
        """Test that CV must be non-negative."""
        with pytest.raises(ValueError):
            StabilityAssessment(
                level=StabilityLevel.STABLE,
                cv=-0.1,
                message="Invalid",
            )


class TestStatisticalResult:
    """Tests for StatisticalResult model."""

    def test_create_basic_result(self) -> None:
        """Test creating a basic statistical result."""
        result = StatisticalResult(
            mean=50.0,
            std=5.0,
            min=40.0,
            max=60.0,
            median=50.0,
            confidence_interval=(45.0, 55.0),
            n_runs=10,
            coefficient_of_variation=0.1,
        )
        assert result.mean == 50.0
        assert result.std == 5.0
        assert result.min == 40.0
        assert result.max == 60.0
        assert result.median == 50.0
        assert result.confidence_interval == (45.0, 55.0)
        assert result.n_runs == 10
        assert result.coefficient_of_variation == 0.1

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        result = StatisticalResult(
            mean=50.123456,
            std=5.123456,
            min=40.123456,
            max=60.123456,
            median=50.123456,
            confidence_interval=(45.123456, 55.123456),
            n_runs=10,
            coefficient_of_variation=0.123456,
        )
        d = result.to_dict()

        assert d["mean"] == 50.1235
        assert d["std"] == 5.1235
        assert d["min"] == 40.1235
        assert d["max"] == 60.1235
        assert d["median"] == 50.1235
        assert d["confidence_interval"] == (45.1235, 55.1235)
        assert d["n_runs"] == 10
        assert d["coefficient_of_variation"] == 0.1235

    def test_std_must_be_non_negative(self) -> None:
        """Test that std must be non-negative."""
        with pytest.raises(ValueError):
            StatisticalResult(
                mean=50.0,
                std=-5.0,
                min=40.0,
                max=60.0,
                median=50.0,
                confidence_interval=(45.0, 55.0),
                n_runs=10,
                coefficient_of_variation=0.1,
            )

    def test_n_runs_must_be_positive(self) -> None:
        """Test that n_runs must be at least 1."""
        with pytest.raises(ValueError):
            StatisticalResult(
                mean=50.0,
                std=5.0,
                min=40.0,
                max=60.0,
                median=50.0,
                confidence_interval=(45.0, 55.0),
                n_runs=0,
                coefficient_of_variation=0.1,
            )


class TestTestRunStatistics:
    """Tests for TestRunStatistics model."""

    def test_create_basic_statistics(self) -> None:
        """Test creating basic test run statistics."""
        stability = StabilityAssessment(
            level=StabilityLevel.STABLE,
            cv=0.03,
            message="Consistent",
        )
        stats = TestRunStatistics(
            test_id="test-001",
            n_runs=5,
            successful_runs=5,
            success_rate=1.0,
            overall_stability=stability,
        )
        assert stats.test_id == "test-001"
        assert stats.n_runs == 5
        assert stats.successful_runs == 5
        assert stats.success_rate == 1.0
        assert stats.overall_stability.level == StabilityLevel.STABLE

    def test_create_with_score_stats(self) -> None:
        """Test creating statistics with score data."""
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
        stability = StabilityAssessment(
            level=StabilityLevel.MODERATE,
            cv=0.059,
            message="Acceptable variance",
        )
        stats = TestRunStatistics(
            test_id="test-002",
            n_runs=5,
            successful_runs=5,
            success_rate=1.0,
            score_stats=score_stats,
            score_stability=stability,
            overall_stability=stability,
        )
        assert stats.score_stats is not None
        assert stats.score_stats.mean == 85.0
        assert stats.score_stability is not None

    def test_to_dict_minimal(self) -> None:
        """Test to_dict with minimal data."""
        stability = StabilityAssessment(
            level=StabilityLevel.STABLE,
            cv=0.0,
            message="No data",
        )
        stats = TestRunStatistics(
            test_id="test-003",
            n_runs=5,
            successful_runs=4,
            success_rate=0.8,
            overall_stability=stability,
        )
        d = stats.to_dict()

        assert d["test_id"] == "test-003"
        assert d["n_runs"] == 5
        assert d["successful_runs"] == 4
        assert d["success_rate"] == 0.8
        assert d["overall_stability"]["level"] == "stable"
        assert "score" not in d

    def test_to_dict_complete(self) -> None:
        """Test to_dict with complete data."""
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
        score_stability = StabilityAssessment(
            level=StabilityLevel.MODERATE,
            cv=0.059,
            message="Acceptable variance",
        )
        duration_stats = StatisticalResult(
            mean=2.5,
            std=0.5,
            min=1.8,
            max=3.2,
            median=2.4,
            confidence_interval=(2.0, 3.0),
            n_runs=5,
            coefficient_of_variation=0.2,
        )

        stats = TestRunStatistics(
            test_id="test-004",
            n_runs=5,
            successful_runs=5,
            success_rate=1.0,
            score_stats=score_stats,
            score_stability=score_stability,
            duration_stats=duration_stats,
            overall_stability=score_stability,
        )
        d = stats.to_dict()

        assert "score" in d
        assert d["score"]["mean"] == 85.0
        assert d["score"]["stability"]["level"] == "moderate"
        assert "duration_seconds" in d
        assert d["duration_seconds"]["mean"] == 2.5
