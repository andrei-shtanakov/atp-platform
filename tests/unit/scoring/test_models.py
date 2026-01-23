"""Unit tests for scoring models."""

import pytest

from atp.scoring.models import (
    ComponentScore,
    NormalizationConfig,
    ScoreBreakdown,
    ScoredTestResult,
)


class TestNormalizationConfig:
    """Tests for NormalizationConfig model."""

    def test_default_values(self) -> None:
        """Test default values are None."""
        config = NormalizationConfig()
        assert config.max_steps is None
        assert config.optimal_steps is None
        assert config.max_tokens is None
        assert config.max_cost_usd is None

    def test_with_values(self) -> None:
        """Test creating with specific values."""
        config = NormalizationConfig(
            max_steps=100,
            optimal_steps=5,
            max_tokens=50000,
            max_cost_usd=0.50,
        )
        assert config.max_steps == 100
        assert config.optimal_steps == 5
        assert config.max_tokens == 50000
        assert config.max_cost_usd == 0.50

    def test_optimal_cannot_exceed_max(self) -> None:
        """Test optimal_steps cannot exceed max_steps."""
        with pytest.raises(ValueError, match="cannot exceed max_steps"):
            NormalizationConfig(max_steps=10, optimal_steps=20)

    def test_optimal_equals_max_is_valid(self) -> None:
        """Test optimal_steps can equal max_steps."""
        config = NormalizationConfig(max_steps=10, optimal_steps=10)
        assert config.optimal_steps == config.max_steps

    def test_max_steps_validation(self) -> None:
        """Test max_steps must be >= 1."""
        with pytest.raises(ValueError):
            NormalizationConfig(max_steps=0)

    def test_max_cost_usd_validation(self) -> None:
        """Test max_cost_usd must be > 0."""
        with pytest.raises(ValueError):
            NormalizationConfig(max_cost_usd=0)


class TestComponentScore:
    """Tests for ComponentScore model."""

    def test_create_component_score(self) -> None:
        """Test creating a component score."""
        score = ComponentScore(
            name="quality",
            raw_value=0.85,
            normalized_value=0.85,
            weight=0.4,
            weighted_value=0.34,
        )
        assert score.name == "quality"
        assert score.raw_value == 0.85
        assert score.normalized_value == 0.85
        assert score.weight == 0.4
        assert score.weighted_value == 0.34
        assert score.details is None

    def test_with_details(self) -> None:
        """Test component score with details."""
        score = ComponentScore(
            name="efficiency",
            raw_value=15,
            normalized_value=0.7,
            weight=0.2,
            weighted_value=0.14,
            details={"actual_steps": 15, "max_steps": 50},
        )
        assert score.details == {"actual_steps": 15, "max_steps": 50}

    def test_none_raw_value_allowed(self) -> None:
        """Test that raw_value can be None."""
        score = ComponentScore(
            name="cost",
            raw_value=None,
            normalized_value=1.0,
            weight=0.1,
            weighted_value=0.1,
        )
        assert score.raw_value is None

    def test_normalized_value_bounds(self) -> None:
        """Test normalized_value must be 0-1."""
        with pytest.raises(ValueError):
            ComponentScore(
                name="test",
                normalized_value=1.5,
                weight=0.5,
                weighted_value=0.75,
            )

        with pytest.raises(ValueError):
            ComponentScore(
                name="test",
                normalized_value=-0.1,
                weight=0.5,
                weighted_value=0.0,
            )

    def test_name_required(self) -> None:
        """Test name is required and non-empty."""
        with pytest.raises(ValueError):
            ComponentScore(
                name="",
                normalized_value=0.5,
                weight=0.5,
                weighted_value=0.25,
            )


class TestScoreBreakdown:
    """Tests for ScoreBreakdown model."""

    @pytest.fixture
    def sample_breakdown(self) -> ScoreBreakdown:
        """Create a sample score breakdown."""
        return ScoreBreakdown(
            quality=ComponentScore(
                name="quality",
                normalized_value=0.9,
                weight=0.4,
                weighted_value=0.36,
            ),
            completeness=ComponentScore(
                name="completeness",
                normalized_value=0.8,
                weight=0.3,
                weighted_value=0.24,
            ),
            efficiency=ComponentScore(
                name="efficiency",
                normalized_value=0.7,
                weight=0.2,
                weighted_value=0.14,
            ),
            cost=ComponentScore(
                name="cost",
                normalized_value=1.0,
                weight=0.1,
                weighted_value=0.1,
            ),
        )

    def test_final_score_calculation(self, sample_breakdown: ScoreBreakdown) -> None:
        """Test final score is sum of weighted values * 100."""
        expected = (0.36 + 0.24 + 0.14 + 0.1) * 100
        assert sample_breakdown.final_score == pytest.approx(expected, rel=0.01)

    def test_components_list(self, sample_breakdown: ScoreBreakdown) -> None:
        """Test components property returns all components."""
        components = sample_breakdown.components
        assert len(components) == 4
        names = [c.name for c in components]
        assert "quality" in names
        assert "completeness" in names
        assert "efficiency" in names
        assert "cost" in names

    def test_to_dict(self, sample_breakdown: ScoreBreakdown) -> None:
        """Test to_dict conversion."""
        result = sample_breakdown.to_dict()
        assert "final_score" in result
        assert "components" in result
        assert result["components"]["quality"]["weight"] == 0.4
        assert result["components"]["completeness"]["normalized"] == 0.8

    def test_perfect_score(self) -> None:
        """Test perfect score is 100."""
        breakdown = ScoreBreakdown(
            quality=ComponentScore(
                name="quality",
                normalized_value=1.0,
                weight=0.4,
                weighted_value=0.4,
            ),
            completeness=ComponentScore(
                name="completeness",
                normalized_value=1.0,
                weight=0.3,
                weighted_value=0.3,
            ),
            efficiency=ComponentScore(
                name="efficiency",
                normalized_value=1.0,
                weight=0.2,
                weighted_value=0.2,
            ),
            cost=ComponentScore(
                name="cost",
                normalized_value=1.0,
                weight=0.1,
                weighted_value=0.1,
            ),
        )
        assert breakdown.final_score == 100.0

    def test_zero_score(self) -> None:
        """Test zero score is 0."""
        breakdown = ScoreBreakdown(
            quality=ComponentScore(
                name="quality",
                normalized_value=0.0,
                weight=0.4,
                weighted_value=0.0,
            ),
            completeness=ComponentScore(
                name="completeness",
                normalized_value=0.0,
                weight=0.3,
                weighted_value=0.0,
            ),
            efficiency=ComponentScore(
                name="efficiency",
                normalized_value=0.0,
                weight=0.2,
                weighted_value=0.0,
            ),
            cost=ComponentScore(
                name="cost",
                normalized_value=0.0,
                weight=0.1,
                weighted_value=0.0,
            ),
        )
        assert breakdown.final_score == 0.0


class TestScoredTestResult:
    """Tests for ScoredTestResult model."""

    @pytest.fixture
    def sample_breakdown(self) -> ScoreBreakdown:
        """Create a sample score breakdown."""
        return ScoreBreakdown(
            quality=ComponentScore(
                name="quality",
                normalized_value=0.9,
                weight=0.4,
                weighted_value=0.36,
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

    def test_create_scored_result(self, sample_breakdown: ScoreBreakdown) -> None:
        """Test creating a scored test result."""
        result = ScoredTestResult(
            test_id="test-001",
            score=91.0,
            breakdown=sample_breakdown,
            passed=True,
        )
        assert result.test_id == "test-001"
        assert result.score == 91.0
        assert result.passed is True

    def test_failed_result(self, sample_breakdown: ScoreBreakdown) -> None:
        """Test creating a failed test result."""
        result = ScoredTestResult(
            test_id="test-002",
            score=45.0,
            breakdown=sample_breakdown,
            passed=False,
        )
        assert result.passed is False

    def test_score_validation(self, sample_breakdown: ScoreBreakdown) -> None:
        """Test score must be 0-100."""
        with pytest.raises(ValueError):
            ScoredTestResult(
                test_id="test",
                score=101.0,
                breakdown=sample_breakdown,
                passed=True,
            )

        with pytest.raises(ValueError):
            ScoredTestResult(
                test_id="test",
                score=-1.0,
                breakdown=sample_breakdown,
                passed=True,
            )
