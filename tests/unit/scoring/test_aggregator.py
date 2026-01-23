"""Unit tests for ScoreAggregator."""

import math

import pytest

from atp.evaluators.base import EvalCheck, EvalResult
from atp.loader.models import ScoringWeights
from atp.protocol import ATPResponse, Metrics, ResponseStatus
from atp.scoring.aggregator import ScoreAggregator
from atp.scoring.models import NormalizationConfig


class TestScoreAggregatorInit:
    """Tests for ScoreAggregator initialization."""

    def test_default_weights(self) -> None:
        """Test default weights are applied."""
        aggregator = ScoreAggregator()
        assert aggregator.weights.quality_weight == 0.4
        assert aggregator.weights.completeness_weight == 0.3
        assert aggregator.weights.efficiency_weight == 0.2
        assert aggregator.weights.cost_weight == 0.1

    def test_custom_weights(self) -> None:
        """Test custom weights are applied."""
        weights = ScoringWeights(
            quality_weight=0.5,
            completeness_weight=0.2,
            efficiency_weight=0.2,
            cost_weight=0.1,
        )
        aggregator = ScoreAggregator(weights=weights)
        assert aggregator.weights.quality_weight == 0.5
        assert aggregator.weights.completeness_weight == 0.2

    def test_weights_must_sum_to_one(self) -> None:
        """Test weights validation."""
        weights = ScoringWeights(
            quality_weight=0.5,
            completeness_weight=0.5,
            efficiency_weight=0.5,
            cost_weight=0.1,
        )
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoreAggregator(weights=weights)

    def test_normalization_config(self) -> None:
        """Test normalization config is stored."""
        config = NormalizationConfig(max_steps=100, max_tokens=50000)
        aggregator = ScoreAggregator(normalization=config)
        assert aggregator.normalization.max_steps == 100
        assert aggregator.normalization.max_tokens == 50000


class TestQualityScore:
    """Tests for quality score calculation."""

    def test_empty_results(self) -> None:
        """Test quality with no evaluation results."""
        aggregator = ScoreAggregator()
        score = aggregator.calculate_quality_score([])
        assert score.normalized_value == 1.0
        assert score.weight == 0.4

    def test_perfect_quality(self) -> None:
        """Test perfect quality score."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[
                    EvalCheck(name="check1", passed=True, score=1.0),
                    EvalCheck(name="check2", passed=True, score=1.0),
                ],
            )
        ]
        score = aggregator.calculate_quality_score(results)
        assert score.normalized_value == 1.0
        assert score.weighted_value == 0.4

    def test_partial_quality(self) -> None:
        """Test partial quality score."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[
                    EvalCheck(name="check1", passed=True, score=1.0),
                    EvalCheck(name="check2", passed=True, score=0.5),
                ],
            )
        ]
        score = aggregator.calculate_quality_score(results)
        assert score.normalized_value == pytest.approx(0.75)
        assert score.weighted_value == pytest.approx(0.3)

    def test_zero_quality(self) -> None:
        """Test zero quality score."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[
                    EvalCheck(name="check1", passed=False, score=0.0),
                ],
            )
        ]
        score = aggregator.calculate_quality_score(results)
        assert score.normalized_value == 0.0
        assert score.weighted_value == 0.0

    def test_multiple_evaluators(self) -> None:
        """Test quality with multiple evaluator types."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[EvalCheck(name="check1", passed=True, score=0.8)],
            ),
            EvalResult(
                evaluator="llm_judge",
                checks=[EvalCheck(name="check2", passed=True, score=0.6)],
            ),
        ]
        score = aggregator.calculate_quality_score(results)
        assert score.normalized_value == pytest.approx(0.7)
        assert score.details is not None
        assert score.details["artifact_checks"] == 1
        assert score.details["llm_checks"] == 1

    def test_quality_details(self) -> None:
        """Test quality score includes details."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact_evaluator",
                checks=[
                    EvalCheck(name="c1", passed=True, score=0.9),
                    EvalCheck(name="c2", passed=True, score=0.7),
                ],
            ),
        ]
        score = aggregator.calculate_quality_score(results)
        assert score.details is not None
        assert score.details["total_checks"] == 2
        assert score.details["mean_artifact_score"] == pytest.approx(0.8)

    def test_results_with_empty_checks(self) -> None:
        """Test quality with results that have empty checks lists."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(evaluator="artifact", checks=[]),
            EvalResult(evaluator="behavior", checks=[]),
        ]
        score = aggregator.calculate_quality_score(results)
        assert score.normalized_value == 1.0


class TestCompletenessScore:
    """Tests for completeness score calculation."""

    def test_empty_results(self) -> None:
        """Test completeness with no evaluation results."""
        aggregator = ScoreAggregator()
        score = aggregator.calculate_completeness_score([])
        assert score.normalized_value == 1.0
        assert score.weight == 0.3

    def test_all_passed(self) -> None:
        """Test completeness when all checks pass."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="test",
                checks=[
                    EvalCheck(name="c1", passed=True, score=1.0),
                    EvalCheck(name="c2", passed=True, score=1.0),
                ],
            )
        ]
        score = aggregator.calculate_completeness_score(results)
        assert score.normalized_value == 1.0
        assert score.details == {"passed_checks": 2, "total_checks": 2}

    def test_partial_completeness(self) -> None:
        """Test partial completeness."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="test",
                checks=[
                    EvalCheck(name="c1", passed=True, score=1.0),
                    EvalCheck(name="c2", passed=False, score=0.0),
                    EvalCheck(name="c3", passed=True, score=0.5),
                ],
            )
        ]
        score = aggregator.calculate_completeness_score(results)
        assert score.normalized_value == pytest.approx(2 / 3)
        assert score.details == {"passed_checks": 2, "total_checks": 3}

    def test_zero_completeness(self) -> None:
        """Test zero completeness when all checks fail."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="test",
                checks=[
                    EvalCheck(name="c1", passed=False, score=0.0),
                    EvalCheck(name="c2", passed=False, score=0.0),
                ],
            )
        ]
        score = aggregator.calculate_completeness_score(results)
        assert score.normalized_value == 0.0

    def test_multiple_result_sets(self) -> None:
        """Test completeness across multiple result sets."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[EvalCheck(name="c1", passed=True, score=1.0)],
            ),
            EvalResult(
                evaluator="behavior",
                checks=[
                    EvalCheck(name="c2", passed=True, score=1.0),
                    EvalCheck(name="c3", passed=False, score=0.0),
                ],
            ),
        ]
        score = aggregator.calculate_completeness_score(results)
        assert score.normalized_value == pytest.approx(2 / 3)


class TestEfficiencyScore:
    """Tests for efficiency score calculation."""

    def test_no_response(self) -> None:
        """Test efficiency with no response."""
        aggregator = ScoreAggregator()
        score = aggregator.calculate_efficiency_score(None)
        assert score.normalized_value == 1.0

    def test_no_metrics(self) -> None:
        """Test efficiency with no metrics."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=None,
        )
        score = aggregator.calculate_efficiency_score(response)
        assert score.normalized_value == 1.0

    def test_no_step_count(self) -> None:
        """Test efficiency with no step count in metrics."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_tokens=1000),
        )
        score = aggregator.calculate_efficiency_score(response)
        assert score.normalized_value == 1.0

    def test_optimal_steps(self) -> None:
        """Test efficiency at optimal steps."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=5),
        )
        score = aggregator.calculate_efficiency_score(
            response, max_steps=50, optimal_steps=5
        )
        assert score.normalized_value == 1.0

    def test_max_steps(self) -> None:
        """Test efficiency at max steps."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=50),
        )
        score = aggregator.calculate_efficiency_score(
            response, max_steps=50, optimal_steps=5
        )
        assert score.normalized_value == 0.0

    def test_mid_range_efficiency(self) -> None:
        """Test efficiency between optimal and max."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=27),
        )
        score = aggregator.calculate_efficiency_score(
            response, max_steps=50, optimal_steps=5
        )
        expected = 1 - (27 - 5) / (50 - 5)
        assert score.normalized_value == pytest.approx(expected)

    def test_below_optimal(self) -> None:
        """Test efficiency when steps below optimal."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=2),
        )
        score = aggregator.calculate_efficiency_score(
            response, max_steps=50, optimal_steps=5
        )
        assert score.normalized_value == 1.0

    def test_exceeds_max(self) -> None:
        """Test efficiency when steps exceed max."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=60),
        )
        score = aggregator.calculate_efficiency_score(
            response, max_steps=50, optimal_steps=5
        )
        assert score.normalized_value == 0.0

    def test_no_max_configured(self) -> None:
        """Test efficiency with no max_steps configured."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=10),
        )
        score = aggregator.calculate_efficiency_score(response)
        assert score.details is not None
        assert "heuristic" in score.details.get("note", "")

    def test_max_equals_optimal(self) -> None:
        """Test efficiency when max_steps equals optimal_steps."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=5),
        )
        score = aggregator.calculate_efficiency_score(
            response, max_steps=5, optimal_steps=5
        )
        assert score.normalized_value == 1.0

    def test_max_equals_optimal_exceeds(self) -> None:
        """Test efficiency when max equals optimal but actual exceeds."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=10),
        )
        score = aggregator.calculate_efficiency_score(
            response, max_steps=5, optimal_steps=5
        )
        assert score.normalized_value == 0.0

    def test_uses_normalization_config(self) -> None:
        """Test efficiency uses normalization config."""
        config = NormalizationConfig(max_steps=30, optimal_steps=3)
        aggregator = ScoreAggregator(normalization=config)
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=16),
        )
        score = aggregator.calculate_efficiency_score(response)
        expected = 1 - (16 - 3) / (30 - 3)
        assert score.normalized_value == pytest.approx(expected)


class TestCostScore:
    """Tests for cost score calculation."""

    def test_no_response(self) -> None:
        """Test cost with no response."""
        aggregator = ScoreAggregator()
        score = aggregator.calculate_cost_score(None)
        assert score.normalized_value == 1.0

    def test_no_metrics(self) -> None:
        """Test cost with no metrics."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=None,
        )
        score = aggregator.calculate_cost_score(response)
        assert score.normalized_value == 1.0

    def test_zero_tokens(self) -> None:
        """Test cost with zero tokens."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_tokens=0),
        )
        score = aggregator.calculate_cost_score(response, max_tokens=50000)
        assert score.normalized_value == 1.0

    def test_max_tokens(self) -> None:
        """Test cost at max tokens."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_tokens=50000),
        )
        score = aggregator.calculate_cost_score(response, max_tokens=50000)
        expected = 1 - math.log(1 + 1) / math.log(2)
        assert score.normalized_value == pytest.approx(expected)

    def test_half_max_tokens(self) -> None:
        """Test cost at half max tokens."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_tokens=25000),
        )
        score = aggregator.calculate_cost_score(response, max_tokens=50000)
        expected = 1 - math.log(1 + 0.5) / math.log(2)
        assert score.normalized_value == pytest.approx(expected)

    def test_direct_cost_method(self) -> None:
        """Test cost calculation with direct cost_usd."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(cost_usd=0.25),
        )
        score = aggregator.calculate_cost_score(response, max_cost_usd=1.00)
        assert score.normalized_value == pytest.approx(0.75)
        assert score.details is not None
        assert score.details["method"] == "direct_cost"

    def test_zero_cost(self) -> None:
        """Test cost with zero cost_usd."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(cost_usd=0.0),
        )
        score = aggregator.calculate_cost_score(response, max_cost_usd=1.00)
        assert score.normalized_value == 1.0

    def test_exceeds_max_cost(self) -> None:
        """Test cost when exceeding max_cost_usd."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(cost_usd=1.50),
        )
        score = aggregator.calculate_cost_score(response, max_cost_usd=1.00)
        assert score.normalized_value == 0.0

    def test_no_max_configured_low_tokens(self) -> None:
        """Test cost heuristic with low tokens and no max."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_tokens=500),
        )
        score = aggregator.calculate_cost_score(response)
        assert score.normalized_value == 1.0

    def test_no_max_configured_high_tokens(self) -> None:
        """Test cost heuristic with high tokens and no max."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_tokens=5000),
        )
        score = aggregator.calculate_cost_score(response)
        assert score.normalized_value == 0.5

    def test_metrics_without_cost_or_tokens(self) -> None:
        """Test cost when metrics exist but no cost or token data."""
        aggregator = ScoreAggregator()
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=10, wall_time_seconds=5.0),
        )
        score = aggregator.calculate_cost_score(response)
        assert score.normalized_value == 1.0
        assert score.raw_value is None
        assert score.details is not None
        assert "No cost or token data available" in score.details.get("note", "")

    def test_uses_normalization_config(self) -> None:
        """Test cost uses normalization config."""
        config = NormalizationConfig(max_tokens=10000)
        aggregator = ScoreAggregator(normalization=config)
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_tokens=5000),
        )
        score = aggregator.calculate_cost_score(response)
        expected = 1 - math.log(1 + 0.5) / math.log(2)
        assert score.normalized_value == pytest.approx(expected)


class TestAggregate:
    """Tests for aggregate method."""

    def test_aggregate_all_components(self) -> None:
        """Test aggregation produces all components."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[EvalCheck(name="c1", passed=True, score=0.8)],
            )
        ]
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=10, total_tokens=5000),
        )
        breakdown = aggregator.aggregate(
            results, response, max_steps=50, max_tokens=50000
        )

        assert breakdown.quality.name == "quality"
        assert breakdown.completeness.name == "completeness"
        assert breakdown.efficiency.name == "efficiency"
        assert breakdown.cost.name == "cost"

    def test_aggregate_final_score(self) -> None:
        """Test aggregate produces valid final score."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[EvalCheck(name="c1", passed=True, score=1.0)],
            )
        ]
        breakdown = aggregator.aggregate(results)

        assert 0 <= breakdown.final_score <= 100

    def test_aggregate_with_custom_weights(self) -> None:
        """Test aggregate respects custom weights."""
        weights = ScoringWeights(
            quality_weight=0.6,
            completeness_weight=0.2,
            efficiency_weight=0.1,
            cost_weight=0.1,
        )
        aggregator = ScoreAggregator(weights=weights)
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[EvalCheck(name="c1", passed=True, score=0.5)],
            )
        ]
        breakdown = aggregator.aggregate(results)

        assert breakdown.quality.weight == 0.6
        assert breakdown.completeness.weight == 0.2


class TestScoreTestResult:
    """Tests for score_test_result method."""

    def test_basic_scoring(self) -> None:
        """Test basic test scoring."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[EvalCheck(name="c1", passed=True, score=0.9)],
            )
        ]
        scored = aggregator.score_test_result("test-001", results)

        assert scored.test_id == "test-001"
        assert 0 <= scored.score <= 100
        assert scored.passed is True
        assert scored.breakdown is not None

    def test_failed_test_result(self) -> None:
        """Test scoring with failed checks."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[EvalCheck(name="c1", passed=False, score=0.0)],
            )
        ]
        scored = aggregator.score_test_result("test-002", results)

        assert scored.passed is False

    def test_partial_pass(self) -> None:
        """Test scoring with partial pass."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[
                    EvalCheck(name="c1", passed=True, score=1.0),
                    EvalCheck(name="c2", passed=False, score=0.0),
                ],
            )
        ]
        scored = aggregator.score_test_result("test-003", results)

        assert scored.passed is False

    def test_with_response_metrics(self) -> None:
        """Test scoring with response metrics."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[EvalCheck(name="c1", passed=True, score=1.0)],
            )
        ]
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=5, total_tokens=1000),
        )
        scored = aggregator.score_test_result(
            "test-004",
            results,
            response=response,
            max_steps=50,
            max_tokens=10000,
        )

        assert scored.score > 0
        assert scored.breakdown.efficiency.raw_value == 5
        assert scored.breakdown.cost.raw_value == 1000

    def test_empty_results(self) -> None:
        """Test scoring with no evaluation results."""
        aggregator = ScoreAggregator()
        scored = aggregator.score_test_result("test-005", [])

        assert scored.score == 100.0
        assert scored.passed is True


class TestScoreCalculationExamples:
    """Example-based tests for score calculation."""

    def test_perfect_score_example(self) -> None:
        """Test perfect score scenario."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[
                    EvalCheck(name="file_exists", passed=True, score=1.0),
                    EvalCheck(name="content_valid", passed=True, score=1.0),
                ],
            ),
            EvalResult(
                evaluator="behavior",
                checks=[
                    EvalCheck(name="tools_used", passed=True, score=1.0),
                ],
            ),
        ]
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=3, total_tokens=500, cost_usd=0.01),
        )
        scored = aggregator.score_test_result(
            "perfect-test",
            results,
            response=response,
            max_steps=50,
            optimal_steps=3,
            max_cost_usd=1.00,
        )

        assert scored.score == pytest.approx(100.0, abs=0.1)
        assert scored.passed is True

    def test_mixed_results_example(self) -> None:
        """Test mixed results scenario."""
        aggregator = ScoreAggregator()
        results = [
            EvalResult(
                evaluator="artifact",
                checks=[
                    EvalCheck(name="file_exists", passed=True, score=1.0),
                    EvalCheck(name="content_valid", passed=False, score=0.3),
                ],
            ),
            EvalResult(
                evaluator="behavior",
                checks=[
                    EvalCheck(name="tools_used", passed=True, score=0.8),
                ],
            ),
        ]
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=25, total_tokens=30000),
        )
        scored = aggregator.score_test_result(
            "mixed-test",
            results,
            response=response,
            max_steps=50,
            optimal_steps=5,
            max_tokens=50000,
        )

        assert 0 < scored.score < 100
        assert scored.passed is False

    def test_requirement_example(self) -> None:
        """Test example from REQ-043.

        GIVEN weights: quality 0.4, completeness 0.3, efficiency 0.2, cost 0.1
        WHEN all evaluators complete
        THEN weighted score is computed and normalized to 0-100
        """
        weights = ScoringWeights(
            quality_weight=0.4,
            completeness_weight=0.3,
            efficiency_weight=0.2,
            cost_weight=0.1,
        )
        aggregator = ScoreAggregator(weights=weights)

        results = [
            EvalResult(
                evaluator="artifact",
                checks=[
                    EvalCheck(name="c1", passed=True, score=0.9),
                    EvalCheck(name="c2", passed=True, score=0.8),
                ],
            ),
        ]
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_steps=10, total_tokens=5000),
        )
        scored = aggregator.score_test_result(
            "req-test",
            results,
            response=response,
            max_steps=50,
            optimal_steps=5,
            max_tokens=50000,
        )

        breakdown = scored.breakdown.to_dict()
        assert "final_score" in breakdown
        assert "components" in breakdown
        assert 0 <= scored.score <= 100
