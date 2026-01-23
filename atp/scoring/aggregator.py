"""Score aggregation for test evaluation results."""

import math

from atp.evaluators.base import EvalResult
from atp.loader.models import ScoringWeights
from atp.protocol import ATPResponse

from .models import (
    ComponentScore,
    NormalizationConfig,
    ScoreBreakdown,
    ScoredTestResult,
)


class ScoreAggregator:
    """
    Aggregates evaluator results into a composite score.

    The final score is calculated as:
        Score = Σ(weight_i × component_i) × 100

    Components:
    - quality: mean of artifact scores and LLM evaluation scores
    - completeness: passed_checks / total_checks
    - efficiency: normalized based on steps used vs max/optimal
    - cost: 1 - log(1 + tokens/max_tokens) / log(2)
    """

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        normalization: NormalizationConfig | None = None,
    ) -> None:
        """
        Initialize the score aggregator.

        Args:
            weights: Scoring weights for each component.
                     If None, uses default weights (0.4, 0.3, 0.2, 0.1).
            normalization: Configuration for normalizing metrics.
                          If None, uses default values.
        """
        self.weights = weights or ScoringWeights()
        self.normalization = normalization or NormalizationConfig()

        self._validate_weights()

    def _validate_weights(self) -> None:
        """Validate that weights sum to 1.0 (within tolerance)."""
        total = (
            self.weights.quality_weight
            + self.weights.completeness_weight
            + self.weights.efficiency_weight
            + self.weights.cost_weight
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total:.3f}")

    def calculate_quality_score(
        self,
        eval_results: list[EvalResult],
    ) -> ComponentScore:
        """
        Calculate quality score from evaluation results.

        Quality is the mean of all artifact and LLM evaluation scores.

        Args:
            eval_results: List of evaluation results.

        Returns:
            ComponentScore with quality metrics.
        """
        if not eval_results:
            return ComponentScore(
                name="quality",
                raw_value=None,
                normalized_value=1.0,
                weight=self.weights.quality_weight,
                weighted_value=self.weights.quality_weight,
                details={"note": "No evaluations, assuming perfect quality"},
            )

        scores: list[float] = []
        artifact_scores: list[float] = []
        llm_scores: list[float] = []

        for result in eval_results:
            for check in result.checks:
                scores.append(check.score)
                if result.evaluator in ("artifact", "artifact_evaluator"):
                    artifact_scores.append(check.score)
                elif result.evaluator in ("llm_judge", "llm_evaluator"):
                    llm_scores.append(check.score)

        if not scores:
            normalized = 1.0
        else:
            normalized = sum(scores) / len(scores)

        weighted = normalized * self.weights.quality_weight

        return ComponentScore(
            name="quality",
            raw_value=normalized,
            normalized_value=normalized,
            weight=self.weights.quality_weight,
            weighted_value=weighted,
            details={
                "total_checks": len(scores),
                "artifact_checks": len(artifact_scores),
                "llm_checks": len(llm_scores),
                "mean_artifact_score": (
                    sum(artifact_scores) / len(artifact_scores)
                    if artifact_scores
                    else None
                ),
                "mean_llm_score": (
                    sum(llm_scores) / len(llm_scores) if llm_scores else None
                ),
            },
        )

    def calculate_completeness_score(
        self,
        eval_results: list[EvalResult],
    ) -> ComponentScore:
        """
        Calculate completeness score from evaluation results.

        Completeness = passed_checks / total_checks

        Args:
            eval_results: List of evaluation results.

        Returns:
            ComponentScore with completeness metrics.
        """
        total_checks = 0
        passed_checks = 0

        for result in eval_results:
            total_checks += len(result.checks)
            passed_checks += sum(1 for c in result.checks if c.passed)

        if total_checks == 0:
            normalized = 1.0
        else:
            normalized = passed_checks / total_checks

        weighted = normalized * self.weights.completeness_weight

        return ComponentScore(
            name="completeness",
            raw_value=passed_checks,
            normalized_value=normalized,
            weight=self.weights.completeness_weight,
            weighted_value=weighted,
            details={
                "passed_checks": passed_checks,
                "total_checks": total_checks,
            },
        )

    def calculate_efficiency_score(
        self,
        response: ATPResponse | None,
        max_steps: int | None = None,
        optimal_steps: int | None = None,
    ) -> ComponentScore:
        """
        Calculate efficiency score based on steps used.

        The efficiency score rewards completing tasks in fewer steps.
        Formula: efficiency = 1 - (steps - optimal) / (max - optimal)
                 Clamped to [0, 1]

        If only max_steps is provided, optimal defaults to 1.
        If neither is provided, assumes max from normalization config.

        Args:
            response: ATP Response containing metrics.
            max_steps: Maximum allowed steps (override).
            optimal_steps: Optimal (minimum) steps expected (override).

        Returns:
            ComponentScore with efficiency metrics.
        """
        max_s = max_steps or self.normalization.max_steps
        optimal_s = optimal_steps or self.normalization.optimal_steps

        if response is None or response.metrics is None:
            return ComponentScore(
                name="efficiency",
                raw_value=None,
                normalized_value=1.0,
                weight=self.weights.efficiency_weight,
                weighted_value=self.weights.efficiency_weight,
                details={"note": "No metrics available, assuming optimal efficiency"},
            )

        actual_steps = response.metrics.total_steps
        if actual_steps is None:
            return ComponentScore(
                name="efficiency",
                raw_value=None,
                normalized_value=1.0,
                weight=self.weights.efficiency_weight,
                weighted_value=self.weights.efficiency_weight,
                details={"note": "No step count in metrics"},
            )

        if max_s is None:
            normalized = 1.0 if actual_steps <= 1 else 0.5
            details = {
                "actual_steps": actual_steps,
                "note": "No max_steps configured, using heuristic",
            }
        else:
            optimal = optimal_s if optimal_s is not None else 1

            if actual_steps <= optimal:
                normalized = 1.0
            elif max_s <= optimal:
                normalized = 1.0 if actual_steps <= max_s else 0.0
            else:
                raw = 1 - (actual_steps - optimal) / (max_s - optimal)
                normalized = max(0.0, min(1.0, raw))

            details = {
                "actual_steps": actual_steps,
                "max_steps": max_s,
                "optimal_steps": optimal,
            }

        weighted = normalized * self.weights.efficiency_weight

        return ComponentScore(
            name="efficiency",
            raw_value=actual_steps,
            normalized_value=normalized,
            weight=self.weights.efficiency_weight,
            weighted_value=weighted,
            details=details,
        )

    def calculate_cost_score(
        self,
        response: ATPResponse | None,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> ComponentScore:
        """
        Calculate cost score based on token usage or direct cost.

        Formula: cost_score = 1 - log(1 + tokens/max_tokens) / log(2)
        This gives 1.0 for 0 tokens and 0.0 for max_tokens.

        Alternatively, if cost_usd is available:
        cost_score = 1 - (cost_usd / max_cost_usd)

        Args:
            response: ATP Response containing metrics.
            max_tokens: Maximum expected tokens (override).
            max_cost_usd: Maximum expected cost in USD (override).

        Returns:
            ComponentScore with cost metrics.
        """
        max_t = max_tokens or self.normalization.max_tokens
        max_c = max_cost_usd or self.normalization.max_cost_usd

        if response is None or response.metrics is None:
            return ComponentScore(
                name="cost",
                raw_value=None,
                normalized_value=1.0,
                weight=self.weights.cost_weight,
                weighted_value=self.weights.cost_weight,
                details={"note": "No metrics available, assuming zero cost"},
            )

        metrics = response.metrics

        if metrics.cost_usd is not None and max_c is not None:
            if metrics.cost_usd <= 0:
                normalized = 1.0
            else:
                ratio = metrics.cost_usd / max_c
                normalized = max(0.0, min(1.0, 1 - ratio))

            details = {
                "cost_usd": metrics.cost_usd,
                "max_cost_usd": max_c,
                "method": "direct_cost",
            }
            raw_value = metrics.cost_usd

        elif metrics.total_tokens is not None and max_t is not None:
            if metrics.total_tokens <= 0:
                normalized = 1.0
            else:
                ratio = metrics.total_tokens / max_t
                log_score = math.log(1 + ratio) / math.log(2)
                normalized = max(0.0, min(1.0, 1 - log_score))

            details = {
                "total_tokens": metrics.total_tokens,
                "max_tokens": max_t,
                "method": "token_based",
            }
            raw_value = metrics.total_tokens

        elif metrics.total_tokens is not None:
            normalized = 1.0 if metrics.total_tokens < 1000 else 0.5
            details = {
                "total_tokens": metrics.total_tokens,
                "note": "No max_tokens configured, using heuristic",
            }
            raw_value = metrics.total_tokens

        else:
            normalized = 1.0
            details = {"note": "No cost or token data available"}
            raw_value = None

        weighted = normalized * self.weights.cost_weight

        return ComponentScore(
            name="cost",
            raw_value=raw_value,
            normalized_value=normalized,
            weight=self.weights.cost_weight,
            weighted_value=weighted,
            details=details,
        )

    def aggregate(
        self,
        eval_results: list[EvalResult],
        response: ATPResponse | None = None,
        max_steps: int | None = None,
        optimal_steps: int | None = None,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> ScoreBreakdown:
        """
        Aggregate all scores into a final breakdown.

        Args:
            eval_results: List of evaluation results.
            response: ATP Response with metrics.
            max_steps: Maximum expected steps.
            optimal_steps: Optimal steps expected.
            max_tokens: Maximum expected tokens.
            max_cost_usd: Maximum expected cost.

        Returns:
            ScoreBreakdown with all component scores.
        """
        quality = self.calculate_quality_score(eval_results)
        completeness = self.calculate_completeness_score(eval_results)
        efficiency = self.calculate_efficiency_score(response, max_steps, optimal_steps)
        cost = self.calculate_cost_score(response, max_tokens, max_cost_usd)

        return ScoreBreakdown(
            quality=quality,
            completeness=completeness,
            efficiency=efficiency,
            cost=cost,
        )

    def score_test_result(
        self,
        test_id: str,
        eval_results: list[EvalResult],
        response: ATPResponse | None = None,
        max_steps: int | None = None,
        optimal_steps: int | None = None,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> ScoredTestResult:
        """
        Calculate final scored result for a test.

        Args:
            test_id: Test identifier.
            eval_results: List of evaluation results.
            response: ATP Response with metrics.
            max_steps: Maximum expected steps.
            optimal_steps: Optimal steps expected.
            max_tokens: Maximum expected tokens.
            max_cost_usd: Maximum expected cost.

        Returns:
            ScoredTestResult with final score and breakdown.
        """
        breakdown = self.aggregate(
            eval_results,
            response,
            max_steps,
            optimal_steps,
            max_tokens,
            max_cost_usd,
        )

        all_passed = all(result.passed for result in eval_results)

        return ScoredTestResult(
            test_id=test_id,
            score=breakdown.final_score,
            breakdown=breakdown,
            passed=all_passed,
        )
