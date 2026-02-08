"""Payoff evaluator for game-theoretic agent assessment.

Evaluates agent performance based on payoff metrics:
average payoff, distribution, Pareto efficiency, and
social welfare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atp.evaluators.base import EvalCheck, EvalResult, Evaluator
from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse

from atp_games.models import GameResult


@dataclass(frozen=True)
class PayoffConfig:
    """Configuration for PayoffEvaluator.

    Attributes:
        min_payoff: Minimum average payoff threshold
            per player for pass (None = no threshold).
        max_payoff: Maximum average payoff threshold
            per player (None = no threshold).
        min_social_welfare: Minimum total social welfare
            threshold (None = no threshold).
        pareto_check: Whether to check Pareto efficiency.
        weights: Per-check weights for scoring.
    """

    min_payoff: dict[str, float] | None = None
    max_payoff: dict[str, float] | None = None
    min_social_welfare: float | None = None
    pareto_check: bool = False
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "average_payoff": 1.0,
            "social_welfare": 1.0,
            "pareto_efficiency": 1.0,
        },
    )


class PayoffEvaluator(Evaluator):
    """Evaluates game outcomes based on payoff metrics.

    Computes:
    - Average payoff per player per episode
    - Payoff distribution (min, max, percentiles)
    - Social welfare (sum of payoffs)
    - Pareto efficiency check

    Compatible with ATP evaluator registry and scoring
    pipeline via the standard Evaluator interface.
    """

    def __init__(
        self,
        config: PayoffConfig | None = None,
    ) -> None:
        self._config = config or PayoffConfig()

    @property
    def name(self) -> str:
        """Return evaluator name."""
        return "payoff"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Evaluate via ATP evaluator interface.

        Extracts GameResult from assertion config and
        delegates to evaluate_game().
        """
        config = assertion.config
        game_result_data = config.get("game_result")
        if game_result_data is None:
            return self._create_result(
                [
                    self._create_check(
                        name="payoff_data",
                        passed=False,
                        message="No game_result in assertion config",
                    ),
                ]
            )
        game_result = GameResult.from_dict(game_result_data)
        return self.evaluate_game(game_result, config)

    def evaluate_game(
        self,
        result: GameResult,
        config: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate a GameResult directly.

        Args:
            result: Completed game result to evaluate.
            config: Optional override config dict with
                keys: min_payoff, max_payoff,
                min_social_welfare, pareto_check.

        Returns:
            EvalResult with payoff checks.
        """
        eval_config = self._resolve_config(config)
        checks: list[EvalCheck] = []

        avg_payoffs = result.average_payoffs
        if not avg_payoffs:
            return self._create_result(
                [
                    self._create_check(
                        name="payoff_data",
                        passed=False,
                        message="No episodes to evaluate",
                    ),
                ]
            )

        # 1. Average payoff check
        checks.append(self._check_average_payoff(result, eval_config))

        # 2. Payoff distribution
        checks.append(self._check_payoff_distribution(result))

        # 3. Social welfare
        checks.append(self._check_social_welfare(result, eval_config))

        # 4. Pareto efficiency
        if eval_config.pareto_check:
            checks.append(self._check_pareto_efficiency(result))

        return self._create_result(checks)

    def _resolve_config(
        self,
        override: dict[str, Any] | None,
    ) -> PayoffConfig:
        """Merge override dict into default config."""
        if override is None:
            return self._config
        return PayoffConfig(
            min_payoff=override.get(
                "min_payoff",
                self._config.min_payoff,
            ),
            max_payoff=override.get(
                "max_payoff",
                self._config.max_payoff,
            ),
            min_social_welfare=override.get(
                "min_social_welfare",
                self._config.min_social_welfare,
            ),
            pareto_check=override.get(
                "pareto_check",
                self._config.pareto_check,
            ),
            weights=override.get(
                "weights",
                self._config.weights,
            ),
        )

    def _check_average_payoff(
        self,
        result: GameResult,
        config: PayoffConfig,
    ) -> EvalCheck:
        """Check average payoffs against thresholds."""
        avg = result.average_payoffs
        details: dict[str, Any] = {"average_payoffs": avg}
        failures: list[str] = []

        if config.min_payoff:
            for pid, threshold in config.min_payoff.items():
                actual = avg.get(pid, 0.0)
                if actual < threshold:
                    failures.append(f"{pid}: {actual:.3f} < {threshold:.3f}")
            details["min_payoff_thresholds"] = config.min_payoff

        if config.max_payoff:
            for pid, threshold in config.max_payoff.items():
                actual = avg.get(pid, 0.0)
                if actual > threshold:
                    failures.append(f"{pid}: {actual:.3f} > {threshold:.3f}")
            details["max_payoff_thresholds"] = config.max_payoff

        if failures:
            return EvalCheck(
                name="average_payoff",
                passed=False,
                score=0.0,
                message=(f"Payoff threshold violations: {'; '.join(failures)}"),
                details=details,
            )

        # Score based on relative payoff performance
        score = self._compute_payoff_score(avg, config)
        return EvalCheck(
            name="average_payoff",
            passed=True,
            score=score,
            message=(
                "Average payoffs: "
                + ", ".join(f"{pid}={v:.3f}" for pid, v in avg.items())
            ),
            details=details,
        )

    def _compute_payoff_score(
        self,
        avg_payoffs: dict[str, float],
        config: PayoffConfig,
    ) -> float:
        """Compute a 0-1 score for payoff performance.

        If thresholds exist, score is fraction of how far
        above min threshold. Otherwise score is 1.0.
        """
        if not config.min_payoff:
            return 1.0

        scores: list[float] = []
        for pid, threshold in config.min_payoff.items():
            actual = avg_payoffs.get(pid, 0.0)
            if threshold == 0:
                scores.append(1.0 if actual >= 0 else 0.0)
            else:
                ratio = actual / abs(threshold)
                scores.append(min(1.0, max(0.0, ratio)))

        return sum(scores) / len(scores) if scores else 1.0

    def _check_payoff_distribution(
        self,
        result: GameResult,
    ) -> EvalCheck:
        """Compute payoff distribution statistics."""
        per_player: dict[str, dict[str, float]] = {}

        for pid in result.average_payoffs:
            payoffs = [ep.payoffs.get(pid, 0.0) for ep in result.episodes]
            if payoffs:
                sorted_p = sorted(payoffs)
                n = len(sorted_p)
                per_player[pid] = {
                    "min": sorted_p[0],
                    "max": sorted_p[-1],
                    "median": sorted_p[n // 2],
                    "mean": sum(sorted_p) / n,
                    "p25": sorted_p[max(0, n // 4)],
                    "p75": sorted_p[min(n - 1, 3 * n // 4)],
                }

        return EvalCheck(
            name="payoff_distribution",
            passed=True,
            score=1.0,
            message="Payoff distribution computed",
            details={"per_player": per_player},
        )

    def _check_social_welfare(
        self,
        result: GameResult,
        config: PayoffConfig,
    ) -> EvalCheck:
        """Check social welfare (sum of average payoffs)."""
        avg = result.average_payoffs
        welfare = sum(avg.values())
        details: dict[str, Any] = {"social_welfare": welfare}

        if config.min_social_welfare is not None:
            passed = welfare >= config.min_social_welfare
            details["threshold"] = config.min_social_welfare
            score = min(
                1.0,
                max(
                    0.0,
                    welfare / config.min_social_welfare
                    if config.min_social_welfare != 0
                    else (1.0 if welfare >= 0 else 0.0),
                ),
            )
            return EvalCheck(
                name="social_welfare",
                passed=passed,
                score=score,
                message=(
                    f"Social welfare: {welfare:.3f}"
                    + (f" (threshold: {config.min_social_welfare:.3f})")
                ),
                details=details,
            )

        return EvalCheck(
            name="social_welfare",
            passed=True,
            score=1.0,
            message=f"Social welfare: {welfare:.3f}",
            details=details,
        )

    def _check_pareto_efficiency(
        self,
        result: GameResult,
    ) -> EvalCheck:
        """Check if the outcome is Pareto efficient.

        An outcome is Pareto efficient if no player can
        improve without making another worse off. We check
        against all observed episode outcomes.
        """
        avg = result.average_payoffs
        players = list(avg.keys())

        if len(result.episodes) < 2:
            return EvalCheck(
                name="pareto_efficiency",
                passed=True,
                score=1.0,
                message=("Pareto check skipped: insufficient episodes"),
                details={"reason": "need >=2 episodes"},
            )

        # Collect all outcome vectors
        outcomes: list[dict[str, float]] = [ep.payoffs for ep in result.episodes]

        # Check if avg outcome is Pareto dominated
        is_dominated = False
        dominator = None
        for outcome in outcomes:
            all_ge = all(outcome.get(p, 0.0) >= avg.get(p, 0.0) for p in players)
            any_gt = any(outcome.get(p, 0.0) > avg.get(p, 0.0) for p in players)
            if all_ge and any_gt:
                is_dominated = True
                dominator = outcome
                break

        if is_dominated:
            return EvalCheck(
                name="pareto_efficiency",
                passed=False,
                score=0.0,
                message="Average outcome is Pareto dominated",
                details={
                    "average": avg,
                    "dominating_outcome": dominator,
                },
            )

        return EvalCheck(
            name="pareto_efficiency",
            passed=True,
            score=1.0,
            message="Average outcome is Pareto efficient",
            details={"average": avg},
        )
