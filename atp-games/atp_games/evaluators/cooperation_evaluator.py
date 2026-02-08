"""Cooperation evaluator for game-theoretic assessment.

Measures cooperative behavior patterns in game play including
cooperation rate, conditional cooperation, and reciprocity.
Wraps game_envs.analysis.cooperation metrics into the ATP
evaluator interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atp.evaluators.base import EvalCheck, EvalResult, Evaluator
from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse
from game_envs.analysis.cooperation import (
    conditional_cooperation,
    cooperation_rate,
    reciprocity_index,
)
from game_envs.core.state import RoundResult

from atp_games.models import GameResult


@dataclass(frozen=True)
class CooperationConfig:
    """Configuration for CooperationEvaluator.

    Attributes:
        min_cooperation_rate: Minimum cooperation rate per
            player for pass (None = no threshold).
        max_cooperation_rate: Maximum cooperation rate per
            player (None = no threshold).
        min_reciprocity: Minimum reciprocity index threshold
            (None = no threshold).
        cooperative_actions: Set of action strings considered
            cooperative. Defaults to {"cooperate", "c"}.
        weights: Per-check weights for scoring.
    """

    min_cooperation_rate: dict[str, float] | None = None
    max_cooperation_rate: dict[str, float] | None = None
    min_reciprocity: float | None = None
    cooperative_actions: frozenset[str] | None = None
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "cooperation_rate": 1.0,
            "conditional_cooperation": 1.0,
            "reciprocity": 1.0,
        },
    )


class CooperationEvaluator(Evaluator):
    """Evaluates cooperative behavior in game results.

    Computes:
    - Cooperation rate per player
    - Conditional cooperation: P(C|C) and P(C|D)
    - Reciprocity index (correlation of cooperation)

    Compatible with ATP evaluator registry and scoring
    pipeline via the standard Evaluator interface.
    """

    def __init__(
        self,
        config: CooperationConfig | None = None,
    ) -> None:
        self._config = config or CooperationConfig()

    @property
    def name(self) -> str:
        """Return evaluator name."""
        return "cooperation"

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
                        name="cooperation_data",
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
        """Evaluate cooperation metrics for a GameResult.

        Args:
            result: Completed game result with action history.
            config: Optional override config dict with keys:
                min_cooperation_rate, max_cooperation_rate,
                min_reciprocity.

        Returns:
            EvalResult with cooperation checks.
        """
        eval_config = self._resolve_config(config)
        checks: list[EvalCheck] = []

        if not result.episodes:
            return self._create_result(
                [
                    self._create_check(
                        name="cooperation_data",
                        passed=False,
                        message="No episodes to evaluate",
                    ),
                ]
            )

        history = self._collect_history(result)

        if not history:
            return self._create_result(
                [
                    self._create_check(
                        name="cooperation_data",
                        passed=False,
                        message="No action history available",
                    ),
                ]
            )

        players = self._extract_players(history)
        if len(players) < 2:
            return self._create_result(
                [
                    self._create_check(
                        name="cooperation_data",
                        passed=False,
                        message=(
                            "Need at least 2 players for "
                            f"cooperation analysis, got {len(players)}"
                        ),
                    ),
                ]
            )

        # 1. Cooperation rate per player
        checks.append(self._check_cooperation_rate(history, players, eval_config))

        # 2. Conditional cooperation (requires >= 2 rounds)
        checks.append(self._check_conditional_cooperation(history, players))

        # 3. Reciprocity index
        checks.append(self._check_reciprocity(history, players, eval_config))

        return self._create_result(checks)

    def _resolve_config(
        self,
        override: dict[str, Any] | None,
    ) -> CooperationConfig:
        """Merge override dict into default config."""
        if override is None:
            return self._config
        return CooperationConfig(
            min_cooperation_rate=override.get(
                "min_cooperation_rate",
                self._config.min_cooperation_rate,
            ),
            max_cooperation_rate=override.get(
                "max_cooperation_rate",
                self._config.max_cooperation_rate,
            ),
            min_reciprocity=override.get(
                "min_reciprocity",
                self._config.min_reciprocity,
            ),
            cooperative_actions=override.get(
                "cooperative_actions",
                self._config.cooperative_actions,
            ),
            weights=override.get(
                "weights",
                self._config.weights,
            ),
        )

    def _collect_history(
        self,
        result: GameResult,
    ) -> list[RoundResult]:
        """Collect RoundResult objects from all episodes."""
        history: list[RoundResult] = []
        for ep in result.episodes:
            for step in ep.actions_log:
                if isinstance(step, dict) and "actions" in step:
                    history.append(
                        RoundResult(
                            round_number=step.get("round_number", len(history)),
                            actions=step["actions"],
                            payoffs=step.get("payoffs", {}),
                        )
                    )
                elif isinstance(step, dict):
                    history.append(
                        RoundResult(
                            round_number=len(history),
                            actions=step,
                            payoffs={},
                        )
                    )
        return history

    def _extract_players(
        self,
        history: list[RoundResult],
    ) -> list[str]:
        """Extract sorted player IDs from history."""
        players: set[str] = set()
        for rr in history:
            players.update(rr.actions.keys())
        return sorted(players)

    def _check_cooperation_rate(
        self,
        history: list[RoundResult],
        players: list[str],
        config: CooperationConfig,
    ) -> EvalCheck:
        """Check cooperation rate per player against thresholds."""
        rates: dict[str, float] = {}
        for pid in players:
            try:
                rates[pid] = cooperation_rate(history, pid)
            except ValueError:
                rates[pid] = 0.0

        details: dict[str, Any] = {"cooperation_rates": rates}
        failures: list[str] = []

        if config.min_cooperation_rate:
            for pid, threshold in config.min_cooperation_rate.items():
                actual = rates.get(pid, 0.0)
                if actual < threshold:
                    failures.append(f"{pid}: {actual:.3f} < {threshold:.3f}")
            details["min_thresholds"] = dict(config.min_cooperation_rate)

        if config.max_cooperation_rate:
            for pid, threshold in config.max_cooperation_rate.items():
                actual = rates.get(pid, 0.0)
                if actual > threshold:
                    failures.append(f"{pid}: {actual:.3f} > {threshold:.3f}")
            details["max_thresholds"] = dict(config.max_cooperation_rate)

        if failures:
            return EvalCheck(
                name="cooperation_rate",
                passed=False,
                score=0.0,
                message=("Cooperation rate violations: " + "; ".join(failures)),
                details=details,
            )

        # Score: average cooperation rate across players
        avg_rate = sum(rates.values()) / len(rates) if rates else 0.0
        return EvalCheck(
            name="cooperation_rate",
            passed=True,
            score=avg_rate,
            message=(
                "Cooperation rates: "
                + ", ".join(f"{pid}={v:.3f}" for pid, v in rates.items())
            ),
            details=details,
        )

    def _check_conditional_cooperation(
        self,
        history: list[RoundResult],
        players: list[str],
    ) -> EvalCheck:
        """Compute conditional cooperation for each player."""
        per_player: dict[str, dict[str, float | None]] = {}

        for pid in players:
            opponent = next((p for p in players if p != pid), None)
            try:
                cond = conditional_cooperation(history, pid, opponent)
                per_player[pid] = cond
            except ValueError:
                per_player[pid] = {
                    "prob_c_given_c": None,
                    "prob_c_given_d": None,
                }

        return EvalCheck(
            name="conditional_cooperation",
            passed=True,
            score=1.0,
            message="Conditional cooperation computed",
            details={"per_player": per_player},
        )

    def _check_reciprocity(
        self,
        history: list[RoundResult],
        players: list[str],
        config: CooperationConfig,
    ) -> EvalCheck:
        """Check reciprocity index against threshold."""
        try:
            recip = reciprocity_index(history, players[0], players[1])
        except ValueError:
            return EvalCheck(
                name="reciprocity",
                passed=True,
                score=1.0,
                message="Reciprocity skipped: insufficient data",
                details={"reason": "insufficient data"},
            )

        details: dict[str, Any] = {"reciprocity_index": recip}

        if config.min_reciprocity is not None:
            passed = recip >= config.min_reciprocity
            details["threshold"] = config.min_reciprocity
            # Score based on how close to threshold
            score = min(
                1.0,
                max(0.0, (recip + 1.0) / 2.0),
            )
            return EvalCheck(
                name="reciprocity",
                passed=passed,
                score=score,
                message=(
                    f"Reciprocity: {recip:.4f}"
                    f" (threshold: {config.min_reciprocity:.4f})"
                ),
                details=details,
            )

        # No threshold: normalize to [0, 1] for score
        score = min(1.0, max(0.0, (recip + 1.0) / 2.0))
        return EvalCheck(
            name="reciprocity",
            passed=True,
            score=score,
            message=f"Reciprocity: {recip:.4f}",
            details=details,
        )
