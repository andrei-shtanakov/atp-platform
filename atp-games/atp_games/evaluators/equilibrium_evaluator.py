"""Equilibrium evaluator for game-theoretic assessment.

Measures how close an agent's empirical strategy is to Nash
equilibrium by computing the distance between observed play
and known equilibria, and detecting convergence patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from atp.evaluators.base import EvalCheck, EvalResult, Evaluator
from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse
from game_envs.analysis.exploitability import EmpiricalStrategy
from game_envs.analysis.models import NashEquilibrium
from game_envs.analysis.nash_solver import NashSolver
from game_envs.core.state import RoundResult

from atp_games.models import GameResult


@dataclass(frozen=True)
class EquilibriumConfig:
    """Configuration for EquilibriumEvaluator.

    Attributes:
        max_nash_distance: Maximum L1 distance from nearest
            Nash equilibrium for pass (None = no threshold).
        convergence_window: Number of recent rounds to use
            for convergence detection. Compares strategy in
            first half vs second half of the window.
        convergence_threshold: Maximum strategy change between
            window halves to consider converged.
        payoff_matrix_1: Payoff matrix for player 1.
        payoff_matrix_2: Payoff matrix for player 2.
        action_names_1: Action labels for player 1.
        action_names_2: Action labels for player 2.
        solver_method: Nash solver method to use.
    """

    max_nash_distance: float | None = None
    convergence_window: int = 20
    convergence_threshold: float = 0.1
    payoff_matrix_1: list[list[float]] | None = None
    payoff_matrix_2: list[list[float]] | None = None
    action_names_1: list[str] | None = None
    action_names_2: list[str] | None = None
    solver_method: str = "support_enumeration"


class EquilibriumEvaluator(Evaluator):
    """Evaluates proximity to Nash equilibrium.

    Computes:
    - Nash equilibria of the game
    - L1 distance from empirical strategy to nearest NE
    - Convergence detection over time
    - Equilibrium classification (pure/mixed)

    Compatible with ATP evaluator registry and scoring
    pipeline via the standard Evaluator interface.
    """

    def __init__(
        self,
        config: EquilibriumConfig | None = None,
    ) -> None:
        self._config = config or EquilibriumConfig()

    @property
    def name(self) -> str:
        """Return evaluator name."""
        return "equilibrium"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Evaluate via ATP evaluator interface.

        Extracts GameResult and payoff matrices from
        assertion config.
        """
        config = assertion.config
        game_result_data = config.get("game_result")
        if game_result_data is None:
            return self._create_result(
                [
                    self._create_check(
                        name="equilibrium_data",
                        passed=False,
                        message=("No game_result in assertion config"),
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
        """Evaluate equilibrium proximity for a GameResult.

        Args:
            result: Completed game result with action history.
            config: Config dict with keys: max_nash_distance,
                payoff_matrix_1, payoff_matrix_2,
                action_names_1, action_names_2.

        Returns:
            EvalResult with equilibrium checks.
        """
        eval_config = self._resolve_config(config)
        checks: list[EvalCheck] = []

        if not result.episodes:
            return self._create_result(
                [
                    self._create_check(
                        name="equilibrium_data",
                        passed=False,
                        message="No episodes to evaluate",
                    ),
                ]
            )

        if eval_config.payoff_matrix_1 is None or eval_config.payoff_matrix_2 is None:
            return self._create_result(
                [
                    self._create_check(
                        name="equilibrium_data",
                        passed=False,
                        message=("Payoff matrices required for equilibrium analysis"),
                    ),
                ]
            )

        history = self._collect_history(result)
        players = list(result.average_payoffs.keys())

        if len(players) != 2:
            return self._create_result(
                [
                    self._create_check(
                        name="equilibrium_data",
                        passed=False,
                        message=(
                            "Equilibrium analysis requires exactly "
                            f"2 players, got {len(players)}"
                        ),
                    ),
                ]
            )

        if not history:
            return self._create_result(
                [
                    self._create_check(
                        name="equilibrium_data",
                        passed=False,
                        message="No action history available",
                    ),
                ]
            )

        p0, p1 = players[0], players[1]

        try:
            emp_0 = EmpiricalStrategy.from_history(history, p0)
            emp_1 = EmpiricalStrategy.from_history(history, p1)
        except ValueError as e:
            return self._create_result(
                [
                    self._create_check(
                        name="equilibrium_data",
                        passed=False,
                        message=f"Cannot extract strategy: {e}",
                    ),
                ]
            )

        payoff_1 = np.array(eval_config.payoff_matrix_1)
        payoff_2 = np.array(eval_config.payoff_matrix_2)

        actions_1 = eval_config.action_names_1 or [
            str(i) for i in range(payoff_1.shape[0])
        ]
        actions_2 = eval_config.action_names_2 or [
            str(i) for i in range(payoff_1.shape[1])
        ]

        # Compute Nash equilibria
        try:
            equilibria = NashSolver.solve_2player(
                payoff_1,
                payoff_2,
                method=eval_config.solver_method,
            )
        except Exception as e:
            return self._create_result(
                [
                    self._create_check(
                        name="equilibrium_data",
                        passed=False,
                        message=f"Nash solver failed: {e}",
                    ),
                ]
            )

        if not equilibria:
            return self._create_result(
                [
                    self._create_check(
                        name="equilibrium_data",
                        passed=False,
                        message="No Nash equilibria found",
                    ),
                ]
            )

        strat_0 = emp_0.to_array(actions_1)
        strat_1 = emp_1.to_array(actions_2)

        # 1. Nash distance check
        checks.append(
            self._check_nash_distance(
                strat_0,
                strat_1,
                equilibria,
                eval_config,
                p0,
                p1,
            )
        )

        # 2. Equilibrium classification
        checks.append(self._check_equilibrium_type(equilibria))

        # 3. Convergence detection
        checks.append(
            self._check_convergence(
                history,
                p0,
                p1,
                actions_1,
                actions_2,
                eval_config,
            )
        )

        return self._create_result(checks)

    def _resolve_config(
        self,
        override: dict[str, Any] | None,
    ) -> EquilibriumConfig:
        """Merge override dict into default config."""
        if override is None:
            return self._config
        return EquilibriumConfig(
            max_nash_distance=override.get(
                "max_nash_distance",
                self._config.max_nash_distance,
            ),
            convergence_window=override.get(
                "convergence_window",
                self._config.convergence_window,
            ),
            convergence_threshold=override.get(
                "convergence_threshold",
                self._config.convergence_threshold,
            ),
            payoff_matrix_1=override.get(
                "payoff_matrix_1",
                self._config.payoff_matrix_1,
            ),
            payoff_matrix_2=override.get(
                "payoff_matrix_2",
                self._config.payoff_matrix_2,
            ),
            action_names_1=override.get(
                "action_names_1",
                self._config.action_names_1,
            ),
            action_names_2=override.get(
                "action_names_2",
                self._config.action_names_2,
            ),
            solver_method=override.get(
                "solver_method",
                self._config.solver_method,
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

    def _check_nash_distance(
        self,
        strat_0: np.ndarray,
        strat_1: np.ndarray,
        equilibria: list[NashEquilibrium],
        config: EquilibriumConfig,
        p0: str,
        p1: str,
    ) -> EvalCheck:
        """Check L1 distance to nearest Nash equilibrium."""
        min_distance = float("inf")
        nearest_ne: NashEquilibrium | None = None

        for ne in equilibria:
            ne_s0 = ne.strategies.get("player_0", np.zeros_like(strat_0))
            ne_s1 = ne.strategies.get("player_1", np.zeros_like(strat_1))
            dist = float(
                np.sum(np.abs(strat_0 - ne_s0)) + np.sum(np.abs(strat_1 - ne_s1))
            )
            if dist < min_distance:
                min_distance = dist
                nearest_ne = ne

        details: dict[str, Any] = {
            "min_nash_distance": min_distance,
            "num_equilibria": len(equilibria),
            "empirical_strategy": {
                p0: strat_0.tolist(),
                p1: strat_1.tolist(),
            },
        }
        if nearest_ne is not None:
            details["nearest_equilibrium"] = nearest_ne.to_dict()

        if config.max_nash_distance is not None:
            passed = min_distance <= config.max_nash_distance
            # Score: exponential decay based on distance
            score = max(
                0.0,
                min(
                    1.0,
                    1.0 - min_distance / (2 * config.max_nash_distance)
                    if config.max_nash_distance > 0
                    else (1.0 if min_distance == 0 else 0.0),
                ),
            )
            return EvalCheck(
                name="nash_distance",
                passed=passed,
                score=score,
                message=(
                    f"Nash distance: {min_distance:.4f}"
                    f" (threshold: {config.max_nash_distance:.4f})"
                ),
                details=details,
            )

        # No threshold — report distance, always pass
        # Score: 1.0 when at NE, decrease as distance grows
        score = max(0.0, min(1.0, 1.0 / (1.0 + min_distance)))
        return EvalCheck(
            name="nash_distance",
            passed=True,
            score=score,
            message=f"Nash distance: {min_distance:.4f}",
            details=details,
        )

    def _check_equilibrium_type(
        self,
        equilibria: list[NashEquilibrium],
    ) -> EvalCheck:
        """Classify the computed Nash equilibria."""
        pure_count = sum(1 for ne in equilibria if ne.is_pure())
        mixed_count = sum(1 for ne in equilibria if ne.is_mixed())

        eq_summaries: list[dict[str, Any]] = []
        for ne in equilibria:
            eq_summaries.append(
                {
                    "type": "pure" if ne.is_pure() else "mixed",
                    "payoffs": ne.payoffs,
                    "support": {k: list(v) for k, v in ne.support.items()},
                }
            )

        return EvalCheck(
            name="equilibrium_type",
            passed=True,
            score=1.0,
            message=(
                f"Found {len(equilibria)} equilibria: "
                f"{pure_count} pure, {mixed_count} mixed"
            ),
            details={
                "equilibria": eq_summaries,
                "pure_count": pure_count,
                "mixed_count": mixed_count,
            },
        )

    def _check_convergence(
        self,
        history: list[RoundResult],
        p0: str,
        p1: str,
        actions_1: list[str],
        actions_2: list[str],
        config: EquilibriumConfig,
    ) -> EvalCheck:
        """Detect convergence in strategy over time.

        Splits the history into two halves and compares the
        empirical strategies. If the L1 distance between the
        two halves is below the threshold, the strategy is
        considered converged.
        """
        window = min(config.convergence_window, len(history))
        if window < 4:
            return EvalCheck(
                name="convergence",
                passed=True,
                score=1.0,
                message=("Convergence check skipped: insufficient rounds"),
                details={
                    "reason": "need >= 4 rounds",
                    "rounds": len(history),
                },
            )

        recent = history[-window:]
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]

        try:
            emp_0_first = EmpiricalStrategy.from_history(first_half, p0)
            emp_0_second = EmpiricalStrategy.from_history(second_half, p0)
            emp_1_first = EmpiricalStrategy.from_history(first_half, p1)
            emp_1_second = EmpiricalStrategy.from_history(second_half, p1)
        except ValueError:
            return EvalCheck(
                name="convergence",
                passed=True,
                score=1.0,
                message="Convergence check skipped: extraction failed",
                details={"reason": "strategy extraction failed"},
            )

        s0_first = emp_0_first.to_array(actions_1)
        s0_second = emp_0_second.to_array(actions_1)
        s1_first = emp_1_first.to_array(actions_2)
        s1_second = emp_1_second.to_array(actions_2)

        change_0 = float(np.sum(np.abs(s0_second - s0_first)))
        change_1 = float(np.sum(np.abs(s1_second - s1_first)))
        max_change = max(change_0, change_1)

        converged = max_change <= config.convergence_threshold
        score = max(
            0.0,
            min(
                1.0,
                1.0 - max_change / (2 * config.convergence_threshold)
                if config.convergence_threshold > 0
                else (1.0 if max_change == 0 else 0.0),
            ),
        )

        return EvalCheck(
            name="convergence",
            passed=converged,
            score=score,
            message=(
                f"Strategy change: {max_change:.4f}"
                f" (threshold: {config.convergence_threshold:.4f})"
                f" — {'converged' if converged else 'not converged'}"
            ),
            details={
                "max_change": max_change,
                "change_player_0": change_0,
                "change_player_1": change_1,
                "converged": converged,
                "window_size": window,
                "threshold": config.convergence_threshold,
            },
        )
