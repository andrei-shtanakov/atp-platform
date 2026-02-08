"""Adversarial stress-test: agent vs best-response oracle.

Tests an agent's robustness by computing its empirical strategy,
generating a best-response opponent, and measuring exploitability
in practice.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from atp.adapters.base import AgentAdapter
from game_envs.analysis.exploitability import (
    EmpiricalStrategy,
    ExploitabilityResult,
    compute_best_response,
    compute_exploitability_from_game,
)
from game_envs.core.action import DiscreteActionSpace
from game_envs.core.game import Game
from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy

from atp_games.models import GameResult, GameRunConfig
from atp_games.runner.builtin_adapter import BuiltinAdapter
from atp_games.runner.game_runner import GameRunner

logger = logging.getLogger(__name__)


class BestResponseStrategy(Strategy):
    """Strategy that plays the best response to an opponent.

    Given payoff matrices and an opponent's empirical strategy,
    plays the pure-strategy best response at every round.

    Attributes:
        _best_action: The best response action name.
        _action_names: Ordered action name list.
    """

    def __init__(
        self,
        best_action: str,
        action_names: list[str],
    ) -> None:
        self._best_action = best_action
        self._action_names = action_names

    @property
    def name(self) -> str:
        """Strategy name."""
        return f"best_response({self._best_action})"

    def choose_action(self, observation: Observation) -> str:
        """Always play the best response action."""
        return self._best_action

    def reset(self) -> None:
        """No state to reset."""


@dataclass
class StressTestIteration:
    """Result of one iteration of the stress test.

    Attributes:
        iteration: Iteration number.
        empirical_strategy: Agent's empirical strategy.
        best_response_action: Best response action name.
        game_result: Result of agent vs best response.
        exploitability: Exploitability measurement.
    """

    iteration: int
    empirical_strategy: EmpiricalStrategy
    best_response_action: str
    game_result: GameResult
    exploitability: ExploitabilityResult

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "iteration": self.iteration,
            "empirical_strategy": self.empirical_strategy.to_dict(),
            "best_response_action": self.best_response_action,
            "exploitability": self.exploitability.to_dict(),
            "game_result_summary": {
                "average_payoffs": self.game_result.average_payoffs,
            },
        }


@dataclass
class StressTestResult:
    """Result of a complete adversarial stress test.

    Attributes:
        agent_name: Name of the tested agent.
        iterations: Per-iteration results.
        final_exploitability: Final exploitability score.
        passed: Whether the agent passed (below threshold).
        threshold: Exploitability threshold used.
    """

    agent_name: str
    iterations: list[StressTestIteration]
    final_exploitability: float
    passed: bool
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_name": self.agent_name,
            "iterations": [it.to_dict() for it in self.iterations],
            "final_exploitability": round(self.final_exploitability, 4),
            "passed": self.passed,
            "threshold": self.threshold,
        }


def _build_best_response_strategy(
    game: Game,
    agent_player_id: str,
    opponent_player_id: str,
    empirical_strategy: EmpiricalStrategy,
) -> BestResponseStrategy:
    """Build a best-response strategy from empirical data.

    Args:
        game: The game (2-player, discrete actions).
        agent_player_id: The agent under test (opponent role).
        opponent_player_id: The BR player role.
        empirical_strategy: Agent's observed strategy.

    Returns:
        BestResponseStrategy that exploits the agent.

    Raises:
        ValueError: If game is not suitable.
    """
    space_agent = game.action_space(agent_player_id)
    space_opp = game.action_space(opponent_player_id)

    if not isinstance(space_agent, DiscreteActionSpace):
        raise ValueError("Stress test requires DiscreteActionSpace")
    if not isinstance(space_opp, DiscreteActionSpace):
        raise ValueError("Stress test requires DiscreteActionSpace")

    actions_agent = space_agent.to_list()
    actions_opp = space_opp.to_list()
    m = len(actions_agent)
    n = len(actions_opp)

    # Build payoff matrix for opponent (BR player)
    payoff_opp = np.zeros((n, m))

    for i, a_opp in enumerate(actions_opp):
        for j, a_agent in enumerate(actions_agent):
            game_copy = copy.deepcopy(game)
            game_copy.reset()
            result = game_copy.step(
                {agent_player_id: a_agent, opponent_player_id: a_opp}
            )
            payoff_opp[i, j] = result.payoffs[opponent_player_id]

    # Convert agent's empirical strategy to probability vector
    agent_probs = empirical_strategy.to_array([str(a) for a in actions_agent])

    # Find best response
    br_idx, _ = compute_best_response(
        payoff_opp,
        agent_probs,
        player_is_row=True,
    )
    best_action = str(actions_opp[br_idx])

    return BestResponseStrategy(
        best_action=best_action,
        action_names=[str(a) for a in actions_opp],
    )


def _extract_empirical_strategy(
    game_result: GameResult,
    player_id: str,
) -> EmpiricalStrategy:
    """Extract empirical strategy from game result history.

    Args:
        game_result: Result containing episode histories.
        player_id: Player whose strategy to extract.

    Returns:
        EmpiricalStrategy with action frequencies.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    total = 0

    for episode in game_result.episodes:
        for actions_round in episode.actions_log:
            if player_id in actions_round:
                counts[str(actions_round[player_id])] += 1
                total += 1

    if total == 0:
        raise ValueError(f"No actions found for player '{player_id}'")

    frequencies = {action: count / total for action, count in counts.items()}
    return EmpiricalStrategy(action_frequencies=frequencies)


async def run_stress_test(
    game: Game,
    agent_name: str,
    agent: AgentAdapter,
    config: GameRunConfig | None = None,
    runner: GameRunner | None = None,
    threshold: float = 0.15,
    iterations: int = 1,
    profiling_episodes: int | None = None,
) -> StressTestResult:
    """Run adversarial stress test against an agent.

    Process:
    1. Profile the agent by playing it against a random/simple
       opponent for profiling_episodes.
    2. Extract empirical strategy.
    3. Compute best response to that strategy.
    4. Play agent vs best response.
    5. Measure exploitability.
    6. Optionally repeat (iterative: agent may adapt).

    Args:
        game: Game template (2-player, discrete actions).
        agent_name: Name of agent under test.
        agent: Agent adapter.
        config: Run configuration per match.
        runner: Optional GameRunner.
        threshold: Exploitability threshold for passing.
        iterations: Number of profile-exploit iterations.
        profiling_episodes: Episodes for profiling phase.
            Defaults to config.episodes if not set.

    Returns:
        StressTestResult with exploitability data.
    """
    config = config or GameRunConfig(episodes=10)
    runner = runner or GameRunner()
    profiling_episodes = profiling_episodes or config.episodes

    players = game.player_ids
    if len(players) != 2:
        raise ValueError("Stress test requires exactly 2 players")

    agent_pid = players[0]
    opp_pid = players[1]

    iteration_results: list[StressTestIteration] = []

    # Create a simple profiling opponent (random/default strategy)
    from game_envs import StrategyRegistry

    # Try to get a random strategy, fall back to first available
    try:
        profiling_strategy = StrategyRegistry.create("random")
    except KeyError:
        available = StrategyRegistry.list_strategies()
        if not available:
            raise ValueError("No strategies available for profiling")
        profiling_strategy = StrategyRegistry.create(available[0])

    profiling_adapter: AgentAdapter = BuiltinAdapter(strategy=profiling_strategy)
    current_opponent: AgentAdapter = profiling_adapter

    for iteration in range(iterations):
        logger.info(
            "Stress test iteration %d/%d for agent '%s'",
            iteration + 1,
            iterations,
            agent_name,
        )

        # Phase 1: Profile the agent
        profiling_config = GameRunConfig(
            episodes=profiling_episodes,
            max_retries=config.max_retries,
            move_timeout=config.move_timeout,
            base_seed=config.base_seed,
        )
        profiling_game = copy.deepcopy(game)
        profiling_agents = {
            agent_pid: copy.deepcopy(agent),
            opp_pid: copy.deepcopy(current_opponent),
        }
        profiling_result = await runner.run_game(
            profiling_game, profiling_agents, profiling_config
        )

        # Phase 2: Extract empirical strategy
        empirical = _extract_empirical_strategy(profiling_result, agent_pid)
        logger.info(
            "Agent empirical strategy: %s",
            empirical.action_frequencies,
        )

        # Phase 3: Build best response
        br_strategy = _build_best_response_strategy(
            game=copy.deepcopy(game),
            agent_player_id=agent_pid,
            opponent_player_id=opp_pid,
            empirical_strategy=empirical,
        )
        br_adapter = BuiltinAdapter(strategy=br_strategy)

        # Phase 4: Play agent vs best response
        exploit_game = copy.deepcopy(game)
        exploit_agents = {
            agent_pid: copy.deepcopy(agent),
            opp_pid: br_adapter,
        }
        exploit_result = await runner.run_game(exploit_game, exploit_agents, config)

        # Phase 5: Compute exploitability
        exploit_empirical = _extract_empirical_strategy(exploit_result, agent_pid)
        br_empirical = _extract_empirical_strategy(exploit_result, opp_pid)
        exploitability = compute_exploitability_from_game(
            game=copy.deepcopy(game),
            empirical_strategies={
                agent_pid: exploit_empirical,
                opp_pid: br_empirical,
            },
        )

        iteration_results.append(
            StressTestIteration(
                iteration=iteration,
                empirical_strategy=empirical,
                best_response_action=br_strategy._best_action,
                game_result=exploit_result,
                exploitability=exploitability,
            )
        )

        # For iterative: use BR as the next opponent
        current_opponent = br_adapter

    final_exploit = (
        iteration_results[-1].exploitability.total if iteration_results else 0.0
    )

    return StressTestResult(
        agent_name=agent_name,
        iterations=iteration_results,
        final_exploitability=final_exploit,
        passed=final_exploit <= threshold,
        threshold=threshold,
    )
