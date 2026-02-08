"""Cross-play matrix: all agents vs all agents.

Runs every agent pair (including self-play) and produces
a payoff matrix with dominance and clustering analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from atp.adapters.base import AgentAdapter
from game_envs.core.game import Game

from atp_games.models import GameResult, GameRunConfig
from atp_games.runner.game_runner import GameRunner
from atp_games.suites.tournament import _run_match

logger = logging.getLogger(__name__)


@dataclass
class CrossPlayEntry:
    """Entry in the cross-play matrix.

    Attributes:
        agent_row: Row agent (player 0).
        agent_col: Column agent (player 1).
        payoff_row: Average payoff for row agent.
        payoff_col: Average payoff for column agent.
        game_result: Full game result.
    """

    agent_row: str
    agent_col: str
    payoff_row: float
    payoff_col: float
    game_result: GameResult

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_row": self.agent_row,
            "agent_col": self.agent_col,
            "payoff_row": round(self.payoff_row, 4),
            "payoff_col": round(self.payoff_col, 4),
        }


@dataclass
class DominanceRelation:
    """Dominance relationship between two agents.

    Attributes:
        dominator: Agent that dominates.
        dominated: Agent that is dominated.
        strict: Whether dominance is strict (all payoffs better).
    """

    dominator: str
    dominated: str
    strict: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "dominator": self.dominator,
            "dominated": self.dominated,
            "strict": self.strict,
        }


@dataclass
class CrossPlayResult:
    """Result of cross-play matrix computation.

    Attributes:
        agents: Ordered list of agent names.
        matrix: 2D payoff matrix [row_agent][col_agent].
        entries: All pairwise entries with full results.
        dominance: Dominance relationships found.
        pareto_frontier: Agents on the Pareto frontier.
        clusters: Agent clusters by similar play.
    """

    agents: list[str]
    matrix: dict[str, dict[str, float]]
    entries: list[CrossPlayEntry] = field(default_factory=list)
    dominance: list[DominanceRelation] = field(default_factory=list)
    pareto_frontier: list[str] = field(default_factory=list)
    clusters: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agents": self.agents,
            "matrix": {
                r: {c: round(v, 4) for c, v in cols.items()}
                for r, cols in self.matrix.items()
            },
            "entries": [e.to_dict() for e in self.entries],
            "dominance": [d.to_dict() for d in self.dominance],
            "pareto_frontier": self.pareto_frontier,
            "clusters": self.clusters,
        }


async def run_cross_play(
    game: Game,
    agents: dict[str, AgentAdapter],
    config: GameRunConfig | None = None,
    runner: GameRunner | None = None,
    include_self_play: bool = True,
) -> CrossPlayResult:
    """Run cross-play matrix: every agent vs every agent.

    Args:
        game: Game template.
        agents: Agent name to adapter mapping.
        config: Run configuration per matchup.
        runner: Optional GameRunner.
        include_self_play: Whether to include agent vs itself.

    Returns:
        CrossPlayResult with payoff matrix and analysis.
    """
    config = config or GameRunConfig()
    runner = runner or GameRunner()
    agent_names = list(agents.keys())

    matrix: dict[str, dict[str, float]] = {a: {} for a in agent_names}
    entries: list[CrossPlayEntry] = []

    total_pairs = (
        len(agent_names) ** 2
        if include_self_play
        else (len(agent_names) * (len(agent_names) - 1))
    )
    logger.info(
        "Cross-play matrix: %d agents, %d matchups",
        len(agent_names),
        total_pairs,
    )

    for row_agent in agent_names:
        for col_agent in agent_names:
            if not include_self_play and row_agent == col_agent:
                matrix[row_agent][col_agent] = 0.0
                continue

            logger.info("Cross-play: %s vs %s", row_agent, col_agent)
            match = await _run_match(
                runner=runner,
                game=game,
                agent_a_name=row_agent,
                agent_a=agents[row_agent],
                agent_b_name=col_agent,
                agent_b=agents[col_agent],
                config=config,
            )

            entry = CrossPlayEntry(
                agent_row=row_agent,
                agent_col=col_agent,
                payoff_row=match.score_a,
                payoff_col=match.score_b,
                game_result=match.game_result,
            )
            entries.append(entry)
            matrix[row_agent][col_agent] = match.score_a

    # Compute dominance relationships
    dominance = _compute_dominance(agent_names, matrix)

    # Compute Pareto frontier
    pareto = _compute_pareto_frontier(agent_names, matrix)

    # Compute clusters
    clusters = _compute_clusters(agent_names, matrix)

    return CrossPlayResult(
        agents=agent_names,
        matrix=matrix,
        entries=entries,
        dominance=dominance,
        pareto_frontier=pareto,
        clusters=clusters,
    )


def _compute_dominance(
    agents: list[str],
    matrix: dict[str, dict[str, float]],
) -> list[DominanceRelation]:
    """Find dominance relationships in payoff matrix.

    Agent A dominates agent B if A's payoff against every
    opponent is at least as good as B's payoff against that
    same opponent (weak dominance), or strictly better
    (strict dominance).

    Args:
        agents: List of agent names.
        matrix: Payoff matrix.

    Returns:
        List of DominanceRelation objects.
    """
    relations: list[DominanceRelation] = []

    for a in agents:
        for b in agents:
            if a == b:
                continue

            # Check if a dominates b
            all_at_least = True
            all_strictly_better = True

            for opponent in agents:
                pa = matrix[a].get(opponent, 0.0)
                pb = matrix[b].get(opponent, 0.0)
                if pa < pb:
                    all_at_least = False
                    all_strictly_better = False
                    break
                if pa <= pb:
                    all_strictly_better = False

            if all_at_least:
                relations.append(
                    DominanceRelation(
                        dominator=a,
                        dominated=b,
                        strict=all_strictly_better,
                    )
                )

    return relations


def _compute_pareto_frontier(
    agents: list[str],
    matrix: dict[str, dict[str, float]],
) -> list[str]:
    """Find agents on the Pareto frontier.

    An agent is on the frontier if no other agent strictly
    dominates it.

    Args:
        agents: Agent names.
        matrix: Payoff matrix.

    Returns:
        List of agents on the Pareto frontier.
    """
    dominated: set[str] = set()

    for a in agents:
        for b in agents:
            if a == b:
                continue

            # Check if a strictly dominates b
            all_strictly = True
            for opponent in agents:
                if matrix[a].get(opponent, 0.0) <= matrix[b].get(opponent, 0.0):
                    all_strictly = False
                    break
            if all_strictly:
                dominated.add(b)

    return [a for a in agents if a not in dominated]


def _compute_clusters(
    agents: list[str],
    matrix: dict[str, dict[str, float]],
    threshold: float = 0.1,
) -> list[list[str]]:
    """Cluster agents by similarity of payoff profiles.

    Uses simple single-linkage: two agents are in the same
    cluster if their payoff vectors (row in matrix) differ
    by less than threshold (normalized).

    Args:
        agents: Agent names.
        matrix: Payoff matrix.
        threshold: Similarity threshold (relative difference).

    Returns:
        List of clusters (each is a list of agent names).
    """
    if not agents:
        return []

    # Build payoff vectors
    vectors: dict[str, list[float]] = {}
    for agent in agents:
        vectors[agent] = [matrix[agent].get(opp, 0.0) for opp in agents]

    # Simple union-find clustering
    parent: dict[str, str] = {a: a for a in agents}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, a in enumerate(agents):
        for j in range(i + 1, len(agents)):
            b = agents[j]
            # Compute normalized L1 distance
            diff = sum(abs(vectors[a][k] - vectors[b][k]) for k in range(len(agents)))
            max_val = max(
                sum(abs(v) for v in vectors[a]),
                sum(abs(v) for v in vectors[b]),
                1e-10,
            )
            if diff / max_val < threshold:
                union(a, b)

    # Group by root
    groups: dict[str, list[str]] = {}
    for agent in agents:
        root = find(agent)
        groups.setdefault(root, []).append(agent)

    return list(groups.values())
