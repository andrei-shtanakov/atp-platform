"""Tournament modes: round-robin and elimination brackets.

Supports round-robin (all-pairs) and single/double elimination
tournament formats for comparing agents in game-theoretic settings.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

from atp.adapters.base import AgentAdapter
from game_envs.core.game import Game

from atp_games.models import GameResult, GameRunConfig
from atp_games.runner.game_runner import GameRunner

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a single match between two agents.

    Attributes:
        agent_a: Name of first agent.
        agent_b: Name of second agent.
        game_result: Full GameResult from the match.
        winner: Name of the winner, or None for draw.
        score_a: Aggregate score for agent A.
        score_b: Aggregate score for agent B.
    """

    agent_a: str
    agent_b: str
    game_result: GameResult
    winner: str | None
    score_a: float
    score_b: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "winner": self.winner,
            "score_a": round(self.score_a, 4),
            "score_b": round(self.score_b, 4),
            "game_result": self.game_result.to_dict(),
        }


@dataclass
class Standing:
    """Tournament standing for one agent.

    Attributes:
        agent: Agent name.
        wins: Number of matches won.
        losses: Number of matches lost.
        draws: Number of draws.
        total_payoff: Sum of average payoffs across matches.
        matches_played: Total matches played.
    """

    agent: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_payoff: float = 0.0
    matches_played: int = 0

    @property
    def points(self) -> float:
        """Points: 3 for win, 1 for draw, 0 for loss."""
        return self.wins * 3.0 + self.draws * 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent": self.agent,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "points": self.points,
            "total_payoff": round(self.total_payoff, 4),
            "matches_played": self.matches_played,
        }


@dataclass
class TournamentResult:
    """Result of a complete tournament.

    Attributes:
        mode: Tournament mode (round_robin, single_elimination,
            double_elimination).
        standings: Sorted list of standings (best first).
        matches: All match results.
        bracket: Elimination bracket structure (if applicable).
    """

    mode: str
    standings: list[Standing]
    matches: list[MatchResult] = field(default_factory=list)
    bracket: list[list[MatchResult]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "mode": self.mode,
            "standings": [s.to_dict() for s in self.standings],
            "matches": [m.to_dict() for m in self.matches],
        }
        if self.bracket is not None:
            result["bracket"] = [[m.to_dict() for m in rnd] for rnd in self.bracket]
        return result


async def _run_match(
    runner: GameRunner,
    game: Game,
    agent_a_name: str,
    agent_a: AgentAdapter,
    agent_b_name: str,
    agent_b: AgentAdapter,
    config: GameRunConfig,
) -> MatchResult:
    """Run a single match between two agents.

    Args:
        runner: GameRunner instance.
        game: Game template.
        agent_a_name: Name of first agent.
        agent_a: First agent adapter.
        agent_b_name: Name of second agent.
        agent_b: Second agent adapter.
        config: Run configuration.

    Returns:
        MatchResult for the matchup.
    """
    match_game = copy.deepcopy(game)
    player_ids = match_game.player_ids
    agents = {
        player_ids[0]: copy.deepcopy(agent_a),
        player_ids[1]: copy.deepcopy(agent_b),
    }

    result = await runner.run_game(match_game, agents, config)

    avg = result.average_payoffs
    score_a = avg.get(player_ids[0], 0.0)
    score_b = avg.get(player_ids[1], 0.0)

    draw_threshold = 1e-6
    if abs(score_a - score_b) < draw_threshold:
        winner = None
    elif score_a > score_b:
        winner = agent_a_name
    else:
        winner = agent_b_name

    return MatchResult(
        agent_a=agent_a_name,
        agent_b=agent_b_name,
        game_result=result,
        winner=winner,
        score_a=score_a,
        score_b=score_b,
    )


async def run_round_robin(
    game: Game,
    agents: dict[str, AgentAdapter],
    config: GameRunConfig | None = None,
    runner: GameRunner | None = None,
) -> TournamentResult:
    """Run a round-robin tournament.

    Every pair of agents plays N episodes. Handles byes for
    odd numbers of agents (agent gets a win with 0 payoff).

    Args:
        game: Game template to use for all matches.
        agents: Mapping of agent name to adapter.
        config: Run configuration per match.
        runner: Optional GameRunner (created if not provided).

    Returns:
        TournamentResult with standings and match details.
    """
    config = config or GameRunConfig()
    runner = runner or GameRunner()
    agent_names = list(agents.keys())

    standings_map: dict[str, Standing] = {
        name: Standing(agent=name) for name in agent_names
    }
    matches: list[MatchResult] = []

    # Generate all pairs
    pairs: list[tuple[str, str]] = []
    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            pairs.append((agent_names[i], agent_names[j]))

    logger.info(
        "Round-robin tournament: %d agents, %d matches",
        len(agent_names),
        len(pairs),
    )

    for a_name, b_name in pairs:
        logger.info("Match: %s vs %s", a_name, b_name)
        match = await _run_match(
            runner=runner,
            game=game,
            agent_a_name=a_name,
            agent_a=agents[a_name],
            agent_b_name=b_name,
            agent_b=agents[b_name],
            config=config,
        )
        matches.append(match)

        # Update standings
        standings_map[a_name].matches_played += 1
        standings_map[b_name].matches_played += 1
        standings_map[a_name].total_payoff += match.score_a
        standings_map[b_name].total_payoff += match.score_b

        if match.winner is None:
            standings_map[a_name].draws += 1
            standings_map[b_name].draws += 1
        elif match.winner == a_name:
            standings_map[a_name].wins += 1
            standings_map[b_name].losses += 1
        else:
            standings_map[b_name].wins += 1
            standings_map[a_name].losses += 1

    # Handle bye for odd number of agents
    if len(agent_names) % 2 == 1:
        for name in agent_names:
            # Each agent already played all others;
            # bye is implicit (no extra matches needed)
            pass

    # Sort standings by points (desc), then total_payoff (desc)
    standings = sorted(
        standings_map.values(),
        key=lambda s: (s.points, s.total_payoff),
        reverse=True,
    )

    return TournamentResult(
        mode="round_robin",
        standings=standings,
        matches=matches,
    )


async def run_single_elimination(
    game: Game,
    agents: dict[str, AgentAdapter],
    config: GameRunConfig | None = None,
    runner: GameRunner | None = None,
    seeding: list[str] | None = None,
) -> TournamentResult:
    """Run a single-elimination tournament.

    Agents are paired in brackets. Losers are eliminated.
    Handles byes when the number of agents is not a power of 2.

    Args:
        game: Game template.
        agents: Agent name to adapter mapping.
        config: Run configuration per match.
        runner: Optional GameRunner.
        seeding: Optional ordered list of agent names for seeding.
            If not provided, uses dict order.

    Returns:
        TournamentResult with bracket and standings.
    """
    config = config or GameRunConfig()
    runner = runner or GameRunner()

    if seeding is not None:
        ordered = list(seeding)
    else:
        ordered = list(agents.keys())

    n = len(ordered)
    if n < 2:
        raise ValueError("Need at least 2 agents for elimination")

    # Pad to next power of 2 with byes
    bracket_size = 1
    while bracket_size < n:
        bracket_size *= 2

    # Fill bracket with None for byes
    bracket_agents: list[str | None] = list(ordered) + [None] * (bracket_size - n)

    all_matches: list[MatchResult] = []
    bracket_rounds: list[list[MatchResult]] = []
    standings_map: dict[str, Standing] = {
        name: Standing(agent=name) for name in ordered
    }

    current_round = bracket_agents
    round_num = 0

    while len(current_round) > 1:
        round_num += 1
        round_matches: list[MatchResult] = []
        next_round: list[str | None] = []

        for i in range(0, len(current_round), 2):
            a = current_round[i]
            b = current_round[i + 1]

            if a is None and b is None:
                next_round.append(None)
                continue
            elif a is None:
                # b gets a bye
                next_round.append(b)
                continue
            elif b is None:
                # a gets a bye
                next_round.append(a)
                continue

            logger.info("Round %d: %s vs %s", round_num, a, b)
            match = await _run_match(
                runner=runner,
                game=game,
                agent_a_name=a,
                agent_a=agents[a],
                agent_b_name=b,
                agent_b=agents[b],
                config=config,
            )
            round_matches.append(match)
            all_matches.append(match)

            standings_map[a].matches_played += 1
            standings_map[b].matches_played += 1
            standings_map[a].total_payoff += match.score_a
            standings_map[b].total_payoff += match.score_b

            if match.winner == a:
                standings_map[a].wins += 1
                standings_map[b].losses += 1
                next_round.append(a)
            elif match.winner == b:
                standings_map[b].wins += 1
                standings_map[a].losses += 1
                next_round.append(b)
            else:
                # Draw: agent_a advances (higher seed)
                standings_map[a].draws += 1
                standings_map[b].draws += 1
                standings_map[b].losses += 1
                next_round.append(a)

        bracket_rounds.append(round_matches)
        current_round = next_round

    # Sort standings by elimination round then payoff
    standings = sorted(
        standings_map.values(),
        key=lambda s: (s.wins, s.total_payoff),
        reverse=True,
    )

    return TournamentResult(
        mode="single_elimination",
        standings=standings,
        matches=all_matches,
        bracket=bracket_rounds,
    )


async def run_double_elimination(
    game: Game,
    agents: dict[str, AgentAdapter],
    config: GameRunConfig | None = None,
    runner: GameRunner | None = None,
    seeding: list[str] | None = None,
) -> TournamentResult:
    """Run a double-elimination tournament.

    Agents must lose twice to be eliminated. Winners bracket
    and losers bracket run in parallel.

    Args:
        game: Game template.
        agents: Agent name to adapter mapping.
        config: Run configuration per match.
        runner: Optional GameRunner.
        seeding: Optional ordered list of agent names.

    Returns:
        TournamentResult with bracket and standings.
    """
    config = config or GameRunConfig()
    runner = runner or GameRunner()

    if seeding is not None:
        ordered = list(seeding)
    else:
        ordered = list(agents.keys())

    n = len(ordered)
    if n < 2:
        raise ValueError("Need at least 2 agents for elimination")

    all_matches: list[MatchResult] = []
    bracket_rounds: list[list[MatchResult]] = []
    standings_map: dict[str, Standing] = {
        name: Standing(agent=name) for name in ordered
    }

    # Track losses per agent
    loss_count: dict[str, int] = {name: 0 for name in ordered}
    winners_bracket: list[str] = list(ordered)
    losers_bracket: list[str] = []

    round_num = 0

    while len(winners_bracket) + len(losers_bracket) > 1:
        round_num += 1
        round_matches: list[MatchResult] = []

        # Winners bracket matches
        if len(winners_bracket) >= 2:
            next_winners: list[str] = []
            for i in range(0, len(winners_bracket) - 1, 2):
                a = winners_bracket[i]
                b = winners_bracket[i + 1]

                match = await _run_match(
                    runner=runner,
                    game=game,
                    agent_a_name=a,
                    agent_a=agents[a],
                    agent_b_name=b,
                    agent_b=agents[b],
                    config=config,
                )
                round_matches.append(match)
                all_matches.append(match)

                standings_map[a].matches_played += 1
                standings_map[b].matches_played += 1
                standings_map[a].total_payoff += match.score_a
                standings_map[b].total_payoff += match.score_b

                if match.winner == b:
                    standings_map[b].wins += 1
                    standings_map[a].losses += 1
                    next_winners.append(b)
                    loss_count[a] += 1
                    if loss_count[a] < 2:
                        losers_bracket.append(a)
                else:
                    # a wins (or draw favors a)
                    standings_map[a].wins += 1
                    standings_map[b].losses += 1
                    next_winners.append(a)
                    loss_count[b] += 1
                    if loss_count[b] < 2:
                        losers_bracket.append(b)

            # Handle odd agent in winners bracket
            if len(winners_bracket) % 2 == 1:
                next_winners.append(winners_bracket[-1])
            winners_bracket = next_winners
        elif len(winners_bracket) == 1 and len(losers_bracket) >= 1:
            # Final: winners bracket champion vs losers survivor
            a = winners_bracket[0]
            b = losers_bracket[0]

            match = await _run_match(
                runner=runner,
                game=game,
                agent_a_name=a,
                agent_a=agents[a],
                agent_b_name=b,
                agent_b=agents[b],
                config=config,
            )
            round_matches.append(match)
            all_matches.append(match)

            standings_map[a].matches_played += 1
            standings_map[b].matches_played += 1
            standings_map[a].total_payoff += match.score_a
            standings_map[b].total_payoff += match.score_b

            if match.winner == b:
                standings_map[b].wins += 1
                standings_map[a].losses += 1
                loss_count[a] += 1
            else:
                standings_map[a].wins += 1
                standings_map[b].losses += 1
                loss_count[b] += 1

            # Tournament over
            winners_bracket = []
            losers_bracket = []
            bracket_rounds.append(round_matches)
            break

        # Losers bracket matches
        if len(losers_bracket) >= 2:
            next_losers: list[str] = []
            for i in range(0, len(losers_bracket) - 1, 2):
                a = losers_bracket[i]
                b = losers_bracket[i + 1]

                match = await _run_match(
                    runner=runner,
                    game=game,
                    agent_a_name=a,
                    agent_a=agents[a],
                    agent_b_name=b,
                    agent_b=agents[b],
                    config=config,
                )
                round_matches.append(match)
                all_matches.append(match)

                standings_map[a].matches_played += 1
                standings_map[b].matches_played += 1
                standings_map[a].total_payoff += match.score_a
                standings_map[b].total_payoff += match.score_b

                if match.winner == b:
                    standings_map[b].wins += 1
                    standings_map[a].losses += 1
                    next_losers.append(b)
                    loss_count[a] += 1
                    # a is now eliminated (2 losses)
                else:
                    standings_map[a].wins += 1
                    standings_map[b].losses += 1
                    next_losers.append(a)
                    loss_count[b] += 1
                    # b is now eliminated (2 losses)

            if len(losers_bracket) % 2 == 1:
                next_losers.append(losers_bracket[-1])
            losers_bracket = next_losers

        bracket_rounds.append(round_matches)

        # Safety: prevent infinite loops
        if round_num > 100:
            logger.warning("Double elimination exceeded 100 rounds")
            break

    standings = sorted(
        standings_map.values(),
        key=lambda s: (s.wins, s.total_payoff),
        reverse=True,
    )

    return TournamentResult(
        mode="double_elimination",
        standings=standings,
        matches=all_matches,
        bracket=bracket_rounds,
    )
