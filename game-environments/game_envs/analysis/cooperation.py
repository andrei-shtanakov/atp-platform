"""Cooperation metrics for game history analysis.

Measures cooperative behavior patterns including cooperation
rates, conditional cooperation, and reciprocity between players.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from game_envs.core.state import RoundResult

COOPERATIVE_ACTIONS = frozenset({"cooperate", "c"})


def _is_cooperative(action: Any) -> bool:
    """Check if an action is cooperative."""
    return str(action).lower() in COOPERATIVE_ACTIONS


@dataclass
class CooperationMetrics:
    """Results of cooperation analysis for a player.

    Attributes:
        cooperation_rate: Fraction of cooperative actions [0, 1].
        conditional_cooperation: P(C|C) vs P(C|D) as a dict
            with keys 'prob_c_given_c' and 'prob_c_given_d'.
        reciprocity_index: Correlation of cooperation between
            players [-1, 1]. Positive means reciprocal.
    """

    cooperation_rate: float
    conditional_cooperation: dict[str, float | None]
    reciprocity_index: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "cooperation_rate": self.cooperation_rate,
            "conditional_cooperation": dict(self.conditional_cooperation),
            "reciprocity_index": self.reciprocity_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CooperationMetrics:
        """Deserialize from dictionary."""
        return cls(
            cooperation_rate=data["cooperation_rate"],
            conditional_cooperation=data["conditional_cooperation"],
            reciprocity_index=data["reciprocity_index"],
        )


def cooperation_rate(
    history: list[RoundResult],
    player: str,
) -> float:
    """Compute fraction of cooperative actions for a player.

    Args:
        history: List of round results from a game.
        player: Player ID to analyze.

    Returns:
        Fraction of rounds where the player cooperated [0, 1].

    Raises:
        ValueError: If history is empty or player not found.
    """
    if not history:
        raise ValueError("Cannot compute cooperation rate from empty history")

    actions = [rr.actions[player] for rr in history if player in rr.actions]
    if not actions:
        raise ValueError(f"Player '{player}' not found in history")

    coop_count = sum(1 for a in actions if _is_cooperative(a))
    return coop_count / len(actions)


def conditional_cooperation(
    history: list[RoundResult],
    player: str,
    opponent: str | None = None,
) -> dict[str, float | None]:
    """Compute conditional cooperation probabilities.

    Calculates P(C|opponent played C last round) and
    P(C|opponent played D last round).

    Args:
        history: List of round results from a game.
        player: Player ID to analyze.
        opponent: Opponent player ID. If None, uses the first
            other player found in history.

    Returns:
        Dict with 'prob_c_given_c' and 'prob_c_given_d'.
        Values are None if no observations for that condition.

    Raises:
        ValueError: If history has fewer than 2 rounds or
            player/opponent not found.
    """
    if len(history) < 2:
        raise ValueError("Need at least 2 rounds for conditional cooperation")

    # Find opponent if not specified
    if opponent is None:
        for rr in history:
            for pid in rr.actions:
                if pid != player:
                    opponent = pid
                    break
            if opponent is not None:
                break
    if opponent is None:
        raise ValueError("Could not find opponent in history")

    c_given_c = 0
    total_given_c = 0
    c_given_d = 0
    total_given_d = 0

    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]

        if opponent not in prev.actions or player not in curr.actions:
            continue

        opp_cooperated = _is_cooperative(prev.actions[opponent])
        player_cooperates = _is_cooperative(curr.actions[player])

        if opp_cooperated:
            total_given_c += 1
            if player_cooperates:
                c_given_c += 1
        else:
            total_given_d += 1
            if player_cooperates:
                c_given_d += 1

    prob_c_given_c = c_given_c / total_given_c if total_given_c > 0 else None
    prob_c_given_d = c_given_d / total_given_d if total_given_d > 0 else None

    return {
        "prob_c_given_c": prob_c_given_c,
        "prob_c_given_d": prob_c_given_d,
    }


def reciprocity_index(
    history: list[RoundResult],
    player_a: str | None = None,
    player_b: str | None = None,
) -> float:
    """Compute correlation of cooperation between two players.

    Uses Pearson correlation on binary cooperation indicators.
    A value of +1 means perfect reciprocity (both cooperate or
    defect together), -1 means perfect anti-reciprocity.

    Args:
        history: List of round results from a game.
        player_a: First player ID. If None, auto-detected.
        player_b: Second player ID. If None, auto-detected.

    Returns:
        Pearson correlation coefficient in [-1, 1].
        Returns 0.0 if variance is zero (constant behavior).

    Raises:
        ValueError: If history is empty or players not found.
    """
    if not history:
        raise ValueError("Cannot compute reciprocity from empty history")

    # Auto-detect players
    if player_a is None or player_b is None:
        players = set()
        for rr in history:
            players.update(rr.actions.keys())
        player_list = sorted(players)
        if len(player_list) < 2:
            raise ValueError("Need at least 2 players for reciprocity")
        if player_a is None:
            player_a = player_list[0]
        if player_b is None:
            player_b = player_list[1]

    a_coop: list[float] = []
    b_coop: list[float] = []

    for rr in history:
        if player_a in rr.actions and player_b in rr.actions:
            a_coop.append(1.0 if _is_cooperative(rr.actions[player_a]) else 0.0)
            b_coop.append(1.0 if _is_cooperative(rr.actions[player_b]) else 0.0)

    if not a_coop:
        raise ValueError("No rounds with both players found")

    n = len(a_coop)
    mean_a = sum(a_coop) / n
    mean_b = sum(b_coop) / n

    cov = sum((a_coop[i] - mean_a) * (b_coop[i] - mean_b) for i in range(n)) / n
    std_a = (sum((x - mean_a) ** 2 for x in a_coop) / n) ** 0.5
    std_b = (sum((x - mean_b) ** 2 for x in b_coop) / n) ** 0.5

    if std_a == 0.0 or std_b == 0.0:
        return 0.0

    return cov / (std_a * std_b)
