"""Game history tracking with per-player filtering."""

from __future__ import annotations

from typing import Any

from game_envs.core.state import RoundResult


class GameHistory:
    """Tracks game rounds with per-player filtering.

    Stores the full history of rounds and provides methods
    to filter by player and serialize to/from dicts.
    """

    def __init__(self) -> None:
        self._rounds: list[RoundResult] = []

    @property
    def rounds(self) -> list[RoundResult]:
        """All recorded rounds."""
        return list(self._rounds)

    def __len__(self) -> int:
        return len(self._rounds)

    def add_round(self, result: RoundResult) -> None:
        """Record a round result."""
        self._rounds.append(result)

    def for_player(
        self,
        player_id: str,
        visible_players: list[str] | None = None,
    ) -> list[RoundResult]:
        """Get history filtered for a specific player.

        Args:
            player_id: The player requesting the view.
            visible_players: If set, only show actions/payoffs
                for these players. If None, show all (full
                observability).

        Returns:
            List of RoundResults with filtered actions/payoffs.
        """
        if visible_players is None:
            return list(self._rounds)

        filtered: list[RoundResult] = []
        for rr in self._rounds:
            actions = {k: v for k, v in rr.actions.items() if k in visible_players}
            payoffs = {k: v for k, v in rr.payoffs.items() if k in visible_players}
            messages = [
                m
                for m in rr.messages
                if m.sender in visible_players or m.sender == player_id
            ]
            filtered.append(
                RoundResult(
                    round_number=rr.round_number,
                    actions=actions,
                    payoffs=payoffs,
                    messages=messages,
                )
            )
        return filtered

    def get_player_actions(self, player_id: str) -> list[Any]:
        """Get list of all actions by a player."""
        return [rr.actions[player_id] for rr in self._rounds if player_id in rr.actions]

    def get_player_payoffs(self, player_id: str) -> list[float]:
        """Get list of all payoffs for a player."""
        return [rr.payoffs[player_id] for rr in self._rounds if player_id in rr.payoffs]

    def total_payoff(self, player_id: str) -> float:
        """Get total accumulated payoff for a player."""
        return sum(self.get_player_payoffs(player_id))

    def clear(self) -> None:
        """Clear all history."""
        self._rounds.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize history to a dictionary."""
        return {
            "rounds": [r.to_dict() for r in self._rounds],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameHistory:
        """Deserialize history from a dictionary."""
        history = cls()
        for r in data.get("rounds", []):
            history.add_round(RoundResult.from_dict(r))
        return history

    def __repr__(self) -> str:
        return f"GameHistory(rounds={len(self._rounds)})"
