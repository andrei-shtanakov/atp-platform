"""Colonel Blotto game implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from game_envs.core.action import StructuredActionSpace
from game_envs.core.game import Game, GameConfig, GameType, MoveOrder
from game_envs.core.state import (
    GameState,
    Observation,
    RoundResult,
    StepResult,
)
from game_envs.games.registry import register_game


@dataclass(frozen=True)
class BlottoConfig(GameConfig):
    """Colonel Blotto game configuration.

    Attributes:
        num_battlefields: Number of battlefields to contest.
        total_troops: Total troops each player must allocate
            across all battlefields.
    """

    num_battlefields: int = 3
    total_troops: int = 100

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_battlefields < 2:
            raise ValueError(
                f"num_battlefields must be >= 2, got {self.num_battlefields}"
            )
        if self.total_troops < 1:
            raise ValueError(f"total_troops must be >= 1, got {self.total_troops}")


@register_game("colonel_blotto", BlottoConfig)
class ColonelBlotto(Game):
    """Two-player Colonel Blotto game (one-shot or repeated).

    Each player simultaneously allocates troops across
    battlefields. A player wins a battlefield if they assign
    more troops than their opponent; ties split the
    battlefield. The payoff is the fraction of battlefields
    won.

    Features:
        - Configurable number of battlefields and troops
        - StructuredActionSpace (allocation vector)
        - Allocation must sum to total_troops, all non-negative
        - Discount factor for repeated games
    """

    def __init__(self, config: BlottoConfig | None = None) -> None:
        super().__init__(config or BlottoConfig())
        self._terminal = False
        self._cumulative: dict[str, float] = {pid: 0.0 for pid in self.player_ids}

    @property
    def _blotto_config(self) -> BlottoConfig:
        """Typed access to the Blotto-specific config."""
        return self.config  # type: ignore[return-value]

    @property
    def name(self) -> str:
        base = f"Colonel Blotto ({self._blotto_config.num_battlefields} battlefields)"
        if self.config.num_rounds > 1:
            base += f" (repeated x{self.config.num_rounds})"
        return base

    @property
    def game_type(self) -> GameType:
        if self.config.num_rounds > 1:
            return GameType.REPEATED
        return GameType.NORMAL_FORM

    @property
    def move_order(self) -> MoveOrder:
        return MoveOrder.SIMULTANEOUS

    @property
    def player_ids(self) -> list[str]:
        return [f"player_{i}" for i in range(self.config.num_players)]

    def _battlefield_fields(self) -> list[str]:
        """Field names for the allocation vector."""
        return [f"battlefield_{i}" for i in range(self._blotto_config.num_battlefields)]

    def action_space(self, player_id: str) -> StructuredActionSpace:
        """Allocation vector over battlefields."""
        c = self._blotto_config
        return StructuredActionSpace(
            schema={
                "type": "allocation",
                "fields": self._battlefield_fields(),
                "total": c.total_troops,
                "min_value": 0,
                "max_value": c.total_troops,
            },
            description=(
                f"Allocate {c.total_troops} troops across "
                f"{c.num_battlefields} battlefields. "
                f"Values must be non-negative and sum to "
                f"{c.total_troops}."
            ),
        )

    def reset(self) -> StepResult:
        """Reset game to initial state."""
        self._reset_base()
        self._terminal = False
        self._cumulative = {pid: 0.0 for pid in self.player_ids}
        state = GameState(
            round_number=0,
            player_states={},
            public_state={"game": self.name},
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs={pid: 0.0 for pid in self.player_ids},
            is_terminal=False,
        )

    def step(self, actions: dict[str, Any]) -> StepResult:
        """Process one round of troop allocations."""
        if self._terminal:
            raise RuntimeError("Game is already terminal")

        c = self._blotto_config
        fields = self._battlefield_fields()

        # Validate and extract allocations
        allocations: dict[str, dict[str, float]] = {}
        for pid in self.player_ids:
            alloc = actions[pid]
            if not self.action_space(pid).contains(alloc):
                raise ValueError(f"Invalid allocation for {pid}: {alloc}")
            allocations[pid] = {f: float(alloc[f]) for f in fields}

        # Compute payoffs: fraction of battlefields won
        payoffs = self._compute_payoffs(allocations, fields)

        # Apply discount factor
        discount = c.discount_factor**self._current_round
        discounted = {pid: p * discount for pid, p in payoffs.items()}
        for pid in self.player_ids:
            self._cumulative[pid] += discounted[pid]

        current_round_number = self._current_round
        self._current_round += 1

        rr = RoundResult(
            round_number=current_round_number,
            actions=actions,
            payoffs=payoffs,
        )
        self._history.add_round(rr)

        if self._current_round >= self.config.num_rounds:
            self._terminal = True

        state = GameState(
            round_number=self._current_round,
            player_states={},
            public_state={"game": self.name},
            is_terminal=self._terminal,
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs=payoffs,
            is_terminal=self._terminal,
        )

    def _compute_payoffs(
        self,
        allocations: dict[str, dict[str, float]],
        fields: list[str],
    ) -> dict[str, float]:
        """Compute payoffs as fraction of battlefields won.

        A player wins a battlefield if they have strictly more
        troops. Ties split the battlefield equally.
        """
        n_fields = len(fields)
        pids = self.player_ids
        scores: dict[str, float] = {pid: 0.0 for pid in pids}

        for bf in fields:
            troops = {pid: allocations[pid][bf] for pid in pids}
            max_troops = max(troops.values())
            winners = [pid for pid in pids if troops[pid] == max_troops]
            share = 1.0 / len(winners)
            for pid in winners:
                scores[pid] += share

        return {pid: scores[pid] / n_fields for pid in pids}

    def observe(self, player_id: str) -> Observation:
        """Get observation with game-specific state info."""
        c = self._blotto_config
        fields = self._battlefield_fields()
        game_state: dict[str, Any] = {
            "game": self.name,
            "your_role": player_id,
            "num_battlefields": c.num_battlefields,
            "total_troops": c.total_troops,
            "fields": fields,
            "total_units": float(c.total_troops),
            "payoff_rule": (
                "Win a battlefield by assigning more troops "
                "than your opponent. Ties split the "
                "battlefield. Payoff = fraction of "
                "battlefields won."
            ),
        }
        return Observation(
            player_id=player_id,
            game_state=game_state,
            available_actions=self.action_space(player_id).to_list(),
            history=self._history.for_player(player_id),
            round_number=self._current_round,
            total_rounds=self.config.num_rounds,
            messages=self._get_pending_messages(player_id),
        )

    def get_payoffs(self) -> dict[str, float]:
        """Get cumulative (discounted) payoffs."""
        return dict(self._cumulative)

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    def to_prompt(self) -> str:
        """Describe the Colonel Blotto game for LLM agents."""
        c = self._blotto_config
        lines = [
            "This is a Colonel Blotto game.",
            f"There are {c.num_players} players and {c.num_battlefields} battlefields.",
            "",
            "Rules:",
            f"- You have {c.total_troops} troops to "
            f"allocate across {c.num_battlefields} "
            "battlefields.",
            f"- Allocations must be non-negative and sum to exactly {c.total_troops}.",
            "- You win a battlefield if you assign "
            "strictly more troops than your opponent.",
            "- If both players assign equal troops to a "
            "battlefield, it is split (each gets 0.5).",
            "- Your payoff is the fraction of battlefields you win (0.0 to 1.0).",
        ]
        if c.num_rounds > 1:
            lines.append(f"\nThis game is repeated for {c.num_rounds} rounds.")
            if c.discount_factor < 1.0:
                lines.append(
                    f"Future payoffs are discounted by {c.discount_factor} per round."
                )
        lines.extend(
            [
                "",
                "Strategy note: There is no pure strategy "
                "Nash equilibrium in symmetric Colonel "
                "Blotto. Optimal play involves randomizing "
                "your allocations.",
            ]
        )
        return "\n".join(lines)
