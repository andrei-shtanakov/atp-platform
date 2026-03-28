"""Battle of the Sexes game implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from game_envs.core.action import DiscreteActionSpace
from game_envs.core.game import Game, GameConfig, GameType, MoveOrder
from game_envs.core.state import (
    GameState,
    Observation,
    RoundResult,
    StepResult,
)
from game_envs.games.registry import register_game

A = "A"
B = "B"


@dataclass(frozen=True)
class BoSConfig(GameConfig):
    """Battle of the Sexes configuration.

    Two players coordinate on an event. player_0 prefers A and
    player_1 prefers B, but both prefer coordination over mismatch.

    Payoff matrix:
             Player 1
             A             B
    Player 0 A (preferred_a, other_a)  (mismatch, mismatch)
             B (mismatch, mismatch)    (other_b, preferred_b)

    Constraints:
        preferred_a > other_a > mismatch
        preferred_b > other_b > mismatch
    """

    preferred_a: float = 3.0
    other_a: float = 2.0
    preferred_b: float = 3.0
    other_b: float = 2.0
    mismatch: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.preferred_a > self.other_a:
            raise ValueError(
                f"Require preferred_a > other_a: {self.preferred_a} > {self.other_a}"
            )
        if not self.other_a > self.mismatch:
            raise ValueError(
                f"Require other_a > mismatch: {self.other_a} > {self.mismatch}"
            )
        if not self.preferred_b > self.other_b:
            raise ValueError(
                f"Require preferred_b > other_b: {self.preferred_b} > {self.other_b}"
            )
        if not self.other_b > self.mismatch:
            raise ValueError(
                f"Require other_b > mismatch: {self.other_b} > {self.mismatch}"
            )


@register_game("battle_of_sexes", BoSConfig)
class BattleOfSexes(Game):
    """Battle of the Sexes coordination game (one-shot or repeated).

    Two players simultaneously choose event A or B. Both prefer to
    coordinate, but player_0 prefers A while player_1 prefers B.

    Pure Nash Equilibria: (A, A) and (B, B).

    Features:
        - One-shot mode (num_rounds=1)
        - Repeated mode (num_rounds > 1) with discount_factor
        - Noise: action flip with probability config.noise
        - Full observability of past rounds via observe()
    """

    def __init__(self, config: BoSConfig | None = None) -> None:
        super().__init__(config or BoSConfig())
        self._terminal = False
        self._cumulative: dict[str, float] = {pid: 0.0 for pid in self.player_ids}

    @property
    def _bos_config(self) -> BoSConfig:
        """Typed access to the BoS-specific config."""
        return self.config  # type: ignore[return-value]

    @property
    def name(self) -> str:
        if self.config.num_rounds > 1:
            return f"Battle of the Sexes (repeated x{self.config.num_rounds})"
        return "Battle of the Sexes"

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
        return ["player_0", "player_1"]

    def action_space(self, player_id: str) -> DiscreteActionSpace:
        """Both players choose A or B."""
        return DiscreteActionSpace([A, B])

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
        """Process one round of actions.

        Applies noise (action flip) if configured, computes
        payoffs, records the round, and checks for termination.
        """
        if self._terminal:
            raise RuntimeError("Game is already terminal")

        a0 = actions.get("player_0")
        a1 = actions.get("player_1")
        if a0 not in (A, B):
            raise ValueError(f"Invalid action for player_0: {a0}")
        if a1 not in (A, B):
            raise ValueError(f"Invalid action for player_1: {a1}")

        # Apply noise (trembling hand)
        if self._bos_config.noise > 0:
            if self._rng.random() < self._bos_config.noise:
                a0 = B if a0 == A else A
            if self._rng.random() < self._bos_config.noise:
                a1 = B if a1 == A else A

        payoffs = self._compute_payoffs(a0, a1)

        current_round_number = self._current_round

        # Apply discount factor for repeated games
        discount = self._bos_config.discount_factor**self._current_round
        discounted = {pid: p * discount for pid, p in payoffs.items()}
        for pid in self.player_ids:
            self._cumulative[pid] += discounted[pid]

        self._current_round += 1

        rr = RoundResult(
            round_number=current_round_number,
            actions={"player_0": a0, "player_1": a1},
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

    def observe(self, player_id: str) -> Observation:
        """Get observation with game-specific state info."""
        c = self._bos_config
        return Observation(
            player_id=player_id,
            game_state={
                "game": self.name,
                "your_role": player_id,
                "payoff_matrix": {
                    "both_A": (c.preferred_a, c.other_a),
                    "both_B": (c.other_b, c.preferred_b),
                    "mismatch": c.mismatch,
                },
            },
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

    def _compute_payoffs(self, a0: str, a1: str) -> dict[str, float]:
        """Compute round payoffs from the payoff matrix."""
        c = self._bos_config
        matrix: dict[tuple[str, str], tuple[float, float]] = {
            (A, A): (c.preferred_a, c.other_a),
            (B, B): (c.other_b, c.preferred_b),
            (A, B): (c.mismatch, c.mismatch),
            (B, A): (c.mismatch, c.mismatch),
        }
        p0, p1 = matrix[(a0, a1)]
        return {"player_0": p0, "player_1": p1}
