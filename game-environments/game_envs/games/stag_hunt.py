"""Stag Hunt game implementation."""

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

STAG = "stag"
HARE = "hare"


@dataclass(frozen=True)
class SHConfig(GameConfig):
    """Stag Hunt configuration.

    Payoff parameters define the coordination game:

        mutual_stag: Both hunt stag (cooperative success).
        hare: Hunt hare regardless of opponent (safe fallback).
        mutual_hare: Both hunt hare (safe equilibrium).
        sucker: Hunt stag while opponent hunts hare (failure).

    Constraints:
        mutual_stag > hare > sucker  (stag is best if coordinated)
        mutual_stag > mutual_hare    (coordination is Pareto-superior)
    """

    mutual_stag: float = 4.0
    mutual_hare: float = 3.0
    hare: float = 3.0
    sucker: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.mutual_stag > self.hare:
            raise ValueError(
                f"Require mutual_stag > hare: mutual_stag ({self.mutual_stag})"
                f" must exceed hare ({self.hare})"
            )
        if not self.hare > self.sucker:
            raise ValueError(
                f"Require hare > sucker: hare ({self.hare})"
                f" must exceed sucker ({self.sucker})"
            )
        if not self.mutual_stag > self.mutual_hare:
            raise ValueError(
                f"Require mutual_stag > mutual_hare: mutual_stag ({self.mutual_stag})"
                f" must exceed mutual_hare ({self.mutual_hare})"
            )


@register_game("stag_hunt", SHConfig)
class StagHunt(Game):
    """Stag Hunt coordination game (one-shot or repeated).

    Two players simultaneously choose to hunt stag or hare.
    Stag requires coordination: both must choose stag to succeed.
    Hare is safe: always yields a moderate payoff regardless of
    the opponent's choice.

    There are two pure Nash equilibria: (stag, stag) and
    (hare, hare). The stag equilibrium is Pareto-superior but
    riskier; the hare equilibrium is risk-dominant.

    Features:
        - One-shot mode (num_rounds=1)
        - Repeated mode (num_rounds > 1) with discount_factor
        - Noise: action flip with probability config.noise
        - Full observability of past rounds via observe()
    """

    def __init__(self, config: SHConfig | None = None) -> None:
        super().__init__(config or SHConfig())
        self._terminal = False
        self._cumulative: dict[str, float] = {pid: 0.0 for pid in self.player_ids}

    @property
    def _sh_config(self) -> SHConfig:
        """Typed access to the Stag Hunt-specific config."""
        return self.config  # type: ignore[return-value]

    @property
    def name(self) -> str:
        if self.config.num_rounds > 1:
            return f"Stag Hunt (repeated x{self.config.num_rounds})"
        return "Stag Hunt"

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
        """Both players choose stag or hare."""
        return DiscreteActionSpace([STAG, HARE])

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
        if a0 not in (STAG, HARE):
            raise ValueError(f"Invalid action for player_0: {a0}")
        if a1 not in (STAG, HARE):
            raise ValueError(f"Invalid action for player_1: {a1}")

        # Apply noise (trembling hand)
        if self._sh_config.noise > 0:
            if self._rng.random() < self._sh_config.noise:
                a0 = HARE if a0 == STAG else STAG
            if self._rng.random() < self._sh_config.noise:
                a1 = HARE if a1 == STAG else STAG

        payoffs = self._compute_payoffs(a0, a1)

        current_round_number = self._current_round

        # Apply discount factor for repeated games
        discount = self._sh_config.discount_factor**self._current_round
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
        c = self._sh_config
        return Observation(
            player_id=player_id,
            game_state={
                "game": self.name,
                "your_role": player_id,
                "payoff_matrix": {
                    "mutual_stag": c.mutual_stag,
                    "mutual_hare": c.mutual_hare,
                    "hare": c.hare,
                    "sucker": c.sucker,
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
        c = self._sh_config
        matrix: dict[tuple[str, str], tuple[float, float]] = {
            (STAG, STAG): (c.mutual_stag, c.mutual_stag),
            (STAG, HARE): (c.sucker, c.hare),
            (HARE, STAG): (c.hare, c.sucker),
            (HARE, HARE): (c.mutual_hare, c.mutual_hare),
        }
        p0, p1 = matrix[(a0, a1)]
        return {"player_0": p0, "player_1": p1}
