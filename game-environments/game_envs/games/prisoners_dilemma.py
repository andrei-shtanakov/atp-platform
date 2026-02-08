"""Prisoner's Dilemma game implementation."""

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

COOPERATE = "cooperate"
DEFECT = "defect"


@dataclass(frozen=True)
class PDConfig(GameConfig):
    """Prisoner's Dilemma configuration.

    Payoff parameters follow the standard naming:
        T > R > P > S  and  2R > T + S

    Where:
        R (reward): Mutual cooperation payoff.
        S (sucker): Cooperate vs defect payoff.
        T (temptation): Defect vs cooperate payoff.
        P (punishment): Mutual defection payoff.
    """

    reward: float = 3.0
    sucker: float = 0.0
    temptation: float = 5.0
    punishment: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.temptation > self.reward:
            raise ValueError(
                f"Require T > R: temptation ({self.temptation})"
                f" must exceed reward ({self.reward})"
            )
        if not self.reward > self.punishment:
            raise ValueError(
                f"Require R > P: reward ({self.reward})"
                f" must exceed punishment ({self.punishment})"
            )
        if not self.punishment > self.sucker:
            raise ValueError(
                f"Require P > S: punishment ({self.punishment})"
                f" must exceed sucker ({self.sucker})"
            )
        if not 2 * self.reward > self.temptation + self.sucker:
            raise ValueError(
                f"Require 2R > T + S: "
                f"2*{self.reward} must exceed "
                f"{self.temptation} + {self.sucker}"
            )


class PrisonersDilemma(Game):
    """Classic Prisoner's Dilemma (one-shot or repeated).

    Two players simultaneously choose to cooperate or defect.
    Payoffs are determined by the payoff matrix defined in
    PDConfig.

    Features:
        - One-shot mode (num_rounds=1)
        - Repeated mode (num_rounds > 1) with discount_factor
        - Noise: action flip with probability config.noise
        - Full observability of past rounds via observe()
    """

    def __init__(self, config: PDConfig | None = None) -> None:
        super().__init__(config or PDConfig())
        self._terminal = False
        self._cumulative: dict[str, float] = {pid: 0.0 for pid in self.player_ids}

    @property
    def _pd_config(self) -> PDConfig:
        """Typed access to the PD-specific config."""
        return self.config  # type: ignore[return-value]

    @property
    def name(self) -> str:
        if self.config.num_rounds > 1:
            return f"Prisoner's Dilemma (repeated x{self.config.num_rounds})"
        return "Prisoner's Dilemma"

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
        """Both players choose cooperate or defect."""
        return DiscreteActionSpace([COOPERATE, DEFECT])

    def reset(self) -> StepResult:
        """Reset game to initial state."""
        self._current_round = 0
        self._terminal = False
        self._history.clear()
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

        a0 = actions["player_0"]
        a1 = actions["player_1"]

        # Apply noise (trembling hand)
        if self._pd_config.noise > 0:
            if self._rng.random() < self._pd_config.noise:
                a0 = DEFECT if a0 == COOPERATE else COOPERATE
            if self._rng.random() < self._pd_config.noise:
                a1 = DEFECT if a1 == COOPERATE else COOPERATE

        payoffs = self._compute_payoffs(a0, a1)

        # Apply discount factor for repeated games
        discount = self._pd_config.discount_factor**self._current_round
        discounted = {pid: p * discount for pid, p in payoffs.items()}
        for pid in self.player_ids:
            self._cumulative[pid] += discounted[pid]

        self._current_round += 1

        # Record round with actual (possibly noised) actions
        rr = RoundResult(
            round_number=self._current_round,
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
        c = self._pd_config
        return Observation(
            player_id=player_id,
            game_state={
                "game": self.name,
                "your_role": player_id,
                "payoff_matrix": {
                    "mutual_cooperation": c.reward,
                    "mutual_defection": c.punishment,
                    "temptation": c.temptation,
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
        c = self._pd_config
        matrix: dict[tuple[str, str], tuple[float, float]] = {
            (COOPERATE, COOPERATE): (c.reward, c.reward),
            (COOPERATE, DEFECT): (c.sucker, c.temptation),
            (DEFECT, COOPERATE): (c.temptation, c.sucker),
            (DEFECT, DEFECT): (c.punishment, c.punishment),
        }
        p0, p1 = matrix[(a0, a1)]
        return {"player_0": p0, "player_1": p1}
