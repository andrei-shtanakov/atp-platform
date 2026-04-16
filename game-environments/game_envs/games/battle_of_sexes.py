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

    # ------------------------------------------------------------------
    # Service-facing methods (mirror PrisonersDilemma / StagHunt surface
    # used by the tournament service — see packages/atp-dashboard/.../
    # tournament/service.py).
    # ------------------------------------------------------------------

    def format_state_for_player(
        self,
        round_number: int,
        total_rounds: int,
        participant_idx: int,
        action_history: list[dict[str, Any]],
        cumulative_scores: list[float],
    ) -> dict[str, Any]:
        """Build a player-private RoundState dict for the given player.

        Unlike PD and Stag Hunt, BoS is asymmetric — player 0 prefers A,
        player 1 prefers B. The dict exposes which role the caller plays
        via ``your_preferred`` so a client can reason about whose focal
        point matters for this round.
        """
        opponent_idx = 1 - participant_idx
        your_history = [
            row["actions"][participant_idx]["choice"] for row in action_history
        ]
        opponent_history = [
            row["actions"][opponent_idx]["choice"] for row in action_history
        ]
        return {
            "tournament_id": -1,
            "round_number": round_number,
            "game_type": "battle_of_sexes",
            "your_history": your_history,
            "opponent_history": opponent_history,
            "your_cumulative_score": cumulative_scores[participant_idx],
            "opponent_cumulative_score": cumulative_scores[opponent_idx],
            "your_preferred": A if participant_idx == 0 else B,
            "action_schema": {
                "type": "choice",
                "options": [A, B],
            },
            "your_turn": True,  # service overwrites based on DB submission state
            "total_rounds": total_rounds,
            "extra": {},
        }

    def validate_action(self, raw: Any) -> dict[str, str]:
        """Validate a client-submitted action and return canonical form.

        Strict path used by tournament submit. Raises ValidationError on
        any malformed input.
        """
        from game_envs.core.errors import (
            ValidationError,  # local import to avoid cycles
        )

        if not isinstance(raw, dict):
            raise ValidationError(f"action must be a dict, got {type(raw).__name__}")
        choice = raw.get("choice")
        if choice not in (A, B):
            raise ValidationError(f"choice must be {A!r} or {B!r}, got {choice!r}")
        return {"choice": choice}

    def default_action_on_timeout(self) -> dict[str, str]:
        """Action substituted when a participant misses the round deadline.

        Defaults to ``A`` — the conventional Schelling focal point for
        BoS write-ups (most English-language game-theory texts order the
        two options with A first and use A-A as the "expected" focal
        coordination). Both timed-out players defaulting to A gives them
        the mutual-coordination payoff.

        Note: the ``Game.default_action_on_timeout()`` signature is
        ``() -> action``, not ``(participant_idx) -> action``, so the
        asymmetric game cannot pick per-player defaults without a
        signature change across all 3 existing games (PD, SH, EF). That
        refactor is deferred. For now, both players timing out on BoS
        both default to A, producing the (preferred_a, other_a) payoff.
        """
        return {"choice": A}

    def compute_round_payoffs(self, actions: dict[int, dict[str, Any]]) -> list[float]:
        """Generic entry point used by the tournament service.

        Args:
            actions: Mapping participant_idx -> action_data dict.

        Returns:
            List of payoffs in participant_idx order.
        """
        a0 = actions[0]["choice"]
        a1 = actions[1]["choice"]
        payoffs = self._compute_payoffs(a0, a1)
        return [payoffs["player_0"], payoffs["player_1"]]
