"""El Farol Bar Problem — N-player minority / congestion game.

Brian Arthur (1994) showed that a population of heterogeneous agents
with bounded rationality can self-organise around a bar's capacity
threshold without any explicit coordination.

Game structure:
  - N players decide independently which time-slots to attend.
  - If attendance at a slot >= threshold, the slot is *crowded* (bad).
  - Players want to maximise time in non-crowded slots.

One *round* in the framework corresponds to one *day* in the original
simulation.  Each player submits a list of slot indices (0 to
num_slots-1) they plan to attend.  The game runs for num_rounds days.

Payoff per round:
  happy_slots - crowded_slots   (net non-crowded slots attended that day)

Final payoffs (get_payoffs):
  t_happy / max(t_crowded, 0.1) for each player.
  Players who attended fewer than min_total_hours hours in total
  receive a disqualification penalty of 0.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from game_envs.core.action import ActionSpace
from game_envs.core.game import Game, GameConfig, GameType, MoveOrder
from game_envs.core.state import (
    GameState,
    Observation,
    RoundResult,
    StepResult,
)
from game_envs.games.registry import register_game

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------


class ElFarolActionSpace(ActionSpace):
    """Action space for El Farol Bar: a list of time-slot indices.

    A valid action is a (possibly empty) list of integers in
    [0, num_slots), each slot representing a 30-minute window.
    """

    def __init__(self, num_slots: int = 16) -> None:
        self.num_slots = num_slots

    def contains(self, action: Any) -> bool:
        if not isinstance(action, list):
            return False
        return all(isinstance(s, int) and 0 <= s < self.num_slots for s in action)

    def sample(self, rng: random.Random | None = None) -> list[int]:
        r = rng or random.Random()
        length = r.randint(4, 8)
        start = r.randint(0, max(0, self.num_slots - length))
        return list(range(start, min(start + length, self.num_slots)))

    def to_list(self) -> list[str]:
        return [f"list of slot indices in 0..{self.num_slots - 1}"]

    def to_description(self) -> str:
        return (
            f"Choose which time slots to attend today. "
            f"Provide a list of integers in 0–{self.num_slots - 1} "
            f"(each slot = 30 min). You may attend at most 8 consecutive "
            f"slots per session. Example: [4, 5, 6, 7, 8]. "
            f"Return an empty list [] to stay home."
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElFarolConfig(GameConfig):
    """Configuration for El Farol Bar.

    Attributes:
        num_players: Total population (default 100).
        num_rounds: Number of days to simulate (default 30).
        num_slots: Time slots per day — default 16 (30-min each, 8 hours).
        capacity_threshold: Bar is crowded when attendance >= this value.
        min_total_hours: Minimum hours required to avoid disqualification.
        slot_duration: Duration of each slot in hours (default 0.5 h).
    """

    num_players: int = 100
    num_rounds: int = 30
    num_slots: int = 16
    capacity_threshold: int = 60
    min_total_hours: float = 0.0
    slot_duration: float = 0.5  # hours

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {self.num_slots}")
        if self.capacity_threshold < 1:
            raise ValueError(
                f"capacity_threshold must be >= 1, got {self.capacity_threshold}"
            )
        if self.min_total_hours < 0:
            raise ValueError(
                f"min_total_hours must be >= 0, got {self.min_total_hours}"
            )
        if self.slot_duration <= 0:
            raise ValueError(f"slot_duration must be > 0, got {self.slot_duration}")


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------


@register_game("el_farol", ElFarolConfig)
class ElFarolBar(Game):
    """El Farol Bar Problem — N-player congestion / minority game.

    Each of N players simultaneously decides which time slots to attend.
    If occupancy at a slot reaches or exceeds the capacity_threshold,
    the slot is labelled *crowded* and attendance there is unpleasant.

    The game captures:
      - Bounded rationality (each player sees only past attendance, not
        others' current plans).
      - Emergent self-organisation from heterogeneous strategies.
      - Negative feedback loop: popularity reduces value.

    Nash equilibrium: In a symmetric mixed-strategy NE, each player
    attends with probability p* = threshold / num_players per slot,
    which makes each slot crowded exactly at the threshold in expectation.
    Heterogeneous strategy populations often outperform the NE in
    aggregate welfare.

    Features:
      - Supports 2 to 1000 players.
      - Discrete per-slot attendance choice.
      - Full attendance history shared with all players (public info).
      - Disqualification rule for low-attendance players.
    """

    def __init__(self, config: ElFarolConfig | None = None) -> None:
        cfg = config or ElFarolConfig()
        super().__init__(cfg)
        n = cfg.num_players
        if n < 2 or n > 1000:
            raise ValueError(f"num_players must be between 2 and 1000, got {n}")

        self._terminal = False
        # Per-player cumulative happy / crowded time (in slots)
        self._t_happy: dict[str, float] = {pid: 0.0 for pid in self.player_ids}
        self._t_crowded: dict[str, float] = {pid: 0.0 for pid in self.player_ids}
        # Attendance history: list of arrays (one per past day)
        self._attendance_history: list[list[float]] = []

    # ------------------------------------------------------------------
    # Game identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        c = self._ef_config
        return (
            f"El Farol Bar (n={c.num_players}, "
            f"threshold={c.capacity_threshold}, "
            f"days={c.num_rounds})"
        )

    @property
    def game_type(self) -> GameType:
        return GameType.REPEATED

    @property
    def move_order(self) -> MoveOrder:
        return MoveOrder.SIMULTANEOUS

    @property
    def player_ids(self) -> list[str]:
        return [f"player_{i}" for i in range(self.config.num_players)]

    # ------------------------------------------------------------------
    # Typed config access
    # ------------------------------------------------------------------

    @property
    def _ef_config(self) -> ElFarolConfig:
        return self.config  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------

    def action_space(self, player_id: str) -> ElFarolActionSpace:
        return ElFarolActionSpace(self._ef_config.num_slots)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> StepResult:
        """Reset game to day 0."""
        self._reset_base()
        self._terminal = False
        self._t_happy = {pid: 0.0 for pid in self.player_ids}
        self._t_crowded = {pid: 0.0 for pid in self.player_ids}
        self._attendance_history = []

        state = GameState(
            round_number=0,
            player_states={},
            public_state={"game": self.name, "attendance_history": []},
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs={pid: 0.0 for pid in self.player_ids},
            is_terminal=False,
        )

    def step(self, actions: dict[str, Any]) -> StepResult:
        """Process one day.

        Args:
            actions: Mapping player_id -> list[int] of slot indices.

        Returns:
            StepResult with per-player payoffs for this day and
            updated attendance history in observations.
        """
        if self._terminal:
            raise RuntimeError("Game is already terminal")

        c = self._ef_config
        num_slots = c.num_slots
        threshold = c.capacity_threshold

        # ------------------------------------------------------------------
        # 1. Compute per-slot occupancy
        # ------------------------------------------------------------------
        daily_occupancy: list[float] = [0.0] * num_slots
        for pid in self.player_ids:
            slots = list(actions.get(pid, []))
            for s in slots:
                if 0 <= s < num_slots:
                    daily_occupancy[s] += 1

        # ------------------------------------------------------------------
        # 2. Update player stats and compute payoffs
        # ------------------------------------------------------------------
        payoffs: dict[str, float] = {}
        for pid in self.player_ids:
            slots = [s for s in actions.get(pid, []) if 0 <= s < num_slots]
            happy = sum(1 for s in slots if daily_occupancy[s] < threshold)
            crowded = sum(1 for s in slots if daily_occupancy[s] >= threshold)
            self._t_happy[pid] += happy
            self._t_crowded[pid] += crowded
            payoffs[pid] = float(happy - crowded)

        # ------------------------------------------------------------------
        # 3. Record attendance history
        # ------------------------------------------------------------------
        self._attendance_history.append(daily_occupancy)

        # ------------------------------------------------------------------
        # 4. Record round in game history
        # ------------------------------------------------------------------
        current_round = self._current_round
        rr = RoundResult(
            round_number=current_round,
            actions={pid: list(actions.get(pid, [])) for pid in self.player_ids},
            payoffs=payoffs,
        )
        self._history.add_round(rr)
        self._current_round += 1

        if self._current_round >= c.num_rounds:
            self._terminal = True

        state = GameState(
            round_number=self._current_round,
            player_states={},
            public_state={
                "game": self.name,
                "attendance_history": self._attendance_history,
                "crowded_slots_today": [
                    s for s, occ in enumerate(daily_occupancy) if occ >= threshold
                ],
                "max_occupancy_today": max(daily_occupancy),
            },
            is_terminal=self._terminal,
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs=payoffs,
            is_terminal=self._terminal,
        )

    # ------------------------------------------------------------------
    # Payoffs
    # ------------------------------------------------------------------

    def get_payoffs(self) -> dict[str, float]:
        """Compute final payoffs.

        Returns:
            score = t_happy / max(t_crowded, 0.1) per player.
            Players who attended fewer than min_total_hours hours receive 0.
        """
        c = self._ef_config
        result: dict[str, float] = {}
        for pid in self.player_ids:
            th = self._t_happy[pid]
            tc = self._t_crowded[pid]
            total_hours = (th + tc) * c.slot_duration
            if total_hours < c.min_total_hours:
                result[pid] = 0.0  # disqualified
            else:
                result[pid] = th / max(tc, 0.1)
        return result

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self, player_id: str) -> Observation:
        """Observation for a player.

        game_state includes:
          - attendance_history: list of per-slot occupancy arrays (past days).
          - capacity_threshold: the crowding threshold.
          - num_slots: time slots per day.
          - slot_duration: hours per slot.
          - your_t_happy / your_t_crowded: cumulative stats for this player.
        """
        c = self._ef_config
        th = self._t_happy.get(player_id, 0.0)
        tc = self._t_crowded.get(player_id, 0.0)

        game_state: dict[str, Any] = {
            "game": self.name,
            "num_slots": c.num_slots,
            "capacity_threshold": c.capacity_threshold,
            "slot_duration_hours": c.slot_duration,
            "min_total_hours": c.min_total_hours,
            "attendance_history": self._attendance_history,
            "your_t_happy_slots": th,
            "your_t_crowded_slots": tc,
            "your_total_hours": (th + tc) * c.slot_duration,
        }

        return Observation(
            player_id=player_id,
            game_state=game_state,
            available_actions=self.action_space(player_id).to_list(),
            history=self._history.for_player(player_id),
            round_number=self._current_round,
            total_rounds=c.num_rounds,
            messages=self._get_pending_messages(player_id),
        )

    # ------------------------------------------------------------------
    # LLM prompt
    # ------------------------------------------------------------------

    def to_prompt(self) -> str:
        """Describe the El Farol scenario for LLM agents."""
        c = self._ef_config
        slot_hours = c.num_slots * c.slot_duration
        return "\n".join(
            [
                f"This is the El Farol Bar Problem with {c.num_players} players.",
                "",
                "Rules:",
                f"- Each day has {c.num_slots} time slots "
                f"of {c.slot_duration:.1f} h each "
                f"({slot_hours:.0f} h total).",
                "- You choose which slots to attend "
                f"(list of integers 0–{c.num_slots - 1}).",
                f"- If {c.capacity_threshold}+ players attend "
                "a slot, it becomes *crowded*.",
                "- You can only observe past attendance — not what others plan today.",
                "",
                "Scoring:",
                "  score = total_happy_slots / max(total_crowded_slots, 0.1)",
                f"  (must attend >= {c.min_total_hours} h to avoid disqualification)",
                "",
                "Strategy note:",
                "  The Nash equilibrium has each player "
                "attend each slot with probability"
                f"  p* = {c.capacity_threshold}/{c.num_players} = "
                f"  {c.capacity_threshold / c.num_players:.2f}. "
                "  Heterogeneous learning strategies often outperform this.",
                "",
                f"This game is repeated for {c.num_rounds} days.",
            ]
        )

    # ------------------------------------------------------------------
    # Convenience: attendance history as numpy array
    # ------------------------------------------------------------------

    @property
    def attendance_array(self) -> np.ndarray:
        """Attendance history as (days, slots) numpy array."""
        if not self._attendance_history:
            return np.zeros((0, self._ef_config.num_slots))
        return np.array(self._attendance_history)
