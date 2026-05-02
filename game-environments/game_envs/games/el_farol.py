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

import math
import random
from dataclasses import dataclass
from typing import Any, Literal

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


# Default cap on total slots per day. The authoritative limit lives on
# ``ElFarolConfig.max_total_slots`` and is threaded through the action space.
MAX_SLOTS_PER_DAY = 8


class ElFarolActionSpace(ActionSpace):
    """Action space for El Farol Bar: contiguous-interval visits.

    A valid action covers at most ``max_intervals`` contiguous runs of
    time slots with at most ``max_total_slots`` slots in total. Agents
    submit one of two equivalent interval shapes:

      * ``{"intervals": [[start, end], ...]}`` — preferred wire form.
      * ``[[start, end], ...]`` — bare list of inclusive ``[start, end]`` pairs.

    The empty list ``[]`` (or ``{"intervals": []}``) means "stay home".

    Intervals are validated as non-overlapping, non-adjacent (at least
    one empty slot between them) and within ``[0, num_slots - 1]``.
    ``sanitize`` normalises any valid input into a flat, sorted,
    deduplicated ``list[int]`` of slot indices for downstream payoff
    math; flat slot lists are no longer accepted as input.
    """

    def __init__(
        self,
        num_slots: int = 16,
        max_intervals: int = 2,
        max_total_slots: int = 8,
    ) -> None:
        self.num_slots = num_slots
        self.max_intervals = max_intervals
        self.max_total_slots = max_total_slots

    # ------------------------------------------------------------------
    # Shape detection
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_intervals_from_dict(action: Any) -> Any:
        if isinstance(action, dict) and "intervals" in action:
            return action["intervals"]
        return None

    @staticmethod
    def _is_pair_list(value: Any) -> bool:
        """True iff ``value`` is a non-empty list of [start, end] pairs."""
        if not isinstance(value, list) or not value:
            return False
        for pair in value:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return False
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in pair):
                return False
        return True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_pairs(self, pairs: list[list[int]] | list[tuple[int, int]]) -> bool:
        if len(pairs) > self.max_intervals:
            return False
        normalised: list[tuple[int, int]] = []
        total = 0
        for pair in pairs:
            start, end = int(pair[0]), int(pair[1])
            if start > end:
                return False
            if start < 0 or end >= self.num_slots:
                return False
            normalised.append((start, end))
            total += end - start + 1
        if total > self.max_total_slots:
            return False
        # Non-overlap + non-adjacency: compare pairs in canonical order.
        ordered = sorted(normalised, key=lambda p: p[0])
        for i in range(len(ordered) - 1):
            prev_end = ordered[i][1]
            next_start = ordered[i + 1][0]
            if next_start <= prev_end + 1:
                return False
        return True

    def contains(self, action: Any) -> bool:
        # Dict form
        dict_intervals = self._extract_intervals_from_dict(action)
        if dict_intervals is not None:
            action = dict_intervals
        # Empty list is always valid ("stay home")
        if isinstance(action, list) and not action:
            return True
        # List of [start, end] pairs — the only accepted non-empty shape.
        if self._is_pair_list(action):
            return self._validate_pairs(list(action))
        return False

    # ------------------------------------------------------------------
    # Lenient normalisation
    # ------------------------------------------------------------------

    def sanitize(self, action: Any) -> list[int]:
        """Convert valid interval input to a sorted, deduped slot list.

        Accepts ``{"intervals": [[start, end], ...]}``, a bare list of
        ``[start, end]`` pairs, or ``[]`` / ``{"intervals": []}`` for
        "stay home". Anything else (including a flat list of slot
        indices) is treated as invalid and yields ``[]`` so the caller's
        default-action fallback can decide what to submit.

        The flat ``list[int]`` return type is the canonical internal
        representation used by payoff math; the wire format is intervals.
        """
        dict_intervals = self._extract_intervals_from_dict(action)
        if dict_intervals is not None:
            action = dict_intervals

        # Empty list = "stay home".
        if isinstance(action, list) and not action:
            return []

        # The only valid non-empty input shape is a list of [start, end] pairs.
        if not self._is_pair_list(action):
            return []
        if len(action) > self.max_intervals:
            return []
        if not self._validate_pairs(list(action)):
            return []
        slots: list[int] = []
        for pair in action:
            start, end = int(pair[0]), int(pair[1])
            slots.extend(range(start, end + 1))
        return sorted(set(slots))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def sample(self, rng: random.Random | None = None) -> list[list[int]]:
        """Sample a random valid action — a single contiguous interval."""
        r = rng or random.Random()
        length = r.randint(1, self.max_total_slots)
        start = r.randint(0, max(0, self.num_slots - length))
        end = min(start + length, self.num_slots) - 1
        if end < start:
            return []
        return [[start, end]]

    def to_list(self) -> list[str]:
        return [
            f"{{'intervals': [[start, end], ...]}} with up to "
            f"{self.max_intervals} non-adjacent contiguous intervals; "
            f"start, end in 0..{self.num_slots - 1}"
        ]

    def to_description(self) -> str:
        return (
            f"Choose which time slots to attend today as up to "
            f"{self.max_intervals} contiguous interval(s). "
            f'Shape: {{"intervals": [[start, end], ...]}} '
            f"with inclusive slot indices in 0..{self.num_slots - 1} "
            f"(each slot = 30 min). "
            f"At most {self.max_total_slots} slots total per day. "
            f"Intervals must be non-overlapping and non-adjacent (at least "
            f"one empty slot between them). Example: "
            f'{{"intervals": [[6, 9], [12, 15]]}}. '
            f'Return {{"intervals": []}} or [] to stay home.'
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
            A value of 0 is a sentinel meaning "derive from
            ``floor(capacity_ratio * num_players)``" at construction time.
        max_intervals: Maximum number of contiguous visit intervals per day.
        max_total_slots: Maximum total number of slots covered per day
            (across both intervals).
        capacity_ratio: Fraction of ``num_players`` used to derive
            ``capacity_threshold`` when it is not set explicitly. Must be
            strictly in ``(0, 1]``.
        min_total_hours: Minimum hours required to avoid disqualification.
        slot_duration: Duration of each slot in hours (default 0.5 h).
    """

    num_players: int = 100
    num_rounds: int = 30
    num_slots: int = 16
    capacity_threshold: int = 0
    max_intervals: int = 2
    max_total_slots: int = 8
    capacity_ratio: float = 0.6
    min_total_hours: float = 0.0
    slot_duration: float = 0.5  # hours
    # Scoring mode — see docs/games/rules/el-farol-bar.{ru,en}.md.
    # Default is "happy_minus_crowded" during incremental rollout; the
    # final commit flips it to "happy_only".
    scoring_mode: Literal["happy_only", "happy_minus_crowded"] = "happy_minus_crowded"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {self.num_slots}")
        if not 0.0 < self.capacity_ratio <= 1.0:
            raise ValueError(
                f"capacity_ratio must be in (0, 1], got {self.capacity_ratio}"
            )
        if self.max_total_slots > self.num_slots:
            raise ValueError(
                f"max_total_slots ({self.max_total_slots}) must be <= "
                f"num_slots ({self.num_slots})"
            )
        if self.max_intervals > self.max_total_slots:
            raise ValueError(
                f"max_intervals ({self.max_intervals}) must be <= "
                f"max_total_slots ({self.max_total_slots})"
            )
        if self.capacity_threshold == 0:
            derived = math.floor(self.capacity_ratio * self.num_players)
            object.__setattr__(self, "capacity_threshold", derived)
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
        if self.scoring_mode not in {"happy_only", "happy_minus_crowded"}:
            raise ValueError(
                f"scoring_mode must be 'happy_only' or "
                f"'happy_minus_crowded', got {self.scoring_mode!r}"
            )


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
        cfg = self._ef_config
        return ElFarolActionSpace(
            num_slots=cfg.num_slots,
            max_intervals=cfg.max_intervals,
            max_total_slots=cfg.max_total_slots,
        )

    def validate_action(self, raw: Any) -> dict[str, list[list[int]]]:
        """Validate a client-submitted action and return canonical form.

        Strict path. Used by tournament submit. Accepts only the
        ``{"intervals": [[start, end], ...]}`` interval shape; the
        legacy flat slot list (``{"slots": [...]}``) is no longer
        supported. See ``ElFarolActionSpace.sanitize`` for the permissive
        replay path.
        """
        from game_envs.core.errors import (
            ValidationError,  # local import to avoid cycles
        )

        if not isinstance(raw, dict):
            raise ValidationError(f"action must be a dict, got {type(raw).__name__}")
        if "intervals" not in raw:
            raise ValidationError("action must have field 'intervals'")
        intervals = raw["intervals"]
        if not isinstance(intervals, list):
            raise ValidationError(
                "intervals must be a list of [start, end] pairs, got "
                f"{type(intervals).__name__}"
            )
        cfg = self._ef_config
        num_slots = cfg.num_slots
        if len(intervals) > cfg.max_intervals:
            raise ValidationError(
                f"at most {cfg.max_intervals} intervals per day, got {len(intervals)}"
            )
        normalised: list[list[int]] = []
        total = 0
        for pair in intervals:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValidationError(
                    f"each interval must be a [start, end] pair, got {pair!r}"
                )
            start, end = pair[0], pair[1]
            if not (isinstance(start, int) and not isinstance(start, bool)):
                raise ValidationError(f"interval start {start!r} is not an int")
            if not (isinstance(end, int) and not isinstance(end, bool)):
                raise ValidationError(f"interval end {end!r} is not an int")
            if start < 0 or end >= num_slots:
                raise ValidationError(
                    f"interval [{start}, {end}] out of range [0, {num_slots})"
                )
            if start > end:
                raise ValidationError(
                    f"interval [{start}, {end}] must satisfy start <= end"
                )
            normalised.append([int(start), int(end)])
            total += end - start + 1
        if total > cfg.max_total_slots:
            raise ValidationError(
                f"intervals cover {total} slots; max is {cfg.max_total_slots}"
            )
        ordered = sorted(normalised, key=lambda p: p[0])
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt[0] <= prev[1] + 1:
                raise ValidationError(
                    f"intervals {prev} and {nxt} overlap or are adjacent"
                )
        return {"intervals": ordered}

    def default_action_on_timeout(self) -> dict[str, list[list[int]]]:
        """Stay home — attend zero slots (spec §3.1)."""
        return {"intervals": []}

    def format_state_for_player(
        self,
        round_number: int,
        total_rounds: int,
        participant_idx: int,
        action_history: list[dict[str, Any]],
        cumulative_scores: list[float],
    ) -> dict[str, Any]:
        """N-player state formatter (see spec §3.1).

        Action data in each ``action_history`` row is expected in the
        canonical interval shape ``{"intervals": [[start, end], ...]}``.
        Per-slot occupancy counts are derived by expanding each interval.

        Does NOT include ``pending_submission`` — service-layer concern.
        """
        num_slots = self._ef_config.num_slots

        def _slots_of(action_data: dict[str, Any]) -> list[int]:
            slots: list[int] = []
            for pair in action_data.get("intervals", []):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                start, end = pair[0], pair[1]
                if not (isinstance(start, int) and isinstance(end, int)):
                    continue
                lo = max(0, int(start))
                hi = min(num_slots - 1, int(end))
                if lo <= hi:
                    slots.extend(range(lo, hi + 1))
            return slots

        your_history = [
            sorted(set(_slots_of(row["actions"].get(participant_idx, {}))))
            for row in action_history
        ]
        attendance_by_round: list[list[int]] = []
        for row in action_history:
            counts = [0] * num_slots
            for _pid, action_data in row["actions"].items():
                for s in _slots_of(action_data):
                    counts[s] += 1
            attendance_by_round.append(counts)

        return {
            "tournament_id": -1,
            "game_type": "el_farol",
            "round_number": round_number,
            "total_rounds": total_rounds,
            "your_history": your_history,
            "attendance_by_round": attendance_by_round,
            "capacity_threshold": self._ef_config.capacity_threshold,
            "your_cumulative_score": cumulative_scores[participant_idx],
            "all_scores": list(cumulative_scores),
            "your_participant_idx": participant_idx,
            "num_slots": num_slots,
            "action_schema": {
                "type": "intervals",
                "shape": '{"intervals": [[start, end], ...]}',
                "max_intervals": self._ef_config.max_intervals,
                "max_total_slots": self._ef_config.max_total_slots,
                "value_range": [0, num_slots - 1],
            },
            "extra": {},
        }

    def compute_round_payoffs(self, actions: dict[int, dict[str, Any]]) -> list[float]:
        """Per-round payoff = happy slots − crowded slots (spec §3.3).

        Args:
            actions: participant_idx -> ``{"intervals": [[start, end], ...]}``.

        Returns:
            List of per-round payoffs in participant_idx order.
        """
        n = self._ef_config.num_players
        threshold = self._ef_config.capacity_threshold
        num_slots = self._ef_config.num_slots

        def _slots_of(action_data: dict[str, Any]) -> list[int]:
            slots: list[int] = []
            for pair in action_data.get("intervals", []):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                start, end = pair[0], pair[1]
                if not (isinstance(start, int) and isinstance(end, int)):
                    continue
                lo = max(0, int(start))
                hi = min(num_slots - 1, int(end))
                if lo <= hi:
                    slots.extend(range(lo, hi + 1))
            return slots

        per_player_slots: list[list[int]] = [
            _slots_of(actions.get(p_idx, {})) for p_idx in range(n)
        ]

        counts = [0] * num_slots
        for slots in per_player_slots:
            for s in slots:
                counts[s] += 1

        crowded = {i for i, c in enumerate(counts) if c >= threshold}

        payoffs: list[float] = [0.0] * n
        for p_idx, slots in enumerate(per_player_slots):
            happy = 0
            crowded_count = 0
            for s in slots:
                if s in crowded:
                    crowded_count += 1
                else:
                    happy += 1
            payoffs[p_idx] = float(happy - crowded_count)
        return payoffs

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
        aspace = self.action_space(self.player_ids[0])

        # ------------------------------------------------------------------
        # 1. Sanitize actions (deduplicate, bound, handle bad input)
        # ------------------------------------------------------------------
        clean: dict[str, list[int]] = {}
        for pid in self.player_ids:
            clean[pid] = aspace.sanitize(actions.get(pid))

        # ------------------------------------------------------------------
        # 2. Compute per-slot occupancy
        # ------------------------------------------------------------------
        daily_occupancy: list[float] = [0.0] * num_slots
        for pid in self.player_ids:
            for s in clean[pid]:
                daily_occupancy[s] += 1

        # ------------------------------------------------------------------
        # 3. Update player stats and compute payoffs
        # ------------------------------------------------------------------
        payoffs: dict[str, float] = {}
        for pid in self.player_ids:
            slots = clean[pid]
            happy = sum(1 for s in slots if daily_occupancy[s] < threshold)
            crowded = sum(1 for s in slots if daily_occupancy[s] >= threshold)
            self._t_happy[pid] += happy
            self._t_crowded[pid] += crowded
            payoffs[pid] = float(happy - crowded)

        # ------------------------------------------------------------------
        # 4. Record attendance history
        # ------------------------------------------------------------------
        self._attendance_history.append(daily_occupancy)

        # ------------------------------------------------------------------
        # 5. Record round in game history
        # ------------------------------------------------------------------
        current_round = self._current_round
        rr = RoundResult(
            round_number=current_round,
            actions={pid: clean[pid] for pid in self.player_ids},
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
                "- You choose which slots to attend as up to "
                f"{c.max_intervals} contiguous interval(s): "
                f'{{"intervals": [[start, end], ...]}} with inclusive '
                f"indices in 0–{c.num_slots - 1}, at most "
                f"{c.max_total_slots} slots total per day, intervals "
                "non-overlapping and non-adjacent.",
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
