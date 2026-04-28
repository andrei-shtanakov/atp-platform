"""El Farol Bar strategies — six heterogeneous agent types.

Each strategy mirrors one of the agent archetypes from the original
El Farol Bar simulation (Arthur, 1994).  All strategies work with
the `ElFarolBar` game environment and receive attendance history
through `observation.game_state["attendance_history"]`.

Available strategies:
    Traditionalist  — copy the quietest window from yesterday
    TrendFollower   — pick the quietest window by weekly moving average
    Contrarian      — go where yesterday was busiest (expects herd to flee)
    Gambler         — random window each day
    SmartPredictor  — linear trend extrapolation (second-order analysis)
    Scout           — recon slot first, then commit based on occupancy
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_best_window(
    occupancy: list[float] | np.ndarray,
    length: int,
    find_max: bool = False,
) -> list[list[int]]:
    """Find the contiguous window of `length` slots with min (or max) sum.

    Args:
        occupancy: Per-slot occupancy values.
        length: Window length (number of consecutive slots).
        find_max: If True, find the maximum-sum window instead.

    Returns:
        Single-interval action ``[[start, end]]`` (inclusive end), or
        ``[]`` ("stay home") if a window of the requested length cannot
        fit. Matches the canonical El Farol interval action shape.
    """
    data = list(occupancy)
    n = len(data)
    if length <= 0 or length > n:
        return []

    best_start = 0
    best_sum = float("-inf") if find_max else float("inf")

    for i in range(n - length + 1):
        window_sum = sum(data[i : i + length])
        if find_max and window_sum > best_sum:
            best_sum = window_sum
            best_start = i
        elif not find_max and window_sum < best_sum:
            best_sum = window_sum
            best_start = i

    return [[best_start, best_start + length - 1]]


def _history_arrays(observation: Observation) -> list[list[float]]:
    """Extract attendance history from an observation."""
    return observation.game_state.get("attendance_history", [])


def _num_slots(observation: Observation) -> int:
    """Extract number of slots per day from an observation."""
    return int(observation.game_state.get("num_slots", 16))


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class Traditionalist(Strategy):
    """Repeat the quietest time window from yesterday.

    The Traditionalist believes the world is stable: if a slot was empty
    yesterday, it will be empty today.  Uses a contiguous window of
    ``window_size`` slots from the previous day's occupancy.

    Equivalent to: zero-lag, single-day memory agent.
    """

    def __init__(self, window_size: int = 6) -> None:
        self._window_size = window_size

    @property
    def name(self) -> str:
        return "traditionalist"

    def choose_action(self, observation: Observation) -> Any:
        history = _history_arrays(observation)
        if not history:
            return []
        yesterday = history[-1]
        return _get_best_window(yesterday, self._window_size, find_max=False)


def _midday_interval(num_slots: int, window_size: int) -> list[list[int]]:
    """Single contiguous interval starting near midday."""
    mid = num_slots // 4
    end = min(mid + window_size, num_slots) - 1
    if end < mid:
        return []
    return [[mid, end]]


class TrendFollower(Strategy):
    """Choose the quietest window by moving average over the past week.

    Computes the mean occupancy per slot over the last ``lookback`` days
    and attends the ``window_size``-slot contiguous block with the
    smallest average.  Falls back to a midday window on the first visit.

    Equivalent to: moving-average (MA) agent, typical of quantitative
    strategies that dampen noise with temporal smoothing.
    """

    def __init__(self, window_size: int = 8, lookback: int = 7) -> None:
        self._window_size = window_size
        self._lookback = lookback

    @property
    def name(self) -> str:
        return "trend_follower"

    def choose_action(self, observation: Observation) -> Any:
        history = _history_arrays(observation)
        slots = _num_slots(observation)
        if not history:
            return _midday_interval(slots, self._window_size)
        data = np.array(history[-self._lookback :])
        avg = np.mean(data, axis=0).tolist()
        return _get_best_window(avg, self._window_size, find_max=False)


class Contrarian(Strategy):
    """Go where yesterday was most crowded — expecting the herd to flee.

    The Contrarian assumes other agents are deterred by yesterday's crowds
    and will avoid those slots.  By targeting the previously busiest window
    they bet that the slot will empty out today.

    Equivalent to: negative-feedback / anti-momentum agent.
    """

    def __init__(self, window_size: int = 8) -> None:
        self._window_size = window_size

    @property
    def name(self) -> str:
        return "contrarian"

    def choose_action(self, observation: Observation) -> Any:
        history = _history_arrays(observation)
        slots = _num_slots(observation)
        if not history:
            # Default to evening on day 0
            start = max(0, slots - self._window_size)
            return [[start, slots - 1]]
        yesterday = history[-1]
        return _get_best_window(yesterday, self._window_size, find_max=True)


class Gambler(Strategy):
    """Attend a random contiguous window each day.

    Ignores all history.  Provides a stochastic baseline equivalent to
    a zero-intelligence agent.  Useful for measuring how much strategy
    matters relative to pure luck.

    Attributes:
        min_window: Minimum session length in slots.
        max_window: Maximum session length in slots.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        min_window: int = 4,
        max_window: int = 8,
        seed: int | None = None,
    ) -> None:
        self._min_window = min_window
        self._max_window = max_window
        self._seed = seed
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "gambler"

    def choose_action(self, observation: Observation) -> Any:
        slots = _num_slots(observation)
        length = self._rng.randint(self._min_window, self._max_window)
        max_start = max(0, slots - length)
        start = self._rng.randint(0, max_start)
        end = min(start + length, slots) - 1
        if end < start:
            return []
        return [[start, end]]

    def reset(self) -> None:
        self._rng = random.Random(self._seed)


class SmartPredictor(Strategy):
    """Linear trend extrapolation — second-order prediction.

    Computes the expected occupancy for each slot by extrapolating the
    linear trend of the last ``lookback`` days:

        prediction[s] = occ[d-1][s] + (occ[d-1][s] - occ[d-2][s])

    Selects the ``window_size``-slot contiguous block with the lowest
    predicted occupancy.  Falls back to TrendFollower logic when there
    are fewer than 2 days of history.

    Equivalent to: momentum-aware adaptive agent (ARIMA(1,1,0) analogy).
    """

    def __init__(self, window_size: int = 8, lookback: int = 3) -> None:
        self._window_size = window_size
        self._lookback = lookback

    @property
    def name(self) -> str:
        return "smart_predictor"

    def choose_action(self, observation: Observation) -> Any:
        history = _history_arrays(observation)
        slots = _num_slots(observation)
        if len(history) < 2:
            if not history:
                return _midday_interval(slots, self._window_size)
            return _get_best_window(history[-1], self._window_size, find_max=False)

        data = np.array(history[-self._lookback :])
        # Simple linear extrapolation from the last two available days
        prediction = data[-1] + (data[-1] - data[-2])
        return _get_best_window(prediction.tolist(), self._window_size, find_max=False)


class Scout(Strategy):
    """Recon-then-commit: probe one slot, then decide the main visit.

    Day strategy (two logical phases collapsed into a single action):

    Phase 1 — Choose a short random recon slot early in the day.
    Phase 2 — Based on yesterday's occupancy at that same slot, infer
               whether the bar will be quiet and plan the main session.

    Because the framework submits all slots at the start of the day,
    the Scout approximates its real-time scouting using the previous
    day's occupancy at the chosen recon slot as a proxy for today's.

    Attributes:
        recon_window: Slot range (inclusive) from which the recon slot
            is picked.  Default (0, 4) = first two hours of the day.
        main_window_size: Length of the main session in slots.
        threshold: If yesterday's recon-slot occupancy was below this,
            the main visit is planned; otherwise only a brief fallback.
        seed: Optional RNG seed.
    """

    def __init__(
        self,
        recon_window: tuple[int, int] = (0, 4),
        main_window_size: int = 7,
        threshold: int = 60,
        seed: int | None = None,
    ) -> None:
        self._recon_window = recon_window
        self._main_window_size = main_window_size
        self._threshold = threshold
        self._seed = seed
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "scout"

    def choose_action(self, observation: Observation) -> Any:
        history = _history_arrays(observation)
        slots = _num_slots(observation)
        lo, hi = self._recon_window
        hi = min(hi, slots - 1)
        recon_slot = self._rng.randint(lo, hi)

        # Recon slot as a single-point interval [recon, recon].
        intervals: list[list[int]] = [[recon_slot, recon_slot]]

        if history:
            yesterday_at_recon = history[-1][recon_slot]
            if yesterday_at_recon < self._threshold:
                # Bar was quiet at recon time → plan a long main visit
                # starting two slots after recon (one empty gap so the
                # intervals stay non-adjacent).
                start = recon_slot + 2
                end = min(start + self._main_window_size, slots) - 1
                if start < slots and end >= start:
                    intervals.append([start, end])
            else:
                # Bar was crowded → sneak in only at the end.
                tail = max(0, slots - 2)
                end = slots - 1
                if tail > recon_slot + 1 and end >= tail:
                    intervals.append([tail, end])

        return intervals

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
