"""End-to-end interval-action support through the GameRunner.

The runner and ElFarol action space accept the canonical interval shape
in two equivalent forms — list-of-pairs ``[[start, end], ...]`` and
dict-with-intervals ``{"intervals": [[start, end], ...]}``. These tests
verify that agents submitting either shape produce correct
``ActionRecord`` entries on ``EpisodeResult.actions`` after running
through the full ``GameRunner`` pipeline.

Style mirrors ``test_runner_action_records.py`` — inline strategies
for deterministic, contiguous picks and a small async helper that
runs a 2-agent match and returns the first episode.
"""

from __future__ import annotations

from typing import Any

import pytest
from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy
from game_envs.games.el_farol import ElFarolBar, ElFarolConfig

from atp_games.models import ActionRecord, EpisodeResult, GameRunConfig
from atp_games.runner.builtin_adapter import BuiltinAdapter
from atp_games.runner.game_runner import GameRunner

# ---------------------------------------------------------------------------
# Test strategies — each submits one of the interval-action shapes
# ---------------------------------------------------------------------------


class PairFormatMorning(Strategy):
    """Always returns a list-of-pairs interval action."""

    @property
    def name(self) -> str:
        return "pair_morning"

    def choose_action(self, observation: Observation) -> Any:
        return [[0, 2]]  # one interval = 3 slots [0, 1, 2]


class DictFormatEvening(Strategy):
    """Always returns a dict-with-intervals action."""

    @property
    def name(self) -> str:
        return "dict_evening"

    def choose_action(self, observation: Observation) -> Any:
        return {"intervals": [[13, 15]]}  # one interval = 3 slots [13, 14, 15]


class TwoIntervalMidday(Strategy):
    """Returns two disjoint intervals via dict shape."""

    @property
    def name(self) -> str:
        return "two_interval_midday"

    def choose_action(self, observation: Observation) -> Any:
        return {"intervals": [[4, 5], [8, 9]]}  # total 4 slots


class ThreeIntervalsAgent(Strategy):
    """Invalid: submits three intervals (exceeds max_intervals=2)."""

    @property
    def name(self) -> str:
        return "three_intervals"

    def choose_action(self, observation: Observation) -> Any:
        return {"intervals": [[0, 0], [2, 2], [4, 4]]}


class OverlappingIntervalsAgent(Strategy):
    """Invalid: submits two overlapping intervals."""

    @property
    def name(self) -> str:
        return "overlapping_intervals"

    def choose_action(self, observation: Observation) -> Any:
        return {"intervals": [[0, 4], [3, 7]]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_match(
    *,
    strategy_a: Strategy,
    strategy_b: Strategy,
    num_rounds: int,
    num_players: int = 2,
    capacity_threshold: int = 2,
    num_slots: int = 16,
) -> EpisodeResult:
    """Run a 2-agent El Farol match and return the first episode."""
    config = ElFarolConfig(
        num_players=num_players,
        num_rounds=num_rounds,
        capacity_threshold=capacity_threshold,
        num_slots=num_slots,
    )
    game = ElFarolBar(config)
    agents = {
        "player_0": BuiltinAdapter(strategy_a),
        "player_1": BuiltinAdapter(strategy_b),
    }
    runner = GameRunner()
    result = await runner.run_game(game, agents, GameRunConfig(episodes=1))
    return result.episodes[0]


# ---------------------------------------------------------------------------
# Tests — happy paths for every interval shape
# ---------------------------------------------------------------------------


class TestIntervalFormatsYieldActionRecords:
    """Agents submitting any supported interval shape produce correct records."""

    @pytest.mark.anyio
    async def test_pair_format_agent_produces_action_records(self) -> None:
        # GIVEN a 3-day, 2-agent match (list-of-pairs vs dict-with-intervals)
        ep = await _run_match(
            strategy_a=PairFormatMorning(),
            strategy_b=DictFormatEvening(),
            num_rounds=3,
            num_players=2,
            capacity_threshold=2,
            num_slots=16,
        )

        # THEN 3 days x 2 agents = 6 records are produced
        assert len(ep.actions) == 6
        for record in ep.actions:
            assert isinstance(record, ActionRecord)

        # AND PairFormatMorning records resolve to interval (0, 2)
        morning_records = [r for r in ep.actions if r.agent_id == "player_0"]
        assert len(morning_records) == 3
        for record in morning_records:
            assert record.picks == (0, 1, 2)
            assert record.intervals.first == (0, 2)
            assert record.intervals.second == ()
            assert record.num_visits == 1
            assert record.total_slots == 3

        # AND DictFormatEvening records resolve to interval (13, 15)
        evening_records = [r for r in ep.actions if r.agent_id == "player_1"]
        assert len(evening_records) == 3
        for record in evening_records:
            assert record.picks == (13, 14, 15)
            assert record.intervals.first == (13, 15)

    @pytest.mark.anyio
    async def test_two_interval_dict_format_records_both_intervals(self) -> None:
        # GIVEN a 2-day match where player_0 submits TWO disjoint intervals
        ep = await _run_match(
            strategy_a=TwoIntervalMidday(),
            strategy_b=PairFormatMorning(),
            num_rounds=2,
            num_players=2,
            capacity_threshold=2,
            num_slots=16,
        )

        # THEN player_0's records carry both intervals
        midday_records = [r for r in ep.actions if r.agent_id == "player_0"]
        assert len(midday_records) == 2
        for record in midday_records:
            assert record.picks == (4, 5, 8, 9)
            assert record.intervals.first == (4, 5)
            assert record.intervals.second == (8, 9)
            assert record.num_visits == 2
            assert record.total_slots == 4

    @pytest.mark.anyio
    async def test_per_day_payoff_sum_matches_expected(self) -> None:
        # GIVEN disjoint contiguous picks (morning 0-2 vs evening 13-15),
        # threshold=2 and only 1 attendee per slot — every slot is happy.
        ep = await _run_match(
            strategy_a=PairFormatMorning(),
            strategy_b=DictFormatEvening(),
            num_rounds=3,
            num_players=2,
            capacity_threshold=2,
            num_slots=16,
        )

        # THEN each agent earns happy=3, crowded=0 per day → sum 9.0 over 3 days
        for pid in ("player_0", "player_1"):
            daily_sum = sum(r.payoff for r in ep.actions if r.agent_id == pid)
            assert daily_sum == pytest.approx(9.0), (
                f"expected 9.0 (3 happy slots × 3 days) for {pid}, got {daily_sum}"
            )


# ---------------------------------------------------------------------------
# Tests — invalid interval input triggers retry / fallback gracefully
# ---------------------------------------------------------------------------


class TestActionSpaceRejectsInvalidIntervalInput:
    """Invalid interval input is rejected by ``contains`` and the runner
    falls back to a default action after max_retries.

    ``ElFarolActionSpace.sample()`` now returns an interval-shaped action
    (e.g. ``[[start, end]]``), so the fallback path is interval-convertible
    and ActionRecords are produced for both players — even the one whose
    original action was rejected.
    """

    @pytest.mark.anyio
    async def test_too_many_intervals_triggers_retry_or_fallback(self) -> None:
        # GIVEN a 1-day match where player_0 always submits 3 intervals
        # (exceeds max_intervals=2) — invalid per the action space.
        ep = await _run_match(
            strategy_a=ThreeIntervalsAgent(),
            strategy_b=PairFormatMorning(),
            num_rounds=1,
            num_players=2,
            capacity_threshold=2,
            num_slots=16,
        )

        # THEN the match completes; both players get an ActionRecord
        # (1 day x 2 agents) — player_0's via the interval-shaped fallback.
        assert len(ep.round_payoffs) == 1
        assert len(ep.actions) == 2
        agent_ids = {r.agent_id for r in ep.actions}
        assert agent_ids == {"player_0", "player_1"}

    @pytest.mark.anyio
    async def test_overlapping_intervals_rejected(self) -> None:
        # GIVEN a 1-day match where player_0 submits two overlapping
        # intervals (invalid per the action space).
        ep = await _run_match(
            strategy_a=OverlappingIntervalsAgent(),
            strategy_b=PairFormatMorning(),
            num_rounds=1,
            num_players=2,
            capacity_threshold=2,
            num_slots=16,
        )

        # THEN the match completes; both players get an ActionRecord
        # (1 day x 2 agents) — player_0's via the interval-shaped fallback.
        assert len(ep.round_payoffs) == 1
        assert len(ep.actions) == 2
        agent_ids = {r.agent_id for r in ep.actions}
        assert agent_ids == {"player_0", "player_1"}


# ---------------------------------------------------------------------------
# Tests — config-driven max_total_slots is honoured
# ---------------------------------------------------------------------------


class TestConfigDrivenMaxTotalSlots:
    """Phase 5's ElFarolConfig.max_total_slots is threaded into the space."""

    def test_action_space_enforces_config_max_total_slots(self) -> None:
        # GIVEN an El Farol config with max_total_slots=4
        game = ElFarolBar(ElFarolConfig(num_players=8, num_slots=16, max_total_slots=4))
        aspace = game.action_space("player_0")

        # THEN a 5-slot interval [[0, 4]] is rejected (5 > 4)
        assert aspace.contains([[0, 4]]) is False

        # AND a 4-slot interval [[0, 3]] is accepted
        assert aspace.contains([[0, 3]]) is True
