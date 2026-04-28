"""GameRunner populates EpisodeResult.actions for interval-shaped games.

These tests cover the runner's population of the
``actions: list[ActionRecord]`` field on ``EpisodeResult`` for El Farol
(canonical interval-shaped action) and the no-op behaviour for non-interval
games (e.g. Prisoner's Dilemma whose actions are string labels).

The runner accepts the canonical El Farol shape — either a bare list of
``[start, end]`` pairs or ``{"intervals": [[start, end], ...]}`` — and
converts it via the module-level ``_action_to_interval_pair`` helper.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from atp.adapters.base import AgentAdapter
from atp.protocol.models import ArtifactStructured, ATPResponse, ResponseStatus
from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy
from game_envs.games.el_farol import ElFarolBar, ElFarolConfig
from game_envs.games.prisoners_dilemma import PDConfig, PrisonersDilemma
from game_envs.strategies.pd_strategies import AlwaysCooperate, AlwaysDefect

from atp_games.models import (
    ActionRecord,
    EpisodeResult,
    GameResult,
    GameRunConfig,
    IntervalPair,
)
from atp_games.runner.builtin_adapter import BuiltinAdapter
from atp_games.runner.game_runner import GameRunner, _action_to_interval_pair

# ---------------------------------------------------------------------------
# Deterministic El Farol test strategies
# ---------------------------------------------------------------------------
#
# Two tiny inline strategies producing disjoint contiguous intervals so
# the runner's interval -> IntervalPair conversion is unambiguous.


class MorningAttendee(Strategy):
    """Always attends the first three slots of the day."""

    @property
    def name(self) -> str:
        return "morning_attendee"

    def choose_action(self, observation: Observation) -> Any:
        return [[0, 2]]


class EveningAttendee(Strategy):
    """Always attends the last three slots of the day."""

    @property
    def name(self) -> str:
        return "evening_attendee"

    def choose_action(self, observation: Observation) -> Any:
        return [[13, 15]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_el_farol_game_result(
    *,
    num_rounds: int = 3,
    capacity_threshold: int = 2,
) -> GameResult:
    """Run a 2-agent El Farol match and return the full GameResult."""
    config = ElFarolConfig(
        num_players=2,
        num_rounds=num_rounds,
        capacity_threshold=capacity_threshold,
    )
    game = ElFarolBar(config)
    agents = {
        "player_0": BuiltinAdapter(MorningAttendee()),
        "player_1": BuiltinAdapter(EveningAttendee()),
    }
    runner = GameRunner()
    return await runner.run_game(game, agents, GameRunConfig(episodes=1))


async def _run_el_farol_match(
    *,
    num_rounds: int = 3,
    capacity_threshold: int = 2,
) -> EpisodeResult:
    """Run a 2-agent El Farol match and return the first episode result."""
    result = await _run_el_farol_game_result(
        num_rounds=num_rounds,
        capacity_threshold=capacity_threshold,
    )
    return result.episodes[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEpisodeResultActionsField:
    """The ``actions`` field on EpisodeResult."""

    def test_episode_actions_field_exists_and_empty_by_default(self) -> None:
        # GIVEN an EpisodeResult constructed with only the required args
        ep = EpisodeResult(episode=0, payoffs={"p0": 1.0})

        # THEN the actions field is present and defaults to []
        assert ep.actions == []


class TestRunnerBuildsActionRecordsForElFarol:
    """The runner populates EpisodeResult.actions for El Farol runs."""

    @pytest.mark.anyio
    async def test_el_farol_3day_2agent_run_yields_6_records(self) -> None:
        # GIVEN a 3-day, 2-agent El Farol match with contiguous-pick agents
        ep = await _run_el_farol_match(num_rounds=3)

        # THEN 3 days x 2 agents = 6 ActionRecords are produced
        assert len(ep.actions) == 6
        for record in ep.actions:
            assert isinstance(record, ActionRecord)

    @pytest.mark.anyio
    async def test_day_ordering(self) -> None:
        # GIVEN a 3-day, 2-agent El Farol match
        ep = await _run_el_farol_match(num_rounds=3)

        # WHEN records are grouped by agent_id
        by_agent: dict[str, list[int]] = {}
        for record in ep.actions:
            by_agent.setdefault(record.agent_id, []).append(record.day)

        # THEN every agent has days [1, 2, 3] in increasing order
        #      (days are 1-indexed to match the dashboard UX)
        assert set(by_agent.keys()) == {"player_0", "player_1"}
        for agent_id, days in by_agent.items():
            assert days == [1, 2, 3], (
                f"agent {agent_id} days were {days}, expected [1, 2, 3]"
            )

    @pytest.mark.anyio
    async def test_per_day_payoff_sum_equals_expected(self) -> None:
        # GIVEN a 3-day, 2-agent El Farol match with disjoint contiguous picks
        # (morning vs evening, threshold=2) — each player has happy=3, crowded=0
        # per day, so per-day payoff is 3 and the 3-day sum is 9.
        ep = await _run_el_farol_match(num_rounds=3)

        for pid in ("player_0", "player_1"):
            daily_sum = sum(r.payoff for r in ep.actions if r.agent_id == pid)
            assert daily_sum == pytest.approx(9.0), (
                f"expected sum of per-day payoffs for {pid} to equal 9.0 "
                f"(3 happy slots per day × 3 days), got {daily_sum}"
            )

    @pytest.mark.anyio
    async def test_picks_match_slot_picks(self) -> None:
        # GIVEN a 3-day, 2-agent El Farol match
        ep = await _run_el_farol_match(num_rounds=3)

        # THEN the cached picks on each record mirror the intervals
        assert ep.actions, "no ActionRecords produced"
        record = ep.actions[0]
        assert isinstance(record.intervals, IntervalPair)
        assert record.picks == record.intervals.covered_slots()

    @pytest.mark.anyio
    async def test_match_id_stable_across_days_for_same_episode(self) -> None:
        # GIVEN a 3-day, 2-agent El Farol match
        ep = await _run_el_farol_match(num_rounds=3)

        # THEN all 6 records share the same match_id
        assert ep.actions, "no ActionRecords produced"
        match_ids = {record.match_id for record in ep.actions}
        assert len(match_ids) == 1, (
            f"expected a single match_id across the episode, got {match_ids}"
        )

    @pytest.mark.anyio
    async def test_retry_count_default_zero_in_happy_path(self) -> None:
        # GIVEN a clean 3-day, 2-agent El Farol match (no validation errors)
        ep = await _run_el_farol_match(num_rounds=3)

        # THEN every record reports zero retries
        assert ep.actions, "no ActionRecords produced"
        for record in ep.actions:
            assert record.retry_count == 0, (
                f"expected retry_count=0 for {record.agent_id} day "
                f"{record.day}, got {record.retry_count}"
            )


class TestRunIdAndMatchIdUniqueness:
    """Independent runs produce distinct run_id and match_id values."""

    @pytest.mark.anyio
    async def test_match_id_differs_across_independent_runs(self) -> None:
        # GIVEN two separate run_game() invocations with identical config/agents
        run_a = await _run_el_farol_game_result(num_rounds=2)
        run_b = await _run_el_farol_game_result(num_rounds=2)

        # WHEN collecting match_ids from each run's first episode
        match_ids_a = {record.match_id for record in run_a.episodes[0].actions}
        match_ids_b = {record.match_id for record in run_b.episodes[0].actions}

        # THEN the two sets are disjoint — no match_id leaks across runs
        assert match_ids_a, "expected ActionRecords in run A"
        assert match_ids_b, "expected ActionRecords in run B"
        assert match_ids_a.isdisjoint(match_ids_b), (
            f"expected disjoint match_ids across independent runs, "
            f"got overlap {match_ids_a & match_ids_b}"
        )

    @pytest.mark.anyio
    async def test_game_result_run_id_is_populated_and_unique(self) -> None:
        # GIVEN two successive runs of the same El Farol config
        run_a = await _run_el_farol_game_result(num_rounds=1)
        run_b = await _run_el_farol_game_result(num_rounds=1)

        # THEN each result has a non-empty run_id string
        assert isinstance(run_a.run_id, str) and run_a.run_id, (
            f"expected non-empty run_id on run A, got {run_a.run_id!r}"
        )
        assert isinstance(run_b.run_id, str) and run_b.run_id, (
            f"expected non-empty run_id on run B, got {run_b.run_id!r}"
        )

        # AND the two run_ids differ
        assert run_a.run_id != run_b.run_id, (
            f"expected distinct run_ids across runs, got {run_a.run_id!r}"
        )

    @pytest.mark.anyio
    async def test_match_id_contains_run_id(self) -> None:
        # GIVEN a single El Farol run
        result = await _run_el_farol_game_result(num_rounds=2)

        # THEN every ActionRecord.match_id starts with "{game_name}#{run_id}"
        assert result.episodes[0].actions, "expected ActionRecords to be produced"
        expected_prefix = f"{result.game_name}#{result.run_id}"
        for record in result.episodes[0].actions:
            assert record.match_id.startswith(expected_prefix), (
                f"expected match_id {record.match_id!r} to start with "
                f"{expected_prefix!r}"
            )


class TestRunnerSkipsActionRecordsForNonIntervalGames:
    """Games whose actions aren't interval-shaped yield no ActionRecords."""

    @pytest.mark.anyio
    async def test_non_el_farol_game_yields_empty_actions(self) -> None:
        # GIVEN a 1-round Prisoner's Dilemma match (actions are strings)
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        runner = GameRunner()
        result = await runner.run_game(game, agents, GameRunConfig(episodes=1))
        ep = result.episodes[0]

        # THEN no ActionRecords are built (PD actions aren't interval-shaped)
        assert ep.actions == []


# ---------------------------------------------------------------------------
# Intent plumbing tests
# ---------------------------------------------------------------------------


def _make_mock_el_farol_adapter(
    *,
    action: Any,
    intent: str | None = None,
) -> AgentAdapter:
    """Create an AsyncMock adapter returning a structured El Farol action.

    When ``intent`` is ``None`` the ``intent`` key is omitted from the
    response payload entirely (rather than sent as an explicit ``None``),
    so the ActionMapper sees no ``intent`` field at all.
    """
    mock_agent = AsyncMock(spec=AgentAdapter)
    mock_agent.adapter_type = "mock"

    data: dict[str, Any] = {"action": action}
    if intent is not None:
        data["intent"] = intent

    mock_agent.execute.return_value = ATPResponse(
        task_id="test",
        status=ResponseStatus.COMPLETED,
        artifacts=[
            ArtifactStructured(
                name="game_action",
                data=data,
            ),
        ],
    )
    return mock_agent


async def _run_minimal_el_farol_with_mock(
    *,
    player_0: AgentAdapter,
    player_1: AgentAdapter,
    max_retries: int = 2,
) -> EpisodeResult:
    """Run a 2-player, 1-round El Farol match with mock adapters."""
    config = ElFarolConfig(
        num_players=2,
        num_rounds=1,
        capacity_threshold=2,
    )
    game = ElFarolBar(config)
    agents = {"player_0": player_0, "player_1": player_1}
    runner = GameRunner()
    result = await runner.run_game(
        game,
        agents,
        GameRunConfig(episodes=1, max_retries=max_retries),
    )
    return result.episodes[0]


class TestActionRecordIntentPlumbing:
    """GameAction.intent is plumbed through to ActionRecord.intent."""

    @pytest.mark.anyio
    async def test_action_record_preserves_intent_from_agent_response(
        self,
    ) -> None:
        # GIVEN a 1-day 2-player El Farol match where player_0's mock
        #       adapter returns both a valid action and an intent string
        mock_agent = _make_mock_el_farol_adapter(
            action=[[0, 2]],
            intent="avoid the crowd",
        )
        ep = await _run_minimal_el_farol_with_mock(
            player_0=mock_agent,
            player_1=BuiltinAdapter(EveningAttendee()),
        )

        # THEN at least one ActionRecord carries the exact intent string
        assert ep.actions, "expected ActionRecords to be produced"
        intents = [r.intent for r in ep.actions if r.agent_id == "player_0"]
        assert "avoid the crowd" in intents, (
            f"expected 'avoid the crowd' in player_0 intents, got {intents}"
        )

    @pytest.mark.anyio
    async def test_action_record_intent_none_when_response_omits_intent(
        self,
    ) -> None:
        # GIVEN both players return valid actions without any intent field
        ep = await _run_minimal_el_farol_with_mock(
            player_0=_make_mock_el_farol_adapter(action=[[0, 2]]),
            player_1=_make_mock_el_farol_adapter(action=[[13, 15]]),
        )

        # THEN every ActionRecord has intent is None
        assert ep.actions, "expected ActionRecords to be produced"
        for record in ep.actions:
            assert record.intent is None, (
                f"expected intent=None for {record.agent_id} day "
                f"{record.day}, got {record.intent!r}"
            )

    @pytest.mark.anyio
    async def test_action_record_intent_is_none_on_fallback_path(self) -> None:
        # GIVEN player_1 always returns an invalid action (wrong type) so
        # the runner exhausts retries and falls back to the default.
        # ``ElFarolActionSpace.sample()`` now returns an interval-shaped
        # action (e.g. ``[[start, end]]``), so the fallback IS
        # interval-convertible and player_1 still gets an ActionRecord —
        # but the original intent must be discarded since the action
        # itself was replaced.
        invalid_agent = _make_mock_el_farol_adapter(
            action="not-an-interval-list",
            intent="this intent should be discarded",
        )
        ep = await _run_minimal_el_farol_with_mock(
            player_0=BuiltinAdapter(MorningAttendee()),
            player_1=invalid_agent,
            max_retries=1,
        )

        # THEN both players have records (1 day x 2 agents = 2 records),
        # and player_1's record has intent=None (the agent's original
        # intent string is discarded along with the rejected action).
        assert len(ep.actions) == 2
        player_1_records = [r for r in ep.actions if r.agent_id == "player_1"]
        assert player_1_records, "expected player_1 to have an ActionRecord"
        for record in player_1_records:
            assert record.intent is None, (
                f"expected intent=None on fallback path, got {record.intent!r}"
            )


# ---------------------------------------------------------------------------
# IntervalPair cap plumbing tests
# ---------------------------------------------------------------------------


async def _run_el_farol_with_config(
    config: ElFarolConfig,
    *,
    player_0_action: Any,
    player_1_action: Any,
) -> EpisodeResult:
    """Run a 2-player, 1-round El Farol match with mock adapters."""
    game = ElFarolBar(config)
    agents = {
        "player_0": _make_mock_el_farol_adapter(action=player_0_action),
        "player_1": _make_mock_el_farol_adapter(action=player_1_action),
    }
    runner = GameRunner()
    result = await runner.run_game(game, agents, GameRunConfig(episodes=1))
    return result.episodes[0]


class TestActionRecordIntervalPairCap:
    """ActionRecord.intervals carries the match's configured slot cap."""

    @pytest.mark.anyio
    async def test_action_record_interval_pair_honors_config_max_total_slots(
        self,
    ) -> None:
        # GIVEN an El Farol match with max_total_slots=4 (custom cap)
        config = ElFarolConfig(
            num_players=2,
            num_rounds=1,
            num_slots=16,
            max_total_slots=4,
            max_intervals=2,
            capacity_threshold=2,
        )

        # WHEN both players submit a valid 2-slot action
        ep = await _run_el_farol_with_config(
            config,
            player_0_action=[[0, 1]],
            player_1_action=[[0, 1]],
        )

        # THEN every ActionRecord.intervals.max_total_slots is the configured
        # cap (4), not the old len(sorted_slots)-or-8 default
        assert ep.actions, "expected ActionRecords to be produced"
        for record in ep.actions:
            assert record.intervals.max_total_slots == 4, (
                f"expected max_total_slots=4 for {record.agent_id} day "
                f"{record.day}, got {record.intervals.max_total_slots}"
            )

    @pytest.mark.anyio
    async def test_action_record_interval_pair_preserves_num_slots_from_config(
        self,
    ) -> None:
        # GIVEN an El Farol match with num_slots=8
        config = ElFarolConfig(
            num_players=2,
            num_rounds=1,
            num_slots=8,
            max_total_slots=4,
            max_intervals=2,
            capacity_threshold=2,
        )

        # WHEN both players submit a valid 2-slot action
        ep = await _run_el_farol_with_config(
            config,
            player_0_action=[[0, 1]],
            player_1_action=[[0, 1]],
        )

        # THEN every ActionRecord.intervals.num_slots is 8
        assert ep.actions, "expected ActionRecords to be produced"
        for record in ep.actions:
            assert record.intervals.num_slots == 8, (
                f"expected num_slots=8 for {record.agent_id} day "
                f"{record.day}, got {record.intervals.num_slots}"
            )

    @pytest.mark.anyio
    async def test_action_record_interval_pair_uses_default_config_max_total_slots(
        self,
    ) -> None:
        # GIVEN a default ElFarolConfig (max_total_slots=8), overriding only
        # the number of players/rounds/threshold so the match runs quickly
        config = ElFarolConfig(
            num_players=2,
            num_rounds=1,
            capacity_threshold=2,
        )
        assert config.max_total_slots == 8, (
            "pre-condition: default ElFarolConfig.max_total_slots should be 8"
        )

        # WHEN both players submit a small valid action
        ep = await _run_el_farol_with_config(
            config,
            player_0_action=[[0, 1]],
            player_1_action=[[0, 1]],
        )

        # THEN every ActionRecord.intervals.max_total_slots equals the default 8
        assert ep.actions, "expected ActionRecords to be produced"
        for record in ep.actions:
            assert record.intervals.max_total_slots == 8, (
                f"expected default max_total_slots=8 for {record.agent_id} "
                f"day {record.day}, got {record.intervals.max_total_slots}"
            )


# ---------------------------------------------------------------------------
# Direct unit tests for the _action_to_interval_pair helper
# ---------------------------------------------------------------------------


class TestActionToIntervalPair:
    """Unit tests for the module-level _action_to_interval_pair helper."""

    def test_empty_list_returns_empty_pair(self) -> None:
        # GIVEN an empty list (stay home)
        result = _action_to_interval_pair(
            [], num_slots=16, max_total_slots=8
        )
        # THEN the returned pair is empty but well-formed
        assert result is not None
        assert result.first == ()
        assert result.second == ()
        assert result.num_slots == 16
        assert result.max_total_slots == 8

    def test_single_pair_list(self) -> None:
        # GIVEN a single [start, end] pair
        result = _action_to_interval_pair(
            [[0, 2]], num_slots=16, max_total_slots=8
        )
        # THEN the IntervalPair holds that pair as ``first``
        assert result is not None
        assert result.first == (0, 2)
        assert result.second == ()

    def test_two_pairs_sorted_by_start(self) -> None:
        # GIVEN two pairs supplied out of order
        result = _action_to_interval_pair(
            [[10, 12], [0, 1]], num_slots=16, max_total_slots=8
        )
        # THEN the pair is normalised by start index
        assert result is not None
        assert result.first == (0, 1)
        assert result.second == (10, 12)

    def test_dict_with_intervals(self) -> None:
        # GIVEN the canonical dict shape
        result = _action_to_interval_pair(
            {"intervals": [[3, 5]]}, num_slots=16, max_total_slots=8
        )
        # THEN intervals are extracted and IntervalPair is built
        assert result is not None
        assert result.first == (3, 5)

    def test_dict_with_empty_intervals(self) -> None:
        # GIVEN dict shape with an empty list
        result = _action_to_interval_pair(
            {"intervals": []}, num_slots=16, max_total_slots=8
        )
        # THEN an empty IntervalPair is returned
        assert result is not None
        assert result.first == ()
        assert result.second == ()

    def test_dict_without_intervals_key_returns_none(self) -> None:
        # GIVEN a dict that is not interval-shaped (e.g. legacy "slots")
        result = _action_to_interval_pair(
            {"slots": [0, 1, 2]}, num_slots=16, max_total_slots=8
        )
        # THEN None — caller bails out
        assert result is None

    def test_none_input_returns_none(self) -> None:
        result = _action_to_interval_pair(
            None, num_slots=16, max_total_slots=8
        )
        assert result is None

    def test_string_input_returns_none(self) -> None:
        result = _action_to_interval_pair(
            "cooperate", num_slots=16, max_total_slots=8
        )
        assert result is None

    def test_more_than_two_pairs_returns_none(self) -> None:
        result = _action_to_interval_pair(
            [[0, 0], [2, 2], [4, 4]], num_slots=16, max_total_slots=8
        )
        assert result is None

    def test_malformed_pair_shape_returns_none(self) -> None:
        # GIVEN a triple instead of a [start, end] pair
        result = _action_to_interval_pair(
            [[0, 1, 2]], num_slots=16, max_total_slots=8
        )
        # THEN None
        assert result is None

    def test_non_int_bound_returns_none(self) -> None:
        # GIVEN a non-int bound
        result = _action_to_interval_pair(
            [["a", 2]], num_slots=16, max_total_slots=8
        )
        assert result is None

    def test_out_of_range_pair_returns_none(self) -> None:
        # GIVEN a pair that exceeds num_slots
        result = _action_to_interval_pair(
            [[0, 99]], num_slots=16, max_total_slots=8
        )
        # THEN IntervalPair raises and the helper swallows it as None
        assert result is None

    def test_exceeds_max_total_slots_returns_none(self) -> None:
        # GIVEN a pair covering 5 slots while max_total_slots=4
        result = _action_to_interval_pair(
            [[0, 4]], num_slots=16, max_total_slots=4
        )
        # THEN None
        assert result is None

    def test_max_total_slots_propagated(self) -> None:
        # GIVEN a custom cap
        result = _action_to_interval_pair(
            [[0, 1]], num_slots=16, max_total_slots=4
        )
        # THEN the cap is preserved on the returned IntervalPair
        assert result is not None
        assert result.max_total_slots == 4
