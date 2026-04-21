"""Phase 2 TDD tests: GameRunner populates EpisodeResult.actions.

These tests drive the Phase 2 runner change that adds a new
``actions: list[ActionRecord]`` field to ``EpisodeResult`` and has
``GameRunner._run_episode`` build per-day ``ActionRecord`` objects for
El Farol (and any other game whose actions are ``list[int]``).

The change is additive: existing ``actions_log`` / ``history`` fields
are untouched.  For games whose actions are not list-of-ints (e.g.
Prisoner's Dilemma, whose actions are string labels), ``ep.actions``
stays an empty list because no ``IntervalPair`` can be built.

Expected failure modes before Phase 2 implementation:
  * ``EpisodeResult`` has no ``actions`` attribute yet, so tests that
    access ``ep.actions`` fail with AttributeError on construction or
    attribute lookup.
  * Even if the field is added as default ``[]``, the runner does not
    populate it, so length / content assertions fail.
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

from atp_games.models import ActionRecord, EpisodeResult, GameRunConfig, IntervalPair
from atp_games.runner.builtin_adapter import BuiltinAdapter
from atp_games.runner.game_runner import GameRunner

# ---------------------------------------------------------------------------
# Deterministic El Farol test strategies
# ---------------------------------------------------------------------------
#
# The built-in El Farol strategies (Traditionalist, TrendFollower, ...) all
# produce contiguous windows, but their exact picks depend on attendance
# history and may collide in small 2-player matches.  For deterministic,
# disjoint, always-contiguous picks, we define two tiny inline strategies:
# a "morning" attendee that always picks slots [0, 1, 2], and an "evening"
# attendee that always picks slots [13, 14, 15].  Both are single
# contiguous intervals, so the runner's list[int] -> IntervalPair
# conversion must succeed.


class MorningAttendee(Strategy):
    """Always attends the first three slots of the day."""

    @property
    def name(self) -> str:
        return "morning_attendee"

    def choose_action(self, observation: Observation) -> Any:
        return [0, 1, 2]


class EveningAttendee(Strategy):
    """Always attends the last three slots of the day."""

    @property
    def name(self) -> str:
        return "evening_attendee"

    def choose_action(self, observation: Observation) -> Any:
        return [13, 14, 15]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_el_farol_match(
    *,
    num_rounds: int = 3,
    capacity_threshold: int = 2,
) -> EpisodeResult:
    """Run a 2-agent El Farol match and return the first episode result."""
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
    result = await runner.run_game(game, agents, GameRunConfig(episodes=1))
    return result.episodes[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEpisodeResultActionsField:
    """The new ``actions`` field on EpisodeResult."""

    def test_episode_actions_field_exists_and_empty_by_default(self) -> None:
        # GIVEN an EpisodeResult constructed with only the required args
        ep = EpisodeResult(episode=0, payoffs={"p0": 1.0})

        # THEN the new actions field is present and defaults to []
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

        # THEN the sum of per-day ActionRecord.payoff for each agent matches
        # the deterministic per-day delta from the El Farol step resolution
        # (not ep.payoffs, which uses the non-linear t_happy/max(t_crowded, 0.1)
        # formula that is intentionally out of scope for this plan).
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


class TestRunnerSkipsActionRecordsForNonIntervalGames:
    """Games whose actions aren't list[int] yield no ActionRecords."""

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

        # THEN no ActionRecords are built (PD actions aren't list-of-ints)
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
            action=[0, 1, 2],
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
            player_0=_make_mock_el_farol_adapter(action=[0, 1, 2]),
            player_1=_make_mock_el_farol_adapter(action=[13, 14, 15]),
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
        # GIVEN player_0 always returns an invalid action (wrong type) so
        #       the runner exhausts retries and falls back to the default
        invalid_agent = _make_mock_el_farol_adapter(
            action="not-a-slot-list",
            intent="this intent should be discarded",
        )
        ep = await _run_minimal_el_farol_with_mock(
            player_0=invalid_agent,
            player_1=BuiltinAdapter(EveningAttendee()),
            max_retries=1,
        )

        # THEN player_0's fallback ActionRecord(s) must not carry any intent,
        #      even though the agent's response contained one on every attempt
        assert ep.actions, "expected ActionRecords to be produced"
        player_0_records = [r for r in ep.actions if r.agent_id == "player_0"]
        assert player_0_records, "expected at least one record for player_0"
        for record in player_0_records:
            assert record.intent is None, (
                f"expected fallback path to drop intent, got {record.intent!r}"
            )
