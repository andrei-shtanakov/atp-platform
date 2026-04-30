"""Unit tests for El Farol tournament → DashboardPayload reshape (LABS-106)."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.tournament.models import (
    Action,
    Participant,
    Round,
    Tournament,
)
from atp.dashboard.v2.routes.el_farol_dashboard import DashboardAgent
from atp.dashboard.v2.routes.el_farol_from_tournament import (
    _build_agents_from_participants,
    _build_decision_from_action,
    _build_rounds_from_actions,
    _reshape_from_tournament,
)


@pytest.mark.anyio
async def test_single_round_two_builtins(db_session: AsyncSession) -> None:
    """Smallest useful tournament: 2 builtin participants, 1 completed
    round with both actions recorded.
    """
    t = Tournament(
        game_type="el_farol",
        num_players=2,
        total_rounds=1,
        status="completed",
    )
    db_session.add(t)
    await db_session.flush()

    p1 = Participant(
        tournament_id=t.id,
        agent_name="traditionalist",
        builtin_strategy="el_farol/traditionalist",
    )
    p2 = Participant(
        tournament_id=t.id,
        agent_name="random",
        builtin_strategy="el_farol/random",
    )
    db_session.add_all([p1, p2])
    await db_session.flush()

    r = Round(
        tournament_id=t.id,
        round_number=1,
        status="completed",
    )
    db_session.add(r)
    await db_session.flush()

    db_session.add_all(
        [
            Action(
                round_id=r.id,
                participant_id=p1.id,
                action_data={"intervals": [[0, 1]]},
                payoff=2.0,
            ),
            Action(
                round_id=r.id,
                participant_id=p2.id,
                action_data={"intervals": [[1, 2]]},
                payoff=1.0,
            ),
        ]
    )
    await db_session.commit()

    payload = await _reshape_from_tournament(t.id, db_session)

    assert payload.shape_version == 1
    assert payload.NUM_SLOTS == 16
    assert payload.NUM_DAYS == 1
    assert len(payload.AGENTS) == 2
    assert len(payload.DATA) == 1
    day1 = payload.DATA[0]
    assert day1.round == 1
    picks_by_agent = {d.agent: d.picks for d in day1.decisions}
    assert picks_by_agent == {
        "traditionalist": [0, 1],
        "random": [1, 2],
    }


def test_build_agents_mixes_builtins_and_users() -> None:
    p_builtin = Participant(
        id=1,
        tournament_id=99,
        agent_name="traditionalist",
        builtin_strategy="el_farol/traditionalist",
        user_id=None,
    )
    p_user = Participant(
        id=2,
        tournament_id=99,
        agent_name="my-agent",
        builtin_strategy=None,
        user_id=42,
        agent_id=7,
    )

    agents = _build_agents_from_participants([p_builtin, p_user])

    assert [a.id for a in agents] == ["traditionalist", "my-agent"]
    assert agents[0].profile == "traditionalist"
    assert agents[0].user == "unknown"
    assert agents[1].profile == ""
    assert agents[1].user == "42"
    assert agents[0].color == "#6e7781"
    assert agents[1].color == "#6e7781"


def test_build_rounds_computes_slot_attendance_and_over_slots() -> None:
    """3 participants — 2 pick slot 0, 1 picks slot 5. Capacity
    threshold 2 → slot 0 is at threshold (renders -1 per the canonical
    rule att < threshold → +1), slot 5 is under (+1). over_slots
    counts slots where att >= threshold: one slot.
    """
    agents = [
        DashboardAgent(id=f"a{i}", color="#6e7781", user="unknown") for i in range(1, 4)
    ]

    # Pair shape is (agent_roster_index, Action ORM row). Action columns
    # are all nullable / have defaults so kwargs-only construction is
    # safe outside a session — no DB binding required.
    actions_by_round: dict[int, list[tuple[int, Action]]] = {
        1: [
            (0, Action(action_data={"intervals": [[0, 0]]}, payoff=1.0)),
            (1, Action(action_data={"intervals": [[0, 0]]}, payoff=-1.0)),
            (2, Action(action_data={"intervals": [[5, 5]]}, payoff=1.0)),
        ],
    }

    rounds = _build_rounds_from_actions(
        actions_by_round=actions_by_round,
        agents=agents,
        num_slots=16,
        capacity_threshold=2,
    )

    assert len(rounds) == 1
    day1 = rounds[0]
    assert day1.round == 1
    expected_attendance = [0] * 16
    expected_attendance[0] = 2
    expected_attendance[5] = 1
    assert day1.slotAttendance == expected_attendance
    assert day1.overSlots == 1
    decs = {d.agent: d for d in day1.decisions}
    assert decs["a1"].picks == [0]
    assert decs["a1"].slotPayoffs[0].attendance == 2
    assert decs["a1"].slotPayoffs[0].payoff == -1
    assert decs["a3"].slotPayoffs[0].payoff == 1


# ---------------------------------------------------------------------------
# Tier-2 telemetry projection (LABS observability)
# ---------------------------------------------------------------------------


def test_build_decision_forwards_all_tier2_telemetry_fields() -> None:
    """All five tier-2 fields on the Action row must round-trip onto the
    DashboardDecision verbatim — the drawer's DEBUG · OBSERVABILITY panel
    reads them directly off the projection.
    """
    # GIVEN an Action carrying every tier-2 telemetry column
    action = Action(
        action_data={"intervals": [[3, 3]]},
        payoff=1.0,
        model_id="gpt-4o-mini-2024-07-18",
        tokens_in=512,
        tokens_out=128,
        cost_usd=0.000234,
        decide_ms=874,
    )

    # WHEN we project it to a DashboardDecision (slot 3 alone, under cap)
    decision = _build_decision_from_action(
        agent_id="alice",
        action=action,
        slot_attendance=[0, 0, 0, 1, 0],
        capacity_threshold=2,
    )

    # THEN every tier-2 field flows through unchanged
    assert decision.agent == "alice"
    assert decision.picks == [3]
    assert decision.payoff == 1.0
    assert decision.model_id == "gpt-4o-mini-2024-07-18"
    assert decision.tokens_in == 512
    assert decision.tokens_out == 128
    assert decision.cost_usd == 0.000234
    assert decision.decide_ms == 874


def test_build_decision_preserves_none_for_unset_telemetry() -> None:
    """When the Action has no telemetry captured (NULL columns), the
    DashboardDecision keeps them as ``None`` — must not coerce to 0 / "",
    because the drawer differentiates "—" (missing) from "0" (measured).
    """
    # GIVEN an Action with telemetry columns left NULL
    action = Action(action_data={"intervals": [[0, 0]]}, payoff=-1.0)

    # WHEN we project it
    decision = _build_decision_from_action(
        agent_id="bob",
        action=action,
        slot_attendance=[2],
        capacity_threshold=2,
    )

    # THEN the tier-2 fields stay None (not 0 / "")
    assert decision.model_id is None
    assert decision.tokens_in is None
    assert decision.tokens_out is None
    assert decision.cost_usd is None
    assert decision.decide_ms is None


def test_build_rounds_threads_telemetry_through_to_decisions() -> None:
    """End-to-end check at the rounds level: telemetry on the Action
    must surface on the per-decision projection inside the DashboardRound.
    """
    # GIVEN two agents with distinct telemetry payloads in the same round
    agents = [
        DashboardAgent(id="a1", color="#6e7781", user="unknown"),
        DashboardAgent(id="a2", color="#6e7781", user="unknown"),
    ]
    actions_by_round: dict[int, list[tuple[int, Action]]] = {
        1: [
            (
                0,
                Action(
                    action_data={"intervals": [[1, 1]]},
                    payoff=1.0,
                    model_id="gpt-4o",
                    tokens_in=100,
                    tokens_out=50,
                    cost_usd=0.01,
                    decide_ms=200,
                ),
            ),
            (
                1,
                Action(
                    action_data={"intervals": [[2, 2]]},
                    payoff=1.0,
                    model_id="claude-3-5-sonnet",
                    tokens_in=300,
                    tokens_out=75,
                    cost_usd=0.02,
                    decide_ms=450,
                ),
            ),
        ],
    }

    # WHEN we build rounds from those pairs
    rounds = _build_rounds_from_actions(
        actions_by_round=actions_by_round,
        agents=agents,
        num_slots=16,
        capacity_threshold=2,
    )

    # THEN each decision carries the originating Action's telemetry
    decs = {d.agent: d for d in rounds[0].decisions}
    assert decs["a1"].model_id == "gpt-4o"
    assert decs["a1"].tokens_in == 100
    assert decs["a1"].tokens_out == 50
    assert decs["a1"].cost_usd == 0.01
    assert decs["a1"].decide_ms == 200
    assert decs["a2"].model_id == "claude-3-5-sonnet"
    assert decs["a2"].tokens_in == 300
    assert decs["a2"].tokens_out == 75
    assert decs["a2"].cost_usd == 0.02
    assert decs["a2"].decide_ms == 450


# ---------------------------------------------------------------------------
# intervals projection (regression: drawer + make_move(...) example)
# ---------------------------------------------------------------------------


def test_build_decision_pads_single_interval_to_two_tuple() -> None:
    """A single interval must round-trip and be padded to the documented
    2-tuple shape — the drawer and generated ``make_move(...)`` example
    read both slots off ``DashboardDecision.intervals``.
    """
    # GIVEN an Action with one interval
    action = Action(action_data={"intervals": [[3, 6]]}, payoff=1.0)

    # WHEN we project it
    decision = _build_decision_from_action(
        agent_id="alice",
        action=action,
        slot_attendance=[0] * 16,
        capacity_threshold=2,
    )

    # THEN the single pair is preserved and the second slot is empty
    assert decision.intervals == [[3, 6], []]


def test_build_decision_preserves_two_intervals_in_order() -> None:
    """Two intervals must round-trip verbatim and in original order."""
    # GIVEN an Action with two distinct intervals
    action = Action(
        action_data={"intervals": [[0, 1], [10, 12]]},
        payoff=1.0,
    )

    # WHEN we project it
    decision = _build_decision_from_action(
        agent_id="alice",
        action=action,
        slot_attendance=[0] * 16,
        capacity_threshold=2,
    )

    # THEN both pairs land in the projection unchanged and in order
    assert decision.intervals == [[0, 1], [10, 12]]


def test_build_decision_empty_intervals_stays_empty() -> None:
    """Stay-home (no intervals) must project to ``[[], []]`` so the
    drawer renders an empty interval pair rather than a synthetic one.
    """
    # GIVEN an Action with no intervals (stay-home)
    action = Action(action_data={"intervals": []}, payoff=0.0)

    # WHEN we project it
    decision = _build_decision_from_action(
        agent_id="alice",
        action=action,
        slot_attendance=[0] * 16,
        capacity_threshold=2,
    )

    # THEN the projection is the documented empty 2-tuple
    assert decision.intervals == [[], []]


@pytest.mark.anyio
async def test_tournament_with_no_rounds_returns_empty_data(
    db_session: AsyncSession,
) -> None:
    t = Tournament(
        game_type="el_farol",
        num_players=2,
        total_rounds=5,
        status="cancelled",
    )
    db_session.add(t)
    await db_session.flush()
    db_session.add_all(
        [
            Participant(
                tournament_id=t.id,
                agent_name="a",
                builtin_strategy="el_farol/a",
            ),
            Participant(
                tournament_id=t.id,
                agent_name="b",
                builtin_strategy="el_farol/b",
            ),
        ]
    )
    await db_session.commit()

    payload = await _reshape_from_tournament(t.id, db_session)

    # NUM_DAYS must equal len(DATA) — Cards JS indexes DATA[d] for
    # every d < NUM_DAYS, so padding from total_rounds would crash
    # the scrubber.
    assert payload.DATA == []
    assert payload.NUM_DAYS == 0
    assert len(payload.AGENTS) == 2


@pytest.mark.anyio
async def test_match_id_is_threaded_into_payload(
    db_session: AsyncSession,
) -> None:
    """The Cards JS keys per-match localStorage off
    ``window.__ATP_MATCH__.match_id``, so the payload's match id has
    to match the ``/ui/matches/{id}`` URL the user is viewing.
    Falls back to a tournament-id surrogate only when no match_id is
    supplied (e.g. unit tests with no GameResult row).
    """
    t = Tournament(
        game_type="el_farol",
        num_players=2,
        total_rounds=1,
        status="completed",
    )
    db_session.add(t)
    await db_session.commit()

    payload_with_match = await _reshape_from_tournament(
        t.id, db_session, match_id="m-real-id"
    )
    payload_default = await _reshape_from_tournament(t.id, db_session)

    assert payload_with_match.match_id == "m-real-id"
    assert payload_default.match_id == f"tournament-{t.id}"


@pytest.mark.anyio
async def test_incomplete_round_is_skipped(db_session: AsyncSession) -> None:
    t = Tournament(game_type="el_farol", num_players=2, total_rounds=2, status="active")
    db_session.add(t)
    await db_session.flush()
    p1 = Participant(tournament_id=t.id, agent_name="a", builtin_strategy="el_farol/a")
    p2 = Participant(tournament_id=t.id, agent_name="b", builtin_strategy="el_farol/b")
    db_session.add_all([p1, p2])
    await db_session.flush()

    r_done = Round(tournament_id=t.id, round_number=1, status="completed")
    r_pending = Round(tournament_id=t.id, round_number=2, status="pending")
    db_session.add_all([r_done, r_pending])
    await db_session.flush()
    db_session.add_all(
        [
            Action(
                round_id=r_done.id,
                participant_id=p1.id,
                action_data={"intervals": [[0, 0]]},
                payoff=1.0,
            ),
            Action(
                round_id=r_done.id,
                participant_id=p2.id,
                action_data={"intervals": [[1, 1]]},
                payoff=1.0,
            ),
        ]
    )
    await db_session.commit()

    payload = await _reshape_from_tournament(t.id, db_session)

    assert len(payload.DATA) == 1
    assert payload.DATA[0].round == 1
