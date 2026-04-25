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
                action_data={"slots": [0, 1]},
                payoff=2.0,
            ),
            Action(
                round_id=r.id,
                participant_id=p2.id,
                action_data={"slots": [1, 2]},
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

    actions_by_round: dict[int, list[tuple[int, dict, float | None]]] = {
        1: [
            (0, {"slots": [0]}, 1.0),
            (1, {"slots": [0]}, -1.0),
            (2, {"slots": [5]}, 1.0),
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

    assert payload.DATA == []
    assert payload.NUM_DAYS == 5
    assert len(payload.AGENTS) == 2


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
                action_data={"slots": [0]},
                payoff=1.0,
            ),
            Action(
                round_id=r_done.id,
                participant_id=p2.id,
                action_data={"slots": [1]},
                payoff=1.0,
            ),
        ]
    )
    await db_session.commit()

    payload = await _reshape_from_tournament(t.id, db_session)

    assert len(payload.DATA) == 1
    assert payload.DATA[0].round == 1
