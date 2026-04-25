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
from atp.dashboard.v2.routes.el_farol_from_tournament import (
    _build_agents_from_participants,
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
