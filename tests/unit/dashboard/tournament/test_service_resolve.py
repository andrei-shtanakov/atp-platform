"""Tests for TournamentService.submit_action and round resolution."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEventBus


@pytest.mark.anyio
async def test_submit_action_first_player_returns_waiting(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t = await svc.create_tournament(
        admin=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    result = await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
    assert result["status"] == "waiting"
    assert result["round_number"] == 1


@pytest.mark.anyio
async def test_submit_action_last_player_resolves_round(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """Alice cooperates, Bob defects → Bob gets 5, Alice gets 0 (PD payoff)."""
    from sqlalchemy import select

    from atp.dashboard.tournament.models import Action, Participant, Round
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t = await svc.create_tournament(
        admin=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
    result = await svc.submit_action(t.id, bob, action={"choice": "defect"})

    assert result["status"] == "round_resolved"
    assert result["round_number"] == 1

    rounds = (
        (
            await session.execute(
                select(Round)
                .where(Round.tournament_id == t.id)
                .order_by(Round.round_number)
            )
        )
        .scalars()
        .all()
    )
    assert len(rounds) == 2
    assert rounds[0].status == "completed"
    assert rounds[1].status == "waiting_for_actions"
    assert rounds[1].round_number == 2

    actions = (
        (await session.execute(select(Action).where(Action.round_id == rounds[0].id)))
        .scalars()
        .all()
    )
    by_user: dict[int, Action] = {}
    for a in actions:
        p = await session.get(Participant, a.participant_id)
        assert p is not None
        by_user[p.user_id] = a
    assert by_user[alice.id].action_data["choice"] == "cooperate"
    assert by_user[alice.id].payoff == 0.0
    assert by_user[bob.id].action_data["choice"] == "defect"
    assert by_user[bob.id].payoff == 5.0


@pytest.mark.anyio
async def test_full_3_round_pd_tournament_completes(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """End-to-end at the service level: alice always cooperates, bob
    always defects, 3 rounds. Final scores: alice=0, bob=15."""
    from sqlalchemy import select

    from atp.dashboard.tournament.models import Participant
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t = await svc.create_tournament(
        admin=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    for round_n in range(1, 4):
        await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
        result = await svc.submit_action(t.id, bob, action={"choice": "defect"})
        assert result["round_number"] == round_n

    await session.refresh(t)
    assert t.status == "completed"

    parts = (
        (
            await session.execute(
                select(Participant).where(Participant.tournament_id == t.id)
            )
        )
        .scalars()
        .all()
    )
    by_user = {p.user_id: p for p in parts}
    assert by_user[alice.id].total_score == 0.0
    assert by_user[bob.id].total_score == 15.0
