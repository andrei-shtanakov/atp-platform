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
    t, _ = await svc.create_tournament(
        creator=admin_user,
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
    t, _ = await svc.create_tournament(
        creator=admin_user,
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
    t, _ = await svc.create_tournament(
        creator=admin_user,
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


@pytest.mark.anyio
async def test_full_3_round_publishes_round_started_and_tournament_completed(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """Verify the slice's two notification events flow through the bus."""
    import asyncio

    from atp.dashboard.tournament.events import TournamentEvent
    from atp.dashboard.tournament.service import TournamentService

    received: list[TournamentEvent] = []

    svc = TournamentService(session, event_bus)

    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )

    async def collect() -> None:
        async with event_bus.subscribe(t.id) as queue:
            for _ in range(4):
                event = await queue.get()
                received.append(event)

    collector = asyncio.create_task(collect())
    await asyncio.sleep(0)

    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")
    for _ in range(3):
        await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
        await svc.submit_action(t.id, bob, action={"choice": "defect"})

    await asyncio.wait_for(collector, timeout=2.0)

    assert [e.event_type for e in received] == [
        "round_started",
        "round_started",
        "round_started",
        "tournament_completed",
    ]
    assert [e.round_number for e in received[:3]] == [1, 2, 3]


@pytest.mark.anyio
async def test_force_resolve_round_computes_payoffs(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """LABS-7 regression: force_resolve_round must write Action.payoff.

    Alice cooperates (submitted), Bob times out (TIMEOUT_DEFAULT → defect).
    PD matrix: CD → alice=0, bob=5. Before the fix, timeout rounds had
    NULL payoffs and SUM(payoff) in _complete_tournament returned 0.
    """
    from sqlalchemy import func, select

    from atp.dashboard.tournament.models import (
        Action,
        ActionSource,
        Participant,
        Round,
        RoundStatus,
    )
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    # Alice submits, Bob doesn't → round still waiting_for_actions
    await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
    round_one = (
        await session.execute(
            select(Round).where(Round.tournament_id == t.id, Round.round_number == 1)
        )
    ).scalar_one()
    assert round_one.status == RoundStatus.WAITING_FOR_ACTIONS

    await svc.force_resolve_round(round_one.id)
    await session.flush()

    await session.refresh(round_one)
    assert round_one.status == RoundStatus.COMPLETED

    actions = (
        (await session.execute(select(Action).where(Action.round_id == round_one.id)))
        .scalars()
        .all()
    )
    assert len(actions) == 2
    by_user: dict[int, Action] = {}
    for a in actions:
        p = await session.get(Participant, a.participant_id)
        assert p is not None
        by_user[p.user_id] = a

    # Payoffs must be populated, not NULL
    assert by_user[alice.id].payoff == 0.0
    assert by_user[alice.id].source == ActionSource.SUBMITTED
    assert by_user[bob.id].payoff == 5.0
    assert by_user[bob.id].source == ActionSource.TIMEOUT_DEFAULT

    # And the aggregate used by _complete_tournament is non-zero
    total_payoff = await session.scalar(
        select(func.sum(Action.payoff)).where(Action.round_id == round_one.id)
    )
    assert total_payoff == 5.0
