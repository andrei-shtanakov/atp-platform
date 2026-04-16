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


@pytest.mark.anyio
async def test_el_farol_resolve_round_writes_payoffs(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """El Farol round resolves synchronously on last submit; payoffs are written."""
    from sqlalchemy import select

    from atp.dashboard.tournament.models import Action
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="ef",
        game_type="el_farol",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    # Both players attend slot 0 — with capacity_threshold = max(1, int(0.6*2)) = 1,
    # slot 0 will be crowded (count >= threshold) → each gets 0 happy - 1 crowded = -1.
    await svc.submit_action(t.id, alice, action={"slots": [0]})
    result = await svc.submit_action(t.id, bob, action={"slots": [0]})

    assert result["status"] == "round_resolved"
    assert result["round_number"] == 1

    actions = (await session.execute(select(Action))).scalars().all()
    assert len(actions) >= 2
    for a in actions:
        assert a.payoff is not None, f"action {a.id} has None payoff"
        # Both attended the same crowded slot → payoff should be -1.0 each
        assert a.payoff == -1.0


@pytest.mark.anyio
async def test_el_farol_resolve_round_payoffs_differ_on_choice(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """Alice attends crowded slot, Bob stays home → different payoffs."""
    from sqlalchemy import select

    from atp.dashboard.tournament.models import Action, Participant
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="ef",
        game_type="el_farol",
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    # Alice picks slot 0 alone (count=1, threshold=1 → CROWDED → -1 payoff)
    # Bob stays home (slots=[] → 0 payoff)
    await svc.submit_action(t.id, alice, action={"slots": [0]})
    await svc.submit_action(t.id, bob, action={"slots": []})

    # Look up participant → action mapping
    participants = (
        (
            await session.execute(
                select(Participant).where(Participant.tournament_id == t.id)
            )
        )
        .scalars()
        .all()
    )
    alice_p = next(p for p in participants if p.user_id == alice.id)
    bob_p = next(p for p in participants if p.user_id == bob.id)

    alice_action = (
        (
            await session.execute(
                select(Action).where(Action.participant_id == alice_p.id)
            )
        )
        .scalars()
        .first()
    )
    bob_action = (
        (await session.execute(select(Action).where(Action.participant_id == bob_p.id)))
        .scalars()
        .first()
    )

    assert alice_action.payoff == -1.0
    assert bob_action.payoff == 0.0


@pytest.mark.anyio
async def test_resolve_round_logs_structured_fields(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
    caplog,
) -> None:
    import logging

    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="ef-logs",
        game_type="el_farol",
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    with caplog.at_level(logging.INFO, logger="atp.dashboard.tournament.service"):
        await svc.submit_action(t.id, alice, action={"slots": [0]})
        await svc.submit_action(t.id, bob, action={"slots": [0]})

    rec = next(
        r for r in caplog.records if getattr(r, "event", None) == "round_resolved"
    )
    assert rec.game_type == "el_farol"
    assert rec.tournament_id == t.id
    assert rec.round_number == 1
    assert rec.round_resolution_ms >= 0


@pytest.mark.anyio
async def test_submit_action_rejected_emits_structured_log(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
    caplog,
) -> None:
    import logging

    from atp.dashboard.tournament.errors import ValidationError
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="ef-rej",
        game_type="el_farol",
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    with caplog.at_level(logging.INFO, logger="atp.dashboard.tournament.service"):
        with pytest.raises(ValidationError):
            # wrong shape for el_farol (PD's choice)
            await svc.submit_action(t.id, alice, action={"choice": "cooperate"})

    rec = next(
        r for r in caplog.records if getattr(r, "event", None) == "action_rejected"
    )
    assert rec.game_type == "el_farol"
    assert rec.tournament_id == t.id


@pytest.mark.anyio
async def test_tournament_completion_releases_participants(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """When a tournament transitions to COMPLETED, every participant's
    ``released_at`` must be set so the user is no longer matched by the
    ``uq_participant_user_active`` partial unique index and is free to
    join another tournament. Mirrors the symmetric release done by
    ``_cancel_impl`` step 5.
    """
    from sqlalchemy import select

    from atp.dashboard.tournament.models import (
        Participant,
        TournamentStatus,
    )
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="release-on-complete",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")
    await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
    await svc.submit_action(t.id, bob, action={"choice": "defect"})

    await session.refresh(t)
    assert t.status == TournamentStatus.COMPLETED

    parts = (
        (
            await session.execute(
                select(Participant).where(Participant.tournament_id == t.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(parts) == 2
    for p in parts:
        assert p.released_at is not None, (
            f"participant user_id={p.user_id} still active after completion; "
            "would block future tournament participation"
        )


@pytest.mark.anyio
async def test_completed_participants_can_join_new_tournament(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """After tournament COMPLETED, same users must be able to start a
    new tournament without hitting ``uq_participant_user_active``.
    """
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)

    t1, _ = await svc.create_tournament(
        creator=admin_user,
        name="first",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
    )
    await svc.join(t1.id, alice, "alice")
    await svc.join(t1.id, bob, "bob")
    await svc.submit_action(t1.id, alice, action={"choice": "cooperate"})
    await svc.submit_action(t1.id, bob, action={"choice": "cooperate"})

    # Must not raise IntegrityError on uq_participant_user_active.
    t2, _ = await svc.create_tournament(
        creator=admin_user,
        name="second",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
    )
    await svc.join(t2.id, alice, "alice")
    await svc.join(t2.id, bob, "bob")
