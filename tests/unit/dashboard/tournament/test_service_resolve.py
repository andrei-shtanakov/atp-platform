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
            for _ in range(6):
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

    # 3-round tournament: each non-final round resolution publishes
    # round_ended (with next_round_number) followed by round_started for
    # the next round. The final round emits tournament_completed instead
    # of round_ended.
    assert [e.event_type for e in received] == [
        "round_started",
        "round_ended",
        "round_started",
        "round_ended",
        "round_started",
        "tournament_completed",
    ]
    assert [e.round_number for e in received] == [1, 1, 2, 2, 3, None]
    assert received[1].data["tournament_completed"] is False
    assert received[1].data["next_round_number"] == 2
    assert received[3].data["tournament_completed"] is False
    assert received[3].data["next_round_number"] == 3


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
    await svc.submit_action(t.id, alice, action={"intervals": [[0, 0]]})
    result = await svc.submit_action(t.id, bob, action={"intervals": [[0, 0]]})

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
    # Bob stays home (intervals=[] → 0 payoff)
    await svc.submit_action(t.id, alice, action={"intervals": [[0, 0]]})
    await svc.submit_action(t.id, bob, action={"intervals": []})

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
        await svc.submit_action(t.id, alice, action={"intervals": [[0, 0]]})
        await svc.submit_action(t.id, bob, action={"intervals": [[0, 0]]})

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
    ``released_at`` must be set so the agent is no longer matched by the
    ``uq_participant_agent_active`` partial unique index and is free to
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
    """After tournament COMPLETED, same agents must be able to start a
    new tournament without hitting ``uq_participant_agent_active``.
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

    # Must not raise IntegrityError on uq_participant_agent_active.
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


@pytest.mark.anyio
async def test_full_3_round_stag_hunt_tournament_completes(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """3-round Stag Hunt at the service level.

    Alice always stag, Bob always hare → alice always gets sucker (0.0),
    bob always gets hare (3.0). Over 3 rounds: alice=0, bob=9.
    """
    from sqlalchemy import select

    from atp.dashboard.tournament.models import Participant
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="sh",
        game_type="stag_hunt",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    for round_n in range(1, 4):
        await svc.submit_action(t.id, alice, action={"choice": "stag"})
        result = await svc.submit_action(t.id, bob, action={"choice": "hare"})
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
    assert by_user[alice.id].total_score == 0.0, "alice should be sucker 3x"
    assert by_user[bob.id].total_score == 9.0, "bob should get hare payoff 3x3"
    # Auto-release on COMPLETED (PR #36 invariant) still holds.
    for p in parts:
        assert p.released_at is not None


@pytest.mark.anyio
async def test_stag_hunt_mutual_stag_pays_best(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """Both choose stag → mutual_stag payoff (default 4.0 each)."""
    from sqlalchemy import select

    from atp.dashboard.tournament.models import Action, Participant
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="sh-coord",
        game_type="stag_hunt",
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    await svc.submit_action(t.id, alice, action={"choice": "stag"})
    await svc.submit_action(t.id, bob, action={"choice": "stag"})

    parts = (
        (
            await session.execute(
                select(Participant).where(Participant.tournament_id == t.id)
            )
        )
        .scalars()
        .all()
    )
    alice_p = next(p for p in parts if p.user_id == alice.id)
    bob_p = next(p for p in parts if p.user_id == bob.id)
    a_action = (
        await session.execute(select(Action).where(Action.participant_id == alice_p.id))
    ).scalar_one()
    b_action = (
        await session.execute(select(Action).where(Action.participant_id == bob_p.id))
    ).scalar_one()
    assert a_action.payoff == 4.0
    assert b_action.payoff == 4.0


@pytest.mark.anyio
async def test_full_3_round_battle_of_sexes_tournament_completes(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """3-round BoS at the service level.

    Alice (p0) and Bob (p1) both pick A every round → alice gets
    preferred_a=3.0, bob gets other_a=2.0. Over 3 rounds: 9.0 vs 6.0.
    """
    from sqlalchemy import select

    from atp.dashboard.tournament.models import Participant
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="bos",
        game_type="battle_of_sexes",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    for round_n in range(1, 4):
        await svc.submit_action(t.id, alice, action={"choice": "A"})
        result = await svc.submit_action(t.id, bob, action={"choice": "A"})
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
    assert by_user[alice.id].total_score == 9.0  # preferred × 3
    assert by_user[bob.id].total_score == 6.0  # other × 3
    for p in parts:
        assert p.released_at is not None


@pytest.mark.anyio
async def test_battle_of_sexes_mismatch_pays_zero(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """Alice picks A, Bob picks B → both get mismatch (0.0)."""
    from sqlalchemy import select

    from atp.dashboard.tournament.models import Action, Participant
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="bos-mismatch",
        game_type="battle_of_sexes",
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    await svc.submit_action(t.id, alice, action={"choice": "A"})
    await svc.submit_action(t.id, bob, action={"choice": "B"})

    parts = (
        (
            await session.execute(
                select(Participant).where(Participant.tournament_id == t.id)
            )
        )
        .scalars()
        .all()
    )
    alice_p = next(p for p in parts if p.user_id == alice.id)
    bob_p = next(p for p in parts if p.user_id == bob.id)
    for p_id in (alice_p.id, bob_p.id):
        a = (
            await session.execute(select(Action).where(Action.participant_id == p_id))
        ).scalar_one()
        assert a.payoff == 0.0


@pytest.mark.anyio
async def test_battle_of_sexes_state_exposes_your_preferred(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """RoundState.your_preferred differs per participant_idx."""
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="bos-preferred",
        game_type="battle_of_sexes",
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    alice_state = await svc.get_state_for(t.id, alice)
    bob_state = await svc.get_state_for(t.id, bob)
    assert alice_state.your_preferred == "A"
    assert bob_state.your_preferred == "B"


@pytest.mark.anyio
async def test_force_resolve_round_persists_final_round_across_sessions(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """Regression for prod tournament 50 (2026-04-28): the deadline
    worker calls ``force_resolve_round`` inside a fresh
    ``async with session_factory() as session:`` block whose
    auto-rollback-on-exit reverted every change made by the
    final-round resolution path because that branch only flushed,
    never committed.

    Symptom in prod: the worker emitted ``round_resolved`` events
    for the same final round every poll interval forever; the
    round stayed at ``WAITING_FOR_ACTIONS`` in the DB; the
    tournament was stuck ``ACTIVE`` with no one able to play.

    This test reproduces the exact lifecycle:

    1. Setup a 1-round tournament in the fixture session and commit
       so the data is visible across sessions.
    2. Open a SEPARATE session, call ``force_resolve_round`` exactly
       like the deadline worker does, and let the session context
       manager exit without a manual commit.
    3. Open a THIRD session and verify the round is actually
       ``COMPLETED`` and the tournament is ``COMPLETED``. Pre-fix
       both would still read as in-flight.
    """
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from atp.dashboard.tournament.models import (
        Round,
        RoundStatus,
        Tournament,
        TournamentStatus,
    )
    from atp.dashboard.tournament.service import TournamentService

    # --- Setup phase (fixture session) ---
    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="t-final-round",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=1,  # final round IS round 1
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")
    # Alice submits, Bob does not — this is the round the deadline
    # worker would force-resolve in prod.
    await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
    round_one = (
        await session.execute(
            select(Round).where(Round.tournament_id == t.id, Round.round_number == 1)
        )
    ).scalar_one()
    assert round_one.status == RoundStatus.WAITING_FOR_ACTIONS
    round_id = round_one.id
    tournament_id = t.id
    # Make all setup state visible to other sessions.
    await session.commit()

    # --- Worker phase (fresh session, mirrors deadline_worker exactly) ---
    engine = session.bind
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as worker_session:
        worker_svc = TournamentService(worker_session, event_bus)
        await worker_svc.force_resolve_round(round_id)
        # Deliberately NO ``await worker_session.commit()`` here —
        # this matches deadlines.py which relies on
        # force_resolve_round to commit internally. Pre-fix the
        # context manager exit would auto-rollback at this point.

    # --- Verify phase (third session) ---
    async with factory() as verify_session:
        round_after = await verify_session.get(Round, round_id)
        assert round_after is not None
        assert round_after.status == RoundStatus.COMPLETED, (
            "round.status must persist as COMPLETED across sessions; "
            "pre-fix the worker's session exit auto-rolled back the "
            "final-round resolution and the worker hot-looped on the "
            "same round forever (see prod tournament 50, 2026-04-28)."
        )

        tournament_after = await verify_session.get(Tournament, tournament_id)
        assert tournament_after is not None
        assert tournament_after.status == TournamentStatus.COMPLETED, (
            "tournament must transition to COMPLETED on final-round "
            "resolution and that transition must persist."
        )
