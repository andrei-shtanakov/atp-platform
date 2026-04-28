"""Tests for the ``round_ended`` bus event published by ``_resolve_round``.

After a non-final round resolves, ``TournamentService._resolve_round``
emits a ``round_ended`` event before the next ``round_started``. The
final round does NOT emit ``round_ended`` -- subscribers treat
``tournament_completed`` as the final-round signal so the SSE handler
projects the post-completion snapshot exactly once.

Drives a 3-round PD tournament because the existing service tests
already verify PD payoff resolution; the bus-event surface is
game-agnostic.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEvent, TournamentEventBus
from atp.dashboard.tournament.service import TournamentService


@pytest.mark.anyio
async def test_round_ended_published_for_non_final_round_with_next_number(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """GIVEN a 3-round PD tournament
    WHEN round 1 (non-final) resolves
    THEN a ``round_ended`` event for round 1 fires before
        ``round_started`` for round 2, with
        ``data["tournament_completed"] is False`` and
        ``data["next_round_number"] == 2``.
    """
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

    async def collect(n: int) -> None:
        async with event_bus.subscribe(t.id) as queue:
            for _ in range(n):
                received.append(await queue.get())

    # Expect: round_started(R1), round_ended(R1), round_started(R2)
    collector = asyncio.create_task(collect(3))
    await asyncio.sleep(0)

    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")
    # Resolve round 1 only
    await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
    await svc.submit_action(t.id, bob, action={"choice": "defect"})

    await asyncio.wait_for(collector, timeout=2.0)

    types = [e.event_type for e in received]
    # The round_ended must arrive AFTER the join's initial round_started(R1)
    # and BEFORE round_started(R2) for the next round.
    assert types == ["round_started", "round_ended", "round_started"]

    round_ended = received[1]
    assert round_ended.event_type == "round_ended"
    assert round_ended.tournament_id == t.id
    assert round_ended.round_number == 1
    assert round_ended.data["tournament_completed"] is False
    assert round_ended.data["next_round_number"] == 2

    # And the round_started immediately after points to round 2.
    assert received[2].round_number == 2


@pytest.mark.anyio
async def test_no_round_ended_on_final_round_only_tournament_completed(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """GIVEN a 2-round PD tournament
    WHEN both rounds resolve in sequence
    THEN we observe round_started(R1), round_ended(R1), round_started(R2),
        tournament_completed -- but NO round_ended for the final round.
    """
    received: list[TournamentEvent] = []

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=2,
        round_deadline_s=30,
    )

    # We expect 4 events: round_started(R1), round_ended(R1),
    # round_started(R2), tournament_completed. No round_ended for R2.
    async def collect(n: int) -> None:
        async with event_bus.subscribe(t.id) as queue:
            for _ in range(n):
                received.append(await queue.get())

    collector = asyncio.create_task(collect(4))
    await asyncio.sleep(0)

    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")
    for _ in range(2):
        await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
        await svc.submit_action(t.id, bob, action={"choice": "defect"})

    await asyncio.wait_for(collector, timeout=2.0)

    assert [e.event_type for e in received] == [
        "round_started",
        "round_ended",
        "round_started",
        "tournament_completed",
    ]
    # round_ended only fired for round 1 (non-final).
    round_ended_events = [e for e in received if e.event_type == "round_ended"]
    assert len(round_ended_events) == 1
    assert round_ended_events[0].round_number == 1
