"""Unit tests for pending-deadline shrink/start behaviour."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEventBus
from atp.dashboard.tournament.models import (
    Participant,
    Round,
    TournamentStatus,
)
from atp.dashboard.tournament.reasons import CancelReason
from atp.dashboard.tournament.service import TournamentService


async def _make_user(session: AsyncSession, username: str) -> User:
    user = User(
        username=username,
        email=f"{username}@example.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    return user


async def _insert_builtin_participants(
    session: AsyncSession, tournament_id: int, names: list[str]
) -> None:
    for name in names:
        session.add(
            Participant(
                tournament_id=tournament_id,
                user_id=None,
                agent_id=None,
                agent_name=name,
                builtin_strategy=name,
            )
        )
    await session.commit()


async def _rounds_for(session: AsyncSession, tournament_id: int) -> list[Round]:
    return (
        (
            await session.execute(
                select(Round)
                .where(Round.tournament_id == tournament_id)
                .order_by(Round.round_number)
            )
        )
        .scalars()
        .all()
    )


@pytest.mark.anyio
async def test_shrinks_when_below_num_players(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    charlie = await _make_user(session, "charlie")
    svc = TournamentService(session, event_bus)
    tournament, _ = await svc.create_tournament(
        creator=admin_user,
        name="shrink-me",
        game_type="el_farol",
        num_players=5,
        total_rounds=2,
        round_deadline_s=30,
    )

    await svc.join(tournament.id, alice, "alice")
    await svc.join(tournament.id, bob, "bob")
    await svc.join(tournament.id, charlie, "charlie")

    captured = []
    original_publish = event_bus.publish

    async def capture(evt):
        captured.append(evt)
        await original_publish(evt)

    event_bus.publish = capture  # type: ignore[method-assign]

    await svc.try_shrink_and_start_or_cancel(tournament.id)

    await session.refresh(tournament)
    assert tournament.num_players == 3
    assert tournament.status == TournamentStatus.ACTIVE
    rounds = await _rounds_for(session, tournament.id)
    assert len(rounds) == 1
    assert rounds[0].round_number == 1

    shrunken = [e for e in captured if e.event_type == "tournament_shrunken"]
    assert len(shrunken) == 1
    assert shrunken[0].tournament_id == tournament.id
    assert shrunken[0].data == {
        "original_num_players": 5,
        "actual_num_players": 3,
    }


@pytest.mark.anyio
async def test_no_shrink_when_full(
    session: AsyncSession,
    admin_user: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    tournament, _ = await svc.create_tournament(
        creator=admin_user,
        name="already-full",
        game_type="el_farol",
        num_players=3,
        total_rounds=2,
        round_deadline_s=30,
    )
    await _insert_builtin_participants(
        session,
        tournament.id,
        [
            "el_farol/traditionalist",
            "el_farol/contrarian",
            "el_farol/gambler",
        ],
    )

    captured = []
    original_publish = event_bus.publish

    async def capture(evt):
        captured.append(evt)
        await original_publish(evt)

    event_bus.publish = capture  # type: ignore[method-assign]

    await svc.try_shrink_and_start_or_cancel(tournament.id)

    await session.refresh(tournament)
    assert tournament.num_players == 3
    assert tournament.status == TournamentStatus.ACTIVE

    shrunken = [e for e in captured if e.event_type == "tournament_shrunken"]
    assert shrunken == []


@pytest.mark.anyio
async def test_cancels_when_below_floor(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    tournament, _ = await svc.create_tournament(
        creator=admin_user,
        name="too-small",
        game_type="public_goods",
        num_players=4,
        total_rounds=2,
        round_deadline_s=30,
    )
    await svc.join(tournament.id, alice, "alice")

    await svc.try_shrink_and_start_or_cancel(tournament.id)
    await session.commit()

    await session.refresh(tournament)
    assert tournament.status == TournamentStatus.CANCELLED
    assert tournament.cancelled_reason == CancelReason.PENDING_TIMEOUT


@pytest.mark.anyio
async def test_cancels_when_zero_live(
    session: AsyncSession,
    admin_user: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    tournament, _ = await svc.create_tournament(
        creator=admin_user,
        name="nobody-home",
        game_type="public_goods",
        num_players=4,
        total_rounds=2,
        round_deadline_s=30,
    )

    await svc.try_shrink_and_start_or_cancel(tournament.id)
    await session.commit()

    await session.refresh(tournament)
    assert tournament.status == TournamentStatus.CANCELLED
    assert tournament.cancelled_reason == CancelReason.PENDING_TIMEOUT


@pytest.mark.anyio
async def test_skips_when_status_not_pending(
    session: AsyncSession,
    admin_user: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    tournament, _ = await svc.create_tournament(
        creator=admin_user,
        name="already-active",
        game_type="el_farol",
        num_players=2,
        total_rounds=2,
        round_deadline_s=30,
    )
    tournament.status = TournamentStatus.ACTIVE
    await session.commit()

    await svc.try_shrink_and_start_or_cancel(tournament.id)

    await session.refresh(tournament)
    assert tournament.status == TournamentStatus.ACTIVE
    assert await _rounds_for(session, tournament.id) == []


@pytest.mark.anyio
async def test_released_participants_excluded_from_count(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    charlie = await _make_user(session, "charlie")
    svc = TournamentService(session, event_bus)
    tournament, _ = await svc.create_tournament(
        creator=admin_user,
        name="pending-leaver",
        game_type="el_farol",
        num_players=4,
        total_rounds=2,
        round_deadline_s=30,
    )

    await svc.join(tournament.id, alice, "alice")
    await svc.join(tournament.id, bob, "bob")
    await svc.join(tournament.id, charlie, "charlie")
    await svc.leave(tournament.id, alice)

    await svc.try_shrink_and_start_or_cancel(tournament.id)

    await session.refresh(tournament)
    assert tournament.status == TournamentStatus.ACTIVE
    assert tournament.num_players == 2

    state = await svc.get_state_for(tournament.id, bob)
    assert len(state.all_scores) == 2


@pytest.mark.anyio
async def test_roster_counts_toward_live(
    session: AsyncSession,
    admin_user: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    tournament, _ = await svc.create_tournament(
        creator=admin_user,
        name="builtin-roster",
        game_type="el_farol",
        num_players=3,
        total_rounds=2,
        round_deadline_s=30,
        roster=[
            "el_farol/traditionalist",
            "el_farol/contrarian",
        ],
    )

    await svc.try_shrink_and_start_or_cancel(tournament.id)

    await session.refresh(tournament)
    assert tournament.status == TournamentStatus.ACTIVE
    assert tournament.num_players == 2
