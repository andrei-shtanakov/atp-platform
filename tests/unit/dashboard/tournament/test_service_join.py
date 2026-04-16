"""Tests for TournamentService.create_tournament and join lifecycle."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEventBus


@pytest.mark.anyio
async def test_create_tournament_persists_basic_fields(
    session: AsyncSession,
    admin_user: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    tournament, _ = await svc.create_tournament(
        creator=admin_user,
        name="slice-test",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )

    assert tournament.id is not None
    assert tournament.game_type == "prisoners_dilemma"
    assert tournament.num_players == 2
    assert tournament.total_rounds == 3
    assert tournament.round_deadline_s == 30
    assert tournament.created_by == admin_user.id
    assert tournament.status == "pending"


@pytest.mark.anyio
async def test_create_tournament_rejects_unknown_game_type(
    session: AsyncSession,
    admin_user: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.errors import ValidationError
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    with pytest.raises(ValidationError, match="unsupported game_type"):
        await svc.create_tournament(
            creator=admin_user,
            name="bad-game",
            game_type="chess",
            num_players=2,
            total_rounds=3,
            round_deadline_s=30,
        )


@pytest.mark.anyio
async def test_create_tournament_rejects_invalid_num_players(
    session: AsyncSession,
    admin_user: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.errors import ValidationError
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    with pytest.raises(ValidationError, match="exactly 2 players"):
        await svc.create_tournament(
            creator=admin_user,
            name="single-player-pd",
            game_type="prisoners_dilemma",
            num_players=1,
            total_rounds=3,
            round_deadline_s=30,
        )


@pytest.mark.anyio
async def test_join_first_player_creates_participant(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    tournament, _ = await svc.create_tournament(
        creator=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )

    participant, is_new = await svc.join(
        tournament_id=tournament.id, user=alice, agent_name="alice-tft"
    )
    assert is_new is True
    assert participant.id is not None
    assert participant.tournament_id == tournament.id
    assert participant.user_id == alice.id
    assert participant.agent_name == "alice-tft"
    await session.refresh(tournament)
    assert tournament.status == "pending"


@pytest.mark.anyio
async def test_join_filling_last_slot_starts_tournament_and_creates_round_1(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.models import Round
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    tournament, _ = await svc.create_tournament(
        creator=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )

    await svc.join(tournament_id=tournament.id, user=alice, agent_name="alice-tft")
    await svc.join(tournament_id=tournament.id, user=bob, agent_name="bob-random")

    await session.refresh(tournament)
    assert tournament.status == "active"

    rounds = (
        (
            await session.execute(
                select(Round).where(Round.tournament_id == tournament.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(rounds) == 1
    assert rounds[0].round_number == 1
    assert rounds[0].status == "waiting_for_actions"


@pytest.mark.anyio
async def test_join_populates_agent_id_from_owned_agent(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """LABS-14 regression: join() must set Participant.agent_id so the
    DELETE /api/v1/agents/{id} 409 guard can actually block soft-delete
    of agents currently playing.
    """
    from atp.dashboard.models import Agent
    from atp.dashboard.tournament.models import Participant
    from atp.dashboard.tournament.service import TournamentService

    alice_agent = Agent(
        name="alice-bot",
        agent_type="cli",
        owner_id=alice.id,
        version="v1",
    )
    session.add(alice_agent)
    await session.commit()

    svc = TournamentService(session, event_bus)
    tournament, _ = await svc.create_tournament(
        creator=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(tournament_id=tournament.id, user=alice, agent_name="alice-bot")
    # bob has no owned Agent — agent_id must remain NULL (legacy path)
    await svc.join(tournament_id=tournament.id, user=bob, agent_name="bob-anon")

    participants = (
        (
            await session.execute(
                select(Participant)
                .where(Participant.tournament_id == tournament.id)
                .order_by(Participant.id)
            )
        )
        .scalars()
        .all()
    )
    by_user = {p.user_id: p for p in participants}
    assert by_user[alice.id].agent_id == alice_agent.id
    assert by_user[bob.id].agent_id is None


@pytest.mark.anyio
async def test_join_ignores_soft_deleted_owned_agent(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """A soft-deleted Agent must NOT be resolved into agent_id on join."""
    from datetime import datetime

    from atp.dashboard.models import Agent
    from atp.dashboard.tournament.models import Participant
    from atp.dashboard.tournament.service import TournamentService

    dead_agent = Agent(
        name="alice-bot",
        agent_type="cli",
        owner_id=alice.id,
        version="v1",
        deleted_at=datetime.now(),
    )
    session.add(dead_agent)
    await session.commit()

    svc = TournamentService(session, event_bus)
    tournament, _ = await svc.create_tournament(
        creator=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(tournament_id=tournament.id, user=alice, agent_name="alice-bot")
    await svc.join(tournament_id=tournament.id, user=bob, agent_name="bob-anon")

    alice_participant = await session.scalar(
        select(Participant)
        .where(Participant.tournament_id == tournament.id)
        .where(Participant.user_id == alice.id)
    )
    assert alice_participant is not None
    assert alice_participant.agent_id is None
