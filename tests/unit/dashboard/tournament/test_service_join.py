"""Tests for TournamentService.create_tournament and join lifecycle."""

from __future__ import annotations

import pytest
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
    tournament = await svc.create_tournament(
        admin=admin_user,
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
            admin=admin_user,
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
    with pytest.raises(ValidationError, match="num_players"):
        await svc.create_tournament(
            admin=admin_user,
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
    tournament = await svc.create_tournament(
        admin=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )

    participant = await svc.join(
        tournament_id=tournament.id, user=alice, agent_name="alice-tft"
    )
    assert participant.id is not None
    assert participant.tournament_id == tournament.id
    assert participant.user_id == alice.id
    assert participant.agent_name == "alice-tft"
    await session.refresh(tournament)
    assert tournament.status == "pending"
