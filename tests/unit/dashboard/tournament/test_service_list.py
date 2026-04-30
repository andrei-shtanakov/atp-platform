"""Tests for TournamentService.list_tournaments filtering."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEventBus


@pytest.mark.anyio
async def test_list_tournaments_filters_by_game_type(
    session: AsyncSession,
    admin_user: User,
    event_bus: TournamentEventBus,
) -> None:
    """Service filters tournaments by game_type at the SQL level."""
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    pd_t, _ = await svc.create_tournament(
        creator=admin_user,
        name="pd1",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
    )
    ef_t, _ = await svc.create_tournament(
        creator=admin_user,
        name="ef1",
        game_type="el_farol",
        num_players=3,
        total_rounds=1,
        round_deadline_s=30,
    )

    # No filter → both
    all_ts = await svc.list_tournaments(user=admin_user)
    ids = {t.id for t in all_ts}
    assert pd_t.id in ids
    assert ef_t.id in ids

    # game_type="prisoners_dilemma" → only PD
    pd_only = await svc.list_tournaments(user=admin_user, game_type="prisoners_dilemma")
    assert {t.id for t in pd_only} == {pd_t.id}

    # game_type="el_farol" → only el_farol
    ef_only = await svc.list_tournaments(user=admin_user, game_type="el_farol")
    assert {t.id for t in ef_only} == {ef_t.id}

    # Unknown game_type → empty
    none_res = await svc.list_tournaments(user=admin_user, game_type="tic_tac_toe")
    assert none_res == []
