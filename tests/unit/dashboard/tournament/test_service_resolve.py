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
