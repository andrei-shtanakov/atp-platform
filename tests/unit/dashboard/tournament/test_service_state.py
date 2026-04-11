"""Tests for TournamentService.get_state_for."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEventBus


@pytest.mark.anyio
async def test_get_state_for_round_1_no_history(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
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
    await svc.join(tournament.id, alice, "alice")
    await svc.join(tournament.id, bob, "bob")

    state = await svc.get_state_for(tournament.id, alice)

    assert state.tournament_id == tournament.id
    assert state.round_number == 1
    assert state.game_type == "prisoners_dilemma"
    assert state.your_history == []
    assert state.opponent_history == []
    assert state.your_cumulative_score == 0.0
    assert state.opponent_cumulative_score == 0.0
    assert state.action_schema["options"] == ["cooperate", "defect"]
    assert state.your_turn is True
    assert state.total_rounds == 3
