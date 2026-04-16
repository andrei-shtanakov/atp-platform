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


@pytest.mark.anyio
async def test_state_el_farol_has_pending_submission(
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
        name="ef",
        game_type="el_farol",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    state = await svc.get_state_for(t.id, alice)
    assert state.game_type == "el_farol"
    assert state.pending_submission is True
    assert hasattr(state, "attendance_by_round")
    assert hasattr(state, "capacity_threshold")


@pytest.mark.anyio
async def test_state_el_farol_pending_flips_after_submit(
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
        name="ef",
        game_type="el_farol",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")
    await svc.submit_action(t.id, alice, action={"slots": [0]})

    state = await svc.get_state_for(t.id, alice)
    assert state.pending_submission is False


@pytest.mark.anyio
async def test_state_pd_has_your_turn_still(
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
        name="pd",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    state = await svc.get_state_for(t.id, alice)
    assert state.game_type == "prisoners_dilemma"
    assert isinstance(state.your_turn, bool)


@pytest.mark.anyio
async def test_state_after_tournament_completes_flags_are_false(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """After the final round resolves, your_turn must NOT be True."""
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="pd-done",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
    await svc.submit_action(t.id, bob, action={"choice": "defect"})

    state = await svc.get_state_for(t.id, alice)
    assert state.game_type == "prisoners_dilemma"
    assert state.your_turn is False  # tournament over — nothing to submit
