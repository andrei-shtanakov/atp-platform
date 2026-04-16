"""submit_action: server-side game_type injection + mismatch detection."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.errors import ValidationError
from atp.dashboard.tournament.events import TournamentEventBus
from atp.dashboard.tournament.service import TournamentService


async def _make_pd(svc: TournamentService, admin: User, a: User, b: User) -> object:
    t, _ = await svc.create_tournament(
        creator=admin,
        name="pd",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, a, "alice")
    await svc.join(t.id, b, "bob")
    return t


async def _make_el_farol(
    svc: TournamentService, admin: User, a: User, b: User
) -> object:
    t, _ = await svc.create_tournament(
        creator=admin,
        name="ef",
        game_type="el_farol",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, a, "alice")
    await svc.join(t.id, b, "bob")
    return t


@pytest.mark.anyio
async def test_pd_submit_without_game_type_still_works(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    t = await _make_pd(svc, admin_user, alice, bob)
    result = await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
    assert result["status"] == "waiting"


@pytest.mark.anyio
async def test_pd_submit_with_mismatched_game_type_rejected(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    t = await _make_pd(svc, admin_user, alice, bob)
    with pytest.raises(ValidationError, match="does not match"):
        await svc.submit_action(
            t.id,
            alice,
            action={"game_type": "el_farol", "choice": "cooperate"},
        )


@pytest.mark.anyio
async def test_pd_action_to_el_farol_tournament_error_has_hint(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    t = await _make_el_farol(svc, admin_user, alice, bob)
    with pytest.raises(ValidationError) as exc:
        await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
    text = str(exc.value)
    assert "el_farol" in text
    assert "slots" in text


@pytest.mark.anyio
async def test_el_farol_submit_happy(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    t = await _make_el_farol(svc, admin_user, alice, bob)
    result = await svc.submit_action(t.id, alice, action={"slots": [0, 3]})
    assert result["status"] == "waiting"
