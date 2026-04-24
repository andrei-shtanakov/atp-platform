"""submit_action: server-side game_type injection + mismatch detection."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.errors import ValidationError
from atp.dashboard.tournament.events import TournamentEventBus
from atp.dashboard.tournament.models import Action, Participant
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
    assert "intervals" in text


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
    result = await svc.submit_action(
        t.id, alice, action={"intervals": [[0, 0], [3, 3]]}
    )
    assert result["status"] == "waiting"


async def _make_public_goods(
    svc: TournamentService, admin: User, a: User, b: User
) -> object:
    t, _ = await svc.create_tournament(
        creator=admin,
        name="pg",
        game_type="public_goods",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, a, "alice")
    await svc.join(t.id, b, "bob")
    return t


@pytest.mark.anyio
async def test_public_goods_submit_happy(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    t = await _make_public_goods(svc, admin_user, alice, bob)
    result = await svc.submit_action(t.id, alice, action={"contribution": 7.5})
    assert result["status"] == "waiting"


@pytest.mark.anyio
async def test_public_goods_submit_rejects_negative(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    t = await _make_public_goods(svc, admin_user, alice, bob)
    with pytest.raises(ValidationError):
        await svc.submit_action(t.id, alice, action={"contribution": -1.0})


@pytest.mark.anyio
async def test_public_goods_hint_in_error_message(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    t = await _make_public_goods(svc, admin_user, alice, bob)
    with pytest.raises(ValidationError) as exc:
        await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
    text = str(exc.value)
    assert "public_goods" in text
    assert "contribution" in text


async def _alice_action(session: AsyncSession, alice_user_id: int) -> Action:
    stmt = (
        select(Action)
        .join(Participant, Action.participant_id == Participant.id)
        .where(Participant.user_id == alice_user_id)
    )
    return (await session.execute(stmt)).scalar_one()


@pytest.mark.anyio
async def test_reasoning_persists_pd(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    t = await _make_pd(svc, admin_user, alice, bob)
    await svc.submit_action(
        t.id,
        alice,
        action={"choice": "cooperate", "reasoning": "start nice, then mirror"},
    )
    a = await _alice_action(session, alice.id)
    assert a.reasoning == "start nice, then mirror"
    # Canonical action_data must not contain the reasoning key
    assert a.action_data == {"choice": "cooperate"}


@pytest.mark.anyio
async def test_reasoning_persists_el_farol(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    t = await _make_el_farol(svc, admin_user, alice, bob)
    await svc.submit_action(
        t.id,
        alice,
        action={"intervals": [[0, 0], [3, 3]], "reasoning": "non-crowded window"},
    )
    a = await _alice_action(session, alice.id)
    assert a.reasoning == "non-crowded window"
    assert a.action_data == {"slots": [0, 3]}


@pytest.mark.anyio
async def test_reasoning_optional(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    t = await _make_pd(svc, admin_user, alice, bob)
    await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
    a = await _alice_action(session, alice.id)
    assert a.reasoning is None


@pytest.mark.anyio
@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
async def test_reasoning_blank_normalized_to_none(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
    blank: str,
) -> None:
    svc = TournamentService(session, event_bus)
    t = await _make_pd(svc, admin_user, alice, bob)
    await svc.submit_action(
        t.id, alice, action={"choice": "cooperate", "reasoning": blank}
    )
    a = await _alice_action(session, alice.id)
    assert a.reasoning is None


@pytest.mark.anyio
async def test_reasoning_over_limit_rejected(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    t = await _make_pd(svc, admin_user, alice, bob)
    with pytest.raises(ValidationError):
        await svc.submit_action(
            t.id,
            alice,
            action={"choice": "cooperate", "reasoning": "x" * 8001},
        )


@pytest.mark.anyio
async def test_reasoning_unicode_roundtrip(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    svc = TournamentService(session, event_bus)
    t = await _make_pd(svc, admin_user, alice, bob)
    text = "💭 защищаю равновесие по Нэшу"
    await svc.submit_action(
        t.id, alice, action={"choice": "cooperate", "reasoning": text}
    )
    a = await _alice_action(session, alice.id)
    assert a.reasoning == text
