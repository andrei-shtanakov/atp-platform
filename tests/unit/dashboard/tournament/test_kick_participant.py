"""Unit tests for TournamentService.kick_participant."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEventBus
from atp.dashboard.tournament.models import (
    Action,
    ActionSource,
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.service import TournamentService


async def _seed_live_el_farol_with_stalled(
    session: AsyncSession, creator: User
) -> tuple[Tournament, dict[str, Participant]]:
    """ACTIVE El Farol, 2 participants, round 2 in progress. alpha has
    submitted, stalled_bot has not."""
    now = datetime.now(tz=UTC).replace(tzinfo=None)

    t = Tournament(
        game_type="el_farol",
        status=TournamentStatus.ACTIVE.value,
        num_players=2,
        total_rounds=5,
        round_deadline_s=30,
        created_by=creator.id,
        created_at=now - timedelta(minutes=3),
        starts_at=now - timedelta(minutes=2),
        pending_deadline=now - timedelta(minutes=2),
    )
    session.add(t)
    await session.flush()

    participants: dict[str, Participant] = {}
    for name in ("alpha", "stalled_bot"):
        u = User(
            username=f"bot_{name}",
            email=f"{name}@t.com",
            hashed_password="x",
            is_active=True,
        )
        session.add(u)
        await session.flush()
        p = Participant(
            tournament_id=t.id,
            user_id=u.id,
            agent_name=name,
            total_score=0.0,
        )
        session.add(p)
        participants[name] = p
    await session.flush()

    r2 = Round(
        tournament_id=t.id,
        round_number=2,
        status=RoundStatus.IN_PROGRESS.value,
        started_at=now - timedelta(seconds=10),
        deadline=now + timedelta(seconds=20),
    )
    session.add(r2)
    await session.flush()
    # Only alpha has submitted this round.
    session.add(
        Action(
            round_id=r2.id,
            participant_id=participants["alpha"].id,
            action_data={"slots": [0, 1]},
            submitted_at=now - timedelta(seconds=5),
            source=ActionSource.SUBMITTED.value,
        )
    )
    await session.commit()
    return t, participants


@pytest.mark.anyio
async def test_kick_sets_released_at(session: AsyncSession, admin_user: User):
    t, participants = await _seed_live_el_farol_with_stalled(session, admin_user)
    target = participants["alpha"]
    service = TournamentService(session=session, bus=TournamentEventBus())
    await service.kick_participant(t.id, target.id)

    refreshed = await session.get(Participant, target.id)
    assert refreshed is not None
    assert refreshed.released_at is not None


@pytest.mark.anyio
async def test_kick_inserts_timeout_action_for_missing_current_round_submission(
    session: AsyncSession, admin_user: User
):
    t, participants = await _seed_live_el_farol_with_stalled(session, admin_user)
    target = participants["stalled_bot"]
    service = TournamentService(session=session, bus=TournamentEventBus())
    await service.kick_participant(t.id, target.id)

    stmt = select(Action).where(Action.participant_id == target.id)
    rows = (await session.execute(stmt)).scalars().all()
    assert len(rows) == 1
    assert rows[0].source == ActionSource.TIMEOUT_DEFAULT.value
    # For El Farol the default is {"slots": []}.
    assert rows[0].action_data == {"slots": []}


@pytest.mark.anyio
async def test_kick_does_not_double_insert_when_participant_already_submitted(
    session: AsyncSession, admin_user: User
):
    t, participants = await _seed_live_el_farol_with_stalled(session, admin_user)
    target = participants["alpha"]  # already submitted for round 2
    service = TournamentService(session=session, bus=TournamentEventBus())
    await service.kick_participant(t.id, target.id)

    stmt = select(Action).where(Action.participant_id == target.id)
    rows = (await session.execute(stmt)).scalars().all()
    assert len(rows) == 1
    assert rows[0].source == ActionSource.SUBMITTED.value


@pytest.mark.anyio
async def test_kick_rejects_double_release(session: AsyncSession, admin_user: User):
    t, participants = await _seed_live_el_farol_with_stalled(session, admin_user)
    target = participants["stalled_bot"]
    service = TournamentService(session=session, bus=TournamentEventBus())
    await service.kick_participant(t.id, target.id)
    with pytest.raises(ValueError):
        await service.kick_participant(t.id, target.id)


@pytest.mark.anyio
async def test_kick_raises_lookup_error_for_unknown_participant(
    session: AsyncSession, admin_user: User
):
    t, _ = await _seed_live_el_farol_with_stalled(session, admin_user)
    service = TournamentService(session=session, bus=TournamentEventBus())
    with pytest.raises(LookupError):
        await service.kick_participant(t.id, 99999999)


@pytest.mark.anyio
async def test_kick_rejects_completed_tournament(
    session: AsyncSession, admin_user: User
):
    """Copilot review PR #58 comment 6 — no post-mortem mutation."""
    t, participants = await _seed_live_el_farol_with_stalled(session, admin_user)
    # Flip the tournament to COMPLETED to simulate post-mortem state.
    t.status = TournamentStatus.COMPLETED.value
    await session.commit()

    service = TournamentService(session=session, bus=TournamentEventBus())
    with pytest.raises(ValueError, match="live"):
        await service.kick_participant(t.id, participants["alpha"].id)

    # And released_at must still be None.
    refreshed = await session.get(Participant, participants["alpha"].id)
    assert refreshed is not None
    assert refreshed.released_at is None


@pytest.mark.anyio
async def test_kick_rejects_cancelled_tournament(
    session: AsyncSession, admin_user: User
):
    t, participants = await _seed_live_el_farol_with_stalled(session, admin_user)
    t.status = TournamentStatus.CANCELLED.value
    await session.commit()

    service = TournamentService(session=session, bus=TournamentEventBus())
    with pytest.raises(ValueError, match="live"):
        await service.kick_participant(t.id, participants["stalled_bot"].id)
