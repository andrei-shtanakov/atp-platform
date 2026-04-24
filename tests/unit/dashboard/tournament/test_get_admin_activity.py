"""Unit tests for TournamentService.get_admin_activity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import Agent, User
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


async def _seed_live_el_farol(
    session: AsyncSession, creator: User
) -> tuple[Tournament, list[Participant]]:
    """3-player El Farol: round 1 completed (one timeout), round 2 in progress
    with only one submitted so far, round 3+ not started, total_rounds=5."""
    now = datetime.now(tz=UTC).replace(tzinfo=None)

    t = Tournament(
        game_type="el_farol",
        status=TournamentStatus.ACTIVE.value,
        num_players=3,
        total_rounds=5,
        round_deadline_s=30,
        created_by=creator.id,
        created_at=now - timedelta(minutes=3),
        starts_at=now - timedelta(minutes=2),
        pending_deadline=now - timedelta(minutes=2),
    )
    session.add(t)
    await session.flush()

    participants: list[Participant] = []
    for name in ("alpha", "beta", "stalled_bot"):
        u = User(
            username=f"bot_{name}",
            email=f"{name}@t.com",
            hashed_password="x",
            is_active=True,
        )
        session.add(u)
        await session.flush()
        # LABS-TSA PR-4: non-builtin Participant rows need agent_id.
        ag = Agent(
            tenant_id="default",
            name=name,
            agent_type="mcp",
            owner_id=u.id,
            config={},
            purpose="tournament",
        )
        session.add(ag)
        await session.flush()
        p = Participant(
            tournament_id=t.id,
            user_id=u.id,
            agent_id=ag.id,
            agent_name=name,
            total_score=0.0,
        )
        session.add(p)
        participants.append(p)
    await session.flush()

    # Round 1 — COMPLETED. alpha + beta submitted, stalled_bot was timed out.
    r1 = Round(
        tournament_id=t.id,
        round_number=1,
        status=RoundStatus.COMPLETED.value,
        started_at=now - timedelta(minutes=2),
        deadline=now - timedelta(minutes=1, seconds=30),
    )
    session.add(r1)
    await session.flush()
    for p, is_timeout in zip(participants, [False, False, True], strict=True):
        session.add(
            Action(
                round_id=r1.id,
                participant_id=p.id,
                action_data={"slots": [0, 1] if not is_timeout else []},
                submitted_at=now - timedelta(minutes=1, seconds=35),
                source=(
                    ActionSource.TIMEOUT_DEFAULT.value
                    if is_timeout
                    else ActionSource.SUBMITTED.value
                ),
                payoff=0.0,
            )
        )

    # Round 2 — IN_PROGRESS. Only alpha has submitted so far.
    r2_deadline = now + timedelta(seconds=20)
    r2 = Round(
        tournament_id=t.id,
        round_number=2,
        status=RoundStatus.IN_PROGRESS.value,
        started_at=now - timedelta(seconds=10),
        deadline=r2_deadline,
    )
    session.add(r2)
    await session.flush()
    session.add(
        Action(
            round_id=r2.id,
            participant_id=participants[0].id,
            action_data={"slots": [2, 3]},
            submitted_at=now - timedelta(seconds=5),
            source=ActionSource.SUBMITTED.value,
        )
    )
    await session.commit()
    return t, participants


@pytest.mark.anyio
async def test_snapshot_shape(session: AsyncSession, admin_user: User):
    t, _ = await _seed_live_el_farol(session, admin_user)
    service = TournamentService(session=session, bus=TournamentEventBus())
    snap = await service.get_admin_activity(t.id)

    assert snap["tournament_id"] == t.id
    assert snap["status"] == TournamentStatus.ACTIVE.value
    assert snap["total_rounds"] == 5
    assert snap["current_round"] == 2
    assert isinstance(snap["deadline_remaining_s"], int)
    assert 0 < snap["deadline_remaining_s"] <= 30
    assert len(snap["participants"]) == 3
    for p in snap["participants"]:
        assert p["current_round_status"] in {
            "submitted",
            "waiting",
            "timeout",
            "released",
        }
        assert len(p["row_per_round"]) == 5
        for cell in p["row_per_round"]:
            assert cell in {"submitted", "timeout", "waiting"}


@pytest.mark.anyio
async def test_snapshot_counts_submitted_this_round(
    session: AsyncSession, admin_user: User
):
    t, _ = await _seed_live_el_farol(session, admin_user)
    service = TournamentService(session=session, bus=TournamentEventBus())
    snap = await service.get_admin_activity(t.id)

    # Round 2: alpha submitted; beta + stalled_bot still waiting.
    assert snap["submitted_this_round"] == 1
    assert snap["total_this_round"] == 3


@pytest.mark.anyio
async def test_snapshot_current_round_statuses(session: AsyncSession, admin_user: User):
    t, participants = await _seed_live_el_farol(session, admin_user)
    service = TournamentService(session=session, bus=TournamentEventBus())
    snap = await service.get_admin_activity(t.id)

    by_name = {p["agent_name"]: p for p in snap["participants"]}
    assert by_name["alpha"]["current_round_status"] == "submitted"
    assert by_name["beta"]["current_round_status"] == "waiting"
    assert by_name["stalled_bot"]["current_round_status"] == "waiting"


@pytest.mark.anyio
async def test_snapshot_heatmap_reflects_timeout_in_round_1(
    session: AsyncSession, admin_user: User
):
    t, _ = await _seed_live_el_farol(session, admin_user)
    service = TournamentService(session=session, bus=TournamentEventBus())
    snap = await service.get_admin_activity(t.id)

    by_name = {p["agent_name"]: p for p in snap["participants"]}
    # Round 1 (index 0): alpha + beta submitted, stalled_bot timed out.
    assert by_name["alpha"]["row_per_round"][0] == "submitted"
    assert by_name["beta"]["row_per_round"][0] == "submitted"
    assert by_name["stalled_bot"]["row_per_round"][0] == "timeout"
    # Rounds 3..5 (index 2..4) are not started yet → waiting for everyone.
    for p in snap["participants"]:
        assert p["row_per_round"][2] == "waiting"
        assert p["row_per_round"][3] == "waiting"
        assert p["row_per_round"][4] == "waiting"


@pytest.mark.anyio
async def test_get_admin_activity_raises_lookup_for_unknown_id(
    session: AsyncSession,
):
    service = TournamentService(session=session, bus=TournamentEventBus())
    with pytest.raises(LookupError):
        await service.get_admin_activity(99999999)
