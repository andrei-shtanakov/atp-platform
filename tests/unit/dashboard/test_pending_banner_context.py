"""Unit tests for _pending_banner_context helper."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import DEFAULT_TENANT_ID, Agent, User
from atp.dashboard.tournament.models import (
    Participant,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.v2.routes.ui import _pending_banner_context


async def _make_tournament(
    db_session: AsyncSession,
    *,
    status: TournamentStatus = TournamentStatus.PENDING,
    tenant_id: str = DEFAULT_TENANT_ID,
    num_players: int = 4,
) -> Tournament:
    starts = datetime(2026, 5, 1, 12, 0, 0)
    t = Tournament(
        tenant_id=tenant_id,
        game_type="el_farol",
        config={"name": "T"},
        status=status,
        starts_at=starts,
        ends_at=starts + timedelta(minutes=10),
        num_players=num_players,
        total_rounds=5,
        round_deadline_s=30,
        join_token=None,
        pending_deadline=starts + timedelta(minutes=15),
    )
    db_session.add(t)
    await db_session.flush()
    return t


@pytest.mark.anyio
async def test_pending_default_tenant_returns_full_context(db_session: AsyncSession):
    t = await _make_tournament(db_session)
    await db_session.commit()
    # Refresh to ensure participants relationship is empty list, not unloaded.
    await db_session.refresh(t, attribute_names=["participants"])

    ctx = _pending_banner_context(t)
    assert ctx["pending_banner_show"] is True
    assert ctx["pending_planned_count"] == 4
    assert ctx["pending_registered_count"] == 0
    # ISO must carry a UTC marker so the browser parses it correctly.
    assert "+00:00" in ctx["pending_deadline_iso"] or ctx[
        "pending_deadline_iso"
    ].endswith("Z")


async def _make_user(db_session: AsyncSession, username: str) -> User:
    u = User(
        username=username,
        email=f"{username}@example.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    db_session.add(u)
    await db_session.flush()
    return u


async def _make_agent(db_session: AsyncSession, *, owner: User, name: str) -> Agent:
    a = Agent(
        tenant_id=DEFAULT_TENANT_ID,
        name=name,
        agent_type="tournament",
        owner_id=owner.id,
        purpose="tournament",
    )
    db_session.add(a)
    await db_session.flush()
    return a


@pytest.mark.anyio
async def test_active_status_returns_show_false(db_session: AsyncSession):
    t = await _make_tournament(db_session, status=TournamentStatus.ACTIVE)
    await db_session.commit()
    assert _pending_banner_context(t) == {"pending_banner_show": False}


@pytest.mark.anyio
async def test_completed_status_returns_show_false(db_session: AsyncSession):
    t = await _make_tournament(db_session, status=TournamentStatus.COMPLETED)
    await db_session.commit()
    assert _pending_banner_context(t) == {"pending_banner_show": False}


@pytest.mark.anyio
async def test_cancelled_status_returns_show_false(db_session: AsyncSession):
    t = await _make_tournament(db_session, status=TournamentStatus.CANCELLED)
    await db_session.commit()
    assert _pending_banner_context(t) == {"pending_banner_show": False}


@pytest.mark.anyio
async def test_non_default_tenant_returns_show_false(db_session: AsyncSession):
    t = await _make_tournament(db_session, tenant_id="other-tenant")
    await db_session.commit()
    assert _pending_banner_context(t) == {"pending_banner_show": False}


@pytest.mark.anyio
async def test_counter_excludes_released_participants(db_session: AsyncSession):
    t = await _make_tournament(db_session, num_players=4)
    alice = await _make_user(db_session, "alice")
    bob = await _make_user(db_session, "bob")
    carol = await _make_user(db_session, "carol")
    a = await _make_agent(db_session, owner=alice, name="a")
    b = await _make_agent(db_session, owner=bob, name="b")
    c = await _make_agent(db_session, owner=carol, name="c")
    # Three participants; one released.
    p1 = Participant(
        tournament_id=t.id, user_id=alice.id, agent_id=a.id, agent_name="a"
    )
    p2 = Participant(tournament_id=t.id, user_id=bob.id, agent_id=b.id, agent_name="b")
    p3 = Participant(
        tournament_id=t.id,
        user_id=carol.id,
        agent_id=c.id,
        agent_name="c",
        released_at=datetime(2026, 5, 1, 12, 5, 0),
    )
    db_session.add_all([p1, p2, p3])
    await db_session.commit()
    await db_session.refresh(t, attribute_names=["participants"])

    ctx = _pending_banner_context(t)
    assert ctx["pending_registered_count"] == 2  # carol excluded
    assert ctx["pending_planned_count"] == 4
