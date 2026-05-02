"""HTML route tests for /ui/tournaments/{id}/winners."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import DEFAULT_TENANT_ID, Agent, User
from atp.dashboard.tournament.models import (
    Participant,
    Tournament,
    TournamentStatus,
)


async def _seed_tournament(
    session: AsyncSession,
    *,
    status_: TournamentStatus = TournamentStatus.COMPLETED,
    join_token: str | None = None,
    game_type: str = "el_farol",
    tenant_id: str = DEFAULT_TENANT_ID,
    archived: bool = False,
) -> int:
    alice = User(
        username="alice",
        email="a@e.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    session.add(alice)
    await session.flush()
    agent = Agent(
        tenant_id=DEFAULT_TENANT_ID,
        name="alfa",
        agent_type="tournament",
        owner_id=alice.id,
        description="greedy",
        purpose="tournament",
        deleted_at=datetime(2026, 5, 1, 0, 0, 0) if archived else None,
    )
    session.add(agent)
    await session.flush()
    starts = datetime(2026, 5, 1, 12, 0, 0)
    ends = starts + timedelta(minutes=12, seconds=4)
    t = Tournament(
        tenant_id=tenant_id,
        game_type=game_type,
        config={"name": "T1"},
        status=status_,
        starts_at=starts,
        ends_at=ends,
        num_players=2,
        total_rounds=5,
        round_deadline_s=30,
        join_token=join_token,
        pending_deadline=starts,
    )
    session.add(t)
    await session.flush()
    p = Participant(
        tournament_id=t.id,
        user_id=alice.id,
        agent_id=agent.id,
        agent_name="alfa",
        total_score=42.0,
    )
    session.add(p)
    await session.commit()
    return t.id


@pytest.mark.anyio
async def test_winners_page_renders_for_completed_public_tournament(
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    tid = await _seed_tournament(db_session)
    r = await client.get(f"/ui/tournaments/{tid}/winners")
    assert r.status_code == 200
    assert "alfa" in r.text
    assert "Players: 2" in r.text
    assert "Days: 5" in r.text
    assert "Capacity: 1" in r.text  # max(1, int(0.6 * 2)) = 1
    assert "12m 4s" in r.text
    assert r.headers["Cache-Control"] == "public, s-maxage=60"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"status_": TournamentStatus.PENDING},
        {"status_": TournamentStatus.ACTIVE},
        {"status_": TournamentStatus.CANCELLED},
        {"join_token": "secret"},
        {"game_type": "prisoners_dilemma"},
        {"tenant_id": "other"},
    ],
)
async def test_winners_page_404_on_gate_violations(
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
    kwargs,
):
    tid = await _seed_tournament(db_session, **kwargs)
    r = await client.get(f"/ui/tournaments/{tid}/winners")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_winners_page_404_for_missing_tournament(
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    r = await client.get("/ui/tournaments/9999/winners")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_winners_page_archived_agent_suffix(
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    tid = await _seed_tournament(db_session, archived=True)
    r = await client.get(f"/ui/tournaments/{tid}/winners")
    assert r.status_code == 200
    assert "alfa (archived)" in r.text
