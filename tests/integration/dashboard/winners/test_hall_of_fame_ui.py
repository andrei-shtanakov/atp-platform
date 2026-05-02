"""HTML tests for /ui/leaderboard/el-farol."""

from __future__ import annotations

import re
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


@pytest.mark.anyio
async def test_hof_empty_state(
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    r = await client.get("/ui/leaderboard/el-farol")
    assert r.status_code == 200
    assert "No completed El Farol tournaments yet." in r.text
    assert r.headers["Cache-Control"] == "public, s-maxage=60"
    # The sidebar link must be present (not just the page H2).
    assert '<a href="/ui/leaderboard/el-farol"' in r.text


@pytest.mark.anyio
async def test_hof_renders_top_agents(
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    starts = datetime(2026, 5, 1, 12, 0, 0)
    alice = User(
        username="alice",
        email="a@e.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    bob = User(
        username="bob",
        email="b@e.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    db_session.add_all([alice, bob])
    await db_session.flush()
    a = Agent(
        tenant_id=DEFAULT_TENANT_ID,
        name="alfa",
        agent_type="tournament",
        owner_id=alice.id,
        purpose="tournament",
    )
    b = Agent(
        tenant_id=DEFAULT_TENANT_ID,
        name="beta",
        agent_type="tournament",
        owner_id=bob.id,
        purpose="tournament",
    )
    db_session.add_all([a, b])
    await db_session.flush()
    t = Tournament(
        tenant_id=DEFAULT_TENANT_ID,
        game_type="el_farol",
        config={"name": "T"},
        status=TournamentStatus.COMPLETED,
        starts_at=starts,
        ends_at=starts + timedelta(minutes=10),
        num_players=2,
        total_rounds=5,
        round_deadline_s=30,
        join_token=None,
        pending_deadline=starts,
    )
    db_session.add(t)
    await db_session.flush()
    db_session.add_all(
        [
            Participant(
                tournament_id=t.id,
                user_id=alice.id,
                agent_id=a.id,
                agent_name="alfa",
                total_score=20.0,
            ),
            Participant(
                tournament_id=t.id,
                user_id=bob.id,
                agent_id=b.id,
                agent_name="beta",
                total_score=30.0,
            ),
        ]
    )
    await db_session.commit()

    r = await client.get("/ui/leaderboard/el-farol")
    assert r.status_code == 200
    # bob (30) ranks above alice (20) by total score.
    # Extract the leaderboard <tbody> contents — that's where the agent
    # names should appear in score-DESC order.
    tbody = re.search(r"<tbody>(.*?)</tbody>", r.text, re.DOTALL)
    assert tbody is not None, "no <tbody> in HoF response"
    body = tbody.group(1)
    beta_idx = body.find("beta")
    alfa_idx = body.find("alfa")
    assert 0 <= beta_idx < alfa_idx, (
        f"expected 'beta' before 'alfa' inside <tbody>, got {beta_idx=} {alfa_idx=}"
    )


@pytest.mark.anyio
async def test_hof_strict_bounds(
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    for path in (
        "/ui/leaderboard/el-farol?limit=0",
        "/ui/leaderboard/el-farol?limit=201",
        "/ui/leaderboard/el-farol?offset=-1",
    ):
        r = await client.get(path)
        assert r.status_code == 422, path


@pytest.mark.anyio
async def test_hof_offset_overshoot_shows_back_link_not_empty_message(
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    """When offset exceeds total but tournaments exist, render a 'no
    entries on this page' message with a link back to page 1, not the
    misleading 'no tournaments' empty state."""
    # Seed exactly one qualifying participant so total=1.
    starts = datetime(2026, 5, 1, 12, 0, 0)
    alice = User(
        username="alice",
        email="a@e.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    db_session.add(alice)
    await db_session.flush()
    a = Agent(
        tenant_id=DEFAULT_TENANT_ID,
        name="alfa",
        agent_type="tournament",
        owner_id=alice.id,
        purpose="tournament",
    )
    db_session.add(a)
    await db_session.flush()
    t = Tournament(
        tenant_id=DEFAULT_TENANT_ID,
        game_type="el_farol",
        config={"name": "T"},
        status=TournamentStatus.COMPLETED,
        starts_at=starts,
        ends_at=starts + timedelta(minutes=10),
        num_players=2,
        total_rounds=5,
        round_deadline_s=30,
        join_token=None,
        pending_deadline=starts,
    )
    db_session.add(t)
    await db_session.flush()
    db_session.add(
        Participant(
            tournament_id=t.id,
            user_id=alice.id,
            agent_id=a.id,
            agent_name="alfa",
            total_score=10.0,
        )
    )
    await db_session.commit()

    r = await client.get("/ui/leaderboard/el-farol?limit=10&offset=50")
    assert r.status_code == 200
    assert "No entries on this page." in r.text
    assert "Back to first page" in r.text
    # The misleading empty-state copy must NOT appear:
    assert "No completed El Farol tournaments yet." not in r.text


@pytest.mark.anyio
async def test_hof_partial_swap_returns_table_fragment_only(
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    """``?partial=1`` returns only the table-body fragment for HTMX swap,
    NOT the full page (no ``<h2>`` heading, no sidebar)."""
    r = await client.get("/ui/leaderboard/el-farol?partial=1")
    assert r.status_code == 200
    # Empty leaderboard renders the "no tournaments" message in the
    # partial. The full-page heading must be absent.
    assert "El Farol Hall of Fame" not in r.text  # only on the H2 in full page
    assert "No completed El Farol tournaments yet." in r.text
