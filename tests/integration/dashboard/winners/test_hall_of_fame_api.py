"""HTTP-level tests for /api/public/leaderboard/el-farol."""

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
from atp.dashboard.v2.services.winners import SCHEMA_VERSION


async def _seed(session: AsyncSession) -> None:
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
    )
    session.add(agent)
    await session.flush()
    starts = datetime(2026, 5, 1, 12, 0, 0)
    t = Tournament(
        tenant_id=DEFAULT_TENANT_ID,
        game_type="el_farol",
        config={"name": "T1"},
        status=TournamentStatus.COMPLETED,
        starts_at=starts,
        ends_at=starts + timedelta(minutes=10),
        num_players=2,
        total_rounds=5,
        round_deadline_s=30,
        join_token=None,
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


@pytest.mark.anyio
async def test_hall_of_fame_returns_payload_shape(
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    await _seed(db_session)

    r = await client.get("/api/public/leaderboard/el-farol")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["schema_version"] == SCHEMA_VERSION
    assert "generated_at" in body
    assert body["total"] == 1
    assert body["limit"] == 50
    assert body["offset"] == 0
    entry = body["entries"][0]
    assert entry["agent_name"] == "alfa"
    assert entry["owner_username"] == "alice"
    assert entry["total_score"] == 42.0
    assert entry["tournaments_count"] == 1
    assert r.headers["Cache-Control"] == "public, s-maxage=60"


@pytest.mark.anyio
async def test_hall_of_fame_rejects_out_of_bounds(
    disable_dashboard_auth,
    client: AsyncClient,
):
    for path in (
        "/api/public/leaderboard/el-farol?limit=0",
        "/api/public/leaderboard/el-farol?limit=201",
        "/api/public/leaderboard/el-farol?offset=-1",
    ):
        r = await client.get(path)
        assert r.status_code == 422, path


@pytest.mark.anyio
async def test_hall_of_fame_pagination_keys_are_distinct_in_cache(
    db_session: AsyncSession,
    disable_dashboard_auth,
    monkeypatch,
    client: AsyncClient,
):
    """Two consecutive requests with different ``offset`` must NOT share
    a cache hit. Patch ``_hall_of_fame_query`` to count calls."""
    await _seed(db_session)

    from atp.dashboard.v2.services import winners as winners_service

    real_query = winners_service._hall_of_fame_query
    calls: list[tuple[int, int]] = []

    async def _counting(sess, *, limit, offset):
        calls.append((limit, offset))
        return await real_query(sess, limit=limit, offset=offset)

    monkeypatch.setattr(winners_service, "_hall_of_fame_query", _counting)

    await client.get("/api/public/leaderboard/el-farol?limit=2&offset=0")
    await client.get("/api/public/leaderboard/el-farol?limit=2&offset=2")
    # Second hit at offset=0 should be cached.
    await client.get("/api/public/leaderboard/el-farol?limit=2&offset=0")

    # Two distinct (limit, offset) keys → two underlying queries; the
    # third request hits the cache and does not append.
    assert calls == [(2, 0), (2, 2)]


@pytest.mark.anyio
async def test_hall_of_fame_html_and_json_return_same_top_entry(
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    """The shared ``_hall_of_fame_query`` must yield the same top
    entry whether consumed via HTML or JSON. Catches future drift."""
    await _seed(db_session)

    json_r = await client.get("/api/public/leaderboard/el-farol")
    html_r = await client.get("/ui/leaderboard/el-farol")
    assert json_r.status_code == 200
    assert html_r.status_code == 200

    top = json_r.json()["entries"][0]
    # Both surfaces must mention the top agent name; the JSON gives the
    # exact value and the HTML must contain it inside <tbody>.
    assert top["agent_name"] in html_r.text
    assert top["owner_username"] in html_r.text
    # The score formatted by the template is "%.2f" — JSON value must
    # round to the same string.
    assert f"{top['total_score']:.2f}" in html_r.text
