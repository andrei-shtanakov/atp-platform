"""HTTP-level tests for the pending tournament banner."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.models import DEFAULT_TENANT_ID, Agent, User
from atp.dashboard.tournament.models import (
    Participant,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
async def client(test_database: Database) -> AsyncIterator[AsyncClient]:
    app = create_test_app()

    async def _override_session() -> AsyncIterator[AsyncSession]:
        async with test_database.session() as s:
            yield s

    app.dependency_overrides[get_db_session] = _override_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _seed_pending_tournament(
    session: AsyncSession,
    *,
    tenant_id: str = DEFAULT_TENANT_ID,
    num_players: int = 4,
    registered: int = 0,
    deadline_minutes: int = 15,
) -> int:
    starts = datetime(2026, 5, 1, 12, 0, 0)
    t = Tournament(
        tenant_id=tenant_id,
        game_type="el_farol",
        config={"name": "T"},
        status=TournamentStatus.PENDING,
        starts_at=starts,
        ends_at=starts + timedelta(minutes=10),
        num_players=num_players,
        total_rounds=5,
        round_deadline_s=30,
        join_token=None,
        pending_deadline=starts + timedelta(minutes=deadline_minutes),
    )
    session.add(t)
    await session.flush()
    for i in range(registered):
        u = User(
            username=f"u{i}",
            email=f"u{i}@e.com",
            hashed_password="x",
            is_admin=False,
            is_active=True,
        )
        session.add(u)
        await session.flush()
        a = Agent(
            tenant_id=DEFAULT_TENANT_ID,
            name=f"agent{i}",
            agent_type="tournament",
            owner_id=u.id,
            purpose="tournament",
        )
        session.add(a)
        await session.flush()
        session.add(
            Participant(
                tournament_id=t.id,
                user_id=u.id,
                agent_id=a.id,
                agent_name=f"agent{i}",
            )
        )
    await session.commit()
    return t.id


@pytest.mark.anyio
async def test_detail_page_renders_banner_for_pending_tournament(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    tid = await _seed_pending_tournament(db_session, num_players=4, registered=2)
    r = await client.get(f"/ui/tournaments/{tid}")
    assert r.status_code == 200
    assert 'id="pending-banner"' in r.text
    assert 'hx-trigger="every 10s"' in r.text
    assert "Registered:" in r.text
    assert "<strong>2</strong>" in r.text
    assert "/ 4" in r.text
    # Wrapper URL must contain the tournament id, not be empty.
    assert f"/ui/tournaments/{tid}?partial=pending-banner" in r.text
    # ISO string must carry a UTC marker.
    assert ("+00:00" in r.text) or ('Z"' in r.text)


@pytest.mark.anyio
async def test_partial_endpoint_returns_wrapper_with_hx_trigger_and_no_store(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    tid = await _seed_pending_tournament(db_session, num_players=4, registered=1)
    r = await client.get(f"/ui/tournaments/{tid}?partial=pending-banner")
    assert r.status_code == 200
    assert 'id="pending-banner"' in r.text
    assert 'hx-trigger="every 10s"' in r.text
    assert "<strong>1</strong>" in r.text
    assert r.headers["Cache-Control"] == "no-store"


@pytest.mark.anyio
async def test_partial_endpoint_after_status_flip_drops_hx_trigger(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    """Race window: pending → active between two HTMX pulls."""
    tid = await _seed_pending_tournament(db_session, num_players=4, registered=2)

    r1 = await client.get(f"/ui/tournaments/{tid}?partial=pending-banner")
    assert 'hx-trigger="every 10s"' in r1.text

    # Flip status; the next pull must return an empty wrapper (no
    # hx-trigger, no inner content) — this is the gracefully-retire
    # swap the design relies on.
    async with test_database.session() as s:
        await s.execute(
            update(Tournament)
            .where(Tournament.id == tid)
            .values(status=TournamentStatus.ACTIVE)
        )
        await s.commit()

    r2 = await client.get(f"/ui/tournaments/{tid}?partial=pending-banner")
    assert r2.status_code == 200
    assert 'id="pending-banner"' in r2.text
    assert "hx-trigger" not in r2.text
    assert "Registered:" not in r2.text


@pytest.mark.anyio
async def test_detail_page_non_default_tenant_renders_empty_wrapper(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    tid = await _seed_pending_tournament(
        db_session, tenant_id="other-tenant", num_players=4
    )
    r = await client.get(f"/ui/tournaments/{tid}")
    assert r.status_code == 200
    assert 'id="pending-banner"' in r.text
    assert "hx-trigger" not in r.text
    assert "Registered:" not in r.text


@pytest.mark.anyio
async def test_live_page_renders_banner_with_correct_wrapper_url(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    """Regression: wrapper template references tournament_id (not
    tournament.id from a non-existent ORM-row context). If the live
    route forgets to put tournament_id in the template context, the
    URL would render as '/ui/tournaments/?partial=pending-banner' and
    HTMX would 404-loop."""
    tid = await _seed_pending_tournament(db_session, num_players=4, registered=2)
    r = await client.get(f"/ui/tournaments/{tid}/live")
    assert r.status_code == 200
    assert 'id="pending-banner"' in r.text
    assert f"/ui/tournaments/{tid}?partial=pending-banner" in r.text
    assert "/ui/tournaments/?partial=pending-banner" not in r.text


@pytest.mark.anyio
async def test_detail_and_live_pages_have_identical_wrapper_url(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    """Both pages must point HTMX at the same partial URL — that's the
    contract for 'one partial endpoint serves both hosts'."""
    tid = await _seed_pending_tournament(db_session, num_players=4, registered=1)
    r_detail = await client.get(f"/ui/tournaments/{tid}")
    r_live = await client.get(f"/ui/tournaments/{tid}/live")
    expected_url = f"/ui/tournaments/{tid}?partial=pending-banner"
    assert expected_url in r_detail.text
    assert expected_url in r_live.text


@pytest.mark.anyio
async def test_match_detail_replay_does_not_render_banner(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    """The same match_detail.html template serves /ui/matches/{id} for
    replay. It must NOT render the banner wrapper there because the
    replay route has no tournament context. The {% if ... is defined %}
    guard is what protects this."""
    r = await client.get("/ui/matches/nonexistent-match")
    assert r.status_code == 200  # placeholder page
    assert 'id="pending-banner"' not in r.text
