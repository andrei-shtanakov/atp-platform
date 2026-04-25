"""Integration tests for /ui/tournaments/{id} (LABS-106 cross-link).

After the read-time tournament reshape lands in /ui/matches/{id}, the
tournament detail page should expose a one-click "Cards replay" link to
that match instead of pointing at the generic /ui/matches listing.
Cancelled tournaments have no GameResult row, so the link is omitted.
"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
def v2_app(test_database: Database):
    app = create_test_app(use_v2_routes=True)

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session
    return app


@pytest.mark.anyio
async def test_completed_tournament_renders_cards_replay_link(
    v2_app,
    db_session: AsyncSession,
    disable_dashboard_auth,
) -> None:
    from atp.dashboard.models import GameResult
    from atp.dashboard.tournament.models import Tournament

    t = Tournament(
        game_type="el_farol",
        num_players=2,
        total_rounds=1,
        status="completed",
    )
    db_session.add(t)
    await db_session.flush()

    db_session.add(
        GameResult(
            game_name="el_farol",
            game_type="el_farol_interval",
            num_players=2,
            num_rounds=1,
            status="completed",
            match_id="m-replay-link",
            tournament_id=t.id,
        )
    )
    await db_session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=v2_app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/ui/tournaments/{t.id}")

    assert resp.status_code == 200
    html = resp.text
    assert "/ui/matches/m-replay-link" in html
    assert "Cards replay" in html


@pytest.mark.anyio
async def test_cancelled_tournament_has_no_cards_replay_link(
    v2_app,
    db_session: AsyncSession,
    disable_dashboard_auth,
) -> None:
    from atp.dashboard.tournament.models import Tournament

    t = Tournament(
        game_type="el_farol",
        num_players=2,
        total_rounds=1,
        status="cancelled",
    )
    db_session.add(t)
    await db_session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=v2_app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/ui/tournaments/{t.id}")

    assert resp.status_code == 200
    assert "Cards replay" not in resp.text
