"""LABS-TSA PR-5 — /ui/tournaments/new self-service form.

Exercises the GET form renderer and the POST handler that calls
``TournamentService.create_tournament`` and server-renders the detail
page directly (no redirect) so the one-time ``join_token`` reveal
cannot be exposed via a follow-up GET.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.models import Agent, User
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


async def _seed_user_with_tournament_agent(db_session: AsyncSession) -> User:
    user = User(
        username="ui-user",
        email="ui-user@test.com",
        hashed_password="x",
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    db_session.add(
        Agent(
            name="t-agent",
            agent_type="mcp",
            owner_id=user.id,
            purpose="tournament",
        )
    )
    await db_session.commit()
    return user


class TestTournamentNewGet:
    @pytest.mark.anyio
    async def test_form_renders(
        self,
        v2_app,
        disable_dashboard_auth,
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/tournaments/new")
        assert resp.status_code == 200
        html = resp.text.lower()
        assert "<form" in html
        assert 'name="game_type"' in resp.text
        assert 'name="num_players"' in resp.text
        assert 'name="private"' in resp.text


class TestTournamentNewPost:
    @pytest.mark.anyio
    async def test_post_creates_and_renders_detail_with_token(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """POSTing a valid form creates a private tournament and renders
        the detail page inline with the one-time join_token reveal."""
        from atp.dashboard.tournament.models import Tournament

        await _seed_user_with_tournament_agent(db_session)

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/ui/tournaments/new",
                data={
                    "game_type": "el_farol",
                    "num_players": "3",
                    "total_rounds": "5",
                    "round_deadline_s": "30",
                    "private": "on",
                    "roster[]": [
                        "el_farol/traditionalist",
                        "el_farol/contrarian",
                    ],
                },
            )
        assert resp.status_code == 200
        body = resp.text

        # Assert the tournament was created
        from sqlalchemy import select

        rows = (await db_session.execute(select(Tournament))).scalars().all()
        assert len(rows) == 1
        t = rows[0]
        assert t.game_type == "el_farol"
        assert t.num_players == 3
        assert t.total_rounds == 5
        # Private tournament has a join_token
        assert t.join_token is not None
        # One-time reveal: the detail HTML must echo the plaintext token once
        assert t.join_token in body
