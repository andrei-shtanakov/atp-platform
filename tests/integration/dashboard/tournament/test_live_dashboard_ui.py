"""HTML-shape tests for the live tournament dashboard UI route.

Covers ``GET /ui/tournaments/{tournament_id}/live`` which renders the
same ``match_detail.html`` skeleton whether or not any rounds have
resolved. The empty-state behaviour is now driven entirely by
``dashboard.js`` short-circuiting empty ``DATA``; the server no longer
branches on ``payload.NUM_DAYS == 0`` and the template no longer has a
``live_waiting`` placeholder block.

These tests assert the static HTML shape produced by the route. They
intentionally do not exercise the JS empty-state DOM (manual QA covers
that).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.tournament.models import (
    Action,
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app

# ---------- fixtures ----------


@pytest.fixture
def v2_app(test_database: Database):
    """Create a test app with v2 routes wired to the in-memory test DB.

    Mirrors the fixture in ``test_live_dashboard_endpoints.py`` — the UI
    route resolves its session via ``get_db_session`` so we override
    that to read/write the seeded rows.
    """
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


# ---------- seeding helpers ----------


async def _make_pending_el_farol(
    session: AsyncSession,
    *,
    num_players: int = 2,
    total_rounds: int = 1,
) -> Tournament:
    t = Tournament(
        game_type="el_farol",
        status=TournamentStatus.PENDING,
        num_players=num_players,
        total_rounds=total_rounds,
        round_deadline_s=30,
        created_by=None,
        config={},
        rules={},
        join_token=None,
    )
    session.add(t)
    await session.commit()
    await session.refresh(t)
    return t


async def _seed_two_participants(
    session: AsyncSession, tournament: Tournament
) -> tuple[Participant, Participant]:
    p1 = Participant(
        tournament_id=tournament.id,
        user_id=None,
        agent_name="builtin-1",
        agent_id=None,
        builtin_strategy="el_farol/calibrated",
    )
    p2 = Participant(
        tournament_id=tournament.id,
        user_id=None,
        agent_name="builtin-2",
        agent_id=None,
        builtin_strategy="el_farol/conservative",
    )
    session.add_all([p1, p2])
    await session.commit()
    await session.refresh(p1)
    await session.refresh(p2)
    return p1, p2


async def _add_resolved_round(
    session: AsyncSession,
    tournament: Tournament,
    p1: Participant,
    p2: Participant,
    *,
    round_number: int = 1,
) -> Round:
    r = Round(
        tournament_id=tournament.id,
        round_number=round_number,
        status=RoundStatus.COMPLETED,
        state={},
    )
    session.add(r)
    await session.flush()

    a1 = Action(
        round_id=r.id,
        participant_id=p1.id,
        action_data={"intervals": [[0, 0]]},
        payoff=-1.0,
        source="submitted",
    )
    a2 = Action(
        round_id=r.id,
        participant_id=p2.id,
        action_data={"intervals": [[0, 0]]},
        payoff=-1.0,
        source="submitted",
    )
    session.add_all([a1, a2])
    await session.commit()
    return r


# ---------- tests ----------


class TestLiveDashboardUI:
    @pytest.mark.anyio
    async def test_live_ui_zero_rounds_renders_skeleton(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """GIVEN a public El Farol tournament with two participants and
            zero completed rounds
        WHEN an anonymous viewer GETs ``/ui/tournaments/{id}/live``
        THEN the full match_detail dashboard skeleton is rendered
            (no ``live_waiting`` placeholder), the SSE subscriber script
            is loaded, and the topbar shows ``0 days``.
        """
        t = await _make_pending_el_farol(db_session)
        await _seed_two_participants(db_session, t)

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get(f"/ui/tournaments/{t.id}/live")

        assert resp.status_code == 200, resp.text
        body = resp.text

        # Full dashboard skeleton is present even with zero rounds.
        assert 'id="atp-el-farol"' in body
        assert 'id="cards"' in body
        assert 'id="heatmap"' in body

        # SSE subscriber is wired up.
        assert "/static/v2/js/el_farol/live_subscriber.js" in body

        # The removed live_waiting microcopy must be gone.
        assert "Tournament starting" not in body

        # Sanity check on num_days=0 plumbing (topbar match-meta line:
        # ``· {{ num_days }} days ·``).
        assert "0 days" in body

    @pytest.mark.anyio
    async def test_live_ui_with_rounds_renders_dashboard(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """GIVEN a public El Farol tournament with two participants and
            one completed round
        WHEN an anonymous viewer GETs ``/ui/tournaments/{id}/live``
        THEN the full dashboard is rendered with the JSON payload script
            tag and ``num_days`` reflects the resolved round count.
        """
        t = await _make_pending_el_farol(db_session)
        p1, p2 = await _seed_two_participants(db_session, t)
        await _add_resolved_round(db_session, t, p1, p2, round_number=1)

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get(f"/ui/tournaments/{t.id}/live")

        assert resp.status_code == 200, resp.text
        body = resp.text

        # Full dashboard skeleton + payload script tag.
        assert 'id="atp-el-farol"' in body
        assert 'id="atp-match-payload"' in body

        # num_days reflects the one resolved round in the topbar
        # match-meta line ``· {{ num_days }} days ·``.
        assert "1 days" in body
