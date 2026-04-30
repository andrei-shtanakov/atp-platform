"""Integration tests for /ui/matches/{match_id} (LABS-102).

The view is a thin Jinja shell — it reshapes the GameResult row via the
LABS-99 helper and inlines the result as ``window.__ATP_MATCH__`` so
the client bundle (``/static/v2/js/el_farol/*``) can boot without a
second fetch. Tests cover: happy path renders dashboard skeleton,
legacy row shows schema message, in-progress row shows waiting page
with HTMX auto-refresh, unknown match 404s in-place.
"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.models import GameResult
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


def _complete_row(match_id: str = "m-ui") -> GameResult:
    """A small, Phase-7-complete GameResult row."""
    return GameResult(
        game_name="el_farol",
        game_type="el_farol_interval",
        num_players=2,
        num_rounds=3,
        status="completed",
        match_id=match_id,
        num_days=3,
        num_slots=16,
        max_intervals=2,
        max_total_slots=8,
        capacity_ratio=0.6,
        capacity_threshold=1,
        agents_json=[
            {"agent_id": "a1", "display_name": "a1", "user_id": "u1", "color": "#00f"},
            {"agent_id": "a2", "display_name": "a2", "user_id": "u2", "color": "#0f0"},
        ],
        actions_json=[
            {
                "match_id": match_id,
                "day": d,
                "agent_id": aid,
                "intervals": {"first": [0, 1], "second": []},
                "picks": [0, 1],
                "num_visits": 1,
                "total_slots": 2,
                "payoff": 1.0,
                "num_under": 2,
                "num_over": 0,
                "intent": "",
            }
            for d in (1, 2, 3)
            for aid in ("a1", "a2")
        ],
        day_aggregates_json=[
            {
                "match_id": match_id,
                "day": d,
                "slot_attendance": [0] * 16,
                "over_slots": 0,
                "total_attendances": 0,
            }
            for d in (1, 2, 3)
        ],
    )


class TestMatchDetailUI:
    @pytest.mark.anyio
    async def test_happy_path_renders_dashboard_skeleton(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        db_session.add(_complete_row())
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches/m-ui")

        assert resp.status_code == 200
        html = resp.text
        # shell is the scoped root div
        assert 'id="atp-el-farol"' in html
        assert 'class="atp-dashboard-dark' in html
        # bundle scripts are referenced
        assert "/static/v2/js/el_farol/data_helpers.js" in html
        assert "/static/v2/js/el_farol/dashboard.js" in html
        # scoped stylesheet
        assert "/static/v2/css/el_farol.css" in html
        # server-injected payload shows up as JSON with the expected
        # shape_version
        assert "window.__ATP_MATCH__" in html
        assert '"shape_version":1' in html
        # header reflects the match
        assert "Match <b>m-ui</b>" in html
        # scrubber max is templated from num_days
        assert 'max="3"' in html

    @pytest.mark.anyio
    async def test_legacy_row_shows_predates_schema_message(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        db_session.add(
            GameResult(
                game_name="el_farol",
                game_type="el_farol",
                num_players=2,
                num_rounds=3,
                status="completed",
                match_id="m-legacy",
            )
        )
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches/m-legacy")

        assert resp.status_code == 200
        html = resp.text
        assert "pre-Phase-7" in html
        # no dashboard skeleton
        assert 'id="atp-el-farol"' not in html

    @pytest.mark.anyio
    async def test_tournament_backed_row_renders_cards_dashboard(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """Tournament-backed GameResult rows are rendered by reading the
        authoritative Round/Action/Participant ORM rows (LABS-106). The
        old stopgap that punted to /ui/tournaments/{id} is gone — the
        same /ui/matches/{id} URL now serves the full Cards dashboard.
        """
        from atp.dashboard.tournament.models import (
            Action,
            Participant,
            Round,
            Tournament,
        )

        t = Tournament(
            game_type="el_farol",
            num_players=2,
            total_rounds=1,
            status="completed",
        )
        db_session.add(t)
        await db_session.flush()

        p1 = Participant(
            tournament_id=t.id,
            agent_name="a1",
            builtin_strategy="el_farol/a1",
        )
        p2 = Participant(
            tournament_id=t.id,
            agent_name="a2",
            builtin_strategy="el_farol/a2",
        )
        db_session.add_all([p1, p2])
        await db_session.flush()

        r = Round(
            tournament_id=t.id,
            round_number=1,
            status="completed",
        )
        db_session.add(r)
        await db_session.flush()

        db_session.add_all(
            [
                Action(
                    round_id=r.id,
                    participant_id=p1.id,
                    action_data={"intervals": [[0, 1]]},
                    payoff=2.0,
                ),
                Action(
                    round_id=r.id,
                    participant_id=p2.id,
                    action_data={"intervals": [[2, 3]]},
                    payoff=2.0,
                ),
            ]
        )

        db_session.add(
            GameResult(
                game_name="el_farol",
                game_type="el_farol_interval",
                num_players=2,
                num_rounds=1,
                status="completed",
                match_id="m-tourney",
                tournament_id=t.id,
            )
        )
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches/m-tourney")

        assert resp.status_code == 200
        html = resp.text
        assert 'id="atp-el-farol"' in html
        assert "window.__ATP_MATCH__" in html
        assert "pre-Phase-7" not in html
        assert "tournament replay not yet available" not in html
        # Payload's match_id must match the URL — Cards JS keys
        # per-match localStorage off window.__ATP_MATCH__.match_id.
        assert '"match_id":"m-tourney"' in html

    @pytest.mark.anyio
    async def test_non_el_farol_tournament_match_falls_back_to_placeholder(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """``/ui/matches/{id}`` is an El Farol-only dashboard. A PD or
        public_goods tournament match must not be force-rendered through
        the El Farol reshape (it would produce nonsense — 16-slot grid,
        wrong builtin prefix, etc.). It should fall through to the
        legacy ``predates_schema`` placeholder instead.
        """
        from atp.dashboard.tournament.models import Tournament

        t = Tournament(
            game_type="prisoners_dilemma",
            num_players=2,
            total_rounds=1,
            status="completed",
        )
        db_session.add(t)
        await db_session.flush()

        db_session.add(
            GameResult(
                game_name="prisoners_dilemma",
                game_type="prisoners_dilemma",
                num_players=2,
                num_rounds=1,
                status="completed",
                match_id="m-pd-tourney",
                tournament_id=t.id,
            )
        )
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches/m-pd-tourney")

        assert resp.status_code == 200
        html = resp.text
        assert 'id="atp-el-farol"' not in html
        assert "pre-Phase-7" in html

    @pytest.mark.anyio
    async def test_in_progress_row_shows_waiting_page(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        db_session.add(
            GameResult(
                game_name="el_farol",
                game_type="el_farol_interval",
                num_players=2,
                num_rounds=3,
                status="running",
                match_id="m-run",
                num_days=3,
                num_slots=16,
            )
        )
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches/m-run")

        assert resp.status_code == 200
        html = resp.text
        assert "still running" in html
        # htmx auto-refresh wired
        assert 'hx-trigger="every 15s"' in html
        assert 'hx-get="/ui/matches/m-run"' in html

    @pytest.mark.anyio
    async def test_unknown_match_shows_not_found_in_place(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches/nope")

        # The UI route returns 200 with a friendly message rather than a
        # 404 page — consistent with other /ui/* pages.
        assert resp.status_code == 200
        html = resp.text
        assert "not found" in html.lower()
        assert 'id="atp-el-farol"' not in html

    @pytest.mark.anyio
    async def test_private_tournament_match_hidden_from_non_owner(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """IDOR guard: detail route must not leak private tournament
        matches to non-owner callers, even by PK enumeration."""
        from atp.dashboard.models import User
        from atp.dashboard.tournament.models import Tournament

        owner = User(
            username="owner",
            email="o@t.com",
            hashed_password="x",
            is_active=True,
        )
        db_session.add(owner)
        await db_session.commit()

        priv = Tournament(
            game_type="el_farol",
            num_players=2,
            total_rounds=1,
            created_by=owner.id,
            join_token="secret-invite",
        )
        db_session.add(priv)
        await db_session.commit()

        row = _complete_row(match_id="m-private")
        row.tournament_id = priv.id
        db_session.add(row)
        await db_session.commit()
        await db_session.refresh(row)
        pk = row.id

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # Anonymous should see the "not found" message for both the
            # match_id and the autoincrement PK — no enumeration vector.
            resp = await client.get("/ui/matches/m-private")
            html = resp.text
            assert resp.status_code == 200
            assert 'id="atp-el-farol"' not in html
            assert "not found" in html.lower()

            resp = await client.get(f"/ui/matches/{pk}")
            html = resp.text
            assert resp.status_code == 200
            assert 'id="atp-el-farol"' not in html
            assert "not found" in html.lower()
