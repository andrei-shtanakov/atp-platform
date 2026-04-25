"""Integration tests for /ui/matches listing."""

from collections.abc import AsyncGenerator
from datetime import datetime

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


class TestMatchesListing:
    @pytest.mark.anyio
    async def test_empty_state_prompts_how_to_run(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches")
        assert resp.status_code == 200
        assert "No completed El Farol matches yet" in resp.text

    @pytest.mark.anyio
    async def test_lists_phase7_el_farol_matches_only(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        # Three rows covering the filter logic:
        #   A. el_farol + actions_json populated → listed
        #   B. el_farol + actions_json null (legacy) → hidden
        #   C. prisoners_dilemma + actions_json populated → hidden
        actions = [{"day": 1, "agent_id": "a1"}]
        aggs = [{"day": 1, "slot_attendance": [0] * 16, "over_slots": 0}]
        db_session.add_all(
            [
                # Production CLI writer stores game_name as the engine's
                # pretty name, e.g. "El Farol Bar (n=6, threshold=4,
                # days=30)". Filter must not depend on the literal
                # "el_farol" key.
                GameResult(
                    game_name="El Farol Bar (n=6, threshold=4, days=30)",
                    game_type="repeated",
                    num_players=6,
                    num_rounds=30,
                    status="completed",
                    completed_at=datetime(2026, 4, 20, 12, 0),
                    match_id="m-pretty-name",
                    num_days=30,
                    num_slots=16,
                    actions_json=actions,
                    day_aggregates_json=aggs,
                ),
                # Legacy row: matches on game_name but has no Phase-7
                # columns → hidden because not renderable.
                GameResult(
                    game_name="el_farol",
                    game_type="el_farol",
                    num_players=2,
                    num_rounds=3,
                    status="completed",
                    match_id="m-legacy",
                    # actions_json deliberately null
                ),
            ]
        )
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches")

        assert resp.status_code == 200
        html = resp.text
        assert "m-pretty-name" in html
        assert "/ui/matches/m-pretty-name" in html
        # legacy row hidden because Phase-7 columns missing
        assert "m-legacy" not in html
        assert "1 completed El Farol matches" in html or ">1 completed" in html

    @pytest.mark.anyio
    async def test_lists_tournament_backed_el_farol_match(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """Tournament-backed El Farol matches have NULL Phase-7 JSON
        columns by design (LABS-106 reshapes them at read time from
        Round/Action ORM rows). The listing must surface them anyway,
        otherwise the row is reachable only via the
        /ui/tournaments/{id} cross-link.
        """
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
                # Production writer sets ``game_type=variant`` from
                # ``tournament.config`` (default "tournament"), not the
                # tournament's game_type. The listing filter keys off
                # ``game_name``, not ``game_type``, so this is fine —
                # but the fixture mirrors prod for less surprise.
                game_type="tournament",
                num_players=2,
                num_rounds=1,
                status="completed",
                completed_at=datetime(2026, 4, 25, 12, 0),
                match_id="m-tourney-listed",
                tournament_id=t.id,
                # actions_json / day_aggregates_json deliberately NULL —
                # the tournament writer does not populate them.
            )
        )
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches")

        assert resp.status_code == 200
        html = resp.text
        assert "m-tourney-listed" in html
        assert "/ui/matches/m-tourney-listed" in html

    @pytest.mark.anyio
    async def test_excludes_non_el_farol_tournament_match(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """A Prisoner's Dilemma tournament also writes a GameResult row
        with NULL Phase-7 JSON, but /ui/matches/{id} is an El Farol-only
        dashboard. Listing such a row would only let the user click
        through to a placeholder, which is worse than hiding it.
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
                # game_name = tournament.game_type per the prod writer
                # (TournamentService._write_game_result_for_tournament).
                # game_type = variant default ("tournament").
                game_name="prisoners_dilemma",
                game_type="tournament",
                num_players=2,
                num_rounds=1,
                status="completed",
                completed_at=datetime(2026, 4, 25, 12, 0),
                match_id="m-pd-tourney",
                tournament_id=t.id,
            )
        )
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches")

        assert resp.status_code == 200
        assert "m-pd-tourney" not in resp.text

    @pytest.mark.anyio
    async def test_excludes_rows_that_would_not_render(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """The listing must only show rows whose link lands on a
        rendered dashboard. Non-completed, missing-day-aggregates, and
        legacy rows are all excluded."""
        actions = [{"day": 1, "agent_id": "a1"}]
        aggs = [{"day": 1, "slot_attendance": [0] * 16, "over_slots": 0}]
        db_session.add_all(
            [
                # A. fully renderable → listed
                GameResult(
                    game_name="el_farol",
                    game_type="el_farol_interval",
                    num_players=2,
                    num_rounds=3,
                    status="completed",
                    completed_at=datetime(2026, 4, 20, 12, 0),
                    match_id="m-good",
                    actions_json=actions,
                    day_aggregates_json=aggs,
                ),
                # B. status=running → hidden (would show "still running")
                GameResult(
                    game_name="el_farol",
                    game_type="el_farol_interval",
                    num_players=2,
                    num_rounds=3,
                    status="running",
                    match_id="m-running",
                    actions_json=actions,
                    day_aggregates_json=aggs,
                ),
                # C. status=failed → hidden
                GameResult(
                    game_name="el_farol",
                    game_type="el_farol_interval",
                    num_players=2,
                    num_rounds=3,
                    status="failed",
                    match_id="m-failed",
                    actions_json=actions,
                    day_aggregates_json=aggs,
                ),
                # D. actions_json populated but day_aggregates_json null
                #    → hidden (would show "predates schema")
                GameResult(
                    game_name="el_farol",
                    game_type="el_farol_interval",
                    num_players=2,
                    num_rounds=3,
                    status="completed",
                    completed_at=datetime(2026, 4, 20, 12, 0),
                    match_id="m-no-aggs",
                    actions_json=actions,
                    # day_aggregates_json deliberately null
                ),
            ]
        )
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches")

        assert resp.status_code == 200
        html = resp.text
        assert "m-good" in html
        assert "m-running" not in html
        assert "m-failed" not in html
        assert "m-no-aggs" not in html

    @pytest.mark.anyio
    async def test_orders_newest_first(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        actions = [{"day": 1, "agent_id": "a1"}]
        aggs = [{"day": 1, "slot_attendance": [0] * 16, "over_slots": 0}]
        db_session.add_all(
            [
                GameResult(
                    game_name="el_farol",
                    game_type="el_farol_interval",
                    num_players=2,
                    num_rounds=3,
                    status="completed",
                    completed_at=datetime(2026, 4, 1, 12, 0),
                    match_id="m-older",
                    actions_json=actions,
                    day_aggregates_json=aggs,
                ),
                GameResult(
                    game_name="el_farol",
                    game_type="el_farol_interval",
                    num_players=2,
                    num_rounds=3,
                    status="completed",
                    completed_at=datetime(2026, 4, 22, 12, 0),
                    match_id="m-newer",
                    actions_json=actions,
                    day_aggregates_json=aggs,
                ),
            ]
        )
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches")

        html = resp.text
        # newer appears first
        assert html.index("m-newer") < html.index("m-older")


class TestVisibilityFilter:
    """LABS-TSA PR-5: /ui/matches must filter rows linked to private
    tournaments so non-owners cannot see them. Legacy rows with
    ``tournament_id IS NULL`` (CLI standalone runs) stay visible to
    everyone — that's the non-regression path.
    """

    @pytest.mark.anyio
    async def test_anonymous_does_not_see_private_tournament_match(
        self,
        v2_app,
        db_session: AsyncSession,
    ) -> None:
        """A match linked to a private tournament (``join_token`` set)
        is hidden from anonymous viewers."""
        from atp.dashboard.models import User
        from atp.dashboard.tournament.models import Tournament

        user = User(
            username="creator",
            email="c@test.com",
            hashed_password="x",
            is_active=True,
        )
        db_session.add(user)
        await db_session.commit()

        t = Tournament(
            game_type="el_farol",
            num_players=2,
            total_rounds=1,
            round_deadline_s=30,
            created_by=user.id,
            join_token="secret-private-token",
        )
        db_session.add(t)
        await db_session.commit()

        db_session.add(
            GameResult(
                game_name="El Farol Bar (n=2, days=1)",
                game_type="tournament",
                num_players=2,
                num_rounds=1,
                status="completed",
                completed_at=datetime(2026, 4, 20, 12, 0),
                tournament_id=t.id,
                match_id="uuid-private",
                actions_json=[{"day": 1}],
                day_aggregates_json=[
                    {"day": 1, "slot_attendance": [0] * 16, "over_slots": 0}
                ],
            )
        )
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches")

        assert resp.status_code == 200
        assert "uuid-private" not in resp.text

    @pytest.mark.anyio
    async def test_anonymous_sees_legacy_null_tournament_match(
        self,
        v2_app,
        db_session: AsyncSession,
    ) -> None:
        """Rows with ``tournament_id IS NULL`` (CLI standalone runs) stay
        visible to everyone — non-regression for legacy matches.
        """
        db_session.add(
            GameResult(
                game_name="El Farol Bar (n=6, days=30)",
                game_type="repeated",
                num_players=6,
                num_rounds=30,
                status="completed",
                completed_at=datetime(2026, 4, 20, 12, 0),
                tournament_id=None,
                match_id="uuid-legacy",
                actions_json=[{"day": 1}],
                day_aggregates_json=[
                    {"day": 1, "slot_attendance": [0] * 16, "over_slots": 0}
                ],
            )
        )
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches")

        assert resp.status_code == 200
        assert "uuid-legacy" in resp.text

    @pytest.mark.anyio
    async def test_anonymous_sees_public_tournament_match(
        self,
        v2_app,
        db_session: AsyncSession,
    ) -> None:
        """A match linked to a public tournament (``join_token IS NULL``)
        stays visible to everyone.
        """
        from atp.dashboard.models import User
        from atp.dashboard.tournament.models import Tournament

        user = User(
            username="creator2",
            email="c2@test.com",
            hashed_password="x",
            is_active=True,
        )
        db_session.add(user)
        await db_session.commit()

        t = Tournament(
            game_type="el_farol",
            num_players=2,
            total_rounds=1,
            round_deadline_s=30,
            created_by=user.id,
            join_token=None,  # public
        )
        db_session.add(t)
        await db_session.commit()

        db_session.add(
            GameResult(
                game_name="El Farol Bar (public)",
                game_type="tournament",
                num_players=2,
                num_rounds=1,
                status="completed",
                completed_at=datetime(2026, 4, 20, 12, 0),
                tournament_id=t.id,
                match_id="uuid-public",
                actions_json=[{"day": 1}],
                day_aggregates_json=[
                    {"day": 1, "slot_attendance": [0] * 16, "over_slots": 0}
                ],
            )
        )
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches")

        assert resp.status_code == 200
        assert "uuid-public" in resp.text
