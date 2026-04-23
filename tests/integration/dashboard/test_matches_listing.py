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
                GameResult(
                    game_name="el_farol",
                    game_type="el_farol_interval",
                    num_players=2,
                    num_rounds=3,
                    status="completed",
                    completed_at=datetime(2026, 4, 20, 12, 0),
                    match_id="m-listed",
                    num_days=3,
                    num_slots=16,
                    actions_json=actions,
                    day_aggregates_json=aggs,
                ),
                GameResult(
                    game_name="el_farol",
                    game_type="el_farol",
                    num_players=2,
                    num_rounds=3,
                    status="completed",
                    match_id="m-legacy",
                    # actions_json deliberately null
                ),
                GameResult(
                    game_name="prisoners_dilemma",
                    game_type="repeated",
                    num_players=2,
                    num_rounds=5,
                    status="completed",
                    match_id="m-pd",
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

        assert resp.status_code == 200
        html = resp.text
        assert "m-listed" in html
        assert "/ui/matches/m-listed" in html
        # legacy and non-el-farol rows hidden
        assert "m-legacy" not in html
        assert "m-pd" not in html
        assert "1 completed El Farol matches" in html or ">1 completed" in html

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
