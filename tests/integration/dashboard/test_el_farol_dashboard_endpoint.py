"""Integration tests for the El Farol dashboard fetch endpoint (LABS-99).

Covers ``GET /api/v1/games/{match_id}/dashboard`` — the adapter that
reshapes a ``GameResult`` row (PR #63 Phase 7 columns) into the
``window.ATP`` payload the dashboard JS consumes.
"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.models import GameResult
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app
from atp.dashboard.v2.routes.el_farol_dashboard import SHAPE_VERSION

# ---------- fixtures ----------


@pytest.fixture
def v2_app(test_database: Database):
    """Create a test app with v2 routes and the in-memory DB wired in."""
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


def _seed_payload(match_id: str = "m-01", include_tier2: bool = True) -> dict:
    """Build the GameResult column kwargs for one 2-agent 3-day El Farol."""
    agents = [
        {
            "agent_id": "gpt-5",
            "display_name": "gpt-5",
            "user_id": "alice",
            "color": "#1f6feb",
            "family": "calibrated",
            "adapter_type": "MCP",
        },
        {
            "agent_id": "claude-4",
            "display_name": "claude-4",
            "user_id": "bob",
            "color": "#8957e5",
            "family": "conservative",
            "adapter_type": "MCP",
        },
    ]

    def action(day: int, aid: str, payoff: float, **tier2) -> dict:
        base = {
            "match_id": match_id,
            "day": day,
            "agent_id": aid,
            "intervals": {"first": [0, 2], "second": []},
            "picks": [0, 1, 2],
            "num_visits": 1,
            "total_slots": 3,
            "payoff": payoff,
            "num_over": 0,
            "num_under": 3,
            "intent": f"{aid} day {day}",
            "retry_count": 0,
            "validation_error": None,
            "submitted_at": None,
        }
        if include_tier2:
            base.update(
                tokens_in=1200,
                tokens_out=120,
                decide_ms=450,
                cost_usd=0.0042,
                model_id=f"{aid}-latest",
                trace_id=f"trace-{aid}-{day}",
                span_id=f"span-{aid}-{day}",
            )
        base.update(tier2)
        return base

    actions = []
    for d in (1, 2, 3):
        actions.append(action(d, "gpt-5", 3.0))
        actions.append(action(d, "claude-4", -1.0))

    day_aggregates = [
        {
            "match_id": match_id,
            "day": d,
            "slot_attendance": [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "over_slots": 0,
            "total_attendances": 6,
        }
        for d in (1, 2, 3)
    ]

    round_payoffs = [{"gpt-5": 3.0, "claude-4": -1.0} for _ in (1, 2, 3)]

    return {
        "game_name": "el_farol",
        "game_type": "el_farol_interval",
        "num_players": 2,
        "num_rounds": 3,
        "num_episodes": 1,
        "status": "completed",
        "match_id": match_id,
        "game_version": "1.0.0",
        "num_days": 3,
        "num_slots": 16,
        "max_intervals": 2,
        "max_total_slots": 8,
        "capacity_ratio": 0.6,
        "capacity_threshold": 1,
        "actions_json": actions,
        "day_aggregates_json": day_aggregates,
        "round_payoffs_json": round_payoffs,
        "agents_json": agents,
    }


# ---------- tests ----------


class TestElFarolDashboardEndpoint:
    """GET /api/v1/games/{match_id}/dashboard."""

    @pytest.mark.anyio
    async def test_happy_path_returns_full_payload(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """GIVEN a GameResult row with Phase 7 columns populated
        WHEN we GET the dashboard endpoint
        THEN the payload matches the window.ATP shape.
        """
        row = GameResult(**_seed_payload(match_id="m-happy"))
        db_session.add(row)
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/games/m-happy/dashboard")

        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["shape_version"] == SHAPE_VERSION
        assert data["match_id"] == "m-happy"
        assert data["NUM_SLOTS"] == 16
        assert data["MAX_TOTAL"] == 8
        assert data["MAX_INTERVALS"] == 2
        assert data["CAPACITY"] == 1
        assert data["NUM_DAYS"] == 3
        assert len(data["AGENTS"]) == 2
        assert data["AGENTS"][0] == {
            "id": "gpt-5",
            "color": "#1f6feb",
            "user": "alice",
            "profile": "calibrated",
        }
        assert len(data["DATA"]) == 3
        day1 = data["DATA"][0]
        assert day1["round"] == 1
        assert day1["slotAttendance"][:3] == [2, 2, 2]
        assert day1["overSlots"] == 0
        assert len(day1["decisions"]) == 2
        d0 = day1["decisions"][0]
        assert d0["agent"] == "gpt-5"
        assert d0["intervals"] == [[0, 2], []]
        assert d0["picks"] == [0, 1, 2]
        assert d0["numVisits"] == 1
        assert d0["intent"] == "gpt-5 day 1"
        # Tier-2
        assert d0["model_id"] == "gpt-5-latest"
        assert d0["trace_id"] == "trace-gpt-5-1"
        assert d0["tokens_in"] == 1200

    @pytest.mark.anyio
    async def test_legacy_row_returns_404(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """GIVEN a GameResult row without actions_json (pre-PR #63)
        WHEN we GET the dashboard endpoint
        THEN we get 404 with a schema-version message.
        """
        row = GameResult(
            game_name="el_farol",
            game_type="el_farol",
            num_players=2,
            num_rounds=3,
            status="completed",
            match_id="m-legacy",
            # actions_json / day_aggregates_json deliberately left null
        )
        db_session.add(row)
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/games/m-legacy/dashboard")

        assert resp.status_code == 404
        assert "predates" in resp.json()["detail"].lower()

    @pytest.mark.anyio
    async def test_missing_tier2_fields_are_null(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """GIVEN a row where ActionRecords omit OTel Tier-2 fields
        WHEN we GET the dashboard endpoint
        THEN those fields serialise as null (renderer treats as "—").
        """
        payload = _seed_payload(match_id="m-notier2", include_tier2=False)
        row = GameResult(**payload)
        db_session.add(row)
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/games/m-notier2/dashboard")

        assert resp.status_code == 200
        d = resp.json()["DATA"][0]["decisions"][0]
        assert d["tokens_in"] is None
        assert d["tokens_out"] is None
        assert d["decide_ms"] is None
        assert d["cost_usd"] is None
        assert d["model_id"] is None
        assert d["trace_id"] is None

    @pytest.mark.anyio
    async def test_unknown_match_returns_404(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """Unknown match_id 404s with a 'not found' message (not the
        schema-version one)."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/games/nope/dashboard")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.anyio
    async def test_numeric_match_id_falls_back_to_primary_key(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """A caller passing the numeric ``id`` (not match_id) still resolves."""
        payload = _seed_payload(match_id="m-with-pk")
        row = GameResult(**payload)
        db_session.add(row)
        await db_session.commit()
        await db_session.refresh(row)

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get(f"/api/v1/games/{row.id}/dashboard")

        assert resp.status_code == 200
        assert resp.json()["match_id"] == "m-with-pk"
