"""Integration tests for the game evaluation dashboard routes (TASK-920).

Tests verify the full request/response cycle with an in-memory database,
covering game results, tournament results, cross-play, and export endpoints.
"""

from collections.abc import AsyncGenerator
from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import (
    Base,
    GameResult,
    TournamentResult,
    User,
)
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
async def test_database():
    """Create and configure a test database."""
    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)
    yield db
    await db.close()
    set_database(None)  # type: ignore


@pytest.fixture
async def async_session(
    test_database: Database,
) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for testing."""
    async with test_database.session() as session:
        yield session


@pytest.fixture
def v2_app(test_database: Database):
    """Create a test app with v2 routes."""
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


@pytest.fixture
async def admin_user(async_session: AsyncSession) -> User:
    """Create an admin user for testing."""
    user = User(
        username="admin_test",
        email="admin@test.com",
        hashed_password=get_password_hash("password123"),
        is_admin=True,
        is_active=True,
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user: User) -> str:
    """Generate JWT token for admin user."""
    return create_access_token(
        data={"sub": admin_user.username, "user_id": admin_user.id}
    )


@pytest.fixture
def admin_headers(admin_token: str) -> dict[str, str]:
    """Return authorization headers for admin."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
async def sample_game_result(
    async_session: AsyncSession,
) -> GameResult:
    """Create a sample game result in the database."""
    game = GameResult(
        game_name="Prisoner's Dilemma",
        game_type="normal_form",
        num_players=2,
        num_rounds=10,
        num_episodes=5,
        status="completed",
        created_at=datetime.now(),
        completed_at=datetime.now(),
        players_json=[
            {
                "player_id": "player_1",
                "strategy": "tit_for_tat",
                "average_payoff": 3.0,
                "total_payoff": 15.0,
                "cooperation_rate": 0.8,
            },
            {
                "player_id": "player_2",
                "strategy": "always_cooperate",
                "average_payoff": 2.5,
                "total_payoff": 12.5,
                "cooperation_rate": 1.0,
            },
        ],
        payoff_matrix_json={
            "player_1": {"player_1": 0.0, "player_2": 3.0},
            "player_2": {"player_1": 2.5, "player_2": 0.0},
        },
        strategy_timeline_json=[
            {"round": 1, "cooperate": 0.8, "defect": 0.2},
        ],
        cooperation_dynamics_json=[
            {"round": 1, "cooperation_rate": 0.8},
        ],
        episodes_json=[
            {
                "episode": 0,
                "payoffs": {"player_1": 3.0, "player_2": 3.0},
            },
            {
                "episode": 1,
                "payoffs": {"player_1": 3.5, "player_2": 2.0},
            },
        ],
        metadata_json={"seed": 42},
    )
    async_session.add(game)
    await async_session.commit()
    await async_session.refresh(game)
    return game


@pytest.fixture
async def sample_tournament_result(
    async_session: AsyncSession,
) -> TournamentResult:
    """Create a sample tournament result in the database."""
    tournament = TournamentResult(
        name="PD Tournament",
        game_name="Prisoner's Dilemma",
        tournament_type="round_robin",
        num_agents=4,
        episodes_per_matchup=50,
        status="completed",
        created_at=datetime.now(),
        completed_at=datetime.now(),
        standings_json=[
            {
                "rank": 1,
                "agent": "tit_for_tat",
                "wins": 3,
                "losses": 0,
                "draws": 0,
                "total_payoff": 150.0,
                "average_payoff": 3.0,
            },
            {
                "rank": 2,
                "agent": "always_defect",
                "wins": 2,
                "losses": 1,
                "draws": 0,
                "total_payoff": 120.0,
                "average_payoff": 2.4,
            },
        ],
        matchups_json=[
            {
                "player_1": "tit_for_tat",
                "player_2": "always_defect",
                "player_1_avg_payoff": 2.5,
                "player_2_avg_payoff": 2.0,
                "episodes": 50,
                "winner": "tit_for_tat",
            },
        ],
        cross_play_matrix_json={
            "tit_for_tat": {
                "tit_for_tat": 3.0,
                "always_defect": 2.5,
            },
            "always_defect": {
                "tit_for_tat": 2.0,
                "always_defect": 1.0,
            },
        },
    )
    async_session.add(tournament)
    await async_session.commit()
    await async_session.refresh(tournament)
    return tournament


class TestListGameResults:
    """Tests for GET /games."""

    @pytest.mark.anyio
    async def test_list_empty(self, v2_app, admin_user, admin_headers) -> None:
        """Test listing with no game results."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/games", headers=admin_headers)
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 0
            assert data["items"] == []

    @pytest.mark.anyio
    async def test_list_with_results(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_game_result,
    ) -> None:
        """Test listing with game results."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/games", headers=admin_headers)
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert len(data["items"]) == 1
            assert data["items"][0]["game_name"] == "Prisoner's Dilemma"

    @pytest.mark.anyio
    async def test_list_filter_by_game_name(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_game_result,
    ) -> None:
        """Test filtering by game name."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                "/api/games?game_name=Nonexistent",
                headers=admin_headers,
            )
            assert response.status_code == 200
            assert response.json()["total"] == 0

    @pytest.mark.anyio
    async def test_list_filter_by_game_type(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_game_result,
    ) -> None:
        """Test filtering by game type."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                "/api/games?game_type=normal_form",
                headers=admin_headers,
            )
            assert response.status_code == 200
            assert response.json()["total"] == 1

    @pytest.mark.anyio
    async def test_list_returns_200(self, v2_app, admin_user, admin_headers) -> None:
        """Test that list endpoint returns 200."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/games", headers=admin_headers)
            assert response.status_code == 200


class TestGetGameResult:
    """Tests for GET /games/{game_id}."""

    @pytest.mark.anyio
    async def test_get_existing(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_game_result,
    ) -> None:
        """Test getting an existing game result."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/api/games/{sample_game_result.id}",
                headers=admin_headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["game_name"] == "Prisoner's Dilemma"
            assert data["num_players"] == 2
            assert len(data["players"]) == 2
            assert data["players"][0]["player_id"] == "player_1"
            assert data["payoff_matrix"] is not None
            assert data["metadata"]["seed"] == 42

    @pytest.mark.anyio
    async def test_get_not_found(self, v2_app, admin_user, admin_headers) -> None:
        """Test getting a non-existent game result."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/games/999", headers=admin_headers)
            assert response.status_code == 404


class TestExportGameCSV:
    """Tests for GET /games/{game_id}/export/csv."""

    @pytest.mark.anyio
    async def test_export_csv(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_game_result,
    ) -> None:
        """Test CSV export returns valid CSV."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/api/games/{sample_game_result.id}/export/csv",
                headers=admin_headers,
            )
            assert response.status_code == 200
            assert "text/csv" in response.headers["content-type"]
            assert "attachment" in response.headers.get("content-disposition", "")
            content = response.text
            lines = content.strip().split("\n")
            assert len(lines) >= 2  # header + at least 1 data row

    @pytest.mark.anyio
    async def test_export_csv_not_found(
        self, v2_app, admin_user, admin_headers
    ) -> None:
        """Test CSV export for non-existent game."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                "/api/games/999/export/csv",
                headers=admin_headers,
            )
            assert response.status_code == 404


class TestExportGameJSON:
    """Tests for GET /games/{game_id}/export/json."""

    @pytest.mark.anyio
    async def test_export_json(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_game_result,
    ) -> None:
        """Test JSON export returns valid JSON."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/api/games/{sample_game_result.id}/export/json",
                headers=admin_headers,
            )
            assert response.status_code == 200
            assert "application/json" in response.headers["content-type"]
            data = response.json()
            assert data["game_name"] == "Prisoner's Dilemma"
            assert data["num_players"] == 2

    @pytest.mark.anyio
    async def test_export_json_not_found(
        self, v2_app, admin_user, admin_headers
    ) -> None:
        """Test JSON export for non-existent game."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                "/api/games/999/export/json",
                headers=admin_headers,
            )
            assert response.status_code == 404


class TestListTournamentResults:
    """Tests for GET /tournaments."""

    @pytest.mark.anyio
    async def test_list_empty(self, v2_app, admin_user, admin_headers) -> None:
        """Test listing with no tournament results."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/tournaments", headers=admin_headers)
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 0

    @pytest.mark.anyio
    async def test_list_with_results(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_tournament_result,
    ) -> None:
        """Test listing with tournament results."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/tournaments", headers=admin_headers)
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert data["items"][0]["name"] == "PD Tournament"

    @pytest.mark.anyio
    async def test_list_returns_200(self, v2_app, admin_user, admin_headers) -> None:
        """Test that tournaments endpoint returns 200."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/tournaments", headers=admin_headers)
            assert response.status_code == 200


class TestGetTournamentResult:
    """Tests for GET /tournaments/{tournament_id}."""

    @pytest.mark.anyio
    async def test_get_existing(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_tournament_result,
    ) -> None:
        """Test getting an existing tournament result."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/api/tournaments/{sample_tournament_result.id}",
                headers=admin_headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "PD Tournament"
            assert data["num_agents"] == 4
            assert len(data["standings"]) == 2
            assert data["standings"][0]["agent"] == "tit_for_tat"
            assert len(data["matchups"]) == 1

    @pytest.mark.anyio
    async def test_get_not_found(self, v2_app, admin_user, admin_headers) -> None:
        """Test getting a non-existent tournament."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/tournaments/999", headers=admin_headers)
            assert response.status_code == 404


class TestGetCrossplayMatrix:
    """Tests for GET /crossplay/{tournament_id}."""

    @pytest.mark.anyio
    async def test_get_crossplay(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_tournament_result,
    ) -> None:
        """Test getting cross-play matrix."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/api/crossplay/{sample_tournament_result.id}",
                headers=admin_headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["tournament_name"] == "PD Tournament"
            assert "cross_play_matrix" in data
            assert "tit_for_tat" in data["cross_play_matrix"]

    @pytest.mark.anyio
    async def test_crossplay_not_found(self, v2_app, admin_user, admin_headers) -> None:
        """Test cross-play for non-existent tournament."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/crossplay/999", headers=admin_headers)
            assert response.status_code == 404


class TestRoutesReturn200:
    """Verify all game routes return 200 for valid requests."""

    @pytest.mark.anyio
    async def test_games_returns_200(self, v2_app, admin_user, admin_headers) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/games", headers=admin_headers)
            assert response.status_code == 200

    @pytest.mark.anyio
    async def test_tournaments_returns_200(
        self, v2_app, admin_user, admin_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/tournaments", headers=admin_headers)
            assert response.status_code == 200

    @pytest.mark.anyio
    async def test_game_detail_returns_200(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_game_result,
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/api/games/{sample_game_result.id}",
                headers=admin_headers,
            )
            assert response.status_code == 200

    @pytest.mark.anyio
    async def test_tournament_detail_returns_200(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_tournament_result,
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/api/tournaments/{sample_tournament_result.id}",
                headers=admin_headers,
            )
            assert response.status_code == 200

    @pytest.mark.anyio
    async def test_csv_export_returns_200(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_game_result,
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/api/games/{sample_game_result.id}/export/csv",
                headers=admin_headers,
            )
            assert response.status_code == 200

    @pytest.mark.anyio
    async def test_json_export_returns_200(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_game_result,
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/api/games/{sample_game_result.id}/export/json",
                headers=admin_headers,
            )
            assert response.status_code == 200

    @pytest.mark.anyio
    async def test_crossplay_returns_200(
        self,
        v2_app,
        admin_user,
        admin_headers,
        sample_tournament_result,
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/api/crossplay/{sample_tournament_result.id}",
                headers=admin_headers,
            )
            assert response.status_code == 200
