"""Integration tests for the cost and budget dashboard routes.

These tests verify the cost analytics and budget management API endpoints.
"""

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from atp.analytics.database import AnalyticsDatabase
from atp.analytics.models import AnalyticsBase, CostBudget, CostRecord
from atp.dashboard.database import Database
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
async def analytics_engine():
    """Create an async SQLite in-memory engine for analytics models."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(AnalyticsBase.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def analytics_session(
    analytics_engine,
) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for analytics testing."""
    async_session_factory = sessionmaker(
        analytics_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def v2_app(test_database: Database, analytics_engine, monkeypatch):
    """Create a test app with v2 routes and analytics DB override."""
    app = create_test_app(use_v2_routes=True)

    # Override the dependency to use our test session
    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session

    # Monkey-patch the AnalyticsDatabase to use our test engine
    original_init = AnalyticsDatabase.__init__

    def mock_init(self, database_url: str | None = None):
        original_init(self, database_url)
        self._engine = analytics_engine

    monkeypatch.setattr(AnalyticsDatabase, "__init__", mock_init)

    return app


@pytest.fixture
async def cost_data(analytics_engine) -> dict:
    """Create test cost data."""
    async_session_factory = sessionmaker(
        analytics_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        now = datetime.now()

        # Create cost records
        records = [
            CostRecord(
                timestamp=now - timedelta(days=2),
                provider="openai",
                model="gpt-4",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=Decimal("0.05"),
                agent_name="agent-one",
                suite_id="suite-001",
                test_id="test-001",
            ),
            CostRecord(
                timestamp=now - timedelta(days=1),
                provider="openai",
                model="gpt-4",
                input_tokens=2000,
                output_tokens=800,
                cost_usd=Decimal("0.09"),
                agent_name="agent-one",
                suite_id="suite-001",
                test_id="test-002",
            ),
            CostRecord(
                timestamp=now - timedelta(hours=12),
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=1500,
                output_tokens=600,
                cost_usd=Decimal("0.04"),
                agent_name="agent-two",
                suite_id="suite-002",
                test_id="test-001",
            ),
            CostRecord(
                timestamp=now - timedelta(hours=6),
                provider="anthropic",
                model="claude-3-opus",
                input_tokens=500,
                output_tokens=200,
                cost_usd=Decimal("0.10"),
                agent_name="agent-two",
                suite_id="suite-002",
                test_id="test-003",
            ),
        ]
        session.add_all(records)
        await session.flush()

        # Create budgets
        budgets = [
            CostBudget(
                name="daily-budget",
                period="daily",
                limit_usd=Decimal("10.00"),
                alert_threshold=0.8,
                is_active=True,
                description="Daily cost limit",
            ),
            CostBudget(
                name="monthly-budget",
                period="monthly",
                limit_usd=Decimal("500.00"),
                alert_threshold=0.9,
                scope_json={"provider": "openai"},
                alert_channels_json=["slack", "email"],
                is_active=True,
            ),
            CostBudget(
                name="inactive-budget",
                period="weekly",
                limit_usd=Decimal("100.00"),
                is_active=False,
            ),
        ]
        session.add_all(budgets)
        await session.commit()

        return {
            "records": records,
            "budgets": budgets,
        }


class TestCostRoutes:
    """Test cost analytics routes."""

    @pytest.mark.anyio
    async def test_get_cost_summary(self, v2_app, cost_data, auth_headers):
        """Test GET /api/costs returns cost summary."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/costs", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()

            assert "total_cost" in data
            assert "total_input_tokens" in data
            assert "total_output_tokens" in data
            assert "total_records" in data
            assert "by_provider" in data
            assert "by_model" in data
            assert "by_agent" in data
            assert "daily_trend" in data

            # Verify totals
            assert data["total_cost"] > 0
            assert data["total_records"] == 4

    @pytest.mark.anyio
    async def test_get_cost_summary_with_filters(self, v2_app, cost_data, auth_headers):
        """Test GET /api/costs with provider filter."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/costs", params={"provider": "openai"}, headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()

            # With filter, total should reflect filtered records
            assert data["total_cost"] > 0

    @pytest.mark.anyio
    async def test_list_cost_records(self, v2_app, cost_data, auth_headers):
        """Test GET /api/costs/records returns paginated records."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/costs/records", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()

            assert "total" in data
            assert "items" in data
            assert "limit" in data
            assert "offset" in data
            assert len(data["items"]) == 4

            # Verify record structure
            record = data["items"][0]
            assert "id" in record
            assert "timestamp" in record
            assert "provider" in record
            assert "model" in record
            assert "input_tokens" in record
            assert "output_tokens" in record
            assert "cost_usd" in record

    @pytest.mark.anyio
    async def test_list_cost_records_with_filters(
        self, v2_app, cost_data, auth_headers
    ):
        """Test GET /api/costs/records with filters."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/costs/records",
                params={"provider": "anthropic", "limit": 10, "offset": 0},
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()

            assert len(data["items"]) == 2
            for record in data["items"]:
                assert record["provider"] == "anthropic"

    @pytest.mark.anyio
    async def test_get_costs_by_provider(self, v2_app, cost_data, auth_headers):
        """Test GET /api/costs/by-provider returns breakdown."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/costs/by-provider", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()

            assert isinstance(data, list)
            assert len(data) >= 2

            # Check item structure
            for item in data:
                assert "name" in item
                assert "total_cost" in item
                assert "total_input_tokens" in item
                assert "total_output_tokens" in item
                assert "record_count" in item
                assert "percentage" in item

    @pytest.mark.anyio
    async def test_get_costs_by_model(self, v2_app, cost_data, auth_headers):
        """Test GET /api/costs/by-model returns breakdown."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/costs/by-model", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()

            assert isinstance(data, list)
            assert len(data) >= 3  # gpt-4, claude-3-sonnet, claude-3-opus

    @pytest.mark.anyio
    async def test_get_costs_by_agent(self, v2_app, cost_data, auth_headers):
        """Test GET /api/costs/by-agent returns breakdown."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/costs/by-agent", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()

            assert isinstance(data, list)
            assert len(data) >= 2  # agent-one, agent-two

    @pytest.mark.anyio
    async def test_get_cost_trend(self, v2_app, cost_data, auth_headers):
        """Test GET /api/costs/trend returns daily trend."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/costs/trend", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()

            assert isinstance(data, list)
            # Should have at least 2 days of data
            assert len(data) >= 2

            # Check item structure
            for item in data:
                assert "date" in item
                assert "total_cost" in item
                assert "total_tokens" in item
                assert "record_count" in item


class TestBudgetRoutes:
    """Test budget management routes."""

    @pytest.mark.anyio
    async def test_list_budgets(self, v2_app, cost_data, auth_headers):
        """Test GET /api/budgets returns list of budgets."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/budgets", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()

            assert "items" in data
            assert "total" in data
            assert data["total"] == 3

            # Check budget structure
            budget = data["items"][0]
            assert "id" in budget
            assert "name" in budget
            assert "period" in budget
            assert "limit_usd" in budget
            assert "alert_threshold" in budget
            assert "is_active" in budget
            assert "usage" in budget

    @pytest.mark.anyio
    async def test_list_budgets_filter_active(self, v2_app, cost_data, auth_headers):
        """Test GET /api/budgets with active filter."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/budgets", params={"is_active": True}, headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()

            assert data["total"] == 2
            for budget in data["items"]:
                assert budget["is_active"] is True

    @pytest.mark.anyio
    async def test_list_budgets_filter_period(self, v2_app, cost_data, auth_headers):
        """Test GET /api/budgets with period filter."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/budgets", params={"period": "daily"}, headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()

            assert data["total"] == 1
            assert data["items"][0]["period"] == "daily"

    @pytest.mark.anyio
    async def test_get_budget(self, v2_app, cost_data, auth_headers):
        """Test GET /api/budgets/{id} returns budget details."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # Get list first to find an ID
            list_response = await client.get("/api/budgets", headers=auth_headers)
            budget_id = list_response.json()["items"][0]["id"]

            response = await client.get(
                f"/api/budgets/{budget_id}", headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()

            assert data["id"] == budget_id
            assert "usage" in data

    @pytest.mark.anyio
    async def test_get_budget_not_found(self, v2_app, cost_data, auth_headers):
        """Test GET /api/budgets/{id} returns 404 for missing budget."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/budgets/99999", headers=auth_headers)
            assert response.status_code == 404

    @pytest.mark.anyio
    async def test_get_budget_usage(self, v2_app, cost_data, auth_headers):
        """Test GET /api/budgets/{id}/usage returns usage data."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # Get list first to find an ID
            list_response = await client.get("/api/budgets", headers=auth_headers)
            budget_id = list_response.json()["items"][0]["id"]

            response = await client.get(
                f"/api/budgets/{budget_id}/usage", headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()

            assert "budget_id" in data
            assert "budget_name" in data
            assert "period" in data
            assert "spent" in data
            assert "limit" in data
            assert "remaining" in data
            assert "percentage" in data
            assert "is_over_threshold" in data
            assert "is_over_limit" in data

    @pytest.mark.anyio
    async def test_check_all_budgets(self, v2_app, cost_data, auth_headers):
        """Test GET /api/budgets/status/all returns all budget usages."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/budgets/status/all", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()

            assert isinstance(data, list)
            # Should return only active budgets
            assert len(data) == 2

    @pytest.mark.anyio
    async def test_create_budget_unauthorized(self, v2_app, cost_data):
        """Test POST /api/budgets requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/budgets",
                json={
                    "name": "new-budget",
                    "period": "daily",
                    "limit_usd": 50.0,
                },
            )
            assert response.status_code == 401

    @pytest.mark.anyio
    async def test_update_budget_unauthorized(self, v2_app, cost_data):
        """Test PUT /api/budgets/{id} requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.put(
                "/api/budgets/1",
                json={"limit_usd": 100.0},
            )
            assert response.status_code == 401

    @pytest.mark.anyio
    async def test_delete_budget_unauthorized(self, v2_app, cost_data):
        """Test DELETE /api/budgets/{id} requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.delete("/api/budgets/1")
            assert response.status_code == 401


class TestCostRouteConsistency:
    """Test that cost routes follow expected patterns."""

    @pytest.mark.anyio
    async def test_cost_routes_registered(self, v2_app, cost_data):
        """Verify cost routes are registered."""
        routes = [route.path for route in v2_app.routes]

        expected_routes = [
            "/api/costs",
            "/api/costs/records",
            "/api/costs/by-provider",
            "/api/costs/by-model",
            "/api/costs/by-agent",
            "/api/costs/trend",
            "/api/budgets",
            "/api/budgets/{budget_id}",
            "/api/budgets/{budget_id}/usage",
            "/api/budgets/status/all",
        ]

        for expected in expected_routes:
            matching = [r for r in routes if expected in r]
            assert len(matching) > 0, f"Route {expected} not found"
