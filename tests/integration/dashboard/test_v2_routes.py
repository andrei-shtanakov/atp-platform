"""Integration tests for the v2 dashboard routes.

These tests verify that the modular v2 routes provide the same
functionality as the v1 monolithic api.py router.
"""

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from atp.dashboard.models import Agent, Base, RunResult, SuiteExecution, TestExecution
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
async def async_engine():
    """Create an async SQLite in-memory engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for testing."""
    async_session_factory = sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def v2_app(async_engine, async_session):
    """Create a test app with v2 routes."""
    app = create_test_app(use_v2_routes=True)

    # Override the dependency to use our test session
    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async_session_factory = sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session
    return app


@pytest.fixture
async def test_data(async_session: AsyncSession) -> dict:
    """Create test data for v2 route tests."""
    # Create agents
    agent1 = Agent(
        name="agent-one",
        agent_type="http",
        config={"endpoint": "http://localhost:8001"},
        description="First test agent",
    )
    agent2 = Agent(
        name="agent-two",
        agent_type="docker",
        config={"image": "test:latest"},
        description="Second test agent",
    )
    async_session.add_all([agent1, agent2])
    await async_session.flush()

    # Create suite executions
    now = datetime.now()
    suite1 = SuiteExecution(
        suite_name="test-suite",
        agent_id=agent1.id,
        started_at=now - timedelta(hours=2),
        completed_at=now - timedelta(hours=1),
        duration_seconds=3600.0,
        runs_per_test=1,
        total_tests=3,
        passed_tests=2,
        failed_tests=1,
        success_rate=0.67,
        status="completed",
    )
    suite2 = SuiteExecution(
        suite_name="test-suite",
        agent_id=agent2.id,
        started_at=now - timedelta(hours=1),
        completed_at=now,
        duration_seconds=3000.0,
        runs_per_test=1,
        total_tests=3,
        passed_tests=3,
        failed_tests=0,
        success_rate=1.0,
        status="completed",
    )
    async_session.add_all([suite1, suite2])
    await async_session.flush()

    # Create test executions
    test1 = TestExecution(
        suite_execution_id=suite1.id,
        test_id="test-001",
        test_name="First Test",
        started_at=now - timedelta(hours=2),
        completed_at=now - timedelta(hours=1, minutes=50),
        duration_seconds=600.0,
        success=True,
        score=85.0,
        status="completed",
    )
    test2 = TestExecution(
        suite_execution_id=suite2.id,
        test_id="test-001",
        test_name="First Test",
        started_at=now - timedelta(hours=1),
        completed_at=now - timedelta(minutes=50),
        duration_seconds=500.0,
        success=True,
        score=95.0,
        status="completed",
    )
    async_session.add_all([test1, test2])
    await async_session.flush()

    # Create run results with events
    events = [
        {
            "sequence": 1,
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "event_type": "tool_call",
            "payload": {"tool": "read_file", "status": "success"},
        },
        {
            "sequence": 2,
            "timestamp": (now - timedelta(hours=1, minutes=55)).isoformat(),
            "event_type": "llm_request",
            "payload": {"model": "gpt-4", "input_tokens": 100, "output_tokens": 50},
        },
    ]
    run1 = RunResult(
        test_execution_id=test1.id,
        run_number=1,
        started_at=now - timedelta(hours=2),
        completed_at=now - timedelta(hours=1, minutes=50),
        duration_seconds=600.0,
        response_status="completed",
        success=True,
        total_tokens=150,
        total_steps=2,
        tool_calls=1,
        llm_calls=1,
        cost_usd=0.01,
        events_json=events,
    )
    async_session.add(run1)
    await async_session.commit()

    return {
        "agents": [agent1, agent2],
        "suites": [suite1, suite2],
        "tests": [test1, test2],
        "runs": [run1],
    }


class TestV2HomeRoutes:
    """Test v2 home/dashboard routes."""

    @pytest.mark.anyio
    async def test_dashboard_summary(self, v2_app, test_data):
        """Test GET /api/dashboard/summary returns summary stats."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/dashboard/summary")
            assert response.status_code == 200
            data = response.json()
            assert "total_agents" in data
            assert "total_suites" in data
            assert "total_executions" in data
            assert "recent_success_rate" in data


class TestV2AgentRoutes:
    """Test v2 agent management routes."""

    @pytest.mark.anyio
    async def test_list_agents(self, v2_app, test_data):
        """Test GET /api/agents returns list of agents."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/agents")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) >= 2
            agent_names = [a["name"] for a in data]
            assert "agent-one" in agent_names
            assert "agent-two" in agent_names

    @pytest.mark.anyio
    async def test_get_agent(self, v2_app, test_data):
        """Test GET /api/agents/{id} returns agent details."""
        agent_id = test_data["agents"][0].id
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(f"/api/agents/{agent_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "agent-one"
            assert data["agent_type"] == "http"

    @pytest.mark.anyio
    async def test_get_agent_not_found(self, v2_app, test_data):
        """Test GET /api/agents/{id} returns 404 for missing agent."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/agents/99999")
            assert response.status_code == 404


class TestV2SuiteRoutes:
    """Test v2 suite execution routes."""

    @pytest.mark.anyio
    async def test_list_suite_executions(self, v2_app, test_data):
        """Test GET /api/suites returns list of suite executions."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/suites")
            assert response.status_code == 200
            data = response.json()
            assert "total" in data
            assert "items" in data
            assert data["total"] >= 2

    @pytest.mark.anyio
    async def test_list_suite_names(self, v2_app, test_data):
        """Test GET /api/suites/names/list returns unique suite names."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/suites/names/list")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert "test-suite" in data

    @pytest.mark.anyio
    async def test_get_suite_execution(self, v2_app, test_data):
        """Test GET /api/suites/{id} returns suite details."""
        suite_id = test_data["suites"][0].id
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(f"/api/suites/{suite_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["suite_name"] == "test-suite"
            assert "tests" in data


class TestV2TestRoutes:
    """Test v2 test execution routes."""

    @pytest.mark.anyio
    async def test_list_test_executions(self, v2_app, test_data):
        """Test GET /api/tests returns list of test executions."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/tests")
            assert response.status_code == 200
            data = response.json()
            assert "total" in data
            assert "items" in data
            assert data["total"] >= 2

    @pytest.mark.anyio
    async def test_get_test_execution(self, v2_app, test_data):
        """Test GET /api/tests/{id} returns test details."""
        test_id = test_data["tests"][0].id
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(f"/api/tests/{test_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["test_id"] == "test-001"
            assert "runs" in data
            assert "evaluations" in data


class TestV2TrendRoutes:
    """Test v2 trend analysis routes."""

    @pytest.mark.anyio
    async def test_get_suite_trends(self, v2_app, test_data):
        """Test GET /api/trends/suite returns trend data."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/trends/suite", params={"suite_name": "test-suite"}
            )
            assert response.status_code == 200
            data = response.json()
            assert "suite_trends" in data

    @pytest.mark.anyio
    async def test_get_test_trends(self, v2_app, test_data):
        """Test GET /api/trends/test returns trend data."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/trends/test",
                params={"suite_name": "test-suite", "test_id": "test-001"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "test_trends" in data


class TestV2ComparisonRoutes:
    """Test v2 comparison routes."""

    @pytest.mark.anyio
    async def test_compare_agents(self, v2_app, test_data):
        """Test GET /api/compare/agents returns comparison data."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/compare/agents",
                params={
                    "suite_name": "test-suite",
                    "agents": ["agent-one", "agent-two"],
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["suite_name"] == "test-suite"
            assert "agents" in data
            assert "tests" in data

    @pytest.mark.anyio
    async def test_side_by_side_comparison(self, v2_app, test_data):
        """Test GET /api/compare/side-by-side returns detailed comparison."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/compare/side-by-side",
                params={
                    "suite_name": "test-suite",
                    "test_id": "test-001",
                    "agents": ["agent-one", "agent-two"],
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["suite_name"] == "test-suite"
            assert data["test_id"] == "test-001"
            assert "agents" in data


class TestV2TimelineRoutes:
    """Test v2 timeline routes."""

    @pytest.mark.anyio
    async def test_get_timeline_events(self, v2_app, test_data):
        """Test GET /api/timeline/events returns event timeline."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/timeline/events",
                params={
                    "suite_name": "test-suite",
                    "test_id": "test-001",
                    "agent_name": "agent-one",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["suite_name"] == "test-suite"
            assert "events" in data
            assert "total_events" in data


class TestV2TemplateRoutes:
    """Test v2 template routes."""

    @pytest.mark.anyio
    async def test_list_templates(self, v2_app, test_data):
        """Test GET /api/templates returns available templates."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/templates")
            assert response.status_code == 200
            data = response.json()
            assert "templates" in data
            assert "categories" in data
            assert "total" in data


class TestV2AuthRoutes:
    """Test v2 authentication routes."""

    @pytest.mark.anyio
    async def test_auth_me_unauthorized(self, v2_app, test_data):
        """Test GET /api/auth/me returns 401 without token."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/auth/me")
            assert response.status_code == 401


class TestV2RouteConsistency:
    """Test that v2 routes are consistent with v1 routes."""

    @pytest.mark.anyio
    async def test_all_expected_routes_exist(self, v2_app, test_data):
        """Verify all expected v2 routes are registered."""
        routes = [route.path for route in v2_app.routes]

        # Check that all major route categories exist
        expected_prefixes = [
            "/api/auth",
            "/api/agents",
            "/api/suites",
            "/api/tests",
            "/api/trends",
            "/api/compare",
            "/api/leaderboard",
            "/api/timeline",
            "/api/suite-definitions",
            "/api/templates",
            "/api/dashboard",
        ]

        for prefix in expected_prefixes:
            matching_routes = [r for r in routes if r.startswith(prefix)]
            assert len(matching_routes) > 0, f"No routes found for {prefix}"
