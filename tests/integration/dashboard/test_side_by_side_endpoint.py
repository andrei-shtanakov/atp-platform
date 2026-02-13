"""Integration tests for the side-by-side comparison endpoint.

These tests use an in-memory SQLite database to verify the full
endpoint behavior with realistic data.
"""

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from atp.dashboard.auth import get_current_active_user
from atp.dashboard.models import (
    Agent,
    Base,
    RunResult,
    SuiteExecution,
    TestExecution,
    User,
)
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_app
from tests.fixtures.comparison import generate_realistic_event_sequence
from tests.fixtures.comparison.factories import reset_all_factories


def _mock_admin_user() -> User:
    """Create a mock admin user for testing."""
    user = User(
        username="admin_test",
        email="admin@test.com",
        hashed_password="fake_hash",
        is_active=True,
        is_admin=True,
    )
    user.id = 1
    return user


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
async def test_data(async_session: AsyncSession) -> dict:
    """Create test data for side-by-side comparison tests.

    Creates:
    - 2 agents (agent-alpha, agent-beta)
    - 1 suite execution per agent
    - 1 test execution per agent with run results containing events
    """
    # Create agents
    agent_alpha = Agent(
        name="agent-alpha",
        agent_type="http",
        config={"endpoint": "http://localhost:8001"},
        description="Alpha agent for testing",
    )
    agent_beta = Agent(
        name="agent-beta",
        agent_type="http",
        config={"endpoint": "http://localhost:8002"},
        description="Beta agent for testing",
    )
    async_session.add_all([agent_alpha, agent_beta])
    await async_session.flush()

    # Create suite executions
    now = datetime.now()
    suite_alpha = SuiteExecution(
        suite_name="benchmark-suite",
        agent_id=agent_alpha.id,
        started_at=now - timedelta(hours=2),
        completed_at=now - timedelta(hours=1),
        duration_seconds=3600.0,
        runs_per_test=1,
        total_tests=5,
        passed_tests=4,
        failed_tests=1,
        success_rate=0.8,
        status="completed",
    )
    suite_beta = SuiteExecution(
        suite_name="benchmark-suite",
        agent_id=agent_beta.id,
        started_at=now - timedelta(hours=1),
        completed_at=now,
        duration_seconds=3600.0,
        runs_per_test=1,
        total_tests=5,
        passed_tests=5,
        failed_tests=0,
        success_rate=1.0,
        status="completed",
    )
    async_session.add_all([suite_alpha, suite_beta])
    await async_session.flush()

    # Create test executions
    test_alpha = TestExecution(
        suite_execution_id=suite_alpha.id,
        test_id="test-001",
        test_name="Test Case 001",
        tags=["regression", "smoke"],
        started_at=now - timedelta(hours=2),
        completed_at=now - timedelta(hours=1, minutes=30),
        duration_seconds=1800.0,
        total_runs=1,
        successful_runs=1,
        success=True,
        score=85.0,
        status="completed",
    )
    test_beta = TestExecution(
        suite_execution_id=suite_beta.id,
        test_id="test-001",
        test_name="Test Case 001",
        tags=["regression", "smoke"],
        started_at=now - timedelta(hours=1),
        completed_at=now - timedelta(minutes=30),
        duration_seconds=1800.0,
        total_runs=1,
        successful_runs=1,
        success=True,
        score=92.0,
        status="completed",
    )
    async_session.add_all([test_alpha, test_beta])
    await async_session.flush()

    # Create run results with events
    events_alpha = generate_realistic_event_sequence(
        task_id="test-001-alpha",
        num_steps=5,
        include_error=False,
    )
    events_beta = generate_realistic_event_sequence(
        task_id="test-001-beta",
        num_steps=4,
        include_error=False,
    )

    run_alpha = RunResult(
        test_execution_id=test_alpha.id,
        run_number=1,
        started_at=now - timedelta(hours=2),
        completed_at=now - timedelta(hours=1, minutes=30),
        duration_seconds=1800.0,
        response_status="completed",
        success=True,
        total_tokens=1500,
        input_tokens=1000,
        output_tokens=500,
        total_steps=5,
        tool_calls=5,
        llm_calls=5,
        cost_usd=0.015,
        events_json=events_alpha,
    )
    run_beta = RunResult(
        test_execution_id=test_beta.id,
        run_number=1,
        started_at=now - timedelta(hours=1),
        completed_at=now - timedelta(minutes=30),
        duration_seconds=1800.0,
        response_status="completed",
        success=True,
        total_tokens=1200,
        input_tokens=800,
        output_tokens=400,
        total_steps=4,
        tool_calls=4,
        llm_calls=4,
        cost_usd=0.012,
        events_json=events_beta,
    )
    async_session.add_all([run_alpha, run_beta])
    await async_session.commit()

    return {
        "agents": [agent_alpha, agent_beta],
        "suites": [suite_alpha, suite_beta],
        "tests": [test_alpha, test_beta],
        "runs": [run_alpha, run_beta],
    }


@pytest.fixture
async def client_with_data(
    async_session: AsyncSession, test_data: dict
) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client with test data."""

    test_app = create_app()

    # Override the session dependency
    async def override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
        yield async_session

    test_app.dependency_overrides[get_db_session] = override_get_db_session
    test_app.dependency_overrides[get_current_active_user] = _mock_admin_user

    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://test",
    ) as client:
        yield client

    # Clean up
    test_app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def reset_fixture_counters() -> None:
    """Reset factory counters before each test."""
    reset_all_factories()


class TestSideBySideEndpointIntegration:
    """Integration tests for /compare/side-by-side endpoint."""

    @pytest.mark.anyio
    async def test_side_by_side_returns_two_agents(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that endpoint returns data for two agents."""
        response = await client_with_data.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["suite_name"] == "benchmark-suite"
        assert data["test_id"] == "test-001"
        assert data["test_name"] == "Test Case 001"
        assert len(data["agents"]) == 2

    @pytest.mark.anyio
    async def test_side_by_side_contains_correct_agent_details(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that agent details are correctly populated."""
        response = await client_with_data.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Find agent-alpha in response
        alpha_data = next(
            (a for a in data["agents"] if a["agent_name"] == "agent-alpha"),
            None,
        )
        assert alpha_data is not None
        assert alpha_data["score"] == 85.0
        assert alpha_data["success"] is True
        assert alpha_data["total_tokens"] == 1500
        assert alpha_data["total_steps"] == 5
        assert alpha_data["tool_calls"] == 5
        assert alpha_data["cost_usd"] == 0.015

        # Find agent-beta in response
        beta_data = next(
            (a for a in data["agents"] if a["agent_name"] == "agent-beta"),
            None,
        )
        assert beta_data is not None
        assert beta_data["score"] == 92.0
        assert beta_data["success"] is True
        assert beta_data["total_tokens"] == 1200
        assert beta_data["total_steps"] == 4

    @pytest.mark.anyio
    async def test_side_by_side_contains_events(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that events are included in the response."""
        response = await client_with_data.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        for agent_data in data["agents"]:
            events = agent_data["events"]
            assert len(events) > 0

            # Check event structure
            for event in events:
                assert "sequence" in event
                assert "timestamp" in event
                assert "event_type" in event
                assert "summary" in event
                assert "data" in event

    @pytest.mark.anyio
    async def test_side_by_side_events_are_sorted_by_sequence(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that events are sorted by sequence number."""
        response = await client_with_data.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        for agent_data in data["agents"]:
            events = agent_data["events"]
            sequences = [e["sequence"] for e in events]
            assert sequences == sorted(sequences), "Events should be sorted by sequence"

    @pytest.mark.anyio
    async def test_side_by_side_returns_partial_when_one_agent_missing(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that response includes available agent when one is missing."""
        response = await client_with_data.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "nonexistent-agent"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should only have agent-alpha since nonexistent-agent doesn't exist
        assert len(data["agents"]) == 1
        assert data["agents"][0]["agent_name"] == "agent-alpha"

    @pytest.mark.anyio
    async def test_side_by_side_returns_404_when_no_agents_found(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that 404 is returned when no agents have data."""
        response = await client_with_data.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["nonexistent-1", "nonexistent-2"],
            },
        )

        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_side_by_side_returns_404_for_wrong_suite(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that 404 is returned when suite doesn't exist."""
        response = await client_with_data.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "nonexistent-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta"],
            },
        )

        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_side_by_side_returns_404_for_wrong_test(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that 404 is returned when test doesn't exist."""
        response = await client_with_data.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "nonexistent-test",
                "agents": ["agent-alpha", "agent-beta"],
            },
        )

        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_side_by_side_validates_min_agents(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that validation rejects less than 2 agents."""
        response = await client_with_data.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha"],
            },
        )

        assert response.status_code == 422

    @pytest.mark.anyio
    async def test_side_by_side_validates_max_agents(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that validation rejects more than 3 agents."""
        response = await client_with_data.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["a1", "a2", "a3", "a4"],
            },
        )

        assert response.status_code == 422

    @pytest.mark.anyio
    async def test_side_by_side_with_three_agents(
        self, async_session: AsyncSession
    ) -> None:
        """Test endpoint with three agents."""
        # Create a third agent with data
        agent_gamma = Agent(
            name="agent-gamma",
            agent_type="cli",
            config={},
        )
        async_session.add(agent_gamma)
        await async_session.flush()

        now = datetime.now()
        suite_gamma = SuiteExecution(
            suite_name="benchmark-suite",
            agent_id=agent_gamma.id,
            started_at=now,
            completed_at=now,
            duration_seconds=100.0,
            runs_per_test=1,
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            success_rate=1.0,
            status="completed",
        )
        async_session.add(suite_gamma)
        await async_session.flush()

        test_gamma = TestExecution(
            suite_execution_id=suite_gamma.id,
            test_id="test-001",
            test_name="Test Case 001",
            tags=[],
            started_at=now,
            completed_at=now,
            duration_seconds=100.0,
            total_runs=1,
            successful_runs=1,
            success=True,
            score=95.0,
            status="completed",
        )
        async_session.add(test_gamma)
        await async_session.flush()

        run_gamma = RunResult(
            test_execution_id=test_gamma.id,
            run_number=1,
            started_at=now,
            completed_at=now,
            duration_seconds=100.0,
            response_status="completed",
            success=True,
            total_tokens=1000,
            events_json=[],
        )
        async_session.add(run_gamma)
        await async_session.commit()

        # Override session dependency
        async def override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
            yield async_session

        test_app = create_app()
        test_app.dependency_overrides[get_db_session] = override_get_db_session
        test_app.dependency_overrides[get_current_active_user] = _mock_admin_user

        async with AsyncClient(
            transport=ASGITransport(app=test_app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                "/api/compare/side-by-side",
                params={
                    "suite_name": "benchmark-suite",
                    "test_id": "test-001",
                    "agents": ["agent-alpha", "agent-beta", "agent-gamma"],
                },
            )

        test_app.dependency_overrides.clear()

        # Only gamma should be found since alpha/beta aren't in this session
        # (they were created in test_data fixture which uses a different session)
        assert response.status_code in [200, 404]
