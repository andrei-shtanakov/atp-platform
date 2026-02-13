"""Integration tests for the multi-agent timeline comparison endpoint.

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
async def multi_agent_data(async_session: AsyncSession) -> dict:
    """Create test data for multi-agent timeline tests.

    Creates:
    - 3 agents (agent-alpha, agent-beta, agent-gamma)
    - 3 suite executions (one per agent)
    - 3 test executions with run results containing events
    """
    now = datetime.now()

    # Create agents
    agents = []
    for name in ["agent-alpha", "agent-beta", "agent-gamma"]:
        agent = Agent(
            name=name,
            agent_type="http",
            config={"endpoint": f"http://localhost:800{len(agents) + 1}"},
            description=f"{name.title()} agent for testing",
        )
        async_session.add(agent)
        agents.append(agent)
    await async_session.flush()

    # Create suite executions and test executions for each agent
    suite_executions = []
    test_executions = []
    run_results = []

    for i, agent in enumerate(agents):
        # Stagger start times to simulate different execution timing
        start_offset = timedelta(hours=2 - i * 0.1)

        suite = SuiteExecution(
            suite_name="benchmark-suite",
            agent_id=agent.id,
            started_at=now - start_offset,
            completed_at=now - start_offset + timedelta(hours=1),
            duration_seconds=3600.0,
            runs_per_test=1,
            total_tests=5,
            passed_tests=4 + i % 2,
            failed_tests=1 - i % 2,
            success_rate=0.8 + i * 0.05,
            status="completed",
        )
        async_session.add(suite)
        suite_executions.append(suite)
        await async_session.flush()

        # Create test execution with different durations
        duration_factor = 1.0 + i * 0.5
        test_exec = TestExecution(
            suite_execution_id=suite.id,
            test_id="test-001",
            test_name="Test Case 001",
            tags=["regression", "smoke"],
            started_at=now - start_offset,
            completed_at=now - start_offset + timedelta(minutes=30 * duration_factor),
            duration_seconds=1800.0 * duration_factor,
            total_runs=1,
            successful_runs=1,
            success=True,
            score=85.0 + i * 5,
            status="completed",
        )
        async_session.add(test_exec)
        test_executions.append(test_exec)
        await async_session.flush()

        # Create run results with events
        # Different agents have different numbers of steps
        num_steps = 5 + i * 2
        events = generate_realistic_event_sequence(
            task_id=f"test-001-{agent.name}",
            num_steps=num_steps,
            include_error=i == 1,  # Only agent-beta has an error
            start_time=now - start_offset,
        )

        run = RunResult(
            test_execution_id=test_exec.id,
            run_number=1,
            started_at=now - start_offset,
            completed_at=now - start_offset + timedelta(minutes=30 * duration_factor),
            duration_seconds=1800.0 * duration_factor,
            response_status="completed",
            success=True,
            total_tokens=1500 + i * 500,
            input_tokens=1000 + i * 300,
            output_tokens=500 + i * 200,
            total_steps=num_steps,
            tool_calls=num_steps,
            llm_calls=num_steps,
            cost_usd=0.015 + i * 0.005,
            events_json=events,
        )
        async_session.add(run)
        run_results.append(run)

    await async_session.commit()

    return {
        "agents": agents,
        "suite_executions": suite_executions,
        "test_executions": test_executions,
        "run_results": run_results,
    }


@pytest.fixture
async def client_with_data(
    async_session: AsyncSession, multi_agent_data: dict
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


class TestMultiTimelineEndpointIntegration:
    """Integration tests for /timeline/compare endpoint."""

    @pytest.mark.anyio
    async def test_multi_timeline_returns_timelines_for_two_agents(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that endpoint returns timelines for 2 agents."""
        response = await client_with_data.get(
            "/api/timeline/compare",
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
        assert len(data["timelines"]) == 2

    @pytest.mark.anyio
    async def test_multi_timeline_returns_timelines_for_three_agents(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that endpoint returns timelines for 3 agents."""
        response = await client_with_data.get(
            "/api/timeline/compare",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta", "agent-gamma"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["timelines"]) == 3
        agent_names = [t["agent_name"] for t in data["timelines"]]
        assert "agent-alpha" in agent_names
        assert "agent-beta" in agent_names
        assert "agent-gamma" in agent_names

    @pytest.mark.anyio
    async def test_multi_timeline_has_correct_structure(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that timeline data has correct structure."""
        response = await client_with_data.get(
            "/api/timeline/compare",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check timeline structure
        for timeline in data["timelines"]:
            assert "agent_name" in timeline
            assert "test_execution_id" in timeline
            assert "start_time" in timeline
            assert "total_duration_ms" in timeline
            assert "events" in timeline
            assert isinstance(timeline["events"], list)

    @pytest.mark.anyio
    async def test_multi_timeline_events_have_relative_timing(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that events have relative timing from first event."""
        response = await client_with_data.get(
            "/api/timeline/compare",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        for timeline in data["timelines"]:
            events = timeline["events"]
            if events:
                # First event should have relative_time_ms = 0
                assert events[0]["relative_time_ms"] == 0.0

                # Subsequent events should have increasing relative times
                prev_time = 0.0
                for event in events[1:]:
                    assert event["relative_time_ms"] >= prev_time
                    prev_time = event["relative_time_ms"]

    @pytest.mark.anyio
    async def test_multi_timeline_total_duration_per_agent(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that each agent has its own total duration."""
        response = await client_with_data.get(
            "/api/timeline/compare",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta", "agent-gamma"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Each agent should have different durations
        # (based on how we set up the test data)
        durations = [t["total_duration_ms"] for t in data["timelines"]]
        assert all(d >= 0 for d in durations)

    @pytest.mark.anyio
    async def test_multi_timeline_filter_by_event_type(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test filtering events by event type."""
        response = await client_with_data.get(
            "/api/timeline/compare",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta"],
                "event_types": ["tool_call"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # All events in all timelines should be tool_call type
        for timeline in data["timelines"]:
            for event in timeline["events"]:
                assert event["event_type"] == "tool_call"

    @pytest.mark.anyio
    async def test_multi_timeline_filter_multiple_types(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test filtering events by multiple event types."""
        response = await client_with_data.get(
            "/api/timeline/compare",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta"],
                "event_types": ["tool_call", "llm_request"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # All events should be tool_call or llm_request type
        for timeline in data["timelines"]:
            for event in timeline["events"]:
                assert event["event_type"] in ["tool_call", "llm_request"]

    @pytest.mark.anyio
    async def test_multi_timeline_skips_missing_agent(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that endpoint skips agents without executions."""
        response = await client_with_data.get(
            "/api/timeline/compare",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "nonexistent-agent"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should return only the existing agent's timeline
        assert len(data["timelines"]) == 1
        assert data["timelines"][0]["agent_name"] == "agent-alpha"

    @pytest.mark.anyio
    async def test_multi_timeline_404_all_agents_missing(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that 404 is returned when all agents are missing."""
        response = await client_with_data.get(
            "/api/timeline/compare",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["nonexistent-1", "nonexistent-2"],
            },
        )

        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_multi_timeline_404_wrong_suite(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that 404 is returned when suite doesn't exist."""
        response = await client_with_data.get(
            "/api/timeline/compare",
            params={
                "suite_name": "nonexistent-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta"],
            },
        )

        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_multi_timeline_404_wrong_test(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that 404 is returned when test doesn't exist."""
        response = await client_with_data.get(
            "/api/timeline/compare",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "nonexistent-test",
                "agents": ["agent-alpha", "agent-beta"],
            },
        )

        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_multi_timeline_test_execution_ids(
        self, client_with_data: AsyncClient, multi_agent_data: dict
    ) -> None:
        """Test that test_execution_id is returned correctly."""
        response = await client_with_data.get(
            "/api/timeline/compare",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Each timeline should have a valid test_execution_id
        for timeline in data["timelines"]:
            assert timeline["test_execution_id"] > 0


class TestMultiTimelineEdgeCases:
    """Edge case tests for multi-agent timeline endpoint."""

    @pytest.mark.anyio
    async def test_multi_timeline_with_empty_events(
        self, async_session: AsyncSession
    ) -> None:
        """Test endpoint with agent having no events."""
        # Create agents
        agent1 = Agent(
            name="agent-with-events",
            agent_type="http",
            config={},
        )
        agent2 = Agent(
            name="agent-no-events",
            agent_type="http",
            config={},
        )
        async_session.add_all([agent1, agent2])
        await async_session.flush()

        now = datetime.now()

        # Create suite executions for both agents
        for agent in [agent1, agent2]:
            suite = SuiteExecution(
                suite_name="empty-test-suite",
                agent_id=agent.id,
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
            async_session.add(suite)
            await async_session.flush()

            test_exec = TestExecution(
                suite_execution_id=suite.id,
                test_id="test-empty",
                test_name="Empty Test",
                tags=[],
                started_at=now,
                completed_at=now,
                duration_seconds=100.0,
                total_runs=1,
                successful_runs=1,
                success=True,
                score=100.0,
                status="completed",
            )
            async_session.add(test_exec)
            await async_session.flush()

            # First agent has events, second doesn't
            events = []
            if agent.name == "agent-with-events":
                events = generate_realistic_event_sequence(
                    task_id="test-empty",
                    num_steps=3,
                    start_time=now,
                )

            run = RunResult(
                test_execution_id=test_exec.id,
                run_number=1,
                started_at=now,
                completed_at=now,
                duration_seconds=100.0,
                response_status="completed",
                success=True,
                events_json=events,
            )
            async_session.add(run)

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
                "/api/timeline/compare",
                params={
                    "suite_name": "empty-test-suite",
                    "test_id": "test-empty",
                    "agents": ["agent-with-events", "agent-no-events"],
                },
            )

        test_app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()

        assert len(data["timelines"]) == 2

        # Find each agent's timeline
        timelines_by_agent = {t["agent_name"]: t for t in data["timelines"]}

        # Agent with events should have events
        assert len(timelines_by_agent["agent-with-events"]["events"]) > 0

        # Agent without events should have empty events list
        assert len(timelines_by_agent["agent-no-events"]["events"]) == 0
        assert timelines_by_agent["agent-no-events"]["total_duration_ms"] == 0.0

    @pytest.mark.anyio
    async def test_multi_timeline_events_sorted_by_sequence(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that events are sorted by sequence number."""
        response = await client_with_data.get(
            "/api/timeline/compare",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agents": ["agent-alpha", "agent-beta"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        for timeline in data["timelines"]:
            sequences = [e["sequence"] for e in timeline["events"]]
            assert sequences == sorted(sequences), (
                f"Events for {timeline['agent_name']} should be sorted by sequence"
            )
