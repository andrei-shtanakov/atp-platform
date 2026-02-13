"""Integration tests for the timeline events endpoint.

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
    """Create test data for timeline events tests.

    Creates:
    - 1 agent (agent-alpha)
    - 1 suite execution
    - 1 test execution with run results containing events
    """
    # Create agent
    agent = Agent(
        name="agent-alpha",
        agent_type="http",
        config={"endpoint": "http://localhost:8001"},
        description="Alpha agent for testing",
    )
    async_session.add(agent)
    await async_session.flush()

    # Create suite execution
    now = datetime.now()
    suite = SuiteExecution(
        suite_name="benchmark-suite",
        agent_id=agent.id,
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
    async_session.add(suite)
    await async_session.flush()

    # Create test execution
    test_exec = TestExecution(
        suite_execution_id=suite.id,
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
    async_session.add(test_exec)
    await async_session.flush()

    # Create run results with events
    events = generate_realistic_event_sequence(
        task_id="test-001-alpha",
        num_steps=5,
        include_error=False,
        start_time=now - timedelta(hours=2),
    )

    run = RunResult(
        test_execution_id=test_exec.id,
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
        events_json=events,
    )
    async_session.add(run)
    await async_session.commit()

    return {
        "agent": agent,
        "suite": suite,
        "test": test_exec,
        "run": run,
        "events": events,
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


class TestTimelineEventsEndpointIntegration:
    """Integration tests for /timeline/events endpoint."""

    @pytest.mark.anyio
    async def test_timeline_events_returns_events(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that endpoint returns events for valid parameters."""
        response = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agent_name": "agent-alpha",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["suite_name"] == "benchmark-suite"
        assert data["test_id"] == "test-001"
        assert data["test_name"] == "Test Case 001"
        assert data["agent_name"] == "agent-alpha"
        assert data["total_events"] > 0
        assert len(data["events"]) > 0

    @pytest.mark.anyio
    async def test_timeline_events_contains_correct_structure(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that event data has correct structure."""
        response = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agent_name": "agent-alpha",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check event structure
        for event in data["events"]:
            assert "sequence" in event
            assert "timestamp" in event
            assert "event_type" in event
            assert "summary" in event
            assert "data" in event
            assert "relative_time_ms" in event
            assert "duration_ms" in event

    @pytest.mark.anyio
    async def test_timeline_events_has_relative_timing(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that events have relative timing from first event."""
        response = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agent_name": "agent-alpha",
            },
        )

        assert response.status_code == 200
        data = response.json()

        events = data["events"]
        assert len(events) > 1

        # First event should have relative_time_ms = 0
        assert events[0]["relative_time_ms"] == 0.0

        # Subsequent events should have increasing relative times
        prev_time = 0.0
        for event in events[1:]:
            assert event["relative_time_ms"] >= prev_time
            prev_time = event["relative_time_ms"]

    @pytest.mark.anyio
    async def test_timeline_events_sorted_by_sequence(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that events are sorted by sequence number."""
        response = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agent_name": "agent-alpha",
            },
        )

        assert response.status_code == 200
        data = response.json()

        sequences = [e["sequence"] for e in data["events"]]
        assert sequences == sorted(sequences), "Events should be sorted by sequence"

    @pytest.mark.anyio
    async def test_timeline_events_total_duration(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that total duration is calculated correctly."""
        response = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agent_name": "agent-alpha",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Total duration should be present and positive
        assert data["total_duration_ms"] is not None
        assert data["total_duration_ms"] > 0

    @pytest.mark.anyio
    async def test_timeline_events_filter_by_type(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test filtering events by event type."""
        response = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agent_name": "agent-alpha",
                "event_types": ["tool_call"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # All events should be tool_call type
        for event in data["events"]:
            assert event["event_type"] == "tool_call"

    @pytest.mark.anyio
    async def test_timeline_events_filter_multiple_types(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test filtering events by multiple event types."""
        response = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agent_name": "agent-alpha",
                "event_types": ["tool_call", "llm_request"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # All events should be tool_call or llm_request type
        for event in data["events"]:
            assert event["event_type"] in ["tool_call", "llm_request"]

    @pytest.mark.anyio
    async def test_timeline_events_limit(self, client_with_data: AsyncClient) -> None:
        """Test that limit parameter works correctly."""
        response = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agent_name": "agent-alpha",
                "limit": 5,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should return at most 5 events
        assert len(data["events"]) <= 5
        # Total events should be the full count
        assert data["total_events"] >= len(data["events"])

    @pytest.mark.anyio
    async def test_timeline_events_offset(self, client_with_data: AsyncClient) -> None:
        """Test that offset parameter works correctly."""
        # Get all events first
        response_all = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agent_name": "agent-alpha",
            },
        )
        all_events = response_all.json()["events"]

        # Get events with offset
        response_offset = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agent_name": "agent-alpha",
                "offset": 2,
            },
        )

        assert response_offset.status_code == 200
        data = response_offset.json()

        # Offset events should start from index 2
        if len(all_events) > 2:
            assert data["events"][0]["sequence"] == all_events[2]["sequence"]

    @pytest.mark.anyio
    async def test_timeline_events_max_limit(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that limit cannot exceed 1000."""
        response = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agent_name": "agent-alpha",
                "limit": 2000,
            },
        )

        # Should return 422 validation error
        assert response.status_code == 422

    @pytest.mark.anyio
    async def test_timeline_events_404_wrong_suite(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that 404 is returned when suite doesn't exist."""
        response = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "nonexistent-suite",
                "test_id": "test-001",
                "agent_name": "agent-alpha",
            },
        )

        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_timeline_events_404_wrong_test(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that 404 is returned when test doesn't exist."""
        response = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "nonexistent-test",
                "agent_name": "agent-alpha",
            },
        )

        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_timeline_events_404_wrong_agent(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that 404 is returned when agent doesn't exist."""
        response = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agent_name": "nonexistent-agent",
            },
        )

        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_timeline_events_execution_id(
        self, client_with_data: AsyncClient, test_data: dict
    ) -> None:
        """Test that execution_id is returned correctly."""
        response = await client_with_data.get(
            "/api/timeline/events",
            params={
                "suite_name": "benchmark-suite",
                "test_id": "test-001",
                "agent_name": "agent-alpha",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Execution ID should match the test execution
        assert data["execution_id"] == test_data["test"].id


class TestTimelineEventsEdgeCases:
    """Edge case tests for timeline events endpoint."""

    @pytest.mark.anyio
    async def test_timeline_events_empty_events(
        self, async_session: AsyncSession
    ) -> None:
        """Test endpoint with no events in run result."""
        # Create agent
        agent = Agent(
            name="agent-empty",
            agent_type="http",
            config={},
        )
        async_session.add(agent)
        await async_session.flush()

        now = datetime.now()
        suite = SuiteExecution(
            suite_name="empty-suite",
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

        run = RunResult(
            test_execution_id=test_exec.id,
            run_number=1,
            started_at=now,
            completed_at=now,
            duration_seconds=100.0,
            response_status="completed",
            success=True,
            events_json=[],  # Empty events
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
                "/api/timeline/events",
                params={
                    "suite_name": "empty-suite",
                    "test_id": "test-empty",
                    "agent_name": "agent-empty",
                },
            )

        test_app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["total_events"] == 0
        assert data["events"] == []
        assert data["total_duration_ms"] is None

    @pytest.mark.anyio
    async def test_timeline_events_no_run_result(
        self, async_session: AsyncSession
    ) -> None:
        """Test endpoint with no run results."""
        # Create agent
        agent = Agent(
            name="agent-no-run",
            agent_type="http",
            config={},
        )
        async_session.add(agent)
        await async_session.flush()

        now = datetime.now()
        suite = SuiteExecution(
            suite_name="no-run-suite",
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
            test_id="test-no-run",
            test_name="No Run Test",
            tags=[],
            started_at=now,
            completed_at=now,
            duration_seconds=100.0,
            total_runs=0,
            successful_runs=0,
            success=False,
            score=0.0,
            status="completed",
        )
        async_session.add(test_exec)
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
                "/api/timeline/events",
                params={
                    "suite_name": "no-run-suite",
                    "test_id": "test-no-run",
                    "agent_name": "agent-no-run",
                },
            )

        test_app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["total_events"] == 0
        assert data["events"] == []

    @pytest.mark.anyio
    async def test_timeline_events_duration_extraction(
        self, async_session: AsyncSession
    ) -> None:
        """Test that duration_ms is extracted from event payload."""
        # Create agent
        agent = Agent(
            name="agent-duration",
            agent_type="http",
            config={},
        )
        async_session.add(agent)
        await async_session.flush()

        now = datetime.now()
        suite = SuiteExecution(
            suite_name="duration-suite",
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
            test_id="test-duration",
            test_name="Duration Test",
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

        # Create events with duration_ms in payload
        events = [
            {
                "version": "1.0",
                "task_id": "test-duration",
                "timestamp": now.isoformat(),
                "sequence": 0,
                "event_type": "tool_call",
                "payload": {
                    "tool": "test_tool",
                    "status": "success",
                    "duration_ms": 1234.5,
                },
            },
            {
                "version": "1.0",
                "task_id": "test-duration",
                "timestamp": (now + timedelta(seconds=2)).isoformat(),
                "sequence": 1,
                "event_type": "llm_request",
                "payload": {
                    "model": "test-model",
                    "duration_ms": 5678.9,
                },
            },
        ]

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
                "/api/timeline/events",
                params={
                    "suite_name": "duration-suite",
                    "test_id": "test-duration",
                    "agent_name": "agent-duration",
                },
            )

        test_app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()

        # Check that duration_ms was extracted from payload
        assert data["events"][0]["duration_ms"] == 1234.5
        assert data["events"][1]["duration_ms"] == 5678.9
