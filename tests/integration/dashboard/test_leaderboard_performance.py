"""Performance tests for the leaderboard matrix endpoint.

These tests verify that the leaderboard endpoint meets the performance
requirement of < 2 seconds response time with 50 tests × 10 agents.
"""

import time
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from atp.dashboard.api import get_session
from atp.dashboard.app import app
from atp.dashboard.models import (
    Agent,
    Base,
    RunResult,
    SuiteExecution,
    TestExecution,
)
from atp.dashboard.query_cache import clear_all_query_caches


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
async def large_test_data(async_session: AsyncSession) -> dict:
    """Create large test data set: 50 tests × 10 agents.

    This fixture creates a realistic dataset to test performance:
    - 10 agents
    - 5 suite executions per agent
    - 50 test executions per suite
    - 1 run result per test execution

    Total records:
    - 10 agents
    - 50 suite executions (10 agents × 5 executions)
    - 2500 test executions (50 suites × 50 tests)
    - 2500 run results
    """
    now = datetime.now()

    # Create 10 agents
    agents = []
    for i in range(10):
        agent = Agent(
            name=f"agent-{chr(65 + i)}",  # agent-A through agent-J
            agent_type="http",
            config={"endpoint": f"http://localhost:800{i}"},
            description=f"Performance test agent {chr(65 + i)}",
        )
        agents.append(agent)

    async_session.add_all(agents)
    await async_session.flush()

    suite_executions = []
    test_executions = []
    run_results = []

    # Create 5 suite executions per agent
    for agent in agents:
        for exec_num in range(5):
            suite = SuiteExecution(
                suite_name="perf-benchmark-suite",
                agent_id=agent.id,
                started_at=now - timedelta(hours=(exec_num + 1) * 2),
                completed_at=now - timedelta(hours=exec_num * 2),
                duration_seconds=7200.0,
                runs_per_test=1,
                total_tests=50,
                passed_tests=40,
                failed_tests=10,
                success_rate=0.8,
                status="completed",
            )
            suite_executions.append(suite)

    async_session.add_all(suite_executions)
    await async_session.flush()

    # Create 50 test executions per suite
    for suite in suite_executions:
        for test_num in range(50):
            # Vary scores across agents to create realistic data
            base_score = 60.0 + (test_num % 30)  # 60-90 range
            test_exec = TestExecution(
                suite_execution_id=suite.id,
                test_id=f"test-{test_num + 1:03d}",
                test_name=f"Test Case {test_num + 1}",
                tags=[
                    f"category-{test_num % 5}",
                    "smoke" if test_num < 10 else "regression",
                ],
                started_at=suite.started_at + timedelta(minutes=test_num * 2),
                completed_at=suite.started_at + timedelta(minutes=test_num * 2 + 1),
                duration_seconds=60.0,
                total_runs=1,
                successful_runs=1 if base_score >= 50 else 0,
                success=base_score >= 50,
                score=base_score,
                status="completed",
            )
            test_executions.append(test_exec)

    async_session.add_all(test_executions)
    await async_session.flush()

    # Create run results for each test execution
    for test_exec in test_executions:
        run = RunResult(
            test_execution_id=test_exec.id,
            run_number=1,
            started_at=test_exec.started_at,
            completed_at=test_exec.completed_at,
            duration_seconds=test_exec.duration_seconds,
            response_status="completed",
            success=test_exec.success,
            total_tokens=1000,
            input_tokens=700,
            output_tokens=300,
            cost_usd=0.01,
            events_json=[],
        )
        run_results.append(run)

    async_session.add_all(run_results)
    await async_session.commit()

    return {
        "agents": agents,
        "suite_executions": suite_executions,
        "test_executions": test_executions,
        "run_results": run_results,
    }


@pytest.fixture
async def client_with_large_data(
    async_session: AsyncSession, large_test_data: dict
) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client with large test data."""

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield async_session

    app.dependency_overrides[get_session] = override_get_session

    # Clear caches before each test
    clear_all_query_caches()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client

    app.dependency_overrides.clear()


class TestLeaderboardPerformance:
    """Performance tests for leaderboard endpoint."""

    @pytest.mark.anyio
    async def test_leaderboard_response_time_under_2_seconds(
        self, client_with_large_data: AsyncClient
    ) -> None:
        """Test that leaderboard responds in < 2 seconds with 50 tests × 10 agents.

        This is the primary performance requirement from TASK-011.
        """
        start_time = time.perf_counter()

        response = await client_with_large_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "perf-benchmark-suite"},
        )

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        assert response.status_code == 200
        data = response.json()

        # Verify we got the expected amount of data
        assert data["total_tests"] == 50
        assert len(data["agents"]) == 10

        # Performance assertion: < 2 seconds
        assert elapsed < 2.0, (
            f"Leaderboard response took {elapsed:.3f}s, "
            f"which exceeds the 2 second requirement"
        )

    @pytest.mark.anyio
    async def test_leaderboard_cached_response_faster(
        self, client_with_large_data: AsyncClient
    ) -> None:
        """Test that cached responses are significantly faster."""
        # First request (cache miss)
        start_time = time.perf_counter()
        response1 = await client_with_large_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "perf-benchmark-suite"},
        )
        first_request_time = time.perf_counter() - start_time

        assert response1.status_code == 200

        # Second request (cache hit)
        start_time = time.perf_counter()
        response2 = await client_with_large_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "perf-benchmark-suite"},
        )
        second_request_time = time.perf_counter() - start_time

        assert response2.status_code == 200

        # Cached request should be faster
        # (Note: in practice, the improvement should be significant)
        assert second_request_time <= first_request_time

    @pytest.mark.anyio
    async def test_leaderboard_pagination_performance(
        self, client_with_large_data: AsyncClient
    ) -> None:
        """Test that paginated requests maintain performance."""
        # Request with pagination
        start_time = time.perf_counter()

        response = await client_with_large_data.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "perf-benchmark-suite",
                "limit": 10,
                "offset": 20,
            },
        )

        elapsed = time.perf_counter() - start_time

        assert response.status_code == 200
        data = response.json()

        # Verify pagination works
        assert len(data["tests"]) == 10
        assert data["total_tests"] == 50
        assert data["offset"] == 20

        # Should still be fast
        assert elapsed < 2.0

    @pytest.mark.anyio
    async def test_leaderboard_filtered_agents_performance(
        self, client_with_large_data: AsyncClient
    ) -> None:
        """Test that filtering by agents maintains performance."""
        start_time = time.perf_counter()

        response = await client_with_large_data.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "perf-benchmark-suite",
                "agents": ["agent-A", "agent-B", "agent-C"],
            },
        )

        elapsed = time.perf_counter() - start_time

        assert response.status_code == 200
        data = response.json()

        # Verify filtering works
        assert len(data["agents"]) == 3
        assert data["total_agents"] == 3

        # Should be even faster with fewer agents
        assert elapsed < 2.0

    @pytest.mark.anyio
    async def test_leaderboard_data_integrity_with_large_dataset(
        self, client_with_large_data: AsyncClient
    ) -> None:
        """Test that data integrity is maintained with large dataset."""
        response = await client_with_large_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "perf-benchmark-suite"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify all 10 agents are present
        agent_names = {a["agent_name"] for a in data["agents"]}
        expected_names = {f"agent-{chr(65 + i)}" for i in range(10)}
        assert agent_names == expected_names

        # Verify all 50 tests are present (in total_tests)
        assert data["total_tests"] == 50

        # Verify each test has scores for all agents
        for test in data["tests"]:
            assert len(test["scores_by_agent"]) == 10

        # Verify agents have rankings
        ranks = [a["rank"] for a in data["agents"]]
        assert sorted(ranks) == list(range(1, 11))

        # Verify agents have metrics
        for agent in data["agents"]:
            assert agent["avg_score"] is not None
            assert agent["pass_rate"] is not None
            assert agent["total_tokens"] >= 0
