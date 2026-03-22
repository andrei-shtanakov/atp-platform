"""Tests for the optimized queries module."""

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from atp.dashboard.models import (
    Agent,
    Base,
    RunResult,
    SuiteExecution,
    TestExecution,
)
from atp.dashboard.optimized_queries import (
    build_leaderboard_data,
    get_agents_by_names,
    get_run_results_for_test_executions,
    get_suite_executions_for_agents,
)


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
    """Create test data for optimized query tests.

    Creates:
    - 3 agents
    - 2 suite executions per agent
    - 3 test executions per suite
    - 1 run result per test
    """
    now = datetime.now()

    # Create agents
    agents = [
        Agent(name="agent-alpha", agent_type="http", config={}),
        Agent(name="agent-beta", agent_type="http", config={}),
        Agent(name="agent-gamma", agent_type="cli", config={}),
    ]
    async_session.add_all(agents)
    await async_session.flush()

    suite_executions = []
    test_executions = []
    run_results = []

    # Create suite executions for each agent
    for agent in agents:
        for exec_num in range(2):
            suite = SuiteExecution(
                suite_name="benchmark-suite",
                agent_id=agent.id,
                started_at=now - timedelta(hours=exec_num + 1),
                completed_at=now - timedelta(hours=exec_num),
                duration_seconds=3600.0,
                runs_per_test=1,
                total_tests=3,
                passed_tests=2,
                failed_tests=1,
                success_rate=0.67,
                status="completed",
            )
            suite_executions.append(suite)

    async_session.add_all(suite_executions)
    await async_session.flush()

    # Create test executions for each suite
    for suite in suite_executions:
        for test_num in range(3):
            test_exec = TestExecution(
                suite_execution_id=suite.id,
                test_id=f"test-{test_num + 1:03d}",
                test_name=f"Test {test_num + 1}",
                tags=["test"],
                started_at=suite.started_at,
                completed_at=suite.completed_at,
                duration_seconds=1200.0,
                total_runs=1,
                successful_runs=1 if test_num < 2 else 0,
                success=test_num < 2,
                score=80.0 - (test_num * 20),
                status="completed",
            )
            test_executions.append(test_exec)

    async_session.add_all(test_executions)
    await async_session.flush()

    # Create run results for each test
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


class TestGetAgentsByNames:
    """Tests for get_agents_by_names function."""

    @pytest.mark.anyio
    async def test_get_all_agents(
        self, async_session: AsyncSession, test_data: dict
    ) -> None:
        """Test getting all agents when names is None."""
        agents = await get_agents_by_names(async_session, None)
        assert len(agents) == 3
        names = {a.name for a in agents}
        assert names == {"agent-alpha", "agent-beta", "agent-gamma"}

    @pytest.mark.anyio
    async def test_get_agents_by_specific_names(
        self, async_session: AsyncSession, test_data: dict
    ) -> None:
        """Test getting specific agents by name."""
        agents = await get_agents_by_names(async_session, ["agent-alpha", "agent-beta"])
        assert len(agents) == 2
        names = {a.name for a in agents}
        assert names == {"agent-alpha", "agent-beta"}

    @pytest.mark.anyio
    async def test_get_agents_empty_list(
        self, async_session: AsyncSession, test_data: dict
    ) -> None:
        """Test getting agents with empty list returns all (same as None)."""
        # An empty list is treated the same as None - returns all agents
        agents = await get_agents_by_names(async_session, [])
        assert len(agents) == 3

    @pytest.mark.anyio
    async def test_get_agents_nonexistent(
        self, async_session: AsyncSession, test_data: dict
    ) -> None:
        """Test getting nonexistent agents."""
        agents = await get_agents_by_names(async_session, ["nonexistent"])
        assert len(agents) == 0


class TestGetSuiteExecutionsForAgents:
    """Tests for get_suite_executions_for_agents function."""

    @pytest.mark.anyio
    async def test_get_suite_executions(
        self, async_session: AsyncSession, test_data: dict
    ) -> None:
        """Test getting suite executions for multiple agents."""
        agent_ids = [a.id for a in test_data["agents"]]
        executions = await get_suite_executions_for_agents(
            async_session, "benchmark-suite", agent_ids, limit_per_agent=5
        )
        # 3 agents x 2 executions each = 6 total
        assert len(executions) == 6

    @pytest.mark.anyio
    async def test_get_suite_executions_with_limit(
        self, async_session: AsyncSession, test_data: dict
    ) -> None:
        """Test limiting executions per agent."""
        agent_ids = [a.id for a in test_data["agents"]]
        executions = await get_suite_executions_for_agents(
            async_session, "benchmark-suite", agent_ids, limit_per_agent=1
        )
        # 3 agents x 1 execution each = 3 total
        assert len(executions) == 3

    @pytest.mark.anyio
    async def test_get_suite_executions_empty_agent_list(
        self, async_session: AsyncSession, test_data: dict
    ) -> None:
        """Test with empty agent list."""
        executions = await get_suite_executions_for_agents(
            async_session, "benchmark-suite", [], limit_per_agent=5
        )
        assert len(executions) == 0

    @pytest.mark.anyio
    async def test_get_suite_executions_nonexistent_suite(
        self, async_session: AsyncSession, test_data: dict
    ) -> None:
        """Test with nonexistent suite."""
        agent_ids = [a.id for a in test_data["agents"]]
        executions = await get_suite_executions_for_agents(
            async_session, "nonexistent-suite", agent_ids, limit_per_agent=5
        )
        assert len(executions) == 0


class TestGetRunResultsForTestExecutions:
    """Tests for get_run_results_for_test_executions function."""

    @pytest.mark.anyio
    async def test_get_run_results(
        self, async_session: AsyncSession, test_data: dict
    ) -> None:
        """Test getting run results for test executions."""
        test_exec_ids = [te.id for te in test_data["test_executions"][:3]]
        results = await get_run_results_for_test_executions(
            async_session, test_exec_ids
        )
        assert len(results) == 3
        for test_id in test_exec_ids:
            assert test_id in results
            assert len(results[test_id]) == 1

    @pytest.mark.anyio
    async def test_get_run_results_empty_list(
        self, async_session: AsyncSession, test_data: dict
    ) -> None:
        """Test with empty test execution list."""
        results = await get_run_results_for_test_executions(async_session, [])
        assert len(results) == 0


class TestBuildLeaderboardData:
    """Tests for build_leaderboard_data function."""

    @pytest.mark.anyio
    async def test_build_leaderboard_data(
        self, async_session: AsyncSession, test_data: dict
    ) -> None:
        """Test building leaderboard data."""
        agent_names = ["agent-alpha", "agent-beta", "agent-gamma"]
        (
            test_data_result,
            test_names,
            test_tags,
            agent_metrics,
        ) = await build_leaderboard_data(
            async_session,
            "benchmark-suite",
            agent_names,
            limit_executions=5,
        )

        # Should have 3 tests
        assert len(test_data_result) == 3
        assert len(test_names) == 3
        assert len(test_tags) == 3

        # Each agent should have metrics
        for name in agent_names:
            assert name in agent_metrics
            assert "scores" in agent_metrics[name]
            assert "successes" in agent_metrics[name]
            assert "tokens" in agent_metrics[name]
            assert "cost" in agent_metrics[name]

    @pytest.mark.anyio
    async def test_build_leaderboard_data_empty_agents(
        self, async_session: AsyncSession, test_data: dict
    ) -> None:
        """Test with no agents."""
        (
            test_data_result,
            test_names,
            test_tags,
            agent_metrics,
        ) = await build_leaderboard_data(
            async_session,
            "benchmark-suite",
            [],
            limit_executions=5,
        )
        assert len(test_data_result) == 0
        assert len(agent_metrics) == 0

    @pytest.mark.anyio
    async def test_build_leaderboard_data_nonexistent_suite(
        self, async_session: AsyncSession, test_data: dict
    ) -> None:
        """Test with nonexistent suite."""
        agent_names = ["agent-alpha", "agent-beta", "agent-gamma"]
        (
            test_data_result,
            test_names,
            test_tags,
            agent_metrics,
        ) = await build_leaderboard_data(
            async_session,
            "nonexistent-suite",
            agent_names,
            limit_executions=5,
        )
        # Test data should be empty
        assert len(test_data_result) == 0
        # Agent metrics should be initialized but empty
        for name in agent_names:
            assert name in agent_metrics
            assert len(agent_metrics[name]["scores"]) == 0
