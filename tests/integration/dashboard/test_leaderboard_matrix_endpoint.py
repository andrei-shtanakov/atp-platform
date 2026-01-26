"""Integration tests for the leaderboard matrix endpoint.

These tests use an in-memory SQLite database to verify the full
endpoint behavior with realistic data.
"""

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
from tests.fixtures.comparison.factories import reset_all_factories


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
async def leaderboard_test_data(async_session: AsyncSession) -> dict:
    """Create test data for leaderboard matrix tests.

    Creates:
    - 3 agents (agent-alpha, agent-beta, agent-gamma)
    - 1 suite execution per agent for "benchmark-suite"
    - 5 test executions per agent with varying scores
    """
    # Create agents
    agent_alpha = Agent(
        name="agent-alpha",
        agent_type="http",
        config={"endpoint": "http://localhost:8001"},
        description="Alpha agent - best performer",
    )
    agent_beta = Agent(
        name="agent-beta",
        agent_type="http",
        config={"endpoint": "http://localhost:8002"},
        description="Beta agent - medium performer",
    )
    agent_gamma = Agent(
        name="agent-gamma",
        agent_type="cli",
        config={},
        description="Gamma agent - low performer",
    )
    async_session.add_all([agent_alpha, agent_beta, agent_gamma])
    await async_session.flush()

    now = datetime.now()

    # Create suite executions
    suite_alpha = SuiteExecution(
        suite_name="benchmark-suite",
        agent_id=agent_alpha.id,
        started_at=now - timedelta(hours=3),
        completed_at=now - timedelta(hours=2),
        duration_seconds=3600.0,
        runs_per_test=1,
        total_tests=5,
        passed_tests=5,
        failed_tests=0,
        success_rate=1.0,
        status="completed",
    )
    suite_beta = SuiteExecution(
        suite_name="benchmark-suite",
        agent_id=agent_beta.id,
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
    suite_gamma = SuiteExecution(
        suite_name="benchmark-suite",
        agent_id=agent_gamma.id,
        started_at=now - timedelta(hours=1),
        completed_at=now,
        duration_seconds=3600.0,
        runs_per_test=1,
        total_tests=5,
        passed_tests=2,
        failed_tests=3,
        success_rate=0.4,
        status="completed",
    )
    async_session.add_all([suite_alpha, suite_beta, suite_gamma])
    await async_session.flush()

    # Test definitions with varying difficulty
    test_definitions = [
        ("test-001", "Easy Test", ["smoke"], 90.0, 85.0, 80.0),  # Easy
        ("test-002", "Medium Test", ["regression"], 75.0, 65.0, 55.0),  # Medium
        ("test-003", "Hard Test", ["complex"], 50.0, 45.0, 30.0),  # Hard
        ("test-004", "Very Hard Test", ["expert"], 35.0, 25.0, 15.0),  # Very hard
        ("test-005", "Variable Test", ["variable"], 95.0, 50.0, 25.0),  # High variance
    ]

    test_executions = []
    run_results = []

    for test_def in test_definitions:
        test_id, test_name, tags, alpha_score, beta_score, gamma_score = test_def
        # Alpha test execution
        test_alpha = TestExecution(
            suite_execution_id=suite_alpha.id,
            test_id=test_id,
            test_name=test_name,
            tags=tags,
            started_at=now - timedelta(hours=3),
            completed_at=now - timedelta(hours=2, minutes=30),
            duration_seconds=1800.0,
            total_runs=1,
            successful_runs=1 if alpha_score >= 50 else 0,
            success=alpha_score >= 50,
            score=alpha_score,
            status="completed",
        )
        # Beta test execution
        test_beta = TestExecution(
            suite_execution_id=suite_beta.id,
            test_id=test_id,
            test_name=test_name,
            tags=tags,
            started_at=now - timedelta(hours=2),
            completed_at=now - timedelta(hours=1, minutes=30),
            duration_seconds=1800.0,
            total_runs=1,
            successful_runs=1 if beta_score >= 50 else 0,
            success=beta_score >= 50,
            score=beta_score,
            status="completed",
        )
        # Gamma test execution
        test_gamma = TestExecution(
            suite_execution_id=suite_gamma.id,
            test_id=test_id,
            test_name=test_name,
            tags=tags,
            started_at=now - timedelta(hours=1),
            completed_at=now - timedelta(minutes=30),
            duration_seconds=1800.0,
            total_runs=1,
            successful_runs=1 if gamma_score >= 50 else 0,
            success=gamma_score >= 50,
            score=gamma_score,
            status="completed",
        )
        test_executions.extend([test_alpha, test_beta, test_gamma])

    async_session.add_all(test_executions)
    await async_session.flush()

    # Create run results for each test execution
    for i, test_exec in enumerate(test_executions):
        run = RunResult(
            test_execution_id=test_exec.id,
            run_number=1,
            started_at=test_exec.started_at,
            completed_at=test_exec.completed_at,
            duration_seconds=test_exec.duration_seconds,
            response_status="completed",
            success=test_exec.success,
            total_tokens=1000 + (i * 100),
            input_tokens=700 + (i * 70),
            output_tokens=300 + (i * 30),
            total_steps=5,
            tool_calls=3,
            llm_calls=5,
            cost_usd=0.01 + (i * 0.001),
            events_json=[],
        )
        run_results.append(run)

    async_session.add_all(run_results)
    await async_session.commit()

    return {
        "agents": [agent_alpha, agent_beta, agent_gamma],
        "suites": [suite_alpha, suite_beta, suite_gamma],
        "tests": test_executions,
        "runs": run_results,
    }


@pytest.fixture
async def client_with_data(
    async_session: AsyncSession, leaderboard_test_data: dict
) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client with test data."""

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield async_session

    app.dependency_overrides[get_session] = override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def reset_fixture_counters() -> None:
    """Reset factory counters before each test."""
    reset_all_factories()


class TestLeaderboardMatrixEndpointIntegration:
    """Integration tests for /leaderboard/matrix endpoint."""

    @pytest.mark.anyio
    async def test_leaderboard_returns_all_agents(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that endpoint returns data for all agents."""
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "benchmark-suite"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["suite_name"] == "benchmark-suite"
        assert len(data["agents"]) == 3
        agent_names = {a["agent_name"] for a in data["agents"]}
        assert agent_names == {"agent-alpha", "agent-beta", "agent-gamma"}

    @pytest.mark.anyio
    async def test_leaderboard_returns_all_tests(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that endpoint returns all tests."""
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "benchmark-suite"},
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["tests"]) == 5
        assert data["total_tests"] == 5

    @pytest.mark.anyio
    async def test_leaderboard_agents_have_correct_rankings(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that agents are ranked correctly by score."""
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "benchmark-suite"},
        )

        assert response.status_code == 200
        data = response.json()

        # Find agents by name
        alpha = next(a for a in data["agents"] if a["agent_name"] == "agent-alpha")
        beta = next(a for a in data["agents"] if a["agent_name"] == "agent-beta")
        gamma = next(a for a in data["agents"] if a["agent_name"] == "agent-gamma")

        # Alpha should be rank 1 (highest scores)
        assert alpha["rank"] == 1
        # Beta should be rank 2
        assert beta["rank"] == 2
        # Gamma should be rank 3 (lowest scores)
        assert gamma["rank"] == 3

    @pytest.mark.anyio
    async def test_leaderboard_tests_have_correct_difficulty(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that tests have correct difficulty ratings."""
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "benchmark-suite"},
        )

        assert response.status_code == 200
        data = response.json()

        # Find tests by id
        tests_by_id = {t["test_id"]: t for t in data["tests"]}

        # test-001 avg score ~85 -> easy
        assert tests_by_id["test-001"]["difficulty"] == "easy"
        # test-002 avg score ~65 -> medium
        assert tests_by_id["test-002"]["difficulty"] == "medium"
        # test-003 avg score ~42 -> hard
        assert tests_by_id["test-003"]["difficulty"] == "hard"
        # test-004 avg score ~25 -> very_hard
        assert tests_by_id["test-004"]["difficulty"] == "very_hard"

    @pytest.mark.anyio
    async def test_leaderboard_tests_have_scores_by_agent(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that tests have scores for each agent."""
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "benchmark-suite"},
        )

        assert response.status_code == 200
        data = response.json()

        for test in data["tests"]:
            assert "scores_by_agent" in test
            assert len(test["scores_by_agent"]) == 3

            # Check score structure
            for agent_name, score_data in test["scores_by_agent"].items():
                assert "score" in score_data
                assert "success" in score_data
                assert "execution_count" in score_data

    @pytest.mark.anyio
    async def test_leaderboard_filters_by_agents(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test filtering leaderboard to specific agents."""
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "agents": ["agent-alpha", "agent-beta"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["agents"]) == 2
        agent_names = {a["agent_name"] for a in data["agents"]}
        assert agent_names == {"agent-alpha", "agent-beta"}
        assert data["total_agents"] == 2

    @pytest.mark.anyio
    async def test_leaderboard_pagination_limit(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test pagination with limit parameter."""
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "limit": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should return only 2 tests
        assert len(data["tests"]) == 2
        # But total_tests should still be 5
        assert data["total_tests"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 0

    @pytest.mark.anyio
    async def test_leaderboard_pagination_offset(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test pagination with offset parameter."""
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "limit": 2,
                "offset": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should return 2 tests starting from offset 2
        assert len(data["tests"]) == 2
        assert data["total_tests"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 2

    @pytest.mark.anyio
    async def test_leaderboard_returns_empty_tests_for_nonexistent_suite(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that no tests are returned for nonexistent suite.

        Note: Agents still appear in response (they exist in DB)
        but have no scores/data for this suite.
        """
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "nonexistent-suite"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["suite_name"] == "nonexistent-suite"
        assert len(data["tests"]) == 0
        assert data["total_tests"] == 0
        # Agents exist in DB so they're returned, but with no data
        for agent in data["agents"]:
            assert agent["avg_score"] is None
            assert agent["pass_rate"] == 0.0
            assert agent["total_tokens"] == 0

    @pytest.mark.anyio
    async def test_leaderboard_agents_have_metrics(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that agents have aggregate metrics."""
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "benchmark-suite"},
        )

        assert response.status_code == 200
        data = response.json()

        for agent in data["agents"]:
            assert "avg_score" in agent
            assert "pass_rate" in agent
            assert "total_tokens" in agent
            assert "total_cost" in agent
            assert "rank" in agent

            # Verify types
            assert agent["avg_score"] is None or isinstance(agent["avg_score"], float)
            assert isinstance(agent["pass_rate"], float)
            assert isinstance(agent["total_tokens"], int)

    @pytest.mark.anyio
    async def test_leaderboard_detects_high_variance_pattern(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that high variance pattern is detected."""
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "benchmark-suite"},
        )

        assert response.status_code == 200
        data = response.json()

        # test-005 has scores 95, 50, 25 - should have high_variance pattern
        test_005 = next(t for t in data["tests"] if t["test_id"] == "test-005")
        # Note: Pattern detection depends on exact algorithm
        # Score range is 70 which is >= 40, so should be high_variance
        assert test_005["pattern"] == "high_variance"

    @pytest.mark.anyio
    async def test_leaderboard_detects_hard_for_all_pattern(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that hard_for_all pattern is detected for low scoring tests."""
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "benchmark-suite"},
        )

        assert response.status_code == 200
        data = response.json()

        # test-004 has scores 35, 25, 15 - avg ~25 < 40, should be hard_for_all
        test_004 = next(t for t in data["tests"] if t["test_id"] == "test-004")
        assert test_004["pattern"] == "hard_for_all"

    @pytest.mark.anyio
    async def test_leaderboard_validates_min_agents(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that validation passes with empty agents filter."""
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "agents": [],
            },
        )

        # Empty agents list means "use all agents"
        # The endpoint should handle this gracefully
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_leaderboard_returns_correct_test_tags(
        self, client_with_data: AsyncClient
    ) -> None:
        """Test that test tags are returned correctly."""
        response = await client_with_data.get(
            "/api/leaderboard/matrix",
            params={"suite_name": "benchmark-suite"},
        )

        assert response.status_code == 200
        data = response.json()

        tests_by_id = {t["test_id"]: t for t in data["tests"]}

        assert tests_by_id["test-001"]["tags"] == ["smoke"]
        assert tests_by_id["test-002"]["tags"] == ["regression"]
        assert tests_by_id["test-003"]["tags"] == ["complex"]
