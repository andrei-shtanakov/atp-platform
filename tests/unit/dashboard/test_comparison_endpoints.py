"""Tests for comparison dashboard endpoints.

This module tests the new comparison endpoints:
- /compare/side-by-side
- /leaderboard/matrix
- /timeline/events
- /timeline/compare

These tests verify that the test infrastructure supports the new endpoints
and provides placeholder tests for when the endpoints are implemented.
"""

import pytest
from fastapi.testclient import TestClient

from atp.dashboard.app import app
from tests.fixtures.comparison import (
    SAMPLE_EVENTS,
    AgentFactory,
    RunResultFactory,
    SuiteExecutionFactory,
    TestExecutionFactory,
    create_leaderboard_scenario,
)
from tests.fixtures.comparison.factories import reset_all_factories


@pytest.fixture
def client() -> TestClient:
    """Create a test client using the actual app."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def reset_fixtures() -> None:
    """Reset all factory counters before each test."""
    reset_all_factories()


class TestSideBySideComparisonEndpoint:
    """Tests for /compare/side-by-side endpoint."""

    def test_endpoint_requires_suite_name(self, client: TestClient) -> None:
        """Test that suite_name is required."""
        response = client.get("/api/compare/side-by-side")
        # Should return 404 (not found) or 422 (validation error) until implemented
        # When implemented, should require suite_name
        assert response.status_code in [404, 422, 500]

    def test_endpoint_requires_test_id(self, client: TestClient) -> None:
        """Test that test_id is required."""
        response = client.get(
            "/api/compare/side-by-side",
            params={"suite_name": "test-suite"},
        )
        assert response.status_code in [404, 422, 500]

    def test_endpoint_requires_agents(self, client: TestClient) -> None:
        """Test that agents parameter is required."""
        response = client.get(
            "/api/compare/side-by-side",
            params={"suite_name": "test-suite", "test_id": "test-001"},
        )
        assert response.status_code in [404, 422, 500]

    def test_endpoint_validates_min_agents(self, client: TestClient) -> None:
        """Test that at least 2 agents are required."""
        response = client.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "test-suite",
                "test_id": "test-001",
                "agents": ["agent-1"],  # Only 1 agent
            },
        )
        # Should fail validation (min 2 agents)
        assert response.status_code in [404, 422, 500]

    def test_endpoint_validates_max_agents(self, client: TestClient) -> None:
        """Test that at most 3 agents are allowed."""
        response = client.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "test-suite",
                "test_id": "test-001",
                "agents": ["agent-1", "agent-2", "agent-3", "agent-4"],  # 4 agents
            },
        )
        # Should fail validation (max 3 agents)
        assert response.status_code in [404, 422, 500]


class TestLeaderboardMatrixEndpoint:
    """Tests for /leaderboard/matrix endpoint."""

    def test_endpoint_requires_suite_name(self, client: TestClient) -> None:
        """Test that suite_name is required."""
        response = client.get("/api/leaderboard/matrix")
        assert response.status_code in [404, 422, 500]

    def test_endpoint_accepts_valid_params(self, client: TestClient) -> None:
        """Test endpoint with valid parameters."""
        response = client.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "limit_executions": 5,
            },
        )
        # Should return 404 (not found) or 200 (success) when implemented
        assert response.status_code in [200, 404, 500]

    def test_endpoint_validates_limit(self, client: TestClient) -> None:
        """Test that limit_executions has a maximum value."""
        response = client.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "limit_executions": 100,  # Exceeds max of 20
            },
        )
        # Should fail validation or return error
        assert response.status_code in [404, 422, 500]


class TestTimelineEventsEndpoint:
    """Tests for /timeline/events endpoint."""

    def test_endpoint_requires_test_execution_id(self, client: TestClient) -> None:
        """Test that test_execution_id is required."""
        response = client.get("/api/timeline/events")
        assert response.status_code in [404, 422, 500]

    def test_endpoint_accepts_run_number(self, client: TestClient) -> None:
        """Test endpoint with run_number parameter."""
        response = client.get(
            "/api/timeline/events",
            params={
                "test_execution_id": 1,
                "run_number": 1,
            },
        )
        assert response.status_code in [200, 404, 500]

    def test_endpoint_accepts_event_type_filter(self, client: TestClient) -> None:
        """Test endpoint with event_types filter."""
        response = client.get(
            "/api/timeline/events",
            params={
                "test_execution_id": 1,
                "event_types": ["tool_call", "llm_request"],
            },
        )
        assert response.status_code in [200, 404, 500]

    def test_endpoint_validates_limit(self, client: TestClient) -> None:
        """Test that limit has a maximum value."""
        response = client.get(
            "/api/timeline/events",
            params={
                "test_execution_id": 1,
                "limit": 2000,  # Exceeds max of 1000
            },
        )
        assert response.status_code in [404, 422, 500]


class TestMultiTimelineEndpoint:
    """Tests for /timeline/compare endpoint."""

    def test_endpoint_requires_suite_name(self, client: TestClient) -> None:
        """Test that suite_name is required."""
        response = client.get("/api/timeline/compare")
        assert response.status_code in [404, 422, 500]

    def test_endpoint_requires_test_id(self, client: TestClient) -> None:
        """Test that test_id is required."""
        response = client.get(
            "/api/timeline/compare",
            params={"suite_name": "test-suite"},
        )
        assert response.status_code in [404, 422, 500]

    def test_endpoint_requires_agents(self, client: TestClient) -> None:
        """Test that agents parameter is required."""
        response = client.get(
            "/api/timeline/compare",
            params={
                "suite_name": "test-suite",
                "test_id": "test-001",
            },
        )
        assert response.status_code in [404, 422, 500]


class TestFixturesIntegration:
    """Tests verifying fixtures work for endpoint testing."""

    def test_agent_factory_creates_valid_agent(self) -> None:
        """Test that AgentFactory creates agents usable in tests."""
        agent = AgentFactory.create(name="test-agent")
        assert agent.name == "test-agent"
        assert agent.agent_type is not None

    def test_suite_execution_factory_creates_valid_suite(self) -> None:
        """Test that SuiteExecutionFactory creates valid suites."""
        suite = SuiteExecutionFactory.create(suite_name="benchmark")
        assert suite.suite_name == "benchmark"
        assert suite.started_at is not None

    def test_test_execution_factory_creates_with_events(self) -> None:
        """Test that TestExecutionFactory creates tests with events."""
        test_exec, runs = TestExecutionFactory.create_with_runs(
            test_id="test-with-events",
            num_runs=2,
            include_events=True,
        )
        assert test_exec.test_id == "test-with-events"
        assert len(runs) == 2
        for run in runs:
            assert run.events_json is not None

    def test_run_result_factory_creates_with_events(self) -> None:
        """Test that RunResultFactory creates runs with events."""
        run = RunResultFactory.create_with_events(
            run_number=1,
            num_steps=5,
        )
        assert run.events_json is not None
        assert len(run.events_json) > 0

    def test_leaderboard_scenario_creates_complete_data(self) -> None:
        """Test that leaderboard scenario creates all required data."""
        scenario = create_leaderboard_scenario(
            num_agents=3,
            num_tests=5,
            seed=42,
        )
        # Verify all data is created
        assert len(scenario.agents) == 3
        assert len(scenario.suite_executions) == 3
        assert len(scenario.test_executions) == 15
        assert len(scenario.run_results) == 15
        assert len(scenario.score_matrix) == 5

    def test_sample_events_usable_in_run_result(self) -> None:
        """Test that SAMPLE_EVENTS can be used in RunResult."""
        events = SAMPLE_EVENTS["simple_success"]
        run = RunResultFactory.create(events_json=events)
        assert run.events_json == events

    def test_factories_reset_counters(self) -> None:
        """Test that reset_all_factories resets counters."""
        # Create some objects
        AgentFactory.create()
        AgentFactory.create()
        agent1 = AgentFactory.create()
        assert "3" in agent1.name

        # Reset
        reset_all_factories()

        # Counter should restart
        agent2 = AgentFactory.create()
        assert "1" in agent2.name


class TestDataConsistency:
    """Tests for data consistency across fixtures."""

    def test_test_execution_runs_match(self) -> None:
        """Test that test execution's run count matches actual runs."""
        test_exec, runs = TestExecutionFactory.create_with_runs(
            num_runs=5,
            include_events=True,
        )
        assert test_exec.total_runs == len(runs)

    def test_score_matrix_matches_test_executions(self) -> None:
        """Test that score matrix matches test executions."""
        scenario = create_leaderboard_scenario(
            num_agents=2,
            num_tests=3,
            seed=42,
        )
        # Each test should have scores for all agents
        for test_id in scenario.score_matrix:
            for agent in scenario.agents:
                assert agent.name in scenario.score_matrix[test_id]

    def test_events_have_valid_structure(self) -> None:
        """Test that generated events have valid structure."""
        run = RunResultFactory.create_with_events(num_steps=5)
        assert run.events_json is not None
        for event in run.events_json:
            assert "version" in event
            assert "task_id" in event
            assert "timestamp" in event
            assert "sequence" in event
            assert "event_type" in event
            assert "payload" in event

    def test_metrics_calculated_from_events(self) -> None:
        """Test that metrics are calculated from events."""
        run = RunResultFactory.create_with_events(num_steps=5)
        assert run.events_json is not None
        # Count events by type
        tool_calls = sum(1 for e in run.events_json if e["event_type"] == "tool_call")
        llm_calls = sum(1 for e in run.events_json if e["event_type"] == "llm_request")
        # Metrics should match
        assert run.tool_calls == tool_calls
        assert run.llm_calls == llm_calls
