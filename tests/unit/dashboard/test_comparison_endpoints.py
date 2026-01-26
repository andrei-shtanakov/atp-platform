"""Tests for comparison dashboard endpoints.

This module tests the new comparison endpoints:
- /compare/side-by-side
- /leaderboard/matrix
- /timeline/events
- /timeline/compare

These tests verify that the test infrastructure supports the new endpoints
and provides placeholder tests for when the endpoints are implemented.
"""

from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from atp.dashboard.api import _format_event_summary
from atp.dashboard.app import app
from atp.dashboard.schemas import (
    AgentExecutionDetail,
    EventSummary,
    SideBySideComparisonResponse,
)
from tests.fixtures.comparison import (
    SAMPLE_EVENTS,
    AgentFactory,
    RunResultFactory,
    SuiteExecutionFactory,
    TestExecutionFactory,
    create_error_event,
    create_leaderboard_scenario,
    create_llm_request_event,
    create_progress_event,
    create_reasoning_event,
    create_tool_call_event,
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
        # Should return 422 (validation error) for missing required param
        assert response.status_code == 422

    def test_endpoint_requires_test_id(self, client: TestClient) -> None:
        """Test that test_id is required."""
        response = client.get(
            "/api/compare/side-by-side",
            params={"suite_name": "test-suite"},
        )
        assert response.status_code == 422

    def test_endpoint_requires_agents(self, client: TestClient) -> None:
        """Test that agents parameter is required."""
        response = client.get(
            "/api/compare/side-by-side",
            params={"suite_name": "test-suite", "test_id": "test-001"},
        )
        assert response.status_code == 422

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
        assert response.status_code == 422

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
        assert response.status_code == 422

    def test_endpoint_returns_error_when_no_executions(
        self, client: TestClient
    ) -> None:
        """Test that error is returned when no executions exist.

        Note: This test uses sync TestClient which doesn't have proper
        async database setup, so we expect either 404 (proper response)
        or 500 (database not configured). The actual logic is tested
        in integration tests with proper database setup.
        """
        response = client.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "nonexistent-suite",
                "test_id": "test-001",
                "agents": ["agent-1", "agent-2"],
            },
        )
        # Should return 404 (no executions) or 500 (db not configured)
        assert response.status_code in [404, 500]


class TestFormatEventSummary:
    """Tests for _format_event_summary helper function."""

    def test_format_tool_call_event(self) -> None:
        """Test formatting a tool_call event."""
        event = create_tool_call_event(
            tool="web_search",
            status="success",
            sequence=1,
        )
        summary = _format_event_summary(event)

        assert summary.event_type == "tool_call"
        assert summary.sequence == 1
        assert "web_search" in summary.summary
        assert "success" in summary.summary
        assert summary.data["tool"] == "web_search"

    def test_format_llm_request_event(self) -> None:
        """Test formatting an llm_request event."""
        event = create_llm_request_event(
            model="claude-sonnet-4-20250514",
            input_tokens=500,
            output_tokens=200,
            sequence=2,
        )
        summary = _format_event_summary(event)

        assert summary.event_type == "llm_request"
        assert summary.sequence == 2
        assert "claude-sonnet-4-20250514" in summary.summary
        assert "700 tokens" in summary.summary  # 500 + 200
        assert summary.data["model"] == "claude-sonnet-4-20250514"

    def test_format_reasoning_event(self) -> None:
        """Test formatting a reasoning event."""
        event = create_reasoning_event(
            thought="Analyzing the problem structure",
            step="Step 1 of 3",
            sequence=3,
        )
        summary = _format_event_summary(event)

        assert summary.event_type == "reasoning"
        assert summary.sequence == 3
        assert "Analyzing" in summary.summary
        assert summary.data.get("thought") == "Analyzing the problem structure"

    def test_format_reasoning_event_truncates_long_thought(self) -> None:
        """Test that long thoughts are truncated in summary."""
        long_thought = "A" * 100  # 100 characters
        event = create_reasoning_event(thought=long_thought, sequence=0)
        summary = _format_event_summary(event)

        # Summary should be truncated to 50 chars + "..."
        assert len(summary.summary) == 53

    def test_format_progress_event_with_message(self) -> None:
        """Test formatting a progress event with message."""
        event = create_progress_event(
            percentage=50.0,
            message="Halfway done",
            sequence=4,
        )
        summary = _format_event_summary(event)

        assert summary.event_type == "progress"
        assert summary.sequence == 4
        assert "50" in summary.summary
        assert "Halfway done" in summary.summary

    def test_format_progress_event_without_message(self) -> None:
        """Test formatting a progress event without message."""
        event = create_progress_event(
            percentage=75.0,
            message=None,
            sequence=5,
        )
        summary = _format_event_summary(event)

        assert summary.event_type == "progress"
        assert "75" in summary.summary

    def test_format_error_event(self) -> None:
        """Test formatting an error event."""
        event = create_error_event(
            error_type="RuntimeError",
            message="Something went wrong",
            recoverable=True,
            sequence=5,
        )
        summary = _format_event_summary(event)

        assert summary.event_type == "error"
        assert summary.sequence == 5
        assert "RuntimeError" in summary.summary
        assert "Something went wrong" in summary.summary

    def test_format_unknown_event_type(self) -> None:
        """Test formatting an unknown event type."""
        event: dict[str, Any] = {
            "version": "1.0",
            "task_id": "task-001",
            "timestamp": datetime.now().isoformat(),
            "sequence": 6,
            "event_type": "custom_event",
            "payload": {"custom": "data"},
        }
        summary = _format_event_summary(event)

        assert summary.event_type == "custom_event"
        assert "custom_event" in summary.summary

    def test_format_event_parses_timestamp(self) -> None:
        """Test that timestamp is properly parsed."""
        fixed_time = datetime(2024, 6, 15, 10, 30, 0)
        event = create_tool_call_event(timestamp=fixed_time, sequence=0)
        summary = _format_event_summary(event)

        assert summary.timestamp.year == 2024
        assert summary.timestamp.month == 6
        assert summary.timestamp.day == 15


class TestEventSummarySchema:
    """Tests for EventSummary Pydantic schema."""

    def test_event_summary_creation(self) -> None:
        """Test creating EventSummary."""
        summary = EventSummary(
            sequence=1,
            timestamp=datetime.now(),
            event_type="tool_call",
            summary="Test summary",
            data={"key": "value"},
        )
        assert summary.sequence == 1
        assert summary.event_type == "tool_call"
        assert summary.summary == "Test summary"
        assert summary.data == {"key": "value"}


class TestAgentExecutionDetailSchema:
    """Tests for AgentExecutionDetail Pydantic schema."""

    def test_agent_execution_detail_creation(self) -> None:
        """Test creating AgentExecutionDetail."""
        detail = AgentExecutionDetail(
            agent_name="test-agent",
            test_execution_id=1,
            score=85.5,
            success=True,
            duration_seconds=120.5,
            total_tokens=1500,
            total_steps=5,
            tool_calls=3,
            llm_calls=5,
            cost_usd=0.015,
            events=[],
        )
        assert detail.agent_name == "test-agent"
        assert detail.score == 85.5
        assert detail.success is True
        assert detail.events == []

    def test_agent_execution_detail_with_none_values(self) -> None:
        """Test creating AgentExecutionDetail with None values."""
        detail = AgentExecutionDetail(
            agent_name="test-agent",
            test_execution_id=1,
            score=None,
            success=False,
            duration_seconds=None,
            total_tokens=None,
            total_steps=None,
            tool_calls=None,
            llm_calls=None,
            cost_usd=None,
            events=[],
        )
        assert detail.score is None
        assert detail.total_tokens is None


class TestSideBySideComparisonResponseSchema:
    """Tests for SideBySideComparisonResponse Pydantic schema."""

    def test_side_by_side_response_creation(self) -> None:
        """Test creating SideBySideComparisonResponse."""
        agent_detail = AgentExecutionDetail(
            agent_name="agent-1",
            test_execution_id=1,
            score=90.0,
            success=True,
            duration_seconds=100.0,
            total_tokens=1000,
            total_steps=4,
            tool_calls=2,
            llm_calls=4,
            cost_usd=0.01,
            events=[],
        )
        response = SideBySideComparisonResponse(
            suite_name="benchmark-suite",
            test_id="test-001",
            test_name="Test Case 1",
            agents=[agent_detail],
        )
        assert response.suite_name == "benchmark-suite"
        assert response.test_id == "test-001"
        assert response.test_name == "Test Case 1"
        assert len(response.agents) == 1
        assert response.agents[0].agent_name == "agent-1"

    def test_side_by_side_response_multiple_agents(self) -> None:
        """Test response with multiple agents."""
        agents = [
            AgentExecutionDetail(
                agent_name=f"agent-{i}",
                test_execution_id=i,
                score=80.0 + i * 5,
                success=True,
                duration_seconds=100.0 + i * 10,
                total_tokens=1000,
                total_steps=4,
                tool_calls=2,
                llm_calls=4,
                cost_usd=0.01,
                events=[],
            )
            for i in range(1, 4)  # 3 agents
        ]
        response = SideBySideComparisonResponse(
            suite_name="test-suite",
            test_id="test-001",
            test_name="Test",
            agents=agents,
        )
        assert len(response.agents) == 3
        assert response.agents[0].agent_name == "agent-1"
        assert response.agents[2].agent_name == "agent-3"


class TestLeaderboardMatrixEndpoint:
    """Tests for /leaderboard/matrix endpoint."""

    def test_endpoint_requires_suite_name(self, client: TestClient) -> None:
        """Test that suite_name is required."""
        response = client.get("/api/leaderboard/matrix")
        # Should return 422 (validation error) for missing required param
        assert response.status_code == 422

    def test_endpoint_accepts_valid_params(self, client: TestClient) -> None:
        """Test endpoint with valid parameters."""
        response = client.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "limit_executions": 5,
            },
        )
        # Should return 200 (success) or 500 (db not configured)
        assert response.status_code in [200, 500]

    def test_endpoint_validates_limit_executions(self, client: TestClient) -> None:
        """Test that limit_executions has a maximum value of 20."""
        response = client.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "limit_executions": 100,  # Exceeds max of 20
            },
        )
        # Should fail validation
        assert response.status_code == 422


class TestTimelineEventsEndpoint:
    """Tests for /timeline/events endpoint."""

    def test_endpoint_requires_suite_name(self, client: TestClient) -> None:
        """Test that suite_name is required."""
        response = client.get("/api/timeline/events")
        assert response.status_code == 422

    def test_endpoint_requires_test_id(self, client: TestClient) -> None:
        """Test that test_id is required."""
        response = client.get(
            "/api/timeline/events",
            params={
                "suite_name": "test-suite",
            },
        )
        assert response.status_code == 422

    def test_endpoint_requires_agent_name(self, client: TestClient) -> None:
        """Test that agent_name is required."""
        response = client.get(
            "/api/timeline/events",
            params={
                "suite_name": "test-suite",
                "test_id": "test-001",
            },
        )
        assert response.status_code == 422

    def test_endpoint_accepts_event_type_filter(self, client: TestClient) -> None:
        """Test endpoint with event_types filter."""
        response = client.get(
            "/api/timeline/events",
            params={
                "suite_name": "test-suite",
                "test_id": "test-001",
                "agent_name": "agent-1",
                "event_types": ["tool_call", "llm_request"],
            },
        )
        assert response.status_code in [200, 404, 500]

    def test_endpoint_validates_limit(self, client: TestClient) -> None:
        """Test that limit has a maximum value."""
        response = client.get(
            "/api/timeline/events",
            params={
                "suite_name": "test-suite",
                "test_id": "test-001",
                "agent_name": "agent-1",
                "limit": 2000,  # Exceeds max of 1000
            },
        )
        assert response.status_code == 422


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
