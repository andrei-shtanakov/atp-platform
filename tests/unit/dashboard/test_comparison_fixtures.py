"""Tests for comparison dashboard fixtures and factories.

This module tests the test infrastructure itself to ensure fixtures
and factory functions work correctly for the comparison dashboard tests.
"""

from datetime import datetime

import pytest

from tests.fixtures.comparison import (
    LEADERBOARD_SCENARIOS,
    SAMPLE_EVENTS,
    AgentFactory,
    ArtifactFactory,
    EvaluationResultFactory,
    EventType,
    LeaderboardScenario,
    RunResultFactory,
    ScoreComponentFactory,
    SuiteExecutionFactory,
    TestExecutionFactory,
    create_error_event,
    create_leaderboard_scenario,
    create_llm_request_event,
    create_progress_event,
    create_reasoning_event,
    create_tool_call_event,
    generate_leaderboard_data,
    generate_realistic_event_sequence,
)
from tests.fixtures.comparison.factories import reset_all_factories


@pytest.fixture(autouse=True)
def reset_factories() -> None:
    """Reset all factory counters before each test."""
    reset_all_factories()


class TestEventCreation:
    """Tests for event creation functions."""

    def test_create_tool_call_event_minimal(self) -> None:
        """Test creating a minimal tool call event."""
        event = create_tool_call_event()
        assert event["task_id"] == "task-001"
        assert event["sequence"] == 0
        assert event["event_type"] == EventType.TOOL_CALL.value
        assert event["payload"]["tool"] == "web_search"
        assert event["payload"]["status"] == "success"
        assert "timestamp" in event

    def test_create_tool_call_event_full(self) -> None:
        """Test creating a full tool call event with all fields."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        event = create_tool_call_event(
            task_id="custom-task",
            sequence=5,
            tool="code_exec",
            input_data={"code": "print('hello')"},
            output_data={"stdout": "hello", "exit_code": 0},
            duration_ms=1500.5,
            status="success",
            timestamp=timestamp,
        )
        assert event["task_id"] == "custom-task"
        assert event["sequence"] == 5
        assert event["payload"]["tool"] == "code_exec"
        assert event["payload"]["input"]["code"] == "print('hello')"
        assert event["payload"]["output"]["stdout"] == "hello"
        assert event["payload"]["duration_ms"] == 1500.5

    def test_create_llm_request_event(self) -> None:
        """Test creating an LLM request event."""
        event = create_llm_request_event(
            task_id="test-task",
            sequence=10,
            model="gpt-4-turbo",
            input_tokens=1000,
            output_tokens=500,
            duration_ms=2500.0,
        )
        assert event["event_type"] == EventType.LLM_REQUEST.value
        assert event["payload"]["model"] == "gpt-4-turbo"
        assert event["payload"]["input_tokens"] == 1000
        assert event["payload"]["output_tokens"] == 500
        assert event["payload"]["duration_ms"] == 2500.0

    def test_create_llm_request_event_minimal(self) -> None:
        """Test creating an LLM request with only required fields."""
        event = create_llm_request_event(
            model="claude-3",
            input_tokens=None,
            output_tokens=None,
            duration_ms=None,
        )
        assert event["payload"]["model"] == "claude-3"
        assert "input_tokens" not in event["payload"]
        assert "output_tokens" not in event["payload"]
        assert "duration_ms" not in event["payload"]

    def test_create_reasoning_event(self) -> None:
        """Test creating a reasoning event."""
        event = create_reasoning_event(
            task_id="reason-task",
            sequence=3,
            thought="I need to analyze the data first",
            plan="1. Load data\n2. Process\n3. Report",
            step="Analysis phase",
        )
        assert event["event_type"] == EventType.REASONING.value
        assert event["payload"]["thought"] == "I need to analyze the data first"
        assert event["payload"]["plan"].startswith("1. Load data")
        assert event["payload"]["step"] == "Analysis phase"

    def test_create_reasoning_event_partial(self) -> None:
        """Test creating a reasoning event with only some fields."""
        event = create_reasoning_event(thought="Just a thought")
        assert event["payload"]["thought"] == "Just a thought"
        assert "plan" not in event["payload"]
        assert "step" not in event["payload"]

    def test_create_error_event(self) -> None:
        """Test creating an error event."""
        event = create_error_event(
            task_id="error-task",
            sequence=7,
            error_type="AuthenticationError",
            message="Invalid API key",
            recoverable=False,
        )
        assert event["event_type"] == EventType.ERROR.value
        assert event["payload"]["error_type"] == "AuthenticationError"
        assert event["payload"]["message"] == "Invalid API key"
        assert event["payload"]["recoverable"] is False

    def test_create_progress_event(self) -> None:
        """Test creating a progress event."""
        event = create_progress_event(
            task_id="progress-task",
            sequence=15,
            current_step=5,
            percentage=50.0,
            message="Halfway done",
        )
        assert event["event_type"] == EventType.PROGRESS.value
        assert event["payload"]["current_step"] == 5
        assert event["payload"]["percentage"] == 50.0
        assert event["payload"]["message"] == "Halfway done"


class TestRealisticEventSequence:
    """Tests for realistic event sequence generation."""

    def test_generate_realistic_sequence_basic(self) -> None:
        """Test generating a basic event sequence."""
        events = generate_realistic_event_sequence(
            task_id="seq-test",
            num_steps=3,
            include_error=False,
        )
        assert len(events) > 0
        # Should start with progress
        assert events[0]["event_type"] == EventType.PROGRESS.value
        # Should end with completion progress
        assert events[-1]["event_type"] == EventType.PROGRESS.value
        assert events[-1]["payload"]["percentage"] == 100.0

    def test_generate_realistic_sequence_with_error(self) -> None:
        """Test generating a sequence with an error."""
        events = generate_realistic_event_sequence(
            task_id="error-seq",
            num_steps=5,
            include_error=True,
        )
        error_events = [e for e in events if e["event_type"] == EventType.ERROR.value]
        assert len(error_events) == 1
        assert error_events[0]["payload"]["recoverable"] is True

    def test_generate_realistic_sequence_timestamps_increase(self) -> None:
        """Test that timestamps are monotonically increasing."""
        events = generate_realistic_event_sequence(num_steps=5)
        timestamps = [
            datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00").split("+")[0])
            for e in events
        ]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_generate_realistic_sequence_sequences_increase(self) -> None:
        """Test that sequence numbers are monotonically increasing."""
        events = generate_realistic_event_sequence(num_steps=5)
        sequences = [e["sequence"] for e in events]
        for i in range(1, len(sequences)):
            assert sequences[i] > sequences[i - 1]

    def test_generate_realistic_sequence_contains_all_types(self) -> None:
        """Test that sequence contains multiple event types."""
        events = generate_realistic_event_sequence(num_steps=5)
        event_types = {e["event_type"] for e in events}
        # Should have progress, reasoning, llm_request, tool_call
        assert EventType.PROGRESS.value in event_types
        assert EventType.REASONING.value in event_types
        assert EventType.LLM_REQUEST.value in event_types
        assert EventType.TOOL_CALL.value in event_types


class TestSampleEvents:
    """Tests for pre-built sample events."""

    def test_sample_events_simple_success(self) -> None:
        """Test simple_success sample events."""
        events = SAMPLE_EVENTS["simple_success"]
        assert len(events) > 0
        assert events[0]["task_id"] == "simple-success"
        # Should not contain errors
        error_events = [e for e in events if e["event_type"] == EventType.ERROR.value]
        assert len(error_events) == 0

    def test_sample_events_with_error(self) -> None:
        """Test with_error sample events."""
        events = SAMPLE_EVENTS["with_error"]
        error_events = [e for e in events if e["event_type"] == EventType.ERROR.value]
        assert len(error_events) > 0

    def test_sample_events_minimal(self) -> None:
        """Test minimal sample events."""
        events = SAMPLE_EVENTS["minimal"]
        assert len(events) == 3  # start, tool_call, done

    def test_sample_events_error_only(self) -> None:
        """Test error_only sample events."""
        events = SAMPLE_EVENTS["error_only"]
        assert len(events) == 2  # start, error
        assert events[1]["event_type"] == EventType.ERROR.value
        assert events[1]["payload"]["recoverable"] is False


class TestAgentFactory:
    """Tests for AgentFactory."""

    def test_create_agent_defaults(self) -> None:
        """Test creating an agent with defaults."""
        agent = AgentFactory.create()
        assert agent.name.startswith("test-agent-")
        assert agent.agent_type == "http"
        assert "endpoint" in agent.config

    def test_create_agent_custom(self) -> None:
        """Test creating an agent with custom values."""
        agent = AgentFactory.create(
            name="custom-agent",
            agent_type="docker",
            config={"image": "test:latest"},
            description="A test agent",
        )
        assert agent.name == "custom-agent"
        assert agent.agent_type == "docker"
        assert agent.config["image"] == "test:latest"
        assert agent.description == "A test agent"

    def test_create_batch(self) -> None:
        """Test creating multiple agents."""
        agents = AgentFactory.create_batch(count=3, prefix="batch")
        assert len(agents) == 3
        assert agents[0].name == "batch-1"
        assert agents[1].name == "batch-2"
        assert agents[2].name == "batch-3"


class TestSuiteExecutionFactory:
    """Tests for SuiteExecutionFactory."""

    def test_create_suite_execution_defaults(self) -> None:
        """Test creating a suite execution with defaults."""
        suite = SuiteExecutionFactory.create()
        assert suite.suite_name.startswith("test-suite-")
        assert suite.status == "completed"
        assert suite.total_tests == 10
        assert suite.success_rate == 0.8

    def test_create_suite_execution_custom(self) -> None:
        """Test creating a suite execution with custom values."""
        suite = SuiteExecutionFactory.create(
            suite_name="benchmark",
            agent_id=5,
            total_tests=20,
            passed_tests=15,
            failed_tests=5,
            status="completed",
        )
        assert suite.suite_name == "benchmark"
        assert suite.agent_id == 5
        assert suite.success_rate == 0.75


class TestTestExecutionFactory:
    """Tests for TestExecutionFactory."""

    def test_create_test_execution_defaults(self) -> None:
        """Test creating a test execution with defaults."""
        test_exec = TestExecutionFactory.create()
        assert test_exec.test_id.startswith("test-")
        assert test_exec.success is True
        assert test_exec.score == 85.0
        assert test_exec.status == "completed"

    def test_create_test_execution_failed(self) -> None:
        """Test creating a failed test execution."""
        test_exec = TestExecutionFactory.create(
            success=False,
            score=35.0,
            error="Test failed due to assertion error",
        )
        assert test_exec.success is False
        assert test_exec.score == 35.0
        assert test_exec.error is not None

    def test_create_with_runs(self) -> None:
        """Test creating a test execution with run results."""
        test_exec, runs = TestExecutionFactory.create_with_runs(
            test_id="run-test",
            num_runs=5,
            include_events=True,
            event_steps=3,
        )
        assert test_exec.test_id == "run-test"
        assert len(runs) == 5
        assert test_exec.total_runs == 5
        # Check events are populated
        for run in runs:
            assert run.events_json is not None
            assert len(run.events_json) > 0


class TestRunResultFactory:
    """Tests for RunResultFactory."""

    def test_create_run_result_defaults(self) -> None:
        """Test creating a run result with defaults."""
        run = RunResultFactory.create()
        assert run.run_number == 1
        assert run.success is True
        assert run.response_status == "completed"
        assert run.total_tokens == 1500

    def test_create_run_result_failed(self) -> None:
        """Test creating a failed run result."""
        run = RunResultFactory.create(success=False)
        assert run.success is False
        assert run.response_status == "failed"
        assert run.error is not None

    def test_create_with_events(self) -> None:
        """Test creating a run result with generated events."""
        run = RunResultFactory.create_with_events(
            run_number=1,
            num_steps=5,
            include_error=False,
        )
        assert run.events_json is not None
        assert len(run.events_json) > 0
        # Check metrics were calculated
        assert run.tool_calls is not None
        assert run.llm_calls is not None

    def test_create_with_events_and_error(self) -> None:
        """Test creating a run result with error events."""
        run = RunResultFactory.create_with_events(
            num_steps=5,
            include_error=True,
        )
        assert run.success is False
        assert run.events_json is not None
        error_events = [
            e for e in run.events_json if e["event_type"] == EventType.ERROR.value
        ]
        assert len(error_events) > 0


class TestArtifactFactory:
    """Tests for ArtifactFactory."""

    def test_create_file_artifact(self) -> None:
        """Test creating a file artifact."""
        artifact = ArtifactFactory.create(artifact_type="file")
        assert artifact.artifact_type == "file"
        assert artifact.path is not None
        assert artifact.path.startswith("output/")

    def test_create_structured_artifact(self) -> None:
        """Test creating a structured artifact."""
        artifact = ArtifactFactory.create(
            artifact_type="structured",
            data_json={"key": "value"},
        )
        assert artifact.artifact_type == "structured"
        assert artifact.name is not None
        assert artifact.data_json == {"key": "value"}


class TestEvaluationResultFactory:
    """Tests for EvaluationResultFactory."""

    def test_create_evaluation_result_passed(self) -> None:
        """Test creating a passing evaluation result."""
        result = EvaluationResultFactory.create(
            evaluator_name="artifact",
            passed=True,
            score=0.95,
        )
        assert result.evaluator_name == "artifact"
        assert result.passed is True
        assert result.score == 0.95
        assert result.checks_json is not None

    def test_create_evaluation_result_failed(self) -> None:
        """Test creating a failing evaluation result."""
        result = EvaluationResultFactory.create(
            passed=False,
            score=0.4,
            total_checks=5,
            passed_checks=2,
            failed_checks=3,
        )
        assert result.passed is False
        assert result.failed_checks == 3


class TestScoreComponentFactory:
    """Tests for ScoreComponentFactory."""

    def test_create_score_component(self) -> None:
        """Test creating a score component."""
        component = ScoreComponentFactory.create(
            component_name="quality",
            raw_value=0.85,
            normalized_value=0.85,
            weight=0.4,
        )
        assert component.component_name == "quality"
        assert component.weighted_value == 0.85 * 0.4

    def test_create_full_breakdown(self) -> None:
        """Test creating a full score breakdown."""
        components = ScoreComponentFactory.create_full_breakdown(
            test_execution_id=1,
            quality=0.9,
            completeness=0.8,
            efficiency=0.7,
            cost=0.85,
        )
        assert len(components) == 4
        names = {c.component_name for c in components}
        assert names == {"quality", "completeness", "efficiency", "cost"}


class TestLeaderboardScenario:
    """Tests for leaderboard scenario generation."""

    def test_create_leaderboard_scenario_basic(self) -> None:
        """Test creating a basic leaderboard scenario."""
        scenario = create_leaderboard_scenario(
            suite_name="test-suite",
            num_agents=3,
            num_tests=5,
            seed=42,
        )
        assert isinstance(scenario, LeaderboardScenario)
        assert len(scenario.agents) == 3
        assert len(scenario.suite_executions) == 3
        assert len(scenario.test_executions) == 15  # 3 agents * 5 tests

    def test_create_leaderboard_scenario_score_matrix(self) -> None:
        """Test that score matrix is properly populated."""
        scenario = create_leaderboard_scenario(
            num_agents=2,
            num_tests=3,
            seed=42,
        )
        # Check score matrix structure
        assert len(scenario.score_matrix) == 3  # 3 tests
        for test_id, agent_scores in scenario.score_matrix.items():
            assert len(agent_scores) == 2  # 2 agents
            for score in agent_scores.values():
                assert score is not None
                assert 0 <= score <= 100

    def test_create_leaderboard_scenario_with_events(self) -> None:
        """Test creating a scenario with events included."""
        scenario = create_leaderboard_scenario(
            num_agents=2,
            num_tests=2,
            include_events=True,
            event_steps=3,
            seed=42,
        )
        # Check that run results have events
        for run in scenario.run_results:
            assert run.events_json is not None
            assert len(run.events_json) > 0

    def test_create_leaderboard_scenario_without_events(self) -> None:
        """Test creating a scenario without events."""
        scenario = create_leaderboard_scenario(
            num_agents=2,
            num_tests=2,
            include_events=False,
            seed=42,
        )
        for run in scenario.run_results:
            assert run.events_json is None

    def test_create_leaderboard_scenario_deterministic(self) -> None:
        """Test that scenarios are deterministic with seed."""
        scenario1 = create_leaderboard_scenario(seed=123)
        reset_all_factories()
        scenario2 = create_leaderboard_scenario(seed=123)

        # Scores should match
        for test_id in scenario1.score_matrix:
            for agent in scenario1.score_matrix[test_id]:
                score1 = scenario1.score_matrix[test_id][agent]
                score2 = scenario2.score_matrix[test_id][agent]
                assert score1 == score2

    def test_generate_leaderboard_data_custom(self) -> None:
        """Test generating leaderboard data with custom configs."""
        scenario = generate_leaderboard_data(
            suite_name="custom-suite",
            agent_configs=[
                {"name": "expert", "skill": 0.95},
                {"name": "novice", "skill": 0.4},
            ],
            test_configs=[
                {"test_id": "easy-test", "difficulty": "easy"},
                {"test_id": "hard-test", "difficulty": "hard"},
            ],
            seed=42,
        )
        assert len(scenario.agents) == 2
        assert scenario.agents[0].name == "expert"
        assert scenario.agents[1].name == "novice"
        # Expert should generally score higher than novice
        expert_scores: list[float] = []
        novice_scores: list[float] = []
        for t in scenario.score_matrix:
            expert_score = scenario.score_matrix[t]["expert"]
            novice_score = scenario.score_matrix[t]["novice"]
            if expert_score is not None:
                expert_scores.append(expert_score)
            if novice_score is not None:
                novice_scores.append(novice_score)
        expert_avg = sum(expert_scores) / len(expert_scores)
        novice_avg = sum(novice_scores) / len(novice_scores)
        # With seed=42, expert should outscore novice
        assert expert_avg > novice_avg


class TestLeaderboardScenarios:
    """Tests for pre-built LEADERBOARD_SCENARIOS."""

    def test_leaderboard_scenarios_available(self) -> None:
        """Test that LEADERBOARD_SCENARIOS dict contains expected keys."""
        expected_keys = {
            "basic",
            "large",
            "hard_for_all",
            "easy_suite",
            "mixed_performance",
        }
        assert expected_keys == set(LEADERBOARD_SCENARIOS.keys())

    def test_basic_scenario_returns_valid_data(self) -> None:
        """Test that basic scenario creates valid leaderboard data."""
        scenario = LEADERBOARD_SCENARIOS["basic"]()
        assert isinstance(scenario, LeaderboardScenario)
        assert len(scenario.agents) == 3
        assert len(scenario.score_matrix) > 0

    def test_large_scenario_returns_valid_data(self) -> None:
        """Test that large scenario creates valid leaderboard data."""
        scenario = LEADERBOARD_SCENARIOS["large"]()
        assert isinstance(scenario, LeaderboardScenario)
        assert len(scenario.agents) == 5
        assert len(scenario.test_executions) > 0

    def test_hard_for_all_scenario_has_low_scores(self) -> None:
        """Test that hard_for_all scenario has generally lower scores."""
        scenario = LEADERBOARD_SCENARIOS["hard_for_all"]()
        all_scores: list[float] = []
        for test_scores in scenario.score_matrix.values():
            for score in test_scores.values():
                if score is not None:
                    all_scores.append(score)
        # With low skills and hard difficulty, average should be below 60
        avg_score = sum(all_scores) / len(all_scores)
        assert avg_score < 60

    def test_easy_suite_scenario_has_high_scores(self) -> None:
        """Test that easy_suite scenario has generally higher scores."""
        scenario = LEADERBOARD_SCENARIOS["easy_suite"]()
        all_scores: list[float] = []
        for test_scores in scenario.score_matrix.values():
            for score in test_scores.values():
                if score is not None:
                    all_scores.append(score)
        # With high skills and easy difficulty, average should be above 80
        avg_score = sum(all_scores) / len(all_scores)
        assert avg_score > 80

    def test_mixed_performance_scenario_shows_variation(self) -> None:
        """Test that mixed_performance scenario shows score variation."""
        scenario = LEADERBOARD_SCENARIOS["mixed_performance"]()
        # Get scores by agent
        agent_scores: dict[str, list[float]] = {}
        for test_scores in scenario.score_matrix.values():
            for agent_name, score in test_scores.items():
                if score is not None:
                    if agent_name not in agent_scores:
                        agent_scores[agent_name] = []
                    agent_scores[agent_name].append(score)

        # Calculate averages
        agent_averages = {
            name: sum(scores) / len(scores) for name, scores in agent_scores.items()
        }
        # Specialist should outperform beginner
        assert agent_averages.get("specialist", 0) > agent_averages.get("beginner", 100)


class TestEventTypeEnum:
    """Tests for EventType enum."""

    def test_event_type_values(self) -> None:
        """Test that EventType has expected values."""
        assert EventType.TOOL_CALL.value == "tool_call"
        assert EventType.LLM_REQUEST.value == "llm_request"
        assert EventType.REASONING.value == "reasoning"
        assert EventType.ERROR.value == "error"
        assert EventType.PROGRESS.value == "progress"

    def test_event_type_is_string(self) -> None:
        """Test that EventType values are strings."""
        for event_type in EventType:
            assert isinstance(event_type.value, str)
