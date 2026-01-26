"""Test fixtures for comparison dashboard tests."""

from tests.fixtures.comparison.events import (
    SAMPLE_EVENTS,
    EventType,
    create_error_event,
    create_llm_request_event,
    create_progress_event,
    create_reasoning_event,
    create_tool_call_event,
    generate_realistic_event_sequence,
)
from tests.fixtures.comparison.factories import (
    AgentFactory,
    ArtifactFactory,
    EvaluationResultFactory,
    RunResultFactory,
    ScoreComponentFactory,
    SuiteExecutionFactory,
    TestExecutionFactory,
)
from tests.fixtures.comparison.leaderboard import (
    LeaderboardScenario,
    create_leaderboard_scenario,
    generate_leaderboard_data,
)

__all__ = [
    # Event generators
    "SAMPLE_EVENTS",
    "EventType",
    "create_tool_call_event",
    "create_llm_request_event",
    "create_reasoning_event",
    "create_error_event",
    "create_progress_event",
    "generate_realistic_event_sequence",
    # Factory functions
    "AgentFactory",
    "SuiteExecutionFactory",
    "TestExecutionFactory",
    "RunResultFactory",
    "ArtifactFactory",
    "EvaluationResultFactory",
    "ScoreComponentFactory",
    # Leaderboard
    "LeaderboardScenario",
    "create_leaderboard_scenario",
    "generate_leaderboard_data",
]
