"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ==================== Agent Schemas ====================


class AgentCreate(BaseModel):
    """Schema for creating an agent."""

    name: str = Field(..., min_length=1, max_length=100)
    agent_type: str = Field(..., min_length=1, max_length=50)
    config: dict[str, Any] = Field(default_factory=dict)
    description: str | None = Field(None, max_length=1000)


class AgentUpdate(BaseModel):
    """Schema for updating an agent."""

    agent_type: str | None = Field(None, min_length=1, max_length=50)
    config: dict[str, Any] | None = None
    description: str | None = None


class AgentResponse(BaseModel):
    """Schema for agent response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    agent_type: str
    config: dict[str, Any]
    description: str | None
    created_at: datetime
    updated_at: datetime


# ==================== Suite Execution Schemas ====================


class SuiteExecutionSummary(BaseModel):
    """Summary of a suite execution."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    suite_name: str
    agent_id: int
    agent_name: str | None = None
    started_at: datetime
    completed_at: datetime | None
    duration_seconds: float | None
    runs_per_test: int
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    status: str
    error: str | None


class SuiteExecutionDetail(SuiteExecutionSummary):
    """Detailed suite execution with test results."""

    tests: list["TestExecutionSummary"] = Field(default_factory=list)


class SuiteExecutionList(BaseModel):
    """Paginated list of suite executions."""

    total: int
    items: list[SuiteExecutionSummary]
    limit: int
    offset: int


# ==================== Test Execution Schemas ====================


class TestExecutionSummary(BaseModel):
    """Summary of a test execution."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    suite_execution_id: int
    test_id: str
    test_name: str
    tags: list[str]
    started_at: datetime
    completed_at: datetime | None
    duration_seconds: float | None
    total_runs: int
    successful_runs: int
    success: bool
    score: float | None
    status: str
    error: str | None


class RunResultSummary(BaseModel):
    """Summary of a run result."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_number: int
    started_at: datetime
    completed_at: datetime | None
    duration_seconds: float | None
    response_status: str
    success: bool
    error: str | None
    total_tokens: int | None
    input_tokens: int | None
    output_tokens: int | None
    total_steps: int | None
    tool_calls: int | None
    llm_calls: int | None
    cost_usd: float | None


class EvaluationResultResponse(BaseModel):
    """Evaluation result response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    evaluator_name: str
    passed: bool
    score: float | None
    total_checks: int
    passed_checks: int
    failed_checks: int
    checks_json: list[dict[str, Any]]


class ScoreComponentResponse(BaseModel):
    """Score component response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    component_name: str
    raw_value: float | None
    normalized_value: float
    weight: float
    weighted_value: float
    details_json: dict[str, Any] | None


class TestExecutionDetail(TestExecutionSummary):
    """Detailed test execution with runs and evaluations."""

    runs: list[RunResultSummary] = Field(default_factory=list)
    evaluations: list[EvaluationResultResponse] = Field(default_factory=list)
    score_components: list[ScoreComponentResponse] = Field(default_factory=list)
    statistics: dict[str, Any] | None = None


class TestExecutionList(BaseModel):
    """Paginated list of test executions."""

    total: int
    items: list[TestExecutionSummary]
    limit: int
    offset: int


# ==================== Historical Trends Schemas ====================


class TrendDataPoint(BaseModel):
    """Single data point in a trend."""

    timestamp: datetime
    value: float
    execution_id: int


class TestTrend(BaseModel):
    """Trend data for a single test."""

    test_id: str
    test_name: str
    data_points: list[TrendDataPoint]
    metric: str  # score, duration, success_rate, etc.


class SuiteTrend(BaseModel):
    """Trend data for a suite."""

    suite_name: str
    agent_name: str
    data_points: list[TrendDataPoint]
    metric: str


class TrendResponse(BaseModel):
    """Response containing trend data."""

    suite_trends: list[SuiteTrend] = Field(default_factory=list)
    test_trends: list[TestTrend] = Field(default_factory=list)


# ==================== Side-by-Side Comparison Schemas ====================


class EventSummary(BaseModel):
    """Summary of a single event for timeline/comparison views."""

    sequence: int
    timestamp: datetime
    event_type: str  # tool_call, llm_request, reasoning, error, progress
    summary: str  # One-line description
    data: dict[str, Any]  # Full event data


class AgentExecutionDetail(BaseModel):
    """Detailed execution data for a single agent in side-by-side comparison."""

    agent_name: str
    test_execution_id: int
    score: float | None
    success: bool
    duration_seconds: float | None
    total_tokens: int | None
    total_steps: int | None
    tool_calls: int | None
    llm_calls: int | None
    cost_usd: float | None
    events: list[EventSummary]  # Ordered list of events


class SideBySideComparisonResponse(BaseModel):
    """Response for side-by-side agent comparison on a specific test."""

    suite_name: str
    test_id: str
    test_name: str
    agents: list[AgentExecutionDetail]


# ==================== Agent Comparison Schemas ====================


class AgentComparisonMetrics(BaseModel):
    """Metrics for agent comparison."""

    agent_name: str
    total_executions: int
    avg_success_rate: float
    avg_score: float | None
    avg_duration_seconds: float | None
    latest_success_rate: float | None
    latest_score: float | None


class TestComparisonMetrics(BaseModel):
    """Test-level metrics for agent comparison."""

    test_id: str
    test_name: str
    metrics_by_agent: dict[str, AgentComparisonMetrics]


class AgentComparisonResponse(BaseModel):
    """Response for agent comparison."""

    suite_name: str
    agents: list[AgentComparisonMetrics]
    tests: list[TestComparisonMetrics] = Field(default_factory=list)


# ==================== Authentication Schemas ====================


class UserCreate(BaseModel):
    """Schema for creating a user."""

    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """Schema for user response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    is_active: bool
    is_admin: bool
    created_at: datetime


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Data extracted from JWT token."""

    username: str | None = None
    user_id: int | None = None


# ==================== Leaderboard Matrix Schemas ====================


class TestScore(BaseModel):
    """Score data for a single test by a single agent."""

    score: float | None
    success: bool
    execution_count: int


class TestRow(BaseModel):
    """Row in the leaderboard matrix representing a test."""

    test_id: str
    test_name: str
    tags: list[str]
    scores_by_agent: dict[str, TestScore]
    avg_score: float | None
    difficulty: str  # easy, medium, hard, very_hard, unknown
    pattern: str | None  # hard_for_all, easy, etc.


class AgentColumn(BaseModel):
    """Column in the leaderboard matrix representing an agent."""

    agent_name: str
    avg_score: float | None
    pass_rate: float
    total_tokens: int
    total_cost: float | None
    rank: int


class LeaderboardMatrixResponse(BaseModel):
    """Response for leaderboard matrix view."""

    suite_name: str
    tests: list[TestRow]
    agents: list[AgentColumn]
    total_tests: int
    total_agents: int
    limit: int
    offset: int


# ==================== Timeline Events Schemas ====================


class TimelineEvent(BaseModel):
    """Event data for timeline visualization.

    Extends EventSummary with relative timing information for timeline rendering.
    """

    sequence: int
    timestamp: datetime
    event_type: str  # tool_call, llm_request, reasoning, error, progress
    summary: str  # One-line description
    data: dict[str, Any]  # Full event payload
    relative_time_ms: float  # Time since first event in milliseconds
    duration_ms: float | None  # Event duration if applicable (from payload)


class TimelineEventsResponse(BaseModel):
    """Response for timeline events API."""

    suite_name: str
    test_id: str
    test_name: str
    agent_name: str
    total_events: int  # Total count before limiting
    events: list[TimelineEvent]  # List of timeline events (max 1000)
    total_duration_ms: float | None  # Total duration from first to last event
    execution_id: int  # Test execution ID for reference


# ==================== Multi-Agent Timeline Schemas ====================


class AgentTimeline(BaseModel):
    """Timeline data for a single agent in multi-agent comparison.

    Contains all events for one agent with timing aligned to a common
    start time for visual comparison across agents.
    """

    agent_name: str
    test_execution_id: int
    start_time: datetime
    total_duration_ms: float  # Total duration from first to last event
    events: list[TimelineEvent]  # Ordered list of events with relative timing


class MultiTimelineResponse(BaseModel):
    """Response for multi-agent timeline comparison API.

    Returns aligned timelines for 2-3 agents on the same test,
    enabling visual comparison of execution strategies and timing.
    """

    suite_name: str
    test_id: str
    test_name: str
    timelines: list[AgentTimeline]  # Timeline for each agent (2-3)


# ==================== Dashboard Summary Schemas ====================


class DashboardSummary(BaseModel):
    """Summary statistics for dashboard home."""

    total_agents: int
    total_suites: int
    total_executions: int
    recent_success_rate: float
    recent_avg_score: float | None
    recent_executions: list[SuiteExecutionSummary]


# Update forward references
SuiteExecutionDetail.model_rebuild()
