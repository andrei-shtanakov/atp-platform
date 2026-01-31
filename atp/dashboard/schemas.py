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


# ==================== Test Suite Management Schemas ====================


class ConstraintsCreate(BaseModel):
    """Schema for creating execution constraints."""

    max_steps: int | None = Field(None, description="Maximum number of steps allowed")
    max_tokens: int | None = Field(None, description="Maximum tokens allowed")
    timeout_seconds: int = Field(300, ge=1, description="Timeout in seconds")
    allowed_tools: list[str] | None = Field(
        None, description="List of allowed tools, None means all allowed"
    )
    budget_usd: float | None = Field(None, ge=0, description="Budget limit in USD")


class AssertionCreate(BaseModel):
    """Schema for creating an assertion."""

    type: str = Field(..., min_length=1, description="Assertion type")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Assertion configuration"
    )


class TaskCreate(BaseModel):
    """Schema for creating a task definition."""

    description: str = Field(
        ..., min_length=1, max_length=10000, description="Task description"
    )
    input_data: dict[str, Any] | None = Field(None, description="Optional input data")
    expected_artifacts: list[str] | None = Field(
        None, description="Expected output artifacts"
    )


class ScoringWeightsCreate(BaseModel):
    """Schema for scoring weights."""

    quality_weight: float = Field(0.4, ge=0.0, le=1.0)
    completeness_weight: float = Field(0.3, ge=0.0, le=1.0)
    efficiency_weight: float = Field(0.2, ge=0.0, le=1.0)
    cost_weight: float = Field(0.1, ge=0.0, le=1.0)


class TestCreateRequest(BaseModel):
    """Schema for creating a test within a suite."""

    id: str = Field(..., min_length=1, max_length=100, description="Test ID")
    name: str = Field(..., min_length=1, max_length=255, description="Test name")
    description: str | None = Field(None, max_length=2000, description="Description")
    tags: list[str] = Field(default_factory=list, description="Test tags")
    task: TaskCreate = Field(..., description="Task specification")
    constraints: ConstraintsCreate = Field(
        default_factory=ConstraintsCreate, description="Execution constraints"
    )
    assertions: list[AssertionCreate] = Field(
        default_factory=list, description="Test assertions"
    )
    scoring: ScoringWeightsCreate | None = Field(
        None, description="Optional scoring weights override"
    )


class TestDefaultsCreate(BaseModel):
    """Schema for test suite defaults."""

    runs_per_test: int = Field(1, ge=1, le=100, description="Number of runs per test")
    timeout_seconds: int = Field(300, ge=1, le=3600, description="Default timeout")
    scoring: ScoringWeightsCreate = Field(
        default_factory=ScoringWeightsCreate, description="Default scoring weights"
    )
    constraints: ConstraintsCreate | None = Field(
        None, description="Default constraints"
    )


class AgentConfigCreate(BaseModel):
    """Schema for agent configuration in suite."""

    name: str = Field(..., min_length=1, max_length=100, description="Agent name")
    type: str | None = Field(
        None, max_length=50, description="Agent type (http, container, etc.)"
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific configuration"
    )


class SuiteCreateRequest(BaseModel):
    """Schema for creating a new test suite definition."""

    name: str = Field(..., min_length=1, max_length=255, description="Suite name")
    version: str = Field("1.0", max_length=20, description="Suite version")
    description: str | None = Field(None, max_length=2000, description="Description")
    defaults: TestDefaultsCreate = Field(
        default_factory=TestDefaultsCreate, description="Default settings"
    )
    agents: list[AgentConfigCreate] = Field(
        default_factory=list, description="Agent configurations"
    )
    tests: list[TestCreateRequest] = Field(
        default_factory=list, description="Initial tests (optional)"
    )


class TestResponse(BaseModel):
    """Schema for test response in suite."""

    id: str
    name: str
    description: str | None
    tags: list[str]
    task: TaskCreate
    constraints: ConstraintsCreate
    assertions: list[AssertionCreate]
    scoring: ScoringWeightsCreate | None


class SuiteDefinitionResponse(BaseModel):
    """Schema for suite definition response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    version: str
    description: str | None
    defaults: TestDefaultsCreate
    agents: list[AgentConfigCreate]
    tests: list[TestResponse]
    created_at: datetime
    updated_at: datetime


class SuiteDefinitionSummary(BaseModel):
    """Summary of a suite definition."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    version: str
    description: str | None
    test_count: int
    agent_count: int
    created_at: datetime
    updated_at: datetime


class SuiteDefinitionList(BaseModel):
    """Paginated list of suite definitions."""

    total: int
    items: list[SuiteDefinitionSummary]
    limit: int
    offset: int


# ==================== Template Schemas ====================


class TemplateVariableInfo(BaseModel):
    """Information about a variable used in a template."""

    name: str
    description: str | None = None
    required: bool = True


class TemplateResponse(BaseModel):
    """Schema for template response."""

    name: str
    description: str
    category: str
    task_template: str
    default_constraints: ConstraintsCreate
    default_assertions: list[AssertionCreate]
    tags: list[str]
    variables: list[str]


class TemplateListResponse(BaseModel):
    """Response containing list of templates."""

    templates: list[TemplateResponse]
    categories: list[str]
    total: int


# ==================== YAML Export Schemas ====================


class YAMLExportResponse(BaseModel):
    """Response for YAML export."""

    yaml_content: str
    suite_name: str
    test_count: int


# Update forward references
SuiteExecutionDetail.model_rebuild()
