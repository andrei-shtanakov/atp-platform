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


class SuiteUpdateRequest(BaseModel):
    """Schema for updating a test suite definition."""

    name: str | None = Field(
        None, min_length=1, max_length=255, description="Suite name"
    )
    version: str | None = Field(None, max_length=20, description="Suite version")
    description: str | None = Field(None, max_length=2000, description="Description")
    defaults: TestDefaultsCreate | None = Field(None, description="Default settings")
    agents: list[AgentConfigCreate] | None = Field(
        None, description="Agent configurations"
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


# ==================== Cost Analytics Schemas ====================


class CostRecordResponse(BaseModel):
    """Schema for cost record response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    test_id: str | None
    suite_id: str | None
    agent_name: str | None
    metadata: dict[str, Any] | None = None


class CostRecordList(BaseModel):
    """Paginated list of cost records."""

    total: int
    items: list[CostRecordResponse]
    limit: int
    offset: int


class CostBreakdownItem(BaseModel):
    """Single item in a cost breakdown (by provider, model, agent, or suite)."""

    name: str
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    record_count: int
    percentage: float = 0.0


class CostTrendPoint(BaseModel):
    """Single data point for cost trend over time."""

    date: str
    total_cost: float
    total_tokens: int
    record_count: int


class CostSummaryResponse(BaseModel):
    """Cost summary with breakdowns and trends."""

    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    total_records: int
    by_provider: list[CostBreakdownItem]
    by_model: list[CostBreakdownItem]
    by_agent: list[CostBreakdownItem]
    daily_trend: list[CostTrendPoint]


# ==================== Budget Schemas ====================


class BudgetCreate(BaseModel):
    """Schema for creating a budget."""

    name: str = Field(..., min_length=1, max_length=100)
    period: str = Field(..., pattern="^(daily|weekly|monthly)$")
    limit_usd: float = Field(..., gt=0)
    alert_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    scope: dict[str, Any] | None = None
    alert_channels: list[str] | None = None
    description: str | None = Field(None, max_length=1000)


class BudgetUpdate(BaseModel):
    """Schema for updating a budget."""

    name: str | None = Field(None, min_length=1, max_length=100)
    period: str | None = Field(None, pattern="^(daily|weekly|monthly)$")
    limit_usd: float | None = Field(None, gt=0)
    alert_threshold: float | None = Field(None, ge=0.0, le=1.0)
    scope: dict[str, Any] | None = None
    alert_channels: list[str] | None = None
    description: str | None = None
    is_active: bool | None = None


class BudgetResponse(BaseModel):
    """Schema for budget response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    period: str
    limit_usd: float
    alert_threshold: float
    scope: dict[str, Any] | None = None
    alert_channels: list[str] | None = None
    description: str | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class BudgetUsageResponse(BaseModel):
    """Schema for budget usage response."""

    budget_id: int
    budget_name: str
    period: str
    period_start: datetime
    spent: float
    limit: float
    remaining: float
    percentage: float
    is_over_threshold: bool
    is_over_limit: bool


class BudgetWithUsageResponse(BudgetResponse):
    """Budget response including current usage."""

    usage: BudgetUsageResponse | None = None


class BudgetList(BaseModel):
    """List of budgets with optional usage."""

    items: list[BudgetWithUsageResponse]
    total: int


# ==================== Advanced Analytics Schemas ====================


class TrendDataPointResponse(BaseModel):
    """Single data point in a trend."""

    date: str
    value: float
    count: int = 1


class TrendAnalysisResponse(BaseModel):
    """Trend analysis result."""

    metric: str
    direction: str
    change_percent: float
    start_value: float
    end_value: float
    average_value: float
    std_deviation: float
    data_points: list[TrendDataPointResponse]
    period_days: int
    confidence: float


class ScoreTrendsResponse(BaseModel):
    """Response for score trends API."""

    suite_name: str | None = None
    agent_name: str | None = None
    trends: list[TrendAnalysisResponse]
    period_start: datetime
    period_end: datetime


class AnomalyResultResponse(BaseModel):
    """Detected anomaly response."""

    anomaly_type: str
    timestamp: datetime
    metric_name: str
    expected_value: float
    actual_value: float
    deviation_sigma: float
    test_id: str | None = None
    suite_id: str | None = None
    agent_name: str | None = None
    severity: str
    details: dict[str, Any]


class AnomalyDetectionResponseSchema(BaseModel):
    """Response for anomaly detection API."""

    anomalies: list[AnomalyResultResponse]
    total_records_analyzed: int
    anomaly_rate: float
    period_start: datetime
    period_end: datetime


class CorrelationResultResponse(BaseModel):
    """Correlation result response."""

    factor_x: str
    factor_y: str
    correlation_coefficient: float
    strength: str
    sample_size: int
    p_value: float | None = None
    details: dict[str, Any]


class CorrelationAnalysisResponseSchema(BaseModel):
    """Response for correlation analysis API."""

    correlations: list[CorrelationResultResponse]
    sample_size: int
    factors_analyzed: list[str]


class ScheduledReportCreate(BaseModel):
    """Schema for creating a scheduled report."""

    name: str = Field(..., min_length=1, max_length=100)
    frequency: str = Field(..., pattern="^(daily|weekly|monthly)$")
    recipients: list[str] = Field(default_factory=list)
    include_trends: bool = True
    include_anomalies: bool = True
    include_correlations: bool = False
    filters: dict[str, Any] = Field(default_factory=dict)


class ScheduledReportUpdate(BaseModel):
    """Schema for updating a scheduled report."""

    name: str | None = Field(None, min_length=1, max_length=100)
    frequency: str | None = Field(None, pattern="^(daily|weekly|monthly)$")
    recipients: list[str] | None = None
    include_trends: bool | None = None
    include_anomalies: bool | None = None
    include_correlations: bool | None = None
    filters: dict[str, Any] | None = None
    is_active: bool | None = None


class ScheduledReportResponse(BaseModel):
    """Response for scheduled report."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    frequency: str
    recipients: list[str]
    include_trends: bool
    include_anomalies: bool
    include_correlations: bool
    filters: dict[str, Any]
    is_active: bool
    last_run: datetime | None = None
    next_run: datetime | None = None
    created_at: datetime | None = None


class ScheduledReportList(BaseModel):
    """List of scheduled reports."""

    items: list[ScheduledReportResponse]
    total: int


class ExportRequest(BaseModel):
    """Request for data export."""

    format: str = Field(default="csv", pattern="^(csv|excel)$")
    suite_name: str | None = None
    agent_name: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    include_runs: bool = False
    include_trends: bool = True
    include_anomalies: bool = True


# ==================== A/B Testing Schemas ====================


class VariantCreate(BaseModel):
    """Schema for creating an experiment variant."""

    name: str = Field(..., min_length=1, max_length=100)
    agent_name: str = Field(..., min_length=1, max_length=100)
    traffic_weight: float = Field(default=50.0, ge=0.0, le=100.0)
    description: str | None = Field(None, max_length=500)


class MetricConfigCreate(BaseModel):
    """Schema for metric configuration."""

    metric_type: str = Field(..., pattern="^(score|success_rate|duration|cost|tokens)$")
    is_primary: bool = False
    minimize: bool = False
    min_effect_size: float = Field(default=0.05, ge=0.0, le=1.0)


class RollbackConfigCreate(BaseModel):
    """Schema for rollback configuration."""

    enabled: bool = True
    degradation_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    min_samples_before_rollback: int = Field(default=30, ge=10)
    consecutive_checks: int = Field(default=3, ge=1)


class ExperimentCreate(BaseModel):
    """Schema for creating an A/B experiment."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=2000)
    suite_name: str = Field(..., min_length=1, max_length=255)
    test_ids: list[str] | None = None
    control_variant: VariantCreate
    treatment_variant: VariantCreate
    metrics: list[MetricConfigCreate] = Field(
        default_factory=lambda: [
            MetricConfigCreate(metric_type="score", is_primary=True)
        ]
    )
    rollback: RollbackConfigCreate = Field(default_factory=RollbackConfigCreate)
    min_sample_size: int = Field(default=30, ge=10)
    max_sample_size: int | None = None
    max_duration_days: int | None = None
    significance_level: float = Field(default=0.05, ge=0.01, le=0.10)


class ExperimentUpdate(BaseModel):
    """Schema for updating an experiment."""

    description: str | None = None
    rollback: RollbackConfigCreate | None = None
    max_sample_size: int | None = None
    max_duration_days: int | None = None


class VariantResponse(BaseModel):
    """Schema for variant response."""

    name: str
    variant_type: str
    agent_name: str
    traffic_weight: float
    description: str | None = None


class VariantMetricsResponse(BaseModel):
    """Schema for variant metrics response."""

    variant_name: str
    sample_size: int
    mean: float
    std: float
    min_value: float | None = None
    max_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None


class StatisticalResultResponse(BaseModel):
    """Schema for statistical result response."""

    metric_type: str
    control_metrics: VariantMetricsResponse
    treatment_metrics: VariantMetricsResponse
    t_statistic: float
    p_value: float
    is_significant: bool
    effect_size: float
    relative_change: float
    confidence_level: float
    winner: str


class ExperimentResponse(BaseModel):
    """Schema for experiment response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    suite_name: str
    test_ids: list[str] | None
    status: str
    control_variant: VariantResponse
    treatment_variant: VariantResponse
    metrics: list[MetricConfigCreate]
    rollback: RollbackConfigCreate
    min_sample_size: int
    max_sample_size: int | None
    max_duration_days: int | None
    significance_level: float
    control_sample_size: int
    treatment_sample_size: int
    winner: str | None
    rollback_triggered: bool
    created_at: datetime
    started_at: datetime | None
    concluded_at: datetime | None
    conclusion_reason: str | None


class ExperimentSummary(BaseModel):
    """Summary of an experiment."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    suite_name: str
    status: str
    control_variant_name: str
    treatment_variant_name: str
    control_sample_size: int
    treatment_sample_size: int
    winner: str | None
    created_at: datetime
    started_at: datetime | None
    concluded_at: datetime | None


class ExperimentList(BaseModel):
    """List of experiments."""

    items: list[ExperimentSummary]
    total: int


class ExperimentReportResponse(BaseModel):
    """Schema for experiment report response."""

    experiment: ExperimentResponse
    statistical_results: list[StatisticalResultResponse]
    recommendation: str
    summary: dict[str, Any]


class ObservationCreate(BaseModel):
    """Schema for creating an experiment observation."""

    variant_name: str = Field(..., min_length=1, max_length=100)
    test_id: str = Field(..., min_length=1, max_length=255)
    run_id: str = Field(..., min_length=1, max_length=255)
    score: float | None = None
    success: bool = False
    duration_seconds: float | None = None
    cost_usd: float | None = None
    tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationResponse(BaseModel):
    """Schema for observation response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    experiment_id: int
    variant_name: str
    test_id: str
    run_id: str
    timestamp: datetime
    score: float | None
    success: bool
    duration_seconds: float | None
    cost_usd: float | None
    tokens: int | None


class TrafficAssignmentResponse(BaseModel):
    """Schema for traffic assignment response."""

    experiment_id: int
    variant_name: str
    agent_name: str
    run_id: str


# ==================== Public Leaderboard Schemas ====================


class BenchmarkCategoryCreate(BaseModel):
    """Schema for creating a benchmark category."""

    name: str = Field(..., min_length=1, max_length=100)
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9-]+$")
    description: str | None = Field(None, max_length=2000)
    icon: str | None = Field(None, max_length=50)
    display_order: int = Field(default=0)
    parent_id: int | None = None
    min_submissions_for_ranking: int = Field(default=1, ge=1)


class BenchmarkCategoryUpdate(BaseModel):
    """Schema for updating a benchmark category."""

    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    icon: str | None = None
    display_order: int | None = None
    is_active: bool | None = None
    min_submissions_for_ranking: int | None = Field(None, ge=1)


class BenchmarkCategoryResponse(BaseModel):
    """Schema for benchmark category response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    description: str | None
    icon: str | None
    display_order: int
    parent_id: int | None
    is_active: bool
    min_submissions_for_ranking: int
    created_at: datetime
    updated_at: datetime


class BenchmarkCategoryList(BaseModel):
    """List of benchmark categories."""

    items: list[BenchmarkCategoryResponse]
    total: int


class AgentProfileCreate(BaseModel):
    """Schema for creating an agent profile."""

    agent_id: int
    display_name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=2000)
    website_url: str | None = Field(None, max_length=500)
    repository_url: str | None = Field(None, max_length=500)
    avatar_url: str | None = Field(None, max_length=500)
    tags: list[str] = Field(default_factory=list)
    is_public: bool = True


class AgentProfileUpdate(BaseModel):
    """Schema for updating an agent profile."""

    display_name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    website_url: str | None = None
    repository_url: str | None = None
    avatar_url: str | None = None
    tags: list[str] | None = None
    is_public: bool | None = None


class AgentProfileResponse(BaseModel):
    """Schema for agent profile response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    agent_id: int
    display_name: str
    description: str | None
    website_url: str | None
    repository_url: str | None
    avatar_url: str | None
    tags: list[str]
    is_verified: bool
    verification_badges: list[str]
    is_public: bool
    total_submissions: int
    best_overall_score: float | None
    best_overall_rank: int | None
    created_at: datetime
    updated_at: datetime


class AgentProfileSummary(BaseModel):
    """Summary of an agent profile for leaderboard display."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    display_name: str
    avatar_url: str | None
    is_verified: bool
    verification_badges: list[str]
    total_submissions: int
    best_overall_score: float | None
    best_overall_rank: int | None


class AgentProfileList(BaseModel):
    """Paginated list of agent profiles."""

    items: list[AgentProfileResponse]
    total: int
    limit: int
    offset: int


class PublishResultRequest(BaseModel):
    """Schema for publishing a result to the leaderboard."""

    suite_execution_id: int
    category: str = Field(..., min_length=1, max_length=100)


class PublishedResultResponse(BaseModel):
    """Schema for published result response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    suite_execution_id: int
    agent_profile_id: int
    category: str
    score: float
    success_rate: float
    total_tests: int
    passed_tests: int
    total_tokens: int | None
    total_cost_usd: float | None
    duration_seconds: float | None
    is_verified: bool
    verified_at: datetime | None
    published_at: datetime
    updated_at: datetime


class PublishedResultWithProfile(PublishedResultResponse):
    """Published result with agent profile information."""

    agent_profile: AgentProfileSummary


class PublishedResultList(BaseModel):
    """Paginated list of published results."""

    items: list[PublishedResultWithProfile]
    total: int
    limit: int
    offset: int


class LeaderboardEntry(BaseModel):
    """Single entry in the public leaderboard."""

    rank: int
    agent_profile: AgentProfileSummary
    score: float
    success_rate: float
    total_tests: int
    passed_tests: int
    total_tokens: int | None
    total_cost_usd: float | None
    duration_seconds: float | None
    is_verified: bool
    published_at: datetime
    # Historical trend data (last 5 submissions)
    score_history: list[float] = Field(default_factory=list)
    trend: str | None = None  # "up", "down", "stable", or None


class PublicLeaderboardResponse(BaseModel):
    """Response for public leaderboard."""

    category: BenchmarkCategoryResponse
    entries: list[LeaderboardEntry]
    total_entries: int
    last_updated: datetime
    limit: int
    offset: int


class LeaderboardTrendPoint(BaseModel):
    """Single point in a leaderboard trend."""

    timestamp: datetime
    rank: int
    score: float


class AgentLeaderboardHistory(BaseModel):
    """Historical leaderboard data for an agent."""

    agent_profile: AgentProfileSummary
    category: str
    data_points: list[LeaderboardTrendPoint]


class VerificationRequest(BaseModel):
    """Schema for requesting verification."""

    published_result_id: int
    verification_type: str = Field(..., pattern=r"^(official|reproducible|community)$")
    notes: str | None = Field(None, max_length=1000)


class VerificationBadgeRequest(BaseModel):
    """Schema for adding a verification badge."""

    agent_profile_id: int
    badge: str = Field(
        ..., pattern=r"^(official|reproducible|open_source|community_verified)$"
    )


# ==================== Test Suite Marketplace Schemas (TASK-803) ====================


class MarketplaceSuiteCreate(BaseModel):
    """Schema for publishing a test suite to the marketplace."""

    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(
        ..., min_length=1, max_length=255, pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$"
    )
    description: str | None = Field(None, max_length=10000)
    short_description: str | None = Field(None, max_length=500)
    category: str = Field(default="general", max_length=100)
    tags: list[str] = Field(default_factory=list)
    license_type: str = Field(default="MIT", max_length=50)
    license_url: str | None = Field(None, max_length=500)
    source_type: str = Field(default="local", pattern=r"^(local|github)$")
    github_url: str | None = Field(None, max_length=500)
    github_branch: str | None = Field(None, max_length=100)
    github_path: str | None = Field(None, max_length=500)
    # Initial version content
    version: str = Field(default="1.0.0", max_length=20)
    changelog: str | None = Field(None, max_length=5000)
    suite_content: dict[str, Any] = Field(default_factory=dict)


class MarketplaceSuiteUpdate(BaseModel):
    """Schema for updating a marketplace suite."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    short_description: str | None = Field(None, max_length=500)
    category: str | None = Field(None, max_length=100)
    tags: list[str] | None = None
    license_type: str | None = Field(None, max_length=50)
    license_url: str | None = None
    github_url: str | None = None
    github_branch: str | None = None
    github_path: str | None = None
    is_published: bool | None = None


class MarketplaceSuiteResponse(BaseModel):
    """Schema for marketplace suite response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    description: str | None
    short_description: str | None
    publisher_id: int
    publisher_name: str
    category: str
    tags: list[str]
    license_type: str
    license_url: str | None
    source_type: str
    github_url: str | None
    github_branch: str | None
    github_path: str | None
    is_published: bool
    is_featured: bool
    is_verified: bool
    total_downloads: int
    total_installs: int
    average_rating: float | None
    total_ratings: int
    latest_version: str
    created_at: datetime
    updated_at: datetime
    published_at: datetime | None


class MarketplaceSuiteSummary(BaseModel):
    """Summary of a marketplace suite for list views."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    short_description: str | None
    publisher_name: str
    category: str
    tags: list[str]
    license_type: str
    is_published: bool
    is_featured: bool
    is_verified: bool
    total_downloads: int
    average_rating: float | None
    total_ratings: int
    latest_version: str
    updated_at: datetime


class MarketplaceSuiteList(BaseModel):
    """Paginated list of marketplace suites."""

    items: list[MarketplaceSuiteSummary]
    total: int
    limit: int
    offset: int


class MarketplaceSuiteDetail(MarketplaceSuiteResponse):
    """Detailed marketplace suite with versions and reviews."""

    versions: list["MarketplaceSuiteVersionResponse"] = Field(default_factory=list)
    recent_reviews: list["MarketplaceSuiteReviewResponse"] = Field(default_factory=list)
    test_count: int = 0


# ==================== Marketplace Version Schemas ====================


class MarketplaceSuiteVersionCreate(BaseModel):
    """Schema for creating a new version."""

    version: str = Field(..., min_length=1, max_length=20)
    changelog: str | None = Field(None, max_length=5000)
    breaking_changes: bool = False
    suite_content: dict[str, Any] = Field(default_factory=dict)


class MarketplaceSuiteVersionResponse(BaseModel):
    """Schema for version response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    marketplace_suite_id: int
    version: str
    version_major: int
    version_minor: int
    version_patch: int
    changelog: str | None
    breaking_changes: bool
    test_count: int
    file_size_bytes: int | None
    downloads: int
    is_latest: bool
    is_deprecated: bool
    deprecation_message: str | None
    created_at: datetime
    released_at: datetime | None


class MarketplaceSuiteVersionList(BaseModel):
    """List of versions."""

    items: list[MarketplaceSuiteVersionResponse]
    total: int


class MarketplaceSuiteVersionUpdate(BaseModel):
    """Schema for updating a version."""

    changelog: str | None = None
    is_deprecated: bool | None = None
    deprecation_message: str | None = None


# ==================== Marketplace Review Schemas ====================


class MarketplaceSuiteReviewCreate(BaseModel):
    """Schema for creating a review."""

    rating: int = Field(..., ge=1, le=5)
    title: str | None = Field(None, max_length=200)
    content: str | None = Field(None, max_length=5000)
    version_reviewed: str | None = Field(None, max_length=20)


class MarketplaceSuiteReviewUpdate(BaseModel):
    """Schema for updating a review."""

    rating: int | None = Field(None, ge=1, le=5)
    title: str | None = Field(None, max_length=200)
    content: str | None = None


class MarketplaceSuiteReviewResponse(BaseModel):
    """Schema for review response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    marketplace_suite_id: int
    user_id: int
    username: str = ""
    rating: int
    title: str | None
    content: str | None
    version_reviewed: str | None
    is_approved: bool
    helpful_count: int
    not_helpful_count: int
    created_at: datetime
    updated_at: datetime


class MarketplaceSuiteReviewList(BaseModel):
    """Paginated list of reviews."""

    items: list[MarketplaceSuiteReviewResponse]
    total: int
    limit: int
    offset: int
    average_rating: float | None
    rating_distribution: dict[int, int] = Field(default_factory=dict)


# ==================== Marketplace Install Schemas ====================


class MarketplaceSuiteInstallResponse(BaseModel):
    """Schema for install response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    marketplace_suite_id: int
    user_id: int
    version_installed: str
    is_active: bool
    installed_at: datetime
    updated_at: datetime | None


class MarketplaceSuiteInstallList(BaseModel):
    """List of installs."""

    items: list[MarketplaceSuiteInstallResponse]
    total: int


# ==================== Marketplace Search/Filter Schemas ====================


class MarketplaceSearchParams(BaseModel):
    """Search parameters for marketplace."""

    query: str | None = Field(None, max_length=200)
    category: str | None = Field(None, max_length=100)
    tags: list[str] | None = None
    license_type: str | None = Field(None, max_length=50)
    verified_only: bool = False
    featured_only: bool = False
    min_rating: float | None = Field(None, ge=0, le=5)
    sort_by: str = Field(
        default="downloads",
        pattern=r"^(downloads|rating|updated|name|created)$",
    )
    sort_order: str = Field(default="desc", pattern=r"^(asc|desc)$")


class MarketplaceCategoryStats(BaseModel):
    """Statistics for a marketplace category."""

    category: str
    count: int
    total_downloads: int
    average_rating: float | None


class MarketplaceStats(BaseModel):
    """Overall marketplace statistics."""

    total_suites: int
    total_downloads: int
    total_installs: int
    total_reviews: int
    categories: list[MarketplaceCategoryStats]
    top_tags: list[tuple[str, int]]


# ==================== GitHub Import Schemas ====================


class GitHubImportRequest(BaseModel):
    """Schema for importing a test suite from GitHub."""

    github_url: str = Field(..., max_length=500)
    branch: str = Field(default="main", max_length=100)
    path: str = Field(default="", max_length=500)
    name: str | None = Field(None, max_length=255)
    slug: str | None = Field(None, max_length=255)
    description: str | None = Field(None, max_length=10000)
    category: str = Field(default="general", max_length=100)
    tags: list[str] = Field(default_factory=list)
    license_type: str = Field(default="MIT", max_length=50)


class GitHubImportResponse(BaseModel):
    """Response for GitHub import."""

    success: bool
    marketplace_suite: MarketplaceSuiteResponse | None = None
    error: str | None = None
    files_imported: list[str] = Field(default_factory=list)


# Update forward references
SuiteExecutionDetail.model_rebuild()
MarketplaceSuiteDetail.model_rebuild()
