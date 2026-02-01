"""SQLAlchemy ORM models for ATP Analytics cost tracking and A/B testing."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class AnalyticsBase(DeclarativeBase):
    """Base class for analytics ORM models."""

    pass


class CostRecord(AnalyticsBase):
    """Record of a single LLM operation cost.

    Tracks costs per operation including provider, model, token usage,
    and associated test/suite/agent metadata.
    """

    __tablename__ = "cost_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Timestamp of the operation
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)

    # Provider information
    provider: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # anthropic, openai, google, azure, bedrock
    model: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )  # claude-3-sonnet, gpt-4, etc.

    # Token usage
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)

    # Cost in USD (using Numeric for precision)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)

    # Context: optional associations to test execution
    test_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    suite_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    agent_name: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True
    )

    # Optional metadata for additional context
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Timestamps for record tracking
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    # Indexes for efficient querying
    __table_args__ = (
        # Composite index for time-based queries by provider
        Index("idx_cost_provider_timestamp", "provider", "timestamp"),
        # Composite index for time-based queries by model
        Index("idx_cost_model_timestamp", "model", "timestamp"),
        # Composite index for agent-based queries
        Index("idx_cost_agent_timestamp", "agent_name", "timestamp"),
        # Composite index for suite-based queries
        Index("idx_cost_suite_timestamp", "suite_id", "timestamp"),
        # Composite index for test-based queries
        Index("idx_cost_test_timestamp", "test_id", "timestamp"),
        # Index for date-range aggregations
        Index("idx_cost_timestamp_provider_model", "timestamp", "provider", "model"),
    )

    def __repr__(self) -> str:
        return (
            f"CostRecord(id={self.id}, provider={self.provider!r}, "
            f"model={self.model!r}, cost_usd={self.cost_usd})"
        )

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used."""
        return self.input_tokens + self.output_tokens


class CostBudget(AnalyticsBase):
    """Budget configuration for cost tracking.

    Supports daily, weekly, and monthly budgets with alert thresholds.
    """

    __tablename__ = "cost_budgets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Budget name for identification
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)

    # Budget period: daily, weekly, monthly
    period: Mapped[str] = mapped_column(String(20), nullable=False)

    # Budget limit in USD
    limit_usd: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)

    # Alert threshold (0.0 to 1.0, e.g., 0.8 = 80% of budget)
    alert_threshold: Mapped[float] = mapped_column(Float, default=0.8)

    # Optional scope filters (JSON for flexibility)
    # Can filter by provider, model, agent, suite, etc.
    scope_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Alert configuration
    alert_channels_json: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    # Description
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Active status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    # Indexes
    __table_args__ = (
        Index("idx_budget_period", "period"),
        Index("idx_budget_active", "is_active"),
    )

    def __repr__(self) -> str:
        return (
            f"CostBudget(id={self.id}, name={self.name!r}, "
            f"period={self.period!r}, limit_usd={self.limit_usd})"
        )

    @property
    def alert_channels(self) -> list[str]:
        """Get alert channels as list."""
        return self.alert_channels_json or []

    @property
    def scope(self) -> dict[str, Any]:
        """Get scope filters as dict."""
        return self.scope_json or {}


class ScheduledReport(AnalyticsBase):
    """Configuration for a scheduled analytics report.

    Stores report settings including frequency, recipients, and what
    analytics to include in the report.
    """

    __tablename__ = "scheduled_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Report name
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)

    # Frequency: daily, weekly, monthly
    frequency: Mapped[str] = mapped_column(String(20), nullable=False)

    # Recipients (email addresses)
    recipients_json: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    # Report content settings
    include_trends: Mapped[bool] = mapped_column(Boolean, default=True)
    include_anomalies: Mapped[bool] = mapped_column(Boolean, default=True)
    include_correlations: Mapped[bool] = mapped_column(Boolean, default=False)

    # Filters to apply to the report (suite_name, agent_name, etc.)
    filters_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Active status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Scheduling
    last_run: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    next_run: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, index=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    # Indexes
    __table_args__ = (Index("idx_report_active_next_run", "is_active", "next_run"),)

    def __repr__(self) -> str:
        return (
            f"ScheduledReport(id={self.id}, name={self.name!r}, "
            f"frequency={self.frequency!r})"
        )

    @property
    def recipients(self) -> list[str]:
        """Get recipients as list."""
        return self.recipients_json or []

    @property
    def filters(self) -> dict[str, Any]:
        """Get filters as dict."""
        return self.filters_json or {}


# ==================== A/B Testing Models ====================


class ABExperiment(AnalyticsBase):
    """A/B testing experiment record.

    Stores experiment configuration, status, and results for comparing
    agent variants with statistical rigor.
    """

    __tablename__ = "ab_experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Basic info
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    suite_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Test IDs filter (JSON array of test IDs, null = all tests)
    test_ids_json: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    # Variants configuration (JSON)
    control_variant_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    treatment_variant_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Metrics configuration (JSON array of MetricConfig)
    metrics_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)

    # Rollback configuration (JSON)
    rollback_config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Sample size configuration
    min_sample_size: Mapped[int] = mapped_column(Integer, default=30)
    max_sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_duration_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    significance_level: Mapped[float] = mapped_column(Float, default=0.05)

    # Status
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="draft", index=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    concluded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    paused_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Results
    conclusion_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    winner: Mapped[str | None] = mapped_column(String(20), nullable=True)
    results_json: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )
    rollback_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    consecutive_degradation_checks: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    observations: Mapped[list["ABExperimentObservation"]] = relationship(
        "ABExperimentObservation",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("idx_ab_experiment_status_created", "status", "created_at"),
        Index("idx_ab_experiment_suite", "suite_name"),
    )

    def __repr__(self) -> str:
        return (
            f"ABExperiment(id={self.id}, name={self.name!r}, "
            f"status={self.status!r}, suite={self.suite_name!r})"
        )

    @property
    def test_ids(self) -> list[str] | None:
        """Get test IDs filter."""
        return self.test_ids_json

    @property
    def control_variant(self) -> dict[str, Any]:
        """Get control variant config."""
        return self.control_variant_json

    @property
    def treatment_variant(self) -> dict[str, Any]:
        """Get treatment variant config."""
        return self.treatment_variant_json

    @property
    def metrics(self) -> list[dict[str, Any]]:
        """Get metrics configuration."""
        return self.metrics_json

    @property
    def rollback_config(self) -> dict[str, Any]:
        """Get rollback configuration."""
        return self.rollback_config_json

    @property
    def results(self) -> list[dict[str, Any]]:
        """Get experiment results."""
        return self.results_json or []


class ABExperimentObservation(AnalyticsBase):
    """Single observation in an A/B experiment.

    Records the result of a single test run assigned to a variant.
    """

    __tablename__ = "ab_experiment_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to experiment
    experiment_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("ab_experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Variant assignment
    variant_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Test info
    test_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.now, index=True
    )

    # Metrics
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Additional metadata
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Relationship
    experiment: Mapped["ABExperiment"] = relationship(
        "ABExperiment", back_populates="observations"
    )

    # Indexes
    __table_args__ = (
        Index("idx_ab_obs_experiment_variant", "experiment_id", "variant_name"),
        Index("idx_ab_obs_experiment_timestamp", "experiment_id", "timestamp"),
        Index("idx_ab_obs_test", "test_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"ABExperimentObservation(id={self.id}, "
            f"experiment_id={self.experiment_id}, "
            f"variant={self.variant_name!r}, test_id={self.test_id!r})"
        )

    @property
    def observation_metadata(self) -> dict[str, Any]:
        """Get observation metadata."""
        return self.metadata_json or {}
