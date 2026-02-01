"""SQLAlchemy ORM models for ATP Analytics cost tracking."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
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
