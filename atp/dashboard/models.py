"""SQLAlchemy ORM models for the ATP Dashboard."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class User(Base):
    """User model for authentication."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    def __repr__(self) -> str:
        return f"User(id={self.id}, username={self.username!r})"


class Agent(Base):
    """Agent configuration stored in the database."""

    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    # Relationships
    suite_executions: Mapped[list["SuiteExecution"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan"
    )

    # Indexes for common queries
    __table_args__ = (Index("idx_agent_name", "name"),)

    def __repr__(self) -> str:
        return f"Agent(id={self.id}, name={self.name!r}, type={self.agent_type!r})"


class SuiteExecution(Base):
    """Execution record of a test suite."""

    __tablename__ = "suite_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    suite_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    agent_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("agents.id"), nullable=False
    )

    # Execution metadata
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Configuration
    runs_per_test: Mapped[int] = mapped_column(Integer, default=1)

    # Results summary
    total_tests: Mapped[int] = mapped_column(Integer, default=0)
    passed_tests: Mapped[int] = mapped_column(Integer, default=0)
    failed_tests: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)

    # Status and error
    status: Mapped[str] = mapped_column(
        String(20), default="running"
    )  # running, completed, failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    agent: Mapped["Agent"] = relationship(back_populates="suite_executions")
    test_executions: Mapped[list["TestExecution"]] = relationship(
        back_populates="suite_execution", cascade="all, delete-orphan"
    )

    # Indexes for common queries (leaderboard, comparison)
    __table_args__ = (
        Index("idx_suite_agent", "suite_name", "agent_id"),
        Index("idx_suite_started", "suite_name", "started_at"),
        Index("idx_agent_started", "agent_id", "started_at"),
    )

    def __repr__(self) -> str:
        return (
            f"SuiteExecution(id={self.id}, suite={self.suite_name!r}, "
            f"agent_id={self.agent_id})"
        )


class TestExecution(Base):
    """Execution record of a single test within a suite."""

    __tablename__ = "test_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    suite_execution_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("suite_executions.id"), nullable=False
    )

    # Test identification
    test_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    test_name: Mapped[str] = mapped_column(String(255), nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)

    # Execution metadata
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Results summary
    total_runs: Mapped[int] = mapped_column(Integer, default=1)
    successful_runs: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(Boolean, default=False)

    # Scoring
    score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Status and error
    status: Mapped[str] = mapped_column(String(20), default="running")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Statistics (stored as JSON for flexibility)
    statistics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Relationships
    suite_execution: Mapped["SuiteExecution"] = relationship(
        back_populates="test_executions"
    )
    run_results: Mapped[list["RunResult"]] = relationship(
        back_populates="test_execution", cascade="all, delete-orphan"
    )
    evaluation_results: Mapped[list["EvaluationResult"]] = relationship(
        back_populates="test_execution", cascade="all, delete-orphan"
    )
    score_components: Mapped[list["ScoreComponent"]] = relationship(
        back_populates="test_execution", cascade="all, delete-orphan"
    )

    # Unique constraint and indexes for common queries
    __table_args__ = (
        UniqueConstraint("suite_execution_id", "test_id", name="uq_suite_test"),
        Index("idx_test_suite_exec", "suite_execution_id", "test_id"),
        Index("idx_test_started", "started_at"),
    )

    def __repr__(self) -> str:
        return f"TestExecution(id={self.id}, test_id={self.test_id!r})"


class RunResult(Base):
    """Individual run result within a test execution."""

    __tablename__ = "run_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    test_execution_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("test_executions.id"), nullable=False
    )

    # Run identification
    run_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Timing
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Response data
    response_status: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # completed, failed, timeout, etc.
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Metrics
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tool_calls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_calls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Full response stored as JSON for detailed inspection
    response_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Events stored as JSON
    events_json: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )

    # Relationships
    test_execution: Mapped["TestExecution"] = relationship(back_populates="run_results")
    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="run_result", cascade="all, delete-orphan"
    )

    # Unique constraint and indexes for common queries
    __table_args__ = (
        UniqueConstraint("test_execution_id", "run_number", name="uq_test_run"),
        Index("idx_run_test_exec", "test_execution_id"),
    )

    def __repr__(self) -> str:
        return (
            f"RunResult(id={self.id}, test_execution_id={self.test_execution_id}, "
            f"run={self.run_number})"
        )


class Artifact(Base):
    """Artifact produced by an agent run."""

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_result_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("run_results.id"), nullable=False
    )

    # Artifact metadata
    artifact_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # file, structured, reference
    path: Mapped[str | None] = mapped_column(String(4096), nullable=True)
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(256), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Content (for small artifacts) or reference
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Relationships
    run_result: Mapped["RunResult"] = relationship(back_populates="artifacts")

    def __repr__(self) -> str:
        return (
            f"Artifact(id={self.id}, type={self.artifact_type!r}, path={self.path!r})"
        )


class EvaluationResult(Base):
    """Evaluation result for a test execution."""

    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    test_execution_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("test_executions.id"), nullable=False
    )

    # Evaluator info
    evaluator_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Results
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_checks: Mapped[int] = mapped_column(Integer, default=0)
    passed_checks: Mapped[int] = mapped_column(Integer, default=0)
    failed_checks: Mapped[int] = mapped_column(Integer, default=0)

    # Detailed checks stored as JSON
    checks_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)

    # Relationships
    test_execution: Mapped["TestExecution"] = relationship(
        back_populates="evaluation_results"
    )

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(id={self.id}, evaluator={self.evaluator_name!r}, "
            f"passed={self.passed})"
        )


class ScoreComponent(Base):
    """Score breakdown component for a test execution."""

    __tablename__ = "score_components"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    test_execution_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("test_executions.id"), nullable=False
    )

    # Component info
    component_name: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # quality, completeness, efficiency, cost

    # Values
    raw_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    normalized_value: Mapped[float] = mapped_column(Float, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    weighted_value: Mapped[float] = mapped_column(Float, nullable=False)

    # Additional details
    details_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Relationships
    test_execution: Mapped["TestExecution"] = relationship(
        back_populates="score_components"
    )

    # Unique constraint: component_name within a test_execution
    __table_args__ = (
        UniqueConstraint(
            "test_execution_id", "component_name", name="uq_test_component"
        ),
    )

    def __repr__(self) -> str:
        return (
            f"ScoreComponent(id={self.id}, name={self.component_name!r}, "
            f"weighted={self.weighted_value:.3f})"
        )
