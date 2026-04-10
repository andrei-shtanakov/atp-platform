"""SQLAlchemy models for the ATP Benchmark platform."""

from datetime import datetime
from enum import StrEnum
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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from atp.dashboard.models import DEFAULT_TENANT_ID, Base


class RunStatus(StrEnum):
    """Status of a benchmark run."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"


class Benchmark(Base):
    """A benchmark definition containing a test suite."""

    __tablename__ = "benchmarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=DEFAULT_TENANT_ID,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    suite: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tasks_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    family_tag: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True
    )
    parent_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("benchmarks.id"),
        nullable=True,
    )
    webhook_url: Mapped[str | None] = mapped_column(
        String(2048), nullable=True, default=None
    )
    is_immutable: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    # Relationships
    parent: Mapped["Benchmark | None"] = relationship(
        "Benchmark",
        remote_side="Benchmark.id",
        back_populates="children",
    )
    children: Mapped[list["Benchmark"]] = relationship(
        "Benchmark",
        back_populates="parent",
    )
    runs: Mapped[list["Run"]] = relationship(
        "Run",
        back_populates="benchmark",
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_benchmark_tenant_name"),
        Index("idx_benchmark_family_tag", "family_tag"),
        Index("idx_benchmark_tenant", "tenant_id"),
    )

    def __repr__(self) -> str:
        return f"Benchmark(id={self.id}, name={self.name!r}, version={self.version!r})"


class Run(Base):
    """A single execution of a benchmark by an agent."""

    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=DEFAULT_TENANT_ID,
        index=True,
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
    )
    benchmark_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("benchmarks.id"),
        nullable=False,
    )
    agent_name: Mapped[str] = mapped_column(String(200), nullable=False)
    adapter_type: Mapped[str] = mapped_column(String(50), nullable=False, default="sdk")
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=RunStatus.PENDING,
    )
    current_task_index: Mapped[int] = mapped_column(Integer, default=0)
    total_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=3600)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    events: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True, default=list
    )

    # Relationships
    benchmark: Mapped["Benchmark"] = relationship(
        "Benchmark",
        back_populates="runs",
    )
    task_results: Mapped[list["TaskResult"]] = relationship(
        "TaskResult",
        back_populates="run",
    )

    __table_args__ = (
        Index("idx_run_user", "user_id"),
        Index("idx_run_benchmark", "benchmark_id"),
        Index("idx_run_status", "status"),
        Index("idx_run_tenant", "tenant_id"),
    )

    def __repr__(self) -> str:
        return f"Run(id={self.id}, agent={self.agent_name!r}, status={self.status!r})"


class TaskResult(Base):
    """Result of a single task within a benchmark run."""

    __tablename__ = "benchmark_task_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("benchmark_runs.id"),
        nullable=False,
    )
    task_index: Mapped[int] = mapped_column(Integer, nullable=False)
    request: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    response: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    events: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    eval_results: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    run: Mapped["Run"] = relationship(
        "Run",
        back_populates="task_results",
    )

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "task_index",
            name="uq_task_result_run_index",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"TaskResult(id={self.id}, run_id={self.run_id}, "
            f"task_index={self.task_index})"
        )
