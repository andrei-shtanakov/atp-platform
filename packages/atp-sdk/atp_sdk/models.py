"""Data models for the ATP SDK."""

from enum import StrEnum

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    """Status of a benchmark run."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class LeaderboardEntry(BaseModel):
    """A single entry in the benchmark leaderboard."""

    user_id: int
    agent_name: str
    best_score: float
    run_count: int


class BenchmarkInfo(BaseModel):
    """Information about a benchmark."""

    id: int
    name: str
    description: str
    tasks_count: int
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0"
    family_tag: str | None = None


class RunInfo(BaseModel):
    """Information about a benchmark run."""

    id: int
    benchmark_id: int
    agent_name: str
    status: RunStatus
    current_task_index: int = 0
    total_score: float | None = None
