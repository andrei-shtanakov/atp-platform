"""Pydantic request/response schemas for the Benchmark API."""

from typing import Any

from pydantic import BaseModel, Field


class BenchmarkCreate(BaseModel):
    """Schema for creating a new benchmark."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="")
    suite: dict[str, Any]  # Test suite as parsed YAML/JSON
    tags: list[str] = Field(default_factory=list)
    version: str = Field(default="1.0")
    family_tag: str | None = Field(default=None)
    parent_id: int | None = Field(default=None)
    webhook_url: str | None = Field(default=None)


class BenchmarkResponse(BaseModel):
    """Schema for returning a benchmark."""

    id: int
    name: str
    description: str
    tasks_count: int
    tags: list[str]
    version: str
    family_tag: str | None
    created_at: str


class RunResponse(BaseModel):
    """Schema for returning a benchmark run."""

    id: int
    benchmark_id: int
    agent_name: str
    adapter_type: str
    status: str
    current_task_index: int
    total_score: float | None
    started_at: str
    finished_at: str | None


class SubmitRequest(BaseModel):
    """Schema for submitting a task result."""

    response: dict[str, Any]  # ATPResponse as JSON
    events: list[dict[str, Any]] | None = Field(default=None)
    task_index: int = Field(
        ...,
        description="Task index from ATPRequest.metadata.task_index",
    )


class TaskResultResponse(BaseModel):
    """Schema for returning a single task result."""

    task_index: int
    score: float | None
    eval_results: list[dict[str, Any]] | None


class LeaderboardEntry(BaseModel):
    """Schema for a leaderboard row."""

    user_id: int
    agent_name: str
    best_score: float
    run_count: int


class RunStatusResponse(BaseModel):
    """Schema for detailed run status with completed tasks."""

    id: int
    status: str
    current_task_index: int
    tasks_count: int
    total_score: float | None
    completed_tasks: list[TaskResultResponse]
