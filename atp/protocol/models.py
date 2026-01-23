"""ATP Protocol data models."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ResponseStatus(str, Enum):
    """Status values for ATP Response."""

    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class EventType(str, Enum):
    """Event types for ATP Event streaming."""

    TOOL_CALL = "tool_call"
    LLM_REQUEST = "llm_request"
    REASONING = "reasoning"
    ERROR = "error"
    PROGRESS = "progress"


class Task(BaseModel):
    """Task specification for the agent."""

    description: str = Field(..., description="Task description", min_length=1)
    input_data: dict[str, Any] | None = Field(
        None, description="Optional input data for the task"
    )
    expected_artifacts: list[str] | None = Field(
        None, description="Expected artifact paths/names"
    )


class Context(BaseModel):
    """Execution context for the agent."""

    tools_endpoint: str | None = Field(None, description="URL for tools API")
    workspace_path: str | None = Field(None, description="Path to workspace directory")
    environment: dict[str, str] | None = Field(
        None, description="Environment variables"
    )


class ATPRequest(BaseModel):
    """ATP Protocol request message."""

    version: str = Field("1.0", description="Protocol version")
    task_id: str = Field(..., description="Unique task identifier")
    task: Task = Field(..., description="Task specification")
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Execution constraints"
    )
    context: Context | None = Field(None, description="Execution context")
    metadata: dict[str, Any] | None = Field(None, description="Pass-through metadata")

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        """Validate task_id is not empty."""
        if not v or not v.strip():
            raise ValueError("task_id cannot be empty")
        return v


class Metrics(BaseModel):
    """Execution metrics for agent run."""

    total_tokens: int | None = Field(None, description="Total tokens used", ge=0)
    input_tokens: int | None = Field(None, description="Input tokens", ge=0)
    output_tokens: int | None = Field(None, description="Output tokens", ge=0)
    total_steps: int | None = Field(None, description="Total execution steps", ge=0)
    tool_calls: int | None = Field(None, description="Number of tool calls", ge=0)
    llm_calls: int | None = Field(None, description="Number of LLM calls", ge=0)
    wall_time_seconds: float | None = Field(
        None, description="Wall clock time in seconds", ge=0
    )
    cost_usd: float | None = Field(None, description="Cost in USD", ge=0)


class ArtifactFile(BaseModel):
    """File artifact produced by agent."""

    type: Literal["file"] = "file"
    path: str = Field(..., description="File path", min_length=1)
    content_type: str | None = Field(None, description="MIME type")
    size_bytes: int | None = Field(None, description="File size in bytes", ge=0)
    content_hash: str | None = Field(
        None, description="Hash of content (e.g., SHA-256)"
    )
    content: str | None = Field(
        None, description="Inline content (optional, base64 for binary)"
    )


class ArtifactStructured(BaseModel):
    """Structured data artifact."""

    type: Literal["structured"] = "structured"
    name: str = Field(..., description="Artifact name", min_length=1)
    data: dict[str, Any] = Field(..., description="Structured data")
    content_type: str | None = Field(
        None, description="Content type (e.g., application/json)"
    )


class ArtifactReference(BaseModel):
    """Reference to external artifact."""

    type: Literal["reference"] = "reference"
    path: str = Field(..., description="Reference path/URL", min_length=1)
    content_type: str | None = Field(None, description="Content type")
    size_bytes: int | None = Field(None, description="Size in bytes", ge=0)


Artifact = ArtifactFile | ArtifactStructured | ArtifactReference


class ATPResponse(BaseModel):
    """ATP Protocol response message."""

    version: str = Field("1.0", description="Protocol version")
    task_id: str = Field(..., description="Task identifier from request")
    status: ResponseStatus = Field(..., description="Execution status")
    artifacts: list[Artifact] = Field(
        default_factory=list, description="Output artifacts"
    )
    metrics: Metrics | None = Field(None, description="Execution metrics")
    error: str | None = Field(None, description="Error message if failed")
    trace_id: str | None = Field(None, description="Optional trace identifier")

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        """Validate task_id is not empty."""
        if not v or not v.strip():
            raise ValueError("task_id cannot be empty")
        return v


class ToolCallPayload(BaseModel):
    """Payload for tool_call event."""

    tool: str = Field(..., description="Tool name", min_length=1)
    input: dict[str, Any] | None = Field(None, description="Tool input")
    output: dict[str, Any] | None = Field(None, description="Tool output")
    duration_ms: float | None = Field(
        None, description="Duration in milliseconds", ge=0
    )
    status: str | None = Field(None, description="Call status (success, error)")


class LLMRequestPayload(BaseModel):
    """Payload for llm_request event."""

    model: str = Field(..., description="Model name", min_length=1)
    input_tokens: int | None = Field(None, description="Input tokens", ge=0)
    output_tokens: int | None = Field(None, description="Output tokens", ge=0)
    duration_ms: float | None = Field(
        None, description="Duration in milliseconds", ge=0
    )


class ReasoningPayload(BaseModel):
    """Payload for reasoning event."""

    thought: str | None = Field(None, description="Agent thought")
    plan: str | None = Field(None, description="Plan or strategy")
    step: str | None = Field(None, description="Current step description")


class ErrorPayload(BaseModel):
    """Payload for error event."""

    error_type: str = Field(..., description="Error type", min_length=1)
    message: str = Field(..., description="Error message", min_length=1)
    recoverable: bool | None = Field(None, description="Whether error is recoverable")


class ProgressPayload(BaseModel):
    """Payload for progress event."""

    current_step: int | None = Field(None, description="Current step number", ge=0)
    percentage: float | None = Field(
        None, description="Progress percentage (0-100)", ge=0, le=100
    )
    message: str | None = Field(None, description="Progress message")


EventPayload = (
    ToolCallPayload
    | LLMRequestPayload
    | ReasoningPayload
    | ErrorPayload
    | ProgressPayload
    | dict[str, Any]
)


class ATPEvent(BaseModel):
    """ATP Protocol event message for streaming."""

    version: str = Field("1.0", description="Protocol version")
    task_id: str = Field(..., description="Task identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Event timestamp"
    )
    sequence: int = Field(..., description="Monotonic sequence number", ge=0)
    event_type: EventType = Field(..., description="Event type")
    payload: dict[str, Any] = Field(..., description="Event-specific payload")

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        """Validate task_id is not empty."""
        if not v or not v.strip():
            raise ValueError("task_id cannot be empty")
        return v
