"""ATP Protocol data models."""

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Maximum lengths for various fields
MAX_TASK_ID_LENGTH = 128
MAX_DESCRIPTION_LENGTH = 100_000
MAX_PATH_LENGTH = 4096
MAX_ERROR_LENGTH = 10_000
MAX_CONTENT_LENGTH = 10_000_000  # 10MB for inline content
MAX_ARTIFACTS_COUNT = 1000  # Maximum artifacts per response
MAX_ENV_VARS_COUNT = 100  # Maximum environment variables
MAX_METADATA_KEYS = 50  # Maximum metadata keys

# Valid task ID pattern - alphanumeric, underscore, hyphen
TASK_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class ResponseStatus(StrEnum):
    """Status values for ATP Response."""

    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class EventType(StrEnum):
    """Event types for ATP Event streaming."""

    TOOL_CALL = "tool_call"
    LLM_REQUEST = "llm_request"
    REASONING = "reasoning"
    ERROR = "error"
    PROGRESS = "progress"


class Task(BaseModel):
    """Task specification for the agent."""

    description: str = Field(
        ...,
        description="Task description",
        min_length=1,
        max_length=MAX_DESCRIPTION_LENGTH,
    )
    input_data: dict[str, Any] | None = Field(
        None, description="Optional input data for the task"
    )
    expected_artifacts: list[str] | None = Field(
        None, description="Expected artifact paths/names"
    )

    @field_validator("expected_artifacts")
    @classmethod
    def validate_expected_artifacts(cls, v: list[str] | None) -> list[str] | None:
        """Validate expected artifact paths."""
        if v is None:
            return v

        validated = []
        for path in v:
            if not path or not path.strip():
                raise ValueError("Artifact path cannot be empty")
            if len(path) > MAX_PATH_LENGTH:
                raise ValueError(
                    f"Artifact path too long: {len(path)} > {MAX_PATH_LENGTH}"
                )
            # Reject obvious path traversal
            if ".." in path or path.startswith("/"):
                raise ValueError(f"Invalid artifact path: {path}")
            validated.append(path.strip())
        return validated


class Context(BaseModel):
    """Execution context for the agent."""

    tools_endpoint: str | None = Field(None, description="URL for tools API")
    workspace_path: str | None = Field(None, description="Path to workspace directory")
    environment: dict[str, str] | None = Field(
        None, description="Environment variables"
    )

    @field_validator("tools_endpoint")
    @classmethod
    def validate_tools_endpoint(cls, v: str | None) -> str | None:
        """Validate tools endpoint URL."""
        if v is None:
            return v
        # Basic URL validation - must be http/https
        v = v.strip()
        if not v:
            return None
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("Tools endpoint must be an HTTP/HTTPS URL")
        return v

    @field_validator("workspace_path")
    @classmethod
    def validate_workspace_path(cls, v: str | None) -> str | None:
        """Validate workspace path."""
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if len(v) > MAX_PATH_LENGTH:
            raise ValueError(f"Workspace path too long: {len(v)} > {MAX_PATH_LENGTH}")
        # Check for null bytes
        if "\x00" in v:
            raise ValueError("Null bytes not allowed in workspace path")
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        """Validate environment variables."""
        if v is None:
            return v

        # Check count limit
        if len(v) > MAX_ENV_VARS_COUNT:
            raise ValueError(
                f"Too many environment variables: {len(v)} > {MAX_ENV_VARS_COUNT}"
            )

        # Check for null bytes in keys and values
        for key, value in v.items():
            if "\x00" in key or "\x00" in value:
                raise ValueError("Null bytes not allowed in environment variables")
            if not key.strip():
                raise ValueError("Environment variable name cannot be empty")
            # Validate key format
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                raise ValueError(
                    f"Invalid environment variable name: {key}. "
                    "Must start with letter or underscore, "
                    "contain only alphanumeric and underscore."
                )
        return v


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
        """Validate task_id format and length."""
        if not v or not v.strip():
            raise ValueError("task_id cannot be empty")
        v = v.strip()
        if len(v) > MAX_TASK_ID_LENGTH:
            raise ValueError(f"task_id too long: {len(v)} > {MAX_TASK_ID_LENGTH}")
        if not TASK_ID_PATTERN.match(v):
            raise ValueError(
                "task_id must contain only alphanumeric characters, "
                "underscores, and hyphens"
            )
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate metadata size constraints."""
        if v is None:
            return v
        if len(v) > MAX_METADATA_KEYS:
            raise ValueError(f"Too many metadata keys: {len(v)} > {MAX_METADATA_KEYS}")
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
    path: str = Field(
        ..., description="File path", min_length=1, max_length=MAX_PATH_LENGTH
    )
    content_type: str | None = Field(None, description="MIME type", max_length=256)
    size_bytes: int | None = Field(None, description="File size in bytes", ge=0)
    content_hash: str | None = Field(
        None, description="Hash of content (e.g., SHA-256)", max_length=128
    )
    content: str | None = Field(
        None,
        description="Inline content (optional, base64 for binary)",
        max_length=MAX_CONTENT_LENGTH,
    )

    @field_validator("path")
    @classmethod
    def validate_artifact_path(cls, v: str) -> str:
        """Validate artifact file path for security."""
        if not v or not v.strip():
            raise ValueError("Artifact path cannot be empty")
        v = v.strip()

        # Check for null bytes
        if "\x00" in v:
            raise ValueError("Null bytes not allowed in artifact path")

        # Check for path traversal
        path_parts = v.replace("\\", "/").split("/")
        if ".." in path_parts:
            raise ValueError("Path traversal (..) not allowed in artifact path")

        # Normalize and validate
        try:
            # Use Path to normalize but keep as relative
            path_obj = Path(v)
            # Reject absolute paths
            if path_obj.is_absolute():
                raise ValueError("Absolute paths not allowed in artifacts")
        except Exception as e:
            raise ValueError(f"Invalid artifact path: {e}")

        return v


class ArtifactStructured(BaseModel):
    """Structured data artifact."""

    type: Literal["structured"] = "structured"
    name: str = Field(..., description="Artifact name", min_length=1, max_length=256)
    data: dict[str, Any] = Field(..., description="Structured data")
    content_type: str | None = Field(
        None, description="Content type (e.g., application/json)", max_length=256
    )

    @field_validator("name")
    @classmethod
    def validate_artifact_name(cls, v: str) -> str:
        """Validate artifact name."""
        if not v or not v.strip():
            raise ValueError("Artifact name cannot be empty")
        v = v.strip()
        # Reject path separators in name
        if "/" in v or "\\" in v:
            raise ValueError("Path separators not allowed in artifact name")
        if "\x00" in v:
            raise ValueError("Null bytes not allowed in artifact name")
        return v


class ArtifactReference(BaseModel):
    """Reference to external artifact."""

    type: Literal["reference"] = "reference"
    path: str = Field(
        ..., description="Reference path/URL", min_length=1, max_length=MAX_PATH_LENGTH
    )
    content_type: str | None = Field(None, description="Content type", max_length=256)
    size_bytes: int | None = Field(None, description="Size in bytes", ge=0)

    @field_validator("path")
    @classmethod
    def validate_reference_path(cls, v: str) -> str:
        """Validate reference path."""
        if not v or not v.strip():
            raise ValueError("Reference path cannot be empty")
        v = v.strip()
        if "\x00" in v:
            raise ValueError("Null bytes not allowed in reference path")
        return v


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
    error: str | None = Field(
        None, description="Error message if failed", max_length=MAX_ERROR_LENGTH
    )
    trace_id: str | None = Field(
        None, description="Optional trace identifier", max_length=256
    )

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        """Validate task_id format and length."""
        if not v or not v.strip():
            raise ValueError("task_id cannot be empty")
        v = v.strip()
        if len(v) > MAX_TASK_ID_LENGTH:
            raise ValueError(f"task_id too long: {len(v)} > {MAX_TASK_ID_LENGTH}")
        # Note: We don't enforce pattern here since this is a response
        # and the task_id comes from the agent
        return v

    @field_validator("artifacts")
    @classmethod
    def validate_artifacts_count(cls, v: list[Artifact]) -> list[Artifact]:
        """Validate artifacts count limit."""
        if len(v) > MAX_ARTIFACTS_COUNT:
            raise ValueError(f"Too many artifacts: {len(v)} > {MAX_ARTIFACTS_COUNT}")
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
        default_factory=lambda: datetime.now(UTC),
        description="Event timestamp",
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
