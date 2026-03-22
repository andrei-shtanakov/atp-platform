"""Data models for chaos engineering configuration."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ErrorType(str, Enum):
    """Types of errors that can be injected."""

    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    INTERNAL_ERROR = "internal_error"
    RATE_LIMIT = "rate_limit"
    VALIDATION_ERROR = "validation_error"
    PERMISSION_DENIED = "permission_denied"


class ToolFailureConfig(BaseModel):
    """Configuration for tool failure injection."""

    tool: str = Field(..., description="Tool name pattern to affect (* for all)")
    probability: float = Field(
        0.0, description="Probability of failure (0.0 to 1.0)", ge=0.0, le=1.0
    )
    error_type: ErrorType = Field(
        default=ErrorType.INTERNAL_ERROR, description="Type of error to inject"
    )
    error_message: str | None = Field(
        None, description="Custom error message (uses default if not specified)"
    )

    @field_validator("tool")
    @classmethod
    def validate_tool_pattern(cls, v: str) -> str:
        """Validate tool name pattern."""
        if not v or not v.strip():
            raise ValueError("Tool name pattern cannot be empty")
        return v.strip()


class LatencyConfig(BaseModel):
    """Configuration for latency injection."""

    min_ms: int = Field(0, description="Minimum latency in milliseconds", ge=0)
    max_ms: int = Field(0, description="Maximum latency in milliseconds", ge=0)
    affected_tools: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Tool patterns to affect (* for all)",
    )

    @field_validator("max_ms")
    @classmethod
    def validate_max_ms(cls, v: int, info: Any) -> int:
        """Ensure max_ms >= min_ms."""
        min_ms = info.data.get("min_ms", 0)
        if v < min_ms:
            raise ValueError(f"max_ms ({v}) must be >= min_ms ({min_ms})")
        return v


class TokenLimitConfig(BaseModel):
    """Configuration for token limit simulation."""

    max_input: int | None = Field(
        None, description="Maximum input tokens allowed", ge=1
    )
    max_output: int | None = Field(
        None, description="Maximum output tokens allowed", ge=1
    )


class PartialResponseConfig(BaseModel):
    """Configuration for partial response simulation."""

    enabled: bool = Field(False, description="Enable partial response simulation")
    truncate_probability: float = Field(
        0.0, description="Probability of truncating response", ge=0.0, le=1.0
    )
    min_percentage: float = Field(
        0.5, description="Minimum percentage of response to keep", ge=0.1, le=1.0
    )
    max_percentage: float = Field(
        0.9, description="Maximum percentage of response to keep", ge=0.1, le=1.0
    )

    @field_validator("max_percentage")
    @classmethod
    def validate_max_percentage(cls, v: float, info: Any) -> float:
        """Ensure max_percentage >= min_percentage."""
        min_pct = info.data.get("min_percentage", 0.5)
        if v < min_pct:
            raise ValueError(
                f"max_percentage ({v}) must be >= min_percentage ({min_pct})"
            )
        return v


class RateLimitConfig(BaseModel):
    """Configuration for rate limit simulation."""

    enabled: bool = Field(False, description="Enable rate limit simulation")
    requests_per_minute: int = Field(
        60, description="Maximum requests per minute", ge=1
    )
    burst_size: int = Field(10, description="Maximum burst size", ge=1)
    retry_after_seconds: int = Field(
        60, description="Retry-After header value when rate limited", ge=1
    )


class ChaosConfig(BaseModel):
    """Complete chaos engineering configuration."""

    enabled: bool = Field(True, description="Enable chaos engineering")
    tool_failures: list[ToolFailureConfig] = Field(
        default_factory=list, description="Tool failure injection rules"
    )
    latency: LatencyConfig | None = Field(None, description="Latency injection config")
    token_limits: TokenLimitConfig | None = Field(
        None, description="Token limit simulation config"
    )
    partial_response: PartialResponseConfig | None = Field(
        None, description="Partial response simulation config"
    )
    rate_limit: RateLimitConfig | None = Field(
        None, description="Rate limit simulation config"
    )
    seed: int | None = Field(
        None, description="Random seed for reproducible chaos (None for random)"
    )

    def is_empty(self) -> bool:
        """Check if chaos config has no active injections."""
        if not self.enabled:
            return True
        has_tool_failures = bool(self.tool_failures)
        has_latency = self.latency is not None and self.latency.max_ms > 0
        has_token_limits = self.token_limits is not None and (
            self.token_limits.max_input is not None
            or self.token_limits.max_output is not None
        )
        has_partial = (
            self.partial_response is not None and self.partial_response.enabled
        )
        has_rate_limit = self.rate_limit is not None and self.rate_limit.enabled
        return not any(
            [
                has_tool_failures,
                has_latency,
                has_token_limits,
                has_partial,
                has_rate_limit,
            ]
        )


class ChaosProfile(str, Enum):
    """Predefined chaos profiles for common test scenarios."""

    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    HIGH_LATENCY = "high_latency"
    FLAKY_TOOLS = "flaky_tools"
    RATE_LIMITED = "rate_limited"
    RESOURCE_CONSTRAINED = "resource_constrained"


# Default error messages for each error type
DEFAULT_ERROR_MESSAGES: dict[ErrorType, str] = {
    ErrorType.TIMEOUT: "Operation timed out",
    ErrorType.CONNECTION_ERROR: "Failed to connect to service",
    ErrorType.INTERNAL_ERROR: "Internal server error occurred",
    ErrorType.RATE_LIMIT: "Rate limit exceeded, please retry later",
    ErrorType.VALIDATION_ERROR: "Request validation failed",
    ErrorType.PERMISSION_DENIED: "Permission denied for this operation",
}
