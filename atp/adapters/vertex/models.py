"""Configuration models for the Google Vertex AI adapter."""

from typing import Any

from pydantic import Field, field_validator, model_validator

from atp.adapters.base import AdapterConfig


class VertexAdapterConfig(AdapterConfig):
    """Configuration for Google Vertex AI adapter."""

    # Project and location
    project_id: str = Field(..., description="Google Cloud project ID")
    location: str = Field(
        default="us-central1",
        description="Google Cloud region for Vertex AI",
    )

    # Agent identification
    agent_id: str | None = Field(
        None,
        description=(
            "Vertex AI Agent Builder agent ID. If not provided, "
            "uses direct Gemini model invocation."
        ),
    )
    agent_display_name: str | None = Field(
        None,
        description=("Display name for the agent (for Agent Builder)"),
    )

    # Model configuration (for direct model usage)
    model_name: str = Field(
        default="gemini-1.5-pro",
        description="Vertex AI model name for direct invocation",
    )

    # Authentication
    credentials_path: str | None = Field(
        None,
        description=(
            "Path to service account JSON key file. "
            "If not provided, uses Application Default "
            "Credentials."
        ),
    )
    service_account_email: str | None = Field(
        None,
        description=("Service account email for impersonation"),
    )

    # Session management
    session_id: str | None = Field(
        None,
        description=(
            "Session ID for conversation continuity. "
            "If not provided, a new session is created "
            "per request."
        ),
    )
    enable_session_persistence: bool = Field(
        default=False,
        description=("Persist session ID across requests for multi-turn conversations"),
    )

    # Generation settings
    temperature: float = Field(
        default=0.7,
        description="Temperature for model generation",
        ge=0.0,
        le=2.0,
    )
    max_output_tokens: int = Field(
        default=8192,
        description="Maximum tokens in the response",
        gt=0,
    )
    top_p: float = Field(
        default=0.95,
        description="Top-p (nucleus) sampling parameter",
        ge=0.0,
        le=1.0,
    )
    top_k: int | None = Field(None, description="Top-k sampling parameter", ge=0)

    # Tool configuration
    enable_function_calling: bool = Field(
        default=True,
        description="Enable function calling capabilities",
    )
    tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description=("List of tool definitions for function calling"),
    )

    # Safety settings
    safety_settings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Safety settings for content filtering",
    )
    block_threshold: str | None = Field(
        None,
        description=(
            "Default block threshold for all categories. "
            "Options: BLOCK_NONE, BLOCK_LOW_AND_ABOVE, "
            "BLOCK_MED_AND_ABOVE, BLOCK_HIGH_AND_ABOVE"
        ),
    )

    # Grounding
    enable_grounding: bool = Field(
        default=False,
        description="Enable Google Search grounding",
    )
    grounding_source: str | None = Field(
        None,
        description=(
            "Grounding source. Options: 'google_search', "
            "'vertex_ai_search', "
            "or a custom data store ID"
        ),
    )

    # System instruction
    system_instruction: str | None = Field(
        None,
        description="System instruction for the model",
    )

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, v: str) -> str:
        """Validate project ID format."""
        if not v or not v.strip():
            raise ValueError("project_id cannot be empty")
        return v.strip()

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        """Validate location/region."""
        v = v.strip().lower()
        if not v:
            raise ValueError("location cannot be empty")
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name."""
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_config(self) -> "VertexAdapterConfig":
        """Validate configuration combinations."""
        if self.grounding_source and not self.enable_grounding:
            raise ValueError(
                "enable_grounding must be True when grounding_source is provided"
            )

        return self
