"""Configuration models for the AWS Bedrock adapter."""

from typing import Any

from pydantic import Field, field_validator, model_validator

from atp.adapters.base import AdapterConfig


class BedrockAdapterConfig(AdapterConfig):
    """Configuration for AWS Bedrock Agents adapter."""

    # Agent identification
    agent_id: str = Field(..., description="Bedrock Agent ID")
    agent_alias_id: str = Field(
        default="TSTALIASID",
        description="Bedrock Agent alias ID (default: TSTALIASID for draft)",
    )

    # AWS configuration
    region: str = Field(default="us-east-1", description="AWS region")
    profile: str | None = Field(
        None, description="AWS profile name for credential resolution"
    )
    access_key_id: str | None = Field(None, description="AWS access key ID")
    secret_access_key: str | None = Field(None, description="AWS secret access key")
    session_token: str | None = Field(None, description="AWS session token")
    endpoint_url: str | None = Field(
        None,
        description="Custom endpoint URL for Bedrock (for testing)",
    )

    # Session management
    session_id: str | None = Field(
        None,
        description=(
            "Session ID for conversation continuity. "
            "If not provided, a new session is created per request."
        ),
    )
    enable_session_persistence: bool = Field(
        default=False,
        description=("Persist session ID across requests for multi-turn conversations"),
    )
    session_ttl_seconds: int = Field(
        default=3600,
        description="Session time-to-live in seconds",
        gt=0,
    )

    # Knowledge base configuration
    knowledge_base_ids: list[str] = Field(
        default_factory=list,
        description="List of knowledge base IDs to attach to the agent",
    )
    retrieve_and_generate: bool = Field(
        default=False,
        description=(
            "Use retrieve and generate mode for knowledge base queries. "
            "When enabled, the agent retrieves relevant documents and "
            "generates responses based on them."
        ),
    )
    retrieval_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for knowledge base retrieval",
    )

    # Action group configuration
    action_groups: list[str] = Field(
        default_factory=list,
        description="List of action group names to enable",
    )

    # Tracing and observability
    enable_trace: bool = Field(
        default=True,
        description="Enable trace output from Bedrock agent",
    )
    trace_include_reasoning: bool = Field(
        default=True,
        description="Include reasoning steps in trace events",
    )

    # Memory and context
    memory_id: str | None = Field(
        None,
        description="Memory ID for agent memory feature",
    )

    # Guardrails
    guardrail_identifier: str | None = Field(
        None,
        description="Guardrail identifier for content filtering",
    )
    guardrail_version: str | None = Field(
        None,
        description=("Guardrail version (required if guardrail_identifier is set)"),
    )

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        """Validate agent ID format."""
        if not v or not v.strip():
            raise ValueError("agent_id cannot be empty")
        return v.strip()

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validate AWS region."""
        v = v.strip().lower()
        if not v:
            raise ValueError("region cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_credentials(self) -> "BedrockAdapterConfig":
        """Validate credential configuration."""
        if self.access_key_id and not self.secret_access_key:
            raise ValueError(
                "secret_access_key is required when access_key_id is provided"
            )
        if self.secret_access_key and not self.access_key_id:
            raise ValueError(
                "access_key_id is required when secret_access_key is provided"
            )

        if self.guardrail_identifier and not self.guardrail_version:
            raise ValueError(
                "guardrail_version is required when guardrail_identifier is set"
            )

        return self
