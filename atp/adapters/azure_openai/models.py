"""Configuration models for the Azure OpenAI adapter."""

from typing import Any

from pydantic import Field, field_validator, model_validator

from atp.adapters.base import AdapterConfig


class AzureOpenAIAdapterConfig(AdapterConfig):
    """Configuration for Azure OpenAI adapter."""

    # Azure OpenAI resource configuration
    endpoint: str = Field(
        ...,
        description=(
            "Azure OpenAI endpoint URL. "
            "Format: https://<resource-name>"
            ".openai.azure.com/"
        ),
    )
    deployment_name: str = Field(
        ...,
        description=("Name of the Azure OpenAI model deployment"),
    )
    api_version: str = Field(
        default="2024-02-15-preview",
        description="Azure OpenAI API version",
    )

    # Authentication - API Key
    api_key: str | None = Field(
        None,
        description=(
            "Azure OpenAI API key. If not provided, uses Azure AD authentication."
        ),
    )

    # Authentication - Azure AD
    use_azure_ad: bool = Field(
        default=False,
        description=(
            "Use Azure AD (Entra ID) authentication "
            "instead of API key. "
            "Requires azure-identity package."
        ),
    )
    tenant_id: str | None = Field(
        None,
        description=("Azure AD tenant ID for authentication"),
    )
    client_id: str | None = Field(
        None,
        description=("Azure AD client/application ID for service principal auth"),
    )
    client_secret: str | None = Field(
        None,
        description=("Azure AD client secret for service principal auth"),
    )
    managed_identity_client_id: str | None = Field(
        None,
        description=(
            "Client ID for user-assigned managed identity. "
            "If not provided, uses system-assigned identity."
        ),
    )

    # Region/Location
    azure_region: str | None = Field(
        None,
        description=(
            "Azure region where the resource is deployed "
            "(e.g., 'eastus', 'westeurope'). "
            "Informational and does not affect connectivity."
        ),
    )

    # Model parameters
    temperature: float = Field(
        default=0.7,
        description="Temperature for model generation",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens in the response",
        gt=0,
    )
    top_p: float = Field(
        default=1.0,
        description="Top-p (nucleus) sampling parameter",
        ge=0.0,
        le=1.0,
    )
    frequency_penalty: float = Field(
        default=0.0,
        description="Frequency penalty for generation",
        ge=-2.0,
        le=2.0,
    )
    presence_penalty: float = Field(
        default=0.0,
        description="Presence penalty for generation",
        ge=-2.0,
        le=2.0,
    )

    # Tool/function calling
    enable_function_calling: bool = Field(
        default=True,
        description=("Enable function/tool calling capabilities"),
    )
    tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description=("List of tool definitions for function calling"),
    )
    tool_choice: str | dict[str, Any] | None = Field(
        None,
        description=(
            "Tool choice setting: 'auto', 'none', "
            "'required', or specific tool specification"
        ),
    )

    # System message
    system_message: str | None = Field(
        None,
        description=("System message to set the behavior of the assistant"),
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
        description=("Persist session across requests for multi-turn conversations"),
    )

    # Response format
    response_format: dict[str, Any] | None = Field(
        None,
        description=(
            "Response format specification. E.g., {'type': 'json_object'} for JSON mode"
        ),
    )

    # Seed for reproducibility
    seed: int | None = Field(
        None,
        description=("Seed for deterministic outputs (beta feature)"),
    )

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Validate and normalize endpoint URL."""
        v = v.strip()
        if not v:
            raise ValueError("endpoint cannot be empty")
        return v.rstrip("/")

    @field_validator("deployment_name")
    @classmethod
    def validate_deployment_name(cls, v: str) -> str:
        """Validate deployment name."""
        if not v or not v.strip():
            raise ValueError("deployment_name cannot be empty")
        return v.strip()

    @field_validator("api_version")
    @classmethod
    def validate_api_version(cls, v: str) -> str:
        """Validate API version."""
        if not v or not v.strip():
            raise ValueError("api_version cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_authentication(
        self,
    ) -> "AzureOpenAIAdapterConfig":
        """Validate authentication configuration."""
        if not self.api_key and not self.use_azure_ad:
            raise ValueError(
                "Either api_key or use_azure_ad=True "
                "must be provided for authentication"
            )

        if self.client_id and not self.client_secret:
            raise ValueError("client_secret is required when client_id is provided")
        if self.client_secret and not self.client_id:
            raise ValueError("client_id is required when client_secret is provided")
        if (self.client_id or self.tenant_id) and not self.use_azure_ad:
            raise ValueError(
                "use_azure_ad must be True when using Azure AD credentials"
            )

        return self
