"""Azure OpenAI adapter implementation."""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from pydantic import Field, field_validator, model_validator

from atp.protocol import (
    ArtifactStructured,
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
    Metrics,
    ResponseStatus,
)

from .base import AdapterConfig, AgentAdapter
from .exceptions import (
    AdapterConnectionError,
    AdapterError,
    AdapterTimeoutError,
)


class AzureOpenAIAdapterConfig(AdapterConfig):
    """Configuration for Azure OpenAI adapter."""

    # Azure OpenAI resource configuration
    endpoint: str = Field(
        ...,
        description=(
            "Azure OpenAI endpoint URL. "
            "Format: https://<resource-name>.openai.azure.com/"
        ),
    )
    deployment_name: str = Field(
        ...,
        description="Name of the Azure OpenAI model deployment",
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
            "Use Azure AD (Entra ID) authentication instead of API key. "
            "Requires azure-identity package."
        ),
    )
    tenant_id: str | None = Field(
        None,
        description="Azure AD tenant ID for authentication",
    )
    client_id: str | None = Field(
        None,
        description="Azure AD client/application ID for service principal auth",
    )
    client_secret: str | None = Field(
        None,
        description="Azure AD client secret for service principal auth",
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
        description="Enable function/tool calling capabilities",
    )
    tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of tool definitions for function calling",
    )
    tool_choice: str | dict[str, Any] | None = Field(
        None,
        description=(
            "Tool choice setting: 'auto', 'none', 'required', "
            "or specific tool specification"
        ),
    )

    # System message
    system_message: str | None = Field(
        None,
        description="System message to set the behavior of the assistant",
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
        description="Persist session across requests for multi-turn conversations",
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
        description="Seed for deterministic outputs (beta feature)",
    )

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Validate and normalize endpoint URL."""
        v = v.strip()
        if not v:
            raise ValueError("endpoint cannot be empty")
        # Remove trailing slash for consistency
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
    def validate_authentication(self) -> "AzureOpenAIAdapterConfig":
        """Validate authentication configuration."""
        if not self.api_key and not self.use_azure_ad:
            raise ValueError(
                "Either api_key or use_azure_ad=True must be provided "
                "for authentication"
            )

        # Service principal auth requires all three fields
        if self.client_id and not self.client_secret:
            raise ValueError("client_secret is required when client_id is provided")
        if self.client_secret and not self.client_id:
            raise ValueError("client_id is required when client_secret is provided")
        if (self.client_id or self.tenant_id) and not self.use_azure_ad:
            raise ValueError(
                "use_azure_ad must be True when using Azure AD credentials"
            )

        return self


class AzureOpenAIAdapter(AgentAdapter):
    """
    Adapter for Azure OpenAI Service.

    This adapter allows ATP to communicate with OpenAI models deployed
    on Azure OpenAI Service. It supports:
    - Chat completions with GPT models
    - Function/tool calling
    - Multi-turn conversations with session persistence
    - Both API key and Azure AD authentication

    Azure OpenAI requires the openai library to be installed:
        uv add openai

    For Azure AD authentication, also install:
        uv add azure-identity

    Authentication methods:
    - API Key: Provide api_key in config
    - Azure AD Default Credential: Set use_azure_ad=True
    - Service Principal: use_azure_ad=True with tenant_id, client_id, client_secret
    - Managed Identity: use_azure_ad=True with optional managed_identity_client_id
    """

    def __init__(self, config: AzureOpenAIAdapterConfig) -> None:
        """
        Initialize Azure OpenAI adapter.

        Args:
            config: Azure OpenAI adapter configuration.

        Raises:
            AdapterError: If openai library is not installed.
        """
        super().__init__(config)
        self._config: AzureOpenAIAdapterConfig = config
        self._client: Any = None
        self._session_id: str | None = config.session_id
        self._conversation_history: list[dict[str, Any]] = []
        self._initialized = False

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""
        return "azure_openai"

    @property
    def session_id(self) -> str | None:
        """Get the current session ID."""
        return self._session_id

    def _get_openai_client(self) -> Any:
        """
        Create or return the Azure OpenAI client.

        Returns:
            AzureOpenAI client instance.

        Raises:
            AdapterError: If openai is not installed.
            AdapterConnectionError: If client creation fails.
        """
        if self._client is not None:
            return self._client

        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise AdapterError(
                "openai is required for Azure OpenAI adapter. "
                "Install it with: uv add openai",
                adapter_type=self.adapter_type,
            ) from e

        try:
            client_kwargs: dict[str, Any] = {
                "azure_endpoint": self._config.endpoint,
                "api_version": self._config.api_version,
            }

            if self._config.api_key:
                client_kwargs["api_key"] = self._config.api_key
            elif self._config.use_azure_ad:
                # Get Azure AD token
                token_provider = self._get_azure_ad_token_provider()
                client_kwargs["azure_ad_token_provider"] = token_provider
            else:
                raise AdapterError(
                    "No authentication method configured",
                    adapter_type=self.adapter_type,
                )

            self._client = AzureOpenAI(**client_kwargs)
            self._initialized = True
            return self._client

        except Exception as e:
            if "openai" in str(type(e).__module__).lower():
                raise AdapterConnectionError(
                    f"Failed to create Azure OpenAI client: {e}",
                    adapter_type=self.adapter_type,
                    cause=e,
                ) from e
            raise

    def _get_azure_ad_token_provider(self) -> Any:
        """
        Get an Azure AD token provider function.

        Returns:
            Callable that returns Azure AD tokens.

        Raises:
            AdapterError: If azure-identity is not installed.
            AdapterConnectionError: If credential setup fails.
        """
        try:
            from azure.identity import (  # pyrefly: ignore[missing-import]
                ClientSecretCredential,
                DefaultAzureCredential,
                ManagedIdentityCredential,
                get_bearer_token_provider,
            )
        except ImportError as e:
            raise AdapterError(
                "azure-identity is required for Azure AD authentication. "
                "Install it with: uv add azure-identity",
                adapter_type=self.adapter_type,
            ) from e

        try:
            # Service principal authentication
            if self._config.client_id and self._config.client_secret:
                if not self._config.tenant_id:
                    raise AdapterError(
                        "tenant_id is required for service principal authentication",
                        adapter_type=self.adapter_type,
                    )
                credential = ClientSecretCredential(
                    tenant_id=self._config.tenant_id,
                    client_id=self._config.client_id,
                    client_secret=self._config.client_secret,
                )
            # Managed identity authentication
            elif self._config.managed_identity_client_id:
                credential = ManagedIdentityCredential(
                    client_id=self._config.managed_identity_client_id
                )
            # Default Azure credential chain
            else:
                credential = DefaultAzureCredential()

            # Create token provider for Azure OpenAI
            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default",
            )

            return token_provider

        except Exception as e:
            raise AdapterConnectionError(
                f"Failed to setup Azure AD authentication: {e}",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

    def _generate_session_id(self) -> str:
        """Generate a new session ID."""
        return str(uuid.uuid4())

    def _build_messages(self, request: ATPRequest) -> list[dict[str, Any]]:
        """
        Build the messages list for the API call.

        Args:
            request: ATP request.

        Returns:
            List of message dictionaries.
        """
        messages: list[dict[str, Any]] = []

        # Add system message if configured
        if self._config.system_message:
            messages.append(
                {
                    "role": "system",
                    "content": self._config.system_message,
                }
            )

        # Add conversation history for session persistence
        if self._config.enable_session_persistence:
            messages.extend(self._conversation_history)

        # Build user message from request
        content = request.task.description

        # Add context from input_data
        if request.task.input_data:
            context = request.task.input_data.get("context")
            if context:
                content = f"Context: {context}\n\n{content}"

        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

        return messages

    def _build_completion_params(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Build parameters for the chat completion API call.

        Args:
            messages: List of messages.

        Returns:
            Dictionary of API parameters.
        """
        params: dict[str, Any] = {
            "model": self._config.deployment_name,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "top_p": self._config.top_p,
            "frequency_penalty": self._config.frequency_penalty,
            "presence_penalty": self._config.presence_penalty,
        }

        # Add tools if configured
        if self._config.enable_function_calling and self._config.tools:
            params["tools"] = self._config.tools
            if self._config.tool_choice:
                params["tool_choice"] = self._config.tool_choice

        # Add response format if configured
        if self._config.response_format:
            params["response_format"] = self._config.response_format

        # Add seed if configured
        if self._config.seed is not None:
            params["seed"] = self._config.seed

        return params

    def _extract_tool_calls(
        self,
        message: Any,
        task_id: str,
        start_sequence: int,
    ) -> tuple[list[ATPEvent], int]:
        """
        Extract tool call events from a response message.

        Args:
            message: The assistant message from the response.
            task_id: Task ID for events.
            start_sequence: Starting sequence number.

        Returns:
            Tuple of (list of ATPEvent objects, next sequence number).
        """
        events: list[ATPEvent] = []
        sequence = start_sequence

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return events, sequence

        for tool_call in tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                arguments = {}

            events.append(
                ATPEvent(
                    task_id=task_id,
                    sequence=sequence,
                    event_type=EventType.TOOL_CALL,
                    payload={
                        "tool": tool_call.function.name,
                        "function": tool_call.function.name,
                        "tool_call_id": tool_call.id,
                        "input": arguments,
                        "status": "started",
                    },
                )
            )
            sequence += 1

        return events, sequence

    def _extract_usage(self, response: Any) -> dict[str, int]:
        """
        Extract token usage from response.

        Args:
            response: Chat completion response.

        Returns:
            Dictionary with token usage.
        """
        usage: dict[str, int] = {}

        if hasattr(response, "usage") and response.usage:
            if hasattr(response.usage, "prompt_tokens"):
                usage["input_tokens"] = response.usage.prompt_tokens
            if hasattr(response.usage, "completion_tokens"):
                usage["output_tokens"] = response.usage.completion_tokens
            if hasattr(response.usage, "total_tokens"):
                usage["total_tokens"] = response.usage.total_tokens

        return usage

    async def _create_completion(
        self,
        request: ATPRequest,
        session_id: str,
    ) -> tuple[Any, dict[str, Any]]:
        """
        Create a chat completion.

        Args:
            request: ATP request.
            session_id: Session ID for the invocation.

        Returns:
            Tuple of (response object, metrics info).

        Raises:
            AdapterTimeoutError: If completion times out.
            AdapterError: If completion fails.
        """
        client = self._get_openai_client()
        messages = self._build_messages(request)
        params = self._build_completion_params(messages)

        try:
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: client.chat.completions.create(**params),
                ),
                timeout=self._config.timeout_seconds,
            )

            metrics_info: dict[str, Any] = {
                "session_id": session_id,
                "deployment": self._config.deployment_name,
                "model": getattr(response, "model", self._config.deployment_name),
            }

            return response, metrics_info

        except TimeoutError as e:
            raise AdapterTimeoutError(
                f"Azure OpenAI completion timed out after "
                f"{self._config.timeout_seconds}s",
                adapter_type=self.adapter_type,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__

            # Handle specific Azure OpenAI errors
            if "AuthenticationError" in error_type or "401" in error_str:
                raise AdapterConnectionError(
                    "Authentication failed. Check API key or Azure AD credentials.",
                    adapter_type=self.adapter_type,
                    cause=e,
                ) from e
            if "PermissionDenied" in error_type or "403" in error_str:
                raise AdapterConnectionError(
                    "Permission denied. Check Azure RBAC permissions.",
                    adapter_type=self.adapter_type,
                    cause=e,
                ) from e
            if "NotFound" in error_type or "404" in error_str:
                raise AdapterError(
                    f"Deployment not found: {self._config.deployment_name}",
                    adapter_type=self.adapter_type,
                ) from e
            if "RateLimitError" in error_type or "429" in error_str:
                raise AdapterError(
                    "Rate limit exceeded. Consider implementing retry logic.",
                    adapter_type=self.adapter_type,
                ) from e
            if "BadRequestError" in error_type or "400" in error_str:
                raise AdapterError(
                    f"Bad request: {e}",
                    adapter_type=self.adapter_type,
                ) from e

            raise AdapterError(
                f"Azure OpenAI completion failed: {e}",
                adapter_type=self.adapter_type,
            ) from e

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """
        Execute a task via Azure OpenAI.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse with execution results.

        Raises:
            AdapterConnectionError: If connection to Azure fails.
            AdapterTimeoutError: If execution times out.
            AdapterError: If completion fails.
        """
        start_time = time.time()

        # Determine session ID
        if self._config.enable_session_persistence and self._session_id:
            session_id = self._session_id
        else:
            session_id = self._generate_session_id()

        try:
            response, metrics_info = await self._create_completion(request, session_id)

            # Update session ID if persistence is enabled
            if self._config.enable_session_persistence:
                self._session_id = session_id

            wall_time = time.time() - start_time

            # Extract response content
            message = response.choices[0].message
            response_text = message.content or ""
            usage = self._extract_usage(response)

            # Extract tool calls
            tool_events, _ = self._extract_tool_calls(message, request.task_id, 0)
            tool_calls = len(tool_events)

            # Update conversation history for session persistence
            if self._config.enable_session_persistence:
                self._conversation_history.append(
                    {
                        "role": "user",
                        "content": request.task.description,
                    }
                )
                self._conversation_history.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                    }
                )

            # Build artifacts
            artifacts = [
                ArtifactStructured(
                    name="output",
                    data={
                        "text": response_text,
                        "session_id": session_id,
                        "deployment": self._config.deployment_name,
                        "model": metrics_info.get("model"),
                        "finish_reason": response.choices[0].finish_reason,
                    },
                    content_type="text/plain",
                )
            ]

            # Add tool calls artifact if any
            if tool_events:
                artifacts.append(
                    ArtifactStructured(
                        name="tool_calls",
                        data={
                            "calls": [
                                {
                                    "tool": e.payload.get("tool"),
                                    "tool_call_id": e.payload.get("tool_call_id"),
                                    "input": e.payload.get("input"),
                                }
                                for e in tool_events
                            ]
                        },
                        content_type="application/json",
                    )
                )

            return ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
                artifacts=artifacts,
                metrics=Metrics(
                    wall_time_seconds=wall_time,
                    tool_calls=tool_calls,
                    llm_calls=1,
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                ),
                trace_id=session_id,
            )

        except (AdapterTimeoutError, AdapterConnectionError, AdapterError):
            raise
        except Exception as e:
            wall_time = time.time() - start_time
            return ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error=str(e),
                metrics=Metrics(wall_time_seconds=wall_time),
            )

    async def stream_events(
        self,
        request: ATPRequest,
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """
        Execute a task with event streaming.

        Yields ATP events during execution and ends with the final response.

        Args:
            request: ATP Request with task specification.

        Yields:
            ATPEvent objects during execution.
            Final ATPResponse when complete.

        Raises:
            AdapterConnectionError: If connection to Azure fails.
            AdapterTimeoutError: If execution times out.
            AdapterError: If completion fails.
        """
        start_time = time.time()
        sequence = 0

        # Determine session ID
        if self._config.enable_session_persistence and self._session_id:
            session_id = self._session_id
        else:
            session_id = self._generate_session_id()

        # Emit start event
        yield ATPEvent(
            task_id=request.task_id,
            sequence=sequence,
            event_type=EventType.PROGRESS,
            payload={
                "message": "Starting Azure OpenAI completion",
                "session_id": session_id,
                "deployment": self._config.deployment_name,
            },
        )
        sequence += 1

        try:
            client = self._get_openai_client()
            messages = self._build_messages(request)
            params = self._build_completion_params(messages)
            params["stream"] = True

            # Emit LLM request event
            yield ATPEvent(
                task_id=request.task_id,
                sequence=sequence,
                event_type=EventType.LLM_REQUEST,
                payload={
                    "type": "chat_completion",
                    "deployment": self._config.deployment_name,
                    "message_count": len(messages),
                },
            )
            sequence += 1

            # Create streaming completion
            loop = asyncio.get_running_loop()
            stream = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: client.chat.completions.create(**params),
                ),
                timeout=self._config.timeout_seconds,
            )

            # Process streaming response
            full_text = ""
            tool_calls: list[dict[str, Any]] = []
            tool_call_accumulators: dict[int, dict[str, Any]] = {}
            finish_reason = None
            model_name = None

            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason or finish_reason
                model_name = getattr(chunk, "model", model_name)

                # Handle text content
                if delta.content:
                    full_text += delta.content
                    yield ATPEvent(
                        task_id=request.task_id,
                        sequence=sequence,
                        event_type=EventType.PROGRESS,
                        payload={
                            "type": "chunk",
                            "text": delta.content,
                        },
                    )
                    sequence += 1

                # Handle tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_accumulators:
                            tool_call_accumulators[idx] = {
                                "id": tc.id or "",
                                "function_name": "",
                                "arguments": "",
                            }

                        if tc.id:
                            tool_call_accumulators[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_call_accumulators[idx]["function_name"] = (
                                tc.function.name
                            )
                        if tc.function and tc.function.arguments:
                            tool_call_accumulators[idx]["arguments"] += (
                                tc.function.arguments
                            )

            # Process accumulated tool calls
            for idx in sorted(tool_call_accumulators.keys()):
                tc_data = tool_call_accumulators[idx]
                try:
                    arguments = json.loads(tc_data["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                tool_calls.append(
                    {
                        "tool": tc_data["function_name"],
                        "tool_call_id": tc_data["id"],
                        "input": arguments,
                    }
                )

                yield ATPEvent(
                    task_id=request.task_id,
                    sequence=sequence,
                    event_type=EventType.TOOL_CALL,
                    payload={
                        "tool": tc_data["function_name"],
                        "function": tc_data["function_name"],
                        "tool_call_id": tc_data["id"],
                        "input": arguments,
                        "status": "started",
                    },
                )
                sequence += 1

            # Update session ID if persistence is enabled
            if self._config.enable_session_persistence:
                self._session_id = session_id
                self._conversation_history.append(
                    {
                        "role": "user",
                        "content": request.task.description,
                    }
                )
                self._conversation_history.append(
                    {
                        "role": "assistant",
                        "content": full_text,
                    }
                )

            wall_time = time.time() - start_time

            # Build artifacts
            artifacts = [
                ArtifactStructured(
                    name="output",
                    data={
                        "text": full_text,
                        "session_id": session_id,
                        "deployment": self._config.deployment_name,
                        "model": model_name,
                        "finish_reason": finish_reason,
                    },
                    content_type="text/plain",
                )
            ]

            if tool_calls:
                artifacts.append(
                    ArtifactStructured(
                        name="tool_calls",
                        data={"calls": tool_calls},
                        content_type="application/json",
                    )
                )

            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
                artifacts=artifacts,
                metrics=Metrics(
                    wall_time_seconds=wall_time,
                    tool_calls=len(tool_calls),
                    llm_calls=1,
                ),
                trace_id=session_id,
            )

        except TimeoutError:
            wall_time = time.time() - start_time
            yield ATPEvent(
                task_id=request.task_id,
                sequence=sequence,
                event_type=EventType.ERROR,
                payload={
                    "error_type": "timeout",
                    "message": (
                        f"Azure OpenAI completion timed out after "
                        f"{self._config.timeout_seconds}s"
                    ),
                    "recoverable": False,
                },
            )
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.TIMEOUT,
                error=(
                    f"Azure OpenAI completion timed out after "
                    f"{self._config.timeout_seconds}s"
                ),
                metrics=Metrics(wall_time_seconds=wall_time),
            )

        except Exception as e:
            wall_time = time.time() - start_time
            yield ATPEvent(
                task_id=request.task_id,
                sequence=sequence,
                event_type=EventType.ERROR,
                payload={
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "recoverable": False,
                },
            )
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error=str(e),
                metrics=Metrics(wall_time_seconds=wall_time),
            )

    async def health_check(self) -> bool:
        """
        Check if Azure OpenAI is accessible.

        Returns:
            True if Azure OpenAI is accessible, False otherwise.
        """
        try:
            self._get_openai_client()
            return True
        except (AdapterError, AdapterConnectionError):
            return False

    async def cleanup(self) -> None:
        """Release resources."""
        self._client = None
        if not self._config.enable_session_persistence:
            self._session_id = None
            self._conversation_history = []
        self._initialized = False

    def reset_session(self) -> None:
        """Reset the session for a new conversation."""
        self._session_id = None
        self._conversation_history = []

    def set_session_id(self, session_id: str) -> None:
        """
        Set a specific session ID.

        Args:
            session_id: Session ID to use for subsequent requests.
        """
        self._session_id = session_id
