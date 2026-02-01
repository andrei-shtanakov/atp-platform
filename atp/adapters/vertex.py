"""Google Vertex AI adapter implementation."""

import asyncio
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


class VertexAdapterConfig(AdapterConfig):
    """Configuration for Google Vertex AI adapter."""

    # Project and location
    project_id: str = Field(..., description="Google Cloud project ID")
    location: str = Field(
        default="us-central1", description="Google Cloud region for Vertex AI"
    )

    # Agent identification
    agent_id: str | None = Field(
        None,
        description=(
            "Vertex AI Agent Builder agent ID. If not provided, uses "
            "direct Gemini model invocation."
        ),
    )
    agent_display_name: str | None = Field(
        None, description="Display name for the agent (for Agent Builder)"
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
            "If not provided, uses Application Default Credentials."
        ),
    )
    service_account_email: str | None = Field(
        None, description="Service account email for impersonation"
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
        description="Persist session ID across requests for multi-turn conversations",
    )

    # Generation settings
    temperature: float = Field(
        default=0.7, description="Temperature for model generation", ge=0.0, le=2.0
    )
    max_output_tokens: int = Field(
        default=8192, description="Maximum tokens in the response", gt=0
    )
    top_p: float = Field(
        default=0.95, description="Top-p (nucleus) sampling parameter", ge=0.0, le=1.0
    )
    top_k: int | None = Field(None, description="Top-k sampling parameter", ge=0)

    # Tool configuration
    enable_function_calling: bool = Field(
        default=True, description="Enable function calling capabilities"
    )
    tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of tool definitions for function calling",
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
            "Options: BLOCK_NONE, BLOCK_LOW_AND_ABOVE, BLOCK_MED_AND_ABOVE, "
            "BLOCK_HIGH_AND_ABOVE"
        ),
    )

    # Grounding
    enable_grounding: bool = Field(
        default=False, description="Enable Google Search grounding"
    )
    grounding_source: str | None = Field(
        None,
        description=(
            "Grounding source. Options: 'google_search', 'vertex_ai_search', "
            "or a custom data store ID"
        ),
    )

    # System instruction
    system_instruction: str | None = Field(
        None, description="System instruction for the model"
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
        # Grounding source requires grounding to be enabled
        if self.grounding_source and not self.enable_grounding:
            raise ValueError(
                "enable_grounding must be True when grounding_source is provided"
            )

        return self


class VertexAdapter(AgentAdapter):
    """
    Adapter for Google Vertex AI.

    This adapter allows ATP to communicate with:
    - Vertex AI Agent Builder agents
    - Vertex AI Gemini models directly

    It supports:
    - Agent/model invocation with session management
    - Function calling (tool use)
    - Google Search grounding
    - Multi-turn conversations with session persistence
    - Safety settings and content filtering

    Vertex AI requires the google-cloud-aiplatform library to be installed:
        uv add google-cloud-aiplatform

    Google Cloud credentials can be provided via:
    - Explicit credentials_path to a service account JSON key file
    - Service account impersonation via service_account_email
    - Application Default Credentials (gcloud auth application-default login)
    - GCE/GKE service account when running on Google Cloud infrastructure
    """

    def __init__(self, config: VertexAdapterConfig) -> None:
        """
        Initialize Vertex AI adapter.

        Args:
            config: Vertex AI adapter configuration.

        Raises:
            AdapterError: If google-cloud-aiplatform is not installed.
        """
        super().__init__(config)
        self._config: VertexAdapterConfig = config
        self._client: Any = None
        self._model: Any = None
        self._chat_session: Any = None
        self._session_id: str | None = config.session_id
        self._initialized = False
        self._conversation_history: list[dict[str, Any]] = []

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""
        return "vertex"

    @property
    def session_id(self) -> str | None:
        """Get the current session ID."""
        return self._session_id

    def _get_vertexai_module(self) -> Any:
        """
        Import and return the vertexai module.

        Returns:
            The vertexai module.

        Raises:
            AdapterError: If google-cloud-aiplatform is not installed.
        """
        try:
            import vertexai

            return vertexai
        except ImportError as e:
            raise AdapterError(
                "google-cloud-aiplatform is required for Vertex AI adapter. "
                "Install it with: uv add google-cloud-aiplatform",
                adapter_type=self.adapter_type,
            ) from e

    def _initialize_vertexai(self) -> None:
        """
        Initialize Vertex AI SDK.

        Raises:
            AdapterConnectionError: If initialization fails.
        """
        if self._initialized:
            return

        vertexai = self._get_vertexai_module()

        try:
            init_kwargs: dict[str, Any] = {
                "project": self._config.project_id,
                "location": self._config.location,
            }

            # Handle credentials
            if self._config.credentials_path:
                from google.oauth2 import service_account

                credentials = service_account.Credentials.from_service_account_file(
                    self._config.credentials_path
                )
                init_kwargs["credentials"] = credentials
            elif self._config.service_account_email:
                import google.auth
                from google.auth import impersonated_credentials

                source_credentials, _ = google.auth.default()
                credentials = impersonated_credentials.Credentials(
                    source_credentials=source_credentials,
                    target_principal=self._config.service_account_email,
                    target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                init_kwargs["credentials"] = credentials

            vertexai.init(**init_kwargs)
            self._initialized = True

        except Exception as e:
            raise AdapterConnectionError(
                f"Failed to initialize Vertex AI: {e}",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

    def _get_model(self) -> Any:
        """
        Get or create the Generative Model instance.

        Returns:
            GenerativeModel instance.

        Raises:
            AdapterError: If model creation fails.
        """
        if self._model is not None:
            return self._model

        self._initialize_vertexai()

        try:
            from vertexai.generative_models import (
                GenerativeModel,
                SafetySetting,
            )

            model_kwargs: dict[str, Any] = {}

            # Add system instruction if provided
            if self._config.system_instruction:
                model_kwargs["system_instruction"] = self._config.system_instruction

            # Configure safety settings
            if self._config.safety_settings:
                safety_settings = []
                for setting in self._config.safety_settings:
                    category = setting.get("category")
                    threshold = setting.get("threshold")
                    if category is not None and threshold is not None:
                        safety_settings.append(
                            SafetySetting(
                                category=category,
                                threshold=threshold,
                            )
                        )
                model_kwargs["safety_settings"] = safety_settings
            elif self._config.block_threshold:
                # Apply default threshold to all categories
                from vertexai.generative_models import HarmBlockThreshold, HarmCategory

                threshold = getattr(
                    HarmBlockThreshold, self._config.block_threshold, None
                )
                if threshold:
                    safety_settings = []
                    for category in HarmCategory:
                        safety_settings.append(
                            SafetySetting(category=category, threshold=threshold)
                        )
                    model_kwargs["safety_settings"] = safety_settings

            # Configure tools if enabled
            if self._config.enable_function_calling and self._config.tools:
                tools = self._build_tools(self._config.tools)
                if tools:
                    model_kwargs["tools"] = tools

            self._model = GenerativeModel(
                self._config.model_name,
                **model_kwargs,
            )

            return self._model

        except Exception as e:
            raise AdapterError(
                f"Failed to create Vertex AI model: {e}",
                adapter_type=self.adapter_type,
            ) from e

    def _build_tools(self, tool_configs: list[dict[str, Any]]) -> list[Any]:
        """
        Build Tool objects from configuration.

        Args:
            tool_configs: List of tool configuration dictionaries.

        Returns:
            List of Tool objects.
        """
        from vertexai.generative_models import FunctionDeclaration, Tool

        function_declarations = []

        for tool_config in tool_configs:
            if "function_declarations" in tool_config:
                # Tool config contains function declarations
                for func_config in tool_config["function_declarations"]:
                    func_decl = FunctionDeclaration(
                        name=func_config.get("name", ""),
                        description=func_config.get("description", ""),
                        parameters=func_config.get("parameters", {}),
                    )
                    function_declarations.append(func_decl)
            elif "name" in tool_config:
                # Direct function declaration
                func_decl = FunctionDeclaration(
                    name=tool_config.get("name", ""),
                    description=tool_config.get("description", ""),
                    parameters=tool_config.get("parameters", {}),
                )
                function_declarations.append(func_decl)

        if function_declarations:
            return [Tool(function_declarations=function_declarations)]

        return []

    def _generate_session_id(self) -> str:
        """Generate a new session ID."""
        return str(uuid.uuid4())

    def _get_generation_config(self) -> dict[str, Any]:
        """
        Build generation configuration.

        Returns:
            Dictionary of generation parameters.
        """
        config: dict[str, Any] = {
            "temperature": self._config.temperature,
            "max_output_tokens": self._config.max_output_tokens,
            "top_p": self._config.top_p,
        }

        if self._config.top_k is not None:
            config["top_k"] = self._config.top_k

        return config

    def _extract_tool_calls(
        self,
        response: Any,
        task_id: str,
        start_sequence: int,
    ) -> tuple[list[ATPEvent], int]:
        """
        Extract tool call events from a response.

        Args:
            response: Vertex AI response object.
            task_id: Task ID for events.
            start_sequence: Starting sequence number.

        Returns:
            Tuple of (list of ATPEvent objects, next sequence number).
        """
        events: list[ATPEvent] = []
        sequence = start_sequence

        for candidate in response.candidates:
            for part in candidate.content.parts:
                # Check for function call
                if hasattr(part, "function_call") and part.function_call:
                    func_call = part.function_call
                    events.append(
                        ATPEvent(
                            task_id=task_id,
                            sequence=sequence,
                            event_type=EventType.TOOL_CALL,
                            payload={
                                "tool": func_call.name,
                                "function": func_call.name,
                                "input": dict(func_call.args) if func_call.args else {},
                                "status": "started",
                            },
                        )
                    )
                    sequence += 1

        return events, sequence

    def _extract_response_text(self, response: Any) -> str:
        """
        Extract text content from a response.

        Args:
            response: Vertex AI response object.

        Returns:
            Combined text from all text parts.
        """
        text_parts: list[str] = []

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

        return "".join(text_parts)

    def _extract_usage_metadata(self, response: Any) -> dict[str, int]:
        """
        Extract token usage metadata from response.

        Args:
            response: Vertex AI response object.

        Returns:
            Dictionary with input_tokens and output_tokens.
        """
        metadata: dict[str, int] = {}

        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            if hasattr(usage, "prompt_token_count"):
                metadata["input_tokens"] = usage.prompt_token_count
            if hasattr(usage, "candidates_token_count"):
                metadata["output_tokens"] = usage.candidates_token_count
            if hasattr(usage, "total_token_count"):
                metadata["total_tokens"] = usage.total_token_count

        return metadata

    def _extract_grounding_metadata(self, response: Any) -> dict[str, Any] | None:
        """
        Extract grounding metadata from response.

        Args:
            response: Vertex AI response object.

        Returns:
            Grounding metadata dictionary or None.
        """
        for candidate in response.candidates:
            if (
                hasattr(candidate, "grounding_metadata")
                and candidate.grounding_metadata
            ):
                grounding = candidate.grounding_metadata
                result: dict[str, Any] = {}

                if hasattr(grounding, "search_entry_point"):
                    result["search_entry_point"] = str(grounding.search_entry_point)

                if hasattr(grounding, "grounding_chunks"):
                    chunks = []
                    for chunk in grounding.grounding_chunks:
                        chunk_data: dict[str, Any] = {}
                        if hasattr(chunk, "web") and chunk.web:
                            chunk_data["web"] = {
                                "uri": chunk.web.uri
                                if hasattr(chunk.web, "uri")
                                else None,
                                "title": chunk.web.title
                                if hasattr(chunk.web, "title")
                                else None,
                            }
                        chunks.append(chunk_data)
                    result["grounding_chunks"] = chunks

                if hasattr(grounding, "grounding_supports"):
                    supports = []
                    for support in grounding.grounding_supports:
                        support_data: dict[str, Any] = {}
                        if hasattr(support, "segment"):
                            support_data["segment"] = str(support.segment)
                        if hasattr(support, "grounding_chunk_indices"):
                            support_data["chunk_indices"] = list(
                                support.grounding_chunk_indices
                            )
                        supports.append(support_data)
                    result["grounding_supports"] = supports

                return result

        return None

    async def _generate_content(
        self,
        request: ATPRequest,
        session_id: str,
    ) -> tuple[Any, dict[str, Any]]:
        """
        Generate content using Vertex AI.

        Args:
            request: ATP request.
            session_id: Session ID for the invocation.

        Returns:
            Tuple of (response object, metrics info).

        Raises:
            AdapterTimeoutError: If generation times out.
            AdapterError: If generation fails.
        """
        model = self._get_model()
        generation_config = self._get_generation_config()

        # Build the prompt/content
        content = request.task.description

        # Add any additional context from input_data
        if request.task.input_data:
            if "context" in request.task.input_data:
                content = f"Context: {request.task.input_data['context']}\n\n{content}"

        try:
            # For session persistence, use chat
            if self._config.enable_session_persistence:
                if self._chat_session is None:
                    self._chat_session = model.start_chat(
                        history=self._conversation_history
                    )

                # Run generation in executor to avoid blocking
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self._chat_session.send_message(
                            content,
                            generation_config=generation_config,
                        ),
                    ),
                    timeout=self._config.timeout_seconds,
                )
            else:
                # Single-turn generation
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: model.generate_content(
                            content,
                            generation_config=generation_config,
                        ),
                    ),
                    timeout=self._config.timeout_seconds,
                )

            metrics_info: dict[str, Any] = {
                "session_id": session_id,
                "model": self._config.model_name,
            }

            return response, metrics_info

        except TimeoutError as e:
            raise AdapterTimeoutError(
                f"Vertex AI generation timed out after {self._config.timeout_seconds}s",
                adapter_type=self.adapter_type,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except Exception as e:
            error_str = str(e)
            if "PermissionDenied" in error_str or "403" in error_str:
                raise AdapterConnectionError(
                    "Permission denied to Vertex AI. Check Google Cloud credentials "
                    "and permissions.",
                    adapter_type=self.adapter_type,
                    cause=e,
                ) from e
            if "NotFound" in error_str or "404" in error_str:
                raise AdapterError(
                    f"Model not found: {self._config.model_name}",
                    adapter_type=self.adapter_type,
                ) from e
            if "ResourceExhausted" in error_str or "429" in error_str:
                raise AdapterError(
                    "Vertex AI quota exceeded. Consider implementing retry logic.",
                    adapter_type=self.adapter_type,
                ) from e
            raise AdapterError(
                f"Vertex AI generation failed: {e}",
                adapter_type=self.adapter_type,
            ) from e

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """
        Execute a task via Vertex AI.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse with execution results.

        Raises:
            AdapterConnectionError: If connection to Google Cloud fails.
            AdapterTimeoutError: If execution times out.
            AdapterError: If generation fails.
        """
        start_time = time.time()

        # Determine session ID
        if self._config.enable_session_persistence and self._session_id:
            session_id = self._session_id
        else:
            session_id = self._generate_session_id()

        try:
            response, metrics_info = await self._generate_content(request, session_id)

            # Update session ID if persistence is enabled
            if self._config.enable_session_persistence:
                self._session_id = session_id

            wall_time = time.time() - start_time

            # Extract response content
            response_text = self._extract_response_text(response)
            usage = self._extract_usage_metadata(response)
            grounding = self._extract_grounding_metadata(response)

            # Extract tool calls
            tool_events, _ = self._extract_tool_calls(response, request.task_id, 0)
            tool_calls = len(tool_events)

            # Build artifacts
            artifacts = [
                ArtifactStructured(
                    name="output",
                    data={
                        "text": response_text,
                        "session_id": session_id,
                        "model": self._config.model_name,
                    },
                    content_type="text/plain",
                )
            ]

            # Add grounding artifact if available
            if grounding:
                artifacts.append(
                    ArtifactStructured(
                        name="grounding",
                        data=grounding,
                        content_type="application/json",
                    )
                )

            # Add tool calls artifact if any
            if tool_events:
                artifacts.append(
                    ArtifactStructured(
                        name="tool_calls",
                        data={
                            "calls": [
                                {
                                    "tool": e.payload.get("tool"),
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
            AdapterConnectionError: If connection to Google Cloud fails.
            AdapterTimeoutError: If execution times out.
            AdapterError: If generation fails.
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
                "message": "Starting Vertex AI generation",
                "session_id": session_id,
                "model": self._config.model_name,
            },
        )
        sequence += 1

        try:
            model = self._get_model()
            generation_config = self._get_generation_config()

            # Build the prompt/content
            content = request.task.description

            # Add any additional context from input_data
            if request.task.input_data:
                if "context" in request.task.input_data:
                    content = (
                        f"Context: {request.task.input_data['context']}\n\n{content}"
                    )

            # Emit LLM request event
            yield ATPEvent(
                task_id=request.task_id,
                sequence=sequence,
                event_type=EventType.LLM_REQUEST,
                payload={
                    "type": "model_invocation",
                    "model": self._config.model_name,
                    "prompt_preview": content[:200] + "..."
                    if len(content) > 200
                    else content,
                },
            )
            sequence += 1

            # Generate content with streaming
            loop = asyncio.get_event_loop()

            if self._config.enable_session_persistence:
                if self._chat_session is None:
                    self._chat_session = model.start_chat(
                        history=self._conversation_history
                    )

                response_stream = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self._chat_session.send_message(
                            content,
                            generation_config=generation_config,
                            stream=True,
                        ),
                    ),
                    timeout=self._config.timeout_seconds,
                )
            else:
                response_stream = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: model.generate_content(
                            content,
                            generation_config=generation_config,
                            stream=True,
                        ),
                    ),
                    timeout=self._config.timeout_seconds,
                )

            # Process streaming response
            full_text = ""
            tool_calls = 0
            usage: dict[str, int] = {}
            grounding: dict[str, Any] | None = None
            tool_events: list[ATPEvent] = []

            # Iterate through streaming chunks
            for chunk in response_stream:
                # Extract text from chunk
                chunk_text = ""
                for candidate in chunk.candidates:
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            chunk_text += part.text

                if chunk_text:
                    full_text += chunk_text
                    yield ATPEvent(
                        task_id=request.task_id,
                        sequence=sequence,
                        event_type=EventType.PROGRESS,
                        payload={
                            "type": "chunk",
                            "text": chunk_text,
                        },
                    )
                    sequence += 1

                # Extract tool calls from chunk
                chunk_tool_events, sequence = self._extract_tool_calls(
                    chunk, request.task_id, sequence
                )
                for tool_event in chunk_tool_events:
                    yield tool_event
                    tool_events.append(tool_event)
                    tool_calls += 1

                # Extract usage metadata (available in final chunk)
                chunk_usage = self._extract_usage_metadata(chunk)
                if chunk_usage:
                    usage = chunk_usage

                # Extract grounding metadata
                chunk_grounding = self._extract_grounding_metadata(chunk)
                if chunk_grounding:
                    grounding = chunk_grounding

            # Update session ID if persistence is enabled
            if self._config.enable_session_persistence:
                self._session_id = session_id

            wall_time = time.time() - start_time

            # Build artifacts
            artifacts = [
                ArtifactStructured(
                    name="output",
                    data={
                        "text": full_text,
                        "session_id": session_id,
                        "model": self._config.model_name,
                    },
                    content_type="text/plain",
                )
            ]

            if grounding:
                artifacts.append(
                    ArtifactStructured(
                        name="grounding",
                        data=grounding,
                        content_type="application/json",
                    )
                )

            if tool_events:
                artifacts.append(
                    ArtifactStructured(
                        name="tool_calls",
                        data={
                            "calls": [
                                {
                                    "tool": e.payload.get("tool"),
                                    "input": e.payload.get("input"),
                                }
                                for e in tool_events
                            ]
                        },
                        content_type="application/json",
                    )
                )

            yield ATPResponse(
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

        except TimeoutError:
            wall_time = time.time() - start_time
            yield ATPEvent(
                task_id=request.task_id,
                sequence=sequence,
                event_type=EventType.ERROR,
                payload={
                    "error_type": "timeout",
                    "message": (
                        f"Vertex AI generation timed out after "
                        f"{self._config.timeout_seconds}s"
                    ),
                    "recoverable": False,
                },
            )
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.TIMEOUT,
                error=(
                    f"Vertex AI generation timed out after "
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
        Check if Vertex AI is accessible.

        Returns:
            True if Vertex AI is accessible, False otherwise.
        """
        try:
            self._initialize_vertexai()
            self._get_model()
            return True
        except (AdapterError, AdapterConnectionError):
            return False

    async def cleanup(self) -> None:
        """Release resources."""
        self._model = None
        self._chat_session = None
        if not self._config.enable_session_persistence:
            self._session_id = None
            self._conversation_history = []
        self._initialized = False

    def reset_session(self) -> None:
        """Reset the session for a new conversation."""
        self._session_id = None
        self._chat_session = None
        self._conversation_history = []

    def set_session_id(self, session_id: str) -> None:
        """
        Set a specific session ID.

        Args:
            session_id: Session ID to use for subsequent requests.
        """
        self._session_id = session_id
