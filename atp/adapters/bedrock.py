"""AWS Bedrock Agents adapter implementation."""

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
        None, description="Custom endpoint URL for Bedrock (for testing)"
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
            "When enabled, the agent retrieves relevant documents and generates "
            "responses based on them."
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
        description="Guardrail version (required if guardrail_identifier is set)",
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
        # If access key is provided, secret key must also be provided
        if self.access_key_id and not self.secret_access_key:
            raise ValueError(
                "secret_access_key is required when access_key_id is provided"
            )
        if self.secret_access_key and not self.access_key_id:
            raise ValueError(
                "access_key_id is required when secret_access_key is provided"
            )

        # Guardrail version required if identifier is set
        if self.guardrail_identifier and not self.guardrail_version:
            raise ValueError(
                "guardrail_version is required when guardrail_identifier is set"
            )

        return self


class BedrockAdapter(AgentAdapter):
    """
    Adapter for AWS Bedrock Agents.

    This adapter allows ATP to communicate with agents deployed on AWS Bedrock.
    It supports:
    - Agent invocation with session management
    - Knowledge base integration for RAG workflows
    - Action groups for custom tool invocation
    - Trace extraction for observability
    - Multi-turn conversations with session persistence

    AWS Bedrock Agents require the boto3 library to be installed:
        uv add boto3

    AWS credentials can be provided via:
    - Explicit access_key_id and secret_access_key in config
    - AWS profile name
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - IAM role (when running on AWS infrastructure)
    """

    def __init__(self, config: BedrockAdapterConfig) -> None:
        """
        Initialize Bedrock adapter.

        Args:
            config: Bedrock adapter configuration.

        Raises:
            AdapterError: If boto3 is not installed.
        """
        super().__init__(config)
        self._config: BedrockAdapterConfig = config
        self._client: Any = None
        self._session_id: str | None = config.session_id
        self._initialized = False

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""
        return "bedrock"

    @property
    def session_id(self) -> str | None:
        """Get the current session ID."""
        return self._session_id

    def _get_boto3_client(self) -> Any:
        """
        Create or return the boto3 client for Bedrock Agent Runtime.

        Returns:
            boto3 client for bedrock-agent-runtime.

        Raises:
            AdapterError: If boto3 is not installed.
            AdapterConnectionError: If client creation fails.
        """
        if self._client is not None:
            return self._client

        try:
            import boto3
        except ImportError as e:
            raise AdapterError(
                "boto3 is required for Bedrock adapter. Install it with: uv add boto3",
                adapter_type=self.adapter_type,
            ) from e

        try:
            # Build session kwargs
            session_kwargs: dict[str, Any] = {
                "region_name": self._config.region,
            }

            if self._config.profile:
                session_kwargs["profile_name"] = self._config.profile

            # Create session
            session = boto3.Session(**session_kwargs)

            # Build client kwargs
            client_kwargs: dict[str, Any] = {}

            if self._config.access_key_id and self._config.secret_access_key:
                client_kwargs["aws_access_key_id"] = self._config.access_key_id
                client_kwargs["aws_secret_access_key"] = self._config.secret_access_key
                if self._config.session_token:
                    client_kwargs["aws_session_token"] = self._config.session_token

            if self._config.endpoint_url:
                client_kwargs["endpoint_url"] = self._config.endpoint_url

            self._client = session.client("bedrock-agent-runtime", **client_kwargs)
            return self._client

        except Exception as e:
            raise AdapterConnectionError(
                f"Failed to create Bedrock client: {e}",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

    def _generate_session_id(self) -> str:
        """Generate a new session ID."""
        return str(uuid.uuid4())

    def _build_invoke_params(
        self,
        request: ATPRequest,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Build parameters for invoke_agent API call.

        Args:
            request: ATP request.
            session_id: Session ID for the invocation.

        Returns:
            Dictionary of invoke_agent parameters.
        """
        params: dict[str, Any] = {
            "agentId": self._config.agent_id,
            "agentAliasId": self._config.agent_alias_id,
            "sessionId": session_id,
            "inputText": request.task.description,
            "enableTrace": self._config.enable_trace,
        }

        # Add knowledge base configuration
        if self._config.knowledge_base_ids:
            kb_configs = []
            for kb_id in self._config.knowledge_base_ids:
                kb_config: dict[str, Any] = {
                    "knowledgeBaseId": kb_id,
                    "retrievalConfiguration": (
                        self._config.retrieval_config
                        if self._config.retrieval_config
                        else {
                            "vectorSearchConfiguration": {
                                "numberOfResults": 5,
                            }
                        }
                    ),
                }
                kb_configs.append(kb_config)

            params["knowledgeBaseConfigurations"] = kb_configs

        # Add memory configuration
        if self._config.memory_id:
            params["memoryId"] = self._config.memory_id

        # Add guardrails
        if self._config.guardrail_identifier and self._config.guardrail_version:
            params["guardrailConfiguration"] = {
                "guardrailIdentifier": self._config.guardrail_identifier,
                "guardrailVersion": self._config.guardrail_version,
            }

        # Add session state if input data contains it
        if request.task.input_data:
            session_state = request.task.input_data.get("session_state")
            if session_state:
                params["sessionState"] = session_state

            # Add prompt session attributes
            prompt_attrs = request.task.input_data.get("prompt_session_attributes")
            if prompt_attrs:
                if "sessionState" not in params:
                    params["sessionState"] = {}
                params["sessionState"]["promptSessionAttributes"] = prompt_attrs

        return params

    def _extract_trace_event(
        self,
        trace: dict[str, Any],
        task_id: str,
        sequence: int,
    ) -> ATPEvent | None:
        """
        Extract an ATP event from a Bedrock trace.

        Args:
            trace: Bedrock trace dictionary.
            task_id: Task ID for the event.
            sequence: Sequence number.

        Returns:
            ATPEvent if trace can be converted, None otherwise.
        """
        trace_data = trace.get("trace", {})

        # Handle orchestration trace
        orchestration_trace = trace_data.get("orchestrationTrace")
        if orchestration_trace:
            # Model invocation input
            model_input = orchestration_trace.get("modelInvocationInput")
            if model_input:
                return ATPEvent(
                    task_id=task_id,
                    sequence=sequence,
                    event_type=EventType.LLM_REQUEST,
                    payload={
                        "type": "model_invocation",
                        "text": model_input.get("text", ""),
                        "trace_id": model_input.get("traceId"),
                    },
                )

            # Model invocation output
            model_output = orchestration_trace.get("modelInvocationOutput")
            if model_output:
                parsed = model_output.get("parsedResponse", {})
                return ATPEvent(
                    task_id=task_id,
                    sequence=sequence,
                    event_type=EventType.LLM_REQUEST,
                    payload={
                        "type": "model_response",
                        "text": parsed.get("text", ""),
                        "trace_id": model_output.get("traceId"),
                    },
                )

            # Rationale (reasoning)
            rationale = orchestration_trace.get("rationale")
            if rationale and self._config.trace_include_reasoning:
                return ATPEvent(
                    task_id=task_id,
                    sequence=sequence,
                    event_type=EventType.REASONING,
                    payload={
                        "thought": rationale.get("text", ""),
                        "trace_id": rationale.get("traceId"),
                    },
                )

            # Invocation input (action group call)
            invocation_input = orchestration_trace.get("invocationInput")
            if invocation_input:
                action_input = invocation_input.get("actionGroupInvocationInput", {})
                return ATPEvent(
                    task_id=task_id,
                    sequence=sequence,
                    event_type=EventType.TOOL_CALL,
                    payload={
                        "tool": action_input.get("actionGroupName", "unknown"),
                        "function": action_input.get("function", ""),
                        "api_path": action_input.get("apiPath", ""),
                        "input": action_input.get("parameters", []),
                        "status": "started",
                        "trace_id": invocation_input.get("traceId"),
                    },
                )

            # Observation (action result)
            observation = orchestration_trace.get("observation")
            if observation:
                action_output = observation.get("actionGroupInvocationOutput", {})
                kb_output = observation.get("knowledgeBaseLookupOutput", {})
                final_response = observation.get("finalResponse", {})

                if action_output:
                    return ATPEvent(
                        task_id=task_id,
                        sequence=sequence,
                        event_type=EventType.TOOL_CALL,
                        payload={
                            "output": action_output.get("text", ""),
                            "status": "success",
                            "trace_id": observation.get("traceId"),
                        },
                    )
                elif kb_output:
                    return ATPEvent(
                        task_id=task_id,
                        sequence=sequence,
                        event_type=EventType.TOOL_CALL,
                        payload={
                            "tool": "knowledge_base_lookup",
                            "retrieved_references": kb_output.get(
                                "retrievedReferences", []
                            ),
                            "status": "success",
                            "trace_id": observation.get("traceId"),
                        },
                    )
                elif final_response:
                    return ATPEvent(
                        task_id=task_id,
                        sequence=sequence,
                        event_type=EventType.PROGRESS,
                        payload={
                            "message": "Final response generated",
                            "text": final_response.get("text", ""),
                            "trace_id": observation.get("traceId"),
                        },
                    )

        # Handle pre-processing trace
        preprocessing_trace = trace_data.get("preProcessingTrace")
        if preprocessing_trace:
            model_output = preprocessing_trace.get("modelInvocationOutput")
            if model_output:
                parsed = model_output.get("parsedResponse", {})
                return ATPEvent(
                    task_id=task_id,
                    sequence=sequence,
                    event_type=EventType.PROGRESS,
                    payload={
                        "type": "preprocessing",
                        "is_valid": parsed.get("isValid", True),
                        "rationale": parsed.get("rationale", ""),
                    },
                )

        # Handle post-processing trace
        postprocessing_trace = trace_data.get("postProcessingTrace")
        if postprocessing_trace:
            model_output = postprocessing_trace.get("modelInvocationOutput")
            if model_output:
                return ATPEvent(
                    task_id=task_id,
                    sequence=sequence,
                    event_type=EventType.PROGRESS,
                    payload={
                        "type": "postprocessing",
                        "output": model_output.get("parsedResponse", {}),
                    },
                )

        # Handle failure trace
        failure_trace = trace_data.get("failureTrace")
        if failure_trace:
            return ATPEvent(
                task_id=task_id,
                sequence=sequence,
                event_type=EventType.ERROR,
                payload={
                    "error_type": failure_trace.get("failureReason", "unknown"),
                    "message": failure_trace.get("failureReason", "Agent failure"),
                    "trace_id": failure_trace.get("traceId"),
                    "recoverable": False,
                },
            )

        # Handle guardrail trace
        guardrail_trace = trace_data.get("guardrailTrace")
        if guardrail_trace:
            return ATPEvent(
                task_id=task_id,
                sequence=sequence,
                event_type=EventType.PROGRESS,
                payload={
                    "type": "guardrail",
                    "action": guardrail_trace.get("action", ""),
                    "input_assessments": guardrail_trace.get("inputAssessments", []),
                    "output_assessments": guardrail_trace.get("outputAssessments", []),
                    "trace_id": guardrail_trace.get("traceId"),
                },
            )

        return None

    async def _invoke_agent(
        self,
        request: ATPRequest,
        session_id: str,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """
        Invoke the Bedrock agent and collect response.

        Args:
            request: ATP request.
            session_id: Session ID for the invocation.

        Returns:
            Tuple of (response text, traces, metrics info).

        Raises:
            AdapterTimeoutError: If invocation times out.
            AdapterError: If invocation fails.
        """
        client = self._get_boto3_client()
        params = self._build_invoke_params(request, session_id)

        try:
            # Run boto3 call in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: client.invoke_agent(**params)),
                timeout=self._config.timeout_seconds,
            )
        except TimeoutError as e:
            raise AdapterTimeoutError(
                f"Bedrock agent invocation timed out after "
                f"{self._config.timeout_seconds}s",
                adapter_type=self.adapter_type,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except Exception as e:
            # Check for specific AWS errors
            error_str = str(e)
            if "ResourceNotFoundException" in error_str:
                raise AdapterError(
                    f"Bedrock agent not found: {self._config.agent_id}",
                    adapter_type=self.adapter_type,
                ) from e
            if "AccessDeniedException" in error_str:
                raise AdapterConnectionError(
                    "Access denied to Bedrock agent. Check AWS credentials and "
                    "permissions.",
                    adapter_type=self.adapter_type,
                    cause=e,
                ) from e
            if "ThrottlingException" in error_str:
                raise AdapterError(
                    "Bedrock request throttled. Consider implementing retry logic.",
                    adapter_type=self.adapter_type,
                ) from e
            raise AdapterError(
                f"Bedrock agent invocation failed: {e}",
                adapter_type=self.adapter_type,
            ) from e

        # Process streaming response
        completion_text = ""
        traces: list[dict[str, Any]] = []
        metrics_info: dict[str, Any] = {}

        event_stream = response.get("completion", [])
        for event in event_stream:
            # Text chunk
            if "chunk" in event:
                chunk_data = event["chunk"]
                if "bytes" in chunk_data:
                    completion_text += chunk_data["bytes"].decode("utf-8")

            # Trace event
            if "trace" in event:
                traces.append(event["trace"])

            # Return control event (for action groups)
            if "returnControl" in event:
                return_control = event["returnControl"]
                traces.append(
                    {
                        "trace": {
                            "returnControl": return_control,
                        }
                    }
                )

            # Files event
            if "files" in event:
                traces.append(
                    {
                        "trace": {
                            "files": event["files"],
                        }
                    }
                )

        # Extract session ID from response metadata if available
        response_metadata = response.get("ResponseMetadata", {})
        metrics_info["request_id"] = response_metadata.get("RequestId")
        metrics_info["http_status"] = response_metadata.get("HTTPStatusCode")

        return completion_text, traces, metrics_info

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """
        Execute a task via Bedrock Agent.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse with execution results.

        Raises:
            AdapterConnectionError: If connection to AWS fails.
            AdapterTimeoutError: If execution times out.
            AdapterError: If agent invocation fails.
        """
        start_time = time.time()

        # Determine session ID
        if self._config.enable_session_persistence and self._session_id:
            session_id = self._session_id
        else:
            session_id = self._generate_session_id()

        try:
            completion_text, traces, metrics_info = await self._invoke_agent(
                request, session_id
            )

            # Update session ID if persistence is enabled
            if self._config.enable_session_persistence:
                self._session_id = session_id

            wall_time = time.time() - start_time

            # Count tool calls from traces
            tool_calls = sum(
                1
                for t in traces
                if t.get("trace", {})
                .get("orchestrationTrace", {})
                .get("invocationInput")
            )

            # Count LLM calls from traces
            llm_calls = sum(
                1
                for t in traces
                if t.get("trace", {})
                .get("orchestrationTrace", {})
                .get("modelInvocationOutput")
            )

            # Build artifacts
            artifacts = [
                ArtifactStructured(
                    name="output",
                    data={
                        "text": completion_text,
                        "session_id": session_id,
                    },
                    content_type="text/plain",
                )
            ]

            # Add trace artifact if traces were collected
            if traces:
                artifacts.append(
                    ArtifactStructured(
                        name="traces",
                        data={"traces": traces},
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
                    llm_calls=llm_calls,
                ),
                trace_id=metrics_info.get("request_id"),
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
            AdapterConnectionError: If connection to AWS fails.
            AdapterTimeoutError: If execution times out.
            AdapterError: If agent invocation fails.
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
                "message": "Starting Bedrock agent invocation",
                "session_id": session_id,
                "agent_id": self._config.agent_id,
            },
        )
        sequence += 1

        try:
            # Get client
            client = self._get_boto3_client()
            params = self._build_invoke_params(request, session_id)

            # Run boto3 call in executor
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: client.invoke_agent(**params)),
                timeout=self._config.timeout_seconds,
            )

            # Process streaming response
            completion_text = ""
            traces: list[dict[str, Any]] = []
            tool_calls = 0
            llm_calls = 0
            metrics_info: dict[str, Any] = {}

            event_stream = response.get("completion", [])
            for event in event_stream:
                # Text chunk
                if "chunk" in event:
                    chunk_data = event["chunk"]
                    if "bytes" in chunk_data:
                        chunk_text = chunk_data["bytes"].decode("utf-8")
                        completion_text += chunk_text

                        # Emit chunk event
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

                # Trace event
                if "trace" in event:
                    trace = event["trace"]
                    traces.append(trace)

                    # Convert trace to ATP event
                    atp_event = self._extract_trace_event(
                        trace, request.task_id, sequence
                    )
                    if atp_event:
                        yield atp_event
                        sequence += 1

                        # Count metrics
                        if atp_event.event_type == EventType.TOOL_CALL:
                            if atp_event.payload.get("status") == "started":
                                tool_calls += 1
                        elif atp_event.event_type == EventType.LLM_REQUEST:
                            if atp_event.payload.get("type") == "model_response":
                                llm_calls += 1

                # Return control event
                if "returnControl" in event:
                    return_control = event["returnControl"]
                    yield ATPEvent(
                        task_id=request.task_id,
                        sequence=sequence,
                        event_type=EventType.TOOL_CALL,
                        payload={
                            "type": "return_control",
                            "invocation_inputs": return_control.get(
                                "invocationInputs", []
                            ),
                            "invocation_id": return_control.get("invocationId"),
                        },
                    )
                    sequence += 1

            # Update session ID if persistence is enabled
            if self._config.enable_session_persistence:
                self._session_id = session_id

            # Extract response metadata
            response_metadata = response.get("ResponseMetadata", {})
            metrics_info["request_id"] = response_metadata.get("RequestId")

            wall_time = time.time() - start_time

            # Build artifacts
            artifacts = [
                ArtifactStructured(
                    name="output",
                    data={
                        "text": completion_text,
                        "session_id": session_id,
                    },
                    content_type="text/plain",
                )
            ]

            if traces:
                artifacts.append(
                    ArtifactStructured(
                        name="traces",
                        data={"traces": traces},
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
                    llm_calls=llm_calls,
                ),
                trace_id=metrics_info.get("request_id"),
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
                        f"Bedrock agent invocation timed out after "
                        f"{self._config.timeout_seconds}s"
                    ),
                    "recoverable": False,
                },
            )
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.TIMEOUT,
                error=(
                    f"Bedrock agent invocation timed out after "
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
        Check if Bedrock agent is accessible.

        Returns:
            True if agent is accessible, False otherwise.
        """
        try:
            # Try to get the client - this validates credentials
            self._get_boto3_client()
            return True
        except (AdapterError, AdapterConnectionError):
            return False

    async def cleanup(self) -> None:
        """Release resources."""
        self._client = None
        if not self._config.enable_session_persistence:
            self._session_id = None
        self._initialized = False

    def reset_session(self) -> None:
        """Reset the session ID for a new conversation."""
        self._session_id = None

    def set_session_id(self, session_id: str) -> None:
        """
        Set a specific session ID.

        Args:
            session_id: Session ID to use for subsequent requests.
        """
        self._session_id = session_id
