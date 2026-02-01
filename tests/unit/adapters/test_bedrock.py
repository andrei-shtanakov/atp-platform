"""Unit tests for BedrockAdapter."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from atp.adapters import (
    AdapterConnectionError,
    AdapterError,
    AdapterTimeoutError,
)
from atp.adapters.bedrock import BedrockAdapter, BedrockAdapterConfig
from atp.protocol import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
    ResponseStatus,
    Task,
)


@pytest.fixture
def sample_request() -> ATPRequest:
    """Create a sample ATP request for testing."""
    return ATPRequest(
        task_id="test-task-123",
        task=Task(description="Test task for Bedrock agent"),
        constraints={"max_steps": 10},
    )


@pytest.fixture
def bedrock_config() -> BedrockAdapterConfig:
    """Create Bedrock adapter config."""
    return BedrockAdapterConfig(
        agent_id="test-agent-id",
        agent_alias_id="TSTALIASID",
        region="us-east-1",
        timeout_seconds=30.0,
    )


@pytest.fixture
def mock_bedrock_response() -> dict[str, Any]:
    """Create a mock Bedrock invoke_agent response."""
    return {
        "completion": [
            {
                "chunk": {
                    "bytes": b"Hello, I'm your Bedrock agent. ",
                },
            },
            {
                "chunk": {
                    "bytes": b"How can I help you?",
                },
            },
            {
                "trace": {
                    "trace": {
                        "orchestrationTrace": {
                            "modelInvocationOutput": {
                                "parsedResponse": {
                                    "text": "Agent reasoning here",
                                },
                                "traceId": "trace-123",
                            },
                        },
                    },
                },
            },
        ],
        "ResponseMetadata": {
            "RequestId": "request-123",
            "HTTPStatusCode": 200,
        },
    }


@pytest.fixture
def mock_bedrock_response_with_tool_call() -> dict[str, Any]:
    """Create a mock Bedrock response with tool invocation."""
    return {
        "completion": [
            {
                "trace": {
                    "trace": {
                        "orchestrationTrace": {
                            "rationale": {
                                "text": "I need to call a tool to help with this.",
                                "traceId": "trace-rationale-1",
                            },
                        },
                    },
                },
            },
            {
                "trace": {
                    "trace": {
                        "orchestrationTrace": {
                            "invocationInput": {
                                "actionGroupInvocationInput": {
                                    "actionGroupName": "my-action-group",
                                    "function": "search",
                                    "apiPath": "/api/search",
                                    "parameters": [{"name": "query", "value": "test"}],
                                },
                                "traceId": "trace-input-1",
                            },
                        },
                    },
                },
            },
            {
                "trace": {
                    "trace": {
                        "orchestrationTrace": {
                            "observation": {
                                "actionGroupInvocationOutput": {
                                    "text": "Search results...",
                                },
                                "traceId": "trace-output-1",
                            },
                        },
                    },
                },
            },
            {
                "chunk": {
                    "bytes": b"Based on my search, here are the results.",
                },
            },
        ],
        "ResponseMetadata": {
            "RequestId": "request-456",
            "HTTPStatusCode": 200,
        },
    }


class TestBedrockAdapterConfig:
    """Tests for BedrockAdapterConfig."""

    def test_minimal_config(self) -> None:
        """Test creating config with minimal required fields."""
        config = BedrockAdapterConfig(agent_id="my-agent")
        assert config.agent_id == "my-agent"
        assert config.agent_alias_id == "TSTALIASID"
        assert config.region == "us-east-1"
        assert config.timeout_seconds == 300.0
        assert config.enable_trace is True

    def test_full_config(self) -> None:
        """Test creating config with all fields."""
        config = BedrockAdapterConfig(
            agent_id="my-agent",
            agent_alias_id="PRODALIASID",
            region="us-west-2",
            profile="my-profile",
            access_key_id="AKIAEXAMPLE",
            secret_access_key="secret123",
            session_token="token123",
            timeout_seconds=60.0,
            enable_session_persistence=True,
            session_id="session-123",
            knowledge_base_ids=["kb-1", "kb-2"],
            enable_trace=True,
            trace_include_reasoning=True,
            memory_id="memory-123",
            guardrail_identifier="guardrail-1",
            guardrail_version="1",
        )
        assert config.agent_id == "my-agent"
        assert config.agent_alias_id == "PRODALIASID"
        assert config.region == "us-west-2"
        assert config.profile == "my-profile"
        assert config.access_key_id == "AKIAEXAMPLE"
        assert config.secret_access_key == "secret123"
        assert config.session_token == "token123"
        assert config.enable_session_persistence is True
        assert config.knowledge_base_ids == ["kb-1", "kb-2"]
        assert config.guardrail_identifier == "guardrail-1"

    def test_empty_agent_id_fails(self) -> None:
        """Test that empty agent_id raises an error."""
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            BedrockAdapterConfig(agent_id="  ")

    def test_empty_region_fails(self) -> None:
        """Test that empty region raises an error."""
        with pytest.raises(ValueError, match="region cannot be empty"):
            BedrockAdapterConfig(agent_id="my-agent", region="  ")

    def test_secret_key_without_access_key_fails(self) -> None:
        """Test that providing secret_access_key without access_key_id fails."""
        with pytest.raises(
            ValueError, match="access_key_id is required when secret_access_key"
        ):
            BedrockAdapterConfig(
                agent_id="my-agent",
                secret_access_key="secret123",
            )

    def test_access_key_without_secret_key_fails(self) -> None:
        """Test that providing access_key_id without secret_access_key fails."""
        with pytest.raises(
            ValueError, match="secret_access_key is required when access_key_id"
        ):
            BedrockAdapterConfig(
                agent_id="my-agent",
                access_key_id="AKIAEXAMPLE",
            )

    def test_guardrail_identifier_without_version_fails(self) -> None:
        """Test that guardrail_identifier without version fails."""
        with pytest.raises(
            ValueError, match="guardrail_version is required when guardrail_identifier"
        ):
            BedrockAdapterConfig(
                agent_id="my-agent",
                guardrail_identifier="guardrail-1",
            )


class TestBedrockAdapter:
    """Tests for BedrockAdapter."""

    def test_adapter_type(self, bedrock_config: BedrockAdapterConfig) -> None:
        """Test adapter type property."""
        adapter = BedrockAdapter(bedrock_config)
        assert adapter.adapter_type == "bedrock"

    def test_boto3_not_installed(self, bedrock_config: BedrockAdapterConfig) -> None:
        """Test error when boto3 is not installed."""
        adapter = BedrockAdapter(bedrock_config)

        with patch.dict("sys.modules", {"boto3": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'boto3'"),
            ):
                with pytest.raises(AdapterError, match="boto3 is required"):
                    adapter._get_boto3_client()

    def test_session_id_management(self, bedrock_config: BedrockAdapterConfig) -> None:
        """Test session ID management."""
        adapter = BedrockAdapter(bedrock_config)
        assert adapter.session_id is None

        adapter.set_session_id("my-session")
        assert adapter.session_id == "my-session"

        adapter.reset_session()
        assert adapter.session_id is None

    @pytest.mark.anyio
    async def test_execute_success(
        self,
        bedrock_config: BedrockAdapterConfig,
        sample_request: ATPRequest,
        mock_bedrock_response: dict[str, Any],
    ) -> None:
        """Test successful execute call."""
        adapter = BedrockAdapter(bedrock_config)

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = mock_bedrock_response

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            response = await adapter.execute(sample_request)

            assert isinstance(response, ATPResponse)
            assert response.task_id == "test-task-123"
            assert response.status == ResponseStatus.COMPLETED
            assert len(response.artifacts) >= 1

            # Check output artifact
            output_artifact = response.artifacts[0]
            assert output_artifact.name == "output"
            assert "Hello, I'm your Bedrock agent." in output_artifact.data["text"]
            assert "session_id" in output_artifact.data

            # Check metrics
            assert response.metrics is not None
            assert response.metrics.wall_time_seconds > 0
            assert response.metrics.llm_calls == 1

    @pytest.mark.anyio
    async def test_execute_with_traces(
        self,
        bedrock_config: BedrockAdapterConfig,
        sample_request: ATPRequest,
        mock_bedrock_response_with_tool_call: dict[str, Any],
    ) -> None:
        """Test execute with trace events."""
        adapter = BedrockAdapter(bedrock_config)

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = mock_bedrock_response_with_tool_call

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            response = await adapter.execute(sample_request)

            assert response.status == ResponseStatus.COMPLETED
            assert response.metrics is not None
            assert response.metrics.tool_calls == 1

            # Should have traces artifact
            trace_artifact = next(
                (a for a in response.artifacts if a.name == "traces"), None
            )
            assert trace_artifact is not None
            assert len(trace_artifact.data["traces"]) > 0

    @pytest.mark.anyio
    async def test_execute_timeout(
        self,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with timeout."""
        config = BedrockAdapterConfig(
            agent_id="test-agent",
            timeout_seconds=0.001,
        )
        adapter = BedrockAdapter(config)

        mock_client = MagicMock()

        def slow_invoke(**kwargs: Any) -> dict[str, Any]:
            import time

            time.sleep(1)  # Simulate slow response
            return {"completion": [], "ResponseMetadata": {}}

        mock_client.invoke_agent.side_effect = slow_invoke

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            with pytest.raises(AdapterTimeoutError):
                await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_execute_resource_not_found(
        self,
        bedrock_config: BedrockAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with resource not found error."""
        adapter = BedrockAdapter(bedrock_config)

        mock_client = MagicMock()
        mock_client.invoke_agent.side_effect = Exception(
            "ResourceNotFoundException: Agent not found"
        )

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            with pytest.raises(AdapterError, match="Bedrock agent not found"):
                await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_execute_access_denied(
        self,
        bedrock_config: BedrockAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with access denied error."""
        adapter = BedrockAdapter(bedrock_config)

        mock_client = MagicMock()
        mock_client.invoke_agent.side_effect = Exception(
            "AccessDeniedException: Access denied"
        )

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            with pytest.raises(AdapterConnectionError, match="Access denied"):
                await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_execute_throttling(
        self,
        bedrock_config: BedrockAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with throttling error."""
        adapter = BedrockAdapter(bedrock_config)

        mock_client = MagicMock()
        mock_client.invoke_agent.side_effect = Exception(
            "ThrottlingException: Rate exceeded"
        )

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            with pytest.raises(AdapterError, match="request throttled"):
                await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_stream_events_success(
        self,
        bedrock_config: BedrockAdapterConfig,
        sample_request: ATPRequest,
        mock_bedrock_response: dict[str, Any],
    ) -> None:
        """Test streaming events success."""
        adapter = BedrockAdapter(bedrock_config)

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = mock_bedrock_response

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            events: list[ATPEvent | ATPResponse] = []
            async for item in adapter.stream_events(sample_request):
                events.append(item)

            # Should have events and a final response
            assert len(events) >= 2

            # Check for progress events
            progress_events = [
                e
                for e in events
                if isinstance(e, ATPEvent) and e.event_type == EventType.PROGRESS
            ]
            assert len(progress_events) >= 1

            # Last item should be response
            assert isinstance(events[-1], ATPResponse)
            assert events[-1].status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_stream_events_with_tool_calls(
        self,
        bedrock_config: BedrockAdapterConfig,
        sample_request: ATPRequest,
        mock_bedrock_response_with_tool_call: dict[str, Any],
    ) -> None:
        """Test streaming events with tool calls."""
        adapter = BedrockAdapter(bedrock_config)

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = mock_bedrock_response_with_tool_call

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            events: list[ATPEvent | ATPResponse] = []
            async for item in adapter.stream_events(sample_request):
                events.append(item)

            # Check for tool call events
            tool_events = [
                e
                for e in events
                if isinstance(e, ATPEvent) and e.event_type == EventType.TOOL_CALL
            ]
            assert len(tool_events) >= 1

            # Check for reasoning events
            reasoning_events = [
                e
                for e in events
                if isinstance(e, ATPEvent) and e.event_type == EventType.REASONING
            ]
            assert len(reasoning_events) >= 1

    @pytest.mark.anyio
    async def test_stream_events_timeout(
        self,
        sample_request: ATPRequest,
    ) -> None:
        """Test streaming events with timeout."""
        config = BedrockAdapterConfig(
            agent_id="test-agent",
            timeout_seconds=0.001,
        )
        adapter = BedrockAdapter(config)

        mock_client = MagicMock()

        def slow_invoke(**kwargs: Any) -> dict[str, Any]:
            import time

            time.sleep(1)
            return {"completion": [], "ResponseMetadata": {}}

        mock_client.invoke_agent.side_effect = slow_invoke

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            events: list[ATPEvent | ATPResponse] = []
            async for item in adapter.stream_events(sample_request):
                events.append(item)

            # Should have error event and failed response
            assert len(events) >= 2

            # Check for error event
            error_events = [
                e
                for e in events
                if isinstance(e, ATPEvent) and e.event_type == EventType.ERROR
            ]
            assert len(error_events) >= 1

            # Last item should be failed response
            assert isinstance(events[-1], ATPResponse)
            assert events[-1].status == ResponseStatus.TIMEOUT

    @pytest.mark.anyio
    async def test_health_check_success(
        self, bedrock_config: BedrockAdapterConfig
    ) -> None:
        """Test health check success."""
        adapter = BedrockAdapter(bedrock_config)

        mock_client = MagicMock()

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            result = await adapter.health_check()
            assert result is True

    @pytest.mark.anyio
    async def test_health_check_failure(
        self, bedrock_config: BedrockAdapterConfig
    ) -> None:
        """Test health check failure."""
        adapter = BedrockAdapter(bedrock_config)

        with patch.object(
            adapter, "_get_boto3_client", side_effect=AdapterError("Connection failed")
        ):
            result = await adapter.health_check()
            assert result is False

    @pytest.mark.anyio
    async def test_cleanup(self, bedrock_config: BedrockAdapterConfig) -> None:
        """Test cleanup releases resources."""
        adapter = BedrockAdapter(bedrock_config)

        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._session_id = "test-session"
        adapter._initialized = True

        await adapter.cleanup()

        assert adapter._client is None
        assert adapter._session_id is None
        assert adapter._initialized is False

    @pytest.mark.anyio
    async def test_session_persistence(
        self,
        sample_request: ATPRequest,
        mock_bedrock_response: dict[str, Any],
    ) -> None:
        """Test session persistence across requests."""
        config = BedrockAdapterConfig(
            agent_id="test-agent",
            enable_session_persistence=True,
        )
        adapter = BedrockAdapter(config)

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = mock_bedrock_response

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            # First call - should create a session
            await adapter.execute(sample_request)
            session_id_1 = adapter.session_id

            assert session_id_1 is not None

            # Second call - should use same session
            await adapter.execute(sample_request)
            session_id_2 = adapter.session_id

            assert session_id_2 == session_id_1

    @pytest.mark.anyio
    async def test_no_session_persistence(
        self,
        bedrock_config: BedrockAdapterConfig,
        sample_request: ATPRequest,
        mock_bedrock_response: dict[str, Any],
    ) -> None:
        """Test without session persistence creates new sessions."""
        adapter = BedrockAdapter(bedrock_config)
        assert adapter._config.enable_session_persistence is False

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = mock_bedrock_response
        call_args: list[dict[str, Any]] = []

        def capture_invoke(**kwargs: Any) -> dict[str, Any]:
            call_args.append(kwargs)
            return mock_bedrock_response

        mock_client.invoke_agent.side_effect = capture_invoke

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            await adapter.execute(sample_request)
            await adapter.execute(sample_request)

            # Should have different session IDs
            assert len(call_args) == 2
            assert call_args[0]["sessionId"] != call_args[1]["sessionId"]


class TestBedrockTraceExtraction:
    """Tests for Bedrock trace event extraction."""

    @pytest.fixture
    def adapter(self) -> BedrockAdapter:
        """Create a Bedrock adapter for testing."""
        config = BedrockAdapterConfig(agent_id="test-agent")
        return BedrockAdapter(config)

    def test_extract_model_invocation_input(self, adapter: BedrockAdapter) -> None:
        """Test extracting model invocation input trace."""
        trace = {
            "trace": {
                "orchestrationTrace": {
                    "modelInvocationInput": {
                        "text": "Test prompt",
                        "traceId": "trace-123",
                    },
                },
            },
        }

        event = adapter._extract_trace_event(trace, "task-1", 1)

        assert event is not None
        assert event.event_type == EventType.LLM_REQUEST
        assert event.payload["type"] == "model_invocation"
        assert event.payload["text"] == "Test prompt"

    def test_extract_model_invocation_output(self, adapter: BedrockAdapter) -> None:
        """Test extracting model invocation output trace."""
        trace = {
            "trace": {
                "orchestrationTrace": {
                    "modelInvocationOutput": {
                        "parsedResponse": {"text": "Model response"},
                        "traceId": "trace-456",
                    },
                },
            },
        }

        event = adapter._extract_trace_event(trace, "task-1", 2)

        assert event is not None
        assert event.event_type == EventType.LLM_REQUEST
        assert event.payload["type"] == "model_response"
        assert event.payload["text"] == "Model response"

    def test_extract_rationale(self, adapter: BedrockAdapter) -> None:
        """Test extracting rationale (reasoning) trace."""
        trace = {
            "trace": {
                "orchestrationTrace": {
                    "rationale": {
                        "text": "I need to think about this",
                        "traceId": "trace-789",
                    },
                },
            },
        }

        event = adapter._extract_trace_event(trace, "task-1", 3)

        assert event is not None
        assert event.event_type == EventType.REASONING
        assert event.payload["thought"] == "I need to think about this"

    def test_extract_action_group_invocation(self, adapter: BedrockAdapter) -> None:
        """Test extracting action group invocation trace."""
        trace = {
            "trace": {
                "orchestrationTrace": {
                    "invocationInput": {
                        "actionGroupInvocationInput": {
                            "actionGroupName": "my-group",
                            "function": "search",
                            "apiPath": "/api/search",
                            "parameters": [{"name": "q", "value": "test"}],
                        },
                        "traceId": "trace-action",
                    },
                },
            },
        }

        event = adapter._extract_trace_event(trace, "task-1", 4)

        assert event is not None
        assert event.event_type == EventType.TOOL_CALL
        assert event.payload["tool"] == "my-group"
        assert event.payload["function"] == "search"
        assert event.payload["status"] == "started"

    def test_extract_action_group_output(self, adapter: BedrockAdapter) -> None:
        """Test extracting action group output trace."""
        trace = {
            "trace": {
                "orchestrationTrace": {
                    "observation": {
                        "actionGroupInvocationOutput": {
                            "text": "Action result",
                        },
                        "traceId": "trace-obs",
                    },
                },
            },
        }

        event = adapter._extract_trace_event(trace, "task-1", 5)

        assert event is not None
        assert event.event_type == EventType.TOOL_CALL
        assert event.payload["output"] == "Action result"
        assert event.payload["status"] == "success"

    def test_extract_knowledge_base_output(self, adapter: BedrockAdapter) -> None:
        """Test extracting knowledge base lookup output trace."""
        trace = {
            "trace": {
                "orchestrationTrace": {
                    "observation": {
                        "knowledgeBaseLookupOutput": {
                            "retrievedReferences": [
                                {"content": "Reference 1"},
                                {"content": "Reference 2"},
                            ],
                        },
                        "traceId": "trace-kb",
                    },
                },
            },
        }

        event = adapter._extract_trace_event(trace, "task-1", 6)

        assert event is not None
        assert event.event_type == EventType.TOOL_CALL
        assert event.payload["tool"] == "knowledge_base_lookup"
        assert len(event.payload["retrieved_references"]) == 2

    def test_extract_failure_trace(self, adapter: BedrockAdapter) -> None:
        """Test extracting failure trace."""
        trace = {
            "trace": {
                "failureTrace": {
                    "failureReason": "Agent error occurred",
                    "traceId": "trace-fail",
                },
            },
        }

        event = adapter._extract_trace_event(trace, "task-1", 7)

        assert event is not None
        assert event.event_type == EventType.ERROR
        assert event.payload["error_type"] == "Agent error occurred"
        assert event.payload["recoverable"] is False

    def test_extract_guardrail_trace(self, adapter: BedrockAdapter) -> None:
        """Test extracting guardrail trace."""
        trace = {
            "trace": {
                "guardrailTrace": {
                    "action": "BLOCKED",
                    "inputAssessments": [{"type": "HARMFUL_CONTENT"}],
                    "outputAssessments": [],
                    "traceId": "trace-guard",
                },
            },
        }

        event = adapter._extract_trace_event(trace, "task-1", 8)

        assert event is not None
        assert event.event_type == EventType.PROGRESS
        assert event.payload["type"] == "guardrail"
        assert event.payload["action"] == "BLOCKED"

    def test_extract_empty_trace(self, adapter: BedrockAdapter) -> None:
        """Test extracting empty trace returns None."""
        trace = {"trace": {}}

        event = adapter._extract_trace_event(trace, "task-1", 9)

        assert event is None


class TestBedrockInvokeParams:
    """Tests for Bedrock invoke parameters building."""

    @pytest.fixture
    def adapter(self) -> BedrockAdapter:
        """Create a Bedrock adapter for testing."""
        config = BedrockAdapterConfig(
            agent_id="test-agent",
            knowledge_base_ids=["kb-1", "kb-2"],
            guardrail_identifier="guardrail-1",
            guardrail_version="1",
            memory_id="memory-123",
        )
        return BedrockAdapter(config)

    def test_build_basic_params(self, adapter: BedrockAdapter) -> None:
        """Test building basic invoke parameters."""
        request = ATPRequest(
            task_id="task-1",
            task=Task(description="Test task"),
        )

        params = adapter._build_invoke_params(request, "session-123")

        assert params["agentId"] == "test-agent"
        assert params["agentAliasId"] == "TSTALIASID"
        assert params["sessionId"] == "session-123"
        assert params["inputText"] == "Test task"
        assert params["enableTrace"] is True

    def test_build_params_with_knowledge_base(self, adapter: BedrockAdapter) -> None:
        """Test building params with knowledge base config."""
        request = ATPRequest(
            task_id="task-1",
            task=Task(description="Test task"),
        )

        params = adapter._build_invoke_params(request, "session-123")

        assert "knowledgeBaseConfigurations" in params
        assert len(params["knowledgeBaseConfigurations"]) == 2
        assert params["knowledgeBaseConfigurations"][0]["knowledgeBaseId"] == "kb-1"

    def test_build_params_with_guardrail(self, adapter: BedrockAdapter) -> None:
        """Test building params with guardrail config."""
        request = ATPRequest(
            task_id="task-1",
            task=Task(description="Test task"),
        )

        params = adapter._build_invoke_params(request, "session-123")

        assert "guardrailConfiguration" in params
        assert params["guardrailConfiguration"]["guardrailIdentifier"] == "guardrail-1"
        assert params["guardrailConfiguration"]["guardrailVersion"] == "1"

    def test_build_params_with_memory(self, adapter: BedrockAdapter) -> None:
        """Test building params with memory config."""
        request = ATPRequest(
            task_id="task-1",
            task=Task(description="Test task"),
        )

        params = adapter._build_invoke_params(request, "session-123")

        assert params["memoryId"] == "memory-123"

    def test_build_params_with_session_state(self) -> None:
        """Test building params with session state from input data."""
        config = BedrockAdapterConfig(agent_id="test-agent")
        adapter = BedrockAdapter(config)

        request = ATPRequest(
            task_id="task-1",
            task=Task(
                description="Test task",
                input_data={
                    "session_state": {"key": "value"},
                    "prompt_session_attributes": {"attr1": "val1"},
                },
            ),
        )

        params = adapter._build_invoke_params(request, "session-123")

        assert "sessionState" in params
        assert params["sessionState"]["key"] == "value"
        assert params["sessionState"]["promptSessionAttributes"]["attr1"] == "val1"
