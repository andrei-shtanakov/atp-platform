"""Unit tests for AzureOpenAIAdapter."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from atp.adapters import (
    AdapterConnectionError,
    AdapterError,
    AdapterTimeoutError,
)
from atp.adapters.azure_openai import AzureOpenAIAdapter, AzureOpenAIAdapterConfig
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
        task=Task(description="Test task for Azure OpenAI"),
        constraints={"max_steps": 10},
    )


@pytest.fixture
def azure_config() -> AzureOpenAIAdapterConfig:
    """Create Azure OpenAI adapter config."""
    return AzureOpenAIAdapterConfig(
        endpoint="https://my-resource.openai.azure.com",
        deployment_name="gpt-4",
        api_key="test-api-key",
        timeout_seconds=30.0,
    )


@pytest.fixture
def mock_completion_response() -> MagicMock:
    """Create a mock chat completion response."""
    mock = MagicMock()

    # Create mock message
    mock_message = MagicMock()
    mock_message.content = "Hello, I'm your Azure OpenAI assistant."
    mock_message.tool_calls = None

    # Create mock choice
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock.choices = [mock_choice]
    mock.model = "gpt-4"

    # Usage metadata
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 30
    mock.usage = mock_usage

    return mock


@pytest.fixture
def mock_completion_response_with_tool_calls() -> MagicMock:
    """Create a mock chat completion response with tool calls."""
    mock = MagicMock()

    # Create mock tool call
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function = MagicMock()
    mock_tool_call.function.name = "search"
    mock_tool_call.function.arguments = '{"query": "test query"}'

    # Create mock message
    mock_message = MagicMock()
    mock_message.content = "I'll search for that."
    mock_message.tool_calls = [mock_tool_call]

    # Create mock choice
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "tool_calls"

    mock.choices = [mock_choice]
    mock.model = "gpt-4"

    # Usage metadata
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 15
    mock_usage.completion_tokens = 25
    mock_usage.total_tokens = 40
    mock.usage = mock_usage

    return mock


class TestAzureOpenAIAdapterConfig:
    """Tests for AzureOpenAIAdapterConfig."""

    def test_minimal_config_with_api_key(self) -> None:
        """Test creating config with minimal required fields and API key."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="my-api-key",
        )
        assert config.endpoint == "https://my-resource.openai.azure.com"
        assert config.deployment_name == "gpt-4"
        assert config.api_key == "my-api-key"
        assert config.api_version == "2024-02-15-preview"
        assert config.timeout_seconds == 300.0
        assert config.enable_function_calling is True
        assert config.use_azure_ad is False

    def test_minimal_config_with_azure_ad(self) -> None:
        """Test creating config with Azure AD authentication."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            use_azure_ad=True,
        )
        assert config.endpoint == "https://my-resource.openai.azure.com"
        assert config.deployment_name == "gpt-4"
        assert config.use_azure_ad is True
        assert config.api_key is None

    def test_full_config(self) -> None:
        """Test creating config with all fields."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com/",
            deployment_name="gpt-4-turbo",
            api_version="2024-05-01-preview",
            api_key="my-api-key",
            azure_region="eastus",
            timeout_seconds=60.0,
            temperature=0.5,
            max_tokens=2048,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            enable_session_persistence=True,
            session_id="session-123",
            system_message="You are a helpful assistant.",
            seed=42,
        )
        assert config.endpoint == "https://my-resource.openai.azure.com"
        assert config.deployment_name == "gpt-4-turbo"
        assert config.api_version == "2024-05-01-preview"
        assert config.azure_region == "eastus"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.5
        assert config.enable_session_persistence is True
        assert config.session_id == "session-123"
        assert config.system_message == "You are a helpful assistant."
        assert config.seed == 42

    def test_service_principal_config(self) -> None:
        """Test creating config with service principal auth."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            use_azure_ad=True,
            tenant_id="tenant-123",
            client_id="client-123",
            client_secret="secret-123",
        )
        assert config.use_azure_ad is True
        assert config.tenant_id == "tenant-123"
        assert config.client_id == "client-123"
        assert config.client_secret == "secret-123"

    def test_managed_identity_config(self) -> None:
        """Test creating config with managed identity auth."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            use_azure_ad=True,
            managed_identity_client_id="identity-123",
        )
        assert config.use_azure_ad is True
        assert config.managed_identity_client_id == "identity-123"

    def test_empty_endpoint_fails(self) -> None:
        """Test that empty endpoint raises an error."""
        with pytest.raises(ValueError, match="endpoint cannot be empty"):
            AzureOpenAIAdapterConfig(
                endpoint="  ",
                deployment_name="gpt-4",
                api_key="key",
            )

    def test_empty_deployment_name_fails(self) -> None:
        """Test that empty deployment_name raises an error."""
        with pytest.raises(ValueError, match="deployment_name cannot be empty"):
            AzureOpenAIAdapterConfig(
                endpoint="https://my-resource.openai.azure.com",
                deployment_name="  ",
                api_key="key",
            )

    def test_no_authentication_fails(self) -> None:
        """Test that missing authentication raises an error."""
        with pytest.raises(
            ValueError, match="Either api_key or use_azure_ad=True must be provided"
        ):
            AzureOpenAIAdapterConfig(
                endpoint="https://my-resource.openai.azure.com",
                deployment_name="gpt-4",
            )

    def test_client_id_without_secret_fails(self) -> None:
        """Test that client_id without client_secret fails."""
        with pytest.raises(
            ValueError, match="client_secret is required when client_id is provided"
        ):
            AzureOpenAIAdapterConfig(
                endpoint="https://my-resource.openai.azure.com",
                deployment_name="gpt-4",
                use_azure_ad=True,
                client_id="client-123",
            )

    def test_client_secret_without_id_fails(self) -> None:
        """Test that client_secret without client_id fails."""
        with pytest.raises(
            ValueError, match="client_id is required when client_secret is provided"
        ):
            AzureOpenAIAdapterConfig(
                endpoint="https://my-resource.openai.azure.com",
                deployment_name="gpt-4",
                use_azure_ad=True,
                client_secret="secret-123",
            )

    def test_azure_ad_credentials_without_flag_fails(self) -> None:
        """Test that Azure AD credentials without use_azure_ad=True fails."""
        with pytest.raises(
            ValueError, match="use_azure_ad must be True when using Azure AD"
        ):
            AzureOpenAIAdapterConfig(
                endpoint="https://my-resource.openai.azure.com",
                deployment_name="gpt-4",
                api_key="key",
                client_id="client-123",
                client_secret="secret-123",
            )

    def test_temperature_bounds(self) -> None:
        """Test temperature validation bounds."""
        # Valid temperature
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="key",
            temperature=1.5,
        )
        assert config.temperature == 1.5

        # Invalid temperature (too high)
        with pytest.raises(ValueError):
            AzureOpenAIAdapterConfig(
                endpoint="https://my-resource.openai.azure.com",
                deployment_name="gpt-4",
                api_key="key",
                temperature=2.5,
            )

        # Invalid temperature (negative)
        with pytest.raises(ValueError):
            AzureOpenAIAdapterConfig(
                endpoint="https://my-resource.openai.azure.com",
                deployment_name="gpt-4",
                api_key="key",
                temperature=-0.1,
            )

    def test_endpoint_trailing_slash_removed(self) -> None:
        """Test that trailing slash is removed from endpoint."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com/",
            deployment_name="gpt-4",
            api_key="key",
        )
        assert config.endpoint == "https://my-resource.openai.azure.com"


class TestAzureOpenAIAdapter:
    """Tests for AzureOpenAIAdapter."""

    def test_adapter_type(self, azure_config: AzureOpenAIAdapterConfig) -> None:
        """Test adapter type property."""
        adapter = AzureOpenAIAdapter(azure_config)
        assert adapter.adapter_type == "azure_openai"

    def test_openai_not_installed(self, azure_config: AzureOpenAIAdapterConfig) -> None:
        """Test error when openai is not installed."""
        adapter = AzureOpenAIAdapter(azure_config)

        with patch.dict("sys.modules", {"openai": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'openai'"),
            ):
                with pytest.raises(AdapterError, match="openai is required"):
                    adapter._get_openai_client()

    def test_session_id_management(
        self, azure_config: AzureOpenAIAdapterConfig
    ) -> None:
        """Test session ID management."""
        adapter = AzureOpenAIAdapter(azure_config)
        assert adapter.session_id is None

        adapter.set_session_id("my-session")
        assert adapter.session_id == "my-session"

        adapter.reset_session()
        assert adapter.session_id is None

    @pytest.mark.anyio
    async def test_execute_success(
        self,
        azure_config: AzureOpenAIAdapterConfig,
        sample_request: ATPRequest,
        mock_completion_response: MagicMock,
    ) -> None:
        """Test successful execute call."""
        adapter = AzureOpenAIAdapter(azure_config)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion_response

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
            response = await adapter.execute(sample_request)

            assert isinstance(response, ATPResponse)
            assert response.task_id == "test-task-123"
            assert response.status == ResponseStatus.COMPLETED
            assert len(response.artifacts) >= 1

            # Check output artifact
            output_artifact = response.artifacts[0]
            assert output_artifact.name == "output"
            assert (
                "Hello, I'm your Azure OpenAI assistant."
                in output_artifact.data["text"]
            )
            assert "session_id" in output_artifact.data
            assert output_artifact.data["deployment"] == "gpt-4"

            # Check metrics
            assert response.metrics is not None
            assert response.metrics.wall_time_seconds > 0
            assert response.metrics.llm_calls == 1
            assert response.metrics.input_tokens == 10
            assert response.metrics.output_tokens == 20

    @pytest.mark.anyio
    async def test_execute_with_tool_calls(
        self,
        azure_config: AzureOpenAIAdapterConfig,
        sample_request: ATPRequest,
        mock_completion_response_with_tool_calls: MagicMock,
    ) -> None:
        """Test execute with tool call response."""
        adapter = AzureOpenAIAdapter(azure_config)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            mock_completion_response_with_tool_calls
        )

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
            response = await adapter.execute(sample_request)

            assert response.status == ResponseStatus.COMPLETED
            assert response.metrics is not None
            assert response.metrics.tool_calls == 1

            # Should have tool_calls artifact
            tool_artifact = next(
                (a for a in response.artifacts if a.name == "tool_calls"), None
            )
            assert tool_artifact is not None
            assert len(tool_artifact.data["calls"]) == 1
            assert tool_artifact.data["calls"][0]["tool"] == "search"
            assert tool_artifact.data["calls"][0]["tool_call_id"] == "call_123"

    @pytest.mark.anyio
    async def test_execute_with_system_message(
        self,
        sample_request: ATPRequest,
        mock_completion_response: MagicMock,
    ) -> None:
        """Test execute includes system message."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="test-key",
            system_message="You are a helpful assistant.",
        )
        adapter = AzureOpenAIAdapter(config)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion_response
        call_args: list[dict[str, Any]] = []

        def capture_create(**kwargs: Any) -> MagicMock:
            call_args.append(kwargs)
            return mock_completion_response

        mock_client.chat.completions.create.side_effect = capture_create

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
            await adapter.execute(sample_request)

            assert len(call_args) == 1
            messages = call_args[0]["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant."

    @pytest.mark.anyio
    async def test_execute_timeout(
        self,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with timeout."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="test-key",
            timeout_seconds=0.001,
        )
        adapter = AzureOpenAIAdapter(config)

        mock_client = MagicMock()

        def slow_create(**kwargs: Any) -> MagicMock:
            import time

            time.sleep(1)  # Simulate slow response
            return MagicMock()

        mock_client.chat.completions.create.side_effect = slow_create

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
            with pytest.raises(AdapterTimeoutError):
                await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_execute_auth_error(
        self,
        azure_config: AzureOpenAIAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with authentication error."""
        adapter = AzureOpenAIAdapter(azure_config)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "AuthenticationError: 401 Invalid API key"
        )

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
            with pytest.raises(AdapterConnectionError, match="Authentication failed"):
                await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_execute_permission_denied(
        self,
        azure_config: AzureOpenAIAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with permission denied error."""
        adapter = AzureOpenAIAdapter(azure_config)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "PermissionDenied: 403 Access denied"
        )

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
            with pytest.raises(AdapterConnectionError, match="Permission denied"):
                await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_execute_deployment_not_found(
        self,
        azure_config: AzureOpenAIAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with deployment not found error."""
        adapter = AzureOpenAIAdapter(azure_config)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "NotFound: 404 Deployment not found"
        )

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
            with pytest.raises(AdapterError, match="Deployment not found"):
                await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_execute_rate_limit(
        self,
        azure_config: AzureOpenAIAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with rate limit error."""
        adapter = AzureOpenAIAdapter(azure_config)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "RateLimitError: 429 Too many requests"
        )

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
            with pytest.raises(AdapterError, match="Rate limit exceeded"):
                await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_stream_events_success(
        self,
        azure_config: AzureOpenAIAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test streaming events success."""
        adapter = AzureOpenAIAdapter(azure_config)

        # Create mock streaming chunks
        mock_chunk1 = MagicMock()
        mock_delta1 = MagicMock()
        mock_delta1.content = "Hello, "
        mock_delta1.tool_calls = None
        mock_choice1 = MagicMock()
        mock_choice1.delta = mock_delta1
        mock_choice1.finish_reason = None
        mock_chunk1.choices = [mock_choice1]
        mock_chunk1.model = "gpt-4"

        mock_chunk2 = MagicMock()
        mock_delta2 = MagicMock()
        mock_delta2.content = "world!"
        mock_delta2.tool_calls = None
        mock_choice2 = MagicMock()
        mock_choice2.delta = mock_delta2
        mock_choice2.finish_reason = "stop"
        mock_chunk2.choices = [mock_choice2]
        mock_chunk2.model = "gpt-4"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(
            [mock_chunk1, mock_chunk2]
        )

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
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
        azure_config: AzureOpenAIAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test streaming events with tool calls."""
        adapter = AzureOpenAIAdapter(azure_config)

        # Create mock streaming chunks with tool call
        mock_chunk1 = MagicMock()
        mock_delta1 = MagicMock()
        mock_delta1.content = None
        mock_tc1 = MagicMock()
        mock_tc1.index = 0
        mock_tc1.id = "call_123"
        mock_tc1.function = MagicMock()
        mock_tc1.function.name = "search"
        mock_tc1.function.arguments = '{"query": '
        mock_delta1.tool_calls = [mock_tc1]
        mock_choice1 = MagicMock()
        mock_choice1.delta = mock_delta1
        mock_choice1.finish_reason = None
        mock_chunk1.choices = [mock_choice1]
        mock_chunk1.model = "gpt-4"

        mock_chunk2 = MagicMock()
        mock_delta2 = MagicMock()
        mock_delta2.content = None
        mock_tc2 = MagicMock()
        mock_tc2.index = 0
        mock_tc2.id = None
        mock_tc2.function = MagicMock()
        mock_tc2.function.name = None
        mock_tc2.function.arguments = '"test"}'
        mock_delta2.tool_calls = [mock_tc2]
        mock_choice2 = MagicMock()
        mock_choice2.delta = mock_delta2
        mock_choice2.finish_reason = "tool_calls"
        mock_chunk2.choices = [mock_choice2]
        mock_chunk2.model = "gpt-4"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(
            [mock_chunk1, mock_chunk2]
        )

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
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

            # Last item should be response with tool calls
            assert isinstance(events[-1], ATPResponse)
            assert events[-1].metrics is not None
            assert events[-1].metrics.tool_calls >= 1

    @pytest.mark.anyio
    async def test_stream_events_timeout(
        self,
        sample_request: ATPRequest,
    ) -> None:
        """Test streaming events with timeout."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="test-key",
            timeout_seconds=0.001,
        )
        adapter = AzureOpenAIAdapter(config)

        mock_client = MagicMock()

        def slow_create(**kwargs: Any) -> list[MagicMock]:
            import time

            time.sleep(1)
            return []

        mock_client.chat.completions.create.side_effect = slow_create

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
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
        self, azure_config: AzureOpenAIAdapterConfig
    ) -> None:
        """Test health check success."""
        adapter = AzureOpenAIAdapter(azure_config)

        mock_client = MagicMock()

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
            result = await adapter.health_check()
            assert result is True

    @pytest.mark.anyio
    async def test_health_check_failure(
        self, azure_config: AzureOpenAIAdapterConfig
    ) -> None:
        """Test health check failure."""
        adapter = AzureOpenAIAdapter(azure_config)

        with patch.object(
            adapter, "_get_openai_client", side_effect=AdapterError("Connection failed")
        ):
            result = await adapter.health_check()
            assert result is False

    @pytest.mark.anyio
    async def test_cleanup(self, azure_config: AzureOpenAIAdapterConfig) -> None:
        """Test cleanup releases resources."""
        adapter = AzureOpenAIAdapter(azure_config)

        adapter._client = MagicMock()
        adapter._session_id = "test-session"
        adapter._conversation_history = [{"role": "user", "content": "test"}]
        adapter._initialized = True

        await adapter.cleanup()

        assert adapter._client is None
        assert adapter._session_id is None
        assert adapter._conversation_history == []
        assert adapter._initialized is False

    @pytest.mark.anyio
    async def test_session_persistence(
        self,
        sample_request: ATPRequest,
        mock_completion_response: MagicMock,
    ) -> None:
        """Test session persistence across requests."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="test-key",
            enable_session_persistence=True,
        )
        adapter = AzureOpenAIAdapter(config)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion_response

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
            # First call - should create a session
            await adapter.execute(sample_request)
            session_id_1 = adapter.session_id

            assert session_id_1 is not None

            # Second call - should use same session
            await adapter.execute(sample_request)
            session_id_2 = adapter.session_id

            assert session_id_2 == session_id_1

            # Conversation history should be populated
            assert len(adapter._conversation_history) == 4  # 2 user + 2 assistant

    @pytest.mark.anyio
    async def test_no_session_persistence(
        self,
        azure_config: AzureOpenAIAdapterConfig,
        sample_request: ATPRequest,
        mock_completion_response: MagicMock,
    ) -> None:
        """Test without session persistence creates new sessions."""
        adapter = AzureOpenAIAdapter(azure_config)
        assert adapter._config.enable_session_persistence is False

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion_response

        with patch.object(adapter, "_get_openai_client", return_value=mock_client):
            # First call
            response1 = await adapter.execute(sample_request)
            session_id_1 = response1.artifacts[0].data.get("session_id")

            # Second call
            response2 = await adapter.execute(sample_request)
            session_id_2 = response2.artifacts[0].data.get("session_id")

            # Should have different session IDs
            assert session_id_1 != session_id_2


class TestAzureOpenAIMessageBuilding:
    """Tests for message building functionality."""

    @pytest.fixture
    def adapter(self) -> AzureOpenAIAdapter:
        """Create an Azure OpenAI adapter for testing."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="test-key",
        )
        return AzureOpenAIAdapter(config)

    def test_build_messages_basic(self, adapter: AzureOpenAIAdapter) -> None:
        """Test building basic messages."""
        request = ATPRequest(
            task_id="task-1",
            task=Task(description="Hello, world!"),
        )

        messages = adapter._build_messages(request)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, world!"

    def test_build_messages_with_system(self) -> None:
        """Test building messages with system message."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="test-key",
            system_message="You are a helpful assistant.",
        )
        adapter = AzureOpenAIAdapter(config)

        request = ATPRequest(
            task_id="task-1",
            task=Task(description="Hello!"),
        )

        messages = adapter._build_messages(request)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"

    def test_build_messages_with_context(self, adapter: AzureOpenAIAdapter) -> None:
        """Test building messages with context from input_data."""
        request = ATPRequest(
            task_id="task-1",
            task=Task(
                description="What does it say?",
                input_data={"context": "The document contains important information."},
            ),
        )

        messages = adapter._build_messages(request)

        assert len(messages) == 1
        assert (
            "Context: The document contains important information."
            in messages[0]["content"]
        )
        assert "What does it say?" in messages[0]["content"]


class TestAzureOpenAICompletionParams:
    """Tests for completion parameter building."""

    def test_build_basic_params(self) -> None:
        """Test building basic completion parameters."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="test-key",
        )
        adapter = AzureOpenAIAdapter(config)

        messages = [{"role": "user", "content": "Hello"}]
        params = adapter._build_completion_params(messages)

        assert params["model"] == "gpt-4"
        assert params["messages"] == messages
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 4096
        assert params["top_p"] == 1.0
        assert "tools" not in params  # No tools configured

    def test_build_params_with_tools(self) -> None:
        """Test building params with tools."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="test-key",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search for information",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                }
            ],
            tool_choice="auto",
        )
        adapter = AzureOpenAIAdapter(config)

        messages = [{"role": "user", "content": "Search for something"}]
        params = adapter._build_completion_params(messages)

        assert "tools" in params
        assert len(params["tools"]) == 1
        assert params["tool_choice"] == "auto"

    def test_build_params_with_response_format(self) -> None:
        """Test building params with response format."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="test-key",
            response_format={"type": "json_object"},
        )
        adapter = AzureOpenAIAdapter(config)

        messages = [{"role": "user", "content": "Generate JSON"}]
        params = adapter._build_completion_params(messages)

        assert params["response_format"] == {"type": "json_object"}

    def test_build_params_with_seed(self) -> None:
        """Test building params with seed."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="test-key",
            seed=42,
        )
        adapter = AzureOpenAIAdapter(config)

        messages = [{"role": "user", "content": "Hello"}]
        params = adapter._build_completion_params(messages)

        assert params["seed"] == 42


class TestAzureOpenAIToolExtraction:
    """Tests for tool call extraction."""

    @pytest.fixture
    def adapter(self) -> AzureOpenAIAdapter:
        """Create an Azure OpenAI adapter for testing."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="test-key",
        )
        return AzureOpenAIAdapter(config)

    def test_extract_single_tool_call(self, adapter: AzureOpenAIAdapter) -> None:
        """Test extracting a single tool call."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_abc123"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "Seattle"}'

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]

        events, sequence = adapter._extract_tool_calls(mock_message, "task-1", 0)

        assert len(events) == 1
        assert events[0].event_type == EventType.TOOL_CALL
        assert events[0].payload["tool"] == "get_weather"
        assert events[0].payload["tool_call_id"] == "call_abc123"
        assert events[0].payload["input"]["location"] == "Seattle"
        assert sequence == 1

    def test_extract_multiple_tool_calls(self, adapter: AzureOpenAIAdapter) -> None:
        """Test extracting multiple tool calls."""
        mock_tc1 = MagicMock()
        mock_tc1.id = "call_1"
        mock_tc1.function = MagicMock()
        mock_tc1.function.name = "search"
        mock_tc1.function.arguments = '{"query": "test"}'

        mock_tc2 = MagicMock()
        mock_tc2.id = "call_2"
        mock_tc2.function = MagicMock()
        mock_tc2.function.name = "calculate"
        mock_tc2.function.arguments = '{"expression": "1+1"}'

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tc1, mock_tc2]

        events, sequence = adapter._extract_tool_calls(mock_message, "task-1", 0)

        assert len(events) == 2
        assert events[0].payload["tool"] == "search"
        assert events[1].payload["tool"] == "calculate"
        assert sequence == 2

    def test_extract_no_tool_calls(self, adapter: AzureOpenAIAdapter) -> None:
        """Test extraction when no tool calls present."""
        mock_message = MagicMock()
        mock_message.tool_calls = None

        events, sequence = adapter._extract_tool_calls(mock_message, "task-1", 5)

        assert len(events) == 0
        assert sequence == 5

    def test_extract_tool_call_invalid_json(self, adapter: AzureOpenAIAdapter) -> None:
        """Test extraction with invalid JSON arguments."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "broken"
        mock_tool_call.function.arguments = "not valid json"

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]

        events, sequence = adapter._extract_tool_calls(mock_message, "task-1", 0)

        assert len(events) == 1
        assert events[0].payload["input"] == {}  # Falls back to empty dict


class TestAzureOpenAIUsageExtraction:
    """Tests for usage metadata extraction."""

    @pytest.fixture
    def adapter(self) -> AzureOpenAIAdapter:
        """Create an Azure OpenAI adapter for testing."""
        config = AzureOpenAIAdapterConfig(
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_key="test-key",
        )
        return AzureOpenAIAdapter(config)

    def test_extract_usage(self, adapter: AzureOpenAIAdapter) -> None:
        """Test extracting usage metadata."""
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 200
        mock_usage.total_tokens = 300

        mock_response = MagicMock()
        mock_response.usage = mock_usage

        usage = adapter._extract_usage(mock_response)

        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 200
        assert usage["total_tokens"] == 300

    def test_extract_usage_missing(self, adapter: AzureOpenAIAdapter) -> None:
        """Test extraction when usage is missing."""
        mock_response = MagicMock()
        mock_response.usage = None

        usage = adapter._extract_usage(mock_response)

        assert usage == {}
