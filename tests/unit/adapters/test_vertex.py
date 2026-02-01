"""Unit tests for VertexAdapter."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from atp.adapters import (
    AdapterConnectionError,
    AdapterError,
    AdapterTimeoutError,
)
from atp.adapters.vertex import VertexAdapter, VertexAdapterConfig
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
        task=Task(description="Test task for Vertex AI"),
        constraints={"max_steps": 10},
    )


@pytest.fixture
def vertex_config() -> VertexAdapterConfig:
    """Create Vertex AI adapter config."""
    return VertexAdapterConfig(
        project_id="test-project",
        location="us-central1",
        model_name="gemini-1.5-pro",
        timeout_seconds=30.0,
    )


@pytest.fixture
def mock_response() -> MagicMock:
    """Create a mock Vertex AI response."""
    mock = MagicMock()

    # Create mock candidate with text part
    mock_text_part = MagicMock()
    mock_text_part.text = "Hello, I'm your Vertex AI assistant."

    mock_content = MagicMock()
    mock_content.parts = [mock_text_part]

    mock_candidate = MagicMock()
    mock_candidate.content = mock_content
    mock_candidate.grounding_metadata = None

    mock.candidates = [mock_candidate]

    # Usage metadata
    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 10
    mock_usage.candidates_token_count = 20
    mock_usage.total_token_count = 30
    mock.usage_metadata = mock_usage

    return mock


@pytest.fixture
def mock_response_with_function_call() -> MagicMock:
    """Create a mock Vertex AI response with function call."""
    mock = MagicMock()

    # Create mock candidate with function call part
    mock_func_call = MagicMock()
    mock_func_call.name = "search"
    mock_func_call.args = {"query": "test query"}

    mock_func_part = MagicMock()
    mock_func_part.function_call = mock_func_call
    mock_func_part.text = None

    mock_text_part = MagicMock()
    mock_text_part.text = "Based on my search, here are the results."
    mock_text_part.function_call = None

    mock_content = MagicMock()
    mock_content.parts = [mock_func_part, mock_text_part]

    mock_candidate = MagicMock()
    mock_candidate.content = mock_content
    mock_candidate.grounding_metadata = None

    mock.candidates = [mock_candidate]

    # Usage metadata
    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 15
    mock_usage.candidates_token_count = 25
    mock_usage.total_token_count = 40
    mock.usage_metadata = mock_usage

    return mock


@pytest.fixture
def mock_response_with_grounding() -> MagicMock:
    """Create a mock Vertex AI response with grounding metadata."""
    mock = MagicMock()

    # Create mock candidate with text part
    mock_text_part = MagicMock()
    mock_text_part.text = "According to recent news..."

    mock_content = MagicMock()
    mock_content.parts = [mock_text_part]

    # Create grounding metadata
    mock_chunk = MagicMock()
    mock_chunk.web = MagicMock()
    mock_chunk.web.uri = "https://example.com/news"
    mock_chunk.web.title = "News Article"

    mock_support = MagicMock()
    mock_support.segment = "According to recent news..."
    mock_support.grounding_chunk_indices = [0]

    mock_grounding = MagicMock()
    mock_grounding.search_entry_point = "https://google.com/search?q=test"
    mock_grounding.grounding_chunks = [mock_chunk]
    mock_grounding.grounding_supports = [mock_support]

    mock_candidate = MagicMock()
    mock_candidate.content = mock_content
    mock_candidate.grounding_metadata = mock_grounding

    mock.candidates = [mock_candidate]

    # Usage metadata
    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 12
    mock_usage.candidates_token_count = 18
    mock_usage.total_token_count = 30
    mock.usage_metadata = mock_usage

    return mock


class TestVertexAdapterConfig:
    """Tests for VertexAdapterConfig."""

    def test_minimal_config(self) -> None:
        """Test creating config with minimal required fields."""
        config = VertexAdapterConfig(project_id="my-project")
        assert config.project_id == "my-project"
        assert config.location == "us-central1"
        assert config.model_name == "gemini-1.5-pro"
        assert config.timeout_seconds == 300.0
        assert config.enable_function_calling is True

    def test_full_config(self) -> None:
        """Test creating config with all fields."""
        config = VertexAdapterConfig(
            project_id="my-project",
            location="us-west1",
            agent_id="agent-123",
            model_name="gemini-1.5-flash",
            credentials_path="/path/to/credentials.json",
            timeout_seconds=60.0,
            enable_session_persistence=True,
            session_id="session-123",
            temperature=0.5,
            max_output_tokens=4096,
            top_p=0.9,
            top_k=40,
            enable_function_calling=True,
            enable_grounding=True,
            grounding_source="google_search",
            system_instruction="You are a helpful assistant.",
        )
        assert config.project_id == "my-project"
        assert config.location == "us-west1"
        assert config.agent_id == "agent-123"
        assert config.model_name == "gemini-1.5-flash"
        assert config.credentials_path == "/path/to/credentials.json"
        assert config.enable_session_persistence is True
        assert config.temperature == 0.5
        assert config.max_output_tokens == 4096
        assert config.top_k == 40
        assert config.enable_grounding is True
        assert config.grounding_source == "google_search"
        assert config.system_instruction == "You are a helpful assistant."

    def test_empty_project_id_fails(self) -> None:
        """Test that empty project_id raises an error."""
        with pytest.raises(ValueError, match="project_id cannot be empty"):
            VertexAdapterConfig(project_id="  ")

    def test_empty_location_fails(self) -> None:
        """Test that empty location raises an error."""
        with pytest.raises(ValueError, match="location cannot be empty"):
            VertexAdapterConfig(project_id="my-project", location="  ")

    def test_empty_model_name_fails(self) -> None:
        """Test that empty model_name raises an error."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            VertexAdapterConfig(project_id="my-project", model_name="  ")

    def test_grounding_source_without_enable_grounding_fails(self) -> None:
        """Test that grounding_source without enable_grounding fails."""
        with pytest.raises(
            ValueError, match="enable_grounding must be True when grounding_source"
        ):
            VertexAdapterConfig(
                project_id="my-project",
                grounding_source="google_search",
            )

    def test_temperature_bounds(self) -> None:
        """Test temperature validation bounds."""
        # Valid temperature
        config = VertexAdapterConfig(project_id="my-project", temperature=1.5)
        assert config.temperature == 1.5

        # Invalid temperature (too high)
        with pytest.raises(ValueError):
            VertexAdapterConfig(project_id="my-project", temperature=2.5)

        # Invalid temperature (negative)
        with pytest.raises(ValueError):
            VertexAdapterConfig(project_id="my-project", temperature=-0.1)


class TestVertexAdapter:
    """Tests for VertexAdapter."""

    def test_adapter_type(self, vertex_config: VertexAdapterConfig) -> None:
        """Test adapter type property."""
        adapter = VertexAdapter(vertex_config)
        assert adapter.adapter_type == "vertex"

    def test_vertexai_not_installed(self, vertex_config: VertexAdapterConfig) -> None:
        """Test error when google-cloud-aiplatform is not installed."""
        adapter = VertexAdapter(vertex_config)

        with patch.dict("sys.modules", {"vertexai": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'vertexai'"),
            ):
                with pytest.raises(
                    AdapterError, match="google-cloud-aiplatform is required"
                ):
                    adapter._get_vertexai_module()

    def test_session_id_management(self, vertex_config: VertexAdapterConfig) -> None:
        """Test session ID management."""
        adapter = VertexAdapter(vertex_config)
        assert adapter.session_id is None

        adapter.set_session_id("my-session")
        assert adapter.session_id == "my-session"

        adapter.reset_session()
        assert adapter.session_id is None

    @pytest.mark.anyio
    async def test_execute_success(
        self,
        vertex_config: VertexAdapterConfig,
        sample_request: ATPRequest,
        mock_response: MagicMock,
    ) -> None:
        """Test successful execute call."""
        adapter = VertexAdapter(vertex_config)

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        with patch.object(adapter, "_get_model", return_value=mock_model):
            response = await adapter.execute(sample_request)

            assert isinstance(response, ATPResponse)
            assert response.task_id == "test-task-123"
            assert response.status == ResponseStatus.COMPLETED
            assert len(response.artifacts) >= 1

            # Check output artifact
            output_artifact = response.artifacts[0]
            assert output_artifact.name == "output"
            assert (
                "Hello, I'm your Vertex AI assistant." in output_artifact.data["text"]
            )
            assert "session_id" in output_artifact.data

            # Check metrics
            assert response.metrics is not None
            assert response.metrics.wall_time_seconds > 0
            assert response.metrics.llm_calls == 1
            assert response.metrics.input_tokens == 10
            assert response.metrics.output_tokens == 20

    @pytest.mark.anyio
    async def test_execute_with_function_call(
        self,
        vertex_config: VertexAdapterConfig,
        sample_request: ATPRequest,
        mock_response_with_function_call: MagicMock,
    ) -> None:
        """Test execute with function call response."""
        adapter = VertexAdapter(vertex_config)

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response_with_function_call

        with patch.object(adapter, "_get_model", return_value=mock_model):
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

    @pytest.mark.anyio
    async def test_execute_with_grounding(
        self,
        vertex_config: VertexAdapterConfig,
        sample_request: ATPRequest,
        mock_response_with_grounding: MagicMock,
    ) -> None:
        """Test execute with grounding metadata."""
        adapter = VertexAdapter(vertex_config)

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response_with_grounding

        with patch.object(adapter, "_get_model", return_value=mock_model):
            response = await adapter.execute(sample_request)

            assert response.status == ResponseStatus.COMPLETED

            # Should have grounding artifact
            grounding_artifact = next(
                (a for a in response.artifacts if a.name == "grounding"), None
            )
            assert grounding_artifact is not None
            assert "grounding_chunks" in grounding_artifact.data

    @pytest.mark.anyio
    async def test_execute_timeout(
        self,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with timeout."""
        config = VertexAdapterConfig(
            project_id="test-project",
            timeout_seconds=0.001,
        )
        adapter = VertexAdapter(config)

        mock_model = MagicMock()

        def slow_generate(*args: Any, **kwargs: Any) -> MagicMock:
            import time

            time.sleep(1)  # Simulate slow response
            return MagicMock()

        mock_model.generate_content.side_effect = slow_generate

        with patch.object(adapter, "_get_model", return_value=mock_model):
            with pytest.raises(AdapterTimeoutError):
                await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_execute_permission_denied(
        self,
        vertex_config: VertexAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with permission denied error."""
        adapter = VertexAdapter(vertex_config)

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception(
            "PermissionDenied: 403 Access denied"
        )

        with patch.object(adapter, "_get_model", return_value=mock_model):
            with pytest.raises(AdapterConnectionError, match="Permission denied"):
                await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_execute_model_not_found(
        self,
        vertex_config: VertexAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with model not found error."""
        adapter = VertexAdapter(vertex_config)

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception(
            "NotFound: 404 Model not found"
        )

        with patch.object(adapter, "_get_model", return_value=mock_model):
            with pytest.raises(AdapterError, match="Model not found"):
                await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_execute_quota_exceeded(
        self,
        vertex_config: VertexAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with quota exceeded error."""
        adapter = VertexAdapter(vertex_config)

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception(
            "ResourceExhausted: 429 Quota exceeded"
        )

        with patch.object(adapter, "_get_model", return_value=mock_model):
            with pytest.raises(AdapterError, match="quota exceeded"):
                await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_stream_events_success(
        self,
        vertex_config: VertexAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test streaming events success."""
        adapter = VertexAdapter(vertex_config)

        # Create mock streaming response
        mock_chunk1 = MagicMock()
        mock_text_part1 = MagicMock()
        mock_text_part1.text = "Hello, "
        mock_content1 = MagicMock()
        mock_content1.parts = [mock_text_part1]
        mock_candidate1 = MagicMock()
        mock_candidate1.content = mock_content1
        mock_candidate1.grounding_metadata = None
        mock_chunk1.candidates = [mock_candidate1]
        mock_chunk1.usage_metadata = None

        mock_chunk2 = MagicMock()
        mock_text_part2 = MagicMock()
        mock_text_part2.text = "world!"
        mock_content2 = MagicMock()
        mock_content2.parts = [mock_text_part2]
        mock_candidate2 = MagicMock()
        mock_candidate2.content = mock_content2
        mock_candidate2.grounding_metadata = None
        mock_chunk2.candidates = [mock_candidate2]
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 5
        mock_usage.candidates_token_count = 10
        mock_usage.total_token_count = 15
        mock_chunk2.usage_metadata = mock_usage

        mock_model = MagicMock()
        mock_model.generate_content.return_value = [mock_chunk1, mock_chunk2]

        with patch.object(adapter, "_get_model", return_value=mock_model):
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
    async def test_stream_events_timeout(
        self,
        sample_request: ATPRequest,
    ) -> None:
        """Test streaming events with timeout."""
        config = VertexAdapterConfig(
            project_id="test-project",
            timeout_seconds=0.001,
        )
        adapter = VertexAdapter(config)

        mock_model = MagicMock()

        def slow_generate(*args: Any, **kwargs: Any) -> list[MagicMock]:
            import time

            time.sleep(1)
            return []

        mock_model.generate_content.side_effect = slow_generate

        with patch.object(adapter, "_get_model", return_value=mock_model):
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
        self, vertex_config: VertexAdapterConfig
    ) -> None:
        """Test health check success."""
        adapter = VertexAdapter(vertex_config)

        mock_model = MagicMock()

        with patch.object(adapter, "_initialize_vertexai"):
            with patch.object(adapter, "_get_model", return_value=mock_model):
                result = await adapter.health_check()
                assert result is True

    @pytest.mark.anyio
    async def test_health_check_failure(
        self, vertex_config: VertexAdapterConfig
    ) -> None:
        """Test health check failure."""
        adapter = VertexAdapter(vertex_config)

        with patch.object(
            adapter, "_initialize_vertexai", side_effect=AdapterError("Init failed")
        ):
            result = await adapter.health_check()
            assert result is False

    @pytest.mark.anyio
    async def test_cleanup(self, vertex_config: VertexAdapterConfig) -> None:
        """Test cleanup releases resources."""
        adapter = VertexAdapter(vertex_config)

        adapter._model = MagicMock()
        adapter._chat_session = MagicMock()
        adapter._session_id = "test-session"
        adapter._initialized = True

        await adapter.cleanup()

        assert adapter._model is None
        assert adapter._chat_session is None
        assert adapter._session_id is None
        assert adapter._initialized is False

    @pytest.mark.anyio
    async def test_session_persistence(
        self,
        sample_request: ATPRequest,
        mock_response: MagicMock,
    ) -> None:
        """Test session persistence across requests."""
        config = VertexAdapterConfig(
            project_id="test-project",
            enable_session_persistence=True,
        )
        adapter = VertexAdapter(config)

        mock_model = MagicMock()
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = mock_response
        mock_model.start_chat.return_value = mock_chat

        with patch.object(adapter, "_get_model", return_value=mock_model):
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
        vertex_config: VertexAdapterConfig,
        sample_request: ATPRequest,
        mock_response: MagicMock,
    ) -> None:
        """Test without session persistence creates new sessions."""
        adapter = VertexAdapter(vertex_config)
        assert adapter._config.enable_session_persistence is False

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        with patch.object(adapter, "_get_model", return_value=mock_model):
            # First call
            response1 = await adapter.execute(sample_request)
            session_id_1 = response1.artifacts[0].data.get("session_id")

            # Second call
            response2 = await adapter.execute(sample_request)
            session_id_2 = response2.artifacts[0].data.get("session_id")

            # Should have different session IDs
            assert session_id_1 != session_id_2


class TestVertexToolExtraction:
    """Tests for Vertex AI tool call extraction."""

    @pytest.fixture
    def adapter(self) -> VertexAdapter:
        """Create a Vertex AI adapter for testing."""
        config = VertexAdapterConfig(project_id="test-project")
        return VertexAdapter(config)

    def test_extract_function_call(self, adapter: VertexAdapter) -> None:
        """Test extracting function call from response."""
        mock_func_call = MagicMock()
        mock_func_call.name = "get_weather"
        mock_func_call.args = {"location": "Seattle", "unit": "celsius"}

        mock_part = MagicMock()
        mock_part.function_call = mock_func_call

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        events, sequence = adapter._extract_tool_calls(mock_response, "task-1", 0)

        assert len(events) == 1
        assert events[0].event_type == EventType.TOOL_CALL
        assert events[0].payload["tool"] == "get_weather"
        assert events[0].payload["input"]["location"] == "Seattle"
        assert events[0].payload["status"] == "started"
        assert sequence == 1

    def test_extract_multiple_function_calls(self, adapter: VertexAdapter) -> None:
        """Test extracting multiple function calls."""
        mock_func_call1 = MagicMock()
        mock_func_call1.name = "search"
        mock_func_call1.args = {"query": "test"}

        mock_func_call2 = MagicMock()
        mock_func_call2.name = "calculate"
        mock_func_call2.args = {"expression": "1+1"}

        mock_part1 = MagicMock()
        mock_part1.function_call = mock_func_call1

        mock_part2 = MagicMock()
        mock_part2.function_call = mock_func_call2

        mock_content = MagicMock()
        mock_content.parts = [mock_part1, mock_part2]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        events, sequence = adapter._extract_tool_calls(mock_response, "task-1", 0)

        assert len(events) == 2
        assert events[0].payload["tool"] == "search"
        assert events[1].payload["tool"] == "calculate"
        assert sequence == 2

    def test_extract_no_function_calls(self, adapter: VertexAdapter) -> None:
        """Test extraction when no function calls present."""
        mock_text_part = MagicMock()
        mock_text_part.text = "Just text, no function call"
        mock_text_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_text_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        events, sequence = adapter._extract_tool_calls(mock_response, "task-1", 5)

        assert len(events) == 0
        assert sequence == 5


class TestVertexResponseExtraction:
    """Tests for Vertex AI response content extraction."""

    @pytest.fixture
    def adapter(self) -> VertexAdapter:
        """Create a Vertex AI adapter for testing."""
        config = VertexAdapterConfig(project_id="test-project")
        return VertexAdapter(config)

    def test_extract_text_from_single_part(self, adapter: VertexAdapter) -> None:
        """Test extracting text from single part response."""
        mock_text_part = MagicMock()
        mock_text_part.text = "Hello, world!"

        mock_content = MagicMock()
        mock_content.parts = [mock_text_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        text = adapter._extract_response_text(mock_response)
        assert text == "Hello, world!"

    def test_extract_text_from_multiple_parts(self, adapter: VertexAdapter) -> None:
        """Test extracting text from multiple parts."""
        mock_part1 = MagicMock()
        mock_part1.text = "Part 1. "

        mock_part2 = MagicMock()
        mock_part2.text = "Part 2."

        mock_content = MagicMock()
        mock_content.parts = [mock_part1, mock_part2]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        text = adapter._extract_response_text(mock_response)
        assert text == "Part 1. Part 2."

    def test_extract_usage_metadata(self, adapter: VertexAdapter) -> None:
        """Test extracting usage metadata."""
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 100
        mock_usage.candidates_token_count = 200
        mock_usage.total_token_count = 300

        mock_response = MagicMock()
        mock_response.usage_metadata = mock_usage

        usage = adapter._extract_usage_metadata(mock_response)

        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 200
        assert usage["total_tokens"] == 300

    def test_extract_grounding_metadata(self, adapter: VertexAdapter) -> None:
        """Test extracting grounding metadata."""
        mock_chunk = MagicMock()
        mock_chunk.web = MagicMock()
        mock_chunk.web.uri = "https://example.com"
        mock_chunk.web.title = "Example"

        mock_support = MagicMock()
        mock_support.segment = "test segment"
        mock_support.grounding_chunk_indices = [0]

        mock_grounding = MagicMock()
        mock_grounding.search_entry_point = "https://google.com/search"
        mock_grounding.grounding_chunks = [mock_chunk]
        mock_grounding.grounding_supports = [mock_support]

        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = mock_grounding

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        grounding = adapter._extract_grounding_metadata(mock_response)

        assert grounding is not None
        assert grounding["search_entry_point"] == "https://google.com/search"
        assert len(grounding["grounding_chunks"]) == 1
        assert grounding["grounding_chunks"][0]["web"]["uri"] == "https://example.com"

    def test_extract_no_grounding_metadata(self, adapter: VertexAdapter) -> None:
        """Test extraction when no grounding metadata present."""
        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = None

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        grounding = adapter._extract_grounding_metadata(mock_response)

        assert grounding is None


class TestVertexGenerationConfig:
    """Tests for Vertex AI generation configuration."""

    def test_get_generation_config_defaults(self) -> None:
        """Test getting default generation config."""
        config = VertexAdapterConfig(project_id="test-project")
        adapter = VertexAdapter(config)

        gen_config = adapter._get_generation_config()

        assert gen_config["temperature"] == 0.7
        assert gen_config["max_output_tokens"] == 8192
        assert gen_config["top_p"] == 0.95
        assert "top_k" not in gen_config  # Not set by default

    def test_get_generation_config_custom(self) -> None:
        """Test getting custom generation config."""
        config = VertexAdapterConfig(
            project_id="test-project",
            temperature=0.5,
            max_output_tokens=4096,
            top_p=0.8,
            top_k=50,
        )
        adapter = VertexAdapter(config)

        gen_config = adapter._get_generation_config()

        assert gen_config["temperature"] == 0.5
        assert gen_config["max_output_tokens"] == 4096
        assert gen_config["top_p"] == 0.8
        assert gen_config["top_k"] == 50
