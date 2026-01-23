"""Unit tests for HTTPAdapter."""

import httpx
import pytest

from atp.adapters import (
    AdapterConnectionError,
    AdapterResponseError,
    AdapterTimeoutError,
    HTTPAdapter,
    HTTPAdapterConfig,
)
from atp.protocol import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    ResponseStatus,
    Task,
)


@pytest.fixture
def sample_request() -> ATPRequest:
    """Create a sample ATP request for testing."""
    return ATPRequest(
        task_id="test-task-123",
        task=Task(description="Test task"),
        constraints={"max_steps": 10},
    )


@pytest.fixture
def sample_response_data() -> dict:
    """Create sample response data."""
    return {
        "version": "1.0",
        "task_id": "test-task-123",
        "status": "completed",
        "artifacts": [],
        "metrics": {"total_tokens": 100, "total_steps": 5},
    }


@pytest.fixture
def http_config() -> HTTPAdapterConfig:
    """Create HTTP adapter config."""
    return HTTPAdapterConfig(
        endpoint="http://localhost:8000/agent",
        timeout_seconds=30.0,
        allow_internal=True,  # Allow localhost for testing
    )


class TestHTTPAdapterConfig:
    """Tests for HTTPAdapterConfig."""

    def test_minimal_config(self) -> None:
        """Test creating config with minimal required fields."""
        config = HTTPAdapterConfig(endpoint="http://example.com")
        assert config.endpoint == "http://example.com"
        assert config.timeout_seconds == 300.0
        assert config.headers == {}
        assert config.verify_ssl is True

    def test_full_config(self) -> None:
        """Test creating config with all fields."""
        config = HTTPAdapterConfig(
            endpoint="http://example.com/agent",
            timeout_seconds=60.0,
            headers={"Authorization": "Bearer token"},
            verify_ssl=False,
            stream_endpoint="http://example.com/stream",
            health_endpoint="http://example.com/health",
        )
        assert config.endpoint == "http://example.com/agent"
        assert config.timeout_seconds == 60.0
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.verify_ssl is False
        assert config.stream_endpoint == "http://example.com/stream"
        assert config.health_endpoint == "http://example.com/health"


class TestHTTPAdapter:
    """Tests for HTTPAdapter."""

    def test_adapter_type(self, http_config: HTTPAdapterConfig) -> None:
        """Test adapter type property."""
        adapter = HTTPAdapter(http_config)
        assert adapter.adapter_type == "http"

    @pytest.mark.anyio
    async def test_execute_success(
        self,
        http_config: HTTPAdapterConfig,
        sample_request: ATPRequest,
        sample_response_data: dict,
    ) -> None:
        """Test successful execute call."""
        adapter = HTTPAdapter(http_config)

        async with adapter:
            # Mock the httpx client
            transport = httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json=sample_response_data,
                )
            )
            adapter._client = httpx.AsyncClient(transport=transport)

            response = await adapter.execute(sample_request)

            assert isinstance(response, ATPResponse)
            assert response.task_id == "test-task-123"
            assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_execute_http_error(
        self,
        http_config: HTTPAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with HTTP error response."""
        adapter = HTTPAdapter(http_config)

        async with adapter:
            transport = httpx.MockTransport(
                lambda request: httpx.Response(
                    500,
                    text="Internal Server Error",
                )
            )
            adapter._client = httpx.AsyncClient(transport=transport)

            with pytest.raises(AdapterResponseError) as exc_info:
                await adapter.execute(sample_request)

            assert exc_info.value.status_code == 500
            assert "500" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_invalid_json(
        self,
        http_config: HTTPAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with invalid JSON response."""
        adapter = HTTPAdapter(http_config)

        async with adapter:
            transport = httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    text="not valid json",
                )
            )
            adapter._client = httpx.AsyncClient(transport=transport)

            with pytest.raises(AdapterResponseError) as exc_info:
                await adapter.execute(sample_request)

            assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_invalid_response_format(
        self,
        http_config: HTTPAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with invalid ATP response format."""
        adapter = HTTPAdapter(http_config)

        async with adapter:
            transport = httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"invalid": "format"},
                )
            )
            adapter._client = httpx.AsyncClient(transport=transport)

            with pytest.raises(AdapterResponseError) as exc_info:
                await adapter.execute(sample_request)

            assert "Invalid ATP Response" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_timeout(
        self,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with timeout."""
        config = HTTPAdapterConfig(
            endpoint="http://localhost:8000/agent",
            timeout_seconds=0.001,  # Very short timeout
            allow_internal=True,  # Allow localhost for testing
        )
        adapter = HTTPAdapter(config)

        async def raise_timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("Connection timed out")

        async with adapter:
            transport = httpx.MockTransport(raise_timeout)
            adapter._client = httpx.AsyncClient(transport=transport)

            with pytest.raises(AdapterTimeoutError) as exc_info:
                await adapter.execute(sample_request)

            assert exc_info.value.timeout_seconds == 0.001

    @pytest.mark.anyio
    async def test_execute_connection_error(
        self,
        http_config: HTTPAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with connection error."""
        adapter = HTTPAdapter(http_config)

        async def raise_connect_error(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        async with adapter:
            transport = httpx.MockTransport(raise_connect_error)
            adapter._client = httpx.AsyncClient(transport=transport)

            with pytest.raises(AdapterConnectionError) as exc_info:
                await adapter.execute(sample_request)

            assert "localhost:8000" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_health_check_success(
        self,
        http_config: HTTPAdapterConfig,
    ) -> None:
        """Test health check success."""
        adapter = HTTPAdapter(http_config)

        async with adapter:
            transport = httpx.MockTransport(
                lambda request: httpx.Response(200, text="OK")
            )
            adapter._client = httpx.AsyncClient(transport=transport)

            result = await adapter.health_check()
            assert result is True

    @pytest.mark.anyio
    async def test_health_check_failure(
        self,
        http_config: HTTPAdapterConfig,
    ) -> None:
        """Test health check failure."""
        adapter = HTTPAdapter(http_config)

        async def raise_error(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        async with adapter:
            transport = httpx.MockTransport(raise_error)
            adapter._client = httpx.AsyncClient(transport=transport)

            result = await adapter.health_check()
            assert result is False

    @pytest.mark.anyio
    async def test_health_check_server_error(
        self,
        http_config: HTTPAdapterConfig,
    ) -> None:
        """Test health check with server error returns False."""
        adapter = HTTPAdapter(http_config)

        async with adapter:
            transport = httpx.MockTransport(
                lambda request: httpx.Response(500, text="Error")
            )
            adapter._client = httpx.AsyncClient(transport=transport)

            result = await adapter.health_check()
            assert result is False

    @pytest.mark.anyio
    async def test_health_check_custom_endpoint(self) -> None:
        """Test health check with custom endpoint."""
        config = HTTPAdapterConfig(
            endpoint="http://localhost:8000/agent",
            health_endpoint="http://localhost:8000/health",
        )
        adapter = HTTPAdapter(config)

        requests_made: list[str] = []

        def capture_request(request: httpx.Request) -> httpx.Response:
            requests_made.append(str(request.url))
            return httpx.Response(200, text="OK")

        async with adapter:
            transport = httpx.MockTransport(capture_request)
            adapter._client = httpx.AsyncClient(transport=transport)

            await adapter.health_check()

            assert len(requests_made) == 1
            assert "health" in requests_made[0]

    @pytest.mark.anyio
    async def test_cleanup(
        self,
        http_config: HTTPAdapterConfig,
    ) -> None:
        """Test cleanup closes client."""
        adapter = HTTPAdapter(http_config)

        # Create client
        await adapter._get_client()
        assert adapter._client is not None
        assert not adapter._client.is_closed

        # Cleanup
        await adapter.cleanup()
        assert adapter._client is None

    @pytest.mark.anyio
    async def test_context_manager(
        self,
        http_config: HTTPAdapterConfig,
    ) -> None:
        """Test adapter as async context manager."""
        adapter = HTTPAdapter(http_config)

        async with adapter as ctx_adapter:
            assert ctx_adapter is adapter
            await adapter._get_client()
            assert adapter._client is not None

        # Client should be closed after exiting context
        assert adapter._client is None

    @pytest.mark.anyio
    async def test_stream_events_with_response(
        self,
        http_config: HTTPAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test streaming events ending with response."""
        adapter = HTTPAdapter(http_config)

        # SSE format response
        sse_content = (
            "event: progress\n"
            'data: {"event_type": "progress", "payload": {"message": "Working"}}\n'
            "\n"
            "event: response\n"
            'data: {"version": "1.0", "task_id": "test-task-123", '
            '"status": "completed", "artifacts": []}\n'
            "\n"
        )

        async def stream_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_content.encode(),
                headers={"Content-Type": "text/event-stream"},
            )

        async with adapter:
            transport = httpx.MockTransport(stream_handler)
            adapter._client = httpx.AsyncClient(transport=transport)

            events: list[ATPEvent | ATPResponse] = []
            async for item in adapter.stream_events(sample_request):
                events.append(item)

            # Should have at least one event and a response
            assert len(events) >= 1
            # Last item should be response
            assert isinstance(events[-1], ATPResponse)

    @pytest.mark.anyio
    async def test_stream_events_http_error(
        self,
        http_config: HTTPAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test streaming with HTTP error."""
        adapter = HTTPAdapter(http_config)

        async def error_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Server Error")

        async with adapter:
            transport = httpx.MockTransport(error_handler)
            adapter._client = httpx.AsyncClient(transport=transport)

            with pytest.raises(AdapterResponseError):
                async for _ in adapter.stream_events(sample_request):
                    pass
