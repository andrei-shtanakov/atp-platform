"""Integration tests for HTTPAdapter with real HTTP server."""

import asyncio
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from atp.adapters import (
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


class MockAgentHandler(BaseHTTPRequestHandler):
    """Mock HTTP handler for testing."""

    def log_message(self, format, *args):
        """Suppress logging."""
        pass

    def do_GET(self):
        """Handle GET requests (health check)."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests (agent execution)."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if self.path == "/agent":
            try:
                request_data = json.loads(body)
                task_id = request_data.get("task_id", "unknown")

                response = {
                    "version": "1.0",
                    "task_id": task_id,
                    "status": "completed",
                    "artifacts": [
                        {
                            "type": "structured",
                            "name": "result",
                            "data": {"message": "Task completed successfully"},
                        }
                    ],
                    "metrics": {
                        "total_tokens": 150,
                        "total_steps": 3,
                    },
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid JSON"}')

        elif self.path == "/agent/error":
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Internal server error"}')

        elif self.path == "/agent/invalid":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"not valid json")

        elif self.path == "/agent/slow":
            # Simulate slow response
            import time

            time.sleep(2)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "completed"}')

        elif self.path == "/agent/stream":
            # SSE response
            accept = self.headers.get("Accept", "")
            if "text/event-stream" in accept:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                try:
                    request_data = json.loads(body)
                    task_id = request_data.get("task_id", "unknown")
                except json.JSONDecodeError:
                    task_id = "unknown"

                # Send events
                events = [
                    {"event_type": "progress", "payload": {"message": "Starting"}},
                    {"event_type": "progress", "payload": {"message": "Processing"}},
                ]
                for event in events:
                    event["task_id"] = task_id
                    self.wfile.write(b"event: progress\n")
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()

                # Send final response
                response = {
                    "version": "1.0",
                    "task_id": task_id,
                    "status": "completed",
                    "artifacts": [],
                }
                self.wfile.write(b"event: response\n")
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                self.wfile.flush()
            else:
                self.send_response(406)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture(scope="module")
def mock_server():
    """Start a mock HTTP server for testing."""
    server = HTTPServer(("127.0.0.1", 0), MockAgentHandler)
    port = server.server_address[1]

    thread = Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()


@pytest.fixture
def sample_request() -> ATPRequest:
    """Create a sample ATP request."""
    return ATPRequest(
        task_id="integration-test-123",
        task=Task(description="Integration test task"),
        constraints={"max_steps": 10},
    )


class TestHTTPAdapterIntegration:
    """Integration tests for HTTPAdapter."""

    @pytest.mark.anyio
    async def test_execute_real_request(
        self, mock_server: str, sample_request: ATPRequest
    ) -> None:
        """Test executing a real HTTP request."""
        config = HTTPAdapterConfig(
            endpoint=f"{mock_server}/agent",
            timeout_seconds=10.0,
        )

        async with HTTPAdapter(config) as adapter:
            response = await adapter.execute(sample_request)

        assert isinstance(response, ATPResponse)
        assert response.task_id == "integration-test-123"
        assert response.status == ResponseStatus.COMPLETED
        assert len(response.artifacts) == 1
        assert response.metrics is not None
        assert response.metrics.total_tokens == 150

    @pytest.mark.anyio
    async def test_health_check_real_server(self, mock_server: str) -> None:
        """Test health check against real server."""
        config = HTTPAdapterConfig(
            endpoint=f"{mock_server}/agent",
            health_endpoint=f"{mock_server}/health",
            timeout_seconds=10.0,
        )

        async with HTTPAdapter(config) as adapter:
            result = await adapter.health_check()

        assert result is True

    @pytest.mark.anyio
    async def test_health_check_unavailable_server(self) -> None:
        """Test health check against unavailable server."""
        config = HTTPAdapterConfig(
            endpoint="http://127.0.0.1:59999/agent",  # Non-existent port
            timeout_seconds=2.0,
        )

        async with HTTPAdapter(config) as adapter:
            result = await adapter.health_check()

        assert result is False

    @pytest.mark.anyio
    async def test_execute_server_error(
        self, mock_server: str, sample_request: ATPRequest
    ) -> None:
        """Test handling server error response."""
        config = HTTPAdapterConfig(
            endpoint=f"{mock_server}/agent/error",
            timeout_seconds=10.0,
        )

        async with HTTPAdapter(config) as adapter:
            with pytest.raises(AdapterResponseError) as exc_info:
                await adapter.execute(sample_request)

        assert exc_info.value.status_code == 500

    @pytest.mark.anyio
    async def test_execute_invalid_json_response(
        self, mock_server: str, sample_request: ATPRequest
    ) -> None:
        """Test handling invalid JSON response."""
        config = HTTPAdapterConfig(
            endpoint=f"{mock_server}/agent/invalid",
            timeout_seconds=10.0,
        )

        async with HTTPAdapter(config) as adapter:
            with pytest.raises(AdapterResponseError) as exc_info:
                await adapter.execute(sample_request)

        assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_timeout(
        self, mock_server: str, sample_request: ATPRequest
    ) -> None:
        """Test handling request timeout."""
        config = HTTPAdapterConfig(
            endpoint=f"{mock_server}/agent/slow",
            timeout_seconds=0.5,  # Short timeout
        )

        async with HTTPAdapter(config) as adapter:
            with pytest.raises(AdapterTimeoutError) as exc_info:
                await adapter.execute(sample_request)

        assert exc_info.value.timeout_seconds == 0.5

    @pytest.mark.anyio
    async def test_execute_with_custom_headers(
        self, mock_server: str, sample_request: ATPRequest
    ) -> None:
        """Test executing with custom headers."""
        config = HTTPAdapterConfig(
            endpoint=f"{mock_server}/agent",
            timeout_seconds=10.0,
            headers={
                "Authorization": "Bearer test-token",
                "X-Custom-Header": "custom-value",
            },
        )

        async with HTTPAdapter(config) as adapter:
            response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_stream_events_real_request(
        self, mock_server: str, sample_request: ATPRequest
    ) -> None:
        """Test streaming events from real server."""
        config = HTTPAdapterConfig(
            endpoint=f"{mock_server}/agent/stream",
            timeout_seconds=10.0,
        )

        async with HTTPAdapter(config) as adapter:
            events: list[ATPEvent | ATPResponse] = []
            async for item in adapter.stream_events(sample_request):
                events.append(item)

        # Should have events and final response
        assert len(events) >= 2
        # Last item should be response
        assert isinstance(events[-1], ATPResponse)
        assert events[-1].status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_multiple_sequential_requests(self, mock_server: str) -> None:
        """Test multiple sequential requests."""
        config = HTTPAdapterConfig(
            endpoint=f"{mock_server}/agent",
            timeout_seconds=10.0,
        )

        async with HTTPAdapter(config) as adapter:
            for i in range(3):
                request = ATPRequest(
                    task_id=f"sequential-test-{i}",
                    task=Task(description=f"Test task {i}"),
                )
                response = await adapter.execute(request)
                assert response.task_id == f"sequential-test-{i}"
                assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_concurrent_requests(self, mock_server: str) -> None:
        """Test multiple concurrent requests."""
        config = HTTPAdapterConfig(
            endpoint=f"{mock_server}/agent",
            timeout_seconds=10.0,
        )

        async def make_request(adapter: HTTPAdapter, task_id: str) -> ATPResponse:
            request = ATPRequest(
                task_id=task_id,
                task=Task(description=f"Concurrent task {task_id}"),
            )
            return await adapter.execute(request)

        async with HTTPAdapter(config) as adapter:
            tasks = [make_request(adapter, f"concurrent-{i}") for i in range(5)]
            responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response.task_id == f"concurrent-{i}"
            assert response.status == ResponseStatus.COMPLETED
