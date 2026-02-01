"""Unit tests for MCP transport layer."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from atp.adapters.exceptions import (
    AdapterConnectionError,
    AdapterTimeoutError,
)
from atp.adapters.mcp.transport import (
    JSONRPCMessage,
    SSETransport,
    SSETransportConfig,
    StdioTransport,
    StdioTransportConfig,
    TransportConfig,
    TransportState,
    create_jsonrpc_request,
    create_jsonrpc_response,
    parse_jsonrpc_message,
)


class TestTransportConfig:
    """Tests for TransportConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TransportConfig()
        assert config.connection_timeout == 30.0
        assert config.read_timeout == 60.0
        assert config.write_timeout == 30.0
        assert config.reconnect_attempts == 3
        assert config.reconnect_delay == 1.0
        assert config.reconnect_backoff == 2.0
        assert config.max_reconnect_delay == 30.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = TransportConfig(
            connection_timeout=60.0,
            read_timeout=120.0,
            write_timeout=60.0,
            reconnect_attempts=5,
            reconnect_delay=2.0,
            reconnect_backoff=3.0,
            max_reconnect_delay=60.0,
        )
        assert config.connection_timeout == 60.0
        assert config.read_timeout == 120.0
        assert config.write_timeout == 60.0
        assert config.reconnect_attempts == 5
        assert config.reconnect_delay == 2.0
        assert config.reconnect_backoff == 3.0
        assert config.max_reconnect_delay == 60.0

    def test_invalid_timeout(self) -> None:
        """Test that invalid timeout raises validation error."""
        with pytest.raises(ValueError):
            TransportConfig(connection_timeout=-1.0)

    def test_invalid_reconnect_attempts(self) -> None:
        """Test that negative reconnect attempts raises validation error."""
        with pytest.raises(ValueError):
            TransportConfig(reconnect_attempts=-1)


class TestJSONRPCHelpers:
    """Tests for JSON-RPC helper functions."""

    def test_create_jsonrpc_request_minimal(self) -> None:
        """Test creating minimal JSON-RPC request."""
        request = create_jsonrpc_request("test_method")
        assert request["jsonrpc"] == "2.0"
        assert request["method"] == "test_method"
        assert "params" not in request
        assert "id" not in request

    def test_create_jsonrpc_request_with_params(self) -> None:
        """Test creating JSON-RPC request with parameters."""
        params = {"key": "value"}
        request = create_jsonrpc_request("test_method", params, request_id=1)
        assert request["jsonrpc"] == "2.0"
        assert request["method"] == "test_method"
        assert request["params"] == params
        assert request["id"] == 1

    def test_create_jsonrpc_request_notification(self) -> None:
        """Test creating JSON-RPC notification (no id)."""
        request = create_jsonrpc_request("notify", {"data": "test"})
        assert request["jsonrpc"] == "2.0"
        assert request["method"] == "notify"
        assert request["params"] == {"data": "test"}
        assert "id" not in request

    def test_create_jsonrpc_response_success(self) -> None:
        """Test creating successful JSON-RPC response."""
        response = create_jsonrpc_response(1, result={"status": "ok"})
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["result"] == {"status": "ok"}
        assert "error" not in response

    def test_create_jsonrpc_response_error(self) -> None:
        """Test creating error JSON-RPC response."""
        error = {"code": -32600, "message": "Invalid Request"}
        response = create_jsonrpc_response(1, error=error)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["error"] == error
        assert "result" not in response

    def test_parse_jsonrpc_message_request(self) -> None:
        """Test parsing JSON-RPC request."""
        data = '{"jsonrpc": "2.0", "method": "test", "params": {}, "id": 1}'
        message = parse_jsonrpc_message(data)
        assert message.jsonrpc == "2.0"
        assert message.method == "test"
        assert message.id == 1
        assert message.is_request()
        assert not message.is_notification()
        assert not message.is_response()

    def test_parse_jsonrpc_message_notification(self) -> None:
        """Test parsing JSON-RPC notification."""
        data = '{"jsonrpc": "2.0", "method": "notify", "params": {}}'
        message = parse_jsonrpc_message(data)
        assert message.method == "notify"
        assert message.id is None
        assert message.is_notification()
        assert not message.is_request()

    def test_parse_jsonrpc_message_response(self) -> None:
        """Test parsing JSON-RPC response."""
        data = '{"jsonrpc": "2.0", "result": {"data": "test"}, "id": 1}'
        message = parse_jsonrpc_message(data)
        assert message.result == {"data": "test"}
        assert message.id == 1
        assert message.is_response()
        assert not message.is_request()

    def test_parse_jsonrpc_message_error(self) -> None:
        """Test parsing JSON-RPC error response."""
        data = (
            '{"jsonrpc": "2.0", "error": '
            '{"code": -32600, "message": "Invalid"}, "id": 1}'
        )
        message = parse_jsonrpc_message(data)
        assert message.error is not None
        assert message.error["code"] == -32600
        assert message.is_error()

    def test_parse_jsonrpc_message_from_bytes(self) -> None:
        """Test parsing JSON-RPC message from bytes."""
        data = b'{"jsonrpc": "2.0", "method": "test", "id": 1}'
        message = parse_jsonrpc_message(data)
        assert message.method == "test"

    def test_parse_jsonrpc_message_invalid_json(self) -> None:
        """Test parsing invalid JSON raises error."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_jsonrpc_message("not valid json")

    def test_parse_jsonrpc_message_not_object(self) -> None:
        """Test parsing non-object JSON raises error."""
        with pytest.raises(ValueError, match="must be an object"):
            parse_jsonrpc_message("[1, 2, 3]")

    def test_parse_jsonrpc_message_wrong_version(self) -> None:
        """Test parsing wrong JSON-RPC version raises error."""
        with pytest.raises(ValueError, match="Invalid JSON-RPC version"):
            parse_jsonrpc_message('{"jsonrpc": "1.0", "method": "test"}')


class TestJSONRPCMessage:
    """Tests for JSONRPCMessage model."""

    def test_is_request(self) -> None:
        """Test is_request method."""
        msg = JSONRPCMessage(method="test", id=1)
        assert msg.is_request() is True

    def test_is_notification(self) -> None:
        """Test is_notification method."""
        msg = JSONRPCMessage(method="notify")
        assert msg.is_notification() is True

    def test_is_response(self) -> None:
        """Test is_response method."""
        msg = JSONRPCMessage(result="data", id=1)
        assert msg.is_response() is True

    def test_is_error(self) -> None:
        """Test is_error method."""
        msg = JSONRPCMessage(error={"code": -32600}, id=1)
        assert msg.is_error() is True


class TestStdioTransportConfig:
    """Tests for StdioTransportConfig."""

    def test_minimal_config(self) -> None:
        """Test creating config with minimal required fields."""
        config = StdioTransportConfig(command="python")
        assert config.command == "python"
        assert config.args == []
        assert config.working_dir is None
        assert config.environment == {}
        assert config.inherit_environment is True

    def test_full_config(self) -> None:
        """Test creating config with all fields."""
        config = StdioTransportConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            working_dir="/workspace",
            environment={"MCP_ROOT": "/workspace"},
            inherit_environment=False,
            allowed_env_vars=["CUSTOM_VAR"],
            connection_timeout=60.0,
        )
        assert config.command == "npx"
        assert config.args == ["-y", "@modelcontextprotocol/server-filesystem"]
        assert config.working_dir == "/workspace"
        assert config.environment == {"MCP_ROOT": "/workspace"}
        assert config.inherit_environment is False
        assert config.allowed_env_vars == ["CUSTOM_VAR"]
        assert config.connection_timeout == 60.0

    def test_empty_command_validation(self) -> None:
        """Test that empty command raises validation error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            StdioTransportConfig(command="")

    def test_whitespace_command_validation(self) -> None:
        """Test that whitespace-only command raises validation error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            StdioTransportConfig(command="   ")


class TestStdioTransport:
    """Tests for StdioTransport."""

    @pytest.fixture
    def stdio_config(self) -> StdioTransportConfig:
        """Create stdio transport config."""
        return StdioTransportConfig(
            command="python",
            args=["-m", "mcp_server"],
            connection_timeout=5.0,
            read_timeout=10.0,
            write_timeout=5.0,
        )

    def test_initial_state(self, stdio_config: StdioTransportConfig) -> None:
        """Test initial transport state."""
        transport = StdioTransport(stdio_config)
        assert transport.state == TransportState.DISCONNECTED
        assert transport.is_connected is False

    @pytest.mark.anyio
    async def test_connect_success(self, stdio_config: StdioTransportConfig) -> None:
        """Test successful connection."""
        transport = StdioTransport(stdio_config)

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            await transport.connect()

        assert transport.state == TransportState.CONNECTED
        assert transport.is_connected is True
        assert transport._process is mock_process

    @pytest.mark.anyio
    async def test_connect_command_not_found(
        self, stdio_config: StdioTransportConfig
    ) -> None:
        """Test connection failure when command not found."""
        transport = StdioTransport(stdio_config)

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("Command not found"),
        ):
            with pytest.raises(AdapterConnectionError) as exc_info:
                await transport.connect()

        assert "not found" in str(exc_info.value)
        assert transport.state == TransportState.DISCONNECTED

    @pytest.mark.anyio
    async def test_connect_timeout(self, stdio_config: StdioTransportConfig) -> None:
        """Test connection timeout."""
        transport = StdioTransport(stdio_config)

        async def slow_connect(*args, **kwargs):
            await asyncio.sleep(100)

        with patch("asyncio.create_subprocess_exec", side_effect=slow_connect):
            with patch("asyncio.wait_for", side_effect=TimeoutError()):
                with pytest.raises(AdapterTimeoutError) as exc_info:
                    await transport.connect()

        assert exc_info.value.timeout_seconds == 5.0
        assert transport.state == TransportState.DISCONNECTED

    @pytest.mark.anyio
    async def test_send_message(self, stdio_config: StdioTransportConfig) -> None:
        """Test sending a message."""
        transport = StdioTransport(stdio_config)

        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin

        transport._process = mock_process
        transport._state = TransportState.CONNECTED

        message = {"jsonrpc": "2.0", "method": "test", "id": 1}
        await transport.send(message)

        mock_stdin.write.assert_called_once()
        written_data = mock_stdin.write.call_args[0][0]

        # Verify Content-Length header framing
        assert b"Content-Length:" in written_data
        assert b'{"jsonrpc": "2.0"' in written_data

    @pytest.mark.anyio
    async def test_send_not_connected(self, stdio_config: StdioTransportConfig) -> None:
        """Test sending when not connected."""
        transport = StdioTransport(stdio_config)

        with pytest.raises(AdapterConnectionError, match="not connected"):
            await transport.send({"test": "message"})

    @pytest.mark.anyio
    async def test_send_timeout(self, stdio_config: StdioTransportConfig) -> None:
        """Test send timeout."""
        transport = StdioTransport(stdio_config)

        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()

        async def slow_drain():
            await asyncio.sleep(100)

        mock_stdin.drain = slow_drain

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin

        transport._process = mock_process
        transport._state = TransportState.CONNECTED

        with patch("asyncio.wait_for", side_effect=TimeoutError()):
            with pytest.raises(AdapterTimeoutError):
                await transport.send({"test": "message"})

    @pytest.mark.anyio
    async def test_send_broken_pipe(self, stdio_config: StdioTransportConfig) -> None:
        """Test send with broken pipe."""
        transport = StdioTransport(stdio_config)

        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock(side_effect=BrokenPipeError())

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin

        transport._process = mock_process
        transport._state = TransportState.CONNECTED

        with pytest.raises(AdapterConnectionError, match="Connection lost"):
            await transport.send({"test": "message"})

        assert transport.state == TransportState.DISCONNECTED

    @pytest.mark.anyio
    async def test_receive_message(self, stdio_config: StdioTransportConfig) -> None:
        """Test receiving a message."""
        transport = StdioTransport(stdio_config)

        message_content = '{"jsonrpc": "2.0", "result": "success", "id": 1}'
        content_bytes = message_content.encode("utf-8")
        header = f"Content-Length: {len(content_bytes)}\r\n\r\n"
        full_response = header.encode("utf-8") + content_bytes

        read_position = 0

        async def mock_readline():
            nonlocal read_position
            # Find the next line ending
            remaining = full_response[read_position:]
            if b"\r\n" in remaining:
                end = remaining.index(b"\r\n") + 2
            elif b"\n" in remaining:
                end = remaining.index(b"\n") + 1
            else:
                end = len(remaining)
            line = remaining[:end]
            read_position += end
            return line

        async def mock_read(n):
            nonlocal read_position
            data = full_response[read_position : read_position + n]
            read_position += n
            return data

        mock_stdout = MagicMock()
        mock_stdout.readline = mock_readline
        mock_stdout.read = mock_read

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout

        transport._process = mock_process
        transport._state = TransportState.CONNECTED

        result = await transport.receive()

        assert result["jsonrpc"] == "2.0"
        assert result["result"] == "success"
        assert result["id"] == 1

    @pytest.mark.anyio
    async def test_receive_not_connected(
        self, stdio_config: StdioTransportConfig
    ) -> None:
        """Test receiving when not connected."""
        transport = StdioTransport(stdio_config)

        with pytest.raises(AdapterConnectionError, match="not connected"):
            await transport.receive()

    @pytest.mark.anyio
    async def test_receive_missing_content_length(
        self, stdio_config: StdioTransportConfig
    ) -> None:
        """Test receiving with missing Content-Length header."""
        transport = StdioTransport(stdio_config)

        lines = [b"Some-Header: value\r\n", b"\r\n"]
        line_index = 0

        async def mock_readline():
            nonlocal line_index
            if line_index < len(lines):
                line = lines[line_index]
                line_index += 1
                return line
            return b""

        mock_stdout = MagicMock()
        mock_stdout.readline = mock_readline

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout

        transport._process = mock_process
        transport._state = TransportState.CONNECTED

        with pytest.raises(AdapterConnectionError, match="Missing Content-Length"):
            await transport.receive()

    @pytest.mark.anyio
    async def test_close(self, stdio_config: StdioTransportConfig) -> None:
        """Test closing the transport."""
        transport = StdioTransport(stdio_config)

        mock_process = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()

        transport._process = mock_process
        transport._state = TransportState.CONNECTED

        await transport.close()

        assert transport.state == TransportState.CLOSED
        assert transport._process is None
        mock_process.terminate.assert_called_once()

    @pytest.mark.anyio
    async def test_close_timeout_kills_process(
        self, stdio_config: StdioTransportConfig
    ) -> None:
        """Test that close kills process if terminate times out."""
        transport = StdioTransport(stdio_config)

        async def slow_wait():
            await asyncio.sleep(100)

        mock_process = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock(side_effect=[TimeoutError(), None])

        transport._process = mock_process
        transport._state = TransportState.CONNECTED

        await transport.close()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    @pytest.mark.anyio
    async def test_context_manager(self, stdio_config: StdioTransportConfig) -> None:
        """Test transport as async context manager."""
        transport = StdioTransport(stdio_config)

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            async with transport as t:
                assert t is transport
                assert transport.state == TransportState.CONNECTED

        assert transport.state == TransportState.CLOSED

    @pytest.mark.anyio
    async def test_health_check_running(
        self, stdio_config: StdioTransportConfig
    ) -> None:
        """Test health check when process is running."""
        transport = StdioTransport(stdio_config)

        mock_process = MagicMock()
        mock_process.returncode = None  # Process still running

        transport._process = mock_process

        result = await transport.health_check()
        assert result is True

    @pytest.mark.anyio
    async def test_health_check_not_running(
        self, stdio_config: StdioTransportConfig
    ) -> None:
        """Test health check when process is not running."""
        transport = StdioTransport(stdio_config)

        mock_process = MagicMock()
        mock_process.returncode = 1  # Process exited

        transport._process = mock_process

        result = await transport.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_health_check_no_process(
        self, stdio_config: StdioTransportConfig
    ) -> None:
        """Test health check with no process."""
        transport = StdioTransport(stdio_config)

        result = await transport.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_reconnect_success(self, stdio_config: StdioTransportConfig) -> None:
        """Test successful reconnection."""
        transport = StdioTransport(stdio_config)

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()

        transport._process = mock_process
        transport._state = TransportState.CONNECTED

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await transport.reconnect()

        assert result is True
        assert transport.state == TransportState.CONNECTED

    @pytest.mark.anyio
    async def test_reconnect_failure(self, stdio_config: StdioTransportConfig) -> None:
        """Test failed reconnection after max attempts."""
        config = StdioTransportConfig(
            command="python",
            reconnect_attempts=2,
            reconnect_delay=0.01,  # Short delay for testing
        )
        transport = StdioTransport(config)
        transport._state = TransportState.CONNECTED

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("Not found"),
        ):
            result = await transport.reconnect()

        assert result is False
        assert transport.state == TransportState.DISCONNECTED

    @pytest.mark.anyio
    async def test_reconnect_when_closed(
        self, stdio_config: StdioTransportConfig
    ) -> None:
        """Test reconnect returns False when transport is closed."""
        transport = StdioTransport(stdio_config)
        transport._state = TransportState.CLOSED

        result = await transport.reconnect()

        assert result is False

    def test_message_id_generation(self, stdio_config: StdioTransportConfig) -> None:
        """Test message ID counter increments."""
        transport = StdioTransport(stdio_config)

        id1 = transport._next_message_id()
        id2 = transport._next_message_id()
        id3 = transport._next_message_id()

        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

    @pytest.mark.anyio
    async def test_send_request_and_wait(
        self, stdio_config: StdioTransportConfig
    ) -> None:
        """Test send_request waits for matching response."""
        transport = StdioTransport(stdio_config)

        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        message_content = '{"jsonrpc": "2.0", "result": "success", "id": 1}'
        content_bytes = message_content.encode("utf-8")
        header = f"Content-Length: {len(content_bytes)}\r\n\r\n"
        full_response = header.encode("utf-8") + content_bytes

        read_position = 0

        async def mock_readline():
            nonlocal read_position
            remaining = full_response[read_position:]
            if b"\r\n" in remaining:
                end = remaining.index(b"\r\n") + 2
            else:
                end = len(remaining) if remaining else 0
            line = remaining[:end] if end else b""
            read_position += end if end else 1
            return line

        async def mock_read(n):
            nonlocal read_position
            data = full_response[read_position : read_position + n]
            read_position += n
            return data

        mock_stdout = MagicMock()
        mock_stdout.readline = mock_readline
        mock_stdout.read = mock_read

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout

        transport._process = mock_process
        transport._state = TransportState.CONNECTED

        result = await transport.send_request("test_method", {"param": "value"})

        assert result["result"] == "success"

    @pytest.mark.anyio
    async def test_send_request_error_response(
        self, stdio_config: StdioTransportConfig
    ) -> None:
        """Test send_request raises error on JSON-RPC error response."""
        transport = StdioTransport(stdio_config)

        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        error_response = json.dumps(
            {
                "jsonrpc": "2.0",
                "error": {"code": -32600, "message": "Invalid Request"},
                "id": 1,
            }
        )
        content_bytes = error_response.encode("utf-8")
        header = f"Content-Length: {len(content_bytes)}\r\n\r\n"
        full_response = header.encode("utf-8") + content_bytes

        read_position = 0

        async def mock_readline():
            nonlocal read_position
            remaining = full_response[read_position:]
            if b"\r\n" in remaining:
                end = remaining.index(b"\r\n") + 2
            else:
                end = len(remaining) if remaining else 0
            line = remaining[:end] if end else b""
            read_position += end if end else 1
            return line

        async def mock_read(n):
            nonlocal read_position
            data = full_response[read_position : read_position + n]
            read_position += n
            return data

        mock_stdout = MagicMock()
        mock_stdout.readline = mock_readline
        mock_stdout.read = mock_read

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout

        transport._process = mock_process
        transport._state = TransportState.CONNECTED

        with pytest.raises(ValueError, match="JSON-RPC error"):
            await transport.send_request("test_method")


class TestSSETransportConfig:
    """Tests for SSETransportConfig."""

    def test_minimal_config(self) -> None:
        """Test creating config with minimal required fields."""
        config = SSETransportConfig(url="http://localhost:8080/sse")
        assert config.url == "http://localhost:8080/sse"
        assert config.headers == {}
        assert config.verify_ssl is True
        assert config.post_endpoint is None

    def test_full_config(self) -> None:
        """Test creating config with all fields."""
        config = SSETransportConfig(
            url="https://mcp.example.com/sse",
            headers={"Authorization": "Bearer token"},
            verify_ssl=False,
            post_endpoint="https://mcp.example.com/messages",
            connection_timeout=60.0,
        )
        assert config.url == "https://mcp.example.com/sse"
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.verify_ssl is False
        assert config.post_endpoint == "https://mcp.example.com/messages"
        assert config.connection_timeout == 60.0

    def test_empty_url_validation(self) -> None:
        """Test that empty URL raises validation error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SSETransportConfig(url="")

    def test_invalid_url_protocol(self) -> None:
        """Test that non-HTTP URL raises validation error."""
        with pytest.raises(ValueError, match="HTTP/HTTPS"):
            SSETransportConfig(url="ftp://example.com/sse")


class TestSSETransport:
    """Tests for SSETransport."""

    @pytest.fixture
    def sse_config(self) -> SSETransportConfig:
        """Create SSE transport config."""
        return SSETransportConfig(
            url="http://localhost:8080/sse",
            connection_timeout=5.0,
            read_timeout=10.0,
            write_timeout=5.0,
        )

    def test_initial_state(self, sse_config: SSETransportConfig) -> None:
        """Test initial transport state."""
        transport = SSETransport(sse_config)
        assert transport.state == TransportState.DISCONNECTED
        assert transport.is_connected is False

    @pytest.mark.anyio
    async def test_connect_success(self, sse_config: SSETransportConfig) -> None:
        """Test successful SSE connection."""
        transport = SSETransport(sse_config)

        async def mock_stream(*args, **kwargs):
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=b"",
            )

        mock_client = MagicMock()
        mock_client.build_request = MagicMock()

        async def mock_send(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200

            async def mock_aiter_lines():
                return
                yield  # Makes this an async generator

            mock_response.aiter_lines = mock_aiter_lines
            return mock_response

        mock_client.send = mock_send

        with patch("httpx.AsyncClient", return_value=mock_client):
            await transport.connect()

        assert transport.state == TransportState.CONNECTED
        assert transport.is_connected is True

    @pytest.mark.anyio
    async def test_connect_http_error(self, sse_config: SSETransportConfig) -> None:
        """Test connection failure on HTTP error."""
        transport = SSETransport(sse_config)

        mock_client = MagicMock()
        mock_client.build_request = MagicMock()

        async def mock_send(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.aread = AsyncMock()
            return mock_response

        mock_client.send = mock_send

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(AdapterConnectionError, match="500"):
                await transport.connect()

        assert transport.state == TransportState.DISCONNECTED

    @pytest.mark.anyio
    async def test_connect_timeout(self, sse_config: SSETransportConfig) -> None:
        """Test connection timeout."""
        transport = SSETransport(sse_config)

        mock_client = MagicMock()
        mock_client.build_request = MagicMock()
        mock_client.send = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(AdapterTimeoutError):
                await transport.connect()

        assert transport.state == TransportState.DISCONNECTED

    @pytest.mark.anyio
    async def test_connect_error(self, sse_config: SSETransportConfig) -> None:
        """Test connection error."""
        transport = SSETransport(sse_config)

        mock_client = MagicMock()
        mock_client.build_request = MagicMock()
        mock_client.send = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(AdapterConnectionError, match="Failed to connect"):
                await transport.connect()

        assert transport.state == TransportState.DISCONNECTED

    @pytest.mark.anyio
    async def test_send_message(self, sse_config: SSETransportConfig) -> None:
        """Test sending a message via HTTP POST."""
        transport = SSETransport(sse_config)

        transport._client = MagicMock()
        transport._client.post = AsyncMock(return_value=MagicMock(status_code=200))
        transport._state = TransportState.CONNECTED

        message = {"jsonrpc": "2.0", "method": "test", "id": 1}
        await transport.send(message)

        transport._client.post.assert_called_once()
        call_kwargs = transport._client.post.call_args[1]
        assert call_kwargs["json"] == message

    @pytest.mark.anyio
    async def test_send_not_connected(self, sse_config: SSETransportConfig) -> None:
        """Test sending when not connected."""
        transport = SSETransport(sse_config)

        with pytest.raises(AdapterConnectionError, match="not connected"):
            await transport.send({"test": "message"})

    @pytest.mark.anyio
    async def test_send_http_error(self, sse_config: SSETransportConfig) -> None:
        """Test send with HTTP error response."""
        transport = SSETransport(sse_config)

        transport._client = MagicMock()
        transport._client.post = AsyncMock(return_value=MagicMock(status_code=500))
        transport._state = TransportState.CONNECTED

        with pytest.raises(AdapterConnectionError, match="500"):
            await transport.send({"test": "message"})

    @pytest.mark.anyio
    async def test_send_timeout(self, sse_config: SSETransportConfig) -> None:
        """Test send timeout."""
        transport = SSETransport(sse_config)

        transport._client = MagicMock()
        transport._client.post = AsyncMock(
            side_effect=httpx.TimeoutException("timeout")
        )
        transport._state = TransportState.CONNECTED

        with pytest.raises(AdapterTimeoutError):
            await transport.send({"test": "message"})

    @pytest.mark.anyio
    async def test_send_with_custom_post_endpoint(
        self, sse_config: SSETransportConfig
    ) -> None:
        """Test sending to custom POST endpoint."""
        config = SSETransportConfig(
            url="http://localhost:8080/sse",
            post_endpoint="http://localhost:8080/messages",
        )
        transport = SSETransport(config)

        transport._client = MagicMock()
        transport._client.post = AsyncMock(return_value=MagicMock(status_code=200))
        transport._state = TransportState.CONNECTED

        await transport.send({"test": "message"})

        transport._client.post.assert_called_once()
        call_args = transport._client.post.call_args[0]
        assert call_args[0] == "http://localhost:8080/messages"

    @pytest.mark.anyio
    async def test_send_with_session_id(self, sse_config: SSETransportConfig) -> None:
        """Test sending includes session ID header when set."""
        transport = SSETransport(sse_config)

        transport._client = MagicMock()
        transport._client.post = AsyncMock(return_value=MagicMock(status_code=200))
        transport._state = TransportState.CONNECTED
        transport._session_id = "test-session-123"

        await transport.send({"test": "message"})

        call_kwargs = transport._client.post.call_args[1]
        assert call_kwargs["headers"]["X-Session-Id"] == "test-session-123"

    @pytest.mark.anyio
    async def test_receive_from_queue(self, sse_config: SSETransportConfig) -> None:
        """Test receiving a message from the queue."""
        transport = SSETransport(sse_config)
        transport._state = TransportState.CONNECTED

        # Pre-populate the queue
        await transport._message_queue.put(
            {"jsonrpc": "2.0", "result": "test", "id": 1}
        )

        result = await transport.receive()

        assert result["result"] == "test"

    @pytest.mark.anyio
    async def test_receive_not_connected(self, sse_config: SSETransportConfig) -> None:
        """Test receiving when not connected."""
        transport = SSETransport(sse_config)

        with pytest.raises(AdapterConnectionError, match="not connected"):
            await transport.receive()

    @pytest.mark.anyio
    async def test_receive_timeout(self, sse_config: SSETransportConfig) -> None:
        """Test receive timeout when queue is empty."""
        config = SSETransportConfig(
            url="http://localhost:8080/sse",
            read_timeout=0.01,  # Very short timeout for testing
        )
        transport = SSETransport(config)
        transport._state = TransportState.CONNECTED

        with pytest.raises(AdapterTimeoutError):
            await transport.receive()

    @pytest.mark.anyio
    async def test_close(self, sse_config: SSETransportConfig) -> None:
        """Test closing the SSE transport."""
        transport = SSETransport(sse_config)

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        mock_response = MagicMock()
        mock_response.aclose = AsyncMock()

        transport._client = mock_client
        transport._sse_response = mock_response
        transport._state = TransportState.CONNECTED

        await transport.close()

        assert transport.state == TransportState.CLOSED
        assert transport._client is None
        assert transport._sse_response is None
        mock_client.aclose.assert_called_once()
        mock_response.aclose.assert_called_once()

    @pytest.mark.anyio
    async def test_close_cancels_reader_task(
        self, sse_config: SSETransportConfig
    ) -> None:
        """Test that close cancels the reader task."""
        transport = SSETransport(sse_config)

        async def long_running():
            await asyncio.sleep(100)

        transport._reader_task = asyncio.create_task(long_running())
        transport._state = TransportState.CONNECTED

        await transport.close()

        assert transport._reader_task is None

    @pytest.mark.anyio
    async def test_health_check_connected(self, sse_config: SSETransportConfig) -> None:
        """Test health check when connected."""
        transport = SSETransport(sse_config)
        transport._client = MagicMock()
        transport._state = TransportState.CONNECTED

        result = await transport.health_check()
        assert result is True

    @pytest.mark.anyio
    async def test_health_check_disconnected(
        self, sse_config: SSETransportConfig
    ) -> None:
        """Test health check when disconnected."""
        transport = SSETransport(sse_config)

        result = await transport.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_context_manager(self, sse_config: SSETransportConfig) -> None:
        """Test SSE transport as async context manager."""
        transport = SSETransport(sse_config)

        mock_client = MagicMock()
        mock_client.build_request = MagicMock()
        mock_client.aclose = AsyncMock()

        async def mock_send(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.aclose = AsyncMock()

            async def mock_aiter_lines():
                return
                yield

            mock_response.aiter_lines = mock_aiter_lines
            return mock_response

        mock_client.send = mock_send

        with patch("httpx.AsyncClient", return_value=mock_client):
            async with transport as t:
                assert t is transport
                assert transport.state == TransportState.CONNECTED

        assert transport.state == TransportState.CLOSED
