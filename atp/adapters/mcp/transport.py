"""MCP transport layer for stdio and SSE communication."""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from atp.adapters.exceptions import (
    AdapterConnectionError,
    AdapterTimeoutError,
)
from atp.core.security import filter_environment_variables


class TransportState(Enum):
    """Transport connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


class TransportConfig(BaseModel):
    """Base configuration for MCP transports."""

    connection_timeout: float = Field(
        default=30.0, description="Timeout for initial connection in seconds", gt=0
    )
    read_timeout: float = Field(
        default=60.0, description="Timeout for read operations in seconds", gt=0
    )
    write_timeout: float = Field(
        default=30.0, description="Timeout for write operations in seconds", gt=0
    )
    reconnect_attempts: int = Field(
        default=3, description="Number of reconnection attempts", ge=0
    )
    reconnect_delay: float = Field(
        default=1.0, description="Initial delay between reconnection attempts", ge=0
    )
    reconnect_backoff: float = Field(
        default=2.0, description="Backoff multiplier for reconnection delay", ge=1.0
    )
    max_reconnect_delay: float = Field(
        default=30.0, description="Maximum delay between reconnection attempts", gt=0
    )


class JSONRPCMessage(BaseModel):
    """JSON-RPC 2.0 message structure."""

    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    method: str | None = Field(default=None, description="Method name (for requests)")
    params: dict[str, Any] | list[Any] | None = Field(
        default=None, description="Method parameters"
    )
    result: Any | None = Field(default=None, description="Result (for responses)")
    error: dict[str, Any] | None = Field(
        default=None, description="Error object (for error responses)"
    )
    id: str | int | None = Field(
        default=None, description="Request ID (null for notifications)"
    )

    def is_request(self) -> bool:
        """Check if this is a request message."""
        return self.method is not None and self.id is not None

    def is_notification(self) -> bool:
        """Check if this is a notification (request without id)."""
        return self.method is not None and self.id is None

    def is_response(self) -> bool:
        """Check if this is a response message."""
        return self.method is None and self.id is not None

    def is_error(self) -> bool:
        """Check if this is an error response."""
        return self.error is not None


class JSONRPCError:
    """Standard JSON-RPC error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


def create_jsonrpc_request(
    method: str,
    params: dict[str, Any] | list[Any] | None = None,
    request_id: str | int | None = None,
) -> dict[str, Any]:
    """Create a JSON-RPC request message.

    Args:
        method: Method name to call.
        params: Optional parameters for the method.
        request_id: Optional request ID. If None, creates a notification.

    Returns:
        JSON-RPC request as a dictionary.
    """
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    if request_id is not None:
        message["id"] = request_id
    return message


def create_jsonrpc_response(
    request_id: str | int,
    result: Any = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-RPC response message.

    Args:
        request_id: ID of the original request.
        result: Result value (for success responses).
        error: Error object (for error responses).

    Returns:
        JSON-RPC response as a dictionary.
    """
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        message["error"] = error
    else:
        message["result"] = result
    return message


def parse_jsonrpc_message(data: str | bytes) -> JSONRPCMessage:
    """Parse a JSON-RPC message from string or bytes.

    Args:
        data: JSON string or bytes to parse.

    Returns:
        Parsed JSONRPCMessage.

    Raises:
        ValueError: If the message is invalid JSON or not a valid JSON-RPC message.
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")

    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError("JSON-RPC message must be an object")

    if parsed.get("jsonrpc") != "2.0":
        raise ValueError("Invalid JSON-RPC version (expected '2.0')")

    return JSONRPCMessage.model_validate(parsed)


class MCPTransport(ABC):
    """Abstract base class for MCP transport implementations.

    Transports handle the low-level communication with MCP servers,
    including connection management, message framing, and reconnection.
    """

    def __init__(self, config: TransportConfig | None = None) -> None:
        """Initialize the transport.

        Args:
            config: Transport configuration. Uses defaults if not provided.
        """
        self.config = config or TransportConfig()
        self._state = TransportState.DISCONNECTED
        self._message_id_counter = 0

    @property
    def state(self) -> TransportState:
        """Get the current transport state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if the transport is connected."""
        return self._state == TransportState.CONNECTED

    def _next_message_id(self) -> int:
        """Generate the next message ID."""
        self._message_id_counter += 1
        return self._message_id_counter

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the MCP server.

        Raises:
            AdapterConnectionError: If connection fails.
            AdapterTimeoutError: If connection times out.
        """

    @abstractmethod
    async def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message to the server.

        Args:
            message: JSON-RPC message as a dictionary.

        Raises:
            AdapterConnectionError: If not connected or send fails.
            AdapterTimeoutError: If send times out.
        """

    @abstractmethod
    async def receive(self) -> dict[str, Any]:
        """Receive a JSON-RPC message from the server.

        Returns:
            JSON-RPC message as a dictionary.

        Raises:
            AdapterConnectionError: If not connected or receive fails.
            AdapterTimeoutError: If receive times out.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the transport connection is healthy.

        Returns:
            True if connection is healthy, False otherwise.
        """

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | list[Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for response.

        Args:
            method: Method name to call.
            params: Optional parameters for the method.
            timeout: Optional timeout override for this request.

        Returns:
            JSON-RPC response as a dictionary.

        Raises:
            AdapterConnectionError: If not connected or communication fails.
            AdapterTimeoutError: If operation times out.
            ValueError: If response indicates an error.
        """
        request_id = self._next_message_id()
        request = create_jsonrpc_request(method, params, request_id)

        await self.send(request)

        read_timeout = timeout or self.config.read_timeout
        try:
            response = await asyncio.wait_for(
                self._wait_for_response(request_id),
                timeout=read_timeout,
            )
        except TimeoutError as e:
            raise AdapterTimeoutError(
                f"Request timed out after {read_timeout}s",
                timeout_seconds=read_timeout,
                adapter_type="mcp",
            ) from e

        if "error" in response:
            error = response["error"]
            raise ValueError(
                f"JSON-RPC error {error.get('code')}: {error.get('message')}"
            )

        return response

    async def _wait_for_response(self, request_id: int | str) -> dict[str, Any]:
        """Wait for a response with the given request ID.

        Args:
            request_id: ID of the request to wait for.

        Returns:
            Response message matching the request ID.
        """
        while True:
            response = await self.receive()
            if response.get("id") == request_id:
                return response
            # Could buffer other messages here for out-of-order responses

    async def send_notification(
        self,
        method: str,
        params: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        """Send a JSON-RPC notification (no response expected).

        Args:
            method: Method name to call.
            params: Optional parameters for the method.

        Raises:
            AdapterConnectionError: If not connected or send fails.
            AdapterTimeoutError: If send times out.
        """
        notification = create_jsonrpc_request(method, params, request_id=None)
        await self.send(notification)

    async def reconnect(self) -> bool:
        """Attempt to reconnect using configured retry settings.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        if self._state == TransportState.CLOSED:
            return False

        self._state = TransportState.RECONNECTING
        delay = self.config.reconnect_delay

        for attempt in range(self.config.reconnect_attempts):
            try:
                await self.close()
                await asyncio.sleep(delay)
                await self.connect()
                return True
            except (AdapterConnectionError, AdapterTimeoutError):
                delay = min(
                    delay * self.config.reconnect_backoff,
                    self.config.max_reconnect_delay,
                )
                continue

        self._state = TransportState.DISCONNECTED
        return False

    async def __aenter__(self) -> "MCPTransport":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()


class StdioTransportConfig(TransportConfig):
    """Configuration for stdio transport."""

    command: str = Field(..., description="Command to execute the MCP server")
    args: list[str] = Field(
        default_factory=list, description="Additional command arguments"
    )
    working_dir: str | None = Field(
        default=None, description="Working directory for the subprocess"
    )
    environment: dict[str, str] = Field(
        default_factory=dict, description="Additional environment variables"
    )
    inherit_environment: bool = Field(
        default=True,
        description="Whether to inherit parent environment variables (filtered)",
    )
    allowed_env_vars: list[str] = Field(
        default_factory=list,
        description="Additional environment variables to allow when inheriting",
    )

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        """Validate that command is not empty."""
        if not v or not v.strip():
            raise ValueError("Command cannot be empty")
        return v.strip()


class StdioTransport(MCPTransport):
    """MCP transport over stdio (subprocess).

    Communicates with MCP servers by spawning them as subprocesses
    and using stdin/stdout for JSON-RPC message exchange.
    """

    def __init__(self, config: StdioTransportConfig) -> None:
        """Initialize stdio transport.

        Args:
            config: Stdio transport configuration.
        """
        super().__init__(config)
        self._config: StdioTransportConfig = config
        self._process: asyncio.subprocess.Process | None = None
        self._read_buffer = b""

    def _get_env(self) -> dict[str, str]:
        """Get environment variables for the subprocess.

        Security: Filters sensitive environment variables to prevent
        credential leakage to subprocesses.
        """
        if self._config.inherit_environment:
            env = filter_environment_variables(
                additional_allowlist=set(self._config.allowed_env_vars)
            )
        else:
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": os.environ.get("HOME", "/tmp"),
                "LANG": os.environ.get("LANG", "en_US.UTF-8"),
                "TERM": os.environ.get("TERM", "xterm"),
            }

        env.update(self._config.environment)
        return env

    async def connect(self) -> None:
        """Start the MCP server subprocess and establish connection.

        Raises:
            AdapterConnectionError: If subprocess cannot be started.
            AdapterTimeoutError: If connection times out.
        """
        if self._state == TransportState.CONNECTED:
            return

        self._state = TransportState.CONNECTING

        try:
            cmd = [self._config.command, *self._config.args]
            self._process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self._config.working_dir,
                    env=self._get_env(),
                ),
                timeout=self._config.connection_timeout,
            )
            self._state = TransportState.CONNECTED
        except FileNotFoundError as e:
            self._state = TransportState.DISCONNECTED
            raise AdapterConnectionError(
                f"MCP server command not found: {self._config.command}",
                adapter_type="mcp",
                cause=e,
            ) from e
        except TimeoutError as e:
            self._state = TransportState.DISCONNECTED
            raise AdapterTimeoutError(
                f"Connection timed out after {self._config.connection_timeout}s",
                timeout_seconds=self._config.connection_timeout,
                adapter_type="mcp",
            ) from e
        except OSError as e:
            self._state = TransportState.DISCONNECTED
            raise AdapterConnectionError(
                f"Failed to start MCP server: {e}",
                adapter_type="mcp",
                cause=e,
            ) from e

    async def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message to the subprocess stdin.

        Uses Content-Length header framing for message boundaries.

        Args:
            message: JSON-RPC message to send.

        Raises:
            AdapterConnectionError: If not connected or write fails.
            AdapterTimeoutError: If write times out.
        """
        if not self._process or not self._process.stdin:
            raise AdapterConnectionError(
                "Transport not connected",
                adapter_type="mcp",
            )

        content = json.dumps(message)
        content_bytes = content.encode("utf-8")

        # MCP uses Content-Length header framing (LSP-style)
        header = f"Content-Length: {len(content_bytes)}\r\n\r\n"
        data = header.encode("utf-8") + content_bytes

        try:
            self._process.stdin.write(data)
            await asyncio.wait_for(
                self._process.stdin.drain(),
                timeout=self._config.write_timeout,
            )
        except TimeoutError as e:
            raise AdapterTimeoutError(
                f"Write timed out after {self._config.write_timeout}s",
                timeout_seconds=self._config.write_timeout,
                adapter_type="mcp",
            ) from e
        except (BrokenPipeError, ConnectionResetError) as e:
            self._state = TransportState.DISCONNECTED
            raise AdapterConnectionError(
                "Connection lost while sending message",
                adapter_type="mcp",
                cause=e,
            ) from e

    async def receive(self) -> dict[str, Any]:
        """Receive a JSON-RPC message from the subprocess stdout.

        Uses Content-Length header framing for message boundaries.

        Returns:
            Parsed JSON-RPC message.

        Raises:
            AdapterConnectionError: If not connected or read fails.
            AdapterTimeoutError: If read times out.
        """
        if not self._process or not self._process.stdout:
            raise AdapterConnectionError(
                "Transport not connected",
                adapter_type="mcp",
            )

        try:
            # Read headers until empty line
            content_length = await self._read_headers()

            # Read content
            content = await asyncio.wait_for(
                self._read_exact(content_length),
                timeout=self._config.read_timeout,
            )

            return json.loads(content.decode("utf-8"))

        except TimeoutError as e:
            raise AdapterTimeoutError(
                f"Read timed out after {self._config.read_timeout}s",
                timeout_seconds=self._config.read_timeout,
                adapter_type="mcp",
            ) from e
        except json.JSONDecodeError as e:
            raise AdapterConnectionError(
                f"Invalid JSON in response: {e}",
                adapter_type="mcp",
                cause=e,
            ) from e

    async def _read_headers(self) -> int:
        """Read and parse headers to get content length.

        Returns:
            Content length from headers.

        Raises:
            AdapterConnectionError: If headers are invalid.
        """
        if not self._process or not self._process.stdout:
            raise AdapterConnectionError(
                "Transport not connected",
                adapter_type="mcp",
            )

        headers: dict[str, str] = {}

        while True:
            line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=self._config.read_timeout,
            )

            if not line:
                raise AdapterConnectionError(
                    "Connection closed while reading headers",
                    adapter_type="mcp",
                )

            line_str = line.decode("utf-8").rstrip("\r\n")

            if not line_str:
                # Empty line indicates end of headers
                break

            if ":" in line_str:
                key, value = line_str.split(":", 1)
                headers[key.strip().lower()] = value.strip()

        content_length_str = headers.get("content-length")
        if not content_length_str:
            raise AdapterConnectionError(
                "Missing Content-Length header",
                adapter_type="mcp",
            )

        try:
            return int(content_length_str)
        except ValueError as e:
            raise AdapterConnectionError(
                f"Invalid Content-Length: {content_length_str}",
                adapter_type="mcp",
                cause=e,
            ) from e

    async def _read_exact(self, num_bytes: int) -> bytes:
        """Read exactly num_bytes from stdout.

        Args:
            num_bytes: Number of bytes to read.

        Returns:
            Bytes read from stdout.
        """
        if not self._process or not self._process.stdout:
            raise AdapterConnectionError(
                "Transport not connected",
                adapter_type="mcp",
            )

        data = b""
        remaining = num_bytes

        while remaining > 0:
            chunk = await self._process.stdout.read(remaining)
            if not chunk:
                raise AdapterConnectionError(
                    "Connection closed while reading content",
                    adapter_type="mcp",
                )
            data += chunk
            remaining -= len(chunk)

        return data

    async def close(self) -> None:
        """Terminate the subprocess and close the connection."""
        self._state = TransportState.CLOSED

        if self._process:
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass  # Process already terminated
            finally:
                self._process = None

    async def health_check(self) -> bool:
        """Check if the subprocess is still running.

        Returns:
            True if process is running, False otherwise.
        """
        if not self._process:
            return False
        return self._process.returncode is None


class SSETransportConfig(TransportConfig):
    """Configuration for SSE transport."""

    url: str = Field(..., description="SSE endpoint URL")
    headers: dict[str, str] = Field(
        default_factory=dict, description="Additional HTTP headers"
    )
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    post_endpoint: str | None = Field(
        default=None,
        description="Optional separate endpoint for POST requests (defaults to url)",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate that URL is a valid HTTP/HTTPS URL."""
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        v = v.strip()
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("URL must be an HTTP/HTTPS URL")
        return v


class SSETransport(MCPTransport):
    """MCP transport over Server-Sent Events (HTTP SSE).

    Communicates with MCP servers using SSE for receiving messages
    and HTTP POST for sending messages.
    """

    def __init__(self, config: SSETransportConfig) -> None:
        """Initialize SSE transport.

        Args:
            config: SSE transport configuration.
        """
        super().__init__(config)
        self._config: SSETransportConfig = config
        self._client: httpx.AsyncClient | None = None
        self._sse_response: httpx.Response | None = None
        self._message_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._reader_task: asyncio.Task[None] | None = None
        self._session_id: str | None = None

    async def connect(self) -> None:
        """Establish SSE connection to the MCP server.

        Raises:
            AdapterConnectionError: If connection fails.
            AdapterTimeoutError: If connection times out.
        """
        if self._state == TransportState.CONNECTED:
            return

        self._state = TransportState.CONNECTING

        try:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self._config.connection_timeout,
                    read=self._config.read_timeout,
                    write=self._config.write_timeout,
                    pool=30.0,
                ),
                verify=self._config.verify_ssl,
                headers=self._config.headers,
            )

            # Start SSE connection
            self._sse_response = await self._client.send(
                self._client.build_request(
                    "GET",
                    self._config.url,
                    headers={"Accept": "text/event-stream"},
                ),
                stream=True,
            )

            if self._sse_response.status_code >= 400:
                await self._sse_response.aread()
                status = self._sse_response.status_code
                self._state = TransportState.DISCONNECTED
                raise AdapterConnectionError(
                    f"SSE connection failed with status {status}",
                    endpoint=self._config.url,
                    adapter_type="mcp",
                )

            # Start background reader task
            self._reader_task = asyncio.create_task(self._read_sse_events())
            self._state = TransportState.CONNECTED

        except httpx.TimeoutException as e:
            self._state = TransportState.DISCONNECTED
            raise AdapterTimeoutError(
                f"Connection timed out after {self._config.connection_timeout}s",
                timeout_seconds=self._config.connection_timeout,
                adapter_type="mcp",
            ) from e
        except httpx.ConnectError as e:
            self._state = TransportState.DISCONNECTED
            raise AdapterConnectionError(
                f"Failed to connect to {self._config.url}",
                endpoint=self._config.url,
                adapter_type="mcp",
                cause=e,
            ) from e
        except httpx.RequestError as e:
            self._state = TransportState.DISCONNECTED
            raise AdapterConnectionError(
                f"Connection failed: {e}",
                endpoint=self._config.url,
                adapter_type="mcp",
                cause=e,
            ) from e

    async def _read_sse_events(self) -> None:
        """Background task to read SSE events and queue them."""
        if not self._sse_response:
            return

        event_type = ""
        event_data = ""

        try:
            async for line in self._sse_response.aiter_lines():
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    event_data = line[5:].strip()
                elif line == "" and event_data:
                    # Empty line signals end of event
                    try:
                        data = json.loads(event_data)

                        # Handle session ID from endpoint event
                        if event_type == "endpoint" and "uri" in data:
                            # Extract session info if provided
                            self._session_id = data.get("sessionId")

                        # Queue the message for receive()
                        await self._message_queue.put(data)

                    except json.JSONDecodeError:
                        pass  # Skip malformed events

                    event_type = ""
                    event_data = ""

        except (httpx.RequestError, httpx.StreamClosed):
            self._state = TransportState.DISCONNECTED

    async def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message via HTTP POST.

        Args:
            message: JSON-RPC message to send.

        Raises:
            AdapterConnectionError: If not connected or send fails.
            AdapterTimeoutError: If send times out.
        """
        if not self._client or self._state != TransportState.CONNECTED:
            raise AdapterConnectionError(
                "Transport not connected",
                adapter_type="mcp",
            )

        post_url = self._config.post_endpoint or self._config.url

        # Include session ID if we have one
        headers = {"Content-Type": "application/json"}
        if self._session_id:
            headers["X-Session-Id"] = self._session_id

        try:
            response = await self._client.post(
                post_url,
                json=message,
                headers=headers,
                timeout=self._config.write_timeout,
            )

            if response.status_code >= 400:
                raise AdapterConnectionError(
                    f"POST request failed with status {response.status_code}",
                    endpoint=post_url,
                    adapter_type="mcp",
                )

        except httpx.TimeoutException as e:
            raise AdapterTimeoutError(
                f"Send timed out after {self._config.write_timeout}s",
                timeout_seconds=self._config.write_timeout,
                adapter_type="mcp",
            ) from e
        except httpx.RequestError as e:
            raise AdapterConnectionError(
                f"Failed to send message: {e}",
                endpoint=post_url,
                adapter_type="mcp",
                cause=e,
            ) from e

    async def receive(self) -> dict[str, Any]:
        """Receive a JSON-RPC message from the SSE stream.

        Returns:
            Parsed JSON-RPC message.

        Raises:
            AdapterConnectionError: If not connected or read fails.
            AdapterTimeoutError: If read times out.
        """
        if self._state != TransportState.CONNECTED:
            raise AdapterConnectionError(
                "Transport not connected",
                adapter_type="mcp",
            )

        try:
            message = await asyncio.wait_for(
                self._message_queue.get(),
                timeout=self._config.read_timeout,
            )
            return message
        except TimeoutError as e:
            raise AdapterTimeoutError(
                f"Receive timed out after {self._config.read_timeout}s",
                timeout_seconds=self._config.read_timeout,
                adapter_type="mcp",
            ) from e

    async def close(self) -> None:
        """Close the SSE connection and HTTP client."""
        self._state = TransportState.CLOSED

        # Cancel reader task
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        # Close SSE response
        if self._sse_response:
            await self._sse_response.aclose()
            self._sse_response = None

        # Close HTTP client
        if self._client:
            await self._client.aclose()
            self._client = None

        # Clear session
        self._session_id = None

    async def health_check(self) -> bool:
        """Check if the SSE connection is healthy.

        Returns:
            True if connected, False otherwise.
        """
        return self._state == TransportState.CONNECTED and self._client is not None

    def stream_events(self) -> AsyncIterator[dict[str, Any]]:
        """Stream events as they arrive from the SSE connection.

        Yields:
            JSON-RPC messages as they are received.
        """
        return _SSEEventIterator(self._message_queue, self._config.read_timeout)


class _SSEEventIterator:
    """Async iterator for SSE events."""

    def __init__(self, queue: asyncio.Queue[dict[str, Any]], timeout: float) -> None:
        self._queue = queue
        self._timeout = timeout

    def __aiter__(self) -> "_SSEEventIterator":
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                self._queue.get(),
                timeout=self._timeout,
            )
        except TimeoutError:
            raise StopAsyncIteration
