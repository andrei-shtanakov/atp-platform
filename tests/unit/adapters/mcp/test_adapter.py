"""Unit tests for MCP adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.adapters.exceptions import (
    AdapterConnectionError,
    AdapterError,
)
from atp.adapters.mcp.adapter import (
    MCPAdapter,
    MCPAdapterConfig,
    MCPPrompt,
    MCPResource,
    MCPServerInfo,
    MCPTool,
)
from atp.adapters.mcp.transport import TransportState
from atp.protocol import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
    ResponseStatus,
    Task,
)


class TestMCPAdapterConfig:
    """Tests for MCPAdapterConfig."""

    def test_stdio_config_minimal(self) -> None:
        """Test minimal stdio config."""
        config = MCPAdapterConfig(
            transport="stdio",
            command="python",
        )
        assert config.transport == "stdio"
        assert config.command == "python"
        assert config.args == []
        assert config.timeout_seconds == 300.0
        assert config.startup_timeout == 30.0

    def test_stdio_config_full(self) -> None:
        """Test full stdio config."""
        config = MCPAdapterConfig(
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            working_dir="/workspace",
            environment={"MCP_ROOT": "/workspace"},
            inherit_environment=False,
            allowed_env_vars=["CUSTOM_VAR"],
            startup_timeout=60.0,
            tools_filter=["read_file", "write_file"],
            timeout_seconds=120.0,
        )
        assert config.transport == "stdio"
        assert config.command == "npx"
        assert config.args == ["-y", "@modelcontextprotocol/server-filesystem"]
        assert config.working_dir == "/workspace"
        assert config.environment == {"MCP_ROOT": "/workspace"}
        assert config.inherit_environment is False
        assert config.allowed_env_vars == ["CUSTOM_VAR"]
        assert config.startup_timeout == 60.0
        assert config.tools_filter == ["read_file", "write_file"]
        assert config.timeout_seconds == 120.0

    def test_sse_config_minimal(self) -> None:
        """Test minimal SSE config."""
        config = MCPAdapterConfig(
            transport="sse",
            url="http://localhost:8080/mcp",
        )
        assert config.transport == "sse"
        assert config.url == "http://localhost:8080/mcp"

    def test_sse_config_full(self) -> None:
        """Test full SSE config."""
        config = MCPAdapterConfig(
            transport="sse",
            url="https://mcp.example.com/api",
            headers={"Authorization": "Bearer token"},
            verify_ssl=False,
            post_endpoint="https://mcp.example.com/messages",
            timeout_seconds=60.0,
        )
        assert config.transport == "sse"
        assert config.url == "https://mcp.example.com/api"
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.verify_ssl is False
        assert config.post_endpoint == "https://mcp.example.com/messages"

    def test_invalid_transport(self) -> None:
        """Test invalid transport type."""
        with pytest.raises(ValueError, match="must be 'stdio' or 'sse'"):
            MCPAdapterConfig(transport="invalid")

    def test_stdio_missing_command(self) -> None:
        """Test stdio transport without command."""
        with pytest.raises(ValueError, match="'command' is required"):
            MCPAdapterConfig(transport="stdio")

    def test_sse_missing_url(self) -> None:
        """Test SSE transport without URL."""
        with pytest.raises(ValueError, match="'url' is required"):
            MCPAdapterConfig(transport="sse")

    def test_client_info_defaults(self) -> None:
        """Test default client info."""
        config = MCPAdapterConfig(
            transport="stdio",
            command="python",
        )
        assert config.client_name == "atp"
        assert config.client_version == "1.0.0"

    def test_custom_client_info(self) -> None:
        """Test custom client info."""
        config = MCPAdapterConfig(
            transport="stdio",
            command="python",
            client_name="my-client",
            client_version="2.0.0",
        )
        assert config.client_name == "my-client"
        assert config.client_version == "2.0.0"


class TestMCPModels:
    """Tests for MCP data models."""

    def test_mcp_tool(self) -> None:
        """Test MCPTool model."""
        tool = MCPTool(
            name="read_file",
            description="Read a file from the filesystem",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        assert tool.name == "read_file"
        assert tool.description == "Read a file from the filesystem"
        assert "properties" in tool.input_schema

    def test_mcp_resource(self) -> None:
        """Test MCPResource model."""
        resource = MCPResource(
            uri="file:///workspace/config.json",
            name="config.json",
            description="Configuration file",
            mime_type="application/json",
        )
        assert resource.uri == "file:///workspace/config.json"
        assert resource.name == "config.json"
        assert resource.mime_type == "application/json"

    def test_mcp_prompt(self) -> None:
        """Test MCPPrompt model."""
        prompt = MCPPrompt(
            name="code_review",
            description="Review code for issues",
            arguments=[
                {"name": "language", "required": True},
                {"name": "style_guide", "required": False},
            ],
        )
        assert prompt.name == "code_review"
        assert len(prompt.arguments) == 2

    def test_mcp_server_info(self) -> None:
        """Test MCPServerInfo model."""
        info = MCPServerInfo(
            name="test-server",
            version="1.0.0",
            protocol_version="2024-11-05",
            capabilities={"tools": True, "resources": True},
        )
        assert info.name == "test-server"
        assert info.version == "1.0.0"
        assert info.capabilities["tools"] is True


class TestMCPAdapter:
    """Tests for MCPAdapter."""

    @pytest.fixture
    def stdio_config(self) -> MCPAdapterConfig:
        """Create stdio config for tests."""
        return MCPAdapterConfig(
            transport="stdio",
            command="python",
            args=["-m", "test_mcp_server"],
            startup_timeout=5.0,
            timeout_seconds=30.0,
        )

    @pytest.fixture
    def sse_config(self) -> MCPAdapterConfig:
        """Create SSE config for tests."""
        return MCPAdapterConfig(
            transport="sse",
            url="http://localhost:8080/mcp",
            startup_timeout=5.0,
            timeout_seconds=30.0,
        )

    @pytest.fixture
    def mock_transport(self) -> MagicMock:
        """Create mock transport for tests."""
        transport = MagicMock()
        transport.connect = AsyncMock()
        transport.close = AsyncMock()
        transport.send = AsyncMock()
        transport.receive = AsyncMock()
        transport.send_request = AsyncMock()
        transport.send_notification = AsyncMock()
        transport.health_check = AsyncMock(return_value=True)
        transport.is_connected = True
        transport.state = TransportState.CONNECTED
        return transport

    def test_adapter_type(self, stdio_config: MCPAdapterConfig) -> None:
        """Test adapter type property."""
        adapter = MCPAdapter(stdio_config)
        assert adapter.adapter_type == "mcp"

    def test_initial_state(self, stdio_config: MCPAdapterConfig) -> None:
        """Test initial adapter state."""
        adapter = MCPAdapter(stdio_config)
        assert adapter.is_initialized is False
        assert adapter.tools == {}
        assert adapter.resources == {}
        assert adapter.prompts == {}
        assert adapter.server_info is None

    @pytest.mark.anyio
    async def test_initialize_success(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test successful initialization."""
        adapter = MCPAdapter(stdio_config)

        # Mock initialize response
        mock_transport.send_request.side_effect = [
            # Initialize response
            {
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "test-server", "version": "1.0.0"},
                    "capabilities": {"tools": {}},
                }
            },
            # Tools list response
            {
                "result": {
                    "tools": [
                        {
                            "name": "read_file",
                            "description": "Read a file",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                }
            },
            # Resources list response
            {"result": {"resources": []}},
            # Prompts list response
            {"result": {"prompts": []}},
        ]

        with patch.object(adapter, "_create_transport", return_value=mock_transport):
            server_info = await adapter.initialize()

        assert adapter.is_initialized is True
        assert server_info.name == "test-server"
        assert "read_file" in adapter.tools
        mock_transport.connect.assert_called_once()
        mock_transport.send_notification.assert_called_once_with(
            "notifications/initialized"
        )

    @pytest.mark.anyio
    async def test_initialize_connection_failure(
        self, stdio_config: MCPAdapterConfig
    ) -> None:
        """Test initialization with connection failure."""
        adapter = MCPAdapter(stdio_config)

        mock_transport = MagicMock()
        mock_transport.connect = AsyncMock(
            side_effect=AdapterConnectionError("Connection refused")
        )
        mock_transport.close = AsyncMock()

        with patch.object(adapter, "_create_transport", return_value=mock_transport):
            with pytest.raises(AdapterConnectionError, match="Connection refused"):
                await adapter.initialize()

        assert adapter.is_initialized is False

    @pytest.mark.anyio
    async def test_initialize_handshake_failure(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test initialization with handshake failure."""
        adapter = MCPAdapter(stdio_config)

        mock_transport.send_request.side_effect = ValueError("Handshake failed")

        with patch.object(adapter, "_create_transport", return_value=mock_transport):
            with pytest.raises(
                AdapterConnectionError, match="MCP initialization failed"
            ):
                await adapter.initialize()

        assert adapter.is_initialized is False
        mock_transport.close.assert_called_once()

    @pytest.mark.anyio
    async def test_tool_filtering(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test tool filtering during discovery."""
        config = MCPAdapterConfig(
            transport="stdio",
            command="python",
            tools_filter=["allowed_tool"],
        )
        adapter = MCPAdapter(config)

        mock_transport.send_request.side_effect = [
            {
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "test", "version": "1.0"},
                    "capabilities": {},
                }
            },
            {
                "result": {
                    "tools": [
                        {"name": "allowed_tool", "description": "Allowed"},
                        {"name": "filtered_tool", "description": "Filtered"},
                    ]
                }
            },
            {"result": {"resources": []}},
            {"result": {"prompts": []}},
        ]

        with patch.object(adapter, "_create_transport", return_value=mock_transport):
            await adapter.initialize()

        assert "allowed_tool" in adapter.tools
        assert "filtered_tool" not in adapter.tools

    @pytest.mark.anyio
    async def test_call_tool_success(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test successful tool call."""
        adapter = MCPAdapter(stdio_config)
        adapter._transport = mock_transport
        adapter._initialized = True
        adapter._tools = {"read_file": MCPTool(name="read_file")}

        mock_transport.send_request.return_value = {
            "result": {"content": [{"type": "text", "text": "File content"}]}
        }

        result = await adapter.call_tool("read_file", {"path": "/test.txt"})

        assert "content" in result
        mock_transport.send_request.assert_called_once_with(
            "tools/call",
            {"name": "read_file", "arguments": {"path": "/test.txt"}},
        )

    @pytest.mark.anyio
    async def test_call_tool_not_found(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test tool call with unknown tool."""
        adapter = MCPAdapter(stdio_config)
        adapter._transport = mock_transport
        adapter._initialized = True
        adapter._tools = {}

        with pytest.raises(AdapterError, match="Tool 'unknown_tool' not found"):
            await adapter.call_tool("unknown_tool")

    @pytest.mark.anyio
    async def test_call_tool_not_connected(
        self, stdio_config: MCPAdapterConfig
    ) -> None:
        """Test tool call when not connected."""
        adapter = MCPAdapter(stdio_config)

        with pytest.raises(AdapterConnectionError, match="Not connected"):
            await adapter.call_tool("read_file")

    @pytest.mark.anyio
    async def test_read_resource_success(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test successful resource read."""
        adapter = MCPAdapter(stdio_config)
        adapter._transport = mock_transport
        adapter._initialized = True

        mock_transport.send_request.return_value = {
            "result": {"contents": [{"uri": "file:///test.txt", "text": "Content"}]}
        }

        result = await adapter.read_resource("file:///test.txt")

        assert "contents" in result
        mock_transport.send_request.assert_called_once_with(
            "resources/read",
            {"uri": "file:///test.txt"},
        )

    @pytest.mark.anyio
    async def test_get_prompt_success(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test successful prompt retrieval."""
        adapter = MCPAdapter(stdio_config)
        adapter._transport = mock_transport
        adapter._initialized = True

        mock_transport.send_request.return_value = {
            "result": {"messages": [{"role": "user", "content": "Test prompt"}]}
        }

        result = await adapter.get_prompt("test_prompt", {"arg": "value"})

        assert "messages" in result
        mock_transport.send_request.assert_called_once_with(
            "prompts/get",
            {"name": "test_prompt", "arguments": {"arg": "value"}},
        )

    @pytest.mark.anyio
    async def test_execute_with_tool_call(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test execute with explicit tool call."""
        adapter = MCPAdapter(stdio_config)
        adapter._transport = mock_transport
        adapter._initialized = True
        adapter._tools = {"read_file": MCPTool(name="read_file")}

        mock_transport.send_request.return_value = {
            "result": {"content": [{"type": "text", "text": "File content here"}]}
        }

        request = ATPRequest(
            task_id="test-1",
            task=Task(
                description="Read a file",
                input_data={"tool": "read_file", "arguments": {"path": "/test.txt"}},
            ),
        )

        response = await adapter.execute(request)

        assert response.status == ResponseStatus.COMPLETED
        assert response.task_id == "test-1"
        assert len(response.artifacts) > 0

    @pytest.mark.anyio
    async def test_execute_with_resource_read(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test execute with resource read."""
        adapter = MCPAdapter(stdio_config)
        adapter._transport = mock_transport
        adapter._initialized = True

        mock_transport.send_request.return_value = {
            "result": {
                "contents": [{"uri": "file:///test.txt", "text": "Resource content"}]
            }
        }

        request = ATPRequest(
            task_id="test-1",
            task=Task(
                description="Read a resource",
                input_data={"resource": "file:///test.txt"},
            ),
        )

        response = await adapter.execute(request)

        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_execute_with_error_result(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test execute with error result from tool."""
        adapter = MCPAdapter(stdio_config)
        adapter._transport = mock_transport
        adapter._initialized = True
        adapter._tools = {"failing_tool": MCPTool(name="failing_tool")}

        mock_transport.send_request.return_value = {
            "result": {
                "content": [{"type": "text", "text": "Error: File not found"}],
                "isError": True,
            }
        }

        request = ATPRequest(
            task_id="test-1",
            task=Task(
                description="Call failing tool",
                input_data={"tool": "failing_tool"},
            ),
        )

        response = await adapter.execute(request)

        assert response.status == ResponseStatus.FAILED
        assert response.error is not None

    @pytest.mark.anyio
    async def test_execute_auto_initialize(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test that execute auto-initializes if needed."""
        adapter = MCPAdapter(stdio_config)

        # Setup mock for initialization
        mock_transport.send_request.side_effect = [
            {
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "test", "version": "1.0"},
                    "capabilities": {},
                }
            },
            {"result": {"tools": [{"name": "test_tool", "description": "Test"}]}},
            {"result": {"resources": []}},
            {"result": {"prompts": []}},
            {"result": {"content": [{"type": "text", "text": "Output"}]}},
        ]

        with patch.object(adapter, "_create_transport", return_value=mock_transport):
            request = ATPRequest(
                task_id="test-1",
                task=Task(description="Test task"),
            )

            response = await adapter.execute(request)

        assert adapter.is_initialized is True
        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_stream_events(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test event streaming."""
        adapter = MCPAdapter(stdio_config)
        adapter._transport = mock_transport
        adapter._initialized = True
        adapter._tools = {"test_tool": MCPTool(name="test_tool")}

        mock_transport.send_request.return_value = {
            "result": {"content": [{"type": "text", "text": "Tool output"}]}
        }

        request = ATPRequest(
            task_id="test-1",
            task=Task(
                description="Test streaming",
                input_data={"tool": "test_tool", "arguments": {}},
            ),
        )

        events = []
        response = None

        async for item in adapter.stream_events(request):
            if isinstance(item, ATPEvent):
                events.append(item)
            elif isinstance(item, ATPResponse):
                response = item

        assert len(events) >= 2  # At least connection + tool call events
        assert response is not None
        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_stream_events_with_error(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test event streaming with tool error."""
        adapter = MCPAdapter(stdio_config)
        adapter._transport = mock_transport
        adapter._initialized = True
        adapter._tools = {"failing_tool": MCPTool(name="failing_tool")}

        mock_transport.send_request.side_effect = AdapterError("Tool execution failed")

        request = ATPRequest(
            task_id="test-1",
            task=Task(
                description="Test streaming error",
                input_data={"tool": "failing_tool"},
            ),
        )

        events = []
        response = None

        async for item in adapter.stream_events(request):
            if isinstance(item, ATPEvent):
                events.append(item)
            elif isinstance(item, ATPResponse):
                response = item

        # Should have error event
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        assert len(error_events) == 1
        assert response is not None
        assert response.status == ResponseStatus.FAILED

    @pytest.mark.anyio
    async def test_health_check_connected(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test health check when connected."""
        adapter = MCPAdapter(stdio_config)
        adapter._transport = mock_transport

        result = await adapter.health_check()

        assert result is True
        mock_transport.health_check.assert_called_once()

    @pytest.mark.anyio
    async def test_health_check_disconnected(
        self, stdio_config: MCPAdapterConfig
    ) -> None:
        """Test health check when disconnected."""
        adapter = MCPAdapter(stdio_config)

        result = await adapter.health_check()

        assert result is False

    @pytest.mark.anyio
    async def test_cleanup(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test cleanup releases resources."""
        adapter = MCPAdapter(stdio_config)
        adapter._transport = mock_transport
        adapter._initialized = True
        adapter._tools = {"test_tool": MCPTool(name="test_tool")}
        adapter._server_info = MCPServerInfo(name="test", version="1.0")

        await adapter.cleanup()

        assert adapter.is_initialized is False
        assert adapter._transport is None
        assert adapter.tools == {}
        assert adapter.server_info is None
        mock_transport.close.assert_called_once()

    @pytest.mark.anyio
    async def test_reconnect_success(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test successful reconnection."""
        adapter = MCPAdapter(stdio_config)
        adapter._transport = mock_transport
        adapter._initialized = True

        # Setup mock for re-initialization
        mock_transport.send_request.side_effect = [
            {
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "test", "version": "1.0"},
                    "capabilities": {},
                }
            },
            {"result": {"tools": []}},
            {"result": {"resources": []}},
            {"result": {"prompts": []}},
        ]

        with patch.object(adapter, "_create_transport", return_value=mock_transport):
            result = await adapter.reconnect()

        assert result is True
        assert adapter.is_initialized is True

    @pytest.mark.anyio
    async def test_reconnect_failure(self, stdio_config: MCPAdapterConfig) -> None:
        """Test reconnection failure."""
        adapter = MCPAdapter(stdio_config)

        mock_transport = MagicMock()
        mock_transport.connect = AsyncMock(
            side_effect=AdapterConnectionError("Connection failed")
        )
        mock_transport.close = AsyncMock()

        with patch.object(adapter, "_create_transport", return_value=mock_transport):
            result = await adapter.reconnect()

        assert result is False
        assert adapter.is_initialized is False

    @pytest.mark.anyio
    async def test_context_manager(
        self, stdio_config: MCPAdapterConfig, mock_transport: MagicMock
    ) -> None:
        """Test adapter as async context manager."""
        adapter = MCPAdapter(stdio_config)
        adapter._transport = mock_transport
        adapter._initialized = True

        async with adapter as a:
            assert a is adapter

        mock_transport.close.assert_called_once()

    def test_create_transport_stdio(self, stdio_config: MCPAdapterConfig) -> None:
        """Test creating stdio transport."""
        adapter = MCPAdapter(stdio_config)
        transport = adapter._create_transport()

        from atp.adapters.mcp.transport import StdioTransport

        assert isinstance(transport, StdioTransport)

    def test_create_transport_sse(self, sse_config: MCPAdapterConfig) -> None:
        """Test creating SSE transport."""
        adapter = MCPAdapter(sse_config)
        transport = adapter._create_transport()

        from atp.adapters.mcp.transport import SSETransport

        assert isinstance(transport, SSETransport)


class TestMCPEventConversion:
    """Tests for MCP event to ATP event conversion."""

    @pytest.fixture
    def adapter(self) -> MCPAdapter:
        """Create adapter for tests."""
        config = MCPAdapterConfig(transport="stdio", command="python")
        return MCPAdapter(config)

    def test_convert_tool_call_started(self, adapter: MCPAdapter) -> None:
        """Test converting tool call started event."""
        mcp_event = {
            "method": "notifications/tools/call_started",
            "params": {"name": "read_file", "arguments": {"path": "/test.txt"}},
        }

        event = adapter._convert_mcp_event_to_atp(mcp_event, "task-1", 0)

        assert event.event_type == EventType.TOOL_CALL
        assert event.payload["tool"] == "read_file"
        assert event.payload["status"] == "started"

    def test_convert_tool_call_complete(self, adapter: MCPAdapter) -> None:
        """Test converting tool call complete event."""
        mcp_event = {
            "method": "notifications/tools/call_complete",
            "params": {"name": "read_file", "result": {"content": "data"}},
        }

        event = adapter._convert_mcp_event_to_atp(mcp_event, "task-1", 1)

        assert event.event_type == EventType.TOOL_CALL
        assert event.payload["tool"] == "read_file"
        assert event.payload["status"] == "success"

    def test_convert_progress_event(self, adapter: MCPAdapter) -> None:
        """Test converting progress event."""
        mcp_event = {
            "method": "notifications/progress",
            "params": {"progressToken": "token-1", "progress": 50, "total": 100},
        }

        event = adapter._convert_mcp_event_to_atp(mcp_event, "task-1", 2)

        assert event.event_type == EventType.PROGRESS
        assert event.payload["progress"] == 50

    def test_convert_message_event(self, adapter: MCPAdapter) -> None:
        """Test converting message event."""
        mcp_event = {
            "method": "notifications/message",
            "params": {"content": "Processing step 1", "level": "info"},
        }

        event = adapter._convert_mcp_event_to_atp(mcp_event, "task-1", 3)

        assert event.event_type == EventType.REASONING
        assert event.payload["thought"] == "Processing step 1"

    def test_convert_unknown_event(self, adapter: MCPAdapter) -> None:
        """Test converting unknown event type."""
        mcp_event = {
            "method": "unknown/event",
            "params": {"data": "value"},
        }

        event = adapter._convert_mcp_event_to_atp(mcp_event, "task-1", 4)

        assert event.event_type == EventType.PROGRESS
        assert event.payload["method"] == "unknown/event"


class TestMCPResponseConversion:
    """Tests for MCP response to ATP response conversion."""

    @pytest.fixture
    def adapter(self) -> MCPAdapter:
        """Create adapter for tests."""
        config = MCPAdapterConfig(transport="stdio", command="python")
        return MCPAdapter(config)

    def test_convert_text_result(self, adapter: MCPAdapter) -> None:
        """Test converting text result."""
        result = {"content": [{"type": "text", "text": "Hello, world!"}]}

        response = adapter._convert_tool_result_to_response(result, "task-1")

        assert response.status == ResponseStatus.COMPLETED
        assert len(response.artifacts) == 1
        assert response.artifacts[0].data["text"] == "Hello, world!"

    def test_convert_error_result(self, adapter: MCPAdapter) -> None:
        """Test converting error result."""
        result = {
            "content": [{"type": "text", "text": "Error: File not found"}],
            "isError": True,
        }

        response = adapter._convert_tool_result_to_response(result, "task-1")

        assert response.status == ResponseStatus.FAILED
        assert response.error == "Error: File not found"

    def test_convert_image_result(self, adapter: MCPAdapter) -> None:
        """Test converting image result."""
        result = {
            "content": [
                {"type": "image", "data": "base64data", "mimeType": "image/png"}
            ]
        }

        response = adapter._convert_tool_result_to_response(result, "task-1")

        assert response.status == ResponseStatus.COMPLETED
        assert len(response.artifacts) == 1
        assert response.artifacts[0].name == "image"

    def test_convert_resource_result(self, adapter: MCPAdapter) -> None:
        """Test converting resource result."""
        result = {
            "content": [
                {
                    "type": "resource",
                    "resource": {
                        "uri": "file:///test.txt",
                        "text": "Content",
                        "mimeType": "text/plain",
                    },
                }
            ]
        }

        response = adapter._convert_tool_result_to_response(result, "task-1")

        assert response.status == ResponseStatus.COMPLETED
        assert len(response.artifacts) == 1
        # Name should be extracted from URI (last path segment)
        assert response.artifacts[0].name == "test.txt"
        assert response.artifacts[0].data["uri"] == "file:///test.txt"

    def test_convert_mixed_content(self, adapter: MCPAdapter) -> None:
        """Test converting mixed content types."""
        result = {
            "content": [
                {"type": "text", "text": "Results: "},
                {"type": "image", "data": "imgdata", "mimeType": "image/jpeg"},
            ]
        }

        response = adapter._convert_tool_result_to_response(result, "task-1")

        assert response.status == ResponseStatus.COMPLETED
        # Should have image artifact + text output artifact
        assert len(response.artifacts) == 2
