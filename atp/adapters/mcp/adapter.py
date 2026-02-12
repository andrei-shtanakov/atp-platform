"""MCP (Model Context Protocol) adapter implementation."""

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field, field_validator

from atp.protocol import (
    ArtifactStructured,
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
    Metrics,
    ResponseStatus,
)

from ..base import AdapterConfig, AgentAdapter
from ..exceptions import (
    AdapterConnectionError,
    AdapterError,
    AdapterTimeoutError,
)
from .transport import (
    MCPTransport,
    SSETransport,
    SSETransportConfig,
    StdioTransport,
    StdioTransportConfig,
)

logger = logging.getLogger(__name__)


class MCPTool(BaseModel):
    """MCP tool definition."""

    name: str = Field(..., description="Tool name")
    description: str | None = Field(None, description="Tool description")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for tool input"
    )


class MCPResource(BaseModel):
    """MCP resource definition."""

    uri: str = Field(..., description="Resource URI")
    name: str = Field(..., description="Resource name")
    description: str | None = Field(None, description="Resource description")
    mime_type: str | None = Field(None, description="Resource MIME type")


class MCPPrompt(BaseModel):
    """MCP prompt template."""

    name: str = Field(..., description="Prompt name")
    description: str | None = Field(None, description="Prompt description")
    arguments: list[dict[str, Any]] = Field(
        default_factory=list, description="Prompt arguments"
    )


class MCPServerInfo(BaseModel):
    """MCP server information from initialize response."""

    name: str = Field(..., description="Server name")
    version: str = Field(..., description="Server version")
    protocol_version: str = Field(default="1.0", description="MCP protocol version")
    capabilities: dict[str, Any] = Field(
        default_factory=dict, description="Server capabilities"
    )


class MCPAdapterConfig(AdapterConfig):
    """Configuration for MCP adapter."""

    transport: str = Field(
        default="stdio",
        description="Transport type: 'stdio' or 'sse'",
    )

    # Stdio transport settings
    command: str | None = Field(
        None, description="Command to execute MCP server (for stdio transport)"
    )
    args: list[str] = Field(
        default_factory=list, description="Command arguments (for stdio transport)"
    )
    working_dir: str | None = Field(
        None, description="Working directory (for stdio transport)"
    )
    environment: dict[str, str] = Field(
        default_factory=dict, description="Environment variables (for stdio transport)"
    )
    inherit_environment: bool = Field(
        default=True,
        description="Inherit parent environment (filtered) for stdio transport",
    )
    allowed_env_vars: list[str] = Field(
        default_factory=list,
        description="Additional env vars to allow when inheriting",
    )

    # SSE transport settings
    url: str | None = Field(None, description="SSE endpoint URL (for sse transport)")
    headers: dict[str, str] = Field(
        default_factory=dict, description="HTTP headers (for sse transport)"
    )
    verify_ssl: bool = Field(
        default=True, description="Verify SSL certificates (for sse transport)"
    )
    post_endpoint: str | None = Field(
        None, description="Separate POST endpoint (for sse transport)"
    )

    # MCP-specific settings
    startup_timeout: float = Field(
        default=30.0, description="Timeout for MCP server startup in seconds", gt=0
    )
    tools_filter: list[str] | None = Field(
        None, description="List of tool names to expose (None = all tools)"
    )
    resources_filter: list[str] | None = Field(
        None, description="List of resource URIs to expose (None = all resources)"
    )

    # Client info for MCP handshake
    client_name: str = Field(default="atp", description="MCP client name")
    client_version: str = Field(default="1.0.0", description="MCP client version")

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, v: str) -> str:
        """Validate transport type."""
        v = v.lower().strip()
        if v not in ("stdio", "sse"):
            raise ValueError("Transport must be 'stdio' or 'sse'")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate that required fields are present for the transport type."""
        if self.transport == "stdio" and not self.command:
            raise ValueError("'command' is required for stdio transport")
        if self.transport == "sse" and not self.url:
            raise ValueError("'url' is required for sse transport")


class MCPAdapter(AgentAdapter):
    """
    Adapter for agents communicating via Model Context Protocol (MCP).

    MCP is a protocol for tool and resource access that supports:
    - Tool discovery and invocation
    - Resource access (files, data, etc.)
    - Prompt templates
    - Event streaming

    Supports both stdio (subprocess) and SSE (HTTP) transports.
    """

    def __init__(self, config: MCPAdapterConfig) -> None:
        """
        Initialize MCP adapter.

        Args:
            config: MCP adapter configuration.
        """
        super().__init__(config)
        self._config: MCPAdapterConfig = config
        self._transport: MCPTransport | None = None
        self._tools: dict[str, MCPTool] = {}
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}
        self._server_info: MCPServerInfo | None = None
        self._initialized = False

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""
        return "mcp"

    @property
    def is_initialized(self) -> bool:
        """Check if the adapter is initialized."""
        return self._initialized

    @property
    def tools(self) -> dict[str, MCPTool]:
        """Get discovered tools (after initialization)."""
        return self._tools.copy()

    @property
    def resources(self) -> dict[str, MCPResource]:
        """Get discovered resources (after initialization)."""
        return self._resources.copy()

    @property
    def prompts(self) -> dict[str, MCPPrompt]:
        """Get discovered prompts (after initialization)."""
        return self._prompts.copy()

    @property
    def server_info(self) -> MCPServerInfo | None:
        """Get MCP server info (after initialization)."""
        return self._server_info

    def _create_transport(self) -> MCPTransport:
        """Create the appropriate transport based on configuration."""
        if self._config.transport == "stdio":
            if not self._config.command:
                raise AdapterError(
                    "Command required for stdio transport",
                    adapter_type=self.adapter_type,
                )
            stdio_config = StdioTransportConfig(
                command=self._config.command,
                args=self._config.args,
                working_dir=self._config.working_dir,
                environment=self._config.environment,
                inherit_environment=self._config.inherit_environment,
                allowed_env_vars=self._config.allowed_env_vars,
                connection_timeout=self._config.startup_timeout,
                read_timeout=self._config.timeout_seconds,
                write_timeout=30.0,
            )
            return StdioTransport(stdio_config)

        elif self._config.transport == "sse":
            if not self._config.url:
                raise AdapterError(
                    "URL required for SSE transport",
                    adapter_type=self.adapter_type,
                )
            sse_config = SSETransportConfig(
                url=self._config.url,
                headers=self._config.headers,
                verify_ssl=self._config.verify_ssl,
                post_endpoint=self._config.post_endpoint,
                connection_timeout=self._config.startup_timeout,
                read_timeout=self._config.timeout_seconds,
                write_timeout=30.0,
            )
            return SSETransport(sse_config)

        else:
            raise AdapterError(
                f"Unknown transport type: {self._config.transport}",
                adapter_type=self.adapter_type,
            )

    async def initialize(self) -> MCPServerInfo:
        """
        Initialize MCP connection and perform handshake.

        Performs the MCP initialization handshake and discovers
        available tools, resources, and prompts.

        Returns:
            MCPServerInfo with server capabilities.

        Raises:
            AdapterConnectionError: If connection or handshake fails.
            AdapterTimeoutError: If initialization times out.
        """
        if self._initialized and self._transport and self._transport.is_connected:
            if self._server_info:
                return self._server_info
            raise AdapterError(
                "Adapter marked as initialized but has no server info",
                adapter_type=self.adapter_type,
            )

        # Create and connect transport
        self._transport = self._create_transport()
        await self._transport.connect()

        try:
            # Perform MCP handshake
            init_response = await self._transport.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {"listChanged": True},
                        "sampling": {},
                    },
                    "clientInfo": {
                        "name": self._config.client_name,
                        "version": self._config.client_version,
                    },
                },
                timeout=self._config.startup_timeout,
            )

            # Parse server info
            result = init_response.get("result", {})
            self._server_info = MCPServerInfo(
                name=result.get("serverInfo", {}).get("name", "unknown"),
                version=result.get("serverInfo", {}).get("version", "unknown"),
                protocol_version=result.get("protocolVersion", "1.0"),
                capabilities=result.get("capabilities", {}),
            )

            # Send initialized notification
            await self._transport.send_notification("notifications/initialized")

            # Discover tools
            await self._discover_tools()

            # Discover resources
            await self._discover_resources()

            # Discover prompts
            await self._discover_prompts()

            self._initialized = True
            return self._server_info

        except Exception as e:
            # Close transport on initialization failure
            await self._transport.close()
            self._transport = None
            if isinstance(e, (AdapterConnectionError, AdapterTimeoutError)):
                raise
            raise AdapterConnectionError(
                f"MCP initialization failed: {e}",
                adapter_type=self.adapter_type,
                cause=e if isinstance(e, Exception) else None,
            ) from e

    async def _discover_tools(self) -> None:
        """Discover available tools from MCP server."""
        if not self._transport:
            return

        try:
            response = await self._transport.send_request("tools/list")
            tools_data = response.get("result", {}).get("tools", [])

            self._tools = {}
            for tool_data in tools_data:
                tool_name = tool_data.get("name", "")
                if self._should_include_tool(tool_name):
                    self._tools[tool_name] = MCPTool(
                        name=tool_name,
                        description=tool_data.get("description"),
                        input_schema=tool_data.get("inputSchema", {}),
                    )
        except ValueError:
            logger.debug("Server does not support tools/list")

    async def _discover_resources(self) -> None:
        """Discover available resources from MCP server."""
        if not self._transport:
            return

        try:
            response = await self._transport.send_request("resources/list")
            resources_data = response.get("result", {}).get("resources", [])

            self._resources = {}
            for resource_data in resources_data:
                resource_uri = resource_data.get("uri", "")
                if self._should_include_resource(resource_uri):
                    self._resources[resource_uri] = MCPResource(
                        uri=resource_uri,
                        name=resource_data.get("name", resource_uri),
                        description=resource_data.get("description"),
                        mime_type=resource_data.get("mimeType"),
                    )
        except ValueError:
            logger.debug("Server does not support resources/list")

    async def _discover_prompts(self) -> None:
        """Discover available prompts from MCP server."""
        if not self._transport:
            return

        try:
            response = await self._transport.send_request("prompts/list")
            prompts_data = response.get("result", {}).get("prompts", [])

            self._prompts = {}
            for prompt_data in prompts_data:
                prompt_name = prompt_data.get("name", "")
                self._prompts[prompt_name] = MCPPrompt(
                    name=prompt_name,
                    description=prompt_data.get("description"),
                    arguments=prompt_data.get("arguments", []),
                )
        except ValueError:
            logger.debug("Server does not support prompts/list")

    def _should_include_tool(self, tool_name: str) -> bool:
        """Check if a tool should be included based on filter."""
        if not self._config.tools_filter:
            return True
        return tool_name in self._config.tools_filter

    def _should_include_resource(self, resource_uri: str) -> bool:
        """Check if a resource should be included based on filter."""
        if not self._config.resources_filter:
            return True
        return resource_uri in self._config.resources_filter

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Call an MCP tool.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments.

        Returns:
            Tool result as a dictionary.

        Raises:
            AdapterError: If tool call fails.
            AdapterConnectionError: If not connected.
        """
        if not self._transport or not self._transport.is_connected:
            raise AdapterConnectionError(
                "Not connected to MCP server",
                adapter_type=self.adapter_type,
            )

        if tool_name not in self._tools:
            raise AdapterError(
                f"Tool '{tool_name}' not found or not available",
                adapter_type=self.adapter_type,
            )

        response = await self._transport.send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments or {},
            },
        )

        return response.get("result", {})

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """
        Read an MCP resource.

        Args:
            uri: Resource URI.

        Returns:
            Resource content as a dictionary.

        Raises:
            AdapterError: If resource read fails.
            AdapterConnectionError: If not connected.
        """
        if not self._transport or not self._transport.is_connected:
            raise AdapterConnectionError(
                "Not connected to MCP server",
                adapter_type=self.adapter_type,
            )

        response = await self._transport.send_request(
            "resources/read",
            {"uri": uri},
        )

        return response.get("result", {})

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Get an MCP prompt with arguments.

        Args:
            prompt_name: Name of the prompt.
            arguments: Prompt arguments.

        Returns:
            Prompt messages as a dictionary.

        Raises:
            AdapterError: If prompt retrieval fails.
            AdapterConnectionError: If not connected.
        """
        if not self._transport or not self._transport.is_connected:
            raise AdapterConnectionError(
                "Not connected to MCP server",
                adapter_type=self.adapter_type,
            )

        response = await self._transport.send_request(
            "prompts/get",
            {
                "name": prompt_name,
                "arguments": arguments or {},
            },
        )

        return response.get("result", {})

    def _convert_mcp_event_to_atp(
        self,
        mcp_event: dict[str, Any],
        task_id: str,
        sequence: int,
    ) -> ATPEvent:
        """Convert an MCP notification to an ATP event."""
        method = mcp_event.get("method", "")
        params = mcp_event.get("params", {})

        # Map MCP methods to ATP event types
        if method == "notifications/tools/call_started":
            return ATPEvent(
                task_id=task_id,
                sequence=sequence,
                event_type=EventType.TOOL_CALL,
                payload={
                    "tool": params.get("name", "unknown"),
                    "input": params.get("arguments", {}),
                    "status": "started",
                },
            )
        elif method == "notifications/tools/call_complete":
            return ATPEvent(
                task_id=task_id,
                sequence=sequence,
                event_type=EventType.TOOL_CALL,
                payload={
                    "tool": params.get("name", "unknown"),
                    "output": params.get("result", {}),
                    "status": "success",
                },
            )
        elif method == "notifications/progress":
            return ATPEvent(
                task_id=task_id,
                sequence=sequence,
                event_type=EventType.PROGRESS,
                payload={
                    "progress_token": params.get("progressToken"),
                    "progress": params.get("progress"),
                    "total": params.get("total"),
                },
            )
        elif method.startswith("notifications/message"):
            return ATPEvent(
                task_id=task_id,
                sequence=sequence,
                event_type=EventType.REASONING,
                payload={
                    "thought": params.get("content", ""),
                    "level": params.get("level", "info"),
                },
            )
        else:
            # Generic event mapping
            return ATPEvent(
                task_id=task_id,
                sequence=sequence,
                event_type=EventType.PROGRESS,
                payload={
                    "method": method,
                    "params": params,
                },
            )

    def _convert_tool_result_to_response(
        self,
        result: dict[str, Any],
        task_id: str,
        metrics: Metrics | None = None,
    ) -> ATPResponse:
        """Convert MCP tool result to ATP response."""
        # Extract content from MCP result
        content = result.get("content", [])
        is_error = result.get("isError", False)

        artifacts = []
        output_text = ""

        for item in content:
            item_type = item.get("type", "text")
            if item_type == "text":
                output_text += item.get("text", "")
            elif item_type == "image":
                artifacts.append(
                    ArtifactStructured(
                        name="image",
                        data={
                            "data": item.get("data", ""),
                            "mimeType": item.get("mimeType", "image/png"),
                        },
                        content_type=item.get("mimeType", "image/png"),
                    )
                )
            elif item_type == "resource":
                # Extract resource name from URI (last path segment without slashes)
                resource_uri = item.get("resource", {}).get("uri", "resource")
                resource_name = resource_uri.rsplit("/", 1)[-1] or "resource"
                # Further sanitize to remove any remaining path separators
                resource_name = resource_name.replace("\\", "_").replace("/", "_")
                artifacts.append(
                    ArtifactStructured(
                        name=resource_name,
                        data=item.get("resource", {}),
                        content_type=item.get("resource", {}).get("mimeType"),
                    )
                )

        # Add output as artifact if present
        if output_text:
            artifacts.append(
                ArtifactStructured(
                    name="output",
                    data={"text": output_text},
                    content_type="text/plain",
                )
            )

        return ATPResponse(
            task_id=task_id,
            status=ResponseStatus.FAILED if is_error else ResponseStatus.COMPLETED,
            artifacts=artifacts,
            metrics=metrics,
            error=output_text if is_error else None,
        )

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """
        Execute a task via MCP.

        The task description is sent to the MCP server, which processes it
        using available tools and resources. This implementation uses the
        first available tool based on the task description.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse with execution results.

        Raises:
            AdapterConnectionError: If not connected or connection fails.
            AdapterTimeoutError: If execution times out.
            AdapterResponseError: If MCP server returns invalid response.
        """
        start_time = time.time()

        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        if not self._transport or not self._transport.is_connected:
            raise AdapterConnectionError(
                "Not connected to MCP server",
                adapter_type=self.adapter_type,
            )

        # Extract tool call from task if specified
        task_input = request.task.input_data or {}
        tool_name = task_input.get("tool")
        tool_arguments = task_input.get("arguments", {})
        resource_uri = task_input.get("resource")

        tool_calls = 0
        result: dict[str, Any] = {}

        try:
            if tool_name and tool_name in self._tools:
                # Direct tool call
                result = await self.call_tool(tool_name, tool_arguments)
                tool_calls = 1
            elif resource_uri:
                # Resource read
                result = await self.read_resource(resource_uri)
            elif self._tools:
                # Default: use first available tool with task description
                first_tool = next(iter(self._tools.keys()))
                result = await self.call_tool(
                    first_tool,
                    {"input": request.task.description},
                )
                tool_calls = 1
            else:
                # No tools available
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Task received: {request.task.description}",
                        }
                    ]
                }

        except AdapterTimeoutError:
            raise
        except Exception as e:
            wall_time = time.time() - start_time
            return ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error=str(e),
                metrics=Metrics(
                    wall_time_seconds=wall_time,
                    tool_calls=tool_calls,
                ),
            )

        wall_time = time.time() - start_time
        metrics = Metrics(
            wall_time_seconds=wall_time,
            tool_calls=tool_calls,
        )

        return self._convert_tool_result_to_response(result, request.task_id, metrics)

    async def stream_events(
        self, request: ATPRequest
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
            AdapterConnectionError: If not connected or connection fails.
            AdapterTimeoutError: If execution times out.
            AdapterResponseError: If MCP server returns invalid response.
        """
        start_time = time.time()
        sequence = 0

        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        if not self._transport or not self._transport.is_connected:
            raise AdapterConnectionError(
                "Not connected to MCP server",
                adapter_type=self.adapter_type,
            )

        # Emit initialization event
        yield ATPEvent(
            task_id=request.task_id,
            sequence=sequence,
            event_type=EventType.PROGRESS,
            payload={"message": "MCP connection established"},
        )
        sequence += 1

        # Extract tool call from task
        task_input = request.task.input_data or {}
        tool_name = task_input.get("tool")
        tool_arguments = task_input.get("arguments", {})
        resource_uri = task_input.get("resource")

        tool_calls = 0

        # Determine operation
        if tool_name and tool_name in self._tools:
            operation = "tool"
            operation_target = tool_name
        elif resource_uri:
            operation = "resource"
            operation_target = resource_uri
        elif self._tools:
            operation = "tool"
            operation_target = next(iter(self._tools.keys()))
            tool_arguments = {"input": request.task.description}
        else:
            operation = "none"
            operation_target = ""

        # Emit operation started event
        yield ATPEvent(
            task_id=request.task_id,
            sequence=sequence,
            event_type=EventType.TOOL_CALL
            if operation == "tool"
            else EventType.PROGRESS,
            payload={
                "tool": operation_target if operation == "tool" else None,
                "resource": operation_target if operation == "resource" else None,
                "status": "started",
                "input": tool_arguments if operation == "tool" else None,
            },
        )
        sequence += 1

        try:
            if operation == "tool":
                result = await self.call_tool(operation_target, tool_arguments)
                tool_calls = 1

                # Emit tool complete event
                yield ATPEvent(
                    task_id=request.task_id,
                    sequence=sequence,
                    event_type=EventType.TOOL_CALL,
                    payload={
                        "tool": operation_target,
                        "output": result,
                        "status": "success",
                    },
                )
                sequence += 1

            elif operation == "resource":
                result = await self.read_resource(operation_target)

                # Emit resource read event
                yield ATPEvent(
                    task_id=request.task_id,
                    sequence=sequence,
                    event_type=EventType.PROGRESS,
                    payload={
                        "message": f"Resource read: {operation_target}",
                        "resource": result,
                    },
                )
                sequence += 1

            else:
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Task received: {request.task.description}",
                        }
                    ]
                }

        except AdapterTimeoutError:
            raise
        except Exception as e:
            # Emit error event
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
            sequence += 1

            wall_time = time.time() - start_time
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error=str(e),
                metrics=Metrics(
                    wall_time_seconds=wall_time,
                    tool_calls=tool_calls,
                ),
            )
            return

        # Yield final response
        wall_time = time.time() - start_time
        metrics = Metrics(
            wall_time_seconds=wall_time,
            tool_calls=tool_calls,
        )

        yield self._convert_tool_result_to_response(result, request.task_id, metrics)

    async def health_check(self) -> bool:
        """
        Check if MCP server is healthy.

        Returns:
            True if server is connected and responding, False otherwise.
        """
        if not self._transport:
            return False

        if not self._transport.is_connected:
            return False

        # Check transport-specific health
        return await self._transport.health_check()

    async def cleanup(self) -> None:
        """Close MCP connection and release resources."""
        self._initialized = False
        self._tools = {}
        self._resources = {}
        self._prompts = {}
        self._server_info = None

        if self._transport:
            await self._transport.close()
            self._transport = None

    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to MCP server.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        # Close existing connection
        await self.cleanup()

        try:
            await self.initialize()
            return True
        except (AdapterConnectionError, AdapterTimeoutError):
            return False
