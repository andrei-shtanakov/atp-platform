"""Mock tool server using FastAPI."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from atp.mock_tools.loader import MockDefinitionLoader
from atp.mock_tools.models import MockDefinition, MockResponse, MockTool, ToolCall
from atp.mock_tools.recorder import CallRecorder


class ToolCallRequest(BaseModel):
    """Request body for tool calls."""

    tool: str
    input: dict[str, Any] | str | None = None
    task_id: str | None = None


class ToolCallResponse(BaseModel):
    """Response body for tool calls."""

    tool: str
    status: str
    output: dict[str, Any] | str | None = None
    error: str | None = None
    duration_ms: float


class ToolListResponse(BaseModel):
    """Response body for listing tools."""

    tools: list[dict[str, Any]]


class RecordsResponse(BaseModel):
    """Response body for listing call records."""

    records: list[dict[str, Any]]
    total: int


class MockToolServer:
    """Mock tool server with FastAPI backend."""

    def __init__(
        self,
        definition: MockDefinition | None = None,
        record_calls: bool = True,
    ) -> None:
        """Initialize mock tool server.

        Args:
            definition: Mock definition to use
            record_calls: Whether to record calls
        """
        self._definition = definition or MockDefinition(name="default")
        self._recorder = CallRecorder() if record_calls else None
        self._app: FastAPI | None = None

    @property
    def definition(self) -> MockDefinition:
        """Get current mock definition."""
        return self._definition

    @definition.setter
    def definition(self, value: MockDefinition) -> None:
        """Set mock definition."""
        self._definition = value

    @property
    def recorder(self) -> CallRecorder | None:
        """Get call recorder."""
        return self._recorder

    def add_tool(self, tool: MockTool) -> None:
        """Add a tool to the mock definition.

        Args:
            tool: MockTool to add
        """
        # Check if tool already exists and replace it
        existing_tools = [t for t in self._definition.tools if t.name != tool.name]
        existing_tools.append(tool)
        self._definition = MockDefinition(
            name=self._definition.name,
            description=self._definition.description,
            tools=existing_tools,
            default_delay_ms=self._definition.default_delay_ms,
        )

    def load_definition(self, file_path: str | Path) -> None:
        """Load mock definition from YAML file.

        Args:
            file_path: Path to YAML file
        """
        loader = MockDefinitionLoader()
        self._definition = loader.load_file(file_path)

    def load_definition_string(self, content: str) -> None:
        """Load mock definition from YAML string.

        Args:
            content: YAML content
        """
        loader = MockDefinitionLoader()
        self._definition = loader.load_string(content)

    async def call_tool(self, call: ToolCall) -> ToolCallResponse:
        """Execute a tool call.

        Args:
            call: Tool call request

        Returns:
            Tool call response
        """
        start_time = time.perf_counter()

        tool = self._definition.get_tool(call.tool)
        if tool is None:
            error_msg = f"Unknown tool: {call.tool}"
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self._recorder:
                self._recorder.record(
                    tool=call.tool,
                    input_data=call.input,
                    output=None,
                    error=error_msg,
                    status="error",
                    duration_ms=duration_ms,
                    task_id=call.task_id,
                )

            return ToolCallResponse(
                tool=call.tool,
                status="error",
                error=error_msg,
                duration_ms=duration_ms,
            )

        # Get matching response
        response: MockResponse = tool.get_response(call.input)

        # Apply delay
        delay_ms = response.delay_ms or self._definition.default_delay_ms
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Record call
        if self._recorder:
            self._recorder.record(
                tool=call.tool,
                input_data=call.input,
                output=response.output,
                error=response.error,
                status=response.status,
                duration_ms=duration_ms,
                task_id=call.task_id,
            )

        return ToolCallResponse(
            tool=call.tool,
            status=response.status,
            output=response.output,
            error=response.error,
            duration_ms=duration_ms,
        )

    def get_app(self) -> FastAPI:
        """Get or create FastAPI application.

        Returns:
            FastAPI application
        """
        if self._app is None:
            self._app = create_mock_app(self)
        return self._app


def create_mock_app(server: MockToolServer) -> FastAPI:
    """Create FastAPI application for mock tool server.

    Args:
        server: MockToolServer instance

    Returns:
        FastAPI application
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup
        yield
        # Shutdown - clear records
        if server.recorder:
            server.recorder.clear()

    app = FastAPI(
        title="ATP Mock Tool Server",
        description="Mock tool server for testing AI agents",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.post("/tools/call", response_model=ToolCallResponse)
    async def call_tool(request: ToolCallRequest) -> ToolCallResponse:
        """Execute a tool call."""
        call = ToolCall(
            tool=request.tool,
            input=request.input,
            task_id=request.task_id,
        )
        return await server.call_tool(call)

    @app.get("/tools", response_model=ToolListResponse)
    async def list_tools() -> ToolListResponse:
        """List available tools."""
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in server.definition.tools
        ]
        return ToolListResponse(tools=tools)

    @app.get("/tools/{tool_name}")
    async def get_tool(tool_name: str) -> dict[str, Any]:
        """Get tool details."""
        tool = server.definition.get_tool(tool_name)
        if tool is None:
            raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
        return {
            "name": tool.name,
            "description": tool.description,
        }

    @app.get("/records", response_model=RecordsResponse)
    async def list_records(
        tool: str | None = None,
        task_id: str | None = None,
        limit: int | None = None,
    ) -> RecordsResponse:
        """List call records."""
        if server.recorder is None:
            return RecordsResponse(records=[], total=0)

        records = server.recorder.get_records(
            tool=tool,
            task_id=task_id,
            limit=limit,
        )
        return RecordsResponse(
            records=[r.model_dump(mode="json") for r in records],
            total=len(records),
        )

    @app.delete("/records")
    async def clear_records() -> dict[str, int]:
        """Clear all call records."""
        if server.recorder is None:
            return {"cleared": 0}
        count = server.recorder.clear()
        return {"cleared": count}

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    return app
