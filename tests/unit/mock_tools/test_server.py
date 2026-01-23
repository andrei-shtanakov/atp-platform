"""Tests for mock tool server."""

import pytest
from fastapi.testclient import TestClient

from atp.mock_tools.models import (
    MatchType,
    MockDefinition,
    MockResponse,
    MockTool,
    PatternMatcher,
    ToolCall,
)
from atp.mock_tools.server import MockToolServer, create_mock_app


class TestMockToolServer:
    """Tests for MockToolServer class."""

    def test_create_server_default(self) -> None:
        """Test creating server with defaults."""
        server = MockToolServer()

        assert server.definition.name == "default"
        assert server.recorder is not None

    def test_create_server_with_definition(self) -> None:
        """Test creating server with custom definition."""
        definition = MockDefinition(
            name="custom",
            tools=[MockTool(name="tool1")],
        )
        server = MockToolServer(definition=definition)

        assert server.definition.name == "custom"
        assert len(server.definition.tools) == 1

    def test_create_server_without_recording(self) -> None:
        """Test creating server without call recording."""
        server = MockToolServer(record_calls=False)

        assert server.recorder is None

    def test_add_tool(self) -> None:
        """Test adding a tool to server."""
        server = MockToolServer()
        tool = MockTool(name="new_tool")

        server.add_tool(tool)

        assert server.definition.get_tool("new_tool") is not None

    def test_add_tool_replaces_existing(self) -> None:
        """Test adding tool replaces existing with same name."""
        server = MockToolServer()
        tool1 = MockTool(
            name="tool",
            default_response=MockResponse(output={"version": 1}),
        )
        tool2 = MockTool(
            name="tool",
            default_response=MockResponse(output={"version": 2}),
        )

        server.add_tool(tool1)
        server.add_tool(tool2)

        tool = server.definition.get_tool("tool")
        assert tool is not None
        assert tool.default_response.output == {"version": 2}

    @pytest.mark.anyio
    async def test_call_tool_success(self) -> None:
        """Test calling a tool successfully."""
        server = MockToolServer()
        server.add_tool(
            MockTool(
                name="calculator",
                default_response=MockResponse(output={"result": 42}),
            )
        )

        call = ToolCall(tool="calculator", input={"a": 10, "b": 32})
        response = await server.call_tool(call)

        assert response.status == "success"
        assert response.output == {"result": 42}
        assert response.error is None
        assert response.duration_ms >= 0

    @pytest.mark.anyio
    async def test_call_tool_unknown(self) -> None:
        """Test calling unknown tool returns error."""
        server = MockToolServer()

        call = ToolCall(tool="nonexistent")
        response = await server.call_tool(call)

        assert response.status == "error"
        assert "Unknown tool" in str(response.error)

    @pytest.mark.anyio
    async def test_call_tool_with_pattern_matching(self) -> None:
        """Test calling tool with pattern matching."""
        server = MockToolServer()
        server.add_tool(
            MockTool(
                name="search",
                responses=[
                    (
                        PatternMatcher(
                            type=MatchType.CONTAINS,
                            field="query",
                            pattern="python",
                        ),
                        MockResponse(output={"found": True}),
                    ),
                ],
                default_response=MockResponse(output={"found": False}),
            )
        )

        # Matching call
        call1 = ToolCall(tool="search", input={"query": "learn python"})
        response1 = await server.call_tool(call1)
        assert response1.output == {"found": True}

        # Non-matching call
        call2 = ToolCall(tool="search", input={"query": "learn java"})
        response2 = await server.call_tool(call2)
        assert response2.output == {"found": False}

    @pytest.mark.anyio
    async def test_call_tool_records_calls(self) -> None:
        """Test that calls are recorded."""
        server = MockToolServer(record_calls=True)
        server.add_tool(
            MockTool(
                name="tool1",
                default_response=MockResponse(output={"ok": True}),
            )
        )

        call = ToolCall(tool="tool1", input={"key": "value"}, task_id="task-001")
        await server.call_tool(call)

        assert server.recorder is not None
        records = server.recorder.get_records()
        assert len(records) == 1
        assert records[0].tool == "tool1"
        assert records[0].task_id == "task-001"

    @pytest.mark.anyio
    async def test_call_tool_records_errors(self) -> None:
        """Test that error calls are recorded."""
        server = MockToolServer(record_calls=True)

        call = ToolCall(tool="nonexistent")
        await server.call_tool(call)

        assert server.recorder is not None
        records = server.recorder.get_records()
        assert len(records) == 1
        assert records[0].status == "error"

    @pytest.mark.anyio
    async def test_call_tool_with_delay(self) -> None:
        """Test calling tool with delay."""
        import time

        server = MockToolServer()
        server.add_tool(
            MockTool(
                name="slow_tool",
                default_response=MockResponse(output={}, delay_ms=100),
            )
        )

        start = time.perf_counter()
        call = ToolCall(tool="slow_tool")
        response = await server.call_tool(call)
        elapsed = (time.perf_counter() - start) * 1000

        # Should take at least 100ms
        assert elapsed >= 100
        assert response.duration_ms >= 100

    def test_load_definition(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test loading definition from file."""
        yaml_content = """
name: loaded_mock
tools:
  - name: loaded_tool
    default:
      output: {}
"""
        yaml_file = tmp_path / "mock.yaml"
        yaml_file.write_text(yaml_content)

        server = MockToolServer()
        server.load_definition(yaml_file)

        assert server.definition.name == "loaded_mock"
        assert server.definition.get_tool("loaded_tool") is not None

    def test_load_definition_string(self) -> None:
        """Test loading definition from string."""
        yaml_content = """
name: string_mock
tools:
  - name: string_tool
    default:
      output: {}
"""
        server = MockToolServer()
        server.load_definition_string(yaml_content)

        assert server.definition.name == "string_mock"
        assert server.definition.get_tool("string_tool") is not None

    def test_get_app(self) -> None:
        """Test getting FastAPI app."""
        server = MockToolServer()
        app = server.get_app()

        assert app is not None
        # Should return same app on subsequent calls
        assert server.get_app() is app


class TestMockAppEndpoints:
    """Tests for FastAPI endpoints."""

    @pytest.fixture
    def server(self) -> MockToolServer:
        """Create test server."""
        srv = MockToolServer(record_calls=True)
        srv.add_tool(
            MockTool(
                name="test_tool",
                description="A test tool",
                default_response=MockResponse(output={"result": "ok"}),
            )
        )
        return srv

    @pytest.fixture
    def client(self, server: MockToolServer) -> TestClient:
        """Create test client."""
        app = create_mock_app(server)
        return TestClient(app)

    def test_call_tool_endpoint(self, client: TestClient) -> None:
        """Test POST /tools/call endpoint."""
        response = client.post(
            "/tools/call",
            json={
                "tool": "test_tool",
                "input": {"key": "value"},
                "task_id": "task-001",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["tool"] == "test_tool"
        assert data["status"] == "success"
        assert data["output"] == {"result": "ok"}

    def test_call_tool_unknown(self, client: TestClient) -> None:
        """Test calling unknown tool via endpoint."""
        response = client.post(
            "/tools/call",
            json={"tool": "nonexistent"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "Unknown tool" in data["error"]

    def test_list_tools_endpoint(self, client: TestClient) -> None:
        """Test GET /tools endpoint."""
        response = client.get("/tools")

        assert response.status_code == 200
        data = response.json()
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "test_tool"
        assert data["tools"][0]["description"] == "A test tool"

    def test_get_tool_endpoint(self, client: TestClient) -> None:
        """Test GET /tools/{name} endpoint."""
        response = client.get("/tools/test_tool")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_tool"

    def test_get_tool_not_found(self, client: TestClient) -> None:
        """Test GET /tools/{name} for unknown tool."""
        response = client.get("/tools/nonexistent")

        assert response.status_code == 404

    def test_list_records_endpoint(self, client: TestClient) -> None:
        """Test GET /records endpoint."""
        # Make some calls first
        client.post("/tools/call", json={"tool": "test_tool"})
        client.post("/tools/call", json={"tool": "test_tool"})

        response = client.get("/records")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["records"]) == 2

    def test_list_records_with_filter(self, client: TestClient) -> None:
        """Test GET /records with filters."""
        client.post(
            "/tools/call",
            json={"tool": "test_tool", "task_id": "task-001"},
        )
        client.post(
            "/tools/call",
            json={"tool": "test_tool", "task_id": "task-002"},
        )

        response = client.get("/records", params={"task_id": "task-001"})

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1

    def test_clear_records_endpoint(self, client: TestClient) -> None:
        """Test DELETE /records endpoint."""
        client.post("/tools/call", json={"tool": "test_tool"})
        client.post("/tools/call", json={"tool": "test_tool"})

        response = client.delete("/records")

        assert response.status_code == 200
        data = response.json()
        assert data["cleared"] == 2

        # Verify records are cleared
        response = client.get("/records")
        assert response.json()["total"] == 0

    def test_health_endpoint(self, client: TestClient) -> None:
        """Test GET /health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_records_without_recorder(self) -> None:
        """Test records endpoint when recorder is disabled."""
        server = MockToolServer(record_calls=False)
        app = create_mock_app(server)
        client = TestClient(app)

        response = client.get("/records")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["records"] == []
