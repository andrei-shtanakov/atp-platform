"""End-to-end tests for MCP adapter integration.

Tests the complete flow: test suite loading → MCP adapter → evaluators → results.
Uses a mock MCP server for controlled testing without external dependencies.
"""

from typing import Any

import pytest

from atp.adapters.mcp import MCPAdapter, MCPAdapterConfig
from atp.evaluators.artifact import ArtifactEvaluator
from atp.loader.loader import TestLoader
from atp.loader.models import (
    Assertion,
    Constraints,
    TaskDefinition,
    TestDefinition,
    TestSuite,
)
from atp.protocol import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
    ResponseStatus,
    Task,
)
from atp.runner.models import ProgressEvent, ProgressEventType, SuiteResult
from atp.runner.orchestrator import TestOrchestrator

# =============================================================================
# Mock MCP Server Transport
# =============================================================================


class MockMCPTransport:
    """Mock MCP transport for testing without subprocess or network."""

    def __init__(self) -> None:
        """Initialize mock transport."""
        self._connected = False
        self._tools: list[dict[str, Any]] = [
            {
                "name": "read_file",
                "description": "Read a file from the filesystem",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "list_directory",
                "description": "List files in a directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        ]
        self._resources: list[dict[str, Any]] = [
            {
                "uri": "file:///workspace/config.json",
                "name": "config.json",
                "description": "Configuration file",
                "mimeType": "application/json",
            }
        ]
        self._prompts: list[dict[str, Any]] = [
            {
                "name": "summarize",
                "description": "Summarize content",
                "arguments": [{"name": "content", "required": True}],
            }
        ]
        self._call_count = 0
        self._should_fail = False
        self._fail_message = ""

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    async def connect(self) -> None:
        """Connect to mock server."""
        self._connected = True

    async def close(self) -> None:
        """Close connection."""
        self._connected = False

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a request and get response."""
        self._call_count += 1

        if self._should_fail:
            raise RuntimeError(self._fail_message)

        if method == "initialize":
            return {
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "mock-mcp-server",
                        "version": "1.0.0",
                    },
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False},
                        "prompts": {"listChanged": False},
                    },
                }
            }

        if method == "tools/list":
            return {"result": {"tools": self._tools}}

        if method == "tools/call":
            tool_name = params.get("name", "") if params else ""
            arguments = params.get("arguments", {}) if params else {}

            if tool_name == "read_file":
                path = arguments.get("path", "")
                return {
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Contents of {path}: Hello, World!",
                            }
                        ]
                    }
                }

            if tool_name == "write_file":
                path = arguments.get("path", "")
                return {
                    "result": {
                        "content": [
                            {"type": "text", "text": f"Successfully wrote to {path}"}
                        ]
                    }
                }

            if tool_name == "list_directory":
                return {
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": "file1.txt\nfile2.txt\nconfig.json",
                            }
                        ]
                    }
                }

            return {"result": {"content": [{"type": "text", "text": "Unknown tool"}]}}

        if method == "resources/list":
            return {"result": {"resources": self._resources}}

        if method == "resources/read":
            uri = params.get("uri", "") if params else ""
            if uri == "file:///workspace/config.json":
                return {
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": '{"setting": "value"}',
                            }
                        ]
                    }
                }
            return {"result": {"contents": []}}

        if method == "prompts/list":
            return {"result": {"prompts": self._prompts}}

        return {"result": {}}

    async def send_notification(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Send notification (no response expected)."""
        pass

    async def health_check(self) -> bool:
        """Check health."""
        return self._connected

    def set_failure(self, should_fail: bool, message: str = "Mock failure") -> None:
        """Configure mock to fail."""
        self._should_fail = should_fail
        self._fail_message = message


class MockMCPAdapter(MCPAdapter):
    """MCP Adapter with mock transport for testing."""

    def __init__(
        self,
        config: MCPAdapterConfig,
        mock_transport: MockMCPTransport | None = None,
    ) -> None:
        """Initialize with mock transport."""
        super().__init__(config)
        self._mock_transport = mock_transport or MockMCPTransport()

    def _create_transport(self) -> MockMCPTransport:
        """Create mock transport instead of real one."""
        return self._mock_transport


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_mcp_transport() -> MockMCPTransport:
    """Create a mock MCP transport."""
    return MockMCPTransport()


@pytest.fixture
def mcp_config() -> MCPAdapterConfig:
    """Create MCP adapter config for testing."""
    return MCPAdapterConfig(
        transport="stdio",
        command="python",
        args=["-m", "mock_mcp_server"],
        timeout_seconds=30.0,
        startup_timeout=10.0,
    )


@pytest.fixture
def mock_mcp_adapter(
    mcp_config: MCPAdapterConfig,
    mock_mcp_transport: MockMCPTransport,
) -> MockMCPAdapter:
    """Create mock MCP adapter."""
    return MockMCPAdapter(mcp_config, mock_mcp_transport)


@pytest.fixture
def sample_mcp_test_definition() -> TestDefinition:
    """Create a sample MCP test definition."""
    return TestDefinition(
        id="mcp-test-001",
        name="MCP Tool Test",
        description="Test MCP tool invocation",
        tags=["mcp", "tool"],
        task=TaskDefinition(
            description="Read a file using MCP",
            input_data={
                "tool": "read_file",
                "arguments": {"path": "/workspace/test.txt"},
            },
        ),
        constraints=Constraints(
            max_steps=5,
            timeout_seconds=30,
        ),
        assertions=[
            Assertion(
                type="artifact_exists",
                config={"path": "output"},
            ),
        ],
    )


@pytest.fixture
def mcp_test_suite(sample_mcp_test_definition: TestDefinition) -> TestSuite:
    """Create MCP test suite."""
    return TestSuite(
        test_suite="MCP Adapter Test Suite",
        version="1.0",
        description="End-to-end tests for MCP adapter",
        tests=[
            sample_mcp_test_definition,
            TestDefinition(
                id="mcp-test-002",
                name="MCP Multi-Tool Test",
                task=TaskDefinition(
                    description="List directory and read file",
                    input_data={
                        "tool": "list_directory",
                        "arguments": {"path": "/workspace"},
                    },
                ),
                constraints=Constraints(timeout_seconds=30),
            ),
        ],
    )


# =============================================================================
# MCP Adapter E2E Tests
# =============================================================================


class TestMCPAdapterE2E:
    """End-to-end tests for MCP adapter."""

    @pytest.mark.anyio
    async def test_mcp_adapter_initialization(
        self,
        mock_mcp_adapter: MockMCPAdapter,
    ) -> None:
        """Test MCP adapter initializes and discovers tools."""
        server_info = await mock_mcp_adapter.initialize()

        assert mock_mcp_adapter.is_initialized
        assert server_info.name == "mock-mcp-server"
        assert server_info.version == "1.0.0"

        # Verify tools discovered
        tools = mock_mcp_adapter.tools
        assert "read_file" in tools
        assert "write_file" in tools
        assert "list_directory" in tools

        # Verify resources discovered
        resources = mock_mcp_adapter.resources
        assert "file:///workspace/config.json" in resources

        await mock_mcp_adapter.cleanup()

    @pytest.mark.anyio
    async def test_mcp_adapter_tool_call(
        self,
        mock_mcp_adapter: MockMCPAdapter,
    ) -> None:
        """Test MCP adapter can call tools."""
        await mock_mcp_adapter.initialize()

        result = await mock_mcp_adapter.call_tool(
            "read_file",
            {"path": "/workspace/test.txt"},
        )

        assert "content" in result
        assert len(result["content"]) > 0
        assert "text" in result["content"][0]
        assert "/workspace/test.txt" in result["content"][0]["text"]

        await mock_mcp_adapter.cleanup()

    @pytest.mark.anyio
    async def test_mcp_adapter_execute(
        self,
        mock_mcp_adapter: MockMCPAdapter,
    ) -> None:
        """Test MCP adapter execute method."""
        request = ATPRequest(
            task_id="test-task-001",
            task=Task(
                description="Read a configuration file",
                input_data={
                    "tool": "read_file",
                    "arguments": {"path": "/workspace/config.json"},
                },
            ),
        )

        response = await mock_mcp_adapter.execute(request)

        assert response.task_id == "test-task-001"
        assert response.status == ResponseStatus.COMPLETED
        assert len(response.artifacts) > 0
        assert response.metrics is not None
        assert response.metrics.tool_calls == 1

        await mock_mcp_adapter.cleanup()

    @pytest.mark.anyio
    async def test_mcp_adapter_stream_events(
        self,
        mock_mcp_adapter: MockMCPAdapter,
    ) -> None:
        """Test MCP adapter event streaming."""
        request = ATPRequest(
            task_id="test-stream-001",
            task=Task(
                description="List directory contents",
                input_data={
                    "tool": "list_directory",
                    "arguments": {"path": "/workspace"},
                },
            ),
        )

        events: list[ATPEvent] = []
        response: ATPResponse | None = None

        async for item in mock_mcp_adapter.stream_events(request):
            if isinstance(item, ATPEvent):
                events.append(item)
            else:
                response = item

        assert response is not None
        assert response.status == ResponseStatus.COMPLETED
        assert len(events) > 0  # Should emit at least progress events

        # Check for expected event types
        event_types = [e.event_type for e in events]
        assert EventType.PROGRESS in event_types or EventType.TOOL_CALL in event_types

        await mock_mcp_adapter.cleanup()

    @pytest.mark.anyio
    async def test_mcp_adapter_health_check(
        self,
        mock_mcp_adapter: MockMCPAdapter,
    ) -> None:
        """Test MCP adapter health check."""
        # Before initialization
        is_healthy = await mock_mcp_adapter.health_check()
        assert not is_healthy

        # After initialization
        await mock_mcp_adapter.initialize()
        is_healthy = await mock_mcp_adapter.health_check()
        assert is_healthy

        # After cleanup
        await mock_mcp_adapter.cleanup()
        is_healthy = await mock_mcp_adapter.health_check()
        assert not is_healthy


class TestMCPAdapterPipelineIntegration:
    """Integration tests for MCP adapter in test pipeline."""

    @pytest.mark.anyio
    async def test_mcp_adapter_full_pipeline(
        self,
        mock_mcp_adapter: MockMCPAdapter,
        sample_mcp_test_definition: TestDefinition,
    ) -> None:
        """Test MCP adapter in full test pipeline."""
        orchestrator = TestOrchestrator(
            adapter=mock_mcp_adapter,
            runs_per_test=1,
        )

        result = await orchestrator.run_single_test(sample_mcp_test_definition)

        assert result.total_runs == 1
        assert result.runs[0].response.status == ResponseStatus.COMPLETED

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_mcp_suite_execution(
        self,
        mock_mcp_adapter: MockMCPAdapter,
        mcp_test_suite: TestSuite,
    ) -> None:
        """Test MCP adapter executing full suite."""
        progress_events: list[ProgressEvent] = []

        def progress_callback(event: ProgressEvent) -> None:
            progress_events.append(event)

        orchestrator = TestOrchestrator(
            adapter=mock_mcp_adapter,
            progress_callback=progress_callback,
            runs_per_test=1,
        )

        result = await orchestrator.run_suite(
            mcp_test_suite,
            agent_name="mock-mcp-agent",
        )

        assert isinstance(result, SuiteResult)
        assert result.suite_name == "MCP Adapter Test Suite"
        assert result.total_tests == 2
        assert result.success

        # Check progress events
        event_types = [e.event_type for e in progress_events]
        assert ProgressEventType.SUITE_STARTED in event_types
        assert ProgressEventType.SUITE_COMPLETED in event_types

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_mcp_adapter_with_evaluators(
        self,
        mock_mcp_adapter: MockMCPAdapter,
    ) -> None:
        """Test MCP adapter results with evaluators."""
        test = TestDefinition(
            id="mcp-eval-test",
            name="MCP Evaluator Test",
            task=TaskDefinition(
                description="Read file via MCP",
                input_data={
                    "tool": "read_file",
                    "arguments": {"path": "/workspace/test.txt"},
                },
            ),
            constraints=Constraints(timeout_seconds=30),
            assertions=[
                Assertion(type="artifact_exists", config={"path": "output"}),
            ],
        )

        orchestrator = TestOrchestrator(
            adapter=mock_mcp_adapter,
            runs_per_test=1,
        )

        result = await orchestrator.run_single_test(test)

        assert result.success

        # Run evaluator
        evaluator = ArtifactEvaluator()
        eval_result = await evaluator.evaluate(
            task=test,
            response=result.runs[0].response,
            trace=result.runs[0].events,
            assertion=test.assertions[0],
        )

        assert eval_result.passed

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_mcp_adapter_multi_run(
        self,
        mock_mcp_adapter: MockMCPAdapter,
    ) -> None:
        """Test MCP adapter with multiple runs."""
        test = TestDefinition(
            id="mcp-multi-run-test",
            name="MCP Multi-Run Test",
            task=TaskDefinition(
                description="List files multiple times",
                input_data={
                    "tool": "list_directory",
                    "arguments": {"path": "/workspace"},
                },
            ),
            constraints=Constraints(timeout_seconds=30),
        )

        orchestrator = TestOrchestrator(
            adapter=mock_mcp_adapter,
            runs_per_test=3,
        )

        result = await orchestrator.run_single_test(test)

        assert result.total_runs == 3
        assert result.successful_runs == 3
        assert result.success

        # All runs should complete successfully
        for run in result.runs:
            assert run.success
            assert run.response.status == ResponseStatus.COMPLETED

        await orchestrator.cleanup()


class TestMCPAdapterErrorHandling:
    """Tests for MCP adapter error handling."""

    @pytest.mark.anyio
    async def test_mcp_adapter_tool_not_found(
        self,
        mock_mcp_adapter: MockMCPAdapter,
    ) -> None:
        """Test error handling for unknown tool."""
        await mock_mcp_adapter.initialize()

        from atp.adapters.exceptions import AdapterError

        with pytest.raises(AdapterError, match="not found"):
            await mock_mcp_adapter.call_tool("nonexistent_tool", {})

        await mock_mcp_adapter.cleanup()

    @pytest.mark.anyio
    async def test_mcp_adapter_connection_failure(
        self,
        mcp_config: MCPAdapterConfig,
    ) -> None:
        """Test handling of connection failure."""
        transport = MockMCPTransport()
        transport.set_failure(True, "Connection refused")

        adapter = MockMCPAdapter(mcp_config, transport)

        from atp.adapters.exceptions import AdapterConnectionError

        with pytest.raises(AdapterConnectionError):
            await adapter.initialize()

    @pytest.mark.anyio
    async def test_mcp_adapter_reconnection(
        self,
        mock_mcp_adapter: MockMCPAdapter,
        mock_mcp_transport: MockMCPTransport,
    ) -> None:
        """Test MCP adapter reconnection."""
        await mock_mcp_adapter.initialize()
        assert mock_mcp_adapter.is_initialized

        # Simulate disconnect and reconnect
        success = await mock_mcp_adapter.reconnect()

        assert success
        assert mock_mcp_adapter.is_initialized

        await mock_mcp_adapter.cleanup()


class TestMCPAdapterWithYAML:
    """Tests for MCP adapter with YAML test suite loading."""

    @pytest.mark.anyio
    async def test_load_and_run_mcp_suite(
        self,
        mock_mcp_adapter: MockMCPAdapter,
    ) -> None:
        """Test loading MCP suite from YAML and running."""
        yaml_content = """
test_suite: MCP YAML Test Suite
version: "1.0"
description: Test MCP adapter with YAML config

agents:
  - name: mcp-server
    type: mcp
    config:
      transport: stdio
      command: python
      args: ["-m", "mock_server"]

tests:
  - id: mcp-yaml-001
    name: MCP Tool Discovery
    tags: [mcp, discovery]
    task:
      description: Verify MCP tools are discovered
      input_data:
        tool: list_directory
        arguments:
          path: /workspace
    constraints:
      max_steps: 5
      timeout_seconds: 30
    assertions:
      - type: artifact_exists
        config:
          path: output

  - id: mcp-yaml-002
    name: MCP File Read
    tags: [mcp, file]
    task:
      description: Read file using MCP
      input_data:
        tool: read_file
        arguments:
          path: /workspace/config.json
    constraints:
      timeout_seconds: 30
"""
        loader = TestLoader()
        suite = loader.load_string(yaml_content)

        assert suite.test_suite == "MCP YAML Test Suite"
        assert len(suite.tests) == 2

        # Verify agents section was parsed
        assert len(suite.agents) == 1
        assert suite.agents[0].type == "mcp"
        assert suite.agents[0].config["transport"] == "stdio"

        # Run the suite with mock adapter
        orchestrator = TestOrchestrator(
            adapter=mock_mcp_adapter,
            runs_per_test=1,
        )

        result = await orchestrator.run_suite(suite, agent_name="mcp-server")

        assert result.total_tests == 2
        assert result.success

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_mcp_adapter_with_tool_filtering(
        self,
        mock_mcp_transport: MockMCPTransport,
    ) -> None:
        """Test MCP adapter with tool filtering."""
        config = MCPAdapterConfig(
            transport="stdio",
            command="python",
            timeout_seconds=30.0,
            tools_filter=["read_file"],  # Only allow read_file
        )

        adapter = MockMCPAdapter(config, mock_mcp_transport)
        await adapter.initialize()

        # Should have only filtered tools
        tools = adapter.tools
        assert "read_file" in tools
        assert "write_file" not in tools
        assert "list_directory" not in tools

        await adapter.cleanup()


class TestMCPAdapterRegistration:
    """Tests for MCP adapter registration in registry."""

    def test_mcp_adapter_registered(self) -> None:
        """Test MCP adapter is registered in the registry."""
        from atp.adapters import get_registry

        registry = get_registry()

        assert registry.is_registered("mcp")
        assert "mcp" in registry.list_adapters()

    def test_mcp_adapter_creation_via_registry(self) -> None:
        """Test creating MCP adapter via registry."""
        from atp.adapters import create_adapter

        config = {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "mcp_server"],
            "timeout_seconds": 30.0,
        }

        adapter = create_adapter("mcp", config)

        assert adapter.adapter_type == "mcp"
        assert isinstance(adapter, MCPAdapter)

    def test_mcp_config_class_from_registry(self) -> None:
        """Test getting MCP config class from registry."""
        from atp.adapters import get_registry

        registry = get_registry()
        config_class = registry.get_config_class("mcp")

        assert config_class == MCPAdapterConfig

        # Verify we can instantiate it
        config = config_class(
            transport="sse",
            url="http://localhost:8080/mcp",
        )
        assert config.transport == "sse"
