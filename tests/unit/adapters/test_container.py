"""Unit tests for ContainerAdapter."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.adapters import (
    AdapterConnectionError,
    AdapterError,
    AdapterResponseError,
    AdapterTimeoutError,
    ContainerAdapter,
    ContainerAdapterConfig,
    ContainerResources,
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
def container_config() -> ContainerAdapterConfig:
    """Create container adapter config."""
    return ContainerAdapterConfig(
        image="test-agent:latest",
        timeout_seconds=30.0,
    )


class TestContainerAdapterConfig:
    """Tests for ContainerAdapterConfig."""

    def test_minimal_config(self) -> None:
        """Test creating config with minimal required fields."""
        config = ContainerAdapterConfig(image="my-agent:v1")
        assert config.image == "my-agent:v1"
        assert config.timeout_seconds == 300.0
        assert config.network == "none"
        assert config.auto_remove is True

    def test_full_config(self) -> None:
        """Test creating config with all fields."""
        config = ContainerAdapterConfig(
            image="registry/agent:v2",
            timeout_seconds=600.0,
            resources=ContainerResources(memory="4g", cpu="2"),
            network="bridge",
            environment={"API_KEY": "secret"},
            volumes={"/host/data": "/data"},
            working_dir="/app",
            auto_remove=False,
        )
        assert config.image == "registry/agent:v2"
        assert config.timeout_seconds == 600.0
        assert config.resources.memory == "4g"
        assert config.resources.cpu == "2"
        assert config.network == "bridge"
        assert config.environment == {"API_KEY": "secret"}
        assert config.volumes == {"/host/data": "/data"}
        assert config.working_dir == "/app"
        assert config.auto_remove is False


class TestContainerResources:
    """Tests for ContainerResources."""

    def test_default_resources(self) -> None:
        """Test default resource limits."""
        resources = ContainerResources()
        assert resources.memory == "2g"
        assert resources.cpu == "1"

    def test_custom_resources(self) -> None:
        """Test custom resource limits."""
        resources = ContainerResources(memory="512m", cpu="0.5")
        assert resources.memory == "512m"
        assert resources.cpu == "0.5"


class TestContainerAdapter:
    """Tests for ContainerAdapter."""

    def test_adapter_type(self, container_config: ContainerAdapterConfig) -> None:
        """Test adapter type property."""
        adapter = ContainerAdapter(container_config)
        assert adapter.adapter_type == "container"

    def test_build_docker_command_basic(
        self, container_config: ContainerAdapterConfig
    ) -> None:
        """Test building basic docker command."""
        adapter = ContainerAdapter(container_config)
        cmd = adapter._build_docker_command()

        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert "-i" in cmd
        assert "--rm" in cmd
        assert "--memory" in cmd
        assert "--cpus" in cmd
        assert "--network" in cmd
        assert "test-agent:latest" in cmd

    def test_build_docker_command_full(self) -> None:
        """Test building full docker command with all options."""
        config = ContainerAdapterConfig(
            image="my-agent:v1",
            resources=ContainerResources(memory="4g", cpu="2"),
            network="bridge",
            environment={"KEY": "value"},
            volumes={"/host": "/container"},
            working_dir="/app",
            auto_remove=False,
        )
        adapter = ContainerAdapter(config)
        cmd = adapter._build_docker_command()

        assert "--rm" not in cmd
        assert "--memory" in cmd
        assert "4g" in cmd
        assert "--cpus" in cmd
        assert "2" in cmd
        assert "--network" in cmd
        assert "bridge" in cmd
        assert "-e" in cmd
        assert "KEY=value" in cmd
        assert "-v" in cmd
        assert "/host:/container" in cmd
        assert "-w" in cmd
        assert "/app" in cmd

    @pytest.mark.anyio
    async def test_execute_success(
        self,
        container_config: ContainerAdapterConfig,
        sample_request: ATPRequest,
        sample_response_data: dict,
    ) -> None:
        """Test successful execute call."""
        adapter = ContainerAdapter(container_config)

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(json.dumps(sample_response_data).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await adapter.execute(sample_request)

        assert isinstance(response, ATPResponse)
        assert response.task_id == "test-task-123"
        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_execute_docker_not_found(
        self,
        container_config: ContainerAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute when docker is not installed."""
        adapter = ContainerAdapter(container_config)

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("docker not found"),
        ):
            with pytest.raises(AdapterConnectionError) as exc_info:
                await adapter.execute(sample_request)

        assert "Docker command not found" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_non_zero_exit(
        self,
        container_config: ContainerAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with non-zero exit code."""
        adapter = ContainerAdapter(container_config)

        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error message"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(AdapterError) as exc_info:
                await adapter.execute(sample_request)

        assert "exited with code 1" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_no_output(
        self,
        container_config: ContainerAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with no stdout output."""
        adapter = ContainerAdapter(container_config)

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(AdapterResponseError) as exc_info:
                await adapter.execute(sample_request)

        assert "no output" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_invalid_json(
        self,
        container_config: ContainerAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with invalid JSON output."""
        adapter = ContainerAdapter(container_config)

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"not json", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(AdapterResponseError) as exc_info:
                await adapter.execute(sample_request)

        assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_invalid_response_format(
        self,
        container_config: ContainerAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with invalid ATP response format."""
        adapter = ContainerAdapter(container_config)

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b'{"invalid": "format"}', b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(AdapterResponseError) as exc_info:
                await adapter.execute(sample_request)

        assert "Invalid ATP Response" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_timeout(
        self,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with timeout."""
        config = ContainerAdapterConfig(
            image="test:latest",
            timeout_seconds=0.001,
        )
        adapter = ContainerAdapter(config)

        mock_process = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=TimeoutError())

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("asyncio.wait_for", side_effect=TimeoutError()):
                with pytest.raises(AdapterTimeoutError) as exc_info:
                    await adapter.execute(sample_request)

        assert exc_info.value.timeout_seconds == 0.001
        mock_process.kill.assert_called_once()

    @pytest.mark.anyio
    async def test_health_check_docker_available(
        self,
        container_config: ContainerAdapterConfig,
    ) -> None:
        """Test health check when docker is available."""
        adapter = ContainerAdapter(container_config)

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await adapter.health_check()

        assert result is True

    @pytest.mark.anyio
    async def test_health_check_docker_not_running(
        self,
        container_config: ContainerAdapterConfig,
    ) -> None:
        """Test health check when docker daemon is not running."""
        adapter = ContainerAdapter(container_config)

        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await adapter.health_check()

        assert result is False

    @pytest.mark.anyio
    async def test_health_check_docker_not_installed(
        self,
        container_config: ContainerAdapterConfig,
    ) -> None:
        """Test health check when docker is not installed."""
        adapter = ContainerAdapter(container_config)

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError(),
        ):
            result = await adapter.health_check()

        assert result is False

    @pytest.mark.anyio
    async def test_cleanup(
        self,
        container_config: ContainerAdapterConfig,
    ) -> None:
        """Test cleanup (should be no-op for auto-remove containers)."""
        adapter = ContainerAdapter(container_config)
        # Should not raise
        await adapter.cleanup()

    @pytest.mark.anyio
    async def test_context_manager(
        self,
        container_config: ContainerAdapterConfig,
    ) -> None:
        """Test adapter as async context manager."""
        adapter = ContainerAdapter(container_config)

        async with adapter as ctx_adapter:
            assert ctx_adapter is adapter

    @pytest.mark.anyio
    async def test_stream_events_success(
        self,
        container_config: ContainerAdapterConfig,
        sample_request: ATPRequest,
        sample_response_data: dict,
    ) -> None:
        """Test streaming events from container."""
        adapter = ContainerAdapter(container_config)

        # Mock stderr with events
        event_lines = [
            b'{"event_type": "progress", "payload": {"message": "Starting"}}\n',
            b'{"event_type": "progress", "payload": {"message": "Processing"}}\n',
            b"",  # End of stderr
        ]

        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()
        mock_stdin.close = MagicMock()
        mock_stdin.wait_closed = AsyncMock()

        line_index = 0

        async def readline() -> bytes:
            nonlocal line_index
            if line_index < len(event_lines):
                result = event_lines[line_index]
                line_index += 1
                return result
            return b""

        mock_stderr = MagicMock()
        mock_stderr.readline = readline

        mock_stdout = MagicMock()
        mock_stdout.read = AsyncMock(
            return_value=json.dumps(sample_response_data).encode()
        )

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdin = mock_stdin
        mock_process.stderr = mock_stderr
        mock_process.stdout = mock_stdout
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.time.side_effect = [0, 1, 2, 3, 4, 5]

                events: list[ATPEvent | ATPResponse] = []
                async for item in adapter.stream_events(sample_request):
                    events.append(item)

        # Should have 2 events + 1 response
        assert len(events) == 3
        assert isinstance(events[0], ATPEvent)
        assert isinstance(events[1], ATPEvent)
        assert isinstance(events[2], ATPResponse)
        assert events[2].status == ResponseStatus.COMPLETED
