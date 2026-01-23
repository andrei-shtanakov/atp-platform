"""Unit tests for CLIAdapter."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.adapters import (
    AdapterConnectionError,
    AdapterError,
    AdapterResponseError,
    AdapterTimeoutError,
    CLIAdapter,
    CLIAdapterConfig,
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
def cli_config() -> CLIAdapterConfig:
    """Create CLI adapter config."""
    return CLIAdapterConfig(
        command="python agent.py",
        timeout_seconds=30.0,
    )


class TestCLIAdapterConfig:
    """Tests for CLIAdapterConfig."""

    def test_minimal_config(self) -> None:
        """Test creating config with minimal required fields."""
        config = CLIAdapterConfig(command="my-agent")
        assert config.command == "my-agent"
        assert config.timeout_seconds == 300.0
        assert config.args == []
        assert config.shell is False
        assert config.input_format == "json"
        assert config.output_format == "json"

    def test_full_config(self) -> None:
        """Test creating config with all fields."""
        config = CLIAdapterConfig(
            command="python",
            args=["agent.py", "--verbose"],
            timeout_seconds=60.0,
            working_dir="/app",
            environment={"DEBUG": "true"},
            shell=True,
            input_format="file",
            output_format="file",
            input_file="/tmp/input.json",
            output_file="/tmp/output.json",
        )
        assert config.command == "python"
        assert config.args == ["agent.py", "--verbose"]
        assert config.timeout_seconds == 60.0
        assert config.working_dir == "/app"
        assert config.environment == {"DEBUG": "true"}
        assert config.shell is True
        assert config.input_format == "file"
        assert config.output_format == "file"
        assert config.input_file == "/tmp/input.json"
        assert config.output_file == "/tmp/output.json"


class TestCLIAdapter:
    """Tests for CLIAdapter."""

    def test_adapter_type(self, cli_config: CLIAdapterConfig) -> None:
        """Test adapter type property."""
        adapter = CLIAdapter(cli_config)
        assert adapter.adapter_type == "cli"

    def test_build_command_simple(self, cli_config: CLIAdapterConfig) -> None:
        """Test building simple command."""
        adapter = CLIAdapter(cli_config)
        sample_request = ATPRequest(task_id="test", task=Task(description="Test"))
        cmd = adapter._build_command(sample_request)

        assert cmd == ["python", "agent.py"]

    def test_build_command_with_args(self) -> None:
        """Test building command with arguments."""
        config = CLIAdapterConfig(
            command="agent",
            args=["--config", "config.yaml"],
        )
        adapter = CLIAdapter(config)
        sample_request = ATPRequest(task_id="test", task=Task(description="Test"))
        cmd = adapter._build_command(sample_request)

        assert cmd == ["agent", "--config", "config.yaml"]

    def test_build_command_shell_mode(self) -> None:
        """Test building command in shell mode."""
        config = CLIAdapterConfig(
            command="echo hello",
            args=["world"],
            shell=True,
        )
        adapter = CLIAdapter(config)
        sample_request = ATPRequest(task_id="test", task=Task(description="Test"))
        cmd = adapter._build_command(sample_request)

        # In shell mode, command and args are joined
        assert len(cmd) == 1
        assert "echo hello" in cmd[0]
        assert "world" in cmd[0]

    @pytest.mark.anyio
    async def test_execute_success(
        self,
        cli_config: CLIAdapterConfig,
        sample_request: ATPRequest,
        sample_response_data: dict,
    ) -> None:
        """Test successful execute call."""
        adapter = CLIAdapter(cli_config)

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
    async def test_execute_command_not_found(
        self,
        cli_config: CLIAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute when command is not found."""
        adapter = CLIAdapter(cli_config)

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("command not found"),
        ):
            with pytest.raises(AdapterConnectionError) as exc_info:
                await adapter.execute(sample_request)

        assert "Command not found" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_non_zero_exit(
        self,
        cli_config: CLIAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with non-zero exit code."""
        adapter = CLIAdapter(cli_config)

        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error occurred"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(AdapterError) as exc_info:
                await adapter.execute(sample_request)

        assert "exited with code 1" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_no_output(
        self,
        cli_config: CLIAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with no stdout output."""
        adapter = CLIAdapter(cli_config)

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
        cli_config: CLIAdapterConfig,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with invalid JSON output."""
        adapter = CLIAdapter(cli_config)

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"not json", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(AdapterResponseError) as exc_info:
                await adapter.execute(sample_request)

        assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_timeout(
        self,
        sample_request: ATPRequest,
    ) -> None:
        """Test execute with timeout."""
        config = CLIAdapterConfig(
            command="slow-command",
            timeout_seconds=0.001,
        )
        adapter = CLIAdapter(config)

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
    async def test_execute_with_file_input(
        self,
        sample_request: ATPRequest,
        sample_response_data: dict,
        tmp_path: Path,
    ) -> None:
        """Test execute with file-based input."""
        input_file = tmp_path / "input.json"
        config = CLIAdapterConfig(
            command="agent",
            input_format="file",
            input_file=str(input_file),
        )
        adapter = CLIAdapter(config)

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(json.dumps(sample_response_data).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await adapter.execute(sample_request)

        assert input_file.exists()
        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_execute_with_file_output(
        self,
        sample_request: ATPRequest,
        sample_response_data: dict,
        tmp_path: Path,
    ) -> None:
        """Test execute with file-based output."""
        output_file = tmp_path / "output.json"
        output_file.write_text(json.dumps(sample_response_data))

        config = CLIAdapterConfig(
            command="agent",
            output_format="file",
            output_file=str(output_file),
        )
        adapter = CLIAdapter(config)

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_execute_with_environment(
        self,
        sample_request: ATPRequest,
        sample_response_data: dict,
    ) -> None:
        """Test execute with custom environment variables."""
        config = CLIAdapterConfig(
            command="agent",
            environment={"MY_VAR": "my_value"},
        )
        adapter = CLIAdapter(config)

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(json.dumps(sample_response_data).encode(), b"")
        )

        captured_env = None

        async def capture_subprocess(*args, **kwargs):
            nonlocal captured_env
            captured_env = kwargs.get("env")
            return mock_process

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=capture_subprocess,
        ):
            await adapter.execute(sample_request)

        assert captured_env is not None
        assert captured_env.get("MY_VAR") == "my_value"

    @pytest.mark.anyio
    async def test_execute_shell_mode(
        self,
        sample_request: ATPRequest,
        sample_response_data: dict,
    ) -> None:
        """Test execute in shell mode."""
        config = CLIAdapterConfig(
            command="echo hello | agent",
            shell=True,
        )
        adapter = CLIAdapter(config)

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(json.dumps(sample_response_data).encode(), b"")
        )

        with patch(
            "asyncio.create_subprocess_shell",
            return_value=mock_process,
        ) as mock_shell:
            await adapter.execute(sample_request)

        mock_shell.assert_called_once()

    @pytest.mark.anyio
    async def test_health_check_command_exists(
        self,
        cli_config: CLIAdapterConfig,
    ) -> None:
        """Test health check when command exists."""
        adapter = CLIAdapter(cli_config)

        with patch("shutil.which", return_value="/usr/bin/python"):
            result = await adapter.health_check()

        assert result is True

    @pytest.mark.anyio
    async def test_health_check_command_not_found(
        self,
        cli_config: CLIAdapterConfig,
    ) -> None:
        """Test health check when command doesn't exist."""
        adapter = CLIAdapter(cli_config)

        with patch("shutil.which", return_value=None):
            result = await adapter.health_check()

        assert result is False

    @pytest.mark.anyio
    async def test_cleanup_removes_temp_files(
        self,
        tmp_path: Path,
    ) -> None:
        """Test cleanup removes temporary files."""
        input_file = tmp_path / "input.json"
        output_file = tmp_path / "output.json"
        input_file.touch()
        output_file.touch()

        config = CLIAdapterConfig(
            command="agent",
            input_format="file",
            output_format="file",
            input_file=str(input_file),
            output_file=str(output_file),
        )
        adapter = CLIAdapter(config)

        await adapter.cleanup()

        assert not input_file.exists()
        assert not output_file.exists()

    @pytest.mark.anyio
    async def test_context_manager(
        self,
        cli_config: CLIAdapterConfig,
    ) -> None:
        """Test adapter as async context manager."""
        adapter = CLIAdapter(cli_config)

        async with adapter as ctx_adapter:
            assert ctx_adapter is adapter

    @pytest.mark.anyio
    async def test_stream_events_success(
        self,
        cli_config: CLIAdapterConfig,
        sample_request: ATPRequest,
        sample_response_data: dict,
    ) -> None:
        """Test streaming events from CLI."""
        adapter = CLIAdapter(cli_config)

        # Mock stderr with events
        event_lines = [
            b'{"event_type": "progress", "payload": {"message": "Starting"}}\n',
            b'{"event_type": "progress", "payload": {"message": "Done"}}\n',
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
