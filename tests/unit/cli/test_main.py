"""Tests for CLI main module."""

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from atp import __version__
from atp.cli.main import (
    EXIT_ERROR,
    EXIT_FAILURE,
    EXIT_SUCCESS,
    ConfigContext,
    cli,
)


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_suite_content() -> str:
    """Sample test suite YAML content."""
    return """
test_suite: test-suite
version: "1.0"
description: Test suite for CLI tests
tests:
  - id: test-001
    name: Test 1
    tags: [smoke]
    task:
      description: Test task 1
    constraints:
      timeout_seconds: 30
  - id: test-002
    name: Test 2
    tags: [integration]
    task:
      description: Test task 2
    constraints:
      timeout_seconds: 60
"""


@pytest.fixture
def sample_config_content() -> str:
    """Sample config file content."""
    return """
version: "1.0"

defaults:
  timeout_seconds: 300
  runs_per_test: 3
  parallel_workers: 4

agents:
  my-http-agent:
    type: http
    endpoint: "http://localhost:8000"
    timeout_seconds: 300
  my-cli-agent:
    type: cli
    command: "python agent.py"
"""


class TestExitCodes:
    """Tests for exit codes constants."""

    def test_exit_codes_defined(self) -> None:
        """Test that exit codes are defined correctly."""
        assert EXIT_SUCCESS == 0
        assert EXIT_FAILURE == 1
        assert EXIT_ERROR == 2


class TestConfigContext:
    """Tests for ConfigContext class."""

    def test_init(self) -> None:
        """Test ConfigContext initialization."""
        ctx = ConfigContext()
        assert ctx.config == {}
        assert ctx.config_file is None
        assert ctx.verbose is False

    def test_get_default_missing(self) -> None:
        """Test get_default with missing key."""
        ctx = ConfigContext()
        assert ctx.get_default("missing_key") is None
        assert ctx.get_default("missing_key", "default_value") == "default_value"

    def test_get_default_from_config(self) -> None:
        """Test get_default with config loaded."""
        ctx = ConfigContext()
        ctx.config = {"defaults": {"timeout_seconds": 300}}
        assert ctx.get_default("timeout_seconds") == 300
        assert ctx.get_default("missing_key", 100) == 100

    def test_get_agent_config_missing(self) -> None:
        """Test get_agent_config with missing agent."""
        ctx = ConfigContext()
        assert ctx.get_agent_config("missing-agent") is None

    def test_get_agent_config_from_config(self) -> None:
        """Test get_agent_config with config loaded."""
        ctx = ConfigContext()
        ctx.config = {
            "agents": {
                "my-agent": {"type": "http", "endpoint": "http://localhost:8000"}
            }
        }
        agent_cfg = ctx.get_agent_config("my-agent")
        assert agent_cfg is not None
        assert agent_cfg["type"] == "http"
        assert agent_cfg["endpoint"] == "http://localhost:8000"

    def test_load_config_file(self, tmp_path: Path, sample_config_content: str) -> None:
        """Test loading config from file."""
        config_file = tmp_path / "atp.config.yaml"
        config_file.write_text(sample_config_content)

        ctx = ConfigContext()
        ctx.load_config(config_file)

        assert ctx.config_file == config_file
        assert ctx.config["version"] == "1.0"
        assert ctx.get_default("timeout_seconds") == 300
        assert ctx.get_agent_config("my-http-agent") is not None

    def test_load_config_search(
        self, tmp_path: Path, sample_config_content: str
    ) -> None:
        """Test config file search."""
        config_file = tmp_path / "atp.config.yaml"
        config_file.write_text(sample_config_content)

        ctx = ConfigContext()
        # Patch Path.cwd to return tmp_path
        with patch("atp.cli.main.Path.cwd", return_value=tmp_path):
            ctx.load_config()

        assert ctx.config_file == config_file


class TestCLIGroup:
    """Tests for CLI group."""

    def test_cli_help(self, runner: CliRunner) -> None:
        """Test CLI help output."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "ATP - Agent Test Platform CLI" in result.output
        assert "Examples:" in result.output

    def test_cli_version(self, runner: CliRunner) -> None:
        """Test CLI version output."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_cli_no_command_shows_help(self, runner: CliRunner) -> None:
        """Test CLI with no command shows help."""
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert "ATP - Agent Test Platform CLI" in result.output


class TestVersionCommand:
    """Tests for version command."""

    def test_version_command(self, runner: CliRunner) -> None:
        """Test version command output."""
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert f"v{__version__}" in result.output
        assert "Python:" in result.output
        assert "Platform:" in result.output

    def test_version_help(self, runner: CliRunner) -> None:
        """Test version command help."""
        result = runner.invoke(cli, ["version", "--help"])
        assert result.exit_code == 0
        assert "Show ATP version information" in result.output


class TestTestCommand:
    """Tests for test command."""

    def test_test_help(self, runner: CliRunner) -> None:
        """Test test command help."""
        result = runner.invoke(cli, ["test", "--help"])
        assert result.exit_code == 0
        assert "Run tests from a test suite file" in result.output
        assert "--parallel" in result.output
        assert "--tags" in result.output
        assert "--list-only" in result.output
        assert "--adapter" in result.output
        assert "--runs" in result.output
        assert "--fail-fast" in result.output
        assert "--sandbox" in result.output
        assert "--verbose" in result.output
        assert "Exit Codes:" in result.output

    def test_test_list_only(
        self, runner: CliRunner, sample_suite_content: str, tmp_path: Path
    ) -> None:
        """Test test --list-only shows tests without running."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        result = runner.invoke(cli, ["test", str(suite_file), "--list-only"])
        assert result.exit_code == 0
        assert "Test Suite: test-suite" in result.output
        assert "test-001" in result.output
        assert "test-002" in result.output

    def test_test_with_parallel_zero_with_list_only(
        self, runner: CliRunner, sample_suite_content: str, tmp_path: Path
    ) -> None:
        """Test test command with --parallel 0 and --list-only."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        # list-only skips validation since tests don't run
        result = runner.invoke(
            cli, ["test", str(suite_file), "--parallel", "0", "--list-only"]
        )
        # list-only should work regardless of parallel value
        assert result.exit_code == 0

    def test_test_with_runs_zero_with_list_only(
        self, runner: CliRunner, sample_suite_content: str, tmp_path: Path
    ) -> None:
        """Test test command with --runs 0 and --list-only."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        # list-only skips validation since tests don't run
        result = runner.invoke(
            cli, ["test", str(suite_file), "--runs", "0", "--list-only"]
        )
        # list-only should work regardless of runs value
        assert result.exit_code == 0

    def test_test_no_matching_tests(
        self, runner: CliRunner, sample_suite_content: str, tmp_path: Path
    ) -> None:
        """Test test command with no matching tests."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        result = runner.invoke(cli, ["test", str(suite_file), "--tags", "nonexistent"])
        assert result.exit_code == EXIT_FAILURE
        assert "No tests match" in result.output

    def test_test_nonexistent_file(self, runner: CliRunner) -> None:
        """Test test command with nonexistent file."""
        result = runner.invoke(cli, ["test", "nonexistent.yaml"])
        assert result.exit_code == 2
        assert "does not exist" in result.output or "Error" in result.output


class TestRunCommand:
    """Tests for run command (backward compatibility alias)."""

    def test_run_help(self, runner: CliRunner) -> None:
        """Test run command help."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run tests from a test suite file" in result.output
        assert (
            "alias" in result.output.lower()
            or "backward compatibility" in result.output.lower()
        )

    def test_run_with_parallel_option_in_help(self, runner: CliRunner) -> None:
        """Test --parallel option is documented in help."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--parallel" in result.output
        assert "Maximum number of tests to run in parallel" in result.output

    def test_run_list_only(
        self, runner: CliRunner, sample_suite_content: str, tmp_path: Path
    ) -> None:
        """Test run --list-only shows tests without running."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        result = runner.invoke(cli, ["run", str(suite_file), "--list-only"])
        assert result.exit_code == 0
        assert "Test Suite: test-suite" in result.output
        assert "test-001" in result.output
        assert "test-002" in result.output

    def test_run_with_parallel_zero_with_list_only(
        self, runner: CliRunner, sample_suite_content: str, tmp_path: Path
    ) -> None:
        """Test run command with --parallel 0 and --list-only."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        # list-only skips validation since tests don't run
        result = runner.invoke(
            cli, ["run", str(suite_file), "--parallel", "0", "--list-only"]
        )
        # list-only should work regardless of parallel value
        assert result.exit_code == 0

    def test_run_with_parallel_negative_value(
        self, runner: CliRunner, sample_suite_content: str, tmp_path: Path
    ) -> None:
        """Test run command with negative --parallel value."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        result = runner.invoke(
            cli, ["run", str(suite_file), "--parallel", "-1", "--list-only"]
        )
        # Negative values are checked before list-only
        assert result.exit_code == EXIT_FAILURE
        assert "--parallel must be at least 1" in result.output

    def test_run_with_runs_zero_with_list_only(
        self, runner: CliRunner, sample_suite_content: str, tmp_path: Path
    ) -> None:
        """Test run command with --runs 0 and --list-only."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        # list-only skips validation since tests don't run
        result = runner.invoke(
            cli, ["run", str(suite_file), "--runs", "0", "--list-only"]
        )
        # list-only should work regardless of runs value
        assert result.exit_code == 0

    def test_run_no_matching_tests(
        self, runner: CliRunner, sample_suite_content: str, tmp_path: Path
    ) -> None:
        """Test run command with no matching tests."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        result = runner.invoke(cli, ["run", str(suite_file), "--tags", "nonexistent"])
        assert result.exit_code == EXIT_FAILURE
        assert "No tests match" in result.output

    def test_run_nonexistent_file(self, runner: CliRunner) -> None:
        """Test run command with nonexistent file."""
        result = runner.invoke(cli, ["run", "nonexistent.yaml"])
        assert result.exit_code == 2
        assert "does not exist" in result.output or "Error" in result.output


class TestRunCommandWithMockAdapter:
    """Tests for run command with mocked adapter execution."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter that returns success."""
        from atp.protocol import ATPEvent, ATPRequest, ATPResponse, ResponseStatus

        async def mock_stream_events(
            request: ATPRequest,
        ) -> AsyncIterator[ATPEvent | ATPResponse]:
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
            )

        adapter = MagicMock()
        adapter.stream_events = mock_stream_events
        adapter.execute = AsyncMock(
            return_value=ATPResponse(
                task_id="test",
                status=ResponseStatus.COMPLETED,
            )
        )
        adapter.cleanup = AsyncMock()
        return adapter

    def test_run_sequential_execution(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test run command with sequential execution."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter):
            result = runner.invoke(cli, ["run", str(suite_file), "--parallel", "1"])

        # Check exit code (0 = success)
        assert result.exit_code == EXIT_SUCCESS

    def test_run_parallel_execution(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test run command with parallel execution."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter):
            result = runner.invoke(cli, ["run", str(suite_file), "--parallel", "4"])

        assert result.exit_code == EXIT_SUCCESS

    def test_run_with_tag_filter(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test run command with tag filter."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter):
            result = runner.invoke(cli, ["run", str(suite_file), "--tags", "smoke"])

        assert result.exit_code == EXIT_SUCCESS

    def test_run_with_multiple_runs(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test run command with multiple runs per test."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter):
            result = runner.invoke(cli, ["run", str(suite_file), "--runs", "3"])

        assert result.exit_code == EXIT_SUCCESS

    def test_run_with_fail_fast(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test run command with fail-fast option."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter):
            result = runner.invoke(cli, ["run", str(suite_file), "--fail-fast"])

        assert result.exit_code == EXIT_SUCCESS

    def test_run_with_sandbox(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test run command with sandbox enabled."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter):
            result = runner.invoke(cli, ["run", str(suite_file), "--sandbox"])

        # May fail due to sandbox issues, but should not error on the option itself
        # Exit code could be 0 or 1 depending on sandbox behavior
        assert result.exit_code in (EXIT_SUCCESS, EXIT_FAILURE)

    def test_run_with_verbose(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test run command with verbose output."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter):
            result = runner.invoke(cli, ["run", str(suite_file), "--verbose"])

        assert result.exit_code == EXIT_SUCCESS

    def test_run_with_adapter_config(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test run command with adapter configuration."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter) as mock:
            result = runner.invoke(
                cli,
                [
                    "run",
                    str(suite_file),
                    "--adapter",
                    "http",
                    "--adapter-config",
                    "base_url=http://localhost:8080",
                    "--adapter-config",
                    "timeout=30",
                ],
            )

        assert result.exit_code == EXIT_SUCCESS
        # Verify adapter was created with config
        mock.assert_called_once()
        call_args = mock.call_args
        assert call_args[0][0] == "http"
        assert "base_url" in call_args[0][1]
        assert call_args[0][1]["base_url"] == "http://localhost:8080"

    def test_run_with_json_adapter_config(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test run command with JSON adapter configuration."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter) as mock:
            result = runner.invoke(
                cli,
                [
                    "run",
                    str(suite_file),
                    "--adapter-config",
                    'headers={"Authorization": "Bearer token"}',
                ],
            )

        assert result.exit_code == EXIT_SUCCESS
        call_args = mock.call_args
        assert "headers" in call_args[0][1]
        assert call_args[0][1]["headers"]["Authorization"] == "Bearer token"


class TestRunCommandFailures:
    """Tests for run command failure scenarios."""

    @pytest.fixture
    def failing_adapter(self):
        """Create a mock adapter that returns failures."""
        from atp.protocol import ATPEvent, ATPRequest, ATPResponse, ResponseStatus

        async def mock_stream_events(
            request: ATPRequest,
        ) -> AsyncIterator[ATPEvent | ATPResponse]:
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error="Test failed",
            )

        adapter = MagicMock()
        adapter.stream_events = mock_stream_events
        adapter.execute = AsyncMock(
            return_value=ATPResponse(
                task_id="test",
                status=ResponseStatus.FAILED,
                error="Test failed",
            )
        )
        adapter.cleanup = AsyncMock()
        return adapter

    def test_run_with_test_failure(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        failing_adapter: MagicMock,
    ) -> None:
        """Test run command returns non-zero on test failures."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=failing_adapter):
            result = runner.invoke(cli, ["run", str(suite_file)])

        # Should exit with 1 due to test failures
        assert result.exit_code == EXIT_FAILURE


class TestValidateCommand:
    """Tests for validate command."""

    def test_validate_help(self, runner: CliRunner) -> None:
        """Test validate command help."""
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate agent configuration or test suite" in result.output
        assert "--agent" in result.output
        assert "--adapter" in result.output
        assert "--suite" in result.output
        assert "Exit Codes:" in result.output

    def test_validate_nothing_specified(self, runner: CliRunner) -> None:
        """Test validate command with nothing specified."""
        result = runner.invoke(cli, ["validate"])
        assert result.exit_code == EXIT_ERROR
        assert "Nothing to validate" in result.output

    def test_validate_suite_valid(
        self, runner: CliRunner, sample_suite_content: str, tmp_path: Path
    ) -> None:
        """Test validate command with valid suite."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        result = runner.invoke(cli, ["validate", "--suite", str(suite_file)])
        assert result.exit_code == EXIT_SUCCESS
        assert "Suite 'test-suite' is valid" in result.output
        assert "Version: 1.0" in result.output

    def test_validate_suite_invalid(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test validate command with invalid suite."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text("invalid: yaml content: [")

        result = runner.invoke(cli, ["validate", "--suite", str(suite_file)])
        assert result.exit_code == EXIT_FAILURE
        assert "Suite validation failed" in result.output

    def test_validate_adapter_unknown_type(self, runner: CliRunner) -> None:
        """Test validate command with unknown adapter type."""
        result = runner.invoke(cli, ["validate", "--adapter", "unknown"])
        assert result.exit_code == EXIT_FAILURE
        assert "Unknown adapter type" in result.output

    def test_validate_adapter_http(self, runner: CliRunner) -> None:
        """Test validate command with http adapter."""
        result = runner.invoke(
            cli,
            [
                "validate",
                "--adapter",
                "http",
                "--adapter-config",
                "endpoint=http://localhost:8000",
            ],
        )
        assert result.exit_code == EXIT_SUCCESS
        assert "Adapter configuration is valid" in result.output

    def test_validate_agent_not_found(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test validate command with agent not in config."""
        result = runner.invoke(cli, ["validate", "--agent", "nonexistent"])
        assert result.exit_code == EXIT_FAILURE
        assert "not found in config" in result.output


class TestListAgentsCommand:
    """Tests for list-agents command."""

    def test_list_agents_help(self, runner: CliRunner) -> None:
        """Test list-agents command help."""
        result = runner.invoke(cli, ["list-agents", "--help"])
        assert result.exit_code == 0
        assert "List available adapter types" in result.output
        assert "--verbose" in result.output

    def test_list_agents_basic(self, runner: CliRunner) -> None:
        """Test list-agents command basic output."""
        result = runner.invoke(cli, ["list-agents"])
        assert result.exit_code == 0
        assert "Available Adapter Types:" in result.output
        assert "http" in result.output
        assert "container" in result.output
        assert "cli" in result.output
        assert "langgraph" in result.output
        assert "crewai" in result.output
        assert "autogen" in result.output

    def test_list_agents_verbose(self, runner: CliRunner) -> None:
        """Test list-agents command with verbose output."""
        result = runner.invoke(cli, ["list-agents", "--verbose"])
        assert result.exit_code == 0
        assert "Available Adapter Types:" in result.output
        # Verbose should show field information
        assert "Required:" in result.output or "Optional:" in result.output

    def test_list_agents_with_config(
        self,
        runner: CliRunner,
        sample_config_content: str,
        tmp_path: Path,
    ) -> None:
        """Test list-agents command with config file."""
        config_file = tmp_path / "atp.config.yaml"
        config_file.write_text(sample_config_content)

        result = runner.invoke(cli, ["--config", str(config_file), "list-agents"])
        assert result.exit_code == 0
        assert "Available Adapter Types:" in result.output
        assert "Configured Agents:" in result.output
        assert "my-http-agent" in result.output


class TestListCommand:
    """Tests for list command."""

    def test_list_help(self, runner: CliRunner) -> None:
        """Test list command help."""
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0
        assert "List tests in a test suite" in result.output

    def test_list_tests(
        self, runner: CliRunner, sample_suite_content: str, tmp_path: Path
    ) -> None:
        """Test list command shows all tests."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        result = runner.invoke(cli, ["list", str(suite_file)])
        assert result.exit_code == 0
        assert "Test Suite: test-suite" in result.output
        assert "Version: 1.0" in result.output
        assert "test-001" in result.output
        assert "test-002" in result.output
        assert "smoke" in result.output
        assert "integration" in result.output

    def test_list_with_tag_filter(
        self, runner: CliRunner, sample_suite_content: str, tmp_path: Path
    ) -> None:
        """Test list command with tag filter."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        result = runner.invoke(cli, ["list", str(suite_file), "--tags", "smoke"])
        assert result.exit_code == 0
        assert "test-001" in result.output
        assert "test-002" not in result.output


class TestBaselineCommands:
    """Tests for baseline commands."""

    def test_baseline_help(self, runner: CliRunner) -> None:
        """Test baseline command help."""
        result = runner.invoke(cli, ["baseline", "--help"])
        assert result.exit_code == 0
        assert "Manage test baselines" in result.output
        assert "save" in result.output
        assert "compare" in result.output

    def test_baseline_save_help(self, runner: CliRunner) -> None:
        """Test baseline save command help."""
        result = runner.invoke(cli, ["baseline", "save", "--help"])
        assert result.exit_code == 0
        assert "Run tests and save results as a baseline" in result.output
        assert "--output" in result.output
        assert "--runs" in result.output

    def test_baseline_compare_help(self, runner: CliRunner) -> None:
        """Test baseline compare command help."""
        result = runner.invoke(cli, ["baseline", "compare", "--help"])
        assert result.exit_code == 0
        assert "compare results against a baseline" in result.output
        assert "--baseline" in result.output
        assert "--fail-on-regression" in result.output


class TestParallelExecutionIntegration:
    """Integration tests for parallel execution options."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter."""
        from atp.protocol import ATPEvent, ATPRequest, ATPResponse, ResponseStatus

        async def mock_stream_events(
            request: ATPRequest,
        ) -> AsyncIterator[ATPEvent | ATPResponse]:
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
            )

        adapter = MagicMock()
        adapter.stream_events = mock_stream_events
        adapter.cleanup = AsyncMock()
        return adapter

    def test_parallel_with_fail_fast_runs_sequential(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test that --parallel with --fail-fast still works (runs sequentially)."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter):
            # fail-fast takes precedence and disables parallel
            result = runner.invoke(
                cli,
                ["run", str(suite_file), "--parallel", "4", "--fail-fast"],
            )

        assert result.exit_code == EXIT_SUCCESS

    def test_all_options_combined(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test running with all options combined."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter):
            result = runner.invoke(
                cli,
                [
                    "run",
                    str(suite_file),
                    "--parallel",
                    "2",
                    "--runs",
                    "2",
                    "--tags",
                    "smoke,integration",
                    "--agent-name",
                    "test-agent",
                    "--verbose",
                ],
            )

        assert result.exit_code == EXIT_SUCCESS


class TestTestCommandWithMockAdapter:
    """Tests for test command with mocked adapter execution."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter that returns success."""
        from atp.protocol import ATPEvent, ATPRequest, ATPResponse, ResponseStatus

        async def mock_stream_events(
            request: ATPRequest,
        ) -> AsyncIterator[ATPEvent | ATPResponse]:
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
            )

        adapter = MagicMock()
        adapter.stream_events = mock_stream_events
        adapter.execute = AsyncMock(
            return_value=ATPResponse(
                task_id="test",
                status=ResponseStatus.COMPLETED,
            )
        )
        adapter.cleanup = AsyncMock()
        return adapter

    def test_test_command_execution(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test 'test' command executes correctly."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter):
            result = runner.invoke(cli, ["test", str(suite_file)])

        assert result.exit_code == EXIT_SUCCESS

    def test_test_command_with_agent_option(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test 'test' command with --agent option."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter):
            result = runner.invoke(
                cli, ["test", str(suite_file), "--agent", "my-agent"]
            )

        assert result.exit_code == EXIT_SUCCESS

    def test_test_command_with_config_file_agent(
        self,
        runner: CliRunner,
        sample_suite_content: str,
        sample_config_content: str,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test 'test' command uses agent from config file."""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(sample_suite_content)

        config_file = tmp_path / "atp.config.yaml"
        config_file.write_text(sample_config_content)

        with patch("atp.adapters.create_adapter", return_value=mock_adapter) as mock:
            result = runner.invoke(
                cli,
                [
                    "--config",
                    str(config_file),
                    "test",
                    str(suite_file),
                    "--agent",
                    "my-http-agent",
                ],
            )

        assert result.exit_code == EXIT_SUCCESS
        # Verify the adapter was created with config from file
        mock.assert_called_once()
        call_args = mock.call_args
        # Agent type should be "http" from config
        assert call_args[0][0] == "http"
        # Config should include endpoint from config file
        assert "endpoint" in call_args[0][1]
