"""End-to-end tests for ATP CLI commands.

These tests verify the complete CLI workflow including:
- `atp test` command with various scenarios
- `atp validate` command
- JSON report generation
- Exit codes verification
"""

import json
from pathlib import Path

from click.testing import CliRunner

from atp.cli.main import EXIT_ERROR, EXIT_FAILURE, EXIT_SUCCESS, cli

from .conftest import MockAgentServer


class TestAtpTestHappyPath:
    """Test `atp test` command happy path scenarios."""

    def test_atp_test_with_successful_agent(
        self,
        mock_server_url: str,
        valid_suite_path: Path,
    ) -> None:
        """Test running a test suite with a successful agent."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(valid_suite_path),
                "--adapter=http",
                f"--adapter-config=endpoint={mock_server_url}/agent",
                "--adapter-config=allow_internal=true",
                "--agent=test-agent",
            ],
        )

        # Check successful execution
        assert result.exit_code == EXIT_SUCCESS, f"Output: {result.output}"
        # Should see pass indication
        assert "PASSED" in result.output or "passed" in result.output.lower()

    def test_atp_test_list_only(
        self,
        valid_suite_path: Path,
    ) -> None:
        """Test --list-only flag to list tests without running."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(valid_suite_path),
                "--list-only",
            ],
        )

        assert result.exit_code == 0, f"Output: {result.output}"
        # Should list test info
        assert "Test Suite" in result.output
        assert "e2e-test-001" in result.output
        assert "e2e-test-002" in result.output

    def test_atp_test_with_tags_filter(
        self,
        mock_server_url: str,
        valid_suite_path: Path,
    ) -> None:
        """Test --tags filter to run specific tests."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(valid_suite_path),
                "--tags=smoke",
                "--adapter=http",
                f"--adapter-config=endpoint={mock_server_url}/agent",
                "--adapter-config=allow_internal=true",
            ],
        )

        # Should execute successfully (only smoke tests)
        assert result.exit_code == EXIT_SUCCESS, f"Output: {result.output}"


class TestAtpTestFailingTests:
    """Test `atp test` command with failing tests."""

    def test_atp_test_with_failing_agent(
        self,
        mock_server_url: str,
        failing_suite_path: Path,
    ) -> None:
        """Test running a test suite with a failing agent."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(failing_suite_path),
                "--adapter=http",
                f"--adapter-config=endpoint={mock_server_url}/agent/fail",
                "--adapter-config=allow_internal=true",
            ],
        )

        # Should exit with failure code
        assert result.exit_code == EXIT_FAILURE, f"Output: {result.output}"
        # Should indicate test failures
        assert "FAILED" in result.output or "failed" in result.output.lower()

    def test_atp_test_fail_fast(
        self,
        mock_server_url: str,
        failing_suite_path: Path,
    ) -> None:
        """Test --fail-fast stops on first failure."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(failing_suite_path),
                "--adapter=http",
                f"--adapter-config=endpoint={mock_server_url}/agent/fail",
                "--adapter-config=allow_internal=true",
                "--fail-fast",
            ],
        )

        # Should exit with failure code
        assert result.exit_code == EXIT_FAILURE, f"Output: {result.output}"


class TestAtpTestTimeout:
    """Test `atp test` command timeout scenarios."""

    def test_atp_test_with_timeout(
        self,
        mock_agent_server: MockAgentServer,
        timeout_suite_path: Path,
    ) -> None:
        """Test running a test suite where tests timeout."""
        # Configure server to have a delay longer than test timeout
        mock_agent_server.set_delay(5.0)  # 5 second delay
        server_url = mock_agent_server.url  # type: ignore[attr-defined]

        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(timeout_suite_path),
                "--adapter=http",
                f"--adapter-config=endpoint={server_url}/agent/slow",
                "--adapter-config=allow_internal=true",
                # Adapter timeout also set short
                "--adapter-config=timeout_seconds=1",
            ],
        )

        # Should exit with failure code due to timeout
        assert result.exit_code == EXIT_FAILURE, f"Output: {result.output}"
        # Should indicate timeout in output
        assert "timeout" in result.output.lower() or "FAILED" in result.output, (
            f"Output: {result.output}"
        )

        # Reset delay for other tests
        mock_agent_server.set_delay(0.0)


class TestAtpValidate:
    """Test `atp validate` command."""

    def test_validate_valid_suite(
        self,
        valid_suite_path: Path,
    ) -> None:
        """Test validating a valid test suite."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "validate",
                f"--suite={valid_suite_path}",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS, f"Output: {result.output}"
        assert "valid" in result.output.lower()

    def test_validate_adapter_config(
        self,
        mock_server_url: str,
    ) -> None:
        """Test validating adapter configuration."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "validate",
                "--adapter=http",
                f"--adapter-config=endpoint={mock_server_url}/agent",
                "--adapter-config=allow_internal=true",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS, f"Output: {result.output}"
        assert "valid" in result.output.lower()

    def test_validate_agent_with_health_check(
        self,
        mock_server_url: str,
        tmp_path: Path,
    ) -> None:
        """Test validating agent with health check."""
        # Create a temporary config file
        config_file = tmp_path / "atp.config.yaml"
        config_content = f"""
version: "1.0"
agents:
  test-agent:
    type: http
    endpoint: "{mock_server_url}/agent"
    health_endpoint: "{mock_server_url}/health"
    allow_internal: true
"""
        config_file.write_text(config_content)

        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "--config",
                str(config_file),
                "validate",
                "--agent=test-agent",
                "--verbose",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS, f"Output: {result.output}"
        # Should show health check passed
        assert "healthy" in result.output.lower() or "valid" in result.output.lower()

    def test_validate_invalid_suite_path(self) -> None:
        """Test validating a non-existent suite file."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "validate",
                "--suite=/nonexistent/path/suite.yaml",
            ],
        )

        # Click should catch this before our code
        assert result.exit_code != EXIT_SUCCESS
        assert (
            "does not exist" in result.output.lower()
            or "error" in result.output.lower()
        )

    def test_validate_unknown_adapter_type(self) -> None:
        """Test validating an unknown adapter type."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "validate",
                "--adapter=nonexistent",
            ],
        )

        assert result.exit_code == EXIT_FAILURE, f"Output: {result.output}"
        assert (
            "unknown" in result.output.lower() or "not found" in result.output.lower()
        )


class TestJsonReportGeneration:
    """Test JSON report generation."""

    def test_json_output_to_stdout(
        self,
        mock_server_url: str,
        valid_suite_path: Path,
    ) -> None:
        """Test JSON output format to stdout."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(valid_suite_path),
                "--adapter=http",
                f"--adapter-config=endpoint={mock_server_url}/agent",
                "--adapter-config=allow_internal=true",
                "--output=json",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS, f"Output: {result.output}"

        # The output contains progress bar info followed by JSON
        # Find the JSON part (starts with '{')
        output = result.output
        json_start = output.find("{")
        assert json_start != -1, f"No JSON found in output: {output}"
        json_str = output[json_start:]

        # Parse the JSON output
        json_output = json.loads(json_str)

        # Verify JSON structure
        assert "version" in json_output
        assert "summary" in json_output
        assert "tests" in json_output

        # Verify summary fields
        summary = json_output["summary"]
        assert "suite_name" in summary
        assert "total_tests" in summary
        assert "passed_tests" in summary
        assert "failed_tests" in summary
        assert "success_rate" in summary
        assert summary["success"] is True

    def test_json_output_to_file(
        self,
        mock_server_url: str,
        valid_suite_path: Path,
        tmp_output_dir: Path,
    ) -> None:
        """Test JSON output to file."""
        output_file = tmp_output_dir / "results.json"

        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(valid_suite_path),
                "--adapter=http",
                f"--adapter-config=endpoint={mock_server_url}/agent",
                "--adapter-config=allow_internal=true",
                "--output=json",
                f"--output-file={output_file}",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS, f"Output: {result.output}"

        # Verify file was created
        assert output_file.exists()

        # Parse the JSON file
        json_content = json.loads(output_file.read_text())

        # Verify JSON structure
        assert json_content["version"] == "1.0"
        assert "generated_at" in json_content
        assert len(json_content["tests"]) == 2  # Two tests in valid suite

    def test_json_report_with_failed_tests(
        self,
        mock_server_url: str,
        failing_suite_path: Path,
        tmp_output_dir: Path,
    ) -> None:
        """Test JSON report includes failure information."""
        output_file = tmp_output_dir / "failed_results.json"

        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(failing_suite_path),
                "--adapter=http",
                f"--adapter-config=endpoint={mock_server_url}/agent/fail",
                "--adapter-config=allow_internal=true",
                "--output=json",
                f"--output-file={output_file}",
            ],
        )

        # Should fail but still create JSON report
        assert result.exit_code == EXIT_FAILURE

        # Verify file was created
        assert output_file.exists()

        json_content = json.loads(output_file.read_text())

        # Verify failure is captured
        summary = json_content["summary"]
        assert summary["success"] is False
        assert summary["failed_tests"] > 0


class TestExitCodes:
    """Test exit codes for various scenarios."""

    def test_exit_success_on_all_tests_pass(
        self,
        mock_server_url: str,
        valid_suite_path: Path,
    ) -> None:
        """Test EXIT_SUCCESS (0) when all tests pass."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(valid_suite_path),
                "--adapter=http",
                f"--adapter-config=endpoint={mock_server_url}/agent",
                "--adapter-config=allow_internal=true",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert result.exit_code == 0

    def test_exit_failure_on_test_failures(
        self,
        mock_server_url: str,
        failing_suite_path: Path,
    ) -> None:
        """Test EXIT_FAILURE (1) when tests fail."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(failing_suite_path),
                "--adapter=http",
                f"--adapter-config=endpoint={mock_server_url}/agent/fail",
                "--adapter-config=allow_internal=true",
            ],
        )

        assert result.exit_code == EXIT_FAILURE
        assert result.exit_code == 1

    def test_exit_error_on_invalid_suite_file(self) -> None:
        """Test EXIT_ERROR (2) on configuration/file errors."""
        runner = CliRunner()

        # Non-existent file should give error exit code
        result = runner.invoke(
            cli,
            [
                "test",
                "/nonexistent/suite.yaml",
            ],
        )

        # Click handles non-existent files with exit code 2
        assert result.exit_code != EXIT_SUCCESS, f"Output: {result.output}"

    def test_exit_error_on_invalid_parallel_option(
        self,
        valid_suite_path: Path,
    ) -> None:
        """Test EXIT_ERROR on invalid --parallel value (non-integer)."""
        runner = CliRunner()

        # Test with non-integer value - Click will reject this
        result = runner.invoke(
            cli,
            [
                "test",
                str(valid_suite_path),
                "--parallel=invalid",
            ],
        )

        # Click returns 2 for bad parameter values
        assert result.exit_code == EXIT_ERROR, f"Output: {result.output}"
        assert "invalid" in result.output.lower() or "error" in result.output.lower()

    def test_exit_error_on_invalid_runs_option(
        self,
        valid_suite_path: Path,
    ) -> None:
        """Test EXIT_ERROR on invalid --runs value (non-integer)."""
        runner = CliRunner()

        # Test with non-integer value - Click will reject this
        result = runner.invoke(
            cli,
            [
                "test",
                str(valid_suite_path),
                "--runs=invalid",
            ],
        )

        # Click returns 2 for bad parameter values
        assert result.exit_code == EXIT_ERROR, f"Output: {result.output}"
        assert "invalid" in result.output.lower() or "error" in result.output.lower()

    def test_exit_failure_on_no_matching_tests(
        self,
        valid_suite_path: Path,
    ) -> None:
        """Test EXIT_FAILURE when no tests match filter criteria."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(valid_suite_path),
                "--tags=nonexistent-tag",
            ],
        )

        assert result.exit_code == EXIT_FAILURE, f"Output: {result.output}"
        assert "no tests match" in result.output.lower()

    def test_validate_exit_success(
        self,
        valid_suite_path: Path,
    ) -> None:
        """Test EXIT_SUCCESS from validate command on valid input."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "validate",
                f"--suite={valid_suite_path}",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS

    def test_validate_exit_failure_on_unknown_adapter(self) -> None:
        """Test EXIT_FAILURE from validate on unknown adapter."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "validate",
                "--adapter=unknown_adapter_type",
            ],
        )

        assert result.exit_code == EXIT_FAILURE

    def test_validate_exit_error_on_no_arguments(self) -> None:
        """Test EXIT_ERROR from validate when no arguments provided."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "validate",
            ],
        )

        assert result.exit_code == EXIT_ERROR
        assert "nothing to validate" in result.output.lower()


class TestMultipleRuns:
    """Test multiple runs per test."""

    def test_atp_test_with_multiple_runs(
        self,
        mock_server_url: str,
        valid_suite_path: Path,
        tmp_output_dir: Path,
    ) -> None:
        """Test running tests with multiple runs."""
        output_file = tmp_output_dir / "multi_run_results.json"

        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(valid_suite_path),
                "--adapter=http",
                f"--adapter-config=endpoint={mock_server_url}/agent",
                "--adapter-config=allow_internal=true",
                "--runs=3",
                "--output=json",
                f"--output-file={output_file}",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS, f"Output: {result.output}"

        # Check JSON report
        json_content = json.loads(output_file.read_text())

        # Verify runs_per_test in summary
        assert json_content["summary"]["runs_per_test"] == 3

        # Verify each test has run info
        for test in json_content["tests"]:
            assert "runs" in test
            assert test["runs"]["total"] == 3


class TestParallelExecution:
    """Test parallel test execution."""

    def test_atp_test_parallel_execution(
        self,
        mock_server_url: str,
        valid_suite_path: Path,
    ) -> None:
        """Test running tests in parallel."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "test",
                str(valid_suite_path),
                "--adapter=http",
                f"--adapter-config=endpoint={mock_server_url}/agent",
                "--adapter-config=allow_internal=true",
                "--parallel=2",
            ],
        )

        # Should complete successfully
        assert result.exit_code == EXIT_SUCCESS, f"Output: {result.output}"
