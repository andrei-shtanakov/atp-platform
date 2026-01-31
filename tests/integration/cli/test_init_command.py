"""Integration tests for the init command."""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from atp.cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner."""
    return CliRunner()


class TestInitCommandHelp:
    """Tests for init command help and basic options."""

    def test_init_help(self, runner: CliRunner) -> None:
        """Test init command help output."""
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "Initialize a new ATP test suite" in result.output
        assert "--output" in result.output
        assert "--interactive" in result.output
        assert "--no-interactive" in result.output
        assert "Examples:" in result.output
        assert "Exit Codes:" in result.output

    def test_init_in_cli_help(self, runner: CliRunner) -> None:
        """Test init command appears in main CLI help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.output


class TestInitCommandNonInteractive:
    """Tests for init command in non-interactive mode."""

    def test_init_non_interactive_default_output(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init with --no-interactive creates file with default name."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--no-interactive"])

            assert result.exit_code == 0
            assert "Test suite created:" in result.output
            assert "my-test-suite.yaml" in result.output

            # Verify file was created
            output_file = Path("my-test-suite.yaml")
            assert output_file.exists()

            # Verify YAML content
            with open(output_file) as f:
                suite_data = yaml.safe_load(f)

            assert suite_data["test_suite"] == "my-test-suite"
            assert suite_data["version"] == "1.0"
            assert "tests" in suite_data
            assert len(suite_data["tests"]) >= 1

    def test_init_non_interactive_custom_output(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init with --no-interactive and custom output path."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli, ["init", "--no-interactive", "-o", "custom-suite.yaml"]
            )

            assert result.exit_code == 0
            assert "custom-suite.yaml" in result.output

            output_file = Path("custom-suite.yaml")
            assert output_file.exists()

    def test_init_non_interactive_creates_sample_test(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init with --no-interactive creates a sample test."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--no-interactive"])

            assert result.exit_code == 0

            with open("my-test-suite.yaml") as f:
                suite_data = yaml.safe_load(f)

            # Should have a sample test
            assert len(suite_data["tests"]) == 1
            test = suite_data["tests"][0]
            assert test["id"] == "test-001"
            assert test["name"] == "Sample Test"
            assert "sample" in test["tags"]

    def test_init_non_interactive_nested_output_path(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init creates parent directories for output path."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli, ["init", "--no-interactive", "-o", "tests/suites/my-suite.yaml"]
            )

            assert result.exit_code == 0
            output_file = Path("tests/suites/my-suite.yaml")
            assert output_file.exists()


class TestInitCommandInteractive:
    """Tests for init command in interactive mode."""

    def test_init_interactive_basic_suite(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init with interactive prompts for basic suite."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Simulate interactive input:
            # suite name, description, runs_per_test, timeout,
            # no agent, no test, (sample test added automatically)
            user_input = "\n".join(
                [
                    "my-custom-suite",  # Suite name
                    "A test suite",  # Description
                    "3",  # Runs per test
                    "120",  # Timeout
                    "n",  # Add agent? No
                    "n",  # Add test? No
                ]
            )

            result = runner.invoke(cli, ["init", "-o", "output.yaml"], input=user_input)

            assert result.exit_code == 0
            assert "Test suite created:" in result.output

            with open("output.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert suite_data["test_suite"] == "my-custom-suite"
            assert suite_data["description"] == "A test suite"

    def test_init_interactive_with_http_agent(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init with interactive prompts including HTTP agent config."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Simulate interactive input for HTTP agent
            user_input = "\n".join(
                [
                    "agent-test-suite",  # Suite name
                    "",  # Description (empty)
                    "1",  # Runs per test
                    "300",  # Timeout
                    "y",  # Add agent? Yes
                    "my-agent",  # Agent name
                    "http",  # Agent type
                    "http://localhost:8080",  # Endpoint
                    "n",  # Configure optional? No
                    "n",  # Add test? No
                ]
            )

            result = runner.invoke(cli, ["init", "-o", "output.yaml"], input=user_input)

            assert result.exit_code == 0

            with open("output.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert suite_data["test_suite"] == "agent-test-suite"
            assert "agents" in suite_data
            assert len(suite_data["agents"]) == 1
            agent = suite_data["agents"][0]
            assert agent["name"] == "my-agent"
            assert agent["type"] == "http"
            assert agent["config"]["endpoint"] == "http://localhost:8080"

    def test_init_interactive_with_cli_agent(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init with interactive prompts including CLI agent config."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            user_input = "\n".join(
                [
                    "cli-test-suite",  # Suite name
                    "",  # Description
                    "1",  # Runs per test
                    "300",  # Timeout
                    "y",  # Add agent? Yes
                    "cli-agent",  # Agent name
                    "cli",  # Agent type
                    "python agent.py",  # Command
                    "n",  # Configure optional? No
                    "n",  # Add test? No
                ]
            )

            result = runner.invoke(cli, ["init", "-o", "output.yaml"], input=user_input)

            assert result.exit_code == 0

            with open("output.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert "agents" in suite_data
            agent = suite_data["agents"][0]
            assert agent["type"] == "cli"
            assert agent["config"]["command"] == "python agent.py"

    def test_init_interactive_with_container_agent(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init with interactive prompts including container agent config."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            user_input = "\n".join(
                [
                    "container-test-suite",  # Suite name
                    "",  # Description
                    "1",  # Runs per test
                    "300",  # Timeout
                    "y",  # Add agent? Yes
                    "docker-agent",  # Agent name
                    "container",  # Agent type
                    "my-agent:latest",  # Image
                    "n",  # Configure optional? No
                    "n",  # Add test? No
                ]
            )

            result = runner.invoke(cli, ["init", "-o", "output.yaml"], input=user_input)

            assert result.exit_code == 0

            with open("output.yaml") as f:
                suite_data = yaml.safe_load(f)

            agent = suite_data["agents"][0]
            assert agent["type"] == "container"
            assert agent["config"]["image"] == "my-agent:latest"

    def test_init_interactive_with_custom_test(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init with interactive prompts for custom test creation."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            user_input = "\n".join(
                [
                    "custom-test-suite",  # Suite name
                    "",  # Description
                    "1",  # Runs per test
                    "300",  # Timeout
                    "n",  # Add agent? No
                    "y",  # Add test? Yes
                    "2",  # Choice: custom test
                    "custom-001",  # Test ID
                    "My Custom Test",  # Test name
                    "Do something useful",  # Task description
                    "smoke,custom",  # Tags
                    "n",  # Configure constraints? No
                    "n",  # Add assertions? No
                    "n",  # Add another test? No
                ]
            )

            result = runner.invoke(cli, ["init", "-o", "output.yaml"], input=user_input)

            assert result.exit_code == 0

            with open("output.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert len(suite_data["tests"]) == 1
            test = suite_data["tests"][0]
            assert test["id"] == "custom-001"
            assert test["name"] == "My Custom Test"
            assert test["task"]["description"] == "Do something useful"
            assert "smoke" in test["tags"]
            assert "custom" in test["tags"]

    def test_init_interactive_with_template_test(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init with interactive prompts for template-based test."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            user_input = "\n".join(
                [
                    "template-test-suite",  # Suite name
                    "",  # Description
                    "1",  # Runs per test
                    "300",  # Timeout
                    "n",  # Add agent? No
                    "y",  # Add test? Yes
                    "1",  # Choice: template
                    "1",  # Template: file_creation
                    "test.txt",  # filename variable
                    "Hello World",  # content variable
                    "template-001",  # Test ID
                    "File Creation Test",  # Test name
                    "file,test",  # Extra tags
                    "n",  # Add another test? No
                ]
            )

            result = runner.invoke(cli, ["init", "-o", "output.yaml"], input=user_input)

            assert result.exit_code == 0

            with open("output.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert len(suite_data["tests"]) == 1
            test = suite_data["tests"][0]
            assert test["id"] == "template-001"
            assert "test.txt" in test["task"]["description"]
            assert "Hello World" in test["task"]["description"]

    def test_init_interactive_with_assertions(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init with interactive prompts for test with assertions."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            user_input = "\n".join(
                [
                    "assertion-test-suite",  # Suite name
                    "",  # Description
                    "1",  # Runs per test
                    "300",  # Timeout
                    "n",  # Add agent? No
                    "y",  # Add test? Yes
                    "2",  # Choice: custom test
                    "assert-001",  # Test ID
                    "Assertion Test",  # Test name
                    "Create a file",  # Task description
                    "",  # Tags (empty)
                    "n",  # Configure constraints? No
                    "y",  # Add assertions? Yes
                    "1",  # artifact_exists
                    "output.txt",  # Path
                    "4",  # Done adding assertions
                    "n",  # Add another test? No
                ]
            )

            result = runner.invoke(cli, ["init", "-o", "output.yaml"], input=user_input)

            assert result.exit_code == 0

            with open("output.yaml") as f:
                suite_data = yaml.safe_load(f)

            test = suite_data["tests"][0]
            assert "assertions" in test
            assert len(test["assertions"]) == 1
            assert test["assertions"][0]["type"] == "artifact_exists"
            assert test["assertions"][0]["config"]["path"] == "output.txt"

    def test_init_interactive_multiple_tests(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init with multiple tests added interactively."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            user_input = "\n".join(
                [
                    "multi-test-suite",  # Suite name
                    "",  # Description
                    "1",  # Runs per test
                    "300",  # Timeout
                    "n",  # Add agent? No
                    "y",  # Add test? Yes
                    "2",  # Choice: custom
                    "test-001",  # Test ID
                    "First Test",  # Test name
                    "First task",  # Task
                    "",  # Tags
                    "n",  # Constraints? No
                    "n",  # Assertions? No
                    "y",  # Add another test? Yes
                    "2",  # Choice: custom
                    "test-002",  # Test ID
                    "Second Test",  # Test name
                    "Second task",  # Task
                    "",  # Tags
                    "n",  # Constraints? No
                    "n",  # Assertions? No
                    "n",  # Add another test? No
                ]
            )

            result = runner.invoke(cli, ["init", "-o", "output.yaml"], input=user_input)

            assert result.exit_code == 0

            with open("output.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert len(suite_data["tests"]) == 2
            assert suite_data["tests"][0]["id"] == "test-001"
            assert suite_data["tests"][1]["id"] == "test-002"


class TestInitCommandAbort:
    """Tests for init command abort scenarios."""

    def test_init_abort_on_keyboard_interrupt(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init handles keyboard interrupt gracefully."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Send empty input to trigger abort
            result = runner.invoke(cli, ["init"], input="\x03")  # Ctrl+C

            # Should handle abort gracefully
            assert result.exit_code == 2
            assert "cancelled" in result.output.lower() or result.exit_code == 2


class TestInitCommandOutputFormat:
    """Tests for init command output format."""

    def test_init_shows_instructions(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test init shows instructions for running tests."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--no-interactive"])

            assert result.exit_code == 0
            assert "Run tests with:" in result.output
            assert "atp test" in result.output

    def test_init_shows_test_count(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test init shows number of tests created."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--no-interactive"])

            assert result.exit_code == 0
            assert "Tests:" in result.output

    def test_init_yaml_valid_format(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test init creates valid YAML that can be loaded."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--no-interactive"])

            assert result.exit_code == 0

            # Verify YAML is valid and can be loaded
            with open("my-test-suite.yaml") as f:
                suite_data = yaml.safe_load(f)

            # Basic structure validation
            assert "test_suite" in suite_data
            assert "version" in suite_data
            assert "tests" in suite_data
            assert isinstance(suite_data["tests"], list)
            assert len(suite_data["tests"]) > 0

            # Each test should have required fields
            for test in suite_data["tests"]:
                assert "id" in test
                assert "name" in test
                assert "task" in test
                assert "description" in test["task"]


class TestInitCommandWithConstraints:
    """Tests for init command with test constraints."""

    def test_init_with_test_constraints(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init with custom test constraints."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            user_input = "\n".join(
                [
                    "constraints-suite",  # Suite name
                    "",  # Description
                    "1",  # Runs per test
                    "300",  # Timeout
                    "n",  # Add agent? No
                    "y",  # Add test? Yes
                    "2",  # Choice: custom
                    "const-001",  # Test ID
                    "Constrained Test",  # Test name
                    "Task with limits",  # Task
                    "",  # Tags
                    "y",  # Configure constraints? Yes
                    "10",  # Max steps
                    "60",  # Timeout
                    "n",  # Assertions? No
                    "n",  # Add another test? No
                ]
            )

            result = runner.invoke(cli, ["init", "-o", "output.yaml"], input=user_input)

            assert result.exit_code == 0

            with open("output.yaml") as f:
                suite_data = yaml.safe_load(f)

            test = suite_data["tests"][0]
            assert "constraints" in test
            assert test["constraints"]["max_steps"] == 10
            assert test["constraints"]["timeout_seconds"] == 60
