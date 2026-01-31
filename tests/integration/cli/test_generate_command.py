"""Integration tests for the generate command."""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from atp.cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_suite_yaml() -> str:
    """Create sample suite YAML content."""
    return """
test_suite: "existing-suite"
version: "1.0"
description: "An existing test suite"

defaults:
  runs_per_test: 1
  timeout_seconds: 300

tests:
  - id: "existing-001"
    name: "Existing Test"
    tags: ["smoke"]
    task:
      description: "An existing test task"
    constraints:
      max_steps: 5
      timeout_seconds: 60
"""


class TestGenerateCommandHelp:
    """Tests for generate command help and basic options."""

    def test_generate_help(self, runner: CliRunner) -> None:
        """Test generate command help output."""
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "Generate tests and test suites" in result.output
        assert "test" in result.output
        assert "suite" in result.output

    def test_generate_in_cli_help(self, runner: CliRunner) -> None:
        """Test generate command appears in main CLI help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "generate" in result.output

    def test_generate_test_help(self, runner: CliRunner) -> None:
        """Test generate test subcommand help output."""
        result = runner.invoke(cli, ["generate", "test", "--help"])
        assert result.exit_code == 0
        assert "Generate a new test" in result.output
        assert "--suite" in result.output
        assert "--template" in result.output
        assert "--output" in result.output
        assert "--list-templates" in result.output

    def test_generate_suite_help(self, runner: CliRunner) -> None:
        """Test generate suite subcommand help output."""
        result = runner.invoke(cli, ["generate", "suite", "--help"])
        assert result.exit_code == 0
        assert "Generate a new test suite" in result.output
        assert "--output" in result.output
        assert "--template" in result.output
        assert "--count" in result.output
        assert "--interactive" in result.output


class TestGenerateTestSubcommand:
    """Tests for generate test subcommand."""

    def test_generate_test_list_templates(
        self, runner: CliRunner, tmp_path: Path, sample_suite_yaml: str
    ) -> None:
        """Test listing available templates."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create a suite file (required argument)
            suite_file = Path("suite.yaml")
            suite_file.write_text(sample_suite_yaml)

            result = runner.invoke(
                cli, ["generate", "test", "--suite=suite.yaml", "--list-templates"]
            )

            assert result.exit_code == 0
            assert "Available templates:" in result.output
            assert "file_creation" in result.output
            assert "data_processing" in result.output
            assert "web_research" in result.output
            assert "code_generation" in result.output

    def test_generate_test_requires_suite(self, runner: CliRunner) -> None:
        """Test that --suite is required."""
        result = runner.invoke(cli, ["generate", "test"])
        assert result.exit_code == 2
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_generate_test_suite_not_found(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test error when suite file doesn't exist."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli, ["generate", "test", "--suite=nonexistent.yaml"]
            )
            # Click ClickException exits with code 1
            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_generate_test_custom_interactive(
        self, runner: CliRunner, tmp_path: Path, sample_suite_yaml: str
    ) -> None:
        """Test generating a custom test interactively."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            suite_file = Path("suite.yaml")
            suite_file.write_text(sample_suite_yaml)

            user_input = "\n".join(
                [
                    "2",  # Choice: custom test
                    "new-test-001",  # Test ID
                    "New Test",  # Test name
                    "Do something new",  # Task description
                    "new,test",  # Tags
                    "n",  # Configure constraints? No
                    "n",  # Add assertions? No
                ]
            )

            result = runner.invoke(
                cli,
                ["generate", "test", "--suite=suite.yaml"],
                input=user_input,
            )

            assert result.exit_code == 0
            assert "Test 'new-test-001' added" in result.output

            # Verify the test was added
            with open("suite.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert len(suite_data["tests"]) == 2
            new_test = suite_data["tests"][1]
            assert new_test["id"] == "new-test-001"
            assert new_test["name"] == "New Test"
            assert "new" in new_test["tags"]
            assert "test" in new_test["tags"]

    def test_generate_test_with_template(
        self, runner: CliRunner, tmp_path: Path, sample_suite_yaml: str
    ) -> None:
        """Test generating a test using a template."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            suite_file = Path("suite.yaml")
            suite_file.write_text(sample_suite_yaml)

            # Variables are sorted alphabetically: content, filename
            user_input = "\n".join(
                [
                    "Hello World",  # content variable (sorted first)
                    "test.txt",  # filename variable (sorted second)
                    "template-001",  # Test ID
                    "File Creation Test",  # Test name
                    "template",  # Extra tags
                ]
            )

            result = runner.invoke(
                cli,
                ["generate", "test", "--suite=suite.yaml", "--template=file_creation"],
                input=user_input,
            )

            assert result.exit_code == 0
            assert "Test 'template-001' added" in result.output

            with open("suite.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert len(suite_data["tests"]) == 2
            new_test = suite_data["tests"][1]
            assert new_test["id"] == "template-001"
            assert "test.txt" in new_test["task"]["description"]

    def test_generate_test_to_different_output(
        self, runner: CliRunner, tmp_path: Path, sample_suite_yaml: str
    ) -> None:
        """Test generating a test and saving to different file."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            suite_file = Path("suite.yaml")
            suite_file.write_text(sample_suite_yaml)

            user_input = "\n".join(
                [
                    "2",  # Choice: custom test
                    "new-test",  # Test ID
                    "New Test",  # Test name
                    "New task",  # Task description
                    "",  # Tags
                    "n",  # Constraints
                    "n",  # Assertions
                ]
            )

            result = runner.invoke(
                cli,
                [
                    "generate",
                    "test",
                    "--suite=suite.yaml",
                    "-o",
                    "output-suite.yaml",
                ],
                input=user_input,
            )

            assert result.exit_code == 0

            # Original file should be unchanged
            with open("suite.yaml") as f:
                original = yaml.safe_load(f)
            assert len(original["tests"]) == 1

            # New file should have both tests
            with open("output-suite.yaml") as f:
                output = yaml.safe_load(f)
            assert len(output["tests"]) == 2

    def test_generate_test_invalid_template(
        self, runner: CliRunner, tmp_path: Path, sample_suite_yaml: str
    ) -> None:
        """Test error when using invalid template name."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            suite_file = Path("suite.yaml")
            suite_file.write_text(sample_suite_yaml)

            result = runner.invoke(
                cli,
                [
                    "generate",
                    "test",
                    "--suite=suite.yaml",
                    "--template=nonexistent_template",
                ],
            )

            # Click ClickException exits with code 1
            assert result.exit_code == 1
            assert "not found" in result.output.lower()


class TestGenerateSuiteSubcommand:
    """Tests for generate suite subcommand."""

    def test_generate_suite_non_interactive(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test generating a suite in non-interactive mode."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli, ["generate", "suite", "--no-interactive", "-o", "new-suite.yaml"]
            )

            assert result.exit_code == 0
            assert "Test suite created:" in result.output

            with open("new-suite.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert suite_data["test_suite"] == "my-test-suite"
            assert len(suite_data["tests"]) >= 1

    def test_generate_suite_interactive_basic(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test generating a suite interactively with basic options."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            user_input = "\n".join(
                [
                    "my-new-suite",  # Suite name
                    "A new test suite",  # Description
                    "2",  # Runs per test
                    "120",  # Timeout
                    "n",  # Add test? No
                ]
            )

            result = runner.invoke(
                cli,
                ["generate", "suite", "-o", "output.yaml"],
                input=user_input,
            )

            assert result.exit_code == 0
            assert "Test suite created:" in result.output

            with open("output.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert suite_data["test_suite"] == "my-new-suite"
            assert suite_data["description"] == "A new test suite"

    def test_generate_suite_with_custom_test(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test generating a suite with a custom test."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            user_input = "\n".join(
                [
                    "test-suite",  # Suite name
                    "",  # Description
                    "1",  # Runs per test
                    "300",  # Timeout
                    "y",  # Add test? Yes
                    "2",  # Choice: custom test
                    "test-001",  # Test ID
                    "My Test",  # Test name
                    "Do something",  # Task description
                    "smoke",  # Tags
                    "n",  # Constraints
                    "n",  # Assertions
                    "n",  # Add another test? No
                ]
            )

            result = runner.invoke(
                cli,
                ["generate", "suite", "-o", "output.yaml"],
                input=user_input,
            )

            assert result.exit_code == 0

            with open("output.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert len(suite_data["tests"]) == 1
            test = suite_data["tests"][0]
            assert test["id"] == "test-001"
            assert test["name"] == "My Test"
            assert "smoke" in test["tags"]

    def test_generate_suite_with_template_test(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test generating a suite with a template-based test."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            user_input = "\n".join(
                [
                    "template-suite",  # Suite name
                    "",  # Description
                    "1",  # Runs per test
                    "300",  # Timeout
                    "y",  # Add test? Yes
                    "1",  # Choice: template
                    "1",  # Template: file_creation
                    "output.txt",  # filename
                    "Test content",  # content
                    "test-001",  # Test ID
                    "File Test",  # Test name
                    "",  # Extra tags
                    "n",  # Add another test? No
                ]
            )

            result = runner.invoke(
                cli,
                ["generate", "suite", "-o", "output.yaml"],
                input=user_input,
            )

            assert result.exit_code == 0

            with open("output.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert len(suite_data["tests"]) == 1
            test = suite_data["tests"][0]
            assert "output.txt" in test["task"]["description"]
            assert "Test content" in test["task"]["description"]

    def test_generate_suite_batch_mode(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test generating a suite with batch mode (--count)."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Variables are sorted alphabetically: content, filename
            user_input = "\n".join(
                [
                    "batch-suite",  # Suite name
                    "",  # Description
                    "1",  # Runs per test
                    "300",  # Timeout
                    # Test 1
                    "Content 1",  # content (sorted first)
                    "file1.txt",  # filename (sorted second)
                    "Test 1",  # Test name
                    # Test 2
                    "Content 2",  # content (sorted first)
                    "file2.txt",  # filename (sorted second)
                    "Test 2",  # Test name
                ]
            )

            result = runner.invoke(
                cli,
                [
                    "generate",
                    "suite",
                    "-o",
                    "batch.yaml",
                    "--template=file_creation",
                    "--count=2",
                ],
                input=user_input,
            )

            assert result.exit_code == 0
            assert "Generating 2 tests" in result.output

            with open("batch.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert len(suite_data["tests"]) == 2
            assert suite_data["tests"][0]["name"] == "Test 1"
            assert suite_data["tests"][1]["name"] == "Test 2"

    def test_generate_suite_batch_requires_template(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that batch mode requires --template."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            user_input = "\n".join(
                [
                    "batch-suite",  # Suite name
                    "",  # Description
                    "1",  # Runs per test
                    "300",  # Timeout
                ]
            )

            result = runner.invoke(
                cli,
                ["generate", "suite", "-o", "batch.yaml", "--count=3"],
                input=user_input,
            )

            # Click ClickException exits with code 1
            assert result.exit_code == 1
            assert "requires --template" in result.output

    def test_generate_suite_multiple_tests(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test generating a suite with multiple tests added interactively."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            user_input = "\n".join(
                [
                    "multi-suite",  # Suite name
                    "",  # Description
                    "1",  # Runs per test
                    "300",  # Timeout
                    "y",  # Add test? Yes
                    "2",  # Custom
                    "test-001",  # ID
                    "Test One",  # Name
                    "Task one",  # Task
                    "",  # Tags
                    "n",  # Constraints
                    "n",  # Assertions
                    "y",  # Add another? Yes
                    "2",  # Custom
                    "test-002",  # ID
                    "Test Two",  # Name
                    "Task two",  # Task
                    "",  # Tags
                    "n",  # Constraints
                    "n",  # Assertions
                    "n",  # Add another? No
                ]
            )

            result = runner.invoke(
                cli,
                ["generate", "suite", "-o", "multi.yaml"],
                input=user_input,
            )

            assert result.exit_code == 0

            with open("multi.yaml") as f:
                suite_data = yaml.safe_load(f)

            assert len(suite_data["tests"]) == 2
            assert suite_data["tests"][0]["id"] == "test-001"
            assert suite_data["tests"][1]["id"] == "test-002"

    def test_generate_suite_default_output_name(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that default output uses suite name."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["generate", "suite", "--no-interactive"])

            assert result.exit_code == 0
            assert Path("my-test-suite.yaml").exists()


class TestGenerateTestWithAssertions:
    """Tests for generate test with assertions."""

    def test_generate_test_with_artifact_exists(
        self, runner: CliRunner, tmp_path: Path, sample_suite_yaml: str
    ) -> None:
        """Test generating a test with artifact_exists assertion."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            suite_file = Path("suite.yaml")
            suite_file.write_text(sample_suite_yaml)

            user_input = "\n".join(
                [
                    "2",  # Custom
                    "assert-test",  # ID
                    "Assertion Test",  # Name
                    "Create a file",  # Task
                    "",  # Tags
                    "n",  # Constraints
                    "y",  # Assertions? Yes
                    "1",  # artifact_exists
                    "output.txt",  # Path
                    "4",  # Done
                ]
            )

            result = runner.invoke(
                cli,
                ["generate", "test", "--suite=suite.yaml"],
                input=user_input,
            )

            assert result.exit_code == 0

            with open("suite.yaml") as f:
                suite_data = yaml.safe_load(f)

            test = suite_data["tests"][1]
            assert len(test["assertions"]) == 1
            assert test["assertions"][0]["type"] == "artifact_exists"
            assert test["assertions"][0]["config"]["path"] == "output.txt"

    def test_generate_test_with_artifact_contains(
        self, runner: CliRunner, tmp_path: Path, sample_suite_yaml: str
    ) -> None:
        """Test generating a test with artifact_contains assertion."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            suite_file = Path("suite.yaml")
            suite_file.write_text(sample_suite_yaml)

            user_input = "\n".join(
                [
                    "2",  # Custom
                    "contains-test",  # ID
                    "Contains Test",  # Name
                    "Create a file with content",  # Task
                    "",  # Tags
                    "n",  # Constraints
                    "y",  # Assertions? Yes
                    "2",  # artifact_contains
                    "output.txt",  # Path
                    "expected.*pattern",  # Pattern
                    "4",  # Done
                ]
            )

            result = runner.invoke(
                cli,
                ["generate", "test", "--suite=suite.yaml"],
                input=user_input,
            )

            assert result.exit_code == 0

            with open("suite.yaml") as f:
                suite_data = yaml.safe_load(f)

            test = suite_data["tests"][1]
            assert test["assertions"][0]["type"] == "artifact_contains"
            assert test["assertions"][0]["config"]["path"] == "output.txt"
            assert test["assertions"][0]["config"]["pattern"] == "expected.*pattern"


class TestGenerateTestWithConstraints:
    """Tests for generate test with constraints."""

    def test_generate_test_with_constraints(
        self, runner: CliRunner, tmp_path: Path, sample_suite_yaml: str
    ) -> None:
        """Test generating a test with custom constraints."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            suite_file = Path("suite.yaml")
            suite_file.write_text(sample_suite_yaml)

            user_input = "\n".join(
                [
                    "2",  # Custom
                    "constrained-test",  # ID
                    "Constrained Test",  # Name
                    "Limited task",  # Task
                    "",  # Tags
                    "y",  # Constraints? Yes
                    "10",  # Max steps
                    "60",  # Timeout
                    "n",  # Assertions
                ]
            )

            result = runner.invoke(
                cli,
                ["generate", "test", "--suite=suite.yaml"],
                input=user_input,
            )

            assert result.exit_code == 0

            with open("suite.yaml") as f:
                suite_data = yaml.safe_load(f)

            test = suite_data["tests"][1]
            assert test["constraints"]["max_steps"] == 10
            assert test["constraints"]["timeout_seconds"] == 60


class TestGenerateAbort:
    """Tests for abort scenarios."""

    def test_generate_test_abort(
        self, runner: CliRunner, tmp_path: Path, sample_suite_yaml: str
    ) -> None:
        """Test aborting generate test."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            suite_file = Path("suite.yaml")
            suite_file.write_text(sample_suite_yaml)

            # Send Ctrl+C
            result = runner.invoke(
                cli,
                ["generate", "test", "--suite=suite.yaml"],
                input="\x03",
            )

            assert result.exit_code == 2

    def test_generate_suite_abort(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test aborting generate suite."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Send Ctrl+C
            result = runner.invoke(
                cli,
                ["generate", "suite", "-o", "test.yaml"],
                input="\x03",
            )

            assert result.exit_code == 2
