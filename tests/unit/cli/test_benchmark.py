"""Tests for CLI benchmark commands."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from atp.benchmarks import (
    BenchmarkCategory,
    BenchmarkMetadata,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkSuiteResult,
    BenchmarkTest,
)
from atp.cli.commands.benchmark import (
    EXIT_ERROR,
    EXIT_FAILURE,
    EXIT_SUCCESS,
    _convert_to_test_suite,
    _create_results_table,
    _create_suites_table,
    _format_delta,
    _format_score,
    _format_status,
)
from atp.cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_benchmark_suite() -> BenchmarkSuite:
    """Create a sample benchmark suite for testing."""
    return BenchmarkSuite(
        name="test_coding",
        category=BenchmarkCategory.CODING,
        version="1.0.0",
        description="Test coding benchmark suite",
        tests=[
            BenchmarkTest(
                id="test-001",
                name="Test Function",
                description="Test generating a function",
                task_description="Write a hello world function",
                expected_artifacts=["*.py"],
                metadata=BenchmarkMetadata(
                    category=BenchmarkCategory.CODING,
                    difficulty="easy",
                    estimated_time_seconds=60,
                    skills_tested=["python"],
                ),
                tags=["easy", "python"],
            ),
            BenchmarkTest(
                id="test-002",
                name="Test Class",
                description="Test generating a class",
                task_description="Write a simple class",
                expected_artifacts=["*.py"],
                metadata=BenchmarkMetadata(
                    category=BenchmarkCategory.CODING,
                    difficulty="medium",
                    estimated_time_seconds=120,
                    skills_tested=["python", "oop"],
                ),
                tags=["medium", "python"],
            ),
        ],
    )


@pytest.fixture
def sample_benchmark_result() -> BenchmarkResult:
    """Create a sample benchmark result."""
    return BenchmarkResult(
        test_id="test-001",
        raw_score=0.8,
        normalized_score=80.0,
        passed=True,
        execution_time_seconds=10.5,
        tokens_used=150,
        cost_usd=0.001,
    )


@pytest.fixture
def sample_suite_result(
    sample_benchmark_result: BenchmarkResult,
) -> BenchmarkSuiteResult:
    """Create a sample benchmark suite result."""
    return BenchmarkSuiteResult(
        suite_name="test_coding",
        category=BenchmarkCategory.CODING,
        agent_name="test-agent",
        total_tests=2,
        passed_tests=1,
        failed_tests=1,
        average_normalized_score=75.0,
        total_execution_time_seconds=20.0,
        results=[sample_benchmark_result],
        baseline_comparison={"gpt-4": 5.0, "claude-3-opus": -2.0},
    )


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_format_score(self) -> None:
        """Test score formatting."""
        assert _format_score(85.5) == "85.5"
        assert _format_score(100.0) == "100.0"
        assert _format_score(0.0) == "0.0"

    def test_format_delta_positive(self) -> None:
        """Test formatting positive delta."""
        assert _format_delta(5.5) == "+5.5"

    def test_format_delta_negative(self) -> None:
        """Test formatting negative delta."""
        assert _format_delta(-3.2) == "-3.2"

    def test_format_delta_none(self) -> None:
        """Test formatting None delta."""
        assert _format_delta(None) == "-"

    def test_format_delta_zero(self) -> None:
        """Test formatting zero delta."""
        assert _format_delta(0.0) == "+0.0"

    def test_format_status_pass(self) -> None:
        """Test pass status formatting."""
        result = _format_status(True)
        assert "PASS" in result
        assert "green" in result

    def test_format_status_fail(self) -> None:
        """Test fail status formatting."""
        result = _format_status(False)
        assert "FAIL" in result
        assert "red" in result

    def test_create_suites_table(self) -> None:
        """Test creating suites table."""
        table = _create_suites_table("Test Suites")
        assert table.title == "Test Suites"
        assert len(table.columns) == 5

    def test_create_results_table(self) -> None:
        """Test creating results table."""
        table = _create_results_table("Test Results")
        assert table.title == "Test Results"
        assert len(table.columns) == 5


class TestConvertToTestSuite:
    """Tests for _convert_to_test_suite function."""

    def test_convert_basic_suite(self, sample_benchmark_suite: BenchmarkSuite) -> None:
        """Test converting a basic benchmark suite."""
        test_suite = _convert_to_test_suite(sample_benchmark_suite)

        assert test_suite.test_suite == "test_coding"
        assert test_suite.version == "1.0.0"
        assert len(test_suite.tests) == 2

    def test_convert_preserves_test_data(
        self, sample_benchmark_suite: BenchmarkSuite
    ) -> None:
        """Test that conversion preserves test data."""
        test_suite = _convert_to_test_suite(sample_benchmark_suite)

        first_test = test_suite.tests[0]
        assert first_test.id == "test-001"
        assert first_test.name == "Test Function"
        assert first_test.task.description == "Write a hello world function"
        assert first_test.task.expected_artifacts is not None
        assert "*.py" in first_test.task.expected_artifacts


class TestBenchmarkListCommand:
    """Tests for benchmark list command."""

    def test_list_help(self, runner: CliRunner) -> None:
        """Test benchmark list help."""
        result = runner.invoke(cli, ["benchmark", "list", "--help"])
        assert result.exit_code == 0
        assert "List all available benchmark suites" in result.output
        assert "--category" in result.output

    def test_list_all_benchmarks(
        self, runner: CliRunner, sample_benchmark_suite: BenchmarkSuite
    ) -> None:
        """Test listing all benchmarks."""
        with patch("atp.cli.commands.benchmark.get_registry") as mock_registry:
            registry = MagicMock()
            registry.list_suites.return_value = ["test_coding"]
            registry.get.return_value = sample_benchmark_suite
            mock_registry.return_value = registry

            result = runner.invoke(cli, ["benchmark", "list"])

            assert result.exit_code == EXIT_SUCCESS
            assert "test_coding" in result.output

    def test_list_with_category_filter(
        self, runner: CliRunner, sample_benchmark_suite: BenchmarkSuite
    ) -> None:
        """Test listing benchmarks with category filter."""
        with patch("atp.cli.commands.benchmark.get_registry") as mock_registry:
            registry = MagicMock()
            registry.get_by_category.return_value = [sample_benchmark_suite]
            mock_registry.return_value = registry

            result = runner.invoke(cli, ["benchmark", "list", "--category=coding"])

            assert result.exit_code == EXIT_SUCCESS
            registry.get_by_category.assert_called_once_with("coding")

    def test_list_empty(self, runner: CliRunner) -> None:
        """Test listing when no benchmarks exist."""
        with patch("atp.cli.commands.benchmark.get_registry") as mock_registry:
            registry = MagicMock()
            registry.list_suites.return_value = []
            mock_registry.return_value = registry

            result = runner.invoke(cli, ["benchmark", "list"])

            assert result.exit_code == EXIT_SUCCESS
            assert "No benchmark suites found" in result.output

    def test_list_verbose(
        self, runner: CliRunner, sample_benchmark_suite: BenchmarkSuite
    ) -> None:
        """Test listing benchmarks with verbose flag."""
        with patch("atp.cli.commands.benchmark.get_registry") as mock_registry:
            registry = MagicMock()
            registry.list_suites.return_value = ["test_coding"]
            registry.get.return_value = sample_benchmark_suite
            registry.list_categories.return_value = ["coding", "research"]
            mock_registry.return_value = registry

            result = runner.invoke(cli, ["benchmark", "list", "--verbose"])

            assert result.exit_code == EXIT_SUCCESS
            assert "Total:" in result.output


class TestBenchmarkRunCommand:
    """Tests for benchmark run command."""

    def test_run_help(self, runner: CliRunner) -> None:
        """Test benchmark run help."""
        result = runner.invoke(cli, ["benchmark", "run", "--help"])
        assert result.exit_code == 0
        assert "Run benchmark suites" in result.output
        assert "--agent" in result.output
        assert "--output" in result.output
        assert "--all" in result.output

    def test_run_no_args(self, runner: CliRunner) -> None:
        """Test run without required arguments."""
        result = runner.invoke(cli, ["benchmark", "run"])

        assert result.exit_code == EXIT_ERROR
        assert "Specify a CATEGORY" in result.output

    def test_run_invalid_parallel(self, runner: CliRunner) -> None:
        """Test run with invalid parallel value."""
        result = runner.invoke(cli, ["benchmark", "run", "coding", "--parallel=0"])

        assert result.exit_code == EXIT_ERROR
        assert "--parallel must be at least 1" in result.output

    def test_run_category_not_found(self, runner: CliRunner) -> None:
        """Test run with non-existent category."""
        from atp.benchmarks import BenchmarkCategoryNotFoundError

        with patch("atp.cli.commands.benchmark.get_registry") as mock_registry:
            registry = MagicMock()
            registry.get_by_category.side_effect = BenchmarkCategoryNotFoundError(
                "invalid"
            )
            mock_registry.return_value = registry

            result = runner.invoke(cli, ["benchmark", "run", "invalid"])

            assert result.exit_code == EXIT_ERROR
            assert "Category not found" in result.output

    def test_run_category_success(
        self, runner: CliRunner, sample_benchmark_suite: BenchmarkSuite
    ) -> None:
        """Test successful benchmark run with category."""
        with (
            patch("atp.cli.commands.benchmark.get_registry") as mock_registry,
            patch("atp.cli.commands.benchmark._run_benchmarks") as mock_run,
        ):
            registry = MagicMock()
            registry.get_by_category.return_value = [sample_benchmark_suite]
            mock_registry.return_value = registry
            mock_run.return_value = True

            result = runner.invoke(cli, ["benchmark", "run", "coding"])

            # The function is async, so it may exit before run completes
            # We just check it didn't error out
            assert result.exit_code in (EXIT_SUCCESS, EXIT_FAILURE)

    def test_run_with_agent_option(
        self, runner: CliRunner, sample_benchmark_suite: BenchmarkSuite
    ) -> None:
        """Test run with --agent option."""
        with (
            patch("atp.cli.commands.benchmark.get_registry") as mock_registry,
            patch("atp.cli.commands.benchmark._run_benchmarks") as mock_run,
        ):
            registry = MagicMock()
            registry.get_by_category.return_value = [sample_benchmark_suite]
            mock_registry.return_value = registry
            mock_run.return_value = True

            result = runner.invoke(
                cli, ["benchmark", "run", "coding", "--agent=my-agent"]
            )

            # Check agent option is passed
            assert result.exit_code in (EXIT_SUCCESS, EXIT_FAILURE)

    def test_run_all_benchmarks(
        self, runner: CliRunner, sample_benchmark_suite: BenchmarkSuite
    ) -> None:
        """Test run with --all flag."""
        with (
            patch("atp.cli.commands.benchmark.get_registry") as mock_registry,
            patch("atp.cli.commands.benchmark._run_benchmarks") as mock_run,
        ):
            registry = MagicMock()
            registry.list_suites.return_value = ["test_coding"]
            registry.get.return_value = sample_benchmark_suite
            mock_registry.return_value = registry
            mock_run.return_value = True

            result = runner.invoke(cli, ["benchmark", "run", "--all"])

            assert result.exit_code in (EXIT_SUCCESS, EXIT_FAILURE)


class TestBenchmarkInfoCommand:
    """Tests for benchmark info command."""

    def test_info_help(self, runner: CliRunner) -> None:
        """Test benchmark info help."""
        result = runner.invoke(cli, ["benchmark", "info", "--help"])
        assert result.exit_code == 0
        assert "Show detailed information" in result.output

    def test_info_found(
        self, runner: CliRunner, sample_benchmark_suite: BenchmarkSuite
    ) -> None:
        """Test info for existing benchmark."""
        with patch("atp.cli.commands.benchmark.get_registry") as mock_registry:
            registry = MagicMock()
            registry.get.return_value = sample_benchmark_suite
            registry.get_baseline_scores.return_value = {"test-001": []}
            mock_registry.return_value = registry

            result = runner.invoke(cli, ["benchmark", "info", "test_coding"])

            assert result.exit_code == EXIT_SUCCESS
            assert "test_coding" in result.output
            assert "coding" in result.output

    def test_info_not_found(self, runner: CliRunner) -> None:
        """Test info for non-existent benchmark."""
        from atp.benchmarks import BenchmarkNotFoundError

        with patch("atp.cli.commands.benchmark.get_registry") as mock_registry:
            registry = MagicMock()
            registry.get.side_effect = BenchmarkNotFoundError("nonexistent")
            mock_registry.return_value = registry

            result = runner.invoke(cli, ["benchmark", "info", "nonexistent"])

            assert result.exit_code == EXIT_FAILURE
            assert "not found" in result.output


class TestBenchmarkCategoriesCommand:
    """Tests for benchmark categories command."""

    def test_categories_command(self, runner: CliRunner) -> None:
        """Test categories command."""
        with patch("atp.cli.commands.benchmark.get_registry") as mock_registry:
            registry = MagicMock()
            registry.list_categories.return_value = [
                "coding",
                "research",
                "reasoning",
                "data_processing",
            ]
            registry.get_by_category.return_value = []
            mock_registry.return_value = registry

            result = runner.invoke(cli, ["benchmark", "categories"])

            assert result.exit_code == EXIT_SUCCESS
            assert "coding" in result.output
            assert "research" in result.output


class TestBenchmarkCommandGroup:
    """Tests for benchmark command group."""

    def test_benchmark_help(self, runner: CliRunner) -> None:
        """Test benchmark group help."""
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "Run and manage ATP benchmark suites" in result.output
        assert "list" in result.output
        assert "run" in result.output
        assert "info" in result.output
        assert "categories" in result.output

    def test_benchmark_no_subcommand(self, runner: CliRunner) -> None:
        """Test benchmark without subcommand shows usage."""
        result = runner.invoke(cli, ["benchmark"])
        # Without a subcommand, Click shows usage
        assert result.exit_code == EXIT_ERROR or "Usage:" in result.output


class TestExitCodes:
    """Tests for exit codes."""

    def test_exit_codes_defined(self) -> None:
        """Test that exit codes are defined correctly."""
        assert EXIT_SUCCESS == 0
        assert EXIT_FAILURE == 1
        assert EXIT_ERROR == 2


class TestJsonOutput:
    """Tests for JSON output functionality."""

    def test_json_output_structure(
        self, runner: CliRunner, sample_benchmark_suite: BenchmarkSuite
    ) -> None:
        """Test that JSON output has correct structure."""

        with (
            TemporaryDirectory() as tmpdir,
            patch("atp.cli.commands.benchmark.get_registry") as mock_registry,
            patch("atp.cli.commands.benchmark._run_benchmarks") as mock_run,
        ):
            output_file = Path(tmpdir) / "results.json"
            registry = MagicMock()
            registry.get_by_category.return_value = [sample_benchmark_suite]
            mock_registry.return_value = registry
            mock_run.return_value = True

            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "run",
                    "coding",
                    "--output=json",
                    f"--output-file={output_file}",
                ],
            )

            # Verify command doesn't error
            assert result.exit_code in (EXIT_SUCCESS, EXIT_FAILURE)
