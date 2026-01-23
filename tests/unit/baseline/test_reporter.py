"""Tests for baseline reporter module."""

import json
from datetime import UTC, datetime
from pathlib import Path

from atp.baseline.comparison import ComparisonResult, TestComparison
from atp.baseline.models import ChangeType
from atp.baseline.reporter import (
    format_comparison_console,
    format_comparison_json,
    print_comparison,
)


class TestFormatComparisonConsole:
    """Tests for format_comparison_console function."""

    def test_basic_output(self) -> None:
        """Test basic console output formatting."""
        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            baseline_created_at=datetime(2024, 1, 15, tzinfo=UTC),
            total_tests=3,
            regressions=1,
            improvements=1,
            no_changes=1,
            comparisons=[],
        )

        output = format_comparison_console(result, use_colors=False)

        assert "ATP Baseline Comparison" in output
        assert "test-suite" in output
        assert "test-agent" in output
        assert "Regressions: 1" in output
        assert "Improvements: 1" in output
        assert "No changes: 1" in output

    def test_regression_output(self) -> None:
        """Test output with regression details."""
        comparison = TestComparison(
            test_id="test-1",
            test_name="Test One",
            change_type=ChangeType.REGRESSION,
            current_mean=70.0,
            current_std=3.0,
            current_n_runs=5,
            baseline_mean=80.0,
            baseline_std=2.0,
            baseline_n_runs=10,
            delta=-10.0,
            delta_percent=-12.5,
            p_value=0.001,
            is_significant=True,
        )

        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            regressions=1,
            comparisons=[comparison],
        )

        output = format_comparison_console(result, use_colors=False)

        assert "Test One" in output
        assert "70.0" in output
        assert "was 80.0" in output
        assert "-10.0" in output
        assert "REGRESSION DETECTED" in output

    def test_improvement_output(self) -> None:
        """Test output with improvement details."""
        comparison = TestComparison(
            test_id="test-1",
            test_name="Test One",
            change_type=ChangeType.IMPROVEMENT,
            current_mean=90.0,
            current_std=2.0,
            current_n_runs=5,
            baseline_mean=80.0,
            baseline_std=3.0,
            baseline_n_runs=10,
            delta=10.0,
            delta_percent=12.5,
            p_value=0.001,
            is_significant=True,
        )

        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            improvements=1,
            comparisons=[comparison],
        )

        output = format_comparison_console(result, use_colors=False)

        assert "Test One" in output
        assert "+10.0" in output
        assert "IMPROVEMENTS DETECTED" in output

    def test_no_changes_output(self) -> None:
        """Test output with no changes."""
        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            no_changes=1,
            comparisons=[],
        )

        output = format_comparison_console(result, use_colors=False)

        assert "NO SIGNIFICANT CHANGES" in output

    def test_new_test_output(self) -> None:
        """Test output with new test."""
        comparison = TestComparison(
            test_id="test-new",
            test_name="New Test",
            change_type=ChangeType.NEW_TEST,
            current_mean=85.0,
            current_std=2.0,
            current_n_runs=5,
        )

        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            new_tests=1,
            comparisons=[comparison],
        )

        output = format_comparison_console(result, use_colors=False)

        assert "New Test" in output
        assert "(new)" in output

    def test_verbose_output(self) -> None:
        """Test verbose output shows additional details."""
        comparison = TestComparison(
            test_id="test-1",
            test_name="Test One",
            change_type=ChangeType.REGRESSION,
            current_mean=70.0,
            current_std=3.0,
            current_n_runs=5,
            baseline_mean=80.0,
            baseline_std=2.0,
            baseline_n_runs=10,
            delta=-10.0,
            delta_percent=-12.5,
            t_statistic=-5.0,
            p_value=0.001,
            is_significant=True,
        )

        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            regressions=1,
            comparisons=[comparison],
        )

        output = format_comparison_console(result, use_colors=False, verbose=True)

        assert "std:" in output
        assert "t=" in output


class TestFormatComparisonJson:
    """Tests for format_comparison_json function."""

    def test_json_output(self) -> None:
        """Test JSON output formatting."""
        comparison = TestComparison(
            test_id="test-1",
            test_name="Test One",
            change_type=ChangeType.REGRESSION,
            current_mean=70.0,
            current_std=3.0,
            current_n_runs=5,
            baseline_mean=80.0,
            baseline_std=2.0,
            baseline_n_runs=10,
            delta=-10.0,
            delta_percent=-12.5,
            p_value=0.001,
            is_significant=True,
        )

        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            regressions=1,
            comparisons=[comparison],
        )

        output = format_comparison_json(result)
        data = json.loads(output)

        assert data["suite_name"] == "test-suite"
        assert data["summary"]["regressions"] == 1
        assert len(data["comparisons"]) == 1
        assert data["comparisons"][0]["change_type"] == "regression"


class TestPrintComparison:
    """Tests for print_comparison function."""

    def test_print_to_file_console(self, tmp_path: Path) -> None:
        """Test printing to file in console format."""
        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            no_changes=1,
            comparisons=[],
        )

        output_file = tmp_path / "output.txt"
        print_comparison(
            result,
            output_format="console",
            output_file=output_file,
            use_colors=False,
        )

        assert output_file.exists()
        content = output_file.read_text()
        assert "ATP Baseline Comparison" in content

    def test_print_to_file_json(self, tmp_path: Path) -> None:
        """Test printing to file in JSON format."""
        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=1,
            no_changes=1,
            comparisons=[],
        )

        output_file = tmp_path / "output.json"
        print_comparison(result, output_format="json", output_file=output_file)

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["suite_name"] == "test-suite"

    def test_print_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that print_comparison creates parent directories."""
        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=0,
            comparisons=[],
        )

        output_file = tmp_path / "nested" / "dir" / "output.json"
        print_comparison(result, output_format="json", output_file=output_file)

        assert output_file.exists()
