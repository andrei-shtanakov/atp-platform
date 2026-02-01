"""Tests for ATP CLI budget commands."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from atp.cli.commands.budget import budget_command


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


class TestBudgetListCommand:
    """Tests for 'atp budget list' command."""

    def test_list_budgets_empty(self, cli_runner: CliRunner) -> None:
        """Test listing budgets when none exist."""
        with patch("atp.cli.commands.budget._list_budgets_async") as mock_list:
            mock_list.return_value = []

            result = cli_runner.invoke(budget_command, ["list"])

            assert result.exit_code == 0
            assert "No budgets found" in result.output

    def test_list_budgets_with_data(self, cli_runner: CliRunner) -> None:
        """Test listing budgets with data."""
        with patch("atp.cli.commands.budget._list_budgets_async") as mock_list:
            mock_list.return_value = [
                {
                    "id": 1,
                    "name": "daily-limit",
                    "period": "daily",
                    "limit_usd": "100.00",
                    "alert_threshold": 0.8,
                    "is_active": True,
                    "description": "Daily budget",
                    "alert_channels": ["log"],
                    "scope": None,
                }
            ]

            result = cli_runner.invoke(budget_command, ["list"])

            assert result.exit_code == 0
            assert "daily-limit" in result.output
            assert "daily" in result.output

    def test_list_budgets_filter_period(self, cli_runner: CliRunner) -> None:
        """Test filtering budgets by period."""
        with patch("atp.cli.commands.budget._list_budgets_async") as mock_list:
            mock_list.return_value = []

            result = cli_runner.invoke(budget_command, ["list", "--period=daily"])

            assert result.exit_code == 0
            mock_list.assert_called_once()
            call_args = mock_list.call_args
            assert call_args[1]["period"] == "daily"

    def test_list_budgets_json_output(self, cli_runner: CliRunner) -> None:
        """Test JSON output format."""
        with patch("atp.cli.commands.budget._list_budgets_async") as mock_list:
            mock_list.return_value = [
                {
                    "id": 1,
                    "name": "test",
                    "period": "daily",
                    "limit_usd": "100.00",
                    "alert_threshold": 0.8,
                    "is_active": True,
                    "description": None,
                    "alert_channels": [],
                    "scope": None,
                }
            ]

            result = cli_runner.invoke(budget_command, ["list", "--output=json"])

            assert result.exit_code == 0
            assert '"name": "test"' in result.output


class TestBudgetStatusCommand:
    """Tests for 'atp budget status' command."""

    def test_status_empty(self, cli_runner: CliRunner) -> None:
        """Test status when no budgets exist."""
        with patch("atp.cli.commands.budget._check_budgets_async") as mock_check:
            mock_check.return_value = {
                "timestamp": "2026-01-15T10:00:00",
                "has_alerts": False,
                "has_exceeded": False,
                "statuses": [],
            }

            result = cli_runner.invoke(budget_command, ["status"])

            assert result.exit_code == 0
            assert "No budget status available" in result.output

    def test_status_with_data(self, cli_runner: CliRunner) -> None:
        """Test status with budget data."""
        with patch("atp.cli.commands.budget._check_budgets_async") as mock_check:
            mock_check.return_value = {
                "timestamp": "2026-01-15T10:00:00",
                "has_alerts": False,
                "has_exceeded": False,
                "statuses": [
                    {
                        "budget_id": 1,
                        "budget_name": "daily-limit",
                        "period": "daily",
                        "period_start": "2026-01-15T00:00:00",
                        "limit_usd": "100.00",
                        "spent_usd": "50.00",
                        "remaining_usd": "50.00",
                        "percentage": 50.0,
                        "is_over_threshold": False,
                        "is_over_limit": False,
                        "triggered_alerts": [],
                    }
                ],
            }

            result = cli_runner.invoke(budget_command, ["status"])

            assert result.exit_code == 0
            assert "daily-limit" in result.output

    def test_status_budget_exceeded(self, cli_runner: CliRunner) -> None:
        """Test status when budget is exceeded."""
        with patch("atp.cli.commands.budget._check_budgets_async") as mock_check:
            mock_check.return_value = {
                "timestamp": "2026-01-15T10:00:00",
                "has_alerts": True,
                "has_exceeded": True,
                "statuses": [
                    {
                        "budget_id": 1,
                        "budget_name": "daily-limit",
                        "period": "daily",
                        "period_start": "2026-01-15T00:00:00",
                        "limit_usd": "100.00",
                        "spent_usd": "110.00",
                        "remaining_usd": "0.00",
                        "percentage": 110.0,
                        "is_over_threshold": True,
                        "is_over_limit": True,
                        "triggered_alerts": [1.0],
                    }
                ],
            }

            result = cli_runner.invoke(budget_command, ["status"])

            # Should exit with failure code when exceeded
            assert result.exit_code == 1

    def test_status_json_output(self, cli_runner: CliRunner) -> None:
        """Test JSON output format for status."""
        with patch("atp.cli.commands.budget._check_budgets_async") as mock_check:
            mock_check.return_value = {
                "timestamp": "2026-01-15T10:00:00",
                "has_alerts": False,
                "has_exceeded": False,
                "statuses": [],
            }

            result = cli_runner.invoke(budget_command, ["status", "--output=json"])

            assert result.exit_code == 0
            assert '"timestamp"' in result.output


class TestBudgetCreateCommand:
    """Tests for 'atp budget create' command."""

    def test_create_budget(self, cli_runner: CliRunner) -> None:
        """Test creating a budget."""
        with patch("atp.cli.commands.budget._create_budget_async") as mock_create:
            mock_create.return_value = {
                "id": 1,
                "name": "test-budget",
                "period": "daily",
                "limit_usd": "100.00",
            }

            result = cli_runner.invoke(
                budget_command,
                ["create", "--name=test-budget", "--period=daily", "--limit=100"],
            )

            assert result.exit_code == 0
            assert "created successfully" in result.output

    def test_create_budget_with_options(self, cli_runner: CliRunner) -> None:
        """Test creating a budget with all options."""
        with patch("atp.cli.commands.budget._create_budget_async") as mock_create:
            mock_create.return_value = {
                "id": 1,
                "name": "full-budget",
                "period": "monthly",
                "limit_usd": "2000.00",
            }

            result = cli_runner.invoke(
                budget_command,
                [
                    "create",
                    "--name=full-budget",
                    "--period=monthly",
                    "--limit=2000",
                    "--threshold=0.9",
                    "--channels=log",
                    "--channels=email",
                    "--description=Test budget",
                    "--scope=provider=anthropic",
                ],
            )

            assert result.exit_code == 0
            mock_create.assert_called_once()

    def test_create_budget_missing_required(self, cli_runner: CliRunner) -> None:
        """Test creating budget without required options."""
        result = cli_runner.invoke(budget_command, ["create"])

        assert result.exit_code != 0
        assert "Missing option" in result.output

    def test_create_budget_invalid_threshold(self, cli_runner: CliRunner) -> None:
        """Test creating budget with invalid threshold."""
        result = cli_runner.invoke(
            budget_command,
            [
                "create",
                "--name=test",
                "--period=daily",
                "--limit=100",
                "--threshold=2.5",
            ],
        )

        assert result.exit_code == 2
        assert "threshold must be between" in result.output


class TestBudgetUpdateCommand:
    """Tests for 'atp budget update' command."""

    def test_update_budget(self, cli_runner: CliRunner) -> None:
        """Test updating a budget."""
        with patch("atp.cli.commands.budget._update_budget_async") as mock_update:
            mock_update.return_value = {"id": 1, "name": "test-budget"}

            result = cli_runner.invoke(
                budget_command,
                ["update", "test-budget", "--limit=150"],
            )

            assert result.exit_code == 0
            assert "updated successfully" in result.output

    def test_update_budget_not_found(self, cli_runner: CliRunner) -> None:
        """Test updating a non-existent budget."""
        with patch("atp.cli.commands.budget._update_budget_async") as mock_update:
            mock_update.return_value = None

            result = cli_runner.invoke(
                budget_command,
                ["update", "nonexistent", "--limit=100"],
            )

            assert result.exit_code == 1
            assert "not found" in result.output

    def test_update_budget_deactivate(self, cli_runner: CliRunner) -> None:
        """Test deactivating a budget."""
        with patch("atp.cli.commands.budget._update_budget_async") as mock_update:
            mock_update.return_value = {"id": 1, "name": "test-budget"}

            result = cli_runner.invoke(
                budget_command,
                ["update", "test-budget", "--deactivate"],
            )

            assert result.exit_code == 0
            call_args = mock_update.call_args
            assert call_args[1]["is_active"] is False


class TestBudgetDeleteCommand:
    """Tests for 'atp budget delete' command."""

    def test_delete_budget_with_force(self, cli_runner: CliRunner) -> None:
        """Test deleting a budget with force flag."""
        with patch("atp.cli.commands.budget._delete_budget_async") as mock_delete:
            mock_delete.return_value = True

            result = cli_runner.invoke(
                budget_command,
                ["delete", "test-budget", "--force"],
            )

            assert result.exit_code == 0
            assert "deleted successfully" in result.output

    def test_delete_budget_not_found(self, cli_runner: CliRunner) -> None:
        """Test deleting a non-existent budget."""
        with patch("atp.cli.commands.budget._delete_budget_async") as mock_delete:
            mock_delete.return_value = False

            result = cli_runner.invoke(
                budget_command,
                ["delete", "nonexistent", "--force"],
            )

            assert result.exit_code == 1
            assert "not found" in result.output

    def test_delete_budget_cancelled(self, cli_runner: CliRunner) -> None:
        """Test cancelling budget deletion."""
        result = cli_runner.invoke(
            budget_command,
            ["delete", "test-budget"],
            input="n\n",
        )

        assert result.exit_code == 1
        assert "cancelled" in result.output


class TestBudgetSetThresholdsCommand:
    """Tests for 'atp budget set-thresholds' command."""

    def test_set_thresholds(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test setting thresholds from config file."""
        config_file = tmp_path / "cost.yaml"
        config_file.write_text(
            """
cost:
  budgets:
    daily: 100.00
    monthly: 2000.00
  alerts:
    - threshold: 0.8
      channels: ["log"]
"""
        )

        with patch("atp.cli.commands.budget._set_thresholds_async") as mock_set:
            mock_set.return_value = {
                "daily": 100.00,
                "weekly": None,
                "monthly": 2000.00,
                "alerts": 1,
            }

            result = cli_runner.invoke(
                budget_command,
                ["set-thresholds", f"--config={config_file}"],
            )

            assert result.exit_code == 0
            assert "applied" in result.output

    def test_set_thresholds_file_not_found(self, cli_runner: CliRunner) -> None:
        """Test setting thresholds from non-existent file."""
        result = cli_runner.invoke(
            budget_command,
            ["set-thresholds", "--config=/nonexistent/path.yaml"],
        )

        assert result.exit_code != 0


class TestBudgetCheckCommand:
    """Tests for 'atp budget check' command."""

    def test_check_all_ok(self, cli_runner: CliRunner) -> None:
        """Test check when all budgets are OK."""
        with patch("atp.cli.commands.budget._check_budgets_async") as mock_check:
            mock_check.return_value = {
                "timestamp": "2026-01-15T10:00:00",
                "has_alerts": False,
                "has_exceeded": False,
                "statuses": [
                    {
                        "budget_id": 1,
                        "budget_name": "daily",
                        "period": "daily",
                        "period_start": "2026-01-15T00:00:00",
                        "limit_usd": "100.00",
                        "spent_usd": "50.00",
                        "remaining_usd": "50.00",
                        "percentage": 50.0,
                        "is_over_threshold": False,
                        "is_over_limit": False,
                        "triggered_alerts": [],
                    }
                ],
            }

            result = cli_runner.invoke(budget_command, ["check"])

            assert result.exit_code == 0
            assert "All budgets OK" in result.output

    def test_check_fail_on_exceeded(self, cli_runner: CliRunner) -> None:
        """Test check with --fail-on-exceeded flag."""
        with patch("atp.cli.commands.budget._check_budgets_async") as mock_check:
            mock_check.return_value = {
                "timestamp": "2026-01-15T10:00:00",
                "has_alerts": True,
                "has_exceeded": True,
                "statuses": [
                    {
                        "budget_id": 1,
                        "budget_name": "daily",
                        "period": "daily",
                        "period_start": "2026-01-15T00:00:00",
                        "limit_usd": "100.00",
                        "spent_usd": "110.00",
                        "remaining_usd": "0.00",
                        "percentage": 110.0,
                        "is_over_threshold": True,
                        "is_over_limit": True,
                        "triggered_alerts": [1.0],
                    }
                ],
            }

            result = cli_runner.invoke(budget_command, ["check", "--fail-on-exceeded"])

            assert result.exit_code == 1
            assert "EXCEEDED" in result.output

    def test_check_fail_on_warning(self, cli_runner: CliRunner) -> None:
        """Test check with --fail-on-warning flag."""
        with patch("atp.cli.commands.budget._check_budgets_async") as mock_check:
            mock_check.return_value = {
                "timestamp": "2026-01-15T10:00:00",
                "has_alerts": True,
                "has_exceeded": False,
                "statuses": [
                    {
                        "budget_id": 1,
                        "budget_name": "daily",
                        "period": "daily",
                        "period_start": "2026-01-15T00:00:00",
                        "limit_usd": "100.00",
                        "spent_usd": "85.00",
                        "remaining_usd": "15.00",
                        "percentage": 85.0,
                        "is_over_threshold": True,
                        "is_over_limit": False,
                        "triggered_alerts": [0.8],
                    }
                ],
            }

            result = cli_runner.invoke(budget_command, ["check", "--fail-on-warning"])

            assert result.exit_code == 1
            assert "WARNING" in result.output

    def test_check_quiet_mode(self, cli_runner: CliRunner) -> None:
        """Test check with --quiet flag."""
        with patch("atp.cli.commands.budget._check_budgets_async") as mock_check:
            mock_check.return_value = {
                "timestamp": "2026-01-15T10:00:00",
                "has_alerts": False,
                "has_exceeded": False,
                "statuses": [],
            }

            result = cli_runner.invoke(budget_command, ["check", "--quiet"])

            assert result.exit_code == 0
            assert result.output.strip() == ""
