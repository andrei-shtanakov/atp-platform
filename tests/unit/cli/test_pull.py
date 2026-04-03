"""Tests for atp pull command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from atp.cli.commands.pull import pull_command


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestPullCommand:
    """Tests for atp pull."""

    def test_pull_all_suites(self, runner: CliRunner, tmp_path: Path) -> None:
        """Pull all suites from server."""
        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.json.return_value = {
            "items": [
                {
                    "id": 1,
                    "name": "suite-one",
                    "version": "1.0",
                    "description": None,
                    "test_count": 2,
                    "agent_count": 0,
                },
            ],
            "total": 1,
            "limit": 50,
            "offset": 0,
        }

        yaml_resp = MagicMock()
        yaml_resp.status_code = 200
        yaml_resp.json.return_value = {
            "yaml_content": "test_suite: suite-one\ntests: []\n"
        }

        with patch("atp.cli.commands.pull.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(
                return_value=mock_client
            )
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = [list_resp, yaml_resp]

            result = runner.invoke(
                pull_command,
                ["--server", "http://test:8000", "--dir", str(tmp_path)],
            )
            assert result.exit_code == 0
            assert (tmp_path / "suite-one.yaml").exists()

    def test_pull_by_id(self, runner: CliRunner, tmp_path: Path) -> None:
        """Pull a specific suite by ID."""
        yaml_resp = MagicMock()
        yaml_resp.status_code = 200
        yaml_resp.json.return_value = {
            "yaml_content": "test_suite: my-suite\ntests: []\n"
        }

        with patch("atp.cli.commands.pull.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(
                return_value=mock_client
            )
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = yaml_resp

            result = runner.invoke(
                pull_command,
                [
                    "--server",
                    "http://test:8000",
                    "--id",
                    "5",
                    "--dir",
                    str(tmp_path),
                ],
            )
            assert result.exit_code == 0

    def test_pull_skip_existing(self, runner: CliRunner, tmp_path: Path) -> None:
        """Pull skips existing files without --force."""
        (tmp_path / "suite-one.yaml").write_text("existing")

        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.json.return_value = {
            "items": [
                {
                    "id": 1,
                    "name": "suite-one",
                    "version": "1.0",
                    "description": None,
                    "test_count": 2,
                    "agent_count": 0,
                }
            ],
            "total": 1,
            "limit": 50,
            "offset": 0,
        }

        with patch("atp.cli.commands.pull.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(
                return_value=mock_client
            )
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = list_resp

            result = runner.invoke(
                pull_command,
                ["--server", "http://test:8000", "--dir", str(tmp_path)],
            )
            assert "skipped" in result.output.lower() or "skip" in result.output.lower()
