"""Tests for atp push command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from atp.cli.commands.push import push_command


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def valid_yaml(tmp_path: Path) -> Path:
    f = tmp_path / "suite.yaml"
    f.write_text(
        "test_suite: test\ntests:\n  - id: t1\n    name: T1\n"
        "    task:\n      description: do\n"
    )
    return f


@pytest.fixture
def invalid_yaml(tmp_path: Path) -> Path:
    f = tmp_path / "bad.yaml"
    f.write_text("not valid yaml: [[[")
    return f


class TestPushCommand:
    """Tests for atp push."""

    def test_push_single_file_success(
        self, runner: CliRunner, valid_yaml: Path
    ) -> None:
        """Push a single valid file."""
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "suite": {"id": 1, "name": "test"},
            "validation": {"valid": True, "errors": [], "warnings": []},
            "filename": "suite.yaml",
        }

        with patch("atp.cli.commands.push.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(
                return_value=mock_client
            )
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp

            result = runner.invoke(
                push_command,
                [str(valid_yaml), "--server", "http://test:8000"],
            )
            assert result.exit_code == 0
            assert "created" in result.output

    def test_push_no_server_fails(self, runner: CliRunner, valid_yaml: Path) -> None:
        """Push without server URL fails."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("atp.cli.commands.remote.load_token", return_value=None):
                result = runner.invoke(push_command, [str(valid_yaml)])
                assert result.exit_code != 0

    def test_push_validation_error(self, runner: CliRunner, valid_yaml: Path) -> None:
        """Push with validation error returns exit code 1."""
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.json.return_value = {
            "detail": {
                "suite": None,
                "validation": {
                    "valid": False,
                    "errors": ["parse error"],
                    "warnings": [],
                },
                "filename": "suite.yaml",
            }
        }

        with patch("atp.cli.commands.push.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(
                return_value=mock_client
            )
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp

            result = runner.invoke(
                push_command,
                [str(valid_yaml), "--server", "http://test:8000"],
            )
            assert result.exit_code == 1

    def test_push_dry_run(self, runner: CliRunner, valid_yaml: Path) -> None:
        """Dry run shows files without uploading."""
        result = runner.invoke(
            push_command,
            [
                str(valid_yaml),
                "--server",
                "http://test:8000",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "suite.yaml" in result.output
