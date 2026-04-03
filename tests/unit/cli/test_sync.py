"""Tests for atp sync command."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from atp.cli.commands.remote import MANIFEST_FILE, file_sha256
from atp.cli.commands.sync_cmd import sync_command


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_manifest(tmp_path: Path, server: str, files: dict) -> None:
    """Create a manifest file."""
    data = {"server": server, "last_sync": "", "files": files}
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(data))


class TestSyncCommand:
    """Tests for atp sync."""

    def test_sync_new_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """New file gets pushed."""
        (tmp_path / "new.yaml").write_text("test_suite: new\ntests: []\n")

        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "suite": {"id": 1, "name": "new"},
            "validation": {"valid": True, "errors": [], "warnings": []},
            "filename": "new.yaml",
        }

        with patch("atp.cli.commands.sync_cmd.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(
                return_value=mock_client
            )
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp

            result = runner.invoke(
                sync_command,
                [str(tmp_path), "--server", "http://test:8000"],
            )
            assert "created" in result.output

    def test_sync_unchanged_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Unchanged file gets skipped."""
        f = tmp_path / "existing.yaml"
        f.write_text("test_suite: existing\ntests: []\n")
        sha = file_sha256(f)

        _make_manifest(
            tmp_path,
            "http://test:8000",
            {"existing.yaml": {"sha256": sha, "suite_id": 1, "synced_at": ""}},
        )

        result = runner.invoke(
            sync_command,
            [str(tmp_path), "--server", "http://test:8000"],
        )
        assert "unchanged" in result.output

    def test_sync_deleted_file_warns(self, runner: CliRunner, tmp_path: Path) -> None:
        """Deleted file produces warning and is removed from manifest."""
        _make_manifest(
            tmp_path,
            "http://test:8000",
            {"gone.yaml": {"sha256": "old", "suite_id": 5, "synced_at": ""}},
        )

        result = runner.invoke(
            sync_command,
            [str(tmp_path), "--server", "http://test:8000"],
        )
        assert "removed locally" in result.output

        # Verify manifest updated
        manifest = json.loads((tmp_path / MANIFEST_FILE).read_text())
        assert "gone.yaml" not in manifest["files"]

    def test_sync_dry_run(self, runner: CliRunner, tmp_path: Path) -> None:
        """Dry run shows plan without executing."""
        (tmp_path / "new.yaml").write_text("test_suite: new\ntests: []\n")

        result = runner.invoke(
            sync_command,
            [str(tmp_path), "--server", "http://test:8000", "--dry-run"],
        )
        assert "Dry run" in result.output
        assert result.exit_code == 0
