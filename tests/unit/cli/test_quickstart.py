"""Tests for atp quickstart command."""

import os
from pathlib import Path

from click.testing import CliRunner

from atp.cli.main import cli


def test_quickstart_help() -> None:
    """The quickstart command shows help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["quickstart", "--help"])
    assert result.exit_code == 0
    assert "quickstart" in result.output.lower()


def test_quickstart_creates_suite(tmp_path: Path) -> None:
    """Quickstart creates atp-suite.yaml in target directory."""
    runner = CliRunner()
    result = runner.invoke(cli, ["quickstart", "--dir", str(tmp_path), "--no-run"])
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(str(tmp_path), "atp-suite.yaml"))


def test_quickstart_default_dir(tmp_path: Path) -> None:
    """Quickstart creates files in current directory by default."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["quickstart", "--no-run"])
        assert result.exit_code == 0
        assert os.path.exists("atp-suite.yaml")
