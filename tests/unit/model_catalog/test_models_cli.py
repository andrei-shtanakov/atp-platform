from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from atp.cli.commands.models import models_command


def _iso(monkeypatch, tmp_path: Path) -> Path:
    """Point resolution at a tmp file via $ATP_CATALOG; return that path."""
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    target = tmp_path / "agents-catalog.toml"
    monkeypatch.setenv("ATP_CATALOG", str(target))
    return target


def test_init_writes_template(monkeypatch, tmp_path: Path) -> None:
    target = _iso(monkeypatch, tmp_path)
    res = CliRunner().invoke(models_command, ["init"])
    assert res.exit_code == 0, res.output
    assert target.is_file()
    assert "[models]" in target.read_text(encoding="utf-8")


def test_init_creates_parent_dirs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    target = tmp_path / "nested" / "deep" / "agents-catalog.toml"
    monkeypatch.setenv("ATP_CATALOG", str(target))
    res = CliRunner().invoke(models_command, ["init"])
    assert res.exit_code == 0, res.output
    assert target.is_file()


def test_init_refuses_overwrite_without_force(monkeypatch, tmp_path: Path) -> None:
    target = _iso(monkeypatch, tmp_path)
    target.write_text("[models]\n", encoding="utf-8")
    res = CliRunner().invoke(models_command, ["init"])
    assert res.exit_code != 0
    assert "already exists" in res.output


def test_init_force_overwrites(monkeypatch, tmp_path: Path) -> None:
    target = _iso(monkeypatch, tmp_path)
    target.write_text("old", encoding="utf-8")
    res = CliRunner().invoke(models_command, ["init", "--force"])
    assert res.exit_code == 0, res.output
    assert "[models]" in target.read_text(encoding="utf-8")


def test_list_table(monkeypatch, tmp_path: Path) -> None:
    target = _iso(monkeypatch, tmp_path)
    target.write_text(
        '[models."claude-x"]\nvendor="anthropic"\nstatus="active"\n',
        encoding="utf-8",
    )
    res = CliRunner().invoke(models_command, ["list"])
    assert res.exit_code == 0, res.output
    assert "claude-x" in res.output
    assert "anthropic" in res.output


def test_list_json(monkeypatch, tmp_path: Path) -> None:
    target = _iso(monkeypatch, tmp_path)
    target.write_text(
        '[models."claude-x"]\nvendor="anthropic"\nstatus="active"\n',
        encoding="utf-8",
    )
    res = CliRunner().invoke(models_command, ["list", "--format", "json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["claude-x"]["vendor"] == "anthropic"


def test_list_empty_is_friendly(monkeypatch, tmp_path: Path) -> None:
    target = _iso(monkeypatch, tmp_path)
    target.write_text("[models]\n", encoding="utf-8")
    res = CliRunner().invoke(models_command, ["list"])
    assert res.exit_code == 0, res.output
    assert "No models defined" in res.output


def test_list_no_catalog_fails_loud(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ATP_CATALOG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    res = CliRunner().invoke(models_command, ["list"])
    assert res.exit_code != 0
    assert "atp models init" in res.output
