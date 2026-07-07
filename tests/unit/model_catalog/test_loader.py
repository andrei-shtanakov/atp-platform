from __future__ import annotations

from pathlib import Path

import pytest

from atp.model_catalog.errors import (
    CatalogNotConfiguredError,
    CatalogSchemaError,
    CatalogTOMLError,
)
from atp.model_catalog.loader import load_catalog, resolve_catalog_path

_VALID = '[models."m"]\nvendor = "v"\nstatus = "active"\n'


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATP_CATALOG", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)


def test_atp_catalog_takes_precedence(monkeypatch, tmp_path: Path) -> None:
    _clear(monkeypatch)
    f = tmp_path / "explicit.toml"
    f.write_text(_VALID, encoding="utf-8")
    monkeypatch.setenv("ATP_CATALOG", str(f))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert resolve_catalog_path(must_exist=True) == f


def test_xdg_used_when_atp_catalog_unset(monkeypatch, tmp_path: Path) -> None:
    _clear(monkeypatch)
    xdg = tmp_path / "xdg"
    target = xdg / "atp" / "agents-catalog.toml"
    target.parent.mkdir(parents=True)
    target.write_text(_VALID, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    assert resolve_catalog_path(must_exist=True) == target


def test_empty_env_is_treated_as_unset(monkeypatch, tmp_path: Path) -> None:
    _clear(monkeypatch)
    xdg = tmp_path / "xdg"
    target = xdg / "atp" / "agents-catalog.toml"
    target.parent.mkdir(parents=True)
    target.write_text(_VALID, encoding="utf-8")
    monkeypatch.setenv("ATP_CATALOG", "")  # empty -> unset
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    assert resolve_catalog_path(must_exist=True) == target


def test_relative_atp_catalog_is_error(monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ATP_CATALOG", "relative/catalog.toml")
    with pytest.raises(CatalogNotConfiguredError, match="absolute"):
        resolve_catalog_path(must_exist=True)


def test_relative_xdg_config_home_is_error(monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/config")
    with pytest.raises(CatalogNotConfiguredError, match="absolute"):
        resolve_catalog_path(must_exist=True)


def test_empty_xdg_config_home_falls_back_to_home(monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", "")  # empty -> unset
    expected = Path.home() / ".config" / "atp" / "agents-catalog.toml"
    assert resolve_catalog_path(must_exist=False) == expected


def test_nothing_configured_fails_loud(monkeypatch, tmp_path: Path) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    with pytest.raises(CatalogNotConfiguredError, match="atp models init"):
        resolve_catalog_path(must_exist=True)


def test_init_target_returned_even_when_absent(monkeypatch, tmp_path: Path) -> None:
    _clear(monkeypatch)
    f = tmp_path / "will-create.toml"
    monkeypatch.setenv("ATP_CATALOG", str(f))
    assert resolve_catalog_path(must_exist=False) == f  # absent, but the target


def test_load_explicit_path(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text(_VALID, encoding="utf-8")
    cat = load_catalog(f)
    assert cat.models["m"].vendor == "v"


def test_load_empty_models_ok(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text("[models]\n", encoding="utf-8")
    assert load_catalog(f).models == {}


def test_load_invalid_toml(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text("this is = = not toml", encoding="utf-8")
    with pytest.raises(CatalogTOMLError):
        load_catalog(f)


def test_load_bad_status_is_schema_error(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text('[models."m"]\nvendor="v"\nstatus="nope"\n', encoding="utf-8")
    with pytest.raises(CatalogSchemaError):
        load_catalog(f)
