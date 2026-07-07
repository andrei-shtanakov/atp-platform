from __future__ import annotations

import tomllib
from importlib.resources import files

from atp.model_catalog.loader import load_catalog, read_template


def test_read_template_returns_text() -> None:
    text = read_template()
    assert "[models]" in text


def test_packaged_template_is_reachable_as_resource() -> None:
    # The exact path the wheel must ship (ADR-003b packaging boundary).
    res = files("atp.model_catalog").joinpath("data/template.toml")
    assert res.is_file()
    assert res.read_text(encoding="utf-8") == read_template()


def test_template_is_valid_and_loads_without_error(tmp_path) -> None:
    # The active empty [models] table must let a freshly-init'd file load.
    f = tmp_path / "agents-catalog.toml"
    f.write_text(read_template(), encoding="utf-8")
    cat = load_catalog(f)
    assert cat.models == {}


def test_template_endorses_no_active_model() -> None:
    # No real model tables (only commented placeholders) — guard against
    # endorsement creep (ADR D1).
    parsed = tomllib.loads(read_template())
    assert parsed.get("models", {}) == {}
    assert "harnesses" not in parsed
    assert "agents" not in parsed
