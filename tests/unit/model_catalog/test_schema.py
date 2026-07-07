from __future__ import annotations

import pytest
from pydantic import ValidationError

from atp.model_catalog.schema import (
    AgentEntry,
    HarnessEntry,
    ModelCatalog,
    ModelEntry,
)


def test_model_entry_valid() -> None:
    e = ModelEntry(vendor="anthropic", status="active")
    assert e.vendor == "anthropic"
    assert e.aliases == []


def test_model_entry_bad_status_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelEntry(vendor="x", status="experimental")  # not in the Literal


def test_model_entry_allows_extra_fields() -> None:
    e = ModelEntry(vendor="x", status="active", note="future field")
    assert e.vendor == "x"  # unknown field tolerated, not an error


def test_catalog_empty_models_is_valid() -> None:
    c = ModelCatalog(models={})
    assert c.models == {}
    assert c.harnesses is None


def test_catalog_missing_models_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelCatalog(harnesses={})  # `models` is required


def test_harness_entry_defaults() -> None:
    h = HarnessEntry(kind="cli", shim="s.py", model_env="M")
    assert h.model_flag is None
    assert h.routable is False


def test_agent_entry_defaults() -> None:
    a = AgentEntry(harness="h", model="m")
    assert a.tested is False
    assert a.routable is False


def test_entries_allow_extra_fields() -> None:
    h = HarnessEntry(kind="cli", shim="s", model_env="M", note="future")
    a = AgentEntry(harness="h", model="m", note="future")
    assert h.shim == "s" and a.model == "m"


def test_catalog_typed_planes_consistent_ok() -> None:
    c = ModelCatalog(
        models={"m": {"vendor": "v", "status": "active"}},
        harnesses={"h": {"kind": "cli", "shim": "x", "model_env": "Y"}},
        agents=[{"harness": "h", "model": "m", "tested": True}],
    )
    assert isinstance(c.harnesses["h"], HarnessEntry)
    assert c.harnesses["h"].shim == "x"
    assert isinstance(c.agents[0], AgentEntry)
    assert c.agents[0].tested is True


def test_referential_undeclared_harness_rejected() -> None:
    with pytest.raises(ValidationError, match="undeclared harness"):
        ModelCatalog(
            models={"m": {"vendor": "v", "status": "active"}},
            harnesses={"h": {"kind": "cli", "shim": "x", "model_env": "Y"}},
            agents=[{"harness": "MISSING", "model": "m"}],
        )


def test_referential_noop_when_planes_absent() -> None:
    # models-only user catalog (SP-A fork A) — validator must not fire.
    c = ModelCatalog(models={})
    assert c.harnesses is None and c.agents is None


def test_referential_present_empty_both_ok() -> None:
    c = ModelCatalog(models={}, harnesses={}, agents=[])
    assert c.harnesses == {} and c.agents == []


def test_referential_present_empty_harnesses_with_agent_fails() -> None:
    with pytest.raises(ValidationError, match="undeclared harness"):
        ModelCatalog(models={}, harnesses={}, agents=[{"harness": "x", "model": "m"}])
