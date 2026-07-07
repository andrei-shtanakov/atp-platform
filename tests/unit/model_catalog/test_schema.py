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


def test_referential_noop_when_one_plane_absent() -> None:
    # Validator early-returns when EITHER plane is None — asymmetric cases are
    # a no-op (the harness's sweep-shape guard, not the schema, requires both).
    harnesses_only = ModelCatalog(
        models={},
        harnesses={"h": {"kind": "cli", "shim": "s", "model_env": "M"}},
    )
    assert harnesses_only.agents is None

    agents_only = ModelCatalog(
        models={},
        agents=[{"harness": "anything", "model": "m"}],
    )
    # agents present, harnesses None -> validator no-op, so an "undeclared"
    # harness does NOT raise here (it would only raise if harnesses were also present).
    assert agents_only.harnesses is None


def test_catalog_defaults_default_none() -> None:
    from atp.model_catalog.schema import CatalogDefaults

    assert CatalogDefaults().default_model is None


def test_default_model_matching_key_ok() -> None:
    c = ModelCatalog(
        models={"m": {"vendor": "v", "status": "active"}},
        defaults={"default_model": "m"},
    )
    assert c.defaults.default_model == "m"


def test_default_model_matching_alias_ok() -> None:
    c = ModelCatalog(
        models={"m": {"vendor": "v", "status": "active", "aliases": ["m-latest"]}},
        defaults={"default_model": "m-latest"},
    )
    assert c.defaults.default_model == "m-latest"


def test_default_model_unknown_rejected() -> None:
    with pytest.raises(ValidationError, match="not a known model"):
        ModelCatalog(
            models={"m": {"vendor": "v", "status": "active"}},
            defaults={"default_model": "nope"},
        )


def test_default_model_none_is_noop() -> None:
    c = ModelCatalog(models={"m": {"vendor": "v", "status": "active"}}, defaults={})
    assert c.defaults.default_model is None


def test_default_model_with_empty_models_is_noop() -> None:
    # No validation when models is empty (nothing to check against).
    c = ModelCatalog(models={}, defaults={"default_model": "anything"})
    assert c.defaults.default_model == "anything"


def test_no_defaults_plane_is_noop() -> None:
    c = ModelCatalog(models={"m": {"vendor": "v", "status": "active"}})
    assert c.defaults is None
