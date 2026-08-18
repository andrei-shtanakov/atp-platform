from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from atp.model_catalog.errors import CatalogWarning
from atp.model_catalog.schema import (
    KNOWN_HARNESS_KINDS,
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


def test_harness_entry_shim_is_optional() -> None:
    # `shim` is ATP sweep machinery, not part of the shared catalog contract;
    # requiring it would reject contract-valid catalogs (see V-rule tests below).
    h = HarnessEntry(kind="cli", model_env="M")
    assert h.shim is None


def test_agent_entry_agent_id() -> None:
    assert AgentEntry(harness="h", model="m").agent_id == "h@m"


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


# --- cross-plane rules V2..V6 (shared catalog-conformance vocabulary) --------

_ACTIVE = {"vendor": "v", "status": "active"}
_HARNESS = {"kind": "cli", "shim": "x", "model_env": "Y"}


def test_v2_undeclared_model_rejected() -> None:
    with pytest.raises(ValidationError, match="V2:"):
        ModelCatalog(
            models={"m": _ACTIVE},
            harnesses={"h": _HARNESS},
            agents=[{"harness": "h", "model": "MISSING"}],
        )


def test_v2_noop_when_models_plane_empty() -> None:
    # An empty `models` plane declares nothing to check against (same posture as
    # the defaults.default_model validator).
    c = ModelCatalog(
        models={},
        harnesses={"h": _HARNESS},
        agents=[{"harness": "h", "model": "whatever"}],
    )
    assert c.agents is not None


def test_v3_retired_reference_rejected() -> None:
    with pytest.raises(ValidationError, match="V3:"):
        ModelCatalog(
            models={"old": {"vendor": "v", "status": "retired"}},
            harnesses={"h": _HARNESS},
            agents=[{"harness": "h", "model": "old"}],
        )


def test_v3_retired_but_unreferenced_is_ok() -> None:
    # retired-not-deleted is the SSOT regression guard; only a live reference fails.
    c = ModelCatalog(
        models={"m": _ACTIVE, "old": {"vendor": "v", "status": "retired"}},
        harnesses={"h": _HARNESS},
        agents=[{"harness": "h", "model": "m"}],
    )
    assert set(c.models) == {"m", "old"}


def test_v4_duplicate_agent_id_rejected() -> None:
    with pytest.raises(ValidationError, match="V4:"):
        ModelCatalog(
            models={"m": _ACTIVE},
            harnesses={"h": _HARNESS},
            agents=[
                {"harness": "h", "model": "m", "tested": True},
                {"harness": "h", "model": "m", "tested": False},
            ],
        )


def test_v4_fires_without_the_models_plane() -> None:
    # A duplicate join key is ambiguous regardless of which other planes exist.
    with pytest.raises(ValidationError, match="V4:"):
        ModelCatalog(
            models={},
            agents=[{"harness": "h", "model": "m"}, {"harness": "h", "model": "m"}],
        )


def test_v5_routable_agent_under_non_routable_harness_rejected() -> None:
    with pytest.raises(ValidationError, match="V5:"):
        ModelCatalog(
            models={"m": _ACTIVE},
            harnesses={"h": {**_HARNESS, "routable": False}},
            agents=[{"harness": "h", "model": "m", "routable": True}],
        )


def test_v5_routable_agent_under_routable_harness_ok() -> None:
    c = ModelCatalog(
        models={"m": _ACTIVE},
        harnesses={"h": {**_HARNESS, "routable": True}},
        agents=[{"harness": "h", "model": "m", "routable": True}],
    )
    assert c.agents is not None and c.agents[0].routable is True


def test_v6_deprecated_reference_warns_but_loads() -> None:
    with pytest.warns(CatalogWarning, match="V6:"):
        c = ModelCatalog(
            models={"m": {"vendor": "v", "status": "deprecated"}},
            harnesses={"h": _HARNESS},
            agents=[{"harness": "h", "model": "m"}],
        )
    assert c.agents is not None


def test_v6_silent_when_no_deprecated_reference() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", CatalogWarning)
        ModelCatalog(
            models={"m": _ACTIVE, "old": {"vendor": "v", "status": "deprecated"}},
            harnesses={"h": _HARNESS},
            agents=[{"harness": "h", "model": "m"}],
        )


def test_v7_unknown_harness_kind_warns_but_loads() -> None:
    with pytest.warns(CatalogWarning, match="V7:"):
        c = ModelCatalog(
            models={"m": _ACTIVE},
            harnesses={"h": {**_HARNESS, "kind": "container"}},
            agents=[{"harness": "h", "model": "m"}],
        )
    assert c.harnesses is not None and c.harnesses["h"].kind == "container"


def test_v7_names_the_offending_harness_and_kind() -> None:
    with pytest.warns(CatalogWarning, match=r"h=container") as caught:
        ModelCatalog(models={}, harnesses={"h": {**_HARNESS, "kind": "container"}})
    assert "V7:" in str(caught[0].message)


def test_v7_silent_for_every_known_kind() -> None:
    # The vocabulary is ADR-ECO-003's; restating it must not flag its own members.
    with warnings.catch_warnings():
        warnings.simplefilter("error", CatalogWarning)
        for kind in KNOWN_HARNESS_KINDS:
            ModelCatalog(models={}, harnesses={"h": {**_HARNESS, "kind": kind}})


def test_v7_kind_check_needs_no_agents_plane() -> None:
    # A harness-only catalog still gets its kinds checked.
    with pytest.warns(CatalogWarning, match="V7:"):
        ModelCatalog(models={}, harnesses={"h": {**_HARNESS, "kind": "wat"}})
