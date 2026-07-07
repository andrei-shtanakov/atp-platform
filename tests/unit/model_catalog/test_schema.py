from __future__ import annotations

import pytest
from pydantic import ValidationError

from atp.model_catalog.schema import ModelCatalog, ModelEntry


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


def test_catalog_passthrough_planes() -> None:
    c = ModelCatalog(
        models={"m": {"vendor": "v", "status": "active"}},
        harnesses={"h": {"shim": "x", "model_env": "Y"}},
        agents=[{"harness": "h", "model": "m", "tested": True}],
    )
    assert c.models["m"].vendor == "v"
    assert c.harnesses == {"h": {"shim": "x", "model_env": "Y"}}
    assert c.agents[0]["tested"] is True


def test_catalog_missing_models_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelCatalog(harnesses={})  # `models` is required
