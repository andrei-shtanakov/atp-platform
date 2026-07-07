"""Shippable model catalog: schema + D2 resolution + inert template (ADR-003b)."""

from __future__ import annotations

from atp.model_catalog.errors import (
    CatalogError,
    CatalogNotConfiguredError,
    CatalogSchemaError,
    CatalogTOMLError,
)
from atp.model_catalog.loader import (
    load_catalog,
    read_template,
    resolve_catalog_path,
    resolve_default_model,
)
from atp.model_catalog.schema import (
    AgentEntry,
    CatalogDefaults,
    HarnessEntry,
    ModelCatalog,
    ModelEntry,
)

__all__ = [
    "AgentEntry",
    "CatalogDefaults",
    "CatalogError",
    "CatalogNotConfiguredError",
    "CatalogSchemaError",
    "CatalogTOMLError",
    "HarnessEntry",
    "ModelCatalog",
    "ModelEntry",
    "load_catalog",
    "read_template",
    "resolve_catalog_path",
    "resolve_default_model",
]
