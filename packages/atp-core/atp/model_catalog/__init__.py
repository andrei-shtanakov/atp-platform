"""Shippable model catalog: schema + D2 resolution + inert template (ADR-003b)."""

from __future__ import annotations

from atp.model_catalog.errors import (
    CatalogError,
    CatalogMissingFileError,
    CatalogNotConfiguredError,
    CatalogSchemaError,
    CatalogTOMLError,
    CatalogWarning,
)
from atp.model_catalog.loader import (
    load_catalog,
    read_template,
    resolve_catalog_path,
    resolve_default_model,
)
from atp.model_catalog.schema import (
    KNOWN_HARNESS_KINDS,
    AgentEntry,
    CatalogDefaults,
    HarnessEntry,
    ModelCatalog,
    ModelEntry,
)

__all__ = [
    "KNOWN_HARNESS_KINDS",
    "AgentEntry",
    "CatalogDefaults",
    "CatalogError",
    "CatalogMissingFileError",
    "CatalogNotConfiguredError",
    "CatalogSchemaError",
    "CatalogTOMLError",
    "CatalogWarning",
    "HarnessEntry",
    "ModelCatalog",
    "ModelEntry",
    "load_catalog",
    "read_template",
    "resolve_catalog_path",
    "resolve_default_model",
]
