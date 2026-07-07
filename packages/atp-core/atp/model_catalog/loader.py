"""Model-catalog resolution + loading (ADR-ECO-003b D2).

$ATP_CATALOG -> $XDG_CONFIG_HOME/atp/agents-catalog.toml -> ~/.config/atp/... ->
fail-loud. Env paths must be absolute; empty string = unset; relative = error.
"""

from __future__ import annotations

import os
import tomllib
from importlib.resources import files
from pathlib import Path

from pydantic import ValidationError

from atp.model_catalog.errors import (
    CatalogNotConfiguredError,
    CatalogSchemaError,
    CatalogTOMLError,
)
from atp.model_catalog.schema import ModelCatalog

_XDG_RELATIVE = Path("atp") / "agents-catalog.toml"
_INIT_HINT = (
    "model catalog not configured: run 'atp models init' or set $ATP_CATALOG "
    "to an absolute file path"
)


def _env_path(var: str) -> Path | None:
    """Absolute Path from an env var, or None if unset/empty.

    Empty string is unset. A relative path is a misconfiguration ->
    CatalogNotConfiguredError with an explicit 'must be absolute' message
    (distinct from the init hint).
    """
    raw = os.environ.get(var)
    if not raw:  # None or empty string
        return None
    p = Path(raw)
    if not p.is_absolute():
        raise CatalogNotConfiguredError(f"{var} must be an absolute path, got {raw!r}")
    return p


def resolve_catalog_path(*, must_exist: bool) -> Path:
    """Resolve the catalog path (D2). must_exist=True requires an existing file;
    must_exist=False returns the first candidate as a creation target."""
    candidates: list[Path] = []
    explicit = _env_path("ATP_CATALOG")
    if explicit is not None:
        if explicit.exists() and not explicit.is_file():
            raise CatalogNotConfiguredError(
                f"$ATP_CATALOG points at a non-file (expected a file path): {explicit}"
            )
        candidates.append(explicit)
    xdg = _env_path("XDG_CONFIG_HOME")
    if xdg is not None:
        if xdg.exists() and not xdg.is_dir():
            raise CatalogNotConfiguredError(
                f"$XDG_CONFIG_HOME is not a directory: {xdg}"
            )
        candidates.append(xdg / _XDG_RELATIVE)
    else:
        candidates.append(Path.home() / ".config" / _XDG_RELATIVE)

    if not must_exist:
        return candidates[0]
    for c in candidates:
        if c.is_file():
            return c
    raise CatalogNotConfiguredError(_INIT_HINT)


def load_catalog(path: Path | None = None) -> ModelCatalog:
    """Load + validate a catalog. path given -> that file; None -> D2 resolution."""
    target = path if path is not None else resolve_catalog_path(must_exist=True)
    if not target.is_file():
        raise CatalogNotConfiguredError(f"catalog file not found: {target}")
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CatalogTOMLError(f"{target} is not valid TOML: {exc}") from exc
    try:
        return ModelCatalog(**data)
    except ValidationError as exc:
        raise CatalogSchemaError(f"{target} failed schema validation: {exc}") from exc


def read_template() -> str:
    """Return the inert starter catalog shipped as package data.

    Used by `atp models init`; never loaded as live catalog data.
    """
    return (
        files("atp.model_catalog")
        .joinpath("data/template.toml")
        .read_text(encoding="utf-8")
    )
