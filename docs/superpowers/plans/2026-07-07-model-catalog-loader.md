# Model-catalog loader (ADR-003b SP-A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a model-catalog loader (schema + D2 resolution + inert template) in the `atp`
package, plus a minimal `atp models init/list` CLI — a purely additive first increment of 003b.

**Architecture:** A new `packages/atp-core/atp/model_catalog/` package (reached in the dev tree
via an `atp/model_catalog` symlink, matching `atp/cost`). It parses `agents-catalog.toml`:
strict `models` plane, passthrough `harnesses`/`agents` planes. Resolution is `$ATP_CATALOG` →
XDG → fail-loud with typed errors. The wheel ships loader + schema + an inert template; the
active `method/agents-catalog.toml` stays out of the wheel.

**Tech Stack:** Python 3.12+, pydantic v2, tomllib, click (CLI), hatchling (packaging), uv,
pytest.

## Global Constraints

- Package management: **uv only**, never pip. Run tools via `uv run`.
- Line length **88**; full type hints; `uv run ruff format . && uv run ruff check . && uv run
  pyrefly check` clean after every task.
- **Resolution order (ADR-003b D2):** `$ATP_CATALOG` → `$XDG_CONFIG_HOME/atp/agents-catalog.toml`
  → `~/.config/atp/agents-catalog.toml` → fail-loud. `$ATP_CATALOG`/`--path`/`$XDG_CONFIG_HOME`
  are **file/dir paths that must be absolute**; empty string = unset; relative = error.
- **Inert template (D1):** no active `claude-*`/`gpt-*` or any real model tables — only an
  active empty `[models]` table + commented placeholder examples.
- **User contract = `models` plane** (strict); `harnesses`/`agents` = optional passthrough.
- **Additive only:** do not touch `method/run_pipe_check.py`, `atp/core/settings.py`, or
  `atp/catalog/`. Those are later 003b increments (SP-E / SP-C / SP-D).
- Tests must never read or write the real `~/.config`; use `monkeypatch` + `tmp_path`.

Spec: `docs/superpowers/specs/2026-07-07-model-catalog-loader-design.md`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `packages/atp-core/atp/model_catalog/__init__.py` | public exports |
| `packages/atp-core/atp/model_catalog/errors.py` | `CatalogError` + 3 typed subclasses |
| `packages/atp-core/atp/model_catalog/schema.py` | `ModelEntry`, `ModelCatalog` |
| `packages/atp-core/atp/model_catalog/loader.py` | `resolve_catalog_path`, `load_catalog`, `read_template` |
| `packages/atp-core/atp/model_catalog/data/template.toml` | inert starter (package data) |
| `atp/model_catalog` (symlink) | dev-tree import bridge → `../packages/atp-core/atp/model_catalog` |
| `packages/atp-core/pyproject.toml` | add `"atp/model_catalog"` to wheel packages |
| `atp/cli/commands/models.py` | `atp models` group (init, list) |
| `atp/cli/main.py` | register `models_command` |
| `tests/unit/model_catalog/test_schema.py` | schema unit tests |
| `tests/unit/model_catalog/test_loader.py` | resolution + load unit tests |
| `tests/unit/model_catalog/test_template.py` | packaged-template test |
| `tests/unit/model_catalog/test_models_cli.py` | CLI tests |

---

## Task 1: Package skeleton + typed errors + schema

**Files:**
- Create: `packages/atp-core/atp/model_catalog/__init__.py`
- Create: `packages/atp-core/atp/model_catalog/errors.py`
- Create: `packages/atp-core/atp/model_catalog/schema.py`
- Create symlink: `atp/model_catalog` → `../packages/atp-core/atp/model_catalog`
- Create: `tests/unit/model_catalog/__init__.py` (empty), `tests/unit/model_catalog/test_schema.py`

**Interfaces:**
- Produces:
  - `CatalogError(Exception)`, `CatalogNotConfiguredError`, `CatalogTOMLError`,
    `CatalogSchemaError` (all subclass `CatalogError`).
  - `ModelEntry` — pydantic, `vendor: str`, `status: Literal["active","deprecated","retired"]`,
    `aliases: list[str] = []`, `model_config = ConfigDict(extra="allow")`.
  - `ModelCatalog` — pydantic, `models: dict[str, ModelEntry]` (required, may be empty),
    `harnesses: dict[str, Any] | None = None`, `agents: list[dict[str, Any]] | None = None`,
    `model_config = ConfigDict(extra="allow")`.

- [ ] **Step 1: Create the package skeleton + dev-tree symlink**

```bash
mkdir -p packages/atp-core/atp/model_catalog/data
mkdir -p tests/unit/model_catalog
ln -s ../packages/atp-core/atp/model_catalog atp/model_catalog
touch tests/unit/model_catalog/__init__.py
```

Verify the symlink resolves like the sibling `atp/cost`:

```bash
ls -la atp/model_catalog   # -> ../packages/atp-core/atp/model_catalog
```

- [ ] **Step 2: Write `errors.py`**

```python
"""Typed errors for the model catalog (ADR-ECO-003b).

Distinct types so the CLI maps to clear messages + non-zero exit and future
evaluator/harness consumers can catch programmatically.
"""

from __future__ import annotations


class CatalogError(Exception):
    """Base for all model-catalog errors."""


class CatalogNotConfiguredError(CatalogError):
    """No usable catalog location.

    Two message forms (the text differentiates them): 'not configured' (needs
    `atp models init`) vs 'configured but invalid' (an env path must be absolute).
    """


class CatalogTOMLError(CatalogError):
    """The catalog file exists but is not valid TOML."""


class CatalogSchemaError(CatalogError):
    """The catalog parsed as TOML but failed schema validation."""
```

- [ ] **Step 3: Write `schema.py`**

```python
"""Model-catalog schema (ADR-ECO-003b).

The `models` plane is the user-runtime contract: strict on the known fields,
tolerant of unknown ones (forward-compat + dev-SSOT extras). `harnesses`/`agents`
are optional dev-SSOT passthrough planes, kept raw so the dev-SSOT file parses;
they are formalized when the harness migrates (SP-E).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ModelEntry(BaseModel):
    """One model in the `models` plane."""

    model_config = ConfigDict(extra="allow")

    vendor: str
    status: Literal["active", "deprecated", "retired"]
    aliases: list[str] = []


class ModelCatalog(BaseModel):
    """A parsed model catalog."""

    model_config = ConfigDict(extra="allow")

    models: dict[str, ModelEntry]
    harnesses: dict[str, Any] | None = None
    agents: list[dict[str, Any]] | None = None
```

- [ ] **Step 4: Write `__init__.py` (errors + schema exports)**

```python
"""Shippable model catalog: schema + D2 resolution + inert template (ADR-003b)."""

from __future__ import annotations

from atp.model_catalog.errors import (
    CatalogError,
    CatalogNotConfiguredError,
    CatalogSchemaError,
    CatalogTOMLError,
)
from atp.model_catalog.schema import ModelCatalog, ModelEntry

__all__ = [
    "CatalogError",
    "CatalogNotConfiguredError",
    "CatalogSchemaError",
    "CatalogTOMLError",
    "ModelCatalog",
    "ModelEntry",
]
```

- [ ] **Step 5: Write the failing schema tests**

`tests/unit/model_catalog/test_schema.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/model_catalog/test_schema.py -v`
Expected: 6 PASS.

- [ ] **Step 7: Format, lint, type-check**

Run: `uv run ruff format . && uv run ruff check . && uv run pyrefly check`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add packages/atp-core/atp/model_catalog/__init__.py packages/atp-core/atp/model_catalog/errors.py packages/atp-core/atp/model_catalog/schema.py atp/model_catalog tests/unit/model_catalog/
git commit -m "feat(model-catalog): package skeleton, typed errors, schema (ADR-003b SP-A)"
```

---

## Task 2: Loader — D2 resolution + `load_catalog`

**Files:**
- Create: `packages/atp-core/atp/model_catalog/loader.py`
- Modify: `packages/atp-core/atp/model_catalog/__init__.py` (add loader exports)
- Create: `tests/unit/model_catalog/test_loader.py`

**Interfaces:**
- Consumes: `ModelCatalog` (schema), the four error types (Task 1).
- Produces:
  - `resolve_catalog_path(*, must_exist: bool) -> Path`
  - `load_catalog(path: Path | None = None) -> ModelCatalog`

- [ ] **Step 1: Write the failing loader tests**

`tests/unit/model_catalog/test_loader.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/model_catalog/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: atp.model_catalog.loader`.

- [ ] **Step 3: Write `loader.py`**

```python
"""Model-catalog resolution + loading (ADR-ECO-003b D2).

$ATP_CATALOG -> $XDG_CONFIG_HOME/atp/agents-catalog.toml -> ~/.config/atp/... ->
fail-loud. Env paths must be absolute; empty string = unset; relative = error.
"""

from __future__ import annotations

import os
import tomllib
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
        raise CatalogNotConfiguredError(
            f"{var} must be an absolute path, got {raw!r}"
        )
    return p


def resolve_catalog_path(*, must_exist: bool) -> Path:
    """Resolve the catalog path (D2). must_exist=True requires an existing file;
    must_exist=False returns the first candidate as a creation target."""
    candidates: list[Path] = []
    explicit = _env_path("ATP_CATALOG")
    if explicit is not None:
        candidates.append(explicit)
    xdg = _env_path("XDG_CONFIG_HOME")
    if xdg is not None:
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
        raise CatalogSchemaError(
            f"{target} failed schema validation: {exc}"
        ) from exc
```

- [ ] **Step 4: Add loader exports to `__init__.py`**

Add the import and extend `__all__`:

```python
from atp.model_catalog.loader import load_catalog, resolve_catalog_path
```

Add `"load_catalog"` and `"resolve_catalog_path"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/model_catalog/test_loader.py -v`
Expected: 10 PASS.

- [ ] **Step 6: Format, lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add packages/atp-core/atp/model_catalog/loader.py packages/atp-core/atp/model_catalog/__init__.py tests/unit/model_catalog/test_loader.py
git commit -m "feat(model-catalog): D2 resolution + load_catalog with typed errors"
```

---

## Task 3: Inert template + `read_template` + packaging

**Files:**
- Create: `packages/atp-core/atp/model_catalog/data/template.toml`
- Modify: `packages/atp-core/atp/model_catalog/loader.py` (add `read_template`)
- Modify: `packages/atp-core/atp/model_catalog/__init__.py` (export `read_template`)
- Modify: `packages/atp-core/pyproject.toml` (add `"atp/model_catalog"` to wheel packages)
- Create: `tests/unit/model_catalog/test_template.py`

**Interfaces:**
- Consumes: `load_catalog` (Task 2).
- Produces: `read_template() -> str` (the inert starter text shipped as package data).

- [ ] **Step 1: Write the inert template**

`packages/atp-core/atp/model_catalog/data/template.toml`:

```toml
# ATP model catalog — the models your instance uses (ADR-ECO-003b).
#
# ATP resolves this file in order:
#   $ATP_CATALOG  ->  $XDG_CONFIG_HOME/atp/agents-catalog.toml  ->
#   ~/.config/atp/agents-catalog.toml
#
# This starter ships intentionally EMPTY — ATP endorses no model by default.
# Uncomment and adapt an entry below (the names are placeholders, not real
# models), then run `atp models list` to confirm it loads.
#
#   [models."your-model-id"]
#   vendor  = "your-vendor"
#   status  = "active"        # active | deprecated | retired
#   aliases = []

[models]
```

- [ ] **Step 2: Write the failing template test**

`tests/unit/model_catalog/test_template.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/model_catalog/test_template.py -v`
Expected: FAIL — `ImportError: cannot import name 'read_template'`.

- [ ] **Step 4: Add `read_template` to `loader.py`**

Add at the top of `loader.py` imports:

```python
from importlib.resources import files
```

Add the function:

```python
def read_template() -> str:
    """Return the inert starter catalog shipped as package data.

    Used by `atp models init`; never loaded as live catalog data.
    """
    return (
        files("atp.model_catalog")
        .joinpath("data/template.toml")
        .read_text(encoding="utf-8")
    )
```

- [ ] **Step 5: Export `read_template` from `__init__.py`**

Change the loader import line to:

```python
from atp.model_catalog.loader import load_catalog, read_template, resolve_catalog_path
```

Add `"read_template"` to `__all__`.

- [ ] **Step 6: Add the package to the wheel build**

In `packages/atp-core/pyproject.toml`, add `"atp/model_catalog"` to the
`[tool.hatch.build.targets.wheel]` `packages` list (alphabetical-ish is fine; the existing
list has one entry per line):

```toml
packages = [
    "atp/protocol",
    "atp/core",
    "atp/loader",
    "atp/chaos",
    "atp/cost",
    "atp/model_catalog",
    "atp/scoring",
    "atp/statistics",
    "atp/streaming",
]
```

- [ ] **Step 7: Run the template test to verify it passes**

Run: `uv run pytest tests/unit/model_catalog/test_template.py -v`
Expected: 4 PASS.

- [ ] **Step 8: Verify the template actually ships in the wheel (the packaging guard)**

An editable-install resource test passes even if the wheel would drop the file, so build the
wheel and inspect it:

```bash
uv build --wheel --package atp-core -o /tmp/mc-wheelcheck
python -c "import zipfile, glob; w=glob.glob('/tmp/mc-wheelcheck/*.whl')[0]; names=zipfile.ZipFile(w).namelist(); assert any(n.endswith('atp/model_catalog/data/template.toml') for n in names), 'template.toml NOT in wheel!'; print('OK: template.toml shipped in', w)"
```

Expected: `OK: template.toml shipped in …`. If it fails, add an explicit
`[tool.hatch.build.targets.wheel.force-include]` mapping for the `data/` dir and re-run.

- [ ] **Step 9: Format, lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add packages/atp-core/atp/model_catalog/data/template.toml packages/atp-core/atp/model_catalog/loader.py packages/atp-core/atp/model_catalog/__init__.py packages/atp-core/pyproject.toml tests/unit/model_catalog/test_template.py
git commit -m "feat(model-catalog): inert template + read_template + wheel packaging"
```

---

## Task 4: `atp models` CLI (init + list)

**Files:**
- Create: `atp/cli/commands/models.py`
- Modify: `atp/cli/main.py` (import + `cli.add_command(models_command)`)
- Create: `tests/unit/model_catalog/test_models_cli.py`

**Interfaces:**
- Consumes: `resolve_catalog_path`, `load_catalog`, `read_template`, `CatalogError` (Tasks 1-3).
- Produces: `models_command` (a click group named `models`).

- [ ] **Step 1: Write the failing CLI tests**

`tests/unit/model_catalog/test_models_cli.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/model_catalog/test_models_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: atp.cli.commands.models`.

- [ ] **Step 3: Write `atp/cli/commands/models.py`**

```python
"""CLI commands for the ATP model catalog (`atp models`)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from atp.model_catalog import (
    CatalogError,
    load_catalog,
    read_template,
    resolve_catalog_path,
)


@click.group(name="models")
def models_command() -> None:
    """Manage the model catalog (which models this instance uses)."""


@models_command.command(name="init")
@click.option(
    "--path",
    "path",
    type=click.Path(path_type=Path),
    default=None,
    help="Target file to create (overrides $ATP_CATALOG / XDG).",
)
@click.option("--force", is_flag=True, help="Overwrite an existing catalog.")
def init_cmd(path: Path | None, force: bool) -> None:
    """Write a starter catalog to the resolved user-config path."""
    try:
        target = path if path is not None else resolve_catalog_path(must_exist=False)
    except CatalogError as exc:
        raise click.ClickException(str(exc)) from exc
    if target.exists() and not force:
        raise click.ClickException(
            f"{target} already exists; pass --force to overwrite"
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(read_template(), encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(f"cannot write {target}: {exc}") from exc
    click.echo(f"Wrote model catalog: {target}")
    click.echo("Edit it to add your models (the starter ships empty).")


@models_command.command(name="list")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format.",
)
def list_cmd(fmt: str) -> None:
    """List the models in the resolved catalog."""
    try:
        path = resolve_catalog_path(must_exist=True)
        catalog = load_catalog(path)
    except CatalogError as exc:
        raise click.ClickException(str(exc)) from exc
    models = catalog.models
    if fmt == "json":
        click.echo(
            json.dumps(
                {name: m.model_dump() for name, m in models.items()}, indent=2
            )
        )
        return
    if not models:
        click.echo(f"No models defined yet — edit {path}")
        return
    for name, m in models.items():
        aliases = ", ".join(m.aliases) if m.aliases else "-"
        click.echo(f"{name:30s}  {m.vendor:12s}  {m.status:11s}  {aliases}")
```

- [ ] **Step 4: Register the command in `atp/cli/main.py`**

Add an import alongside the other `from atp.cli.commands.… import …_command` lines (keep them
alphabetical — after `init_command`):

```python
from atp.cli.commands.models import models_command
```

Add a registration alongside the other `cli.add_command(...)` calls:

```python
cli.add_command(models_command)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/model_catalog/test_models_cli.py -v`
Expected: 8 PASS.

- [ ] **Step 6: Smoke the wired-up CLI**

Run: `uv run atp models --help` then
`ATP_CATALOG=/tmp/mc-smoke.toml uv run atp models init && ATP_CATALOG=/tmp/mc-smoke.toml uv run atp models list`
Expected: `init` writes `/tmp/mc-smoke.toml`; `list` prints "No models defined yet — edit
/tmp/mc-smoke.toml". Then `rm /tmp/mc-smoke.toml`.

- [ ] **Step 7: Format, lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add atp/cli/commands/models.py atp/cli/main.py tests/unit/model_catalog/test_models_cli.py
git commit -m "feat(model-catalog): atp models init/list CLI"
```

---

## Task 5: Docs pointer

**Files:**
- Modify: `CLAUDE.md` (Development Commands + a one-line note)
- Modify: `TODO.md` (mark 003b SP-A, link spec/plan)

- [ ] **Step 1: Add the command to CLAUDE.md**

In `CLAUDE.md`, under the CLI commands block (near `uv run atp catalog`), add:

```
uv run atp models init             # Create a starter model catalog (~/.config/atp/)
uv run atp models list             # List models in the resolved catalog
```

And append one sentence to the methodology/catalog area noting the shippable model catalog:
the loader/schema/inert-template live in `atp/model_catalog/` (ADR-003b SP-A); resolution is
`$ATP_CATALOG` → XDG → fail-loud; the active `method/agents-catalog.toml` stays the dev-SSOT and
is not shipped.

- [ ] **Step 2: Tick the epic in TODO.md**

Under the active tasks, add a done line for 003b SP-A linking the spec and this plan, and note
the remaining increments (SP-E harness migration, SP-C evaluator unify, SP-D rename).

- [ ] **Step 3: Full-suite sanity + commit**

Run: `uv run pytest tests/unit/model_catalog/ -v && uv run pyrefly check`
Expected: all model_catalog tests pass; pyrefly clean.

```bash
git add CLAUDE.md TODO.md
git commit -m "docs(model-catalog): CLI commands + 003b SP-A status"
```

---

## Self-Review

**Spec coverage:**
- Module home + naming (`model_catalog`) → Task 1. ✓
- Schema: strict `ModelEntry`, `models` required-but-empty-ok, passthrough `harnesses`/`agents`,
  `extra="allow"` → Task 1. ✓
- Typed errors (`CatalogError` + 3) → Task 1. ✓
- D2 resolution: precedence, empty=unset, relative=error (distinct message), `must_exist` two
  modes, file-not-dir → Task 2. ✓
- `load_catalog` (explicit path / D2), TOML vs schema errors → Task 2. ✓
- Inert template with active empty `[models]`, no active models → Task 3. ✓
- Packaging: `"atp/model_catalog"` in hatch packages + `importlib.resources` test + wheel-build
  guard → Task 3. ✓
- Minimal CLI `init`/`list`, `--force`, `--format table|json`, path-first resolution for the
  empty message, exit codes (empty→0, fail-loud→non-zero) → Task 4. ✓
- Symlink import bridge → Task 1. ✓
- Non-goals (harness/settings/`atp/catalog/` untouched) → honored; no task modifies them. ✓

**Placeholder scan:** no "TBD/TODO/handle errors" placeholders; every code step is complete.
The only non-literal is the CLAUDE.md prose sentence (Task 5), which is documentation.

**Type consistency:** `ModelEntry`, `ModelCatalog`, `CatalogError`(+3), `resolve_catalog_path(*,
must_exist)`, `load_catalog(path=None)`, `read_template()`, `models_command` are used with
identical names/signatures across Tasks 1-5. `__init__.py` exports grow monotonically (Task 1
errors+schema → Task 2 loader → Task 3 read_template) with no renames.
