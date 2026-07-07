# SP-E harness catalog migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the pipe-check harness onto the shared `atp.model_catalog.load_catalog`, and
formalize the `harnesses`/`agents` planes into typed schema with a referential-integrity validator.

**Architecture:** Add `HarnessEntry`/`AgentEntry` + a referential validator to
`atp.model_catalog.schema` (schema owns document shape + integrity). Rewrite the harness's
`_load_agent_catalog` to call `load_catalog(path)` and project the typed catalog into its existing
`HARNESSES`/`AGENT_MODELS`/`AGENTS` shapes (harness owns sweep usage: the tested filter, the
projection, and re-adding the dev-SSOT error hint).

**Tech Stack:** Python 3.12+, pydantic v2 (`model_validator`), tomllib, uv, pytest.

## Global Constraints

- Package management **uv only**; run tools via `uv run`. Line length **88**; full type hints;
  `uv run ruff format . && uv run ruff check . && uv run pyrefly check` clean after every task.
- **Schema owns document shape + referential integrity; the harness owns sweep usage.**
- **Referential validator fires only when BOTH `harnesses` and `agents` are present** (not None);
  present-empty counts as present. Absent (None) planes → no-op (preserves SP-A fork A: a
  `models`-only user catalog must still load).
- **Error chain:** a `model_validator` raising `ValueError` surfaces as
  `pydantic.ValidationError`; `load_catalog` wraps validation failures into `CatalogSchemaError`.
  Direct schema-construction tests assert `ValidationError`; loader tests assert
  `CatalogSchemaError`.
- **Sweep-order invariant:** `AGENT_MODELS` preserves the `[[agents]]` order from the TOML.
- **Schema ≠ filesystem:** never validate shim-path/binary existence in the schema; runtime
  availability stays in `_preflight()` (unchanged by SP-E).
- **Error wrapping preserves cause:** `raise RuntimeError(...) from exc`.
- Additive to `atp/core/settings.py` / `atp/catalog/` — **do not touch them** (SP-C / SP-D).

Spec: `docs/superpowers/specs/2026-07-07-sp-e-harness-catalog-migration-design.md`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `packages/atp-core/atp/model_catalog/schema.py` | add `HarnessEntry`/`AgentEntry`; type the planes; referential validator |
| `packages/atp-core/atp/model_catalog/__init__.py` | export `HarnessEntry`, `AgentEntry` |
| `method/run_pipe_check.py` | `_load_agent_catalog` → `load_catalog` + projection + sweep guard + error wrap |
| `tests/unit/model_catalog/test_schema.py` | update the SP-A passthrough test → typed; add entry + validator tests |
| `tests/unit/model_catalog/test_loader.py` | add referential-error-via-loader → `CatalogSchemaError` |
| `tests/unit/method_spawners/test_run_pipe_check.py` | harness projection / order / sweep-guard / error-wrap tests |
| `CLAUDE.md`, `TODO.md` | docs pointer |

---

## Task 1: Formalize the planes + referential validator

**Files:**
- Modify: `packages/atp-core/atp/model_catalog/schema.py`
- Modify: `packages/atp-core/atp/model_catalog/__init__.py`
- Modify: `tests/unit/model_catalog/test_schema.py`
- Modify: `tests/unit/model_catalog/test_loader.py`

**Interfaces:**
- Produces:
  - `HarnessEntry(kind: str, shim: str, model_env: str, model_flag: str | None = None,
    routable: bool = False)` — `extra="allow"`.
  - `AgentEntry(harness: str, model: str, tested: bool = False, routable: bool = False)` —
    `extra="allow"`.
  - `ModelCatalog.harnesses: dict[str, HarnessEntry] | None`, `ModelCatalog.agents:
    list[AgentEntry] | None`, with the referential `model_validator`.
- Consumes: `load_catalog` (SP-A) for the loader test.

- [ ] **Step 1: Update the SP-A passthrough test to the new typed reality (it will fail first)**

The existing `test_catalog_passthrough_planes` in `tests/unit/model_catalog/test_schema.py`
constructs untyped planes and asserts raw-dict access — that contract is changing. Replace it
with a typed version and add the new-entry + validator tests. Append/replace so the file has:

```python
from atp.model_catalog.schema import (
    AgentEntry,
    HarnessEntry,
    ModelCatalog,
    ModelEntry,
)


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
        ModelCatalog(
            models={}, harnesses={}, agents=[{"harness": "x", "model": "m"}]
        )
```

Ensure the file's imports include `import pytest` and `from pydantic import ValidationError`
(add if absent). Delete the old `test_catalog_passthrough_planes` (superseded by
`test_catalog_typed_planes_consistent_ok`).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/model_catalog/test_schema.py -v`
Expected: FAIL — `ImportError: cannot import name 'HarnessEntry'`.

- [ ] **Step 3: Implement the schema changes**

Replace the body of `packages/atp-core/atp/model_catalog/schema.py` with:

```python
"""Model-catalog schema (ADR-ECO-003b).

The `models` plane is the user-runtime contract: strict on the known fields,
tolerant of unknown ones. `harnesses`/`agents` are the dev-SSOT planes, typed in
SP-E; a referential validator ties agents to declared harnesses when both planes
are present (a models-only user catalog is a no-op — SP-A fork A).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class ModelEntry(BaseModel):
    """One model in the `models` plane."""

    model_config = ConfigDict(extra="allow")

    vendor: str
    status: Literal["active", "deprecated", "retired"]
    aliases: list[str] = []


class HarnessEntry(BaseModel):
    """One harness in the dev-SSOT `harnesses` plane."""

    model_config = ConfigDict(extra="allow")

    kind: str
    shim: str
    model_env: str
    model_flag: str | None = None
    routable: bool = False


class AgentEntry(BaseModel):
    """One agent in the dev-SSOT `agents` plane."""

    model_config = ConfigDict(extra="allow")

    harness: str
    model: str
    tested: bool = False
    routable: bool = False


class ModelCatalog(BaseModel):
    """A parsed model catalog."""

    model_config = ConfigDict(extra="allow")

    models: dict[str, ModelEntry]
    harnesses: dict[str, HarnessEntry] | None = None
    agents: list[AgentEntry] | None = None

    @model_validator(mode="after")
    def _agents_reference_declared_harnesses(self) -> "ModelCatalog":
        # Referential integrity fires only when BOTH planes are present
        # (present-empty counts as present); a models-only user catalog is a
        # no-op, preserving SP-A fork A.
        if self.harnesses is None or self.agents is None:
            return self
        declared = set(self.harnesses)
        undeclared = sorted(
            {a.harness for a in self.agents if a.harness not in declared}
        )
        if undeclared:
            raise ValueError(
                f"agents reference undeclared harness(es): {undeclared}"
            )
        return self
```

- [ ] **Step 4: Export the new entries in `__init__.py`**

Change the schema import line to:

```python
from atp.model_catalog.schema import AgentEntry, HarnessEntry, ModelCatalog, ModelEntry
```

Add `"AgentEntry"` and `"HarnessEntry"` to `__all__` (keep it sorted).

- [ ] **Step 5: Run schema tests to verify they pass**

Run: `uv run pytest tests/unit/model_catalog/test_schema.py -v`
Expected: all PASS (the 6 original + the new ones; the old passthrough test is gone).

- [ ] **Step 6: Add the loader-level referential test**

Append to `tests/unit/model_catalog/test_loader.py`:

```python
def test_load_referential_error_is_schema_error(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text(
        '[models."m"]\nvendor="v"\nstatus="active"\n'
        '[harnesses.h]\nkind="cli"\nshim="s"\nmodel_env="M"\n'
        '[[agents]]\nharness="MISSING"\nmodel="m"\n',
        encoding="utf-8",
    )
    with pytest.raises(CatalogSchemaError):
        load_catalog(f)
```

Confirm `CatalogSchemaError` and `pytest`/`Path` are imported in that file (they are, from SP-A).

- [ ] **Step 7: Run loader tests**

Run: `uv run pytest tests/unit/model_catalog/test_loader.py -v`
Expected: all PASS (the referential `ValidationError` is wrapped into `CatalogSchemaError`).

- [ ] **Step 8: Format, lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add packages/atp-core/atp/model_catalog/schema.py packages/atp-core/atp/model_catalog/__init__.py tests/unit/model_catalog/test_schema.py tests/unit/model_catalog/test_loader.py
git commit -m "feat(model-catalog): typed harnesses/agents planes + referential validator (SP-E)"
```

---

## Task 2: Migrate the harness onto `load_catalog`

**Files:**
- Modify: `method/run_pipe_check.py` (`_load_agent_catalog`, ~lines 74-128)
- Modify: `tests/unit/method_spawners/test_run_pipe_check.py`

**Interfaces:**
- Consumes: `load_catalog`, `CatalogError` (atp.model_catalog); `HarnessEntry`/`AgentEntry`
  attributes (`.shim`, `.model_env`, `.harness`, `.model`, `.tested`).
- Produces: `_load_agent_catalog(path=CATALOG_PATH) -> tuple[dict[str, tuple[str, str]],
  list[tuple[str, str]]]` (unchanged signature/return shape); module-level `HARNESSES`,
  `AGENT_MODELS`, `AGENTS` unchanged.

- [ ] **Step 1: Write the failing harness tests**

Append to `tests/unit/method_spawners/test_run_pipe_check.py` (import the module under test as
the existing tests do — check the top of the file for the exact import; below assumes
`import method.run_pipe_check as rpc`):

```python
import pytest

import method.run_pipe_check as rpc

_SWEEP = (
    '[models."m1"]\nvendor="v"\nstatus="active"\n'
    '[models."m2"]\nvendor="v"\nstatus="active"\n'
    '[harnesses.ha]\nkind="cli"\nshim="a.py"\nmodel_env="A_MODEL"\n'
    '[harnesses.hb]\nkind="cli"\nshim="b.py"\nmodel_env="B_MODEL"\n'
    '[[agents]]\nharness="ha"\nmodel="m1"\ntested=true\n'
    '[[agents]]\nharness="hb"\nmodel="m2"\ntested=true\n'
    '[[agents]]\nharness="ha"\nmodel="m2"\ntested=false\n'  # untested → filtered out
)


def _write(tmp_path, text) -> "Path":
    f = tmp_path / "agents-catalog.toml"
    f.write_text(text, encoding="utf-8")
    return f


def test_projection_builds_harnesses_and_agent_models(tmp_path) -> None:
    harnesses, agent_models = rpc._load_agent_catalog(_write(tmp_path, _SWEEP))
    assert harnesses == {"ha": ("a.py", "A_MODEL"), "hb": ("b.py", "B_MODEL")}
    # tested filter: the tested=false pair is excluded.
    assert agent_models == [("ha", "m1"), ("hb", "m2")]


def test_agent_models_preserves_declaration_order(tmp_path) -> None:
    reordered = (
        '[models."m1"]\nvendor="v"\nstatus="active"\n'
        '[models."m2"]\nvendor="v"\nstatus="active"\n'
        '[harnesses.ha]\nkind="cli"\nshim="a.py"\nmodel_env="A"\n'
        '[harnesses.hb]\nkind="cli"\nshim="b.py"\nmodel_env="B"\n'
        '[[agents]]\nharness="hb"\nmodel="m2"\ntested=true\n'
        '[[agents]]\nharness="ha"\nmodel="m1"\ntested=true\n'
    )
    _, agent_models = rpc._load_agent_catalog(_write(tmp_path, reordered))
    assert agent_models == [("hb", "m2"), ("ha", "m1")]  # order == [[agents]] order


def test_missing_planes_is_not_a_sweep_catalog(tmp_path) -> None:
    models_only = '[models."m"]\nvendor="v"\nstatus="active"\n'
    with pytest.raises(RuntimeError, match="not a sweep catalog"):
        rpc._load_agent_catalog(_write(tmp_path, models_only))


def test_bad_catalog_wraps_error_with_ssot_hint_and_preserves_cause(tmp_path) -> None:
    from atp.model_catalog import CatalogError

    bad_ref = (
        '[models."m"]\nvendor="v"\nstatus="active"\n'
        '[harnesses.ha]\nkind="cli"\nshim="a.py"\nmodel_env="A"\n'
        '[[agents]]\nharness="MISSING"\nmodel="m"\n'
    )
    with pytest.raises(RuntimeError, match="ecosystem\\s+SSOT") as ei:
        rpc._load_agent_catalog(_write(tmp_path, bad_ref))
    assert isinstance(ei.value.__cause__, CatalogError)  # cause not lost
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/method_spawners/test_run_pipe_check.py -v -k "projection or order or sweep or ssot"`
Expected: FAIL — the current `_load_agent_catalog` raises `FileNotFoundError`/`KeyError`, not the
new `RuntimeError`, and does not enforce planes the new way.

- [ ] **Step 3: Rewrite `_load_agent_catalog` in `method/run_pipe_check.py`**

Add to the imports near the top (with the other `from atp....` imports):

```python
from atp.model_catalog import CatalogError, load_catalog
```

Replace the whole `_load_agent_catalog` function (current ~lines 77-125) with:

```python
def _load_agent_catalog(
    path: Path = CATALOG_PATH,
) -> tuple[dict[str, tuple[str, str]], list[tuple[str, str]]]:
    """Load HARNESSES + AGENT_MODELS from the SSOT catalog via the shared loader.

    Returns ``(harnesses, agent_models)`` where ``harnesses`` maps a harness
    name to ``(shim_path, model_env)`` and ``agent_models`` is the ordered list
    of ``(harness, model)`` pairs with ``tested = true`` — the ATP sweep set,
    in ``[[agents]]`` declaration order.

    Document shape and referential integrity (every agent references a declared
    harness) are enforced by ``atp.model_catalog`` (SP-E). The tested filter and
    the projection to the harness's shapes stay here (ATP sweep usage). Runtime
    availability (shim binary, API keys) stays in ``_preflight``.
    """
    try:
        cat = load_catalog(path)
    except CatalogError as exc:
        raise RuntimeError(
            f"canonical agents-catalog problem at {path} — this file IS the "
            "ecosystem SSOT (ADR-ECO-003); restore it from git history. "
            "(../_cowork_output/contracts/ holds only a dev mirror.)"
        ) from exc

    if cat.harnesses is None or cat.agents is None:
        raise RuntimeError(
            f"agents-catalog {path} has no harnesses/agents plane — "
            "not a sweep catalog"
        )

    harnesses = {
        name: (h.shim, h.model_env) for name, h in cat.harnesses.items()
    }
    agent_models = [(a.harness, a.model) for a in cat.agents if a.tested]
    return harnesses, agent_models
```

Then remove the now-unused local `import tomllib` (it was inside the old function). If the
file's top-level `import tomllib` (near line 41) is no longer referenced anywhere else, remove it
too; if `grep -n "tomllib" method/run_pipe_check.py` still shows other uses, leave it.

- [ ] **Step 4: Run the new harness tests**

Run: `uv run pytest tests/unit/method_spawners/test_run_pipe_check.py -v -k "projection or order or sweep or ssot"`
Expected: 4 PASS.

- [ ] **Step 5: Regression — the full harness test file + import over the real catalog**

Run: `uv run pytest tests/unit/method_spawners/test_run_pipe_check.py -v`
Expected: all PASS (importing the module runs `HARNESSES, AGENT_MODELS =
_load_agent_catalog()` over the real SSOT — it must load cleanly, and `AGENTS` builds).

- [ ] **Step 6: Sanity — the harness still lists agents**

Run: `uv run python -c "import method.run_pipe_check as r; print(len(r.AGENTS), 'agents'); print(list(r.AGENTS)[:3])"`
Expected: prints the agent count (15) and the first agent_ids (e.g.
`claude_code@claude-sonnet-4-6`).

- [ ] **Step 7: Format, lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add method/run_pipe_check.py tests/unit/method_spawners/test_run_pipe_check.py
git commit -m "refactor(method): pipe-check harness reads the shared model_catalog loader (SP-E)"
```

---

## Task 3: Docs pointer

**Files:**
- Modify: `CLAUDE.md`
- Modify: `TODO.md`

- [ ] **Step 1: Update CLAUDE.md**

In component 25 (the methodology paragraph mentioning `run_pipe_check.py` / `agents-catalog.toml`),
append one sentence: the pipe-check harness now reads the SSOT catalog through the shared
`atp.model_catalog.load_catalog` (ADR-003b SP-E); the `harnesses`/`agents` planes are typed
(`HarnessEntry`/`AgentEntry`) with a referential-integrity validator, while the tested filter and
sweep projection remain in `run_pipe_check.py`.

- [ ] **Step 2: Tick SP-E in TODO.md**

Update the 003b line: mark SP-E done (link this plan + the spec); remaining increments SP-C
(evaluator-model unify) and SP-D (rename `atp/catalog/`).

- [ ] **Step 3: Sanity + commit**

Run: `uv run pytest tests/unit/model_catalog/ tests/unit/method_spawners/test_run_pipe_check.py -q && uv run pyrefly check`
Expected: all green; pyrefly 0 errors.

```bash
git add CLAUDE.md TODO.md
git commit -m "docs(sp-e): harness reads shared catalog loader; 003b SP-E status"
```

---

## Self-Review

**Spec coverage:**
- `HarnessEntry`/`AgentEntry` typed with defaults + `extra="allow"` → Task 1. ✓
- `models` unchanged (required); planes typed + optional → Task 1. ✓
- Referential validator, both-present-only (present-empty counts), None no-op → Task 1
  (`test_referential_*`). ✓
- Error chain `ValueError`→`ValidationError`→`CatalogSchemaError` → Task 1 (schema test asserts
  `ValidationError`, loader test asserts `CatalogSchemaError`). ✓
- User-catalog (models-only) still loads → Task 1 (`test_referential_noop_when_planes_absent`) +
  the existing SP-A loader tests unchanged. ✓
- Harness `_load_agent_catalog` → `load_catalog` + projection + sweep guard + error wrap; drop
  local undeclared check; signature/return unchanged → Task 2. ✓
- Sweep-order invariant → Task 2 (`test_agent_models_preserves_declaration_order`). ✓
- Error wrap preserves `__cause__` → Task 2 (`test_bad_catalog_wraps_error…preserves_cause`). ✓
- `_preflight` / filesystem untouched → honored (no task edits `_preflight`). ✓
- Non-goals (settings.py, atp/catalog/) untouched → honored. ✓

**Placeholder scan:** no "TBD/handle errors" placeholders. The only assumed detail is the harness
test-module import alias (`import method.run_pipe_check as rpc`) — Step 1 instructs the
implementer to match the file's existing import; the `_SWEEP`/`bad_ref` fixtures are complete
inline TOML.

**Type consistency:** `HarnessEntry`/`AgentEntry` fields, `ModelCatalog.harnesses/agents` types,
`_load_agent_catalog(path) -> tuple[dict[str, tuple[str, str]], list[tuple[str, str]]]`,
`CatalogError`, `CatalogSchemaError` are used identically across Tasks 1-2. `.shim`/`.model_env`/
`.harness`/`.model`/`.tested` attribute access matches the entry definitions.
