# Design: SP-E — migrate the pipe-check harness onto the shared catalog loader

**Date:** 2026-07-07
**Status:** Approved (brainstorm) — ready for implementation plan
**ADR:** ADR-ECO-003b (catalog distribution) — `_cowork_output/decisions/2026-07-02-adr-eco-003b-catalog-distribution.md` in the dev-only sibling workspace (not committed to this repo; pointer, not a link — see CLAUDE.md on `../_cowork_output/`)
**Epic:** 003b, increment **SP-E**. Builds on **SP-A** (shipped: `atp.model_catalog`, PR #237).

---

## Problem

The pipe-check harness (`method/run_pipe_check.py`) reads the ecosystem SSOT
`method/agents-catalog.toml` through its **own** loader (`_load_agent_catalog`), duplicating
TOML parsing, error handling, and — crucially — the catalog-integrity check ("a tested agent
must reference a declared harness"). SP-A shipped a shared, tested loader (`atp.model_catalog`)
that keeps the `harnesses`/`agents` planes as **raw passthrough** (`dict[str, Any]` /
`list[dict[str, Any]]`), deferring their formalization to "when the harness migrates."

SP-E migrates the harness onto `load_catalog(path=...)` and **formalizes** those two planes into
typed schema with the referential-integrity validator. The boundary: **the schema owns document
shape + integrity; the harness owns how ATP uses the document for a sweep.**

## Non-goals (later increments / other repos)

- **SP-C** — evaluator-model unification (`atp/core/settings.py` `default_llm_model`) + the D2
  field-level env override.
- **SP-D** — rename `atp/catalog/` (test-suite catalog) → test-catalog.
- **arbiter / Maestro reader migration** — their repos; they vendor the catalog separately.
- **`agent.model ∈ models` validation** — not added now; a transitional catalog can legitimately
  drift, and this would create false blocks. Explicitly out of scope.
- **shim-path / binary existence in the schema** — schema validates *document shape and
  referential integrity only*, never filesystem existence. Runtime availability (shim binary on
  PATH, API keys) stays a **preflight/runtime concern** — `_preflight()` is unchanged by SP-E.

## Architecture

```
atp.model_catalog.schema           method/run_pipe_check.py
  ModelEntry   (SP-A, unchanged)      _load_agent_catalog(path) -> (HARNESSES, AGENT_MODELS)
  HarnessEntry (NEW)      ◄──────────   cat = load_catalog(path)   # shared loader (also
  AgentEntry   (NEW)                      validates models plane now)
  ModelCatalog (planes now typed)        project cat.harnesses/agents -> the harness's shapes
   + referential validator               tested-filter + AGENTS dict stay here
                                          wrap CatalogError -> dev-SSOT hint (from exc)
```

### Component 1 — formalize the planes (`packages/atp-core/atp/model_catalog/schema.py`)

Replace the raw passthrough planes with typed models. All carry `ConfigDict(extra="allow")`
(forward-compat, consistent with SP-A's `ModelEntry`):

```python
class HarnessEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    kind: str
    shim: str
    model_env: str
    model_flag: str | None = None
    routable: bool = False

class AgentEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    harness: str
    model: str
    tested: bool = False
    routable: bool = False
```

`ModelCatalog` changes:
- `models: dict[str, ModelEntry]` — **unchanged** (required plane, may be empty; SP-A fork A).
- `harnesses: dict[str, HarnessEntry] | None = None` — typed, still **optional**.
- `agents: list[AgentEntry] | None = None` — typed, still **optional**; **list order is a
  preserved invariant** (see §Invariants).
- `model_config = ConfigDict(extra="allow")` — unchanged.

**Referential-integrity validator** (`@model_validator(mode="after")`):
- Fires **only when BOTH `harnesses` and `agents` are present** (not None) — **present-empty
  counts as present** (an empty `harnesses={}` with a non-empty `agents=[…]` still validates and
  can fail). Then every `agent.harness` must be a key in `harnesses`; otherwise the validator
  **raises `ValueError`** listing the undeclared harness(es).
- **Error chain (be precise for test-writing):** a `model_validator` raising `ValueError` is
  caught by pydantic and surfaced as a **`pydantic.ValidationError`**; `load_catalog` catches
  `ValidationError` (already, from SP-A) and wraps **all validation failures** — field-shape and
  referential alike — into **`CatalogSchemaError`**. So loader-level tests assert
  `CatalogSchemaError`; direct schema-construction tests assert `ValidationError`.
- When either plane is **absent (None)** — the **user-runtime catalog** (SP-A fork A) — the
  validator is a **no-op**. This is load-bearing: SP-E must NOT make `load_catalog(path)` reject
  a valid user-catalog that has only a `models` plane. The harness separately requires the
  sweep-catalog shape (§Component 2), so the "harness needs both planes" rule lives in the
  harness, not the shared schema.

### Component 2 — migrate the harness (`method/run_pipe_check.py`)

`_load_agent_catalog(path=CATALOG_PATH)` keeps its **signature and return shape**
(`tuple[dict[str, tuple[str, str]], list[tuple[str, str]]]`), so `HARNESSES, AGENT_MODELS =
_load_agent_catalog()` and the downstream `AGENTS` dict are unchanged. Internals:

1. `cat = load_catalog(path)` — the shared loader. Side effect (positive): it now also validates
   the `models` plane; the current SSOT conforms.
2. **Sweep-shape guard:** if `cat.harnesses is None or cat.agents is None`, raise a clear
   fail-loud ("`{path}` has no harnesses/agents plane — not a sweep catalog"). This is the
   harness's requirement, kept out of the shared schema.
3. **Projection:**
   - `harnesses = {name: (h.shim, h.model_env) for name, h in cat.harnesses.items()}`
   - `agent_models = [(a.harness, a.model) for a in cat.agents if a.tested]` — **the tested
     filter (sweep-set selection) stays here.**
4. The harness's **own undeclared-harness check is removed** — the schema validator now
   guarantees `agent.harness ∈ harnesses`, so `HARNESSES[harness]` in the `AGENTS` construction
   is safe.
5. **Error diagnostics preserved:** wrap the shared loader's typed errors so pipe-check
   diagnostics do not degrade:
   ```python
   try:
       cat = load_catalog(path)
   except CatalogError as exc:
       raise RuntimeError(
           f"canonical agents-catalog problem at {path} — this file IS the ecosystem "
           "SSOT (ADR-ECO-003); restore it from git history. "
           "(../_cowork_output/contracts/ holds only a dev mirror.)"
       ) from exc   # `from exc` preserves the cause type + traceback
   ```
   `CATALOG_PATH` (the dev-SSOT path) is unchanged; `_preflight()` is unchanged.

### Invariants (state explicitly; tests pin them)

- **Sweep order:** `AGENT_MODELS` preserves the order of `[[agents]]` in the TOML. `cat.agents`
  is a list (pydantic preserves list order), the projection iterates it in order and the tested
  filter is order-preserving. Sweep order is observable in outputs, so this is a contract, not an
  accident.
- **User-catalog compatibility:** `load_catalog(user_catalog_with_only_models)` still succeeds
  (referential validator no-ops when planes absent). SP-A fork A is preserved.
- **Schema ≠ filesystem:** the schema never checks shim-path/binary existence; that stays in
  `_preflight()` / spawn time.

## Testing

- **Schema** (`tests/unit/model_catalog/test_schema.py`, extend):
  - `HarnessEntry`/`AgentEntry` valid + defaults (`model_flag=None`, `routable=False`,
    `tested=False`); `extra="allow"` tolerates an unknown field.
  - Referential validator (direct schema construction → assert `pydantic.ValidationError`):
    both planes present + consistent → OK; an agent referencing an undeclared harness → error.
  - **None vs present-empty planes** (pins the trigger boundary):
    - `models`-only (harnesses/agents **absent/None**) → OK (no-op case, SP-A preserved).
    - `harnesses={}` **and** `agents=[]` (both present, empty) → OK (nothing to check).
    - `harnesses={}` **and** `agents=[{harness:"x", model:"m"}]` (present, ref into empty) →
      error (undeclared harness `x`).
- **Loader** (`tests/unit/model_catalog/test_loader.py`, extend): a fixture catalog with a bad
  referential link, loaded via `load_catalog` → `CatalogSchemaError` (the `ValidationError` is
  wrapped, per the error chain above).
- **Harness** (`tests/unit/method_spawners/…` — the existing pipe-check test location):
  - projection builds `HARNESSES`/`AGENT_MODELS`/`AGENTS` correctly from a fixture catalog;
  - `AGENT_MODELS` order matches the fixture's `[[agents]]` order (sweep-order invariant);
  - a catalog missing `harnesses`/`agents` → the sweep-shape fail-loud;
  - a malformed catalog → the wrapped `RuntimeError` carries the SSOT hint AND `__cause__` is the
    original `CatalogError` (cause not lost).
- **Regression:** the existing pipe-check / harness tests pass unchanged — the projection yields
  the same `HARNESSES`/`AGENT_MODELS`/`AGENTS` structures as before, and `load_catalog(CATALOG_
  PATH)` over the real SSOT succeeds (all three planes validate).

## Files

| Path | Change |
|---|---|
| `packages/atp-core/atp/model_catalog/schema.py` | add `HarnessEntry`/`AgentEntry`; type the planes; referential validator |
| `packages/atp-core/atp/model_catalog/__init__.py` | export `HarnessEntry`, `AgentEntry` |
| `method/run_pipe_check.py` | `_load_agent_catalog` → `load_catalog` + projection + sweep guard + error wrap; drop the local undeclared-harness check |
| `tests/unit/model_catalog/test_schema.py` | extend — new entries + referential validator (incl. planes-absent no-op) |
| `tests/unit/model_catalog/test_loader.py` | extend — referential error → `CatalogSchemaError` |
| `tests/unit/method_spawners/…` | harness projection / order / sweep-guard / error-wrap tests |

Docs/CLAUDE.md pointer updates are the implementation plan's final step.
