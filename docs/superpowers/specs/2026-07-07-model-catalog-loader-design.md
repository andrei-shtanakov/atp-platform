# Design: model-catalog loader — shippable schema + D2 resolution + `atp models init/list`

**Date:** 2026-07-07
**Status:** Approved (brainstorm) — ready for implementation plan
**ADR:** [ADR-ECO-003b](../../../../_cowork_output/decisions/2026-07-02-adr-eco-003b-catalog-distribution.md)
(catalog distribution), amends 003a / ADR-ECO-003
**Epic:** 003b, increment **SP-A** (foundation). Scope profile **(b) internal unification,
minimal CLI** — not external-facing UX polish.

---

## Problem

The ecosystem model catalog (`method/agents-catalog.toml`) is the git SSOT, read by the
pipe-check harness via a **hardcoded dev path** (`method/run_pipe_check.py`, `CATALOG_PATH =
REPO_ROOT / "method" / "agents-catalog.toml"`). The wheel packs only `atp/`
(`packages/atp-core/pyproject.toml` explicit `packages` list), so `method/agents-catalog.toml`
never reaches a pip-installed user — there is **no shipped loader, schema, or resolution** for a
model catalog. ADR-003b closes this distribution gap: the package ships a **loader + schema +
inert template**, and an installed instance resolves its catalog from user config
(`$ATP_CATALOG` → XDG → fail-loud), never from git.

This increment (SP-A) is **purely additive**: a new shippable module + a minimal
`atp models init/list` CLI. It changes nothing existing — the paid pipe-check path keeps its own
loader for now.

## Non-goals (deferred to later 003b increments)

- **Harness migration** onto the shared loader (SP-E) — the harness keeps its own path this
  increment; two loaders coexist deliberately.
- **Evaluator-model unification** (`atp/core/settings.py` `default_llm_model`) and the D2
  **field-level env override** (level 3: `CLAUDE_MODEL` / `ATP_LLM__…`) — SP-C.
- **Renaming `atp/catalog/`** (the test-suite catalog) → test-catalog — SP-D.
- **`models discover` / `models update`** — no external-user demand yet (YAGNI); discovery stays
  in `devtools/` (ADR-003a D5).
- **arbiter / Maestro readers** — out of this repo.

## Decisions carried from the ADR (not re-litigated)

- **Resolution order (D2):** `$ATP_CATALOG` → `$XDG_CONFIG_HOME/atp/agents-catalog.toml` →
  `~/.config/atp/agents-catalog.toml` → **fail-loud** (no baked default). XDG namespace `atp`
  ratified 2026-07-06.
- **Inert template (D1):** the wheel ships a template resource, **never** loaded as live data;
  **no active `claude-sonnet-*`** or any real model names (no soft endorsement).
- **User-runtime contract = the `models` plane** (brainstorm fork A): `harnesses`/`agents` are
  optional dev-SSOT extensions, tolerated as passthrough, formalized when the harness migrates
  (SP-E).

## Architecture

```
packages/atp-core/atp/model_catalog/
├── __init__.py        exports: ModelCatalog, ModelEntry, load_catalog,
│                      resolve_catalog_path, the typed errors, TEMPLATE_RESOURCE
├── schema.py          pydantic: ModelEntry (strict-on-known), ModelCatalog (passthrough planes)
├── errors.py          CatalogError + CatalogNotConfiguredError / CatalogTOMLError / CatalogSchemaError
├── loader.py          resolve_catalog_path(must_exist), load_catalog(path|None)
└── data/
    └── template.toml  inert starter (package data — MUST ship in the wheel)

atp/cli/…  → new `atp models` group: init, list  (follows the existing subcommand pattern)
```

The loader is **pure and path-aware**: `load_catalog(path)` loads an explicit file (the future
harness/dev-SSOT consumer); `load_catalog(None)` runs D2 resolution (the installed-user path).

### Component 1 — schema (`schema.py`)

- `ModelEntry` — strict on the known fields, tolerant of unknown (forward-compat + dev-SSOT
  extras): `vendor: str` (required), `status: Literal["active", "deprecated", "retired"]`
  (required), `aliases: list[str] = []`; `model_config = ConfigDict(extra="allow")`. A bad
  `status` value is a validation error (strict where it counts).
- `ModelCatalog`:
  - `models: dict[str, ModelEntry]` — **required plane**, may be an **empty table** (valid;
    "no usable model" is a lookup concern for SP-C, not a load error).
  - `harnesses: dict[str, Any] | None = None` — **passthrough** (raw), so the dev-SSOT file
    parses; not validated here. Named passthrough to mark the SP-E boundary.
  - `agents: list[dict[str, Any]] | None = None` — **passthrough** (raw).
  - `model_config = ConfigDict(extra="allow")` — unknown top-level sections pass through.

### Component 2 — typed errors (`errors.py`)

A base `CatalogError(Exception)` with three subclasses, so the CLI maps to clear messages +
non-zero exit and future evaluator/harness consumers can catch programmatically:
- `CatalogNotConfiguredError` — no catalog resolved (nothing found; or a **relative**
  `$ATP_CATALOG` / `$XDG_CONFIG_HOME`). Message carries the `atp models init` / `$ATP_CATALOG`
  hint (the ADR fail-loud text).
- `CatalogTOMLError` — file exists but is not valid TOML.
- `CatalogSchemaError` — parsed, but `models` fails schema validation (wraps pydantic's error).

### Component 3 — loader (`loader.py`)

- `resolve_catalog_path(*, must_exist: bool) -> Path` — D2 precedence, one function, two modes:
  - `$ATP_CATALOG`: **empty string → treated as unset** (fall through); **relative path →
    `CatalogNotConfiguredError`** ("$ATP_CATALOG must be an absolute path, got …"); absolute →
    use it.
  - XDG: `$XDG_CONFIG_HOME` with the same empty=unset / relative=error rule, joined with
    `atp/agents-catalog.toml`; unset → `~/.config/atp/agents-catalog.toml`.
  - **`must_exist=True`** (for load): each candidate must exist to win; none exists →
    `CatalogNotConfiguredError` (init hint). **`must_exist=False`** (for `init`): returns the
    **target** path (whether or not it exists) — `$ATP_CATALOG` set but file absent is NOT an
    error, it is the creation target.
  - Semantics: `$ATP_CATALOG` and `--path` are **file** paths, never directories.
- `load_catalog(path: Path | None = None) -> ModelCatalog`: `path` given → load it directly
  (must exist → `CatalogNotConfiguredError` if not); `None` → `resolve_catalog_path(must_exist=
  True)`. Read bytes → `tomllib` (→ `CatalogTOMLError`) → `ModelCatalog(**data)` (→
  `CatalogSchemaError`).

### Component 4 — inert template (`data/template.toml`, package data)

- Header comment: what it is, that the user edits it, the resolution order.
- **An active, empty `[models]` table** (not only comments) so a freshly-`init`-ed file loads
  **without** a schema error (satisfies "empty `models` is valid" end to end).
- Commented-out example entries with **placeholder** names/vendors (e.g. `# [models."your-model"]`),
  never real active models (ADR D1).
- Exposed as `TEMPLATE_RESOURCE` via `importlib.resources.files("atp.model_catalog").joinpath(
  "data/template.toml")`.

### Component 5 — packaging boundary (`packages/atp-core/pyproject.toml`)

- Add `"atp/model_catalog"` to the explicit `[tool.hatch.build.targets.wheel]` `packages` list —
  **without this the module works in editable/dev and vanishes in the wheel** (the classic
  "works in tests, missing in wheel" trap).
- Ensure `data/template.toml` is included as package data (hatch ships files under a listed
  package by default; if the build excludes non-`.py`, add an explicit
  `force-include`/`artifacts` entry). A test (Component 7) asserts it loads from the installed
  package via `importlib.resources`.
- `method/agents-catalog.toml` stays **out** of the wheel (under `method/`, already excluded) —
  confirm, do not change.

### Component 6 — minimal CLI (`atp models`)

New `atp models` group (siblings the existing `atp catalog` test-catalog command), two commands:
- `atp models init [--path PATH] [--force]` — resolves the **target** path
  (`resolve_catalog_path(must_exist=False)`, or `--path`), creates parent dirs (parent
  inaccessible → fail-loud `CatalogNotConfiguredError`/OSError message), writes the inert
  template. **Refuses to overwrite** an existing file unless `--force`. Prints the path + a
  next-step hint ("edit it to add your models").
- `atp models list [--format table|json]` — `load_catalog(None)`, prints models
  (name/vendor/status/aliases). `--format` default `table`; `json` for scripts (near-free now).
  On any `CatalogError`: print the mapped message, exit non-zero. Empty `models` → a friendly
  "no models defined yet — edit <path>" (still exit 0; the catalog is valid).

## Testing

- **Resolution** (`test_loader.py`): `$ATP_CATALOG` precedence over XDG; `$ATP_CATALOG` empty →
  unset (falls to XDG); `$ATP_CATALOG` relative → `CatalogNotConfiguredError`; `$XDG_CONFIG_HOME`
  empty → unset (falls to `~/.config`); `$XDG_CONFIG_HOME` relative → error; nothing exists +
  `must_exist=True` → `CatalogNotConfiguredError`; `must_exist=False` returns the target path
  even when absent. (Use `monkeypatch.setenv`/`delenv` + `tmp_path`; never touch the real
  `~/.config`.)
- **Parse/schema** (`test_loader.py`): valid `models`; **empty `[models]` loads OK**; dev-SSOT
  fixture with `harnesses`+`agents` passes through (not validated); invalid TOML →
  `CatalogTOMLError`; bad `status` → `CatalogSchemaError`; unknown top-level section tolerated.
- **Template** (`test_template.py`): the packaged `data/template.toml` loads via
  `importlib.resources.files("atp.model_catalog").joinpath("data/template.toml")`, is valid TOML,
  parses through `load_catalog` **without error** (proves the active empty `[models]`), and
  contains **no** active `claude-*`/`gpt-*` model table (guard against endorsement creep).
- **CLI** (`test_models_cli.py`): `init` writes the template + creates parents; `init` refuses to
  overwrite without `--force`, succeeds with it; `list` over a fixture prints rows;
  `list --format json` emits parseable JSON; `list` with no catalog → fail-loud message +
  non-zero exit.

## Files

| Path | Change |
|---|---|
| `packages/atp-core/atp/model_catalog/__init__.py` | new — public exports |
| `packages/atp-core/atp/model_catalog/schema.py` | new — `ModelEntry`, `ModelCatalog` |
| `packages/atp-core/atp/model_catalog/errors.py` | new — `CatalogError` + 3 subclasses |
| `packages/atp-core/atp/model_catalog/loader.py` | new — resolution + `load_catalog` |
| `packages/atp-core/atp/model_catalog/data/template.toml` | new — inert starter |
| `packages/atp-core/pyproject.toml` | add `"atp/model_catalog"` to wheel packages; ensure template ships |
| `atp/cli/…` | new `atp models init/list` group (existing subcommand pattern) |
| `tests/unit/model_catalog/test_loader.py` | new |
| `tests/unit/model_catalog/test_template.py` | new |
| `tests/unit/model_catalog/test_models_cli.py` | new |

Docs/CLAUDE.md pointer updates are the implementation plan's final step.
