# Design: SP-C — unify the evaluator default model through the catalog

**Date:** 2026-07-07
**Status:** Approved (brainstorm) — ready for implementation plan
**ADR:** [ADR-ECO-003b](../../../../_cowork_output/decisions/2026-07-02-adr-eco-003b-catalog-distribution.md)
**Epic:** 003b, increment **SP-C**. Builds on **SP-A** (`atp.model_catalog`, #237) and **SP-E**
(typed planes + validators, #238).

---

## Problem

There are two independent places that name "which model": `settings.default_llm_model`
(`atp/core/settings.py`, env `ATP_DEFAULT_LLM_MODEL`, hardcoded default
`claude-sonnet-4-20250514`) which the LLM judge reads, and the model catalog
(`method/agents-catalog.toml`'s `models` plane). ADR-003b calls for reducing these to **one
resolution mechanism**. SP-C makes the catalog the single non-env place to declare the default
model, without regressing the evaluator's out-of-box behavior (the ADR is explicit that the
evaluator model "already works via env/.env").

**Scope profile (chosen in brainstorm): interpretation (A)** — the catalog is an *optional*
source of the default; one resolution chain (env → catalog → provider fallback); out-of-box eval
stays working because the provider fallback is preserved and the catalog is never *required*.

## Non-goals

- **SP-D** — rename `atp/catalog/` (test-suite catalog) → test-catalog.
- **arbiter / Maestro reader migration** — their repos.
- **Per-role defaults** — one `default_model` (YAGNI; no judge-vs-agent split until needed).
- **`CLAUDE_MODEL` as an evaluator override** — that is the harness's *agent* model_env, not the
  evaluator; the evaluator field override stays `ATP_DEFAULT_LLM_MODEL`.
- **Making the evaluator depend on a configured catalog** — rejected (interpretation C); would
  break out-of-box eval.

## Architecture

```
llm_judge (evaluator model chain, unchanged except one line)
  config.model → ATP_JUDGE_MODEL → resolve_default_model(settings.default_llm_model)
                                    → provider fallback (gpt-4o-mini / bedrock / claude)
                                        │
atp.model_catalog.resolve_default_model(explicit)   # runtime convenience, TOLERANT
  explicit(non-empty) → load_catalog() [defaults].default_model → None
       CatalogNotConfiguredError → None (silent; out-of-box)
       CatalogTOMLError/CatalogSchemaError → warn + None (don't break eval on a bad optional catalog)

atp.model_catalog.schema
  CatalogDefaults(default_model)  +  ModelCatalog.defaults
   + validator: default_model ∈ models keys ∪ all aliases  (only when truthy AND models non-empty)
```

### Component 1 — schema: the `[defaults]` plane (`packages/atp-core/atp/model_catalog/schema.py`)

```python
class CatalogDefaults(BaseModel):
    model_config = ConfigDict(extra="allow")
    default_model: str | None = None


class ModelCatalog(BaseModel):
    ...
    defaults: CatalogDefaults | None = None
    # (models / harnesses / agents unchanged from SP-A/SP-E)
```

**Referential validator** (`@model_validator(mode="after")`, alongside the SP-E
harness-referential one): fires **only when** `defaults` is present with a **truthy**
`default_model` **and** `models` is non-empty. Then `default_model` must be a key in `models`
**or** an alias of some `ModelEntry` (`entry.aliases`) — otherwise raise `ValueError` (→
pydantic `ValidationError` → `load_catalog` wraps into `CatalogSchemaError`). When `defaults` is
absent, `default_model` is falsy, or `models` is empty → **no-op** (user catalogs and the real
SSOT, which have no `[defaults]`, are unaffected).

### Component 2 — the tolerant resolver (`packages/atp-core/atp/model_catalog/loader.py`)

`resolve_default_model` is a **runtime convenience**, not the general loader: an *optional*
catalog with *tolerant* failure; the caller's provider fallback saves the day.

```python
def resolve_default_model(explicit: str | None = None) -> str | None:
    """Resolve the default model: an explicit override, else the catalog's
    [defaults].default_model, else None (the caller supplies a fallback).

    Tolerant by design — a missing catalog is fine (out-of-box eval must not
    require one), and a present-but-broken catalog is logged and ignored rather
    than crashing the evaluator. This is NOT load_catalog()'s fail-loud contract.
    """
    if explicit and explicit.strip():   # empty/whitespace env → treated as unset
        return explicit
    try:
        catalog = load_catalog()        # D2 resolution
    except CatalogNotConfiguredError:
        return None                     # no catalog configured — expected, silent
    except (CatalogTOMLError, CatalogSchemaError) as exc:
        logger.warning("model catalog present but unusable, ignoring: %s", exc)
        return None
    if catalog.defaults is not None and catalog.defaults.default_model:
        return catalog.defaults.default_model
    return None
```

- **`CatalogNotConfiguredError` → silent None** (out-of-box eval must not require a catalog).
- **`CatalogTOMLError` / `CatalogSchemaError` → `logger.warning` + None** (don't crash the
  evaluator over a broken *optional* catalog).
- Empty/whitespace `explicit` is treated as **unset** (so `ATP_DEFAULT_LLM_MODEL=""` defers to
  the catalog rather than becoming an empty explicit value).
- `load_catalog()` (CLI / harness) keeps its **fail-loud** contract — unchanged.

### Component 3 — settings (`atp/core/settings.py`)

`default_llm_model` becomes **optional**, so "unset" is distinguishable from "explicitly set"
(only then can the catalog provide the default):

```python
default_llm_model: str | None = Field(
    default=None,
    description="Default LLM model for evaluators; None defers to the model "
    "catalog's [defaults], then a provider fallback (ADR-003b SP-C).",
)
```

The env var `ATP_DEFAULT_LLM_MODEL` still sets it (non-None). The hardcoded
`claude-sonnet-4-20250514` string leaves the field — its role moves to the resolver chain's
provider fallback (Component 4, already present in `llm_judge`).

### Component 4 — llm_judge (`atp/evaluators/llm_judge.py`)

Change **one line** — the settings read (≈ line 203):

```python
# before:  self._model = settings.default_llm_model
self._model = resolve_default_model(settings.default_llm_model)
```

Everything else is unchanged: `config.model` / `ATP_JUDGE_MODEL` (higher priority, still win),
the base_url/bedrock routing that *skips* this default path, and the provider-specific fallbacks
at ≈ lines 216-222 (`gpt-4o-mini` / bedrock default / `claude-sonnet-4-20250514`) — which now
also catch the `None` the resolver can return. The effective chain becomes:

`config.model` → `ATP_JUDGE_MODEL` → `ATP_DEFAULT_LLM_MODEL` (env) → catalog `[defaults]` →
provider fallback.

**Behavior preserved:** a plain judge with no env and no catalog still resolves to
`claude-sonnet-4-20250514` — now via the line-222 fallback instead of the hardcoded settings
field. Bedrock / base_url judges still skip the catalog default and use their existing fallbacks.

### What is unified

The default model is now set via **env OR the catalog `[defaults]` plane** — one resolution
chain, one non-env place. The settings field's hardcoded default is gone; the catalog gains a
second runtime consumer (the evaluator default). Out-of-box eval is **not** broken (fallback
preserved, catalog optional and tolerant).

## Testing

- **Schema** (`tests/unit/model_catalog/test_schema.py`): `CatalogDefaults` default None;
  `defaults.default_model` = a models key → OK; = an **alias** of a model → OK; = an unknown id
  with non-empty models → `ValidationError`; `[defaults]` present but `default_model=None` →
  no-op; `default_model` set with **empty** `models` → no-op; no `[defaults]` → no-op.
- **Resolver** (`tests/unit/model_catalog/test_loader.py`): explicit non-empty → returned as-is
  (catalog never consulted); explicit empty/`"  "` → treated as unset; catalog `[defaults]`
  used when explicit is None; no catalog (`CatalogNotConfiguredError`) → None silently; a broken
  catalog (bad TOML / bad schema, via a temp `$ATP_CATALOG`) → None **and** a warning logged
  (`caplog`). Never touches the real `~/.config`.
- **Evaluator chain** (`tests/unit/evaluators/…` — the existing llm_judge test location):
  - no env / no catalog / anthropic provider → `claude-sonnet-4-20250514` (line-222 fallback);
  - `ATP_DEFAULT_LLM_MODEL=x` wins over a catalog default;
  - catalog `[defaults].default_model` used when the settings default is None;
  - `ATP_JUDGE_MODEL` wins over both;
  - bedrock / base_url judge still skips the catalog default path and uses its existing fallback.
- **Regression:** existing llm_judge + settings tests pass; the real SSOT (no `[defaults]`) still
  loads (harness/SP-E unaffected); `atp models list/init` unaffected.

## Files

| Path | Change |
|---|---|
| `packages/atp-core/atp/model_catalog/schema.py` | `CatalogDefaults`; `ModelCatalog.defaults`; default-model referential validator (aliases included) |
| `packages/atp-core/atp/model_catalog/loader.py` | `resolve_default_model` (tolerant); a module `logger` |
| `packages/atp-core/atp/model_catalog/__init__.py` | export `CatalogDefaults`, `resolve_default_model` |
| `packages/atp-core/atp/model_catalog/data/template.toml` | add a commented `[defaults]` example |
| `atp/core/settings.py` | `default_llm_model: str | None = None` |
| `atp/evaluators/llm_judge.py` | one-line swap to `resolve_default_model(...)` |
| `tests/unit/model_catalog/test_schema.py` | defaults validator tests |
| `tests/unit/model_catalog/test_loader.py` | resolver tests (incl. tolerant-failure + caplog) |
| `tests/unit/evaluators/…` (llm_judge test) | the 5 chain cases |
| `docs/reference/configuration.md` | note `ATP_DEFAULT_LLM_MODEL` now defers to catalog `[defaults]` when unset |

Docs/CLAUDE.md pointer updates are the implementation plan's final step.
