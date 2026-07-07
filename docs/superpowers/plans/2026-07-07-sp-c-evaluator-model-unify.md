# SP-C evaluator model unify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the model catalog the single non-env place to declare the default LLM model, via a
tolerant `resolve_default_model()` the evaluator consults — without regressing out-of-box eval.

**Architecture:** Add an optional `[defaults]` plane (`CatalogDefaults.default_model`) to the
catalog schema with a referential validator; add a tolerant `resolve_default_model()` to the
loader (explicit → catalog `[defaults]` → None, swallowing a missing/broken *optional* catalog);
make `settings.default_llm_model` optional; swap one line in `llm_judge` to consult the resolver.
The evaluator's provider fallback stays and preserves current behavior.

**Tech Stack:** Python 3.12+, pydantic v2 (`model_validator`), pydantic-settings, uv, pytest.

## Global Constraints

- **uv only**; run tools via `uv run`. Line length **88**; full type hints;
  `uv run ruff format . && uv run ruff check . && uv run pyrefly check` clean after every task.
- **Catalog is optional & tolerant for the evaluator** — `resolve_default_model` swallows a
  missing catalog silently (`CatalogNotConfiguredError` → None) and a broken *optional* catalog
  with a warning (`CatalogTOMLError`/`CatalogSchemaError` → `logger.warning` + None). This is NOT
  `load_catalog()`'s fail-loud contract, which is unchanged for CLI/harness.
- **Referential validator** for `defaults.default_model`: fires **only when** `default_model` is
  truthy **and** `models` is non-empty; the value must be a `models` key **or** a `ModelEntry`
  alias; else `ValueError` → `CatalogSchemaError`. Absent/empty → no-op.
- **Empty/whitespace `explicit` = unset**; a non-empty explicit is returned **stripped**.
- **Resolver does not canonicalize** — a catalog default that is an alias is returned verbatim.
- **Behavior preserved:** a plain judge with no env/catalog still resolves to
  `claude-sonnet-4-20250514` (via the existing provider fallback); bedrock/base_url judges still
  skip the catalog-default path.
- **Do not touch** `atp/catalog/` (SP-D) or `method/` (the real SSOT has no `[defaults]`, stays a
  no-op).

Spec: `docs/superpowers/specs/2026-07-07-sp-c-evaluator-model-unify-design.md`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `packages/atp-core/atp/model_catalog/schema.py` | `CatalogDefaults`; `ModelCatalog.defaults`; default-model referential validator |
| `packages/atp-core/atp/model_catalog/loader.py` | `resolve_default_model` + module `logger` |
| `packages/atp-core/atp/model_catalog/__init__.py` | export `CatalogDefaults`, `resolve_default_model` |
| `packages/atp-core/atp/model_catalog/data/template.toml` | commented `[defaults]` example |
| `atp/core/settings.py` | `default_llm_model: str | None = None` |
| `atp/evaluators/llm_judge.py` | one-line swap to `resolve_default_model(...)` |
| `tests/unit/model_catalog/test_schema.py` | defaults validator tests |
| `tests/unit/model_catalog/test_loader.py` | resolver tests |
| `tests/unit/evaluators/test_llm_judge.py` | 5 chain cases |
| `docs/reference/configuration.md` | note the deferral behavior |

---

## Task 1: `[defaults]` plane + referential validator

**Files:**
- Modify: `packages/atp-core/atp/model_catalog/schema.py`
- Modify: `packages/atp-core/atp/model_catalog/__init__.py`
- Modify: `tests/unit/model_catalog/test_schema.py`

**Interfaces:**
- Produces: `CatalogDefaults(default_model: str | None = None)` (`extra="allow"`);
  `ModelCatalog.defaults: CatalogDefaults | None = None` + validator
  `_default_model_in_models`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/model_catalog/test_schema.py` (add `CatalogDefaults` to the existing
`from atp.model_catalog.schema import (...)` block):

```python
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
    c = ModelCatalog(
        models={"m": {"vendor": "v", "status": "active"}}, defaults={}
    )
    assert c.defaults.default_model is None


def test_default_model_with_empty_models_is_noop() -> None:
    # No validation when models is empty (nothing to check against).
    c = ModelCatalog(models={}, defaults={"default_model": "anything"})
    assert c.defaults.default_model == "anything"


def test_no_defaults_plane_is_noop() -> None:
    c = ModelCatalog(models={"m": {"vendor": "v", "status": "active"}})
    assert c.defaults is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/model_catalog/test_schema.py -v -k default`
Expected: FAIL — `ImportError: cannot import name 'CatalogDefaults'`.

- [ ] **Step 3: Add `CatalogDefaults` + the field + validator in `schema.py`**

Add the class after `AgentEntry`:

```python
class CatalogDefaults(BaseModel):
    """The catalog's optional [defaults] plane (runtime defaults)."""

    model_config = ConfigDict(extra="allow")

    default_model: str | None = None
```

In `ModelCatalog`, add the field (below `agents`) and a second validator (below the existing
`_agents_reference_declared_harnesses`):

```python
    defaults: CatalogDefaults | None = None
```

```python
    @model_validator(mode="after")
    def _default_model_in_models(self) -> ModelCatalog:
        # Fires only when a default_model is set AND models is non-empty: the
        # default must be a models key or a ModelEntry alias (typo-catcher). A
        # catalog with no [defaults], or with empty models, is a no-op.
        if (
            self.defaults is None
            or not self.defaults.default_model
            or not self.models
        ):
            return self
        known = set(self.models) | {
            alias for entry in self.models.values() for alias in entry.aliases
        }
        if self.defaults.default_model not in known:
            raise ValueError(
                f"defaults.default_model {self.defaults.default_model!r} is not a "
                "known model id or alias"
            )
        return self
```

- [ ] **Step 4: Export `CatalogDefaults` in `__init__.py`**

Add `CatalogDefaults` to the `from atp.model_catalog.schema import (...)` line and to `__all__`
(keep sorted).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/model_catalog/test_schema.py -v`
Expected: all PASS (existing + 7 new).

- [ ] **Step 6: Format, lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add packages/atp-core/atp/model_catalog/schema.py packages/atp-core/atp/model_catalog/__init__.py tests/unit/model_catalog/test_schema.py
git commit -m "feat(model-catalog): [defaults] plane + default_model referential validator (SP-C)"
```

---

## Task 2: tolerant `resolve_default_model`

**Files:**
- Modify: `packages/atp-core/atp/model_catalog/loader.py`
- Modify: `packages/atp-core/atp/model_catalog/__init__.py`
- Modify: `tests/unit/model_catalog/test_loader.py`

**Interfaces:**
- Consumes: `load_catalog`, `CatalogNotConfiguredError`, `CatalogTOMLError`,
  `CatalogSchemaError`, `CatalogDefaults` (Task 1).
- Produces: `resolve_default_model(explicit: str | None = None) -> str | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/model_catalog/test_loader.py` (it already imports `Path`, `pytest`,
`load_catalog`; add `resolve_default_model` to the loader import):

```python
def _dm(tmp_path: Path, body: str) -> Path:
    f = tmp_path / "agents-catalog.toml"
    f.write_text(body, encoding="utf-8")
    return f


_MODELS = '[models."m"]\nvendor="v"\nstatus="active"\n'


def test_resolve_explicit_stripped(monkeypatch) -> None:
    _clear(monkeypatch)
    assert resolve_default_model("  gpt-4o  ") == "gpt-4o"


def test_resolve_empty_explicit_is_unset(monkeypatch, tmp_path) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    assert resolve_default_model("   ") is None
    assert resolve_default_model("") is None


def test_resolve_from_catalog_defaults(monkeypatch, tmp_path) -> None:
    _clear(monkeypatch)
    f = _dm(tmp_path, _MODELS + '[defaults]\ndefault_model="m"\n')
    monkeypatch.setenv("ATP_CATALOG", str(f))
    assert resolve_default_model(None) == "m"


def test_resolve_explicit_wins_over_catalog(monkeypatch, tmp_path) -> None:
    _clear(monkeypatch)
    f = _dm(tmp_path, _MODELS + '[defaults]\ndefault_model="m"\n')
    monkeypatch.setenv("ATP_CATALOG", str(f))
    assert resolve_default_model("override") == "override"


def test_resolve_catalog_alias_returned_verbatim(monkeypatch, tmp_path) -> None:
    _clear(monkeypatch)
    body = (
        '[models."m"]\nvendor="v"\nstatus="active"\naliases=["m-latest"]\n'
        '[defaults]\ndefault_model="m-latest"\n'
    )
    monkeypatch.setenv("ATP_CATALOG", str(_dm(tmp_path, body)))
    # The alias passes validation; the resolver returns it verbatim (no
    # canonicalization to the "m" key).
    assert resolve_default_model(None) == "m-latest"


def test_resolve_no_catalog_silent_none(monkeypatch, tmp_path) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    assert resolve_default_model(None) is None


def test_resolve_broken_catalog_warns_and_none(monkeypatch, tmp_path, caplog) -> None:
    import logging

    _clear(monkeypatch)
    monkeypatch.setenv("ATP_CATALOG", str(_dm(tmp_path, "this = = not toml")))
    with caplog.at_level(logging.WARNING):
        assert resolve_default_model(None) is None
    assert "unusable" in caplog.text
```

(`_clear(monkeypatch)` already exists in this file from SP-A — it delenvs `ATP_CATALOG` and
`XDG_CONFIG_HOME`.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/model_catalog/test_loader.py -v -k resolve`
Expected: FAIL — `ImportError: cannot import name 'resolve_default_model'`.

- [ ] **Step 3: Add `resolve_default_model` + logger to `loader.py`**

Add near the top imports:

```python
import logging
```

and after the module constants (near `_INIT_HINT`):

```python
logger = logging.getLogger(__name__)
```

Add the function (after `load_catalog`, before `read_template`):

```python
def resolve_default_model(explicit: str | None = None) -> str | None:
    """Resolve the default model: an explicit override, else the catalog's
    ``[defaults].default_model``, else ``None`` (the caller supplies a fallback).

    Tolerant runtime convenience, NOT ``load_catalog``'s fail-loud contract: a
    missing catalog is expected (out-of-box eval must not require one), and a
    present-but-broken *optional* catalog is logged and ignored rather than
    crashing the evaluator. Whitespace-only ``explicit`` is treated as unset; a
    non-empty ``explicit`` is returned stripped. The catalog value is returned
    verbatim (an alias is not canonicalized).
    """
    if explicit and explicit.strip():
        return explicit.strip()
    try:
        catalog = load_catalog()
    except CatalogNotConfiguredError:
        return None
    except (CatalogTOMLError, CatalogSchemaError) as exc:
        logger.warning("model catalog present but unusable, ignoring: %s", exc)
        return None
    if catalog.defaults is not None and catalog.defaults.default_model:
        return catalog.defaults.default_model
    return None
```

- [ ] **Step 4: Export `resolve_default_model` in `__init__.py`**

Add it to the `from atp.model_catalog.loader import (...)` line and to `__all__` (sorted).

- [ ] **Step 5: Run resolver tests to verify they pass**

Run: `uv run pytest tests/unit/model_catalog/test_loader.py -v`
Expected: all PASS (existing + 7 new).

- [ ] **Step 6: Format, lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add packages/atp-core/atp/model_catalog/loader.py packages/atp-core/atp/model_catalog/__init__.py tests/unit/model_catalog/test_loader.py
git commit -m "feat(model-catalog): tolerant resolve_default_model (SP-C)"
```

---

## Task 3: wire settings + llm_judge to the resolver

**Files:**
- Modify: `atp/core/settings.py` (the `default_llm_model` field, ~line 362)
- Modify: `atp/evaluators/llm_judge.py` (the settings-read line, ~line 203)
- Modify: `tests/unit/evaluators/test_llm_judge.py`

**Interfaces:**
- Consumes: `resolve_default_model` (Task 2).
- Produces: `settings.default_llm_model: str | None` (default None); `llm_judge` resolves its
  default via `resolve_default_model(get_settings().default_llm_model)`.

- [ ] **Step 1: Write the failing chain tests**

Append to `tests/unit/evaluators/test_llm_judge.py`. The file already imports `LLMJudgeConfig`
and `LLMJudgeEvaluator` (verified) — reuse those, do not re-import. **Verified facts:**
`get_settings()` is NOT cached (it builds a fresh `ATPSettings` per call) and, called with no
`config_file` arg (as `llm_judge` does), does NOT auto-load `atp.config.yaml` — so simple
`monkeypatch.delenv` is enough; each `LLMJudgeEvaluator(...)` re-reads the environment. (One
env gotcha: pydantic-settings also reads a repo-root `.env` file; if the dev/CI env has a `.env`
setting `ATP_DEFAULT_LLM_MODEL`, neutralize it too — CI is clean, so this normally does not
arise. If a chain test picks up an unexpected model, check for a local `.env`.)

```python
import logging
from pathlib import Path

import pytest

from atp.evaluators.llm_judge import LLMJudgeConfig, LLMJudgeEvaluator


@pytest.fixture
def clean_model_env(monkeypatch):
    """Isolate every model-resolution input (env only; get_settings is uncached
    and does not auto-load atp.config.yaml)."""
    for var in (
        "ATP_JUDGE_MODEL", "ATP_JUDGE_PROVIDER", "ATP_JUDGE_BASE_URL",
        "ATP_DEFAULT_LLM_MODEL", "ATP_CATALOG", "XDG_CONFIG_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _catalog(tmp_path: Path, default_model: str) -> str:
    f = tmp_path / "agents-catalog.toml"
    f.write_text(
        f'[models."{default_model}"]\nvendor="v"\nstatus="active"\n'
        f'[defaults]\ndefault_model="{default_model}"\n',
        encoding="utf-8",
    )
    return str(f)


def _model(config: LLMJudgeConfig | None = None) -> str:
    return LLMJudgeEvaluator(config=config)._model


def test_chain_no_env_no_catalog_anthropic_fallback(clean_model_env) -> None:
    clean_model_env.setenv("ANTHROPIC_API_KEY", "x")
    assert _model() == "claude-sonnet-4-20250514"


def test_chain_env_default_wins_over_catalog(clean_model_env, tmp_path) -> None:
    clean_model_env.setenv("ATP_DEFAULT_LLM_MODEL", "env-model")
    clean_model_env.setenv("ATP_CATALOG", _catalog(tmp_path, "catalog-model"))
    assert _model() == "env-model"


def test_chain_catalog_used_when_settings_none(clean_model_env, tmp_path) -> None:
    clean_model_env.setenv("ATP_CATALOG", _catalog(tmp_path, "catalog-model"))
    assert _model() == "catalog-model"


def test_chain_judge_model_wins_over_all(clean_model_env, tmp_path) -> None:
    clean_model_env.setenv("ATP_DEFAULT_LLM_MODEL", "env-model")
    clean_model_env.setenv("ATP_CATALOG", _catalog(tmp_path, "catalog-model"))
    assert _model(LLMJudgeConfig(model="explicit-judge")) == "explicit-judge"


def test_chain_bedrock_skips_catalog_default(clean_model_env, tmp_path) -> None:
    from atp.evaluators.llm_judge import DEFAULT_BEDROCK_MODEL

    clean_model_env.setenv("ATP_CATALOG", _catalog(tmp_path, "catalog-model"))
    assert _model(LLMJudgeConfig(provider="bedrock")) == DEFAULT_BEDROCK_MODEL
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/evaluators/test_llm_judge.py -v -k chain`
Expected: FAIL — `test_chain_catalog_used_when_settings_none` fails (today `default_llm_model`
has a hardcoded `"claude-sonnet-4-20250514"` default, so the catalog is never consulted) and
`resolve_default_model` isn't imported into `llm_judge` yet.

- [ ] **Step 3: Make `settings.default_llm_model` optional**

In `atp/core/settings.py` (~line 362) change:

```python
    default_llm_model: str | None = Field(
        default=None,
        description="Default LLM model for evaluators; None defers to the model "
        "catalog's [defaults], then a provider fallback (ADR-003b SP-C).",
    )
```

- [ ] **Step 4: Swap the llm_judge settings read to the resolver**

In `atp/evaluators/llm_judge.py`, add to the top-level imports:

```python
from atp.model_catalog import resolve_default_model
```

In the settings-read block (~line 194-203), replace `self._model = settings.default_llm_model`
with the resolver call:

```python
        if not self._model and self._provider != "bedrock" and not self._base_url:
            try:
                from atp.core.settings import get_settings

                if not self._model:
                    self._model = resolve_default_model(
                        get_settings().default_llm_model
                    )
            except Exception as e:
                logger.debug("Failed to resolve default model: %s", e)
```

(The provider-specific fallback block at ~line 216-222 is unchanged and now also catches the
`None` the resolver can return.)

- [ ] **Step 5: Run the chain tests to verify they pass**

Run: `uv run pytest tests/unit/evaluators/test_llm_judge.py -v -k chain`
Expected: 5 PASS.

- [ ] **Step 6: Regression — full llm_judge + settings suites**

Run: `uv run pytest tests/unit/evaluators/test_llm_judge.py tests/unit/ -q -k "judge or settings"`
Expected: all PASS. If a pre-existing settings test asserted `default_llm_model ==
"claude-sonnet-4-20250514"`, update it to `is None` (the default moved to the resolver fallback)
and note it in the commit.

- [ ] **Step 7: Format, lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add atp/core/settings.py atp/evaluators/llm_judge.py tests/unit/evaluators/test_llm_judge.py
git commit -m "feat(evaluators): llm_judge default model resolves via catalog [defaults] (SP-C)"
```

---

## Task 4: template example + docs

**Files:**
- Modify: `packages/atp-core/atp/model_catalog/data/template.toml`
- Modify: `docs/reference/configuration.md`
- Modify: `CLAUDE.md`, `TODO.md`

- [ ] **Step 1: Add a commented `[defaults]` example to the template**

In `packages/atp-core/atp/model_catalog/data/template.toml`, before the `[models]` table, add a
commented example (keep it inert — commented, placeholder):

```toml
# Optional: a default model for evaluators (must match a [models] id or alias):
#
#   [defaults]
#   default_model = "your-model-id"
```

- [ ] **Step 2: Verify the template still loads inert**

Run: `uv run pytest tests/unit/model_catalog/test_template.py -v`
Expected: PASS (the commented block is inert; parsed `defaults` stays None, `models` stays `{}`).

- [ ] **Step 3: Update `docs/reference/configuration.md`**

Near the `default_llm_model` / `ATP_DEFAULT_LLM_MODEL` row (~line 63), add a note: when
`ATP_DEFAULT_LLM_MODEL` is unset, the evaluator default now defers to the model catalog's
`[defaults].default_model` (if a catalog is configured), then a provider fallback (ADR-003b
SP-C). An empty `ATP_DEFAULT_LLM_MODEL=""` is treated as unset.

- [ ] **Step 4: Update CLAUDE.md + TODO.md**

In `CLAUDE.md` component 25 (or the model-catalog sentence added in SP-E), append: the evaluator's
default model now resolves through the catalog's `[defaults]` plane via
`resolve_default_model()` (ADR-003b SP-C), tolerant of a missing/broken optional catalog. In
`TODO.md`, tick SP-C done (link this plan + the spec); remaining increment: **SP-D** (rename
`atp/catalog/` → test-catalog).

- [ ] **Step 5: Sanity + commit**

Run: `uv run pytest tests/unit/model_catalog/ tests/unit/evaluators/test_llm_judge.py -q && uv run pyrefly check`
Expected: all green; pyrefly 0 errors.

```bash
git add packages/atp-core/atp/model_catalog/data/template.toml docs/reference/configuration.md CLAUDE.md TODO.md
git commit -m "docs(sp-c): [defaults] template example + config docs + 003b SP-C status"
```

---

## Self-Review

**Spec coverage:**
- `CatalogDefaults` + `ModelCatalog.defaults` + referential validator (keys ∪ aliases,
  only-when-truthy+non-empty) → Task 1. ✓
- Tolerant `resolve_default_model` (explicit stripped / empty=unset / catalog / NotConfigured→
  silent None / TOML+Schema→warn+None / alias verbatim) → Task 2. ✓
- `settings.default_llm_model: str | None = None` → Task 3. ✓
- llm_judge one-line swap; provider fallback + config.model/ATP_JUDGE_MODEL/base_url/bedrock
  paths unchanged → Task 3. ✓
- 5 evaluator chain cases (+ alias, + resolver caplog, + schema no-op cases) → Tasks 1-3. ✓
- `load_catalog` fail-loud unchanged; real SSOT (no `[defaults]`) still loads → honored (no task
  touches `load_catalog`/`method/`). ✓
- Template `[defaults]` example + config docs → Task 4. ✓

**Placeholder scan:** no "TBD/handle errors" placeholders. Verified facts baked into Task 3:
`get_settings` is uncached and does not auto-load `atp.config.yaml` (so the fixture is plain
`delenv`, no cache-clear); `LLMJudgeConfig`/`LLMJudgeEvaluator`/`DEFAULT_BEDROCK_MODEL` are
importable from `atp.evaluators.llm_judge`. The only environmental caveat (a local `.env` setting
`ATP_DEFAULT_LLM_MODEL`) is documented in Task 3 Step 1. The template/docs edits are prose with
the exact text to add.

**Type consistency:** `CatalogDefaults`, `ModelCatalog.defaults`, `resolve_default_model(explicit:
str | None) -> str | None`, `settings.default_llm_model: str | None` are used identically across
Tasks 1-4. The validator name `_default_model_in_models` and the SP-E validator coexist.
