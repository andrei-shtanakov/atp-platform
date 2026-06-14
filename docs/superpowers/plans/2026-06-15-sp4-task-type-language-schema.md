# SP-4: task_type + language in the case schema → persisted columns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `task_type` + `language` to the `agent-eval-case` schema and thread them into the persisted store, so `suite_executions.task_type` (added but unpopulated in SP-1) gets filled and a new `language` dimension is captured — upgrading the SP-3 leaderboard from suite-scoped to genuinely task_type-scoped and giving arbiter the language axis.

**Architecture:** Two new optional fields on the case schema flow exactly like SP-1's dimensions: emitted as `task_type_*`/`language_*` tags by the loader, parsed by the pure `dimensions.case_dimensions` (case-level) and `aggregate_run` (run-level "common value"), persisted by storage into columns. A small idempotent Alembic migration adds the missing columns; `_add_missing_columns` covers prod SQLite.

**Tech Stack:** Python 3.12, uv, pydantic, SQLAlchemy + Alembic, pytest; packages `atp-method`, `atp-dashboard`, `atp/cli`.

**Companion docs:** spec `docs/superpowers/specs/2026-06-14-eval-results-architecture-design.md` (§5, SP-4 in §10), ADR-006. SP-1 (store + persist machinery) and SP-3 (leaderboard) merged on `main`.

**Decision (baked in):** `task_type` and `language` are **optional free strings** (lowercase token pattern), NOT a closed enum — not every family maps to an arbiter coding-TaskType (e.g. `req-extraction`), and the closed task_type↔benchmark_id set is enforced at the export boundary by `atp_method.taxonomy.benchmark_id_for` (SP-2's job), not the authoring schema.

**Scope:**
- Schema: `task_type: str | None`, `language: str | None` on `AgentEvalCase` (pydantic + JSON contract).
- Loader: emit `task_type_<v>` / `language_<v>` tags when present.
- Columns (one migration): `suite_executions.language` (run-level) + `test_executions.task_type` + `test_executions.language` (case-level). `suite_executions.task_type` already exists (SP-1).
- Persist: `case_dimensions` adds task_type/language; `aggregate_run` derives run-level task_type/language (the single distinct value across cases, else None); storage threads them.
- Migrate the 2 code-review cases: `task_type: review`, `language: python`.

**Scope guard (NOT in SP-4):** the export sink reading these (SP-2); switching the SP-3 leaderboard's primary grouping to task_type or adding a task_type filter (its column already displays the now-populated value — a filter is a later SP-3 polish); matrix/drill-down (SP-5). `task_type` stays optional (no required-field migration / case backfill beyond the 2 code-review cases).

---

## File Structure

- Modify `packages/atp-method/atp_method/schema.py` — add `task_type`/`language` fields to `AgentEvalCase`.
- Modify `method/agent-eval-case.schema.json` — add the two optional properties (lowercase-token pattern; `additionalProperties:false` already set).
- Modify `packages/atp-method/atp_method/loader.py` (`_tags`) — emit `task_type_<v>`/`language_<v>` when set.
- Modify `packages/atp-dashboard/atp/dashboard/models.py` — add `suite_executions.language`, `test_executions.task_type`, `test_executions.language`.
- Create `migrations/dashboard/versions/d4e5f6a7b8c9_task_type_language.py` — idempotent (inspector-guarded) migration.
- Modify `packages/atp-dashboard/atp/dashboard/dimensions.py` — `case_dimensions` (+task_type/language) and `aggregate_run` (+run-level task_type/language).
- Modify `packages/atp-dashboard/atp/dashboard/storage.py` — `create_test_execution` spreads task_type/language; `update_suite_execution(aggregates=)` sets language (task_type already settable via the aggregates dict — verify).
- Modify `method/cases/code-review/case-code-review-sqli-{clean,moderate}-001.yaml` — add `task_type: review`, `language: python`.
- Tests: `packages/atp-method/tests/test_schema.py`, `test_loader.py`, `test_schema_contract.py`, `test_cases_load.py` (extend); `tests/unit/dashboard/test_eval_dimension_columns.py`, `test_dimensions.py`, `test_storage.py` (extend); `tests/integration/cli/test_persist_dimensions.py` (extend).

**Test cwd:** atp-method tests from `packages/atp-method`; the rest from repo root.

**Migration safety:** all new columns nullable; idempotent upgrade/downgrade (guard with `sa.inspect`), mirroring the SP-1 migration `c3d4e5f6a7b8` (read it for the exact `_columns()`/inspector pattern). Merge auto-deploys; `_add_missing_columns` adds the columns on the live SQLite at startup.

---

## Task 1: Schema fields (pydantic + JSON contract)

**Files:**
- Modify: `packages/atp-method/atp_method/schema.py`
- Modify: `method/agent-eval-case.schema.json`
- Test: `packages/atp-method/tests/test_schema.py` + `test_schema_contract.py` (extend)

- [ ] **Step 1: Write failing tests.** Append to `packages/atp-method/tests/test_schema.py` (reuse its `AgentEvalCase`/`pytest`/`ValidationError` imports + any case-dict helper):

```python
def test_task_type_and_language_optional_and_default_none() -> None:
    from atp_method.schema import AgentEvalCase
    # a minimal valid case WITHOUT task_type/language still validates
    case = AgentEvalCase.model_validate(_VALID_CASE_DICT)  # reuse the file's helper/const
    assert case.task_type is None
    assert case.language is None


def test_task_type_and_language_accepted() -> None:
    from atp_method.schema import AgentEvalCase
    doc = {**_VALID_CASE_DICT, "task_type": "review", "language": "python"}
    case = AgentEvalCase.model_validate(doc)
    assert case.task_type == "review"
    assert case.language == "python"


def test_task_type_rejects_non_token() -> None:
    from atp_method.schema import AgentEvalCase
    with pytest.raises(ValidationError):
        AgentEvalCase.model_validate({**_VALID_CASE_DICT, "task_type": "Code Review!"})
```
If the file has no reusable valid-case dict, add a module-level `_VALID_CASE_DICT` (a known-valid case with `type: programmatic, checker: findings_match, expected_findings: []`). And append to `test_schema_contract.py`:
```python
def test_contract_accepts_task_type_language() -> None:
    case = _case({  # the file's grader-dict helper
        "type": "programmatic", "checker": "findings_match",
        "expected_findings": [], "critical_check": "c", "scoring": "s",
    })
    case["task_type"] = "review"
    case["language"] = "python"
    jsonschema.validate(case, SCHEMA)  # must not raise
```

- [ ] **Step 2: Run, expect FAIL:**
`cd packages/atp-method && uv run pytest tests/test_schema.py tests/test_schema_contract.py -q -k "task_type or language"`

- [ ] **Step 3: Edit `schema.py`.** Add a token type alias near the top (after the other Literals):
```python
_TOKEN_RE = r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$"
```
Add fields to `AgentEvalCase` (after `axis_level`, before `tags`):
```python
    task_type: str | None = Field(default=None, pattern=_TOKEN_RE)
    language: str | None = Field(default=None, pattern=_TOKEN_RE)
```
(Pydantic applies `pattern` only when the value is non-None — confirm with the reject test.)

- [ ] **Step 4: Edit `method/agent-eval-case.schema.json`.** Add to top-level `properties` (NOT to `required`):
```json
    "task_type": {
      "type": "string",
      "pattern": "^[a-z0-9]+(?:[-_][a-z0-9]+)*$",
      "description": "Routing task type (arbiter TaskType canon, e.g. 'review'). Optional; maps to a benchmark_id at the arbiter export boundary."
    },
    "language": {
      "type": "string",
      "pattern": "^[a-z0-9]+(?:[-_][a-z0-9]+)*$",
      "description": "Primary language of the case material (e.g. 'python'). Optional; arbiter routes by language."
    },
```
Verify JSON still parses: `uv run python -c "import json; json.load(open('method/agent-eval-case.schema.json')); print('ok')"`.

- [ ] **Step 5: Run, expect PASS:**
`cd packages/atp-method && uv run pytest tests/test_schema.py tests/test_schema_contract.py -q`

- [ ] **Step 6: Commit:**
```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-method && uv run pyrefly check
git add packages/atp-method/atp_method/schema.py method/agent-eval-case.schema.json packages/atp-method/tests/test_schema.py packages/atp-method/tests/test_schema_contract.py
git commit -m "feat(method): optional task_type + language on agent-eval-case (SP-4)"
```

---

## Task 2: Loader emits task_type/language tags

**Files:**
- Modify: `packages/atp-method/atp_method/loader.py` (`_tags`)
- Test: `packages/atp-method/tests/test_loader.py` (extend)

- [ ] **Step 1: Write failing test** — append to `test_loader.py` (reuse its case-dict pattern; set `task_type: "review"`, `language: "python"` on the case):
```python
def test_tags_include_task_type_and_language_when_set() -> None:
    from atp_method.loader import case_to_test_definition
    from atp_method.schema import AgentEvalCase
    doc = {**_LOADER_CASE_DICT, "task_type": "review", "language": "python"}
    td = case_to_test_definition(AgentEvalCase.model_validate(doc))
    assert "task_type_review" in td.tags
    assert "language_python" in td.tags


def test_tags_omit_task_type_language_when_absent() -> None:
    from atp_method.loader import case_to_test_definition
    from atp_method.schema import AgentEvalCase
    td = case_to_test_definition(AgentEvalCase.model_validate(_LOADER_CASE_DICT))
    assert not any(t.startswith("task_type_") for t in td.tags)
    assert not any(t.startswith("language_") for t in td.tags)
```
(Use the file's existing valid case dict; if none is shared, add `_LOADER_CASE_DICT`.)

- [ ] **Step 2: Run, expect FAIL:**
`cd packages/atp-method && uv run pytest tests/test_loader.py -q -k "task_type or language"`

- [ ] **Step 3: Edit `_tags`** — after building `derived`, conditionally append the two tags (only when the field is set, so absent fields produce no tag):
```python
    if case.task_type:
        derived.append(f"task_type_{case.task_type}")
    if case.language:
        derived.append(f"language_{case.language}")
```
(Place before the dedup loop so they flow through it.)

- [ ] **Step 4: Run, expect PASS (full loader file):**
`cd packages/atp-method && uv run pytest tests/test_loader.py -q`

- [ ] **Step 5: Commit:**
```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-method && uv run pyrefly check
git add packages/atp-method/atp_method/loader.py packages/atp-method/tests/test_loader.py
git commit -m "feat(method): emit task_type_/language_ tags when set (SP-4)"
```

---

## Task 3: Columns + idempotent migration

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/models.py`
- Create: `migrations/dashboard/versions/d4e5f6a7b8c9_task_type_language.py`
- Test: `tests/unit/dashboard/test_eval_dimension_columns.py` (extend)

- [ ] **Step 1: Extend the column-presence test** — in `tests/unit/dashboard/test_eval_dimension_columns.py` add `"language"` to `RUN_COLS` and add `"task_type"`, `"language"` to `CASE_COLS`.

- [ ] **Step 2: Run, expect FAIL:**
`uv run pytest tests/unit/dashboard/test_eval_dimension_columns.py -v`

- [ ] **Step 3: Add columns to `models.py`.** In `TestExecution` (next to the other SP-1 dims):
```python
    task_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    language: Mapped[str | None] = mapped_column(String(50), nullable=True)
```
In `SuiteExecution` (next to `task_type`):
```python
    language: Mapped[str | None] = mapped_column(String(50), nullable=True)
```

- [ ] **Step 4: Create the migration** `migrations/dashboard/versions/d4e5f6a7b8c9_task_type_language.py` — READ `c3d4e5f6a7b8_eval_dimensions.py` first and copy its idempotent inspector-guarded `_columns()` pattern exactly. `revision="d4e5f6a7b8c9"`, `down_revision="c3d4e5f6a7b8"`. Add columns (all nullable):
```python
_TEST_COLS = [("task_type", sa.String(length=50)), ("language", sa.String(length=50))]
_SUITE_COLS = [("language", sa.String(length=50))]
```
`upgrade()` adds each only if absent (inspector); `downgrade()` drops each only if present. No index.

- [ ] **Step 5: Run, expect PASS + migration round-trips on a fresh DB:**
```bash
uv run pytest tests/unit/dashboard/test_eval_dimension_columns.py -v
cd "$(git rev-parse --show-toplevel)"
uv run alembic -c alembic.ini heads   # expect d4e5f6a7b8c9 (head)
# fresh-DB upgrade/downgrade (set a temp ATP_DATABASE_URL=sqlite:///$(mktemp -u).db):
uv run alembic -c alembic.ini upgrade head && uv run alembic -c alembic.ini downgrade -1
```
(If alembic needs the DB url env, set it as the SP-1 review did; confirm upgrade→head and downgrade -1 both succeed.)

- [ ] **Step 6: Commit:**
```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-dashboard migrations && uv run pyrefly check
git add packages/atp-dashboard/atp/dashboard/models.py migrations/dashboard/versions/d4e5f6a7b8c9_task_type_language.py tests/unit/dashboard/test_eval_dimension_columns.py
git commit -m "feat(dashboard): task_type/language columns + idempotent migration (SP-4)"
```

---

## Task 4: Persist task_type/language (dimensions + storage)

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/dimensions.py`
- Modify: `packages/atp-dashboard/atp/dashboard/storage.py`
- Test: `tests/unit/dashboard/test_dimensions.py` + `test_storage.py` (extend)

- [ ] **Step 1: Write failing tests.**
In `test_dimensions.py`: extend the tag-based case to include `task_type_review`/`language_python` and assert `dims["task_type"]=="review"`, `dims["language"]=="python"`; add an `aggregate_run` test where all cases share `task_type="review"`/`language="python"` → run dict has those, and a mixed-task_type case → `task_type=None`.
In `test_storage.py`: extend the SP-3/SP-1 seeding helper or add assertions that `create_test_execution(dimensions={... "task_type":"review","language":"python"})` sets `te.task_type`/`te.language`, and `update_suite_execution(aggregates={... "language":"python", "task_type":"review"})` sets `suite.language`/`suite.task_type`.

- [ ] **Step 2: Run, expect FAIL:**
`uv run pytest tests/unit/dashboard/test_dimensions.py tests/unit/dashboard/test_storage.py -q -k "task_type or language"`

- [ ] **Step 3: Edit `dimensions.py`.** In `case_dimensions(...)` return dict add:
```python
        "task_type": _tag_value(tags, "task_type_"),
        "language": _tag_value(tags, "language_"),
```
In `aggregate_run(...)`, after computing the rate stuff, derive the run-level common value (single distinct non-null value across cases, else None) and add to the returned dict:
```python
    def _common(key: str) -> str | None:
        vals = {c.get(key) for c in case_dims if c.get(key)}
        return next(iter(vals)) if len(vals) == 1 else None
    # ... in the return dict:
        "task_type": _common("task_type"),
        "language": _common("language"),
```
Also add `task_type`/`language` keys (=None) to the early-return empty dict so the shape is stable.

- [ ] **Step 4: Edit `storage.py`.** In `create_test_execution`, add `task_type=d.get("task_type")` and `language=d.get("language")` to the `TestExecution(...)` construction. In `update_suite_execution`, when `aggregates` is provided also set `suite_execution.task_type = aggregates.get("task_type")` and `suite_execution.language = aggregates.get("language")` (alongside the existing rate columns).

- [ ] **Step 5: Run, expect PASS:**
`uv run pytest tests/unit/dashboard/test_dimensions.py tests/unit/dashboard/test_storage.py -q`

- [ ] **Step 6: Commit:**
```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-dashboard && uv run pyrefly check
git add packages/atp-dashboard/atp/dashboard/dimensions.py packages/atp-dashboard/atp/dashboard/storage.py tests/unit/dashboard/test_dimensions.py tests/unit/dashboard/test_storage.py
git commit -m "feat(dashboard): persist task_type + language (case + run level) (SP-4)"
```

---

## Task 5: Migrate the code-review cases + end-to-end persist test

**Files:**
- Modify: `method/cases/code-review/case-code-review-sqli-clean-001.yaml`
- Modify: `method/cases/code-review/case-code-review-sqli-moderate-001.yaml`
- Test: `packages/atp-method/tests/test_cases_load.py` (extend) + `tests/integration/cli/test_persist_dimensions.py` (extend)

- [ ] **Step 1: Extend tests.**
In `test_cases_load.py`: assert each case now has `case.task_type == "review"` and `case.language == "python"` (and still validates against the JSON contract).
In `test_persist_dimensions.py`: extend the seeded method run's `TestDefinition.tags` to include `task_type_review`/`language_python`, and assert the persisted `TestExecution.task_type=="review"`/`.language=="python"` and `SuiteExecution.task_type=="review"`/`.language=="python"`.

- [ ] **Step 2: Run, expect FAIL:**
```bash
cd packages/atp-method && uv run pytest tests/test_cases_load.py -q
cd "$(git rev-parse --show-toplevel)" && uv run pytest tests/integration/cli/test_persist_dimensions.py -q
```

- [ ] **Step 3: Migrate both YAML cases** — add two top-level keys (e.g. after `axis_level:`):
```yaml
task_type: review
language: python
```
Leave everything else unchanged.

- [ ] **Step 4: Run, expect PASS:**
```bash
cd packages/atp-method && uv run pytest tests/test_cases_load.py -q
cd "$(git rev-parse --show-toplevel)" && uv run pytest tests/integration/cli/test_persist_dimensions.py -q
```

- [ ] **Step 5: Commit:**
```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-method && uv run pyrefly check
git add method/cases/code-review/case-code-review-sqli-clean-001.yaml method/cases/code-review/case-code-review-sqli-moderate-001.yaml packages/atp-method/tests/test_cases_load.py tests/integration/cli/test_persist_dimensions.py
git commit -m "feat(method): code-review cases declare task_type=review, language=python (SP-4)"
```

---

## Task 6: Regression + quality gates

**Files:** none (verification only)

- [ ] **Step 1: Affected suites:**
```bash
cd "$(git rev-parse --show-toplevel)"
uv run pytest tests/unit/dashboard tests/integration/cli -q
cd packages/atp-method && uv run pytest -q
```
Expected: all PASS.

- [ ] **Step 2: Fresh-DB columns present (init_database + _add_missing_columns):**
```bash
cd "$(git rev-parse --show-toplevel)"
uv run python -c "
import asyncio, os, tempfile
os.environ['ATP_DATABASE_URL']='sqlite+aiosqlite:///'+tempfile.mktemp(suffix='.db')
from atp.dashboard.database import init_database, get_database
import sqlalchemy as sa
async def chk():
    await init_database(); db=get_database()
    async with db.session() as s:
        t={c[1] for c in (await s.execute(sa.text('PRAGMA table_info(test_executions)'))).fetchall()}
        su={c[1] for c in (await s.execute(sa.text('PRAGMA table_info(suite_executions)'))).fetchall()}
    assert {'task_type','language'} <= t, t
    assert 'language' in su, su
    print('OK: SP-4 columns present')
asyncio.run(chk())
"
```

- [ ] **Step 3: Lint + types:**
```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check atp/ packages/atp-dashboard packages/atp-method migrations
uv run pyrefly check
```
Expected: ruff clean; pyrefly 0 errors.

- [ ] **Step 4: Commit any formatting:**
```bash
cd "$(git rev-parse --show-toplevel)"
git add -A && git commit -m "chore(sp-4): formatting" || echo "nothing to commit"
```

---

## Self-Review (completed during authoring)

- **Spec coverage:** task_type + language in the case schema (Task 1) → tags (Task 2) → columns + migration (Task 3) → persisted case+run level (Task 4) → real cases declare them (Task 5). Fills `suite_executions.task_type` (SP-1 left null) and adds the language dimension.
- **Type/name consistency:** the `task_type_`/`language_` tag prefixes match `_tag_value(...)` extraction; `case_dimensions`/`aggregate_run` keys match the new column names and the `create_test_execution(dimensions=)`/`update_suite_execution(aggregates=)` setters; migration column names == ORM column names == the column-presence test.
- **Optional + backward-compatible:** both fields optional (`None` default), nullable columns, idempotent migration — existing cases (req-extraction) and existing rows are unaffected; only the 2 code-review cases opt in.
- **Run-level "common value":** `aggregate_run` sets run-level task_type/language only when all graded cases share one value (else None) — correct for single-family runs, safe for mixed runs.
- **Placeholders:** Tasks reuse the test files' existing valid-case dicts (named `_VALID_CASE_DICT`/`_LOADER_CASE_DICT` if not already shared) and the SP-1 migration's inspector pattern — the implementer reads those first; all new code is given in full.
- **Prod risk:** nullable columns, no backfill, idempotent reversible migration; deploy-before-migrate covered by `_add_missing_columns`.
