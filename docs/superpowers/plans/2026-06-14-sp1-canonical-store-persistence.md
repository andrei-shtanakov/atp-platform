# SP-1: Canonical dimensioned store + persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve the dashboard's `SuiteExecution`/`TestExecution` tables with the eval dimensions + gate outcomes, and populate them from every `atp test` run, so results become query-friendly (incl. atp-method) — the foundation for SP-2 (export sink) and SP-3 (dashboard leaderboard/trend).

**Architecture:** New **nullable** columns on `test_executions` (case-level dims + the `CaseVerdict` outcomes) and `suite_executions` (run-level aggregates + `run_uuid`). At save time, `_save_results_to_db` already has the `TestDefinition` (with dimension tags) and `all_eval_results` (the `critical_check` EvalCheck's `details` = the `CaseVerdict` dump) — SP-1 maps those into the columns via a pure extractor, then aggregates run-level metrics from the per-case rows. An Alembic migration adds the columns; `_add_missing_columns` covers the live prod SQLite at startup.

**Tech Stack:** Python 3.12, uv, SQLAlchemy, Alembic, pytest (anyio); packages `atp-method`, `atp-dashboard`; `atp/cli/main.py`.

**Companion docs:** spec `docs/superpowers/specs/2026-06-14-eval-results-architecture-design.md` (§5, §11), ADR-006. Phase A (CaseVerdict + checker registry + taxonomy) is merged on `main`.

**Scope (option A — populated-focused):**
- *case-level (`test_executions`):* `axis_level`, `capability`, `family`, `case_version`, `critical_pass`, `malformed`, `recall`, `precision`, `fp_count`, `rubric_score`, `grader_version`.
- *run-level (`suite_executions`):* `task_type` (nullable; sourced in SP-4), `run_uuid`, `critical_pass_rate`, `malformed_rate`, `mean_rubric`, `breakpoint_axis_level`.

**Scope guard (NOT in SP-1):** `language`/`run_mode`/`envelope_pin` columns (deferred to the phase that sources them — SP-4/SP-2); the `report_benchmark` export reading the store (SP-2); dashboard views (SP-3); folding `run_pipe_check.py` into `atp test` (SP-2). All new columns are **nullable** — native (non-method) runs simply leave the method-only columns null.

**Sourcing map (where each column's value comes from at save time):**
- `axis_level`/`capability`/`family`/`case_version` ← parsed from `TestDefinition.tags` (`level_*`/`capability_*`/`family_*`/`version_*`; the `version_*` tag is added in Task 2).
- `critical_pass`/`malformed`/`recall`/`precision`/`fp_count`/`grader_version` ← the `critical_check` EvalCheck's `details` (the `CaseVerdict` dump) in `all_eval_results[test_id]`.
- `rubric_score` ← the `rubric` EvalCheck's `score` (the findings checker reports `rubric_score=0.0` in its own verdict; the real rubric score is the separate `method_rubric` result).
- run-level aggregates ← computed from the per-case values.
- `task_type` ← left null (SP-4).

---

## File Structure

- Modify `packages/atp-dashboard/atp/dashboard/models.py` — add the nullable columns to `TestExecution` + `SuiteExecution`.
- Create `migrations/dashboard/versions/c3d4e5f6a7b8_eval_dimensions.py` — Alembic migration (down_revision `b2c3d4e5f6a7`).
- Modify `packages/atp-method/atp_method/loader.py` (`_tags`) — add `version_{case.version}` tag.
- Create `packages/atp-dashboard/atp/dashboard/dimensions.py` — pure extractor: `(TestDefinition, list[EvalResult]) → case-dims dict`, plus run-level `aggregate(case_dims) → run dict` (incl. breakpoint). No DB, fully unit-testable.
- Modify `packages/atp-dashboard/atp/dashboard/storage.py` — `create_test_execution`/`update_test_execution` accept the case columns; `create_suite_execution_by_name` sets `run_uuid`; `update_suite_execution` accepts the run aggregates.
- Modify `atp/cli/main.py` (`_save_results_to_db`) — call the extractor + thread values into the storage calls.
- Tests: `packages/atp-method/tests/test_loader.py` (version tag), `tests/unit/dashboard/test_dimensions.py` (new), `tests/unit/dashboard/test_storage.py` (extend), `tests/integration/cli/test_persist_dimensions.py` (new).

**Test cwd:** dashboard/cli/core tests run from repo root (`uv run pytest …`); atp-method tests from `packages/atp-method`.

**Migration safety:** all columns nullable, no server-side backfill; `_add_missing_columns` adds them to the prod SQLite at startup (merge auto-deploys). The Alembic migration is the system of record; keep `upgrade`/`downgrade` symmetric.

---

## Task 1: Add the columns to the ORM models + Alembic migration

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/models.py`
- Create: `migrations/dashboard/versions/c3d4e5f6a7b8_eval_dimensions.py`
- Test: `tests/unit/dashboard/test_eval_dimension_columns.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/dashboard/test_eval_dimension_columns.py
"""SP-1: SuiteExecution/TestExecution carry the eval dimension columns."""

from atp.dashboard.models import SuiteExecution, TestExecution

CASE_COLS = {
    "axis_level", "capability", "family", "case_version",
    "critical_pass", "malformed", "recall", "precision", "fp_count",
    "rubric_score", "grader_version",
}
RUN_COLS = {
    "task_type", "run_uuid", "critical_pass_rate", "malformed_rate",
    "mean_rubric", "breakpoint_axis_level",
}


def test_test_execution_has_case_columns() -> None:
    cols = set(TestExecution.__table__.columns.keys())
    assert CASE_COLS <= cols


def test_suite_execution_has_run_columns() -> None:
    cols = set(SuiteExecution.__table__.columns.keys())
    assert RUN_COLS <= cols
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/dashboard/test_eval_dimension_columns.py -v`
Expected: FAIL (columns missing).

- [ ] **Step 3: Add the columns to the models**

In `packages/atp-dashboard/atp/dashboard/models.py`, in `class TestExecution`, after the `statistics` column add:

```python
    # --- SP-1 eval dimensions (nullable; populated for method runs) ---
    axis_level: Mapped[str | None] = mapped_column(String(50), nullable=True)
    capability: Mapped[str | None] = mapped_column(String(50), nullable=True)
    family: Mapped[str | None] = mapped_column(String(120), nullable=True)
    case_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    critical_pass: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    malformed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    fp_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rubric_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    grader_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
```

In `class SuiteExecution`, after the `error` column add:

```python
    # --- SP-1 run-level dimensions + aggregates (nullable) ---
    task_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    run_uuid: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    critical_pass_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    malformed_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_rubric: Mapped[float | None] = mapped_column(Float, nullable=True)
    breakpoint_axis_level: Mapped[str | None] = mapped_column(String(50), nullable=True)
```

Ensure `Boolean`, `Float`, `Integer`, `String` are imported from `sqlalchemy` at the top of the file (Integer/String/Float/DateTime/Text are already imported; add `Boolean` if missing).

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/dashboard/test_eval_dimension_columns.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Create the Alembic migration**

```python
# migrations/dashboard/versions/c3d4e5f6a7b8_eval_dimensions.py
"""SP-1 eval dimensions + run aggregates

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-06-14

Adds nullable dimension/outcome columns to test_executions and run-level
aggregate columns to suite_executions (eval-results architecture SP-1).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: str | Sequence[str] | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TEST_COLS = [
    ("axis_level", sa.String(length=50)),
    ("capability", sa.String(length=50)),
    ("family", sa.String(length=120)),
    ("case_version", sa.Integer()),
    ("critical_pass", sa.Boolean()),
    ("malformed", sa.Boolean()),
    ("recall", sa.Float()),
    ("precision", sa.Float()),
    ("fp_count", sa.Integer()),
    ("rubric_score", sa.Float()),
    ("grader_version", sa.String(length=80)),
]
_SUITE_COLS = [
    ("task_type", sa.String(length=50)),
    ("run_uuid", sa.String(length=36)),
    ("critical_pass_rate", sa.Float()),
    ("malformed_rate", sa.Float()),
    ("mean_rubric", sa.Float()),
    ("breakpoint_axis_level", sa.String(length=50)),
]


def upgrade() -> None:
    for name, type_ in _TEST_COLS:
        op.add_column("test_executions", sa.Column(name, type_, nullable=True))
    for name, type_ in _SUITE_COLS:
        op.add_column("suite_executions", sa.Column(name, type_, nullable=True))
    op.create_index(
        "idx_suite_run_uuid", "suite_executions", ["run_uuid"], unique=False
    )


def downgrade() -> None:
    op.drop_index("idx_suite_run_uuid", table_name="suite_executions")
    for name, _ in _SUITE_COLS:
        op.drop_column("suite_executions", name)
    for name, _ in _TEST_COLS:
        op.drop_column("test_executions", name)
```

- [ ] **Step 6: Verify the migration is the new head and round-trips**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
uv run alembic -c migrations/dashboard/alembic.ini heads
```
Expected: prints `c3d4e5f6a7b8 (head)`. (If the alembic.ini path differs, find it: `find migrations -name alembic.ini`. If alembic isn't wired for offline CLI here, instead assert the file parses: `uv run python -c "import importlib.util,glob; p=glob.glob('migrations/dashboard/versions/c3d4e5f6a7b8_*.py')[0]; importlib.util.spec_from_file_location('m',p); print('ok',p)"`.)

- [ ] **Step 7: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-dashboard migrations && uv run pyrefly check
git add packages/atp-dashboard/atp/dashboard/models.py migrations/dashboard/versions/c3d4e5f6a7b8_eval_dimensions.py tests/unit/dashboard/test_eval_dimension_columns.py
git commit -m "feat(dashboard): SP-1 eval dimension + aggregate columns + migration"
```

---

## Task 2: Source `case_version` via a `version_*` tag

**Files:**
- Modify: `packages/atp-method/atp_method/loader.py` (`_tags`)
- Test: `packages/atp-method/tests/test_loader.py` (extend)

- [ ] **Step 1: Write the failing test** — append to `packages/atp-method/tests/test_loader.py`:

```python
from atp_method.loader import case_to_test_definition as _c2td  # noqa: E402
from atp_method.schema import AgentEvalCase as _AEC  # noqa: E402


def test_tags_include_case_version() -> None:
    case = {
        "id": "case-1", "version": 3, "family": "f", "status": "active",
        "suite_type": "probe", "capability": "safety_compliance",
        "construction_axis": "adversarial_environment", "axis_level": "moderate",
        "instruction": "x", "artifacts": [{"id": "d", "type": "text", "content": "x"}],
        "environment": {"tools": ["file_read"], "side_effects": "none"},
        "expected_failure_mode": "m",
        "grader": {"type": "programmatic", "checker": "findings_match",
                   "expected_findings": [], "critical_check": "c", "scoring": "s"},
        "provenance": {"author": "a", "created": "2026-06-14"},
    }
    td = _c2td(_AEC.model_validate(case))
    assert "version_3" in td.tags
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd packages/atp-method && uv run pytest tests/test_loader.py -k case_version -v`
Expected: FAIL (`version_3` not in tags).

- [ ] **Step 3: Add the tag** — in `_tags`, add `f"version_{case.version}"` to the `derived` list (alongside `family_*`/`level_*`).

- [ ] **Step 4: Run it to verify it passes**

Run: `cd packages/atp-method && uv run pytest tests/test_loader.py -q`
Expected: PASS (all loader tests).

- [ ] **Step 5: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-method && uv run pyrefly check
git add packages/atp-method/atp_method/loader.py packages/atp-method/tests/test_loader.py
git commit -m "feat(method): emit version_<n> tag so SP-1 can persist case_version"
```

---

## Task 3: Pure dimension extractor + aggregator

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/dimensions.py`
- Test: `tests/unit/dashboard/test_dimensions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/dashboard/test_dimensions.py
"""SP-1: pure extractor mapping tags + eval results -> dimension columns."""

from atp.core.results import EvalCheck, EvalResult
from atp.dashboard.dimensions import aggregate_run, case_dimensions
from atp.loader.models import TaskDefinition, TestDefinition


def _td(tags: list[str]) -> TestDefinition:
    return TestDefinition(
        id="case-1", name="f (moderate)", tags=tags,
        task=TaskDefinition(description="x"),
    )


def _critical(details: dict) -> EvalResult:
    return EvalResult(
        evaluator="agent_eval_case",
        checks=[EvalCheck(name="critical_check", passed=details["critical_pass"],
                          score=1.0 if details["critical_pass"] else 0.0,
                          message="", details=details)],
    )


def _rubric(score: float) -> EvalResult:
    return EvalResult(
        evaluator="agent_eval_case",
        checks=[EvalCheck(name="rubric", passed=True, score=score, message="")],
    )


def test_case_dimensions_from_tags_and_verdict() -> None:
    td = _td(["level_moderate", "capability_safety_compliance",
              "family_code_review_planted_defect", "version_2"])
    verdict = {"critical_pass": True, "malformed": False, "recall": 1.0,
               "precision": 1.0, "false_positives": [], "fp_count": 0,
               "grader_version": "findings_match@1"}
    dims = case_dimensions(td, [_critical(verdict), _rubric(0.8)])
    assert dims["axis_level"] == "moderate"
    assert dims["capability"] == "safety_compliance"
    assert dims["family"] == "code_review_planted_defect"
    assert dims["case_version"] == 2
    assert dims["critical_pass"] is True
    assert dims["malformed"] is False
    assert dims["recall"] == 1.0
    assert dims["fp_count"] == 0
    assert dims["rubric_score"] == 0.8
    assert dims["grader_version"] == "findings_match@1"


def test_case_dimensions_native_run_is_all_none() -> None:
    # No method tags, no critical_check details -> dims are None (nullable cols).
    dims = case_dimensions(_td([]), [])
    assert dims["axis_level"] is None
    assert dims["critical_pass"] is None
    assert dims["case_version"] is None


def test_aggregate_run_rates_and_breakpoint() -> None:
    cases = [
        {"axis_level": "clean", "critical_pass": True, "malformed": False,
         "rubric_score": 0.9},
        {"axis_level": "moderate", "critical_pass": False, "malformed": False,
         "rubric_score": 0.4},
        {"axis_level": "severe", "critical_pass": False, "malformed": True,
         "rubric_score": 0.0},
    ]
    agg = aggregate_run(cases)
    assert agg["critical_pass_rate"] == round(1 / 3, 6)
    assert agg["malformed_rate"] == round(1 / 3, 6)
    assert agg["mean_rubric"] == round((0.9 + 0.4 + 0.0) / 3, 6)
    assert agg["breakpoint_axis_level"] == "moderate"


def test_aggregate_run_empty_is_none() -> None:
    agg = aggregate_run([])
    assert agg["critical_pass_rate"] is None
    assert agg["breakpoint_axis_level"] is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/dashboard/test_dimensions.py -v`
Expected: FAIL (`ModuleNotFoundError: atp.dashboard.dimensions`).

- [ ] **Step 3: Create the module**

```python
# packages/atp-dashboard/atp/dashboard/dimensions.py
"""Pure mappers from a run's TestDefinition tags + eval results to the SP-1
dimension/outcome columns. No DB — unit-testable in isolation.

Method runs carry `level_*`/`capability_*`/`family_*`/`version_*` tags and a
`critical_check` EvalCheck whose `details` is the CaseVerdict dump. Native runs
have neither, so every field degrades to None (the columns are nullable).
"""

from typing import Any

from atp.core.results import EvalResult
from atp.loader.models import TestDefinition

_AXIS_ORDER = ["clean", "mild", "moderate", "severe", "very_severe"]


def _tag_value(tags: list[str], prefix: str) -> str | None:
    for t in tags:
        if t.startswith(prefix):
            return t[len(prefix):]
    return None


def _critical_details(results: list[EvalResult]) -> dict[str, Any]:
    """The critical_check EvalCheck.details (CaseVerdict dump), or {}."""
    for r in results:
        for c in r.checks:
            if c.name == "critical_check" and c.details:
                return c.details
    return {}


def _rubric_score(results: list[EvalResult]) -> float | None:
    for r in results:
        for c in r.checks:
            if c.name == "rubric":
                return c.score
    return None


def case_dimensions(
    test: TestDefinition, eval_results: list[EvalResult]
) -> dict[str, Any]:
    """Map one case's tags + eval results into the test_executions columns."""
    tags = test.tags or []
    v = _critical_details(eval_results)
    version = _tag_value(tags, "version_")
    return {
        "axis_level": _tag_value(tags, "level_"),
        "capability": _tag_value(tags, "capability_"),
        "family": _tag_value(tags, "family_"),
        "case_version": int(version) if version and version.isdigit() else None,
        "critical_pass": v.get("critical_pass"),
        "malformed": v.get("malformed"),
        "recall": v.get("recall"),
        "precision": v.get("precision"),
        "fp_count": v.get("fp_count"),
        "rubric_score": _rubric_score(eval_results),
        "grader_version": v.get("grader_version"),
    }


def aggregate_run(case_dims: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-case dims into the suite_executions run-level columns.

    Only cases with a non-null critical_pass count toward the rates (native
    runs contribute nothing). Returns None metrics when there is no signal.
    """
    graded = [c for c in case_dims if c.get("critical_pass") is not None]
    n = len(graded)
    if not n:
        return {
            "critical_pass_rate": None,
            "malformed_rate": None,
            "mean_rubric": None,
            "breakpoint_axis_level": None,
        }
    passed = sum(1 for c in graded if c["critical_pass"])
    malformed = sum(1 for c in graded if c.get("malformed"))
    rubrics = [c["rubric_score"] for c in graded if c.get("rubric_score") is not None]
    failed_levels = [
        c["axis_level"] for c in graded
        if not c["critical_pass"] and c.get("axis_level")
    ]
    breakpoint = (
        min(
            failed_levels,
            key=lambda a: _AXIS_ORDER.index(a) if a in _AXIS_ORDER else 99,
        )
        if failed_levels
        else None
    )
    return {
        "critical_pass_rate": round(passed / n, 6),
        "malformed_rate": round(malformed / n, 6),
        "mean_rubric": round(sum(rubrics) / len(rubrics), 6) if rubrics else None,
        "breakpoint_axis_level": breakpoint,
    }
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/dashboard/test_dimensions.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff format packages/atp-dashboard/atp/dashboard/dimensions.py tests/unit/dashboard/test_dimensions.py
uv run ruff check packages/atp-dashboard && uv run pyrefly check
git add packages/atp-dashboard/atp/dashboard/dimensions.py tests/unit/dashboard/test_dimensions.py
git commit -m "feat(dashboard): pure SP-1 dimension extractor + run aggregator"
```

---

## Task 4: Storage layer accepts the new columns

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/storage.py`
- Test: `tests/unit/dashboard/test_storage.py` (extend)

- [ ] **Step 1: Write the failing tests** — append to `tests/unit/dashboard/test_storage.py` (reuse its existing fixtures/imports; it already exercises `ResultStorage`):

```python
@pytest.mark.anyio
async def test_create_test_execution_sets_dimensions() -> None:
    storage = ResultStorage(_make_session())  # match existing helper in this file
    suite = await storage.create_suite_execution_by_name(
        suite_name="s", agent_name="claude_code"
    )
    te = await storage.create_test_execution(
        suite_execution=suite, test_id="case-1", test_name="f (moderate)",
        dimensions={
            "axis_level": "moderate", "capability": "safety_compliance",
            "family": "fam", "case_version": 2, "critical_pass": True,
            "malformed": False, "recall": 1.0, "precision": 1.0, "fp_count": 0,
            "rubric_score": 0.8, "grader_version": "findings_match@1",
        },
    )
    assert te.axis_level == "moderate"
    assert te.critical_pass is True
    assert te.case_version == 2
    assert te.grader_version == "findings_match@1"


@pytest.mark.anyio
async def test_suite_execution_gets_run_uuid_and_aggregates() -> None:
    storage = ResultStorage(_make_session())
    suite = await storage.create_suite_execution_by_name(
        suite_name="s", agent_name="claude_code"
    )
    assert suite.run_uuid  # a uuid string was assigned
    await storage.update_suite_execution(
        suite,
        aggregates={
            "critical_pass_rate": 0.5, "malformed_rate": 0.0,
            "mean_rubric": 0.7, "breakpoint_axis_level": "moderate",
        },
    )
    assert suite.critical_pass_rate == 0.5
    assert suite.breakpoint_axis_level == "moderate"
```

(If `test_storage.py` builds `ResultStorage` differently, match that pattern — do not invent a new harness.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/dashboard/test_storage.py -k "dimensions or run_uuid" -v`
Expected: FAIL (`create_test_execution` has no `dimensions` kwarg; no `run_uuid`).

- [ ] **Step 3: Edit `storage.py`**

(a) `create_test_execution` — add an optional `dimensions: dict[str, Any] | None = None` parameter and apply it when building the `TestExecution(...)` (spread the known keys, defaulting to None):

```python
    async def create_test_execution(
        self,
        suite_execution: SuiteExecution,
        test_id: str,
        test_name: str,
        tags: list[str] | None = None,
        started_at: datetime | None = None,
        total_runs: int = 1,
        dimensions: dict[str, Any] | None = None,
    ) -> TestExecution:
        d = dimensions or {}
        test_exec = TestExecution(
            suite_execution_id=suite_execution.id,
            test_id=test_id,
            test_name=test_name,
            tags=tags or [],
            started_at=started_at or datetime.now(tz=UTC),
            total_runs=total_runs,
            axis_level=d.get("axis_level"),
            capability=d.get("capability"),
            family=d.get("family"),
            case_version=d.get("case_version"),
            critical_pass=d.get("critical_pass"),
            malformed=d.get("malformed"),
            recall=d.get("recall"),
            precision=d.get("precision"),
            fp_count=d.get("fp_count"),
            rubric_score=d.get("rubric_score"),
            grader_version=d.get("grader_version"),
        )
        self.session.add(test_exec)
        await self.session.flush()
        return test_exec
```
(Preserve any extra fields the current constructor already sets; only ADD the dimension kwargs. Match the file's existing session/commit idiom — `flush` vs `commit`.)

(b) `create_suite_execution_by_name` — assign a `run_uuid` when constructing the `SuiteExecution`:
```python
        import uuid
        ...
        suite_exec = SuiteExecution(
            ...,  # existing fields
            run_uuid=str(uuid.uuid4()),
        )
```
(Put the `import uuid` at module top, not inline, per the file's style.)

(c) `update_suite_execution` — accept an optional `aggregates: dict[str, Any] | None = None` and set the four run-level columns when provided (alongside whatever it already updates):
```python
        if aggregates:
            suite_execution.critical_pass_rate = aggregates.get("critical_pass_rate")
            suite_execution.malformed_rate = aggregates.get("malformed_rate")
            suite_execution.mean_rubric = aggregates.get("mean_rubric")
            suite_execution.breakpoint_axis_level = aggregates.get(
                "breakpoint_axis_level"
            )
```
(If there is no `update_suite_execution` yet, add a minimal one that sets these + flushes; check the file first.)

Ensure `Any` is imported.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/dashboard/test_storage.py -q`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-dashboard && uv run pyrefly check
git add packages/atp-dashboard/atp/dashboard/storage.py tests/unit/dashboard/test_storage.py
git commit -m "feat(dashboard): storage persists eval dimensions + run_uuid + aggregates (SP-1)"
```

---

## Task 5: Wire the persist path in `_save_results_to_db`

**Files:**
- Modify: `atp/cli/main.py` (`_save_results_to_db`)
- Test: `tests/integration/cli/test_persist_dimensions.py`

- [ ] **Step 1: Write the failing integration test** (model it on `tests/integration/cli/test_persist_eval_pass.py` — reuse its DB-setup pattern: tmp `ATP_DATABASE_URL`, build a `SuiteResult` + `all_eval_results`, call `_save_results_to_db`, then query rows):

```python
# tests/integration/cli/test_persist_dimensions.py
"""SP-1: a method run persists eval dimensions on test/suite execution rows."""
# (Mirror the imports + DB fixture of test_persist_eval_pass.py.)

# Build a TestDefinition with method tags, a SuiteResult with one completed run,
# and all_eval_results mapping the test id -> [critical_check EvalResult(details=
# CaseVerdict dump), rubric EvalResult(score=0.8)]. Call _save_results_to_db with
# the same signature test_persist_eval_pass.py uses, then assert:
#   - the TestExecution row has axis_level/capability/critical_pass/malformed/
#     rubric_score/grader_version set from the verdict + tags
#   - the SuiteExecution row has run_uuid set and critical_pass_rate/
#     breakpoint_axis_level aggregated from the case(s)
```

Write the test concretely against the actual `_save_results_to_db` signature and the `test_persist_eval_pass.py` helpers (read both first). Assert at least: `te.axis_level == "moderate"`, `te.critical_pass is True/False as set`, `te.grader_version == "findings_match@1"`, `suite.run_uuid` truthy, `suite.critical_pass_rate is not None`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/cli/test_persist_dimensions.py -v`
Expected: FAIL (columns are None — not yet wired).

- [ ] **Step 3: Wire `_save_results_to_db`** in `atp/cli/main.py`

At the top of the function add the import:
```python
    from atp.dashboard.dimensions import aggregate_run, case_dimensions
```
In the per-test loop where `storage.create_test_execution(...)` is called, compute dims from the test's eval results and pass them; accumulate them for the run aggregate:
```python
        dims = case_dimensions(
            test_result.test, all_eval_results.get(test_result.test.id, [])
        )
        test_exec = await storage.create_test_execution(
            suite_execution=suite_exec,
            test_id=test_result.test.id,
            test_name=test_result.test.name,
            tags=test_result.test.tags if test_result.test.tags else None,
            total_runs=len(test_result.runs),
            dimensions=dims,
        )
        case_dims_all.append(dims)   # init `case_dims_all: list[dict] = []` before the loop
```
After the loop, before/with the suite-execution finalization, set the aggregates:
```python
        await storage.update_suite_execution(
            suite_exec, aggregates=aggregate_run(case_dims_all)
        )
```
(Use the existing `update_suite_execution`/finalization call if present — add the `aggregates=` kwarg rather than a second update. `run_uuid` is already set at creation in Task 4.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/integration/cli/test_persist_dimensions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check atp/cli/main.py && uv run pyrefly check
git add atp/cli/main.py tests/integration/cli/test_persist_dimensions.py
git commit -m "feat(cli): persist eval dimensions + run aggregates from atp test (SP-1)"
```

---

## Task 6: Full regression + quality gates

**Files:** none (verification only)

- [ ] **Step 1: Run the affected suites**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
uv run pytest tests/unit/dashboard tests/integration/cli -q
cd packages/atp-method && uv run pytest -q
```
Expected: all PASS (incl. the new dimension/storage/persist tests; existing persist tests unaffected since new columns are nullable).

- [ ] **Step 2: Backward-compat — a native (non-method) `atp test` run still persists**

Confirm an existing persistence test for a non-method suite still passes (its rows get NULL dimension columns, no error): `uv run pytest tests/integration/cli/test_persist_eval_pass.py -q`. Expected: PASS.

- [ ] **Step 3: Migration round-trips on a scratch DB**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
uv run python -c "
import asyncio
from atp.dashboard.database import init_database, get_database
import os, tempfile
os.environ['ATP_DATABASE_URL'] = 'sqlite+aiosqlite:///' + tempfile.mktemp(suffix='.db')
asyncio.run(init_database())
db = get_database()
import sqlalchemy as sa
async def check():
    async with db.session() as s:
        cols = (await s.execute(sa.text('PRAGMA table_info(test_executions)'))).fetchall()
        names = {c[1] for c in cols}
        assert {'axis_level','critical_pass','grader_version'} <= names, names
        cols = (await s.execute(sa.text('PRAGMA table_info(suite_executions)'))).fetchall()
        names = {c[1] for c in cols}
        assert {'run_uuid','critical_pass_rate','breakpoint_axis_level'} <= names, names
asyncio.run(check())
print('OK: columns present on a fresh DB')
"
```
Expected: `OK: columns present on a fresh DB` (exercises `init_database` + `_add_missing_columns`).

- [ ] **Step 4: Lint + types**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check atp/ packages/atp-dashboard packages/atp-method migrations
uv run pyrefly check
```
Expected: ruff clean; pyrefly 0 errors.

- [ ] **Step 5: Commit any formatting**

```bash
cd "$(git rev-parse --show-toplevel)"
git add -A && git commit -m "chore(sp-1): formatting" || echo "nothing to commit"
```

---

## Self-Review (completed during authoring)

- **Spec coverage (§5/§11, option A):** case-level dims + outcomes (Tasks 1,3,4,5), run-level aggregates + run_uuid (Tasks 1,3,4,5), `case_version` sourcing (Task 2), migration + `_add_missing_columns` safety (Tasks 1,6). `task_type` is added nullable (SP-4 fills it). Deferred columns (`language`/`run_mode`/`envelope_pin`) explicitly out per the scope guard.
- **Type/name consistency:** `case_dimensions(test, eval_results) -> dict` keys match the `TestExecution` columns and the `create_test_execution(dimensions=...)` spread; `aggregate_run(case_dims) -> dict` keys match `update_suite_execution(aggregates=...)` and the `SuiteExecution` columns; migration column names == ORM column names == test assertions.
- **Backward compatibility:** every new column is nullable; native runs and pre-migration prod SQLite are covered (`_add_missing_columns` + nullable). The `rubric_score` comes from the `rubric` check (not the findings verdict, which is 0.0 there) — called out so it isn't mis-sourced.
- **Placeholders:** Task 5's test body is described against the real `_save_results_to_db`/`test_persist_eval_pass.py` rather than pasted, because it must match that file's exact fixture/signature — the implementer reads both first; everything else carries full code.
- **Prod risk:** merge auto-deploys; the migration only ADDs nullable columns (no backfill, no NOT NULL), so it is safe and reversible on the live SQLite.
