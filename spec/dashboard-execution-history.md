# Spec: Dashboard UI for CLI Run History (`/ui/executions`)

**Status:** planned
**Created:** 2026-06-05
**Owner:** dashboard
**Related fix:** `SuiteExecutionSummary.agent_id` made `int | None` (CLI path stores
`agent_id = NULL`, denormalizes `agent_name` — see LABS-54). Without that fix
`GET /api/suites/` returns HTTP 500.

## Problem

`atp test` persists run history into `suite_executions` / `test_executions` /
`run_results` (the **SuiteExecution** model). This history is reachable **only via the
JSON API** (`/api/suites/`, `/api/tests/`). There is **no HTML page** that renders it.

The entire `/ui/*` surface (`/ui/runs`, `/ui/analytics`, home) is wired to the separate
**benchmark `Run`** model (`benchmark_runs` table), which `atp test` does not populate.
`/ui/suites` shows **SuiteDefinition** (uploaded definitions), not executions. So in a
browser the CLI run history is invisible.

## Goal

A new web page `/ui/executions` that renders CLI run history (SuiteExecution), with:

1. **List** — table of executions (suite / agent / runs / pass-fail / success_rate /
   status / time).
2. **Detail** — per-execution: tests, individual runs (statuses), per-test statistics
   for `runs > 1`, and an aggregated **failure-cause breakdown**.

Decision (2026-06-05): **new dedicated page** (not reuse `/ui/runs`, which is the
benchmark surface) + **full scope including failure-cause breakdown**.

## Reuse (already in codebase)

- **Storage:** `list_suite_executions`, `get_suite_execution`, `get_test_execution`,
  `get_test_history` (`packages/atp-dashboard/atp/dashboard/storage.py`).
- **Schemas:** `SuiteExecutionList/Detail`, `TestExecutionDetail` (has `statistics`),
  `RunResultSummary` (`schemas.py`).
- **Statistics:** `StatisticsCalculator` (mean/std/CI/CV/stability) in `atp-core`.
- **Page pattern:** `ui_runs` / `ui_run_detail` + HTMX partials (`v2/routes/ui.py`).

## New logic (the only non-wiring part)

Failure-cause aggregation does not exist yet (run-level `response_status` + `error`
string are present, but never aggregated into a cause histogram).

**New file** `packages/atp-dashboard/atp/dashboard/failure_analysis.py`:

- `compute_failure_breakdown(run_results) -> FailureBreakdown`
- `Counter` over `response_status` (`completed` / `failed` / `timeout` / ...) +
  clustering of `error` strings (normalize: strip numbers/paths, group by prefix) →
  top-N causes with counts.
- Pure, synchronous, table-testable.

## Files & steps

| # | File | Change |
|---|------|--------|
| 1 | `schemas.py` | + `FailureCause`, `FailureBreakdown`; extend `SuiteExecutionDetail` with `failure_breakdown` (+ optional `run_statistics`) |
| 2 | `failure_analysis.py` *(new)* | `compute_failure_breakdown()` + error-string normalizer |
| 3 | `v2/routes/ui.py` | `ui_executions` (list, mirrors `ui_runs`) + `ui_execution_detail` (detail: tests, runs, stats, breakdown; HTMX partials) |
| 4 | `templates/ui/executions.html` *(new)* | List table |
| 5 | `templates/ui/execution_detail.html` *(new)* | Summary header → tests → per-test runs / stats / failure breakdown |
| 6 | `templates/ui/partials/execution_tests.html` *(new)* | HTMX partial (mirrors `run_tasks.html`) |
| 7 | `templates/ui/base_ui.html` | + nav item `Test Runs` → `/ui/executions`, `active_page == 'executions'` (near line 24) |

## Build sequence (iterative, verify each)

1. **Aggregation + schemas** (#1, #2) → unit test `compute_failure_breakdown`
   (timeout / failed / mixed cases). Run, confirm.
2. **List route** (#3 part 1) + `executions.html` (#4) + nav (#7) → open
   `/ui/executions`, see 15 rows.
3. **Detail route** (#3 part 2) + `execution_detail.html` (#5) + partial (#6) → open
   `/ui/executions/10`, see the failed execution with causes.
4. **Statistics block** for `runs > 1` tests (reuse `StatisticsCalculator`) in detail.

## Tests

- `tests/unit/dashboard/test_failure_analysis.py` *(new)* — aggregation (primary new logic).
- `tests/unit/dashboard/test_schemas.py` — new schemas (validation, optional fields).
- `tests/integration/dashboard/test_v2_routes.py` — `GET /ui/executions` (200, renders
  rows), `GET /ui/executions/{id}` (200 + 404), HTMX partial.

## Verification

`ruff format` → `ruff check` → `pyrefly check` → `pytest` (new + touched) → manual:
restart dashboard, `curl` / browser on `/ui/executions` and `/ui/executions/10`.

## Estimate

~7 files (3 new templates, 1 new module, 1 test file, 2 edits), ~250–350 lines.
Only "smart" logic is failure aggregation; the rest mirrors `ui_runs`. ~half a day with tests.

## Risks

- **Error-string clustering** — the only heuristic. Start simple (status + first N words);
  do not over-engineer.
- **Empty `agent_id`** — handled by the fix; template shows `agent_name`, not `agent_id`.
- **HTMX partials** — optional for MVP; list+detail can render plainly first.
- Do not touch benchmark `Run` / `/ui/runs` — keep changes isolated.
