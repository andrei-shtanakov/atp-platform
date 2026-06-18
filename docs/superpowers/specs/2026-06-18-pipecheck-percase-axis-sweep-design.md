# Pipe-check per-case / axis-sweep dashboard view — design

**Date:** 2026-06-18
**Status:** approved (brainstorm), pending implementation plan
**Branch:** `r07/pipecheck-dashboard-import`
**Context:** R-07 Phase 1, follow-up to the `report_benchmark` → dashboard bridge
(`method/import_pipecheck_to_dashboard.py`). Parent EPIC in `TODO.md`:
"унификация просмотра/сравнения/истории результатов".

## Problem

The pipe-check sweep (`method/run_pipe_check.py`) produces, per agent:

- `report_benchmark_<agent>.json` — `report_benchmark-v1` payload (run-level
  aggregates + thin `per_task`), consumed by arbiter. **All 81 runs have this.**
- `case_details_<agent>.jsonl` — full per-case grading
  (`axis_level`, `critical_pass`, `malformed`, `recall`, `precision`,
  `fp_count`, tokens, cost, duration). **Only 25/81 runs have this** (older runs
  predate `_write_case_details`).

The dashboard importer currently writes **one parent `SuiteExecution` per
report** (run-level aggregates only). The `/ui/eval-leaderboard` and
`/ui/eval-trends` views therefore collapse each agent to a single number per
run. The multidimensional structure that matters — *where on the severity axis
an agent breaks*, and *per-case recall/false-positives* — is computed and sits
on disk in `case_details_*.jsonl`, but never reaches the dashboard.

## Goal

A dashboard surface for **visual analysis and management demonstration** of
code-review / req-extraction eval results, showing the axis-level sweep and
per-case precision/recall — not just a run-level aggregate.

This is **dashboard-only**. We are *not* enriching the `report_benchmark-v1`
contract or coordinating with arbiter (arbiter routing is a separate consumer
and is blocked on Maestro R-03 anyway).

## Non-goals

- No change to `report_benchmark-v1` schema or `build_report_benchmark_payload`.
- No `language` axis yet — it is absent from the payload **and** is `None` in
  current `case_details` (0/81). Surfaced only if/when the harness emits it.
- No leaderboard aggregation-semantics change (latest-vs-mean) — tracked
  separately in `TODO.md`; out of scope here.

## Key enabling facts (verified 2026-06-18)

- `TestExecution` (child of `SuiteExecution`, `models.py:218`) **already has**
  every column we need: `axis_level`, `critical_pass`, `malformed`, `recall`,
  `precision`, `fp_count`, `task_type`, `rubric_score`, plus `language`.
  **No migration required** (`init_database._add_missing_columns` backstops a
  lagging deployed DB).
- `run_pipe_check.py:403` calls `_write_case_details(...)` **unconditionally**
  per agent — so any future sweep with the current script writes `case_details`
  for **all** agents. The 56 missing files are historical, not a code gap.

## Design

### 1. Importer enrichment (`method/import_pipecheck_to_dashboard.py`)

For each `report_benchmark_<agent>.json`, look for a sibling
`case_details_<agent>.jsonl`:

- **Present** → after creating the parent `SuiteExecution`, write one
  `TestExecution` child row per case, mapping
  `case_id→test_id/test_name`, `axis_level`, `critical_pass`, `malformed`,
  `recall`, `precision`, `fp_count`, `rubric_score`, `tokens`, `duration`,
  `task_type` (parent's), `success=critical_pass`, `status` from `error_class`.
- **Absent** (legacy 56 runs) → write only the parent, exactly as today. The
  per-case view is empty for those agents; the leaderboard/breakpoint still work.

Sibling lookup is derived from the report filename
(`report_benchmark_<agent>.json` → `case_details_<agent>.jsonl` in the same
directory). Idempotency is unchanged: parent skipped by `run_uuid`; child rows
are written only when the parent is newly created (never duplicated).

### 2. Data replacement strategy: keep-latest + opt-in purge

- **Default (no flag):** import is purely additive. New runs get new
  `run_uuid`s; the leaderboard's "latest completed run per agent"
  (`suite_leaderboard`) means a fresh sweep automatically becomes the displayed
  result, and old/incomplete runs recede into trend history. No data loss.
- **`--replace` flag:** before importing, delete existing
  `adapter='pipe-check'` rows (and their `TestExecution` children via cascade)
  for the targeted suite(s), giving a clean slate for a demo. Explicit,
  opt-in, logged (row counts printed).

Batch/sweep tagging (filter views by sweep id) is deliberately deferred — it is
only worth it once multiple coexisting sweeps must be compared side by side.

### 3. New dashboard view: per-(suite, agent) drill-down

Route `GET /ui/eval-run/{suite_name}/{agent_name}` (HTMX + Pico, same pattern as
existing eval views), reachable via a link from each leaderboard row:

- **Axis sweep:** `critical_pass_rate` per `axis_level`, ordered
  `clean → mild → moderate → severe → very_severe`, as a small chart + table.
  Computed from the agent's latest run's `TestExecution` rows grouped by
  `axis_level`. Levels with no cases render as gaps (honest about coverage).
- **Per-case table:** one row per case — `case_id`, `axis_level`,
  `critical_pass`, `malformed`, `recall`, `precision`, `fp_count`, tokens,
  duration.
- If the agent's latest run has no `TestExecution` rows (legacy/aggregate-only),
  show a clear "no per-case detail for this run" notice rather than an empty
  page.

A new storage method (e.g. `suite_agent_case_detail(suite, agent)`) returns the
latest run's child rows + the per-axis aggregation; the route stays thin.

### 4. Coverage timeline (data, not code)

- **Now:** per-case detail exists for `claude_code`, `codex_cli`, `deepseek`
  only; axis grid skewed to `severe`/`moderate`.
- **This weekend (planned paid run):** re-run `run_pipe_check.py` over **all**
  required agents — current script writes `case_details` for every agent, so the
  gap closes. New models (local + API) will be added to the roster.
- **Demo (~1 week out):** lands *after* the weekend run, so full coverage is
  available; the view is built for the full shape regardless of today's gaps.

## Components & boundaries

| Unit | Responsibility | Depends on |
|------|----------------|------------|
| `parse_case_details(path)` | parse one `case_details_*.jsonl` → list of typed per-case dicts | stdlib json |
| importer child-write path | write `TestExecution` rows under a new `SuiteExecution` | `ResultStorage`, models |
| `--replace` purge | delete prior `pipe-check` rows for targeted suites | `ResultStorage`/session |
| `suite_agent_case_detail` (storage) | latest run's child rows + per-axis aggregate | `SuiteExecution`/`TestExecution` |
| `/ui/eval-run/{suite}/{agent}` route + template | render sweep chart + per-case table | storage method |

## Testing

- **Unit:** `parse_case_details` (well-formed, missing file, malformed line);
  importer writes N child rows when sibling present, 0 when absent; child rows
  carry correct `axis_level`/`recall`/`precision`/`fp_count`; `--replace` purges
  prior pipe-check rows then re-imports.
- **Storage:** `suite_agent_case_detail` returns latest run's rows grouped by
  axis in canonical order; empty when no child rows.
- **Route:** renders for an agent with detail; shows the "no per-case detail"
  notice for an aggregate-only agent.
- Mac/3.12: `pytest`, `pyrefly check`, `ruff check` green before done.

## Risks / open items

- **Depends on the weekend paid run** for full agent coverage. Until then the
  view legitimately shows only 3 agents — acceptable, surfaced in the UI.
- **New models (local + API)** will join the roster in that run; the view is
  roster-agnostic (driven by whatever agents have rows), so no code change
  needed to absorb them — but the axis grid should be balanced across levels in
  the new sweep for a clean curve (a `run_pipe_check.py` case-set concern, noted
  for the run, not this work).
- Re-import after the weekend run uses `--replace` (or relies on keep-latest) to
  supersede today's partial data.
