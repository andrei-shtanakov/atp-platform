# Eval-results architecture — unified store, history & cross-agent comparison

> Date: 2026-06-14. Status: design (approved in brainstorm). Author: andrei + Claude.
> Context: R-07 code-review vertical shipped (PRs #171–#174); more verticals are
> queued (architecture-creation, repo-analysis, coding). The platform's original
> purpose — run tests AND view/compare/history of results — is fragmented.

## TL;DR

The platform stores eval results in three disjoint places with uneven UI, and
the newest run path (`run_pipe_check.py`) is a glue script that bypasses all of
them. As more verticals land, every one risks becoming its own pipeline + its own
result format → no history, no comparison. This design establishes **one run
path and one canonical, dimensioned result store**, with verticals reduced to
**data (cases) + a pluggable grader**. It describes the full target architecture
and a roadmap (SP-1…SP-6), and details the first slice (**SP-1: canonical store +
runner persistence**).

## 1. Problem

Results live in three stores:

1. **`SuiteExecution`/`TestExecution`** (`packages/atp-dashboard/atp/dashboard/models.py`)
   — written by `atp test` (default; `--no-save` disables), including atp-method
   runs. Reachable only via JSON API (timeline/comparison/analytics); **no HTML
   page** renders it (`/ui/executions` is an open TODO).
2. **benchmark `Run`** (table `benchmark_runs`, `benchmark/models.py`) — the
   pull-model benchmark API/SDK for external participants. **Has UI** (leaderboard).
3. **arbiter `benchmark_runs`** (cross-project) — our `report_benchmark-v1`
   payload destined for routing; not surfaced in ATP at all.

Consequences:
- The core use case ("run my agent, see history, compare agents") is unserved for
  local/method runs — data accrues in `SuiteExecution`, but there is nowhere to
  look.
- The R-07 pipe-check (`method/run_pipe_check.py`) drives `TestOrchestrator`
  directly, so it persists **nothing** to either dashboard store; it writes JSON +
  a local sqlite mirror of arbiter only.
- The `report_benchmark` reporter is half-integrated: registered by name, but
  `BenchmarkReporter.report(SuiteReport)` raises `NotImplementedError`; the payload
  is built by a module-level function called from the script.

## 2. Goals / non-goals

**Goals**
- One canonical, **dimensioned** result store for all local/method/campaign runs.
- History (per agent × vertical over time) and comparison (across agents at a
  fixed slice) as first-class, query-driven views.
- Adding a vertical = new cases + a grader plugin, not a new pipeline.
- Keep the cross-project `report_benchmark-v1` export to arbiter working (as a
  sink, not a separate pipeline).

**Non-goals**
- Merging the public pull-model benchmark `Run` product into this store (kept
  separate; it may share the dimension vocabulary later).
- Building the `workspace` (file-writing) run mode now — designed for, deferred to
  the first file-writing vertical (SP-6).
- Re-running historical data through new graders (versioning makes old rows
  legible as-is; no backfill of scores).

## 3. Decisions (locked in brainstorm)

- **Storage = evolve `SuiteExecution`/`TestExecution`** (not a new parallel model,
  not merging benchmark `Run`). They already capture the right shape and are
  already written by `atp test`/method.
- **Vertical extension = shared spine + grader registry + two run modes**
  (`text_out` now, `workspace` later). One run path; only the grader and agent mode
  vary.
- **Dashboard = leaderboard (A) + history/trend (B) first**; matrix (C) and
  per-case drill-down (D) follow.
- **Metrics live in BOTH columns and `score_components` JSON** — columns for
  sort/group/filter, JSON for extensibility.
- **Dimensions `axis_level`/`capability` become columns** (not parsed from tags).
- **Agent comparison key = `agent_name`** (string, e.g. `claude_code`); no FK to
  `agents` required.
- **First detailed slice = SP-1** (canonical store + persistence only; UI and
  campaign follow).

## 4. Target architecture

```
vertical = data (family + cases; fields task_type / language / axis_level)
           + grader (plugin from the GraderType registry)
                         │
                         ▼
RUN PATH (single):  loader → orchestrator → adapter[text_out | workspace]
                          → grader[GraderType registry]
                         │
                         ▼
ONE result store (dimensioned):
   EvalRun (= suite_executions)  1 ──< EvalCaseResult (= test_executions)
   dimensions as columns: agent_name · task_type · axis_level · language ·
   capability · critical_pass · malformed · rubric · tokens · cost · ts ·
   case_version · grader_version · model_pin
                         │
              ┌──────────┴───────────┐
              ▼                      ▼
   DASHBOARD (over columns)     EXPORT (sink)
   A leaderboard · B trend      report_benchmark-v1 → arbiter (routing)
   (C matrix · D drill-down later)
```

Principles:
- **One path, one store** — results share a single shape, so history and
  comparison are queries, not bespoke code.
- **Verticals are data**, not pipelines.
- **Dimensions in columns**, not buried in JSON — else you cannot filter/group.
- **Versions on every row** — else "history" compares different tests.
- **arbiter is an export sink**, not another store.

## 5. Data model (evolution, not new tables)

### 5.1 `EvalRun` (= `suite_executions`, run-level)

Already present: `id, tenant_id, suite_name, agent_id (FK, nullable), agent_name,
adapter, model, started_at, completed_at, duration_seconds, runs_per_test,
total_tests, passed_tests, failed_tests, success_rate, status, error`; children
`test_executions`.

Add:
- **dimensions:** `benchmark_id` (task_type key, e.g. `"code-review"`), `family`,
  `language` (nullable / `n/a` until a vertical sets it), `run_mode`
  (`text_out`|`workspace`), `run_uuid` (stable id for idempotent export).
- **canonical metrics (columns):** `critical_pass_rate`, `malformed_rate`,
  `mean_rubric`, `breakpoint_axis_level` (nullable string) + `score_components`
  (JSON, for any extra numeric components).
- **provenance/versioning:** `family_version`, `grader_version`, `model_pin`,
  `envelope_pin`.

### 5.2 `EvalCaseResult` (= `test_executions`, case-level)

Already present: `test_id, test_name, tags (JSON), started/completed,
duration_seconds, total_runs, successful_runs, success, score, status, error,
statistics (JSON)`; children `run_results`, `evaluation_results`,
`score_components`.

Add:
- **dimensions:** `axis_level`, `capability`, `language`, `critical_pass` (bool),
  `malformed` (bool), `recall`, `precision`, `fp_count`, `rubric_score`.
- **versioning:** `case_version`, `grader_type`, `grader_version`.

### 5.3 Sourcing the new fields

- `critical_pass`/`malformed`/`recall`/`precision`/`fp_count` come from the
  findings_match check's `details` (= `MatchResult.model_dump()` — already carries
  `critical_pass`, `malformed`, `recall`, `precision`, `false_positives`). The
  runner already evaluates assertions post-run (`atp/cli/main.py` ~778–870); we map
  the resulting `EvalCheck.details` into columns at persist time.
- `axis_level`/`capability`/`family` are derivable from atp-method tags
  (`level_*`, `capability_*`, `family_*` from `atp_method.loader._tags`) but are
  lifted to columns for query-ability.
- `case_version` from the case YAML `version`; `grader_type`/`grader_version` from
  the grader (see §6). `model_pin`/`envelope_pin` from run config / shim provenance.

### 5.4 Backward compatibility

All new columns are nullable / defaulted so legacy rows and non-method suites
remain valid. Follow the existing pattern: a real **Alembic migration** owns the
schema change (the codebase already uses Alembic, e.g. `a7b8c9d0e1f2`), plus the
`_add_missing_columns` safety net used by `SuiteExecution` for pre-migration
SQLite DBs. The dashboard reporter/aggregator defaults `malformed`/metrics for
rows that predate the columns.

## 6. Vertical & grader extension

**GraderType registry.** Generalize the existing assertion→evaluator dispatch
(`findings_match` already exists). A grader is a plugin with a uniform contract:

```
grade(case, agent_output | workspace) -> CaseVerdict {
    critical_pass: bool, malformed: bool,
    recall, precision, fp_count, rubric_score,
    details: dict, grader_version: str
}
```

A case names its `grader_type`. Roster: `findings_match` (review, exists) →
`code_exec` (coding: run tests), `structure_match` (architecture: tree/constraint
checks), `retrieval_match` (repo-analysis: answer/evidence match). The spine
(schema, run path, store) is shared; only the grader differs.

**Two agent run modes** (adapter-level):
- `text_out` — current shim (stdin ATPRequest → stdout ATPResponse). Sufficient
  for review/repo-analysis/docs.
- `workspace` — spawner gives the agent a working dir + file-write/(optional) exec;
  the grader reads results from the workspace. Needed for coding/architecture.
  **Designed now, built in SP-6** (YAGNI).

## 7. Run-path unification

Promote "run family × N agents → persist `EvalRun`/`EvalCaseResult` → emit
`report_benchmark`" to a first-class runner capability (e.g. `atp benchmark
<family> --agents a,b,c`, or multi-agent `atp test` with a benchmark sink). Then:
- every vertical persists to the canonical store automatically and is visible on
  the dashboard;
- `report_benchmark` becomes a **reporter/sink built from the store** (fixes the
  current `NotImplementedError`);
- `method/run_pipe_check.py` dissolves into the runner; multi-agent + agent_id
  tagging are built in, not hand-rolled.

## 8. Dashboard views

Over the new columns:
- **A · Leaderboard / vertical** (first) — agents ranked by `critical_pass_rate`
  (+ `malformed_rate`, `breakpoint_axis_level`, cost) at a chosen time.
- **B · History / trend** (first) — `(agent, benchmark_id)` over `ts`; reuse the
  existing OLS slope in `atp/analytics/trend.py`.
- **C · Matrix agent × axis_level** (later) — breakpoint heatmap.
- **D · Run drill-down** (later) — per-case `critical_pass`/recall/FP/malformed.

A+B directly answer "compare agents" and "history"; C+D are supporting and ship in
SP-5. This also subsumes the open `/ui/executions` TODO.

## 9. Versioning

Each result row records `case_version` + `grader_version` + `model_pin`/
`envelope_pin`. Rule: comparisons are within a frozen version
(`suite_type: probe → regression` once frozen); the dashboard renders a version
change as an explicit boundary on trend charts rather than silently comparing
different tests. This is what makes "history" trustworthy.

## 10. Decomposition (roadmap)

| SP | Scope | Depends |
|---|---|---|
| **SP-1** | **Canonical store + runner persistence** — evolve `suite_executions`/`test_executions` (dimension + version columns), Alembic migration + `_add_missing_columns`, runner populates them for ALL runs (CLI + method), findings_match `details`→columns mapping | — |
| **SP-2** | Campaign run-path + `report_benchmark` sink — `atp benchmark family --agents …`, dissolve `run_pipe_check.py`, reporter writes from the store | SP-1 |
| **SP-3** | Dashboard A+B — leaderboard + history/trend over the new columns | SP-1 |
| **SP-4** | Taxonomy in schema — `task_type`/`language`/`benchmark_id` in `agent-eval-case` + GraderType registry generalization | SP-1 |
| **SP-5** | Dashboard C+D — matrix + drill-down | SP-3 |
| **SP-6** | `workspace` run mode + first file-writing grader (`code_exec`) | SP-4 |

Order: **SP-1 → (SP-2 ∥ SP-3 ∥ SP-4) → SP-5 → SP-6**.

## 11. First slice — SP-1 (detailed)

**Goal:** the canonical dimensioned store exists and is populated by every run,
so results become query-friendly (incl. atp-method) — the foundation for SP-2/3/4.

**In scope**
1. Add the columns in §5.1–§5.2 to `SuiteExecution`/`TestExecution` (nullable/
   defaulted).
2. Alembic migration for the new columns + keep `_add_missing_columns` parity for
   SQLite.
3. Persist path (`atp/cli/main.py` `_save_results_to_db` / `dashboard/storage.py`):
   populate the new run-level + case-level columns from the run + evaluated
   assertions; map findings_match `EvalCheck.details` (`critical_pass`,
   `malformed`, `recall`, `precision`, `false_positives`→`fp_count`) into columns;
   lift `axis_level`/`capability`/`family` from tags; record versions.
4. Aggregate run-level `critical_pass_rate`/`malformed_rate`/`mean_rubric`/
   `breakpoint_axis_level` from the case rows at persist time.
5. Tests: a method run (`atp test method/cases/code-review`) persists an `EvalRun`
   with the new columns + per-case `EvalCaseResult` rows carrying `critical_pass`/
   `malformed`/`axis_level`; legacy/non-method suites still persist (columns
   default); migration up/down.

**Out of scope (SP-1)**
- Any UI (SP-3), campaign multi-agent run path & `report_benchmark` sink (SP-2),
  schema `task_type`/`language` authoring fields (SP-4), `workspace` mode (SP-6).
- `run_pipe_check.py` stays as-is until SP-2 dissolves it (no regression).

**Risk / watch-outs**
- Column sprawl on a shared table: keep additions to the dimensions named here;
  anything experimental goes in `score_components`/`statistics` JSON first.
- Don't break the benchmark `Run` product (different table) — SP-1 touches only
  `suite_executions`/`test_executions`.
- Migration must be reversible and safe on the production SQLite DB (prod
  auto-deploys on merge to main).

## 12. Cross-project note

`report_benchmark-v1` export to arbiter is unchanged in contract; in SP-2 it
becomes a sink that reads the canonical store instead of being built ad-hoc in a
script. arbiter remains the routing consumer; the ATP dashboard becomes the
human-facing history/comparison surface.
