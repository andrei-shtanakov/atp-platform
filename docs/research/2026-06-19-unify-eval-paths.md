# Research: unify the two agent-evaluation paths (harness vs `atp test`)

**Status:** open research item (not scheduled)
**Created:** 2026-06-19
**Owner question:** is full convergence worth it, or is the current split correct?

## Context

Two paths evaluate methodology agents over the same `method/cases/`, sharing the
same schema/loader/evaluator (`packages/atp-method/`):

1. **`atp test method/cases/X`** (plugin path) — one agent × one adapter, standard
   reporters, and a built-in save step (`atp/cli/main.py:930+`) that writes
   `SuiteExecution` + per-case `TestExecution(dimensions=case_dimensions(...))`
   into the dashboard store. Lands in `/ui/eval-*` natively.
2. **`method/run_pipe_check.py`** (harness) — a matrix of N spawner shims with
   preflight, emitting `report_benchmark-v1` (for arbiter) + `case_details_*.jsonl`
   + sqlite. No dashboard dependency. Did not reach the dashboard → the bridge
   `method/import_pipecheck_to_dashboard.py` was built to ingest its JSON.

As of 2026-06-19 the ergonomic seam is closed: `run_pipe_check.py --to-dashboard`
runs the bridge automatically after a sweep (PR for `r07/pipecheck-to-dashboard-flag`).

## The north-star idea (full merge)

Finish `BenchmarkReporter` (today `raise NotImplementedError`; the
`SuiteReport → report_benchmark` mapping is marked "Phase-1b" in
`atp/reporters/benchmark_reporter.py`). Then:

- `atp test method/cases/X --adapter cli --model <agent> -o report_benchmark`
  produces the arbiter payload natively, AND writes `SuiteExecution` via the
  existing save step.
- `run_pipe_check.py` collapses to a thin loop over `atp test` per agent
  (preserving the shim matrix + preflight), instead of its own `TestOrchestrator`
  + `_grade_case` + `build_report_benchmark_payload`.
- The bridge is then needed only to backfill historical JSON, not for new runs.
- One evaluation code path; aggregation logic lives in one place.

## Why it is NOT obviously worth doing

- **Decoupling is a feature, not debt.** The harness has no DB dependency and runs
  in the cowork 3.10 sandbox where `atp.dashboard` will not import. Full merge
  couples a clean, portable producer (arbiter artifacts) to a heavy optional
  consumer (dashboard).
- **Cost is real:** finishing `BenchmarkReporter` requires the per-case
  `axis_level`/`critical_pass`/`recall`/`precision`/`fp_count` to survive into the
  reporter in the `report_benchmark` shape — depends on the payload-enrichment
  work already spec'd, and on reconciling the harness's grading output with the
  CLI's `scored_results`/`case_dimensions` path.

## Concrete questions to answer before committing

1. Does the harness's per-agent matrix + preflight map cleanly onto repeated
   `atp test` invocations without losing the cross-agent spread summary?
2. Can `BenchmarkReporter` produce a byte-identical `report_benchmark-v1` payload
   from a `SuiteReport`, including `breakpoint_axis_level`, given the enriched
   per-case data?
3. Is the duplicated aggregation (`benchmark_reporter._breakpoint` vs
   `dashboard/dimensions.py`) worth collapsing into one shared helper regardless
   of whether the full merge happens? (Likely yes — small, independent win.)
4. What is the sandbox story if the harness gains a dashboard code path — stays
   optional behind a flag, or split into two entrypoints?

## Disposition

Deferred. Revisit under the "unify result views" EPIC, after `report_benchmark`
payload enrichment. The DRY cleanup (Q3) can be done independently and earlier.
