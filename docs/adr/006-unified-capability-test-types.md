# ADR-006: Unified Capability Test-Types via agent-eval-case

**Status**: Accepted (2026-06-14)
**Date**: 2026-06-14
**Companion**: `docs/superpowers/specs/2026-06-14-eval-results-architecture-design.md`
— the two are halves of one architecture. This ADR owns the **case/grader/harness
spine** (how a test-type is added); the companion spec owns the **result store,
history and comparison** (where results live and how they are viewed). Read §"Result
store vs export sink" below for how they meet.
**Context**: R-07 ships the first capability eval (`code-review`) on top of the
`agent-eval-case` methodology (`method/agent-eval-case.schema.json`,
`AgentEvalCaseEvaluator`, `report_benchmark-v1` contract). More capability tests
are planned — architecture-design, repo-analysis, coding, documentation
(see the Phase-1 MVP note `docs/2026-06-13-r07-phase1-code-review-mvp.md` §1,
which already maps each onto an arbiter `TaskType`). The concern raised: each new
test-type risks becoming its own bespoke thing — its own grader, its own runner
script, its own envelope, consumed by the dashboard in its own way — i.e. a
"zoo". This ADR decides how a new test-type is added so that does not happen.

The premise is only half-right. A strong unifying spine already exists and
`code-review` sits *inside* it, not beside it:

- **One case schema for every test-type** — `agent-eval-case.schema.json`. A
  test-type is a `family` + `capability` + `construction_axis`, not a new format.
  Axes are orthogonal (what you score × what you vary × how hard).
- **One evaluator** — `AgentEvalCaseEvaluator` (critical_check gate + non-gating
  rubric).
- **One `task_type` dimension** keys every result by test-type on the internal
  store. (Correction — see "Result store vs export sink": the dashboard reads the
  **internal canonical store**, scoped by `task_type`; `report_benchmark-v1` is an
  **export sink to arbiter only**, NOT the dashboard's source, and `benchmark_id`
  lives only on that export — mapped from `task_type` at the sink. An earlier
  draft wrongly used `benchmark_id` as the internal dimension and routed the
  dashboard through the export.)
- **`method/run_pipe_check.py` is a generic harness**, not a per-test script —
  it drives a family through the ATP CLI adapter + spawner shims + benchmark
  reporter.

So the question is not "do we need a framework" — we have one. The question is
how to keep new test-types *additive over data* instead of *forking code*. Three
seams are already starting to leak:

1. **Grader types grow per-capability.** `grader.type` is a closed enum. The
   canonical JSON schema (`agent-eval-case.schema.json`) lists
   `exact|regex|programmatic|rubric|model_graded|human` — it does **not** contain
   `findings_match`. The pydantic mirror (`packages/atp-method/atp_method/schema.py:46`)
   **does**. Contract and code have already diverged on the first new capability.
   Left unchecked, capability *N* adds grader.type *N* plus a branch in
   `_evaluate_critical` — the dispatch grows without bound.
2. **The output envelope is welded into the spawner.** `REVIEW_ENVELOPE` lives
   inside `method/spawners/claude_code_shim.py`. With *N* capabilities × *M*
   spawners that is *N×M* envelope copies that drift (and it breaks the
   API-vs-CLI ablation's equivalence — see the R-07 proposals, file 10, P2).
3. **The harness invites forking.** `run_pipe_check.py` hard-codes
   `BENCHMARK_ID = "code-review"` and is titled a "code-review pipe-check
   harness". If architecture/repo-analysis each get a copy, the standalone-script
   zoo is born.

Plus the dashboard: `task_type` is the unifying key, but if leaderboard/runs
views are not strictly scoped by it, results from different test-types pool and
get read inconsistently.

## Decision

**Adopt one invariant: a new test-type is new data — a `family`, its `cases`,
and (at most) one registered deterministic `checker` — never a new code path.**
Concretely, adding architecture-design or repo-analysis MUST NOT introduce a new
`grader.type` enum value, a new capability-specific spawner, a new harness
script, or a new dashboard page.

To make that invariant enforceable, four directions (principle-level here;
implementation is a follow-up plan, not part of this ADR):

1. **Graders: a named checker registry under one `programmatic` type, not a
   growing enum.** Keep grader *strategies* a small closed vocabulary
   (`exact|regex|programmatic|rubric|model_graded|human`). A deterministic
   capability check is selected by name (e.g. `grader.checker: findings_match`)
   from a registry of pure functions — reusing ATP's existing evaluator plugin
   registry. `findings_match` becomes the first registered checker rather than a
   new enum member; the schema/pydantic divergence is closed by folding it back
   under `programmatic + checker`. A new capability registers a checker; the core
   dispatch stays closed for modification.
2. **Decouple the output contract from transport.** The findings/output envelope
   belongs to the `family`/`capability`, not the spawner. Spawners become
   capability-agnostic transports that relay a contract handed to them. This
   removes the *N×M* drift and makes the API-vs-CLI ablation equivalent by
   construction.
3. **One generic run path, parameterized.** Promote the harness to a single
   entry parameterized by `--family` / `--task-type` (folding toward the
   `atp` CLI), with `task_type ↔ benchmark_id ↔ TaskType` as the one taxonomy
   registry (the store/CLI speak `task_type`; `benchmark_id` is derived for the
   arbiter export). No per-capability script.
4. **One parameterized, task_type-scoped dashboard view, reading the internal
   store.** Leaderboard and runs render per `task_type` from the **canonical
   internal store** (`EvalRun`/`EvalCaseResult` = evolved
   `suite_executions`/`test_executions`; companion spec §5), NOT from the
   `report_benchmark` export. The export stays a one-way sink to arbiter. (The
   arbiter reader fix in the MVP note §4 — scores `task_type`-scoped — is the
   *export/routing* side of the same `benchmark_id` dimension.)

### Rejected alternatives

- **"It's fine, just keep adding grader.type values and shims."** This is the
  status-quo trajectory and is exactly what produces the zoo. The schema/pydantic
  divergence on `findings_match` is the first concrete symptom. Rejected.
- **Rebuild into a new, heavier plugin framework per test-type.** The premise
  that we lack a framework is false — `agent-eval-case` is the framework. A
  rebuild discards a sound, working spine to solve a leak at three seams.
  Disproportionate. Rejected.
- **Open `grader.type` to arbitrary strings (fully open vocabulary).** Maximises
  extensibility but loses validation safety and makes the contract unbounded —
  the dashboard/arbiter can no longer rely on a known set. The registry-under-
  `programmatic` keeps the strategy vocabulary closed while making *checkers*
  open. Rejected in favour of the bounded middle.

## Consequences

**Positive**

- Adding architecture-design / repo-analysis becomes: author a family + cases,
  register one checker if a new deterministic gate is needed. No core edits.
- The `findings_match` schema/code divergence gets closed rather than
  replicated.
- The API-vs-CLI ablation (proposals file 10, Ticket B) becomes valid because
  the envelope is shared by construction.
- The dashboard reads one scoped view; "everyone uses it differently" is removed
  at the source (the taxonomy registry).

**Negative / cost**

- One-time refactor of three seams: fold `findings_match` under
  `programmatic + checker`, lift the envelope out of the spawner, generalize the
  harness. Estimated ~1–2 days, not a rebuild.
- A checker registry adds one indirection vs a direct branch — worth it past the
  second capability, mild overhead at one.
- `grader.checker` is a schema addition; existing cases need a (mechanical)
  migration from `type: findings_match` to `type: programmatic, checker:
  findings_match`.

**Follow-up (not decided here)**

- The implementation plan for the three refactors (sequencing against R-07
  Phase-1b and the P3 strict-schema work in `_cowork_output/10-…`).
- Whether the harness lands as an `atp` subcommand or a single parameterized
  script in the interim.
- `malformed_rate` / `breakpoint` surfacing in the dashboard view (numeric
  `score_components` only — see file 10 P4).

## Result store vs export sink (resolution, 2026-06-14)

The unifying element is the `task_type` **dimension** plus **one internal
canonical store** — NOT the `report_benchmark` contract. There are two distinct
sinks off the same run:

```
run → grader (CaseVerdict) → CANONICAL STORE (EvalRun/EvalCaseResult, dimensioned)
                                  ├──▶ DASHBOARD  (reads store, scoped by task_type)
                                  └──▶ EXPORT     (report_benchmark-v1 → arbiter;
                                                   task_type → benchmark_id at the sink)
```

Why the dashboard must read the store, not the export (each disproves the rejected
"one contract through everything"):

- **Lossy.** `report_benchmark` is a per-agent aggregate; `per_task` is a JSON
  blob → no per-case rows to `GROUP BY` for the matrix/drill-down views (companion
  spec §4, §8 C/D).
- **Coverage.** The export is emitted only by campaign runs; plain `atp test` /
  method runs persist to the store but emit no export — so an export-fed dashboard
  would not show them. That is the original "where do I see atp-method runs" pain
  (companion spec §1).
- **Name overload.** "benchmark_runs" already names three different things
  (`suite_executions`/`test_executions`; the pull-model benchmark `Run` product
  table; the arbiter `benchmark_runs`). The dashboard source is the **first**; the
  export target is the **third**.

**Naming guard (resolved 2026-06-14):** the store-side dimension is **`task_type`**
(the arbiter `TaskType` canon), NOT `benchmark_id` — chosen over `vertical_id` to
avoid inventing a 4th taxonomy name. `benchmark_id` lives only on the arbiter
export; the taxonomy registry maps `task_type ↔ benchmark_id` at the sink. This
keeps "benchmark" off the internal store entirely (it already names three other
things). Applied throughout this ADR and the companion spec (§3, §5.1).

## Resolved decisions (answers to the prior open questions)

1. **Envelope (seam #2):** lift to a capability-level shared location now
   (`method/envelopes/` or a family field) — kill the cross-shim import
   immediately; a full schema field comes later. (YAGNI on the schema, pay-once on
   the duplication.)
2. **`grader.checker` migration:** fold `findings_match` back under
   `programmatic + checker` **now** — it closes a real contract bug (schema↔pydantic
   divergence) and is a mechanical 2-case migration. Part of the first refactor.
3. **Harness landing:** interim — parameterize `run_pipe_check.py` by
   `--family`/`--benchmark-id`; fold into an `atp` subcommand in companion spec
   **SP-2**.
4. **ADR status:** **Accepted**, linked to the companion spec as the two halves.

## Merged roadmap (this ADR + companion spec)

| Phase | Scope | Maps to |
|---|---|---|
| **A — harden the spine** (~1–2 d) | checker registry under `programmatic` + close the `findings_match` divergence + migrate the 2 cases; **unify the grader verdict** (`CaseVerdict` with explicit `malformed`, see dependency note); lift envelope to capability level; parameterize the harness; stand up the `task_type`↔`benchmark_id`↔`TaskType` taxonomy registry | ADR seams #1–#3 + spec §6 grader part |
| **B — canonical store + write** | evolve `suite_executions`/`test_executions` (dimension + version columns), persist for ALL runs | spec **SP-1** |
| **C — dashboard + export sink** | leaderboard + history/trend over columns; campaign run-path + `report_benchmark` reads from the store | spec **SP-3** ∥ **SP-2** |
| later | matrix + drill-down; `workspace` mode + `code_exec` grader | spec **SP-5**, **SP-6** |

**Dependency note (cross-doc, easy to miss):** companion spec **SP-1** persists a
`malformed` column sourced from `MatchResult.details`, but that field does not yet
exist on `MatchResult` — it is introduced by **P3** in
`_cowork_output/10-code-review-eval-improvements-proposals.md` ("collapse format
failure and match failure into one outcome"). P3, ADR direction #1 (checker
registry), and spec §6 (`CaseVerdict`) are **the same refactor** — define one
uniform verdict (`critical_pass`, `malformed`, `recall`, `precision`, …) and the
registry that produces it. Do this together in **Phase A** so SP-1 persists from a
stable `CaseVerdict`, not from `findings_match`-specific internals. Therefore:
**P3 is a precondition of SP-1.**
