# Step 6 — honest `score_semantics` on the benchmark plane

Ratified before implementation, so the contract is not shaped by whatever the code
turned out to do. Steps 1–5 shipped in #267–#271; scoring itself is still
completion-only, because the resolver reaches `app.state.evaluation` and nothing asks
it yet. This step is where it starts being asked.

## Semantics table

| Successful quality evaluators | `kind` | `quality_signal` | `score_components` |
|---|---|---|---|
| 0 | `completion_rate` | `false` | `{}` |
| ≥1 | `aggregated_evaluation` | `true` | only successfully computed components |
| some applied, some skipped/failed | `aggregated_evaluation` | `true` | successful components; the rest reported separately with a reason |

## Invariants

1. **`total_score` is meaningless without `score_semantics`.** It is interpreted only
   together with `kind`, `version` and `unit`. A consumer that reads the number alone
   is reading an unlabelled quantity.
2. **Completion and quality components never mix in one unlabelled aggregate.** If both
   exist they are distinguishable, or they are not combined.
3. **Capability is not evidence.** Assertions in the suite, a connected resolver, and
   `deterministic_allowlist` mode each mean evaluation *could* happen. None of them
   means it *did*.
4. **`quality_signal = true` only when at least one quality evaluator was successfully
   applied.** Not when one was permitted, resolved, or attempted.
5. **An unknown, disallowed, skipped or failed evaluator never becomes a zero
   component.** Zero reads as "assessed and bad"; these are "not assessed", and the
   difference is the whole point of the contract.
6. **No results means the honest completion score**, labelled as such — not a quality
   score of zero.
7. **Partial evaluation carries coverage**: which evaluators were requested, applied,
   skipped and failed, each with a reason.
8. **Component order does not affect the canonical fixture.** Serialization is sorted,
   so a fixture diff means a semantic change rather than an iteration-order change.
9. **No new database persistence.** The map stays an API-layer view; the
   storage-unification EPIC still owns that decision (deferred trigger recorded in
   `TODO.md`).

## Mutation probes

Each must turn a specific test red. A probe that leaves everything green means the
test is not testing what its name claims — the failure mode that hit this workstream
three times (`/etc/hosts` with no extension, `find_module` inert on 3.12, a compose
test reading the wrong file).

| Mutation | Must fail |
|---|---|
| do not call the resolver | deterministic-scoring test |
| return the completion score instead of the evaluator result | semantics/components test |
| drop an applied component | fixture/contract test |
| permit a forbidden evaluator | policy test |
| set `quality_signal=true` with zero applied quality evaluators | semantics test |
| relabel `kind` as `completion_rate` after a real evaluation | semantics test |

**The "do not call the resolver" probe needs a fixture whose suite requires a
deterministic evaluator.** Otherwise the run legitimately takes the completion-only
path, the test passes for a real reason, and the probe proves nothing — green for the
wrong reason once more.

All six were run and each turned its named test red (`6/6`). The caveat earned
its place twice over: the first version of the integration suite used
`config: {"text": ...}` where the artifact evaluator reads `pattern`, so the
*failing*-assertion test passed for a malformed-config reason rather than a
content one. The paired passing-assertion test is what caught it.

## What the wiring exposed

Two evaluators were permitted by `UNTRUSTED_SUBMISSION` that should not have
been. Neither was reachable before this step — no evaluator ran on the
benchmark plane at all — so both are defects created by the wiring, and both
are fixed in it.

**`filesystem` reads the host, not the submission.** It takes `workspace_path`
straight from `assertion.config`, so it inspects a directory named in the
suite rather than the sandbox the submission was materialized into. On this
plane it therefore cannot measure the agent's work at all, and the pass/fail
it does return is an existence answer about the server's own disk, handed to
whoever ran the benchmark. Creating a benchmark is admin-only
(`create_benchmark` requires `AdminUser`), so that is a misconfiguration leak
rather than a self-service oracle — the first reason stands regardless.

**`composite` resolves its own leaves.** `CompositeEvaluator._evaluate_leaf`
calls `get_registry()` directly, so nesting `pytest` under a `composite`
assertion reaches an evaluator the policy exists to withhold. The pipeline
checks the policy before *it* resolves an evaluator and cannot see inside one,
so allowing this class would have made every other exclusion advisory.

Both are now classified in `atp.evaluation.vocabulary`
(`READS_HOST_FILESYSTEM`, `DELEGATES_TO_REGISTRY`) and subtracted from the
allowlist by derivation, not by a hand-kept denylist. The allowlist drops from
25 assertion types to 19.

## Storage

Per-task evidence goes into `TaskResult.eval_results`, an existing nullable
JSON column that has never held anything on this plane. No migration and no
new column: the run-level map stays derived at read time, so invariant 9
holds — what is stored is the evidence, not the published view of it.

## Out of scope

Executing and network evaluators stay excluded until the worker boundary of step 7
exists: durable queue, resource and time limits, network policy, cost budget,
idempotency, cancellation, audit trail.
