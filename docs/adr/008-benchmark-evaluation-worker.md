# ADR-008: Evaluating Executing and Networked Assertions on the Benchmark Plane

**Status**: Proposed (2026-07-27)
**Date**: 2026-07-27
**Builds on**: #272 (step 6 — the benchmark plane runs deterministic evaluators
and labels the result), `packages/atp-core/atp/evaluation/policies.py` (policy per
execution context), `atp/evaluators/container.py` (existing container runtime).
Paths are given as they resolve from the repo root; `atp/evaluation` is a
symlink to the atp-core package, so the package path is cited where the file
actually lives.
**Supersedes the label** "step 7 — worker for executing evaluators" in `TODO.md`,
which turns out to name four different problems.

## Context

#272 wired the shared evaluation pipeline into `POST /api/v1/runs/{id}/submit`.
Of 32 assertion types, **19 run** and **13 are withheld** by
`UNTRUSTED_SUBMISSION`:

| Withheld | Types | Why |
|---|---|---|
| `code_exec` | `code_exec`, `pytest`, `npm`, `lint`, `custom_command` (5) | runs code derived from a stranger's submission, inside the API process |
| `llm_judge` | `llm_eval` (1) | network call; spends money per submission; non-reproducible |
| `factuality` | `factuality` (1) | same |
| `filesystem` | `file_exists`, `file_not_exists`, `file_contains`, `file_count`, `dir_exists` (5) | resolves `workspace_path` from `assertion.config`, so it addresses a directory named in the suite rather than the sandbox the submission was materialized into |
| `composite` | `composite` (1) | resolves its leaves through `get_registry()` directly, so nesting `pytest` under it walks past the policy |

All five landed on one list, so they acquired one blocker: "wait for an
isolated worker". That reading is wrong, and it is expensive — it holds two
cheap interface fixes hostage to the most complex piece of infrastructure on
the list.

Two facts constrain any answer:

**Submission is synchronous.** `submit` evaluates inline in the request
handler, holding a DB session, under a 120/minute rate limit. That is right for
evaluators measured in microseconds. A `pytest` run or an LLM call in the same
place makes the client's HTTP timeout the de-facto evaluation timeout and turns
one slow submission into a stuck worker slot.

**There is no durable background execution.** The only background work today is
webhook delivery via `asyncio.create_task`
(`packages/atp-dashboard/atp/dashboard/webhook.py:151`), which is
fire-and-forget and loses in-flight work on restart. Nothing exists to build on.

## Decision

### 1. The bundle splits into four tracks with different blockers

| Track | Blocker | Needs a worker? |
|---|---|---|
| **A — `composite`** | evaluator reaches for the global registry instead of receiving a resolver | no |
| **B — `filesystem`** | evaluator is *told* a directory instead of being *given* a sandbox | no |
| **C — network (`llm_eval`, `factuality`)** | synchronous request path, unbounded cost, non-reproducibility | deferred execution, not isolation |
| **D — `code_exec`** | executes submission-derived code | deferred execution **and** isolation |

A and B are interface defects in evaluators that happen to be visible from this
plane. C and D are platform capabilities. Only D needs containers.

### 2. Track A: `composite` receives the resolver it is allowed to use

`CompositeEvaluator._evaluate_leaf` calls `get_registry()`
(`atp/evaluators/composite.py:282`). It should be handed the same
`EvaluatorResolver` the pipeline holds, so its leaves pass through the policy
that governs its parent. Then `composite` under `UNTRUSTED_SUBMISSION`
composes permitted leaves and reports a policy refusal for the rest — the same
answer the pipeline gives at the top level.

`EvaluatorRegistry.create_for_assertion` constructs with no arguments
(`atp/evaluators/registry.py:180`), so this is a construction-path change
rather than a one-liner. That is the cost, and it is small.

**Once the leaves are policy-filtered, `composite` returns to the allowlist.**
It never needed a worker; it needed to stop being a hole in the policy.

### 3. Track B: `filesystem` is given a root, or stays out

The evaluator's contract — "name a directory in the suite, I will look in it" —
cannot be made safe by wrapping it, because the directory is its input. Two
honest options:

1. **Give it the sandbox.** The evaluator gains an injected workspace root; the
   `workspace_path` config key is ignored when a root is injected. Server-side
   the root is the `ArtifactWorkspace` the submission was materialized into, so
   `file_exists` finally measures the agent's own output. The CLI keeps
   today's behaviour by injecting the working directory.
2. **Leave it CLI-only.** The benchmark plane checks artifacts through the
   `artifact` evaluator, which reads the response, and never grows a filesystem
   assertion.

**Decision: option 1.** Option 2 permanently splits the vocabulary by plane,
which is the "zoo" ADR-007 exists to prevent, and it gives up a check that is
genuinely meaningful once the root is right. Path arguments from the suite are
confined to the root via the existing `validate_path_within_workspace`.

### 4. Tracks C and D share one prerequisite: evaluation leaves the request path

Both need `submit` to accept, persist, and return — with evaluation happening
afterwards. That is the shared piece, and it is a bigger contract change than
either evaluator class:

- `submit` returns the stored task result with evaluation **pending**, not a
  score;
- a job records what is to be evaluated, its state, its attempts, and its
  outcome;
- `GET /runs/{id}/status` reports pending evaluation as a state.

### 5. Pending is a state, not an absence

This is the contract consequence and the part most likely to be got wrong.

Today a task with no `eval_results` means "nothing was evaluated, the score is
completion". Once evaluation is deferred, the same absence can mean "not
evaluated **yet**". Publishing `completion_rate` for a run whose evaluation is
queued would be exactly the defect #272 removed, reintroduced through the back
door: a confident label on an unfinished measurement.

So `score_semantics` gains an explicit evaluation state, and a run with pending
work is neither `completion_rate` nor `aggregated_evaluation` until its jobs
settle. The existing `null_until_finalized` caveat covers `total_score`; this
covers the label. `coverage` gains the pending count.

### 6. Network evaluators need a budget and an opt-in, not a container

`llm_eval` and `factuality` do not execute anything. Their problems are cost
and reproducibility, and the cost is *asymmetric*: a self-service participant's
submission spends the operator's money, at up to 120 submissions per minute.

Therefore: a per-benchmark budget with a hard cap, an explicit operator opt-in
per benchmark (not a global switch), and a recorded per-submission cost. When
the budget is exhausted, assertions are **skipped with a reason** — they do not
become zeros, and they do not silently fall back to a cheaper judge.

Reproducibility stays a caveat rather than a promise: the semantics already
carry `quality_signal`, and a judged score is honest about being a judgement.

### 7. Code execution needs isolation on top of all of the above

Only track D needs the full list: container isolation, resource and time
limits, no network by default, a filesystem confined to the materialized
workspace, cancellation, and an audit record of what was executed for which
submission.

Worth stating plainly, because it inverts the intuition the name "worker"
creates: **the container is the part that already exists.**
`CodeExecEvaluator` accepts an injected `ContainerRuntime`
(`atp/evaluators/code_exec.py:109`), so isolation is a composition decision,
not a build. What does not exist is everything around it — the job model, the
limits, the accounting, the audit trail, and the contract changes in §4 and §5.
Estimating this track by its scariest word gets the cost backwards.

**Do not build D before there is a suite that needs it.** Nothing shipped in
`examples/`, `method/` or `benchmarks/` asserts `pytest`, `code_exec`, `npm`,
`lint` or `custom_command` — checked 2026-07-27. Building a container-execution
service for a hypothetical suite is how a platform acquires an attack surface
nobody is using.

### 8. A third policy, not a wider one

The worker gets its own `EvaluationPolicy` (`ISOLATED_WORKER`), derived the same
way from the vocabulary classification. `UNTRUSTED_SUBMISSION` — the in-process
policy — never widens. The two contexts are then distinguishable in the code
that grants them, and an evaluator permitted in the worker cannot become
permitted in the API process by accident.

Tracks A and B do change `UNTRUSTED_SUBMISSION`, and correctly: they remove the
reason the classification excluded those evaluators, so the derivation yields a
different answer. That is the classification working, not being overridden.

### 9. Jobs live in the database, not in a broker

The deployment is one VPS running Docker Compose (`deploy/docker-compose.yml`).
Adding Redis or Celery adds a service to operate, monitor, and secure, for a
queue whose depth is bounded by benchmark participation.

A job table plus a poller in the same process gives durability across restart —
the property `asyncio.create_task` lacks — with no new infrastructure. It also
keeps job state in the same transaction as the task result it belongs to, which
is what makes "submitted but not yet evaluated" a consistent read rather than a
race.

Revisit if evaluation ever needs to scale horizontally; a job table migrates to
a broker far more easily than the reverse.

## Non-goals

- Making judged scores reproducible. They are judgements; the contract says so.
- Running participant code with network access. If a suite needs that, it is a
  different ADR with a different threat model.
- Retro-evaluating existing runs. Semantics are derived from stored evidence;
  runs without evidence stay `completion_rate`, honestly.

## Consequences

**Leaderboard ordering becomes eventually consistent.** A run can be complete
but not yet scored. The leaderboard must either exclude unsettled runs or show
them as unsettled; silently ranking a pending run by its completion score would
publish a wrong order.

**Idempotency becomes load-bearing.** A retried job must not double-charge an
LLM budget or double-execute a submission. Jobs need a natural key
(`run_id`, `task_index`, evaluator) and an outcome that is written once.

**Cancellation grows a second meaning.** `POST /runs/{id}/cancel` currently
marks a run cancelled; it must also cancel that run's pending jobs, or the
platform keeps spending on a run its owner abandoned.

**Cost becomes an operational concern for the first time.** Nothing on this
plane has spent money per request before.

## Sequencing and triggers

| Track | When |
|---|---|
| **A — `composite`** | Do it. It is a policy hole, and closing it costs a construction-path change. |
| **B — `filesystem`** | Do it after A. Same shape of fix, and it makes a withheld check meaningful rather than merely safe. |
| **C — network** | When an operator wants a judged benchmark and accepts the bill. Requires §4, §5, §6, §9. |
| **D — `code_exec`** | When a suite that needs it exists. Requires everything above plus §7. Not before. |

A and B are ordinary work. C and D are each their own project, and this ADR
exists so that neither is started by accident when someone reads "step 7" and
assumes it means "build the worker".
