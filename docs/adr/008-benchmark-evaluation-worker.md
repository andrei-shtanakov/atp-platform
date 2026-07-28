# ADR-008: Evaluating Executing and Networked Assertions on the Benchmark Plane

**Status**: Proposed (2026-07-27) — revised 2026-07-27 after review.
Tracks **A** (#274) and **B** (this branch) are implemented; C and D remain
proposed and trigger-gated. The counts in *Context* below describe the plane
at decision time: 6 of the 13 withheld types have since been admitted, leaving
7 (`code_exec` ×5, `llm_eval`, `factuality`).
**Date**: 2026-07-27
**Builds on**: #272 (step 6 — the benchmark plane runs deterministic evaluators
and labels the result), `packages/atp-core/atp/evaluation/policies.py` (policy
per execution context), `atp/evaluators/container.py` (existing container
runtime).
**Supersedes the label** "step 7 — worker for executing evaluators" in
`TODO.md`, which turns out to name four different problems.

Paths are cited from the repo root. `atp/evaluation`, `atp/core` and
`atp/scoring` are symlinks into `packages/atp-core/`; where it matters the
package path is cited, because that is where the file lives.

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
| **D — `code_exec`** | executes submission-derived code | deferred execution **and** a proven isolation profile |

A and B are interface defects in evaluators that happen to be visible from this
plane. C and D are platform capabilities. Only D needs containers.

### 2. Track A: `composite` receives the resolver, and grows a third verdict

`CompositeEvaluator._evaluate_leaf` calls `get_registry()`
(`atp/evaluators/composite.py:282`). It should be handed the same
`EvaluatorResolver` the pipeline holds, so its leaves pass through the policy
that governs its parent. `EvaluatorRegistry.create_for_assertion` constructs
with no arguments (`atp/evaluators/registry.py:180`), so this is a
construction-path change rather than a one-liner.

**Filtering the leaves is the easy half.** The hard half is what a refused leaf
*means* to the operator above it. "Permitted leaves run, refused ones report a
refusal" is not an answer for `AND`/`OR`/`NOT`/`threshold`, and the obvious
shortcut — treating a refusal as `False`/`0.0` — reintroduces the exact defect
#272 removed: an unmeasured thing presented as a bad measurement.

So a leaf yields one of **three** verdicts, propagating by Kleene logic:

| Operator | Rule |
|---|---|
| `NOT` | `NOT PASS = FAIL`, `NOT FAIL = PASS`, **`NOT UNEVALUATED = UNEVALUATED`** |
| `AND` | any `FAIL` → `FAIL` (false regardless of the unknowns); else any `UNEVALUATED` → `UNEVALUATED`; else `PASS` |
| `OR` | any `PASS` → `PASS` (true regardless); else any `UNEVALUATED` → `UNEVALUATED`; else `FAIL` |
| `threshold` | evaluate the comparison at both ends of the score interval, taking each `UNEVALUATED` leaf as `[0.0, 1.0]`. Both ends agree → that verdict; they disagree → `UNEVALUATED` |

The asymmetry in `AND`/`OR` is the point: a conjunction containing one real
failure is genuinely false, and refusing to say so would be its own dishonesty.
The unknown wins only where it could change the answer.

A composite resolving to `UNEVALUATED` contributes **no component** and appears
in `coverage` with the reasons its leaves gave — the same treatment every other
unevaluated assertion gets.

**Acceptance for track A includes a docstring.** `UNTRUSTED_SUBMISSION`
currently states that all four withheld classes wait for an isolated worker
(`packages/atp-core/atp/evaluation/policies.py:34`). That sentence is what this
ADR contradicts; leaving it would leave the codebase asserting the opposite of
the accepted decision.

### 3. Track B: the root is always injected, and the config is never silently ignored

Today `workspace_path` is required and *is* the root
(`atp/evaluators/filesystem.py:41`). The first draft of this ADR proposed
ignoring that key when a root is injected. That is wrong twice: it makes one
spec field mean different things on different planes, and silent ignoring is
how a suite author's intent disappears without a diagnostic.

The contract instead:

1. **The evaluator always receives a trusted root** from its composition — the
   `ArtifactWorkspace` on the server, the working directory on the CLI. It
   never derives a root from the suite.
2. **`workspace_path`, if kept, is a relative subpath *within* that root**, not
   a root of its own. The field keeps its usefulness — address a subdirectory
   of the workspace — and loses its authority.
3. **An absolute path or a traversal is a config error**, reported as such:
   not quietly clamped, not quietly ignored.
   `validate_path_within_workspace`
   (`packages/atp-core/atp/core/security.py:478`) already draws that line.
4. **Existing CLI suites are converted explicitly**, by an adapter that
   rewrites the old meaning into the new one — not by assuming that injecting
   the working directory reproduces it. In general it does not: today's
   `workspace_path` can point anywhere.

**As implemented**, point 4 is `FilesystemEvaluator._legacy_absolute_root`: an
absolute `workspace_path` is honoured as the root, with a deprecation warning
naming the new meaning, **only** where the policy declares the plane trusted —
the operator's own machine, naming the operator's own directory. Anywhere else
it is the config error of point 3, because there it never named the submission
in the first place. The grant carries `trusted` alongside the root for exactly
this, and `trusted` is a declared field on `EvaluationPolicy` rather than
inferred from an empty allowlist: "which evaluators may run" and "whose machine
is this" are different questions, and a CLI that one day restricts the former
must not silently stop being the latter. Transitional; removed once a release
has shipped with the warning. No suite in this repository uses the field, so
the conversion burden is entirely external.

Point 3's answer is a failing check rather than an `AssertionUnevaluated`. A
malformed `workspace_path` is the *suite author's* error: it exists before any
agent runs and is identical for every submission, so it is not the "we could
not measure this participant" case that the three-valued model in §2 protects.
Reporting it as a failure with the reason attached puts it where the author
will see it; routing it into coverage would file a broken suite under "not
tested" and leave it there.

**Separately, a latent bug that deserves its own regression test.** An invalid
path in `file_not_exists` returns `passed=True`, "treated as not existing"
(`atp/evaluators/filesystem.py:129`). On a policy boundary that is backwards: a
path the evaluator could not resolve is an *unanswered* question, and answering
it "absent, therefore pass" lets a malformed assertion score points. It should
be a config error like any other, and a test should keep it that way.

### 4. Tracks C and D share one prerequisite: evaluation leaves the request path

Both need `submit` to accept, persist and return, with evaluation happening
afterwards:

- `submit` returns the stored task result with evaluation **pending**, not a
  score;
- a job records what is to be evaluated, its state, its attempts and its
  outcome;
- `GET /runs/{id}/status` reports pending evaluation as a state.

**This is a wire change, not an internal one.** `submit` returns `202 Accepted`
when it enqueues and stays `200` when it scores inline, so a client can tell
the two apart without parsing the body. The published SDK
(`atp-platform-sdk`) must treat `202` as success and expose the pending state:
a client that polls `status` until `total_score` is non-null already behaves
correctly, but one that reads the score straight out of the submit response
does not. `SCORE_CONTRACT_VERSION` increments.

### 5. Pending is a state, not an absence

Today a task with no `eval_results` means "nothing was evaluated, the score is
completion". Once evaluation is deferred the same absence can mean "not
evaluated **yet**". Publishing `completion_rate` for a run whose evaluation is
queued would be exactly the defect #272 removed, reintroduced through the back
door: a confident label on an unfinished measurement.

So `score_semantics` gains an explicit evaluation state, and a run with pending
work is neither `completion_rate` nor `aggregated_evaluation` until its jobs
reach terminal states. `coverage` gains the pending count.

**The leaderboard must not rank an unsettled run.** The earlier draft left
"exclude or display as unsettled" to the implementation; that is a contract
decision, not a rendering one. A rank is a claim about relative quality, and a
run scored so far only by completion would be ranked on a number about to
change. Unsettled runs are excluded from rank until terminal, and may be shown
as pending.

### 6. Identity, and an honest delivery guarantee

The earlier draft proposed a natural key of `(run_id, task_index, evaluator)`
and called the result idempotent. Both halves were wrong.

**The key collides.** A test may carry two assertions of the same type — two
`contains` with different patterns is ordinary — so the key must include the
assertion's position: `(run_id, task_index, assertion_index)`. The stored
*outcome* additionally records the evaluator name, its version and a digest of
its input, so a re-run under changed code is recognisable as a new measurement
rather than silently overwriting an old one.

(By contrast `score_components` keys by assertion *type* and averages within
it. That is a per-type mean by design — a different question from job
identity.)

**Exactly-once is not available, and claiming it is worse than not having it.**
A job can crash after the provider has answered and before the outcome is
written; the lease expires and the job runs again. A unique key prevents a
duplicate row. It does nothing about a duplicate charge, because the side
effect happened outside the database.

So the guarantee is stated as it actually is:

- **execution is at-least-once**, bounded by `max_attempts`;
- **charging is at-most-once where the provider supports an idempotency key**,
  which is sent and stored with the job;
- **where it does not**, duplicate spend on retry is an accepted, bounded
  operational risk — bounded by `max_attempts` and by the reservation below,
  and visible because every attempt records its provider request ID;
- **budget is reserved before the call and reconciled after**, so a crash leaks
  a reservation (recoverable, conservative) rather than overrunning the budget
  (not recoverable).

Reservation must be atomic against concurrent submissions: two submissions
racing for the last of a budget must not both succeed. That is a conditional
update on the budget row, not a read-then-write.

### 7. Track D: the primitive exists; the isolation profile does not

The container runtime exists. `CodeExecEvaluator` accepts an injected
`ContainerRuntime` (`atp/evaluators/code_exec.py:109`), and the runtime already
sets `network=none`, a read-only root, CPU and memory limits and a temporary
workspace. That is a real head start, and the earlier draft was right about the
*primitive*.

It was wrong about the *profile*. The same module documents a subprocess
fallback — "Falls back to subprocess + rlimits if no runtime"
(`atp/evaluators/container.py:4`) — and `_detect_runtime` returns `None` with
"No container runtime available, using subprocess sandbox"
(`atp/evaluators/container.py:194`). That is **fail-open**: on a host where the
runtime is missing or the socket is unreachable, submission-derived code runs
in the API process's own sandbox. Sensible on a developer's laptop;
unacceptable under `ISOLATED_WORKER`.

So this section claims only what it can support: **a container execution
primitive exists; a production isolation profile has yet to be built and
demonstrated.** That profile requires, at minimum:

- **fail-closed**: no container runtime → the job fails with a reason; never a
  subprocess fallback;
- **an image allowlist with digest pins** — not tags, which are mutable;
- **rootless runtime, and a non-root user inside the container**;
- **`no-new-privileges`, all capabilities dropped, a PID limit**;
- **bounded stdout/stderr**, so output volume cannot exhaust the host;
- **no host path mounts and no Docker socket**, ever;
- **an adversarial isolation test** — a case that *tries* to escape, reach the
  network and exhaust resources, and is expected to fail — rather than
  `is_available()` returning true;
- **a written Docker-vs-Podman threat-model comparison for this use**, since
  rootless Podman and a mounted Docker socket are very different exposures.

**Do not build D before there is a suite that needs it.** Nothing shipped in
`examples/`, `method/` or `benchmarks/` asserts `pytest`, `code_exec`, `npm`,
`lint` or `custom_command` — checked 2026-07-27. Building container execution
for a hypothetical suite is how a platform acquires an attack surface nobody is
using.

### 8. Network evaluators need a budget, an opt-in, and a snapshot

`llm_eval` and `factuality` execute nothing. Their problems are cost and
reproducibility, and the cost is *asymmetric*: a self-service participant's
submission spends the operator's money, at up to 120 submissions per minute.

Therefore: a per-benchmark budget with a hard cap, reserved atomically as in
§6; an explicit operator opt-in per benchmark rather than a global switch,
gated by its own RBAC permission rather than by "is admin"; and a recorded
per-submission cost. An exhausted budget makes assertions **skip with a
reason** — they do not become zeros, and they do not silently fall back to a
cheaper judge.

**The opt-in and its budget are snapshotted when a run starts.** Otherwise a
benchmark edited mid-run scores its early and late tasks under different rules,
and the run's total means nothing in particular.

### 9. Jobs live in a table, and the table needs a protocol

The deployment is one VPS running Docker Compose (`deploy/docker-compose.yml`).
Redis or Celery would add a service to operate, monitor and secure, for a queue
whose depth is bounded by benchmark participation. A job table plus a poller
gives durability across restart — the property `asyncio.create_task` lacks —
with no new infrastructure, and keeps job state in the same transaction as the
task result it belongs to.

"A table and a poller" is not a design, though. Production runs uvicorn and can
run multiple workers, so **two processes will poll the same table**. The
protocol is what stops them both running one job:

- **atomic claim**: a single conditional `UPDATE … WHERE state='queued'`
  returning the claimed row. Never select-then-update.
- **lease with expiry, plus heartbeat**: a claim is time-bounded; a worker that
  dies loses the lease and the job returns to `queued`. Interval and duration
  are configured, not implied.
- **attempts, backoff, `max_attempts`** — with the §6 caveat that retrying a
  networked job may spend money again.
- **terminal states** `SUCCEEDED` / `FAILED` / `CANCELLED`, never re-entered.
- **dead-letter and operator retry**: a job exhausting `max_attempts` is
  inspectable and re-queueable by hand, not silently lost.
- **atomic run finalization**: `total_score` is computed once, when the run's
  last job reaches a terminal state, under a condition two concurrent
  finalizers cannot both satisfy.
- **cancellation**: `POST /runs/{id}/cancel` cancels that run's queued jobs and
  signals running ones, or the platform keeps spending on a run its owner
  abandoned.
- **graceful shutdown**: in-flight jobs are finished or released before exit,
  so a deploy does not park work until lease expiry.
- **recovery on start**: leases held by a previous incarnation of this process
  are reclaimed rather than waited out.
- **observability**: queue depth, oldest-queued age, lease expiries and failure
  rate are metrics, because a queue nobody can see is a queue that silently
  stops.

Revisit the choice if evaluation ever needs more than one host; a job table
migrates to a broker far more easily than the reverse.

## Non-goals

- Making judged scores reproducible. They are judgements, and the contract says
  so.
- Running participant code with network access. Different threat model,
  different ADR.
- Retro-evaluating existing runs. Semantics are derived from stored evidence;
  runs without evidence stay `completion_rate`, honestly.

## Consequences

- **Leaderboard ordering becomes eventually consistent**, and unsettled runs go
  unranked until terminal (§5).
- **Cost becomes an operational concern for the first time.** Nothing on this
  plane has spent money per request before.
- **Retry can cost money** (§6) — a property of the design, stated rather than
  hidden behind the word "idempotent".
- **`submit` gains a second success status** (§4), which is a published-SDK
  concern and not only a server one.
- **Tracks A and B change `UNTRUSTED_SUBMISSION`** — correctly: they remove the
  reason the classification excluded those evaluators. The worker gets its own
  `ISOLATED_WORKER` policy; the in-process policy never widens to accommodate
  it.

## Sequencing and triggers

| Track | When |
|---|---|
| **A — `composite`** | ✅ Done (#274). A policy hole, closed by a construction-path change plus the three-valued model in §2. |
| **B — `filesystem`** | ✅ Done. Same shape of fix — the grant travels down through `composite` exactly as the resolver does — plus the `file_not_exists` regression test. |
| **C — network** | When an operator wants a judged benchmark and accepts the bill. Requires §4, §5, §6, §8, §9. |
| **D — `code_exec`** | When a suite that needs it exists. Requires everything above plus the §7 isolation profile, demonstrated adversarially. Not before. |

A and B are ordinary work. C and D are each their own project, and this ADR
exists so neither is started by accident when someone reads "step 7" and
assumes it means "build the worker".
