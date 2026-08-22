# Handoff to maestro — benchmark score contract v1

**Owner of this contract:** atp-platform (`github:andrei-shtanakov`).
**Consumer:** maestro, `maestro/benchmark/atp_client.py` → `MaestroATPAdapter.finalize()`.
**Status of this document:** both sides shipped. ATP publishes the contract; the
maestro-side consumer test landed in maestro PR #202 (merge `46de3f5`), vendoring the
fixtures below.

**The pins in this document are recomputed in CI**, not maintained by hand — see
[Pins](#pins). They describe the bytes in *the same tree you are reading*, so vendor
against the commit you fetched rather than one quoted here.

## Why this exists

`finalize()` reads `total_score` from `GET /api/v1/runs/{id}/status`. Before v1 that
number arrived bare, and maestro recorded `score_components={}` with the note "пока ATP
не экспортирует breakdown" (`../maestro/TODO.md:123`, now a closed item — kept as the
citation for what the wire used to force). Two problems with the pre-v1 wire:

1. **The breakdown had no agreed shape**, so there was nothing for you to code against.
2. **The number arrived unqualified.** On this plane a task could score 100 purely
   because the agent returned a *completed* response, whatever it contained. A bare
   `total_score` invites a consumer to read completion as quality. That is the failure
   this contract is designed to prevent, so please do not treat it as pedantry.

Evaluators have since been wired into the benchmark plane, which makes the second
problem *worse* without the contract, not better: the number now means quality on some
runs and completion on others, and only the wire can say which.

## What ships now

`score_semantics` and `score_components` are on **both** score-bearing responses
(`RunStatusResponse` and `RunResponse`). They are derived on the API layer from the
per-task evidence in `TaskResult.eval_results`: there is **no new database column**,
deliberately — see "Deferred" below.

```json
{
  "total_score": 50.0,
  "score_semantics": {
    "schema_version": 1,
    "kind": "aggregated_evaluation",
    "level": "run",
    "unit": "percent",
    "range": {"min": 0.0, "max": 100.0},
    "quality_signal": true,
    "aggregation": {"function": "mean", "over": "task_score"},
    "task_score": {
      "kind": "completion_boolean", "level": "task",
      "unit": "percent", "values": [0.0, 100.0]
    },
    "coverage": {
      "tasks_total": 1, "tasks_submitted": 1,
      "tasks_evaluated": 1, "tasks_completion_only": 0,
      "records_unreadable": 0,
      "assertions_applied": {"contains": 1},
      "assertions_skipped": [
        {"assertion_type": "pytest", "reason": "not_allowed_by_policy", "count": 1}
      ]
    },
    "caveats": ["null_until_finalized: ...", "zero_is_ambiguous: ..."],
    "note": "At least one evaluator ran. ..."
  },
  "score_components": {"contains": 50.0}
}
```

### The one field to branch on: `quality_signal`

**`kind` is an open set; `quality_signal` is the stable branch key.** Today's producer
emits one of two kinds, and `quality_signal` tracks them exactly:

| `kind` (today) | `quality_signal` | what `total_score` counts |
|---|---|---|
| `completion_rate` | `false` | completions only — no evaluator ran |
| `aggregated_evaluation` | `true` | at least one evaluator *ran* and produced a result |

That table describes this producer, not the contract's ceiling. A later kind is an
additive change and must not break you — `run_status_forward_compat.json` deliberately
carries an unlisted one (`weighted_quality`) for exactly that reason. So match on
`quality_signal`, and treat `kind` as a label to log rather than a value to exhaust: an
`else` branch that assumes "not `aggregated_evaluation` ⇒ not quality" would misread the
first kind we add.

The boolean follows evidence, not capability. A server wired with evaluators still
publishes `completion_rate` for a suite that asserts nothing, for a submission that did
not complete, and for one whose every assertion the policy withheld — because none of
those is a measurement.

### The two levels are different quantities

Deliberately not merged under one name: **per task** (`completed_tasks[].score`) is
`0.0 | 100.0` — a completion boolean when nothing was evaluated, an aggregated
evaluation score when something was; **per run** (`total_score`) is the mean of those,
i.e. a percentage. Aggregating into a percentage is the only reason the run-level number
is not itself binary.

### `coverage` — what was requested, applied and withheld

This is what separates "assessed and scored low" from "never assessed". Without it a
withheld evaluator is indistinguishable from a failed one, and both look like zero
quality. Note the asymmetry it makes possible: **absence from `score_components` is not
a zero.** Only successfully applied assertion types appear there; a skipped, refused or
unreadable one is reported in `coverage` instead, never as a component worth `0.0`.

`assertions_skipped` carries a `reason` per entry; `records_unreadable` counts per-task
records stored under a `record_version` this server cannot read, reported rather than
dropped.

### Traps stated on the wire rather than left to production

- `null_until_finalized` — `total_score` is `null` until the run completes.
- `zero_is_ambiguous` — a run that scored no tasks finalizes to `0.0`
  (`_finalize_run`: `sum(scores) / len(scores) if scores else 0.0`), which is
  indistinguishable from every task failing. If you need to tell those apart, use
  `tasks_count` and `completed_tasks`.
- `mixed_task_scores` — appended when some tasks were evaluated and others scored by
  completion. The mean is still published, because that is what `total_score` has always
  been, but a blend of two different quantities must not travel unlabelled. The
  composition is in `coverage.tasks_evaluated` / `coverage.tasks_completion_only`.

## What we ask of the consumer

1. **Branch on `quality_signal`** before showing the number to a human or feeding it to
   a router. It is `true` only when an evaluator actually ran.
2. **Ignore unknown keys** — both unknown component names under `score_components` and
   unknown keys inside `score_semantics`. This is what makes adding components additive
   rather than a breaking change; `run_status_forward_compat.json` exists to test it.
   That fixture also carries an *object* component value: type your component map wide
   enough that a future non-scalar does not drop on the floor.
3. **Treat a missing `score_semantics` as legacy/unknown semantics** — never as a
   quality score. Older producers predate v1.
4. **Treat absence from `score_components` as "not measured", never as zero.** It is a
   map keyed by assertion type (not evaluator name — several assertion types share one
   evaluator, and evaluator keys would merge distinct measurements). It is `{}` exactly
   when `kind` is `completion_rate`.

## Pins

Contract version **1**.

Fixtures, in `tests/fixtures/benchmark_score_contract/` in this repo:

| Fixture | sha256 | Purpose |
|---|---|---|
| `run_status_completion_only.json` | `cd5e79e92e46d9ef81be60dd577663eed294057dc9e8935a849846a962dcfca1` | no evaluator ran: `completion_rate`, `score_components: {}` |
| `run_status_evaluated.json` | `0e2169cb04550be8d6b4cccd396ac4ee8ad68b03bae78e706e6913adaf68fa4a` | one applied + one withheld assertion: `aggregated_evaluation`, populated components, `coverage` |
| `run_status_forward_compat.json` | `09f85de6e767ab635ec10755657372877211555aec168689a2afb24e895c3a3b` | unknown component (object-valued), unknown semantics key, `quality_signal: true` |

Canonical `score_semantics`: the block published in `run_status_completion_only.json`,
serialized as JSON with sorted keys and no whitespace
(`json.dumps(..., sort_keys=True, separators=(",", ":"))`). Note this includes
`coverage`, so it does not equal a hash of the bare `run_score_semantics()` default.

```
sha256  43bd52de337bb38737fbafac049ed5ba2f53e3285dd1466a4418422f9ead9de1
```

**These digests are asserted, not asserted-to.** `TestHandoffPinsAreRecomputed` in
`tests/unit/dashboard/test_score_contract.py` recomputes every one of them from the
bytes on disk, and fails if a fixture is published without a row here. Prose pins drift
on the exact commit that changes the bytes and say nothing while they do — which is what
happened between `f58ff7f` and `05bd939`, reported by the consumer during vendoring
(issue #298).

Source of truth: `packages/atp-dashboard/atp/dashboard/benchmark/score_contract.py`
(schema) and `.../benchmark/scoring.py` (derivation).
ATP-side tests: `tests/unit/dashboard/test_score_contract.py` — the fixtures are checked
against live serialization, the pins above against the fixtures, and one test asserts no
persistence column appeared.

## Machine-readable pins: `DIGESTS.json`

The table above is for a human. Everything in it is also published as a sidecar at
`tests/fixtures/benchmark_score_contract/DIGESTS.json`, so a consumer can check drift
by downloading **one file** instead of keeping a checkout of this repo beside theirs.

```json
{
  "contract_version": 1,
  "sidecar_format_version": 1,
  "canonical_score_semantics_sha256": "43bd52de…",
  "files": {
    "packages/atp-dashboard/atp/dashboard/benchmark/score_contract.py": "fb543c00…",
    "tests/fixtures/benchmark_score_contract/run_status_completion_only.json": "cd5e79e9…",
    "…": "…"
  }
}
```

It is generated by `scripts/write_score_contract_digests.py` and asserted against a
fresh recomputation by `TestDigestSidecarIsRecomputed`, so it cannot go stale without a
red test. Keys are repo-relative paths, not basenames: a bare `score_contract.py` is
ambiguous in this repo.

Two things about it that look like omissions and are not:

- **`score_contract.py` is in the map, though it is not a fixture.** A fixture that
  drifts from the parser defining its meaning surfaces on the consumer's side as a
  failure on a live run rather than a red test; pinning the payloads alone would leave
  the sharper failure unguarded. This came from the consumer (maestro#204).
- **There is no commit SHA.** At the moment the sidecar is written, the commit that will
  carry it does not exist, so the field could only ever be stale or empty. Requested to
  be left out; please do not add it back as an obvious improvement.

The sidecar is **not** meant to be vendored — a consumer's own `PIN` already holds the
same digests. Its job is the other guarantee: *upstream has not moved*. Compare the map
against your pins, and flag both a key your pins do not have (a fixture published since
you vendored — the case a hardcoded list of known paths cannot catch) and a
`contract_version` you do not implement.

## Deferred, with an explicit trigger

**No DB column for the run-level map.** The per-task evidence is persisted in the
existing `TaskResult.eval_results`; the run-level `score_components` stays derived at
read time. Storing it would mint a second persistence representation next to the
existing `ScoreComponent` table (on the CLI/`TestExecution` plane) *before* the
storage-unification EPIC has chosen one. When that EPIC picks the persistence model, the
wire map does not change shape, so this contract survives the decision.

If ATP ever needs to **accept** components from outside (round-trip rather than export),
a JSON column on `Run` becomes the preferred interim, and this document must be revised —
the current one-way exporter does not need it.

## Not done here (repo boundaries)

- A pointer in the ecosystem KB — `../prograph-vault/` is read-only from here.
