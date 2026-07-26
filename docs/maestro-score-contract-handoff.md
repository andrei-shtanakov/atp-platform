# Handoff to maestro — benchmark score contract v1

**Owner of this contract:** atp-platform (`github:andrei-shtanakov`).
**Consumer:** maestro, `maestro/benchmark/atp_client.py` → `MaestroATPAdapter.finalize()`.
**Observed at:** 2026-07-26. **Status of this document:** ATP side complete; the
maestro-side consumer contract test is yours to write — it cannot be authored from
this repo.

## Why this exists

`finalize()` reads `total_score` from `GET /api/v1/runs/{id}/status` and records
`score_components={}` with the note "пока ATP не экспортирует breakdown"
(`../maestro/TODO.md:100`). Two problems with the pre-v1 wire:

1. **The breakdown had no agreed shape**, so there was nothing for you to code against.
2. **The number arrived bare.** On this plane a task scores 100 when the agent returned
   a *completed* response, whatever it contained — evaluators are not wired in here yet.
   A bare `total_score` invites a consumer to read completion as quality. That is the
   failure this contract is designed to prevent, so please do not treat it as pedantry.

## What ships now

`score_semantics` and `score_components` are on **both** score-bearing responses
(`RunStatusResponse` and `RunResponse`). They are computed on the API layer:
there is **no new database column**, deliberately — see "Deferred" below.

```json
{
  "total_score": 66.66666666666667,
  "score_semantics": {
    "schema_version": 1,
    "kind": "completion_rate",
    "level": "run",
    "unit": "percent",
    "range": {"min": 0.0, "max": 100.0},
    "quality_signal": false,
    "aggregation": {"function": "mean", "over": "task_score"},
    "task_score": {
      "kind": "completion_boolean", "level": "task",
      "unit": "percent", "values": [0.0, 100.0]
    },
    "caveats": ["null_until_finalized: ...", "zero_is_ambiguous: ..."],
    "note": "Completion, not quality: ..."
  },
  "score_components": {}
}
```

### The two levels are different quantities

Deliberately not merged under one name: **per task** (`completed_tasks[].score`) is a
completion boolean expressed as `0.0 | 100.0`; **per run** (`total_score`) is the mean of
those, i.e. a percentage. Aggregating a boolean into a percentage is the only reason the
run-level number is not itself binary.

### Two traps, stated on the wire rather than left to production

- `null_until_finalized` — `total_score` is `null` until the run completes.
- `zero_is_ambiguous` — a run that scored no tasks finalizes to `0.0`
  (`_finalize_run`: `sum(scores) / len(scores) if scores else 0.0`), which is
  indistinguishable from every task failing. If you need to tell those apart, use
  `tasks_count` and `completed_tasks`.

## What we ask of the consumer

1. **Branch on `quality_signal`** before showing the number to a human or feeding it to a
   router. Today it is `false`.
2. **Ignore unknown keys** — both unknown component names under `score_components` and
   unknown keys inside `score_semantics`. This is what makes adding components additive
   rather than a breaking change; `run_status_forward_compat.json` exists to test it.
3. **Treat a missing `score_semantics` as legacy/unknown semantics** — never as a quality
   score. Older producers predate v1.
4. `score_components` is `{}` today. It is a map, not a list, keyed by component name.

## Pins

Contract version **1**. Canonical `score_semantics` (JSON, sorted keys, no whitespace):

```
sha256  ad883233f5d05cfb826ed545da6bbb5d89b9d694d534af261182cb82b6f0e4fc
```

Fixtures, in `tests/fixtures/benchmark_score_contract/` in this repo:

| Fixture | sha256 | Purpose |
|---|---|---|
| `run_status_completion_only.json` | `d601274726550504c6de0df1e9dde28404f6e63a29de15d67106b2330ea94653` | today's real payload |
| `run_status_forward_compat.json` | `09f85de6e767ab635ec10755657372877211555aec168689a2afb24e895c3a3b` | populated + unknown component, unknown semantics key, `quality_signal: true` |

Source of truth: `packages/atp-dashboard/atp/dashboard/benchmark/score_contract.py`.
ATP-side tests: `tests/unit/dashboard/test_score_contract.py` — including a test that the
fixtures still match live serialization, and one asserting no persistence column appeared.

## Deferred, with an explicit trigger

**No DB column until the first real component is computed.** Storing `{}` on every row
would mint a second persistence representation next to the existing `ScoreComponent`
table (on the CLI/`TestExecution` plane) *before* the storage-unification EPIC has chosen
one. When the first evaluator component lands, that EPIC picks the persistence model; the
wire map does not change shape, so this contract survives the decision.

If ATP ever needs to **accept** components from outside (round-trip rather than export), a
JSON column on `Run` becomes the preferred interim, and this document must be revised —
the current one-way exporter does not need it.

## Not done here (repo boundaries)

- The maestro-side consumer contract test — your repo.
- Updating the dependency status on the maestro side — your repo.
- A pointer in the ecosystem KB — `../prograph-vault/` is read-only from here.
