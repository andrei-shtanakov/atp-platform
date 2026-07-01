# R-07 rank_score tiebreaker — unblock re-rank with the signal ATP already emits

**Status:** Design (approved) · **Date:** 2026-07-01
**Workstream:** R-07 eval-driven routing (track B, slice A)
**Repos touched:** atp-platform (primary), arbiter (small reader change)
**Related:** `../../../../_cowork_output/2026-07-01-reliability-orchestration-paper-eval.md`,
`_cowork_output/decisions/2026-06-13-r07-thin-slice.md` (re-rank mechanism)

## TL;DR

R-07's benchmark re-rank is a **silent no-op** on code-review: both routable agents
emit `score = critical_pass_rate = 0.800`, so `apply_benchmark_rerank` computes an
identical delta for each and never changes the order. But ATP **already computes a
discriminating signal** — `breakpoint_axis_level` (codex/mimo/pi hold to `severe`;
claude_code/opencode break at `moderate`) and `malformed_rate` — which the re-rank
never reads. This slice folds that existing signal into a numeric `rank_score`
component that arbiter's re-rank consumes, breaking the 0.8/0.8 tie **without a new
data pipeline, without changing `score` semantics, and without a DB migration.**

Verifier-reliability (the paper's new signal) is the **follow-on** (track B, slice B)
— see "Not in scope".

## Background: why re-rank is a no-op today

- **ATP emits** (`atp/reporters/benchmark_reporter.py`, `build_report_benchmark_payload`):
  scalar `score = critical_pass_rate`; `score_components = {critical_pass_rate,
  mean_rubric, malformed_rate}` (numbers only, schema constraint); and
  `breakpoint_axis_level` as a **top-level string** (`_AXIS_ORDER =
  [clean, mild, moderate, severe, very_severe]`; the lowest axis at which the
  critical check first fails, `None` if it never fails).
- **arbiter reads** (`arbiter/arbiter-mcp/src/db.rs`, `get_benchmark_score`):
  `SELECT score FROM benchmark_runs ... ORDER BY ts DESC LIMIT 1` → `Option<f64>`,
  clamped `[0,1]`. It ignores `score_components` and `breakpoint_axis_level`.
- **re-rank** (`arbiter/arbiter-mcp/src/tools/route_task.rs`,
  `apply_benchmark_rerank`): `delta = (score - 0.5) * weight`, confidence clamped
  `[0,1]`, re-sorted. No-op when `weight <= 0`, no mapped benchmark, or no score row.
- **Consequence:** two agents at `score = 0.800` → identical delta → tie unbroken.
  The 13-agent paid sweep (2026-06-21) confirmed the top tier is flat at 0.800 on
  `critical_pass_rate` and only separable by `breakpoint_axis_level`.

## Goal / non-goals

**Goal:** make R-07 re-rank stop being a no-op on the current data by having it
consume the tiebreaker ATP already produces.

**Non-goals (explicit follow-on — do NOT build here):**
- **Slice B — verifier-reliability signal.** Gated on spec-runner's additive
  `verification` block (`verifier_id`, `verdict: pass|fail|inconclusive`, `ran`)
  which does not exist yet (spec-runner emits only a scalar `review`), plus an
  undesigned aggregation→`report_benchmark` ingest. Same transport as this slice
  (another numeric component in `score_components`), so this slice is forward-
  compatible with it. Tracked in reliability-eval §2.1/§3.3.
- Changing `score` semantics, adding a DB column/migration, or per-axis weighting
  of `critical_pass_rate` (that remains Phase-1b).

### Known debt — combination policy lives in the measurement layer (MUST revisit before slice B)

The tiebreaker weights (`0.75 * bp_ordinal + 0.25 * (1 - malformed_rate)`) and the
whole `t` formula **are routing policy**, but this slice decides them in
`benchmark_reporter.py` and bakes them into one scalar that arbiter (the policy
engine) swallows as a finished `f64`. That contradicts the principle from the
reliability report — *combination policy belongs to the policy engine, not the
measurement worker*. It is acceptable **only** as a thin slice whose sole job is to
break the current tie.

**This is a hard precondition for slice B, not a nice-to-have.** When
verifier-reliability adds a second scalar, "who combines the signals and with what
weights" becomes acute; leaving it here fragments the combination logic in the
wrong layer (2+ ad-hoc ATP-side blends feeding one delta). To keep the migration
cheap and non-throwaway, this slice **already emits the raw numeric components**
(`bp_ordinal`, `malformed_rate`) in `score_components`. Slice B's repayment is
therefore **arbiter-only**: read the raw components and weight them policy-side,
retiring `rank_score` as the interim combined signal. No further ATP change needed.

## Design

### 1. `rank_score` — ATP side (`benchmark_reporter.py`)

Add to `score_components` in `build_report_benchmark_payload` (all numeric — the
schema is `additionalProperties: {type: number}`, byte-identical across the three
vendored copies, so this needs no schema change and no `payload_version` bump):

- **`rank_score`** — the interim combined routing signal the reader uses today.
- **`bp_ordinal`** — the *raw* numeric breakpoint (forward-compat: lets arbiter
  combine raw signals policy-side at slice B without a second ATP change).
  (`malformed_rate` is already emitted.)

```
rank_score = critical_pass_rate + (t - 1) / (N + 1)
```

- `N` = number of cases (`len(case_results)`).
- `t ∈ [0, 1]` = tiebreaker, higher is better:
  ```
  t = 0.75 * (bp_ordinal / 5) + 0.25 * (1 - malformed_rate)
  bp_ordinal:  None → 5  (never breaks, best)
               very_severe → 4,  severe → 3,  moderate → 2,  mild → 1,  clean → 0
  ```
- Weights: breakpoint is the **primary** tiebreaker (it demonstrably split the top
  tier); `malformed_rate` is secondary. `bp_ordinal / 5 ∈ [0,1]`,
  `1 - malformed_rate ∈ [0,1]`, so `t ∈ [0,1]`.

**Why `(t - 1)` and not `t`:** it keeps `rank_score ≤ critical_pass_rate` with the
band `[cpr - 1/(N+1), cpr]`, so `rank_score ≤ 1.0` **by construction** (max `1.0`
exactly at `cpr = 1.0, t = 1`). This is the fix for the ceiling-clamp collision
(see below). `score` stays exactly `critical_pass_rate` — untouched;
`breakpoint_axis_level` still surfaces at top level for dashboards.

**Semantics note (routing-only):** `rank_score` is a *derived routing signal*, not
a decomposition of `score`. Unlike `critical_pass_rate`/`mean_rubric`/
`malformed_rate` it is not a "part of" `score`. Documented as routing-only so a
dashboard iterating `score_components` as a breakdown does not mis-read it.

**Correctness guarantee (the load-bearing invariant):** `critical_pass_rate` moves
in steps of `1/N`. For two agents differing by exactly one step (`k` vs `k+1`
passes), `rank_score_high − rank_score_low = 1/N + (t_h − t_l)/(N+1) ≥ 1/N −
1/(N+1) = 1/(N(N+1)) > 0`. So `rank_score` **can never reorder two agents whose
`critical_pass_rate` genuinely differs**, even when the higher-cpr agent has the
worst tiebreaker and the lower-cpr agent the best — the epsilon only orders agents
otherwise tied. Verified by a boundary property test (below). (Note: the naive
`(k + t)/(N + 1)` form does NOT satisfy this — opposed tiebreakers collapse the
`1/N` gap — which is why the additive-epsilon form is used.)

**Ceiling & floor:**
- `cpr = 1.0` (all pass) → `breakpoint = None` → `bp_ordinal = 5`; `rank_score ≤
  1.0` by construction, and two perfect agents with different `malformed_rate` get
  **different** `rank_score ≤ 1.0` → the arbiter clamp never fires → the ceiling
  tie **is** broken. (The old `cpr + t/(N+1)` form overflowed >1.0 and the clamp
  re-created the no-op at the top tier — this is the p2 fix.)
- The residual clamp collision moves to the **floor** (`cpr = 0`, `rank_score` may
  dip to `−1/(N+1)` → clamped to 0). Harmless: floor agents fail every case and are
  never routed to.
- `N = 0` → no cases → `rank_score = 0.0` (matches `pass_rate = 0.0`).

### 2. Reader — arbiter side (`db.rs::get_benchmark_score`)

Extend the single read to prefer `rank_score`:

```
SELECT score, score_components FROM benchmark_runs
WHERE agent_id = ?1 AND benchmark_id = ?2 ORDER BY ts DESC LIMIT 1
```

Parse `score_components` (JSON TEXT — column already exists, **no migration**); if
it contains a numeric `rank_score`, return that, else fall back to `score`. Result
still clamped `[0,1]`. `apply_benchmark_rerank` is unchanged — it calls
`get_benchmark_score` and its `delta = (rank_score - 0.5) * weight` now differs
between tied agents, so the re-sort breaks the tie. All still gated by the existing
`weight` A/B knob (`weight <= 0` = off; staged rollout unchanged).

Backward-compatible: rows written before this change have no `rank_score` → fall
back to `score` → identical to today.

**Residual-tie determinism (must-fix, interacts with PR #27).**
`apply_benchmark_rerank` re-sorts by confidence with `unwrap_or(Equal)`. When two
agents have *identical* `rank_score` (genuinely identical cpr + tiebreaker, or two
floor agents clamped to 0), the confidence delta is equal and the sort order is
undefined. The re-sort **must** fall through to the existing PR #27 deterministic
tie-break by `agent_id` rather than an arbitrary order. Confirm the comparator
chains to `agent_id` on `Equal`; add a test with two agents at an identical
`rank_score`.

**Mixed-producer flip-flop (rollout, p5).** `ORDER BY ts DESC LIMIT 1` takes the
latest row. If, mid-rollout, a producer that does not emit `rank_score` writes a
row *after* a `rank_score`-bearing row, the reader falls back to `score` and the
tie returns for that agent. The `ATP-emits-first` rollout order (below) avoids
this; for belt-and-suspenders the reader may instead prefer the latest row that
*has* `rank_score` (decided in the plan).

### 3. Data flow

```
run_pipe_check
  → build_report_benchmark_payload   (score_components += rank_score, bp_ordinal)
  → report_benchmark                 (contract: numeric extra components, additive)
  → benchmark_runs.score_components  (TEXT, unchanged column)
  → get_benchmark_score              (rank_score ?? score)
  → apply_benchmark_rerank(weight)   (existing A/B knob; residual tie → agent_id)
```

## Testing

**ATP (`atp/reporters/`):**
- `rank_score == critical_pass_rate` when `t == 1` (breakpoint `None` &
  `malformed_rate == 0`); `rank_score == cpr - 1/(N+1)` when `t == 0`.
- Monotonic in breakpoint: for equal `critical_pass_rate`, later breakpoint ⇒
  higher `rank_score`.
- **Boundary property (adversarial):** for any `N` and two agents whose
  `critical_pass_rate` differs by exactly `1/N`, the higher-cpr agent has the
  strictly higher `rank_score` **even when it carries the worst tiebreaker
  (`t = 0`) and the lower-cpr agent the best (`t = 1`)** — the case that breaks the
  naive `(k+t)/(N+1)` form.
- **Ceiling:** two agents both at `cpr = 1.0` with different `malformed_rate` get
  **different** `rank_score`, both `≤ 1.0` (no >1.0 overflow → arbiter clamp cannot
  re-create the tie).
- `score` (scalar) is unchanged; `bp_ordinal` and `malformed_rate` are emitted as
  numbers in `score_components`.

**arbiter (`db.rs` / `route_task.rs`):**
- `get_benchmark_score` returns `rank_score` when present in `score_components`.
- Falls back to scalar `score` when `score_components` lacks `rank_score` or is
  malformed JSON.
- `apply_benchmark_rerank` reorders two agents that tie on `critical_pass_rate`
  (0.8/0.8) but differ on breakpoint (severe vs moderate), with `weight > 0`.
- **Residual-tie determinism:** two agents with *identical* `rank_score` resolve to
  a stable order by `agent_id` (PR #27 tie-break), not input/arbitrary order.

## Rollout

Ship behind the existing re-rank `weight` (0 = off). Emit `rank_score` from ATP
first (harmless additive component), then flip the arbiter reader, then stage
`weight` up per the existing A/B procedure. No coordinated deploy required — the
additive component is inert until the reader reads it and `weight > 0`.

## References

- `report_benchmark-v1.schema.json:39-41` — `score_components` is
  `additionalProperties: {type: number}`; byte-identical across the three vendored
  copies (atp method/contract, Maestro/_cowork_output, arbiter/tests/contract), so
  numeric `rank_score`/`bp_ordinal` are valid with no schema change and no
  `payload_version` bump (`const "1.0.0"`).
- `atp/reporters/benchmark_reporter.py` — `build_report_benchmark_payload`,
  `_breakpoint`, `_AXIS_ORDER`
- `arbiter/arbiter-mcp/src/db.rs:817` — `get_benchmark_score` (reader to extend)
- `arbiter/arbiter-mcp/src/tools/route_task.rs:57` — `apply_benchmark_rerank`
  (`delta = (score - 0.5) * weight`)
- `_cowork_output/2026-07-01-reliability-orchestration-paper-eval.md` — track B
  framing; slice B (verifier-reliability) rationale and spec-runner dependency
