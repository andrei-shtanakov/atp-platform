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

## Design

### 1. `rank_score` — ATP side (`benchmark_reporter.py`)

Add a numeric `rank_score` to `score_components` in `build_report_benchmark_payload`:

```
rank_score = critical_pass_rate + t / (N + 1)
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
  `1 - malformed_rate ∈ [0,1]`, so `t ∈ [0,1]` (max `1.0` at breakpoint `None` +
  `malformed_rate == 0`).

`score` stays exactly `critical_pass_rate` — untouched. `breakpoint_axis_level`
still surfaces at top level (unchanged) for dashboards.

**Correctness guarantee (the load-bearing invariant):** `critical_pass_rate` moves
in steps of `1/N`. The tiebreaker is bounded `t/(N+1) ≤ 1/(N+1) < 1/N` (max at
`t = 1.0`). Therefore
`rank_score` **can never reorder two agents whose `critical_pass_rate` genuinely
differs** — the epsilon only orders agents that are otherwise tied. This is
verified by a boundary property test (below).

Edge cases:
- `critical_pass_rate = 1.0` (all pass) → `breakpoint = None` → `bp_ordinal = 5`;
  `rank_score` may exceed `1.0` by up to `<1/(N+1)`. arbiter clamps to `[0,1]`, and
  a perfect agent scoring at the ceiling is correct. (Optionally cap at `1.0` in
  ATP — decided in the plan; clamp on the arbiter side already covers routing.)
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

### 3. Data flow

```
run_pipe_check
  → build_report_benchmark_payload   (score_components.rank_score added)
  → report_benchmark                 (contract: numeric extra component, additive)
  → benchmark_runs.score_components  (TEXT, unchanged column)
  → get_benchmark_score              (rank_score ?? score)
  → apply_benchmark_rerank(weight)   (existing A/B knob)
```

## Testing

**ATP (`atp/reporters/`):**
- `rank_score == critical_pass_rate` when `t == 0` (breakpoint `clean` &
  `malformed_rate == 1.0`).
- Monotonic in breakpoint: for equal `critical_pass_rate`, later breakpoint ⇒
  higher `rank_score`.
- **Boundary property:** for any `N` and any two component sets where
  `critical_pass_rate` differs by exactly `1/N`, the agent with the higher
  `critical_pass_rate` always has the higher `rank_score` (epsilon never crosses the
  gap).
- `score` (scalar) is unchanged by this addition.

**arbiter (`db.rs` / `route_task.rs`):**
- `get_benchmark_score` returns `rank_score` when present in `score_components`.
- Falls back to scalar `score` when `score_components` lacks `rank_score` or is
  malformed.
- `apply_benchmark_rerank` reorders two agents that tie on `critical_pass_rate`
  (0.8/0.8) but differ on breakpoint (severe vs moderate), with `weight > 0`.

## Rollout

Ship behind the existing re-rank `weight` (0 = off). Emit `rank_score` from ATP
first (harmless additive component), then flip the arbiter reader, then stage
`weight` up per the existing A/B procedure. No coordinated deploy required — the
additive component is inert until the reader reads it and `weight > 0`.

## References

- `atp/reporters/benchmark_reporter.py` — `build_report_benchmark_payload`,
  `_breakpoint`, `_AXIS_ORDER`
- `arbiter/arbiter-mcp/src/db.rs:817` — `get_benchmark_score` (reader to extend)
- `arbiter/arbiter-mcp/src/tools/route_task.rs:57` — `apply_benchmark_rerank`
  (`delta = (score - 0.5) * weight`)
- `_cowork_output/2026-07-01-reliability-orchestration-paper-eval.md` — track B
  framing; slice B (verifier-reliability) rationale and spec-runner dependency
