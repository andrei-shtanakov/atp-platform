# R-07 rank_score tiebreaker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break the R-07 re-rank no-op by emitting a numeric `rank_score` tiebreaker (from the signal ATP already computes) and having arbiter's re-rank consume it, with deterministic residual-tie ordering.

**Architecture:** ATP's `build_report_benchmark_payload` adds two numeric keys to `score_components` — `rank_score` (interim combined signal) and `bp_ordinal` (raw, for the slice-B repayment). arbiter's `get_benchmark_score` prefers `rank_score` over the scalar `score`; `apply_benchmark_rerank`'s sort gains a deterministic `agent_id` tiebreak. No schema change, no DB migration, gated by the existing re-rank `weight`.

**Tech Stack:** Python 3.12 (ATP, `uv`/`ruff`/`pyrefly`), Rust (arbiter, `cargo`, `rusqlite`, `serde_json`).

**Spec:** `docs/superpowers/specs/2026-07-01-r07-rank-score-tiebreaker-design.md`

## Global Constraints

- ATP: Python 3.12, line length 88, type hints required, `uv run ruff format .` + `uv run ruff check .` + `uv run pyrefly check` clean before commit.
- `score_components` values MUST all be numbers (schema `additionalProperties: {type: number}`). No `payload_version` bump (`const "1.0.0"`). No arbiter DB migration (`score_components` column already `TEXT`).
- `score` (scalar) stays exactly `critical_pass_rate` — never modified.
- Tiebreaker invariant: `rank_score` MUST NOT reorder two agents whose `critical_pass_rate` genuinely differs (formula `cpr + (t-1)/(N+1)`, `t ∈ [0,1]`).
- Branches: ATP tasks on `r07/rank-score-tiebreaker` (already checked out). arbiter tasks on a fresh branch `git checkout -b r07/rank-score-reader` off arbiter `main`.
- TDD: failing test → run-fail → implement → run-pass → commit, one commit per task.

---

## File Structure

- **Modify** `atp/reporters/benchmark_reporter.py` — add `_bp_ordinal` + `_rank_score` helpers; emit `rank_score` + `bp_ordinal` in `score_components`. (Task 1)
- **Create** `tests/unit/reporters/test_rank_score.py` — pure-function + payload-emission tests. (Task 1)
- **Modify** `arbiter/arbiter-mcp/src/db.rs` — `get_benchmark_score` prefers `rank_score`; 2 tests in the existing `#[cfg(test)]` module. (Task 2)
- **Modify** `arbiter/arbiter-mcp/src/tools/route_task.rs` — add `agent_id` tiebreak to the re-rank sort; 1 test. (Task 3)

---

### Task 1: ATP — emit `rank_score` + `bp_ordinal`

**Files:**
- Modify: `atp/reporters/benchmark_reporter.py` (helpers after `_breakpoint`; `score_components` dict in `build_report_benchmark_payload`)
- Test: `tests/unit/reporters/test_rank_score.py` (create)

**Interfaces:**
- Consumes: nothing new (existing `_breakpoint`, `_AXIS_ORDER`, `case_results`).
- Produces: `_bp_ordinal(bp: str | None) -> int`; `_rank_score(pass_rate: float, bp_ordinal: int, malformed_rate: float, n: int) -> float`; `score_components` now contains numeric `rank_score` and `bp_ordinal`. arbiter (Task 2) reads `score_components.rank_score`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/reporters/test_rank_score.py`:

```python
"""rank_score tiebreaker (R-07 track B slice A) — pure math + payload emission."""

from atp.reporters.benchmark_reporter import (
    _bp_ordinal,
    _rank_score,
    build_report_benchmark_payload,
)


def test_bp_ordinal_none_is_best_and_levels_map_to_index() -> None:
    assert _bp_ordinal(None) == 5  # never breaks = best
    assert _bp_ordinal("clean") == 0
    assert _bp_ordinal("moderate") == 2
    assert _bp_ordinal("severe") == 3
    assert _bp_ordinal("very_severe") == 4
    assert _bp_ordinal("bogus") == 0  # unknown level = worst


def test_rank_score_equals_cpr_at_max_tiebreaker() -> None:
    # t == 1 (bp None + malformed 0), which only happens at the cpr==1.0 ceiling
    assert _rank_score(1.0, 5, 0.0, 5) == 1.0


def test_rank_score_penalizes_below_cpr_at_min_tiebreaker() -> None:
    # t == 0 → cpr - 1/(N+1)
    assert _rank_score(0.8, 0, 1.0, 5) == round(0.8 - 1 / 6, 6)


def test_rank_score_monotonic_in_breakpoint() -> None:
    # equal cpr, later breakpoint ⇒ higher rank_score
    assert _rank_score(0.8, 3, 0.0, 5) > _rank_score(0.8, 2, 0.0, 5)


def test_rank_score_never_crosses_real_gap_adversarial() -> None:
    # lower cpr + best tiebreaker vs higher cpr + worst tiebreaker: higher wins
    low = _rank_score(0.6, 5, 0.0, 5)   # cpr 3/5, t=1 → 0.6
    high = _rank_score(0.8, 0, 1.0, 5)  # cpr 4/5, t=0 → 0.633333
    assert high > low


def test_rank_score_breaks_ceiling_tie_within_bounds() -> None:
    a = _rank_score(1.0, 5, 0.0, 5)  # malformed 0 → 1.0
    b = _rank_score(1.0, 5, 0.2, 5)  # malformed 0.2 → < 1.0
    assert a != b
    assert a <= 1.0 and b <= 1.0


def test_rank_score_zero_cases() -> None:
    assert _rank_score(0.0, 5, 0.0, 0) == 0.0


def test_payload_emits_rank_score_and_bp_ordinal_without_touching_score() -> None:
    cases = [
        {
            "axis_level": "clean", "critical_pass": True, "malformed": False,
            "rubric_score": 1.0, "tokens": 1, "cost_usd": 0.0,
            "duration_seconds": 0.1, "error_class": None,
        },
        {
            "axis_level": "moderate", "critical_pass": False, "malformed": False,
            "rubric_score": 0.0, "tokens": 1, "cost_usd": 0.0,
            "duration_seconds": 0.1, "error_class": None,
        },
    ]
    p = build_report_benchmark_payload(
        run_id="r", benchmark_id="code-review", agent_id="x@m",
        ts="2026-07-01T00:00:00Z", case_results=cases,
    )
    assert p["score"] == 0.5  # critical_pass_rate unchanged
    sc = p["score_components"]
    assert sc["bp_ordinal"] == 2  # breaks at moderate (index 2)
    assert isinstance(sc["rank_score"], float)
    assert sc["rank_score"] <= p["score"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/reporters/test_rank_score.py -v`
Expected: FAIL — `ImportError: cannot import name '_bp_ordinal'`.

- [ ] **Step 3: Add the helpers**

In `atp/reporters/benchmark_reporter.py`, immediately after the `_breakpoint` function, add:

```python
def _bp_ordinal(bp: str | None) -> int:
    """Numeric breakpoint: higher = holds longer = better. ``None`` (never
    breaks) sorts above every real level; an unknown level maps to 0 (worst)."""
    if bp is None:
        return len(_AXIS_ORDER)  # 5
    return _AXIS_ORDER.index(bp) if bp in _AXIS_ORDER else 0


def _rank_score(
    pass_rate: float, bp_ordinal: int, malformed_rate: float, n: int
) -> float:
    """critical_pass_rate plus a sub-1/N tiebreaker (breakpoint + malformed).

    ``rank_score = cpr + (t - 1)/(N + 1)`` with ``t ∈ [0, 1]``: bounded so it
    can never cross a genuine 1/N critical-pass gap, and ``<= pass_rate <= 1.0``
    by construction (fixes the ceiling clamp). See the design spec.
    """
    if n == 0:
        return 0.0
    t = 0.75 * (bp_ordinal / len(_AXIS_ORDER)) + 0.25 * (1.0 - malformed_rate)
    return round(pass_rate + (t - 1.0) / (n + 1), 6)
```

- [ ] **Step 4: Emit the new components**

In `build_report_benchmark_payload`, the block computing `bp` currently reads
`bp = _breakpoint(case_results)`. Replace the `score_components` dict and the
`bp` computation so it becomes:

```python
    bp = _breakpoint(case_results)
    bp_ordinal = _bp_ordinal(bp)
    rank_score = _rank_score(pass_rate, bp_ordinal, malformed_rate, n)
    payload: dict[str, Any] = {
        "payload_version": PAYLOAD_VERSION,
        "run_id": run_id,
        "benchmark_id": benchmark_id,
        "agent_id": agent_id,
        "ts": ts,
        "score": pass_rate,
        # score_components values must all be numbers (schema constraint).
        # rank_score/bp_ordinal are routing-only signals (NOT a breakdown of
        # score): rank_score is the interim combined tiebreaker the arbiter
        # reader consumes; bp_ordinal is the raw signal for the slice-B repayment.
        "score_components": {
            "critical_pass_rate": pass_rate,
            "mean_rubric": mean_rubric,
            "malformed_rate": malformed_rate,
            "rank_score": rank_score,
            "bp_ordinal": bp_ordinal,
        },
```

Leave the rest of the payload (`total_tokens` … `per_task_truncated`) and the
`if bp is not None: payload["breakpoint_axis_level"] = bp` tail unchanged.

- [ ] **Step 5: Run tests + lint + types**

Run: `uv run pytest tests/unit/reporters/test_rank_score.py -v`
Expected: PASS (8 passed).
Run: `uv run pytest tests/unit/reporters/ -q`
Expected: PASS (existing reporter tests still green — regression guard).
Run: `uv run ruff format atp/reporters/benchmark_reporter.py tests/unit/reporters/test_rank_score.py && uv run ruff check atp/reporters/benchmark_reporter.py tests/unit/reporters/test_rank_score.py`
Expected: `All checks passed!`
Run: `uv run pyrefly check atp/reporters/benchmark_reporter.py`
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add atp/reporters/benchmark_reporter.py tests/unit/reporters/test_rank_score.py
git commit -m "feat(R-07): emit rank_score + bp_ordinal tiebreaker in report_benchmark"
```

---

### Task 2: arbiter — `get_benchmark_score` prefers `rank_score`

**Files:**
- Modify: `arbiter/arbiter-mcp/src/db.rs` (`get_benchmark_score` body ~line 817; 2 tests in the `#[cfg(test)]` module near `get_benchmark_score_returns_latest_scoped_by_benchmark`)

**Interfaces:**
- Consumes: `score_components` JSON written by Task 1 (`{"...","rank_score":<number>,...}`).
- Produces: `get_benchmark_score` returns `rank_score` when present & numeric, else the scalar `score`; result still clamped `[0,1]`. Signature unchanged: `pub fn get_benchmark_score(&self, agent_id: &str, benchmark_id: &str) -> Result<Option<f64>>`.

> Setup: `cd arbiter && git checkout main && git checkout -b r07/rank-score-reader`

- [ ] **Step 1: Write the failing tests**

In `arbiter/arbiter-mcp/src/db.rs`, inside the `#[cfg(test)] mod tests` block
(next to `get_benchmark_score_returns_latest_scoped_by_benchmark`), add:

```rust
    #[test]
    fn get_benchmark_score_prefers_rank_score_over_scalar() {
        let db = setup_db();
        let row = |run_id: &'static str, sc: &'static str, score: f64| BenchmarkRunInput {
            run_id,
            payload_version: "1.0.0",
            benchmark_id: "code-review",
            agent_id: "claude_code@claude-sonnet-4-6",
            ts: "2026-06-13T00:00:00Z",
            score,
            score_components: sc,
            total_tokens: None,
            total_cost_usd: None,
            duration_seconds: 0.0,
            per_task: "[]",
            per_task_total_count: 0,
            per_task_truncated: 0,
        };
        db.insert_benchmark_run(&row("r1", r#"{"rank_score":0.63}"#, 0.80))
            .unwrap();
        assert_eq!(
            db.get_benchmark_score("claude_code@claude-sonnet-4-6", "code-review")
                .unwrap(),
            Some(0.63),
            "rank_score is preferred over the scalar score"
        );
    }

    #[test]
    fn get_benchmark_score_falls_back_without_rank_score() {
        let db = setup_db();
        let row = |run_id: &'static str, sc: &'static str, score: f64| BenchmarkRunInput {
            run_id,
            payload_version: "1.0.0",
            benchmark_id: "code-review",
            agent_id: "claude_code@claude-sonnet-4-6",
            ts: "2026-06-13T00:00:00Z",
            score,
            score_components: sc,
            total_tokens: None,
            total_cost_usd: None,
            duration_seconds: 0.0,
            per_task: "[]",
            per_task_total_count: 0,
            per_task_truncated: 0,
        };
        // no rank_score key -> scalar score; malformed JSON -> scalar (no panic)
        db.insert_benchmark_run(&row("r1", "{}", 0.80)).unwrap();
        db.insert_benchmark_run(&row("r2", "not json", 0.80)).unwrap();
        assert_eq!(
            db.get_benchmark_score("claude_code@claude-sonnet-4-6", "code-review")
                .unwrap(),
            Some(0.80)
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd arbiter && cargo test -p arbiter-mcp get_benchmark_score_prefers_rank_score_over_scalar`
Expected: FAIL — assertion `Some(0.80)` (scalar) != `Some(0.63)` (rank_score not yet read).

- [ ] **Step 3: Update the reader**

Replace the body of `get_benchmark_score` in `arbiter/arbiter-mcp/src/db.rs` with:

```rust
    pub fn get_benchmark_score(&self, agent_id: &str, benchmark_id: &str) -> Result<Option<f64>> {
        let row = self
            .conn
            .query_row(
                "SELECT score, score_components FROM benchmark_runs \
                 WHERE agent_id = ?1 AND benchmark_id = ?2 \
                 ORDER BY ts DESC LIMIT 1",
                params![agent_id, benchmark_id],
                |r| Ok((r.get::<_, f64>(0)?, r.get::<_, String>(1)?)),
            )
            .optional()
            .context("Failed to read benchmark score")?;
        // Prefer the routing-only `rank_score` tiebreaker (R-07) when present and
        // numeric; fall back to the scalar critical_pass_rate. Malformed JSON or a
        // missing key falls back with no panic.
        Ok(row.map(|(score, components)| {
            serde_json::from_str::<serde_json::Value>(&components)
                .ok()
                .and_then(|v| v.get("rank_score").and_then(serde_json::Value::as_f64))
                .unwrap_or(score)
                .clamp(0.0, 1.0)
        }))
    }
```

If `serde_json` is not already imported in `db.rs`, the fully-qualified
`serde_json::` paths above need no `use`. `serde_json` is already a workspace
dependency (used by `tools/report_benchmark.rs`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd arbiter && cargo test -p arbiter-mcp get_benchmark_score`
Expected: PASS — the two new tests plus `get_benchmark_score_returns_latest_scoped_by_benchmark` (regression: `score_components: "{}"` falls back to scalar, unchanged).

- [ ] **Step 5: Commit**

```bash
cd arbiter
git add arbiter-mcp/src/db.rs
git commit -m "feat(R-07): get_benchmark_score prefers rank_score, falls back to score"
```

---

### Task 3: arbiter — deterministic residual-tie ordering

**Files:**
- Modify: `arbiter/arbiter-mcp/src/tools/route_task.rs` (`apply_benchmark_rerank` sort ~line 78; 1 test in the `#[cfg(test)]` module near `benchmark_weight_reranks_review_by_per_agent_score`)

**Interfaces:**
- Consumes: `apply_benchmark_rerank(ranked: &mut [(String, PredictionResult)], task_type, db, weight)`; `seed_bench`, `PredictionResult { class, confidence, path }` test helpers already in the module.
- Produces: re-rank sort is total and deterministic — equal adjusted confidence resolves by `agent_id` ascending.

- [ ] **Step 1: Write the failing test**

In `arbiter/arbiter-mcp/src/tools/route_task.rs`, inside the `#[cfg(test)] mod`
block (next to `benchmark_weight_reranks_review_by_per_agent_score`), add:

```rust
    #[test]
    fn rerank_breaks_residual_ties_by_agent_id() {
        // Identical code-review score AND identical base confidence => identical
        // adjusted confidence => the sort must be deterministic by agent_id.
        let db = Database::open_in_memory().unwrap();
        db.migrate().unwrap();
        seed_bench(&db, "a", "zzz@m", "code-review", 0.80);
        seed_bench(&db, "b", "aaa@m", "code-review", 0.80);

        let mk = |conf: f64| PredictionResult {
            class: 0,
            confidence: conf,
            path: vec![],
        };
        // input order deliberately puts zzz first
        let mut ranked = vec![
            ("zzz@m".to_string(), mk(0.50)),
            ("aaa@m".to_string(), mk(0.50)),
        ];
        apply_benchmark_rerank(&mut ranked, &TaskType::Review, &db, 0.15).unwrap();
        assert_eq!(ranked[0].0, "aaa@m", "residual tie resolves by agent_id ascending");
        assert_eq!(ranked[1].0, "zzz@m");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd arbiter && cargo test -p arbiter-mcp rerank_breaks_residual_ties_by_agent_id`
Expected: FAIL — stable sort preserves input order, so `ranked[0].0 == "zzz@m"`.

- [ ] **Step 3: Add the agent_id tiebreak to the sort**

In `apply_benchmark_rerank`, replace the `ranked.sort_by(...)` call with:

```rust
    ranked.sort_by(|a, b| {
        b.1.confidence
            .partial_cmp(&a.1.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
```

> Confirm `a.0.cmp(&b.0)` (ascending `agent_id`) matches the direction of the
> existing PR #27 deterministic tie-break elsewhere in routing; if PR #27 orders
> descending, use `b.0.cmp(&a.0)` instead and update the test's expected order.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd arbiter && cargo test -p arbiter-mcp rerank`
Expected: PASS — new test plus `benchmark_weight_reranks_review_by_per_agent_score` and `benchmark_rerank_is_noop_without_mapping_or_weight` (regression: those have non-equal confidences, so the tiebreak never fires).

- [ ] **Step 5: Commit**

```bash
cd arbiter
git add arbiter-mcp/src/tools/route_task.rs
git commit -m "fix(R-07): deterministic agent_id tie-break in benchmark re-rank sort"
```

---

## Self-Review

**Spec coverage:**
- §1 `rank_score` (ATP) + raw `bp_ordinal` emission → Task 1. ✅
- Correctness invariant + ceiling fix `(t-1)/(N+1)` → Task 1 helper + adversarial/ceiling tests. ✅
- §2 reader prefers `rank_score`, fallback, malformed-safe, no migration → Task 2. ✅
- Residual-tie determinism (p3) → Task 3. ✅
- Backward-compat (rows without `rank_score` → fallback) → Task 2 `get_benchmark_score_falls_back_without_rank_score` + existing `"{}"` tests. ✅
- Semantics/routing-only note (p4) → Task 1 Step 4 comment. ✅
- Rollout flip-flop (p5) → operational (spec §Rollout); no code beyond ATP-first ordering. Not a task — noted here intentionally.
- Slice B / policy debt → spec "Known debt"; out of scope here (raw `bp_ordinal` already emitted in Task 1 to enable it). ✅

**Placeholder scan:** none — all steps carry exact code/commands.

**Type consistency:** `_bp_ordinal`/`_rank_score` signatures match between Task 1 helper and tests; `get_benchmark_score` signature unchanged; `PredictionResult { class, confidence, path }` matches `arbiter-core/src/types.rs:246`; `seed_bench`/`BenchmarkRunInput` field lists copied verbatim from the existing test module.
