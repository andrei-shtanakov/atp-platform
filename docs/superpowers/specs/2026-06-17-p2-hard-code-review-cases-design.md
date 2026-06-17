# Spec: P2 — harder code-review cases that separate frontier agents

**Status:** approved with review revisions (brainstormed 2026-06-17; revised after two reviews)
**Context:** `_cowork_output/status/2026-06-17-r07-matrix-and-spend.md` (routable agents saturate), [[project_r07_phase1]], ADR-007.
**Scope:** `method/cases/code-review/` (new `code-review-correctness` family), a small harness tweak to surface per-case recall/precision, the empirical filter run, a findings report. No grader-semantics changes, no arbiter-side changes.

## Goal

Add code-review cases hard enough that arbiter's **routable** agents (`claude_code`, `codex_cli`) **diverge** on the `code-review` benchmark, so the (working, tested) re-rank has a gap to act on. The lever is **defect class**, not concealment.

## Why defect class, not concealment

Frontier models catch known CWEs (SQLi) at any concealment — the easy axis, which saturates. The frontier is **subtle correctness**, **false-positive discipline**, **spec adherence**. (Concurrency/TOCTOU deferred — gradeability + staging cost.)

## Definition of done (statistically guarded)

Divergence is **measured on a continuous per-case score**, not binary `critical_pass`. The continuous score is the `CaseVerdict` composite **`0.5·recall + 0.5·precision`** (precision = `1 − fp_rate`); `critical_pass` (binary `findings_match`) stays only as the gate. Rationale: at n=3 a binary rate quantizes to {0, .33, .67, 1} and "gap > 2·sd" is near-tautological (sd from min–max → sd=0 makes any gap "infinitely significant"). A continuous composite is meaningful precisely on the multi-finding D/F cases where the real signal lives.

**DoD — all three required (multiple-testing + selection guard):**
1. **Reproduced:** divergence holds on a **held-out confirmation run** (fresh seed, after the filter) — not just the run it was selected on.
2. **Concordant or strong:** either **≥2 kept cases where the same routable agent leads the same defect-class**, *or* a single case whose routable-agent composite gap is **> 3·sd** (not 2·sd) **and** reproduced.
3. **Capability, not format:** the divergence is not explained by `malformed_rate` (reported separately, see §Report).

**Honest no-go is a valid outcome (outcome B):** if nothing clears the bar after one harden/clarify pass — or only *debatable* cases separate (see Integrity Guard) — the finding is "frontier agents are quality-equivalent on these classes." Then the routing lever is **cost/latency**, not quality (see Parallel track).

## The candidate pool (~10 cases) — weighted toward where separation lives

Reviews showed single-defect L-class bugs on a 5-line diff are caught as reliably as SQLi (saturation risk); real separation lives in **distractor recall/precision (D)** and **FP-discipline (F)**. Pool reweighted accordingly. All native-style (`diff` + `kb-rules` → `findings_match`), Python, family `code-review-correctness`, `task_type: review`.

- **L (2) — saturation controls** (expected to be caught by both; included to confirm the easy axis still saturates):
  - **L1 off-by-one boundary** (`>=` vs `>` dropping/duplicating one element). `rule_ids: [LOG-001, off-by-one, boundary-error]`.
  - **L2 inverted predicate** (`is_member or is_banned` grants; `now < expiry` inverted). `rule_ids: [LOG-002, logic-error, incorrect-conditional, cwe-697]`.
- **F (3) — FP-discipline; `expected_findings: []`; `must_not_flag` *every* plausible-safe line** (pass = precision on the safe lines, **not** total silence):
  - **F1 looks-like-SQLi, safe** (`f"... '{Status.ACTIVE.value}'"`, hardcoded enum/int-cast).
  - **F2 looks-unsafe, guarded** (`pickle.loads`/`subprocess.run(shlex.split())`/`eval(literal)` on trusted/quoted input, evident in the diff).
  - **F3 looks-like-a-bug, correct** (e.g. an intentional integer floor-division with a clarifying comment, or a deliberate early-return that looks like a missing branch).
- **S (2) — spec/contract violation; anchor = violating line:**
  - **S1 invariant break.** KB `SPEC-001` "amounts are integer cents"; diff adds `price * 1.0825` / `/ 100`. `rule_ids: [SPEC-001, invariant-violation]`.
  - **S2 policy break.** KB `SPEC-002` "list endpoints paginate (≤100)"; diff adds unbounded `query.all()`. `rule_ids: [SPEC-002, missing-pagination]`.
- **D (3) — distractor / needle-in-noise; the primary separators** (continuous recall/precision):
  - **D1/D2/D3** — each a ~30–40-line diff of believable refactors with **one** real L- or S-class defect buried; `must_not_flag` lists **all** plausible-but-fine lines (so over-flagging costs precision). The three vary the defect class and the noise density. These give a continuous per-case recall/precision signal across the diff, which is what the DoD measures.

**Dropped: D2-prioritization (severity-ranking).** The matcher does not compare agent severity to expected (`matcher.py:158-167`), and severity-matching is out of scope (no grader changes) — so a "surface the critical among minors" case is ungradeable as designed. Deferred as a matcher enhancement (§Out of scope).

## Grading & determinism

- **Checker:** `findings_match` (anchor + `rule_ids` + `severity`, plus `must_not_flag`). No `output_contract`/`json_path`. Shims already emit findings JSON.
- **Continuous metric source:** the `CaseVerdict` already carries `recall`/`precision`/`fp_count` (`atp/evaluators/findings/`). The filter reads these per case. **Harness tweak (in scope):** `run_pipe_check` must surface per-case `recall`/`precision`/`fp_count` (today it emits `critical_pass`/`malformed`) — add them to the per-case output / DB row so the filter can compute the composite. This is the only code change; grader semantics are untouched.
- **FP precision semantics (F + D):** `must_not_flag` must list **every** plausible-safe line, because the matcher only penalizes a flag on a *listed* anchor — a flag on an unlisted safe line goes to `unknown_extras` with `fp=0` and still passes (`matcher.py:170-183`). Pass = no flag on any listed safe line; total silence is **not** required.
- **Rule KB** per case (`kb-rules` artifact, like SEC-011): `LOG-001/002`, `SPEC-001/002`; broad `rule_ids` synonyms (match = right line + defensible reason). F cases cite nothing.
- **Determinism bar (authoring discipline):** every case **unambiguously** a defect or **unambiguously** safe — never style opinion. Logic: one correct fix, exact buggy substring as anchor. FP: defensibly safe with the safety **visible in the diff**. Spec: violated rule **explicit in kb-rules**.
- **Determinism proof per case must include, beyond good→pass / bad→fail:**
  - a **near-miss anchor** check — anchor overlap is bidirectional substring (`matcher.py:55-59`), so in multi-line D/L/S diffs a one-line anchor can collide with another; the proof confirms the correct finding isn't mis-scored and a safe line doesn't match an expected anchor.
  - a **malformed** check — strict-global validation means one bad finding malforms the whole output (`matcher.py:78-95`); the proof confirms a well-formed correct answer is not malformed.

### Integrity guard (pre-committed)
The cases that best separate strong agents sit near the "debatable" boundary, but the determinism bar forbids debatable. The window "unambiguous **and** hard enough that one frontier misses" may be empty — **that is outcome B, not failure.** We **pre-commit NOT to relax the unambiguity bar to manufacture separation.** Concretely: the harden pass may add noise/distractors or shift defect class, but **must not** make a defect's status debatable to induce a miss. If only debatable cases separate, report outcome B; do not crank subtlety into ambiguity. (Same discipline as "don't tune the re-rank weight to force a flip.")

## Structure
- Family `code-review-correctness`, `task_type: review` → `benchmark_id: code-review` (already mapped).
- `capability`: `correctness` (L, S1, D), `safety_compliance` (F, S2).
- `axis_level` tentative; the filter refines true difficulty.
- `tags`: `review` + class tag (`logic_error`/`false_positive`/`spec_violation`/`distractor`), controlled-vocab `^[a-z0-9]+(?:_[a-z0-9]+)*$`.
- Non-gating rubric kept.

## Empirical filter loop

1. **Author** the 10 candidates + determinism proofs (good/bad/near-miss/malformed).
2. **Run** `claude_code`, `codex_cli`, `deepseek`, **n=3**, into `_cowork_output/r07-pipecheck/p2-filter.db` (`run_pipe_check`, repeated 3× — `--runs>1` grades only `runs[0]`).
3. **Classify by the routable pair** on the **continuous composite** (mean ± sd over n=3):
   - **KEEP — divergence only:** routable-agent composite gap meaningful per the DoD bar. *(No "both <1.0" branch — a low tie is "saturated harder" → harden, not keep.)*
   - **SATURATED:** both routable ≈ tie (high or low) → mark to **harden** (noise/class shift, not ambiguity).
   - **DEGENERATE:** all ≈ 0 or high-variance/ambiguous → mark to **clarify**.
4. **One bounded harden/clarify iteration**, then a **held-out confirmation run** (fresh seed) on the KEEP set — divergence must reproduce.
5. **Ship** the reproduced-divergence set into `method/cases/code-review/` + a **report** (§Report). If none reproduce → ship the report as outcome B + the cost analysis.

## Report (`_cowork_output/status/2026-06-17-p2-filter-results.md`)
Per case × agent (n=3): composite mean±sd, `critical_pass_rate`, **`malformed_rate` (separate column — so format-divergence isn't mistaken for capability)**, **`total_cost_usd` + `duration` (separate columns — the cost/latency lever under outcome B)**. Plus: which defect classes separated frontier agents; kept/discarded; held-out reproduction result; spend.

## Parallel track (recommended, cheap)
`codex_cli` cost is currently **unmeasured** (`codex exec --output-last-message` emits no usage). If the filter yields outcome B (quality ties), **cost-per-score becomes the only working re-rank lever.** A mini-task — capture `codex exec --json` usage into the shim's metrics — likely unblocks the arbiter Task-4 A/B faster and cheaper than authoring quality cases. Run it in parallel; it de-risks the no-go branch.

## Out of scope
- Concurrency/TOCTOU (#4).
- Severity-matching in the matcher (would enable D2-prioritization) — deferred enhancement.
- `output_contract`/json_path, arbiter-side changes, `aider` as a third routable agent.

## Notes
- **Signal shelf-life:** `claude_code` vs `codex_cli` divergence is version-specific; a tool update can collapse/invert it. Re-filter on version change (relevant at P5/Task-4).
