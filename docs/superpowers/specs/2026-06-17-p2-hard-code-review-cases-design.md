# Spec: P2 — harder code-review cases that separate frontier agents

**Status:** approved with review revisions (brainstormed 2026-06-17; revised after two reviews)
**Context:** `_cowork_output/status/2026-06-17-r07-matrix-and-spend.md` (routable agents saturate), [[project_r07_phase1]], ADR-007.
**Scope:** `method/cases/code-review/` (new `code-review-correctness` family), a small harness tweak to surface per-case recall/precision, the empirical filter run, a findings report. No grader-semantics changes, no arbiter-side changes.

## Goal

Add code-review cases hard enough that arbiter's **routable** agents (`claude_code`, `codex_cli`) **diverge** on the `code-review` benchmark, so the (working, tested) re-rank has a gap to act on. The lever is **defect class**, not concealment.

## Why defect class, not concealment

Frontier models catch known CWEs (SQLi) at any concealment — the easy axis, which saturates. The frontier is **subtle correctness**, **false-positive discipline**, **spec adherence**. (Concurrency/TOCTOU deferred — gradeability + staging cost.)

## Definition of done (statistically guarded)

Divergence is **measured on a continuous per-case score**, not binary `critical_pass` (which quantizes to {0,.33,.67,1} at n=3, making "gap > 2·sd" near-tautological). `critical_pass` (binary `findings_match`) stays only as the gate.

**Per-type continuous metric (do NOT aggregate across types — different ranges):**
- **Defect cases (L, S, D):** composite `0.5·recall + 0.5·precision`, read straight from `CaseVerdict.recall` / `CaseVerdict.precision` (the matcher's `precision = tp/(tp+fp)`, `matcher.py:198` — there is no `fp_rate` there; the earlier "1 − fp_rate" formula was wrong, removed).
- **FP cases (F):** `expected_findings=[]` ⇒ no positives ⇒ recall is undefined and the blend is quasi-binary. Use an FP-specific continuous score **`fp_score = 1 − fp_count / N_safe`**, where `N_safe` = number of `must_not_flag` lines in the case. This grades *how many* safe lines were flagged, not all-or-nothing.
- The real continuous signal lives in **D** (`tp∈{0,1}`, `fp∈{0..N}` → precision graduates) and **F** (`fp_count` graduates); L/S single-anchor cases are coarse 0/1 (controls).

**DoD — all three required (multiple-testing + selection guard):**
1. **Reproduced (magnitude-preserving):** divergence re-appears on a **held-out confirmation run with a fresh seed at n=5–8** (not n=3 — sd is unestimable on 3 points; the KEEP shortlist is small, so the extra runs are cheap). The held-out gap must preserve **direction AND material size** (not merely "same agent ahead"): the absolute gap floor in (2) must still hold on held-out.
2. **Concordant or strong, with an absolute floor:** either **≥2 kept cases where the same routable agent leads in the same tested *capability*** (capability = fp-discipline / distractor-recall / spec-adherence — **keyed on the skill under test, NOT the buried CWE**, so the D group can form a concordant pair), *or* a single case whose routable-agent composite gap is **> 3·sd AND ≥ 0.34 absolute** (≈ one full recall miss or ~two FPs on a D case) and reproduced per (1). Pre-registered: the 0.34 floor is fixed now, not eyeballed after.
3. **Capability, not format:** the divergence is not explained by `malformed_rate` (reported separately, see §Report).

Lean on the **concordant path** over the single-case path — it is the robust signal; `>3·sd` at n=3 is the weakest leg and only counts with the absolute floor + held-out reproduction.

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
- **D (3) — distractor / needle-in-noise; the primary separators** (continuous recall/precision). **All three share one tested capability — `distractor-recall` — the concordance key** (D1+D2+D3 form a concordant group: "agent X is consistently better at finding the needle under noise"). **Targeting (triage finding):** the live triage located the separation axis at **cross-file / multi-hop taint** (claude_code cleanly misses the existing very-severe cross-file SQLi 3/3; codex catches it). So **≥2 of the 3 D cases bury a *cross-file / multi-hop* defect** (a value tainted through a helper across a function/file boundary), aiming where the signal already bites; the 3rd can be a single-file buried defect for contrast.
  - **D1/D2/D3** — each a ~30–40-line diff of believable refactors with **one** real defect buried; `must_not_flag` lists **all** plausible-but-fine lines (so over-flagging costs precision). Continuous recall/precision across the diff — what the DoD measures.
- **Anchor (seeded, not new):** promote the existing `case-code-review-sqli-very-severe-001` (cross-file taint) into the P2 filter set as a **seeded anchor** on the `distractor-recall`/cross-file axis. It already separates the routable agents on *detection* (malformed=0), so if the new cross-file D cases agree, it contributes to concordance — cheapening the DoD from "find separation cold" to "reproduce + extend a located axis." **Discipline:** this is a *selection-side* signal (same scout/triage), NOT a closed result — it still requires held-out reproduction per the DoD. Treat it as a seeded axis, not a kept case.

**Dropped: D2-prioritization (severity-ranking).** The matcher does not compare agent severity to expected (`matcher.py:158-167`), and severity-matching is out of scope (no grader changes) — so a "surface the critical among minors" case is ungradeable as designed. Deferred as a matcher enhancement (§Out of scope).

## Grading & determinism

- **Checker:** `findings_match` (anchor + `rule_ids` + `severity`, plus `must_not_flag`). No `output_contract`/`json_path`. Shims already emit findings JSON.
- **Continuous metric source:** the `CaseVerdict` already carries `recall`/`precision`/`fp_count` (`atp/evaluators/findings/`). The filter reads these per case. **Harness tweak (in scope):** `run_pipe_check` must surface per-case `recall`/`precision`/`fp_count` (today it emits `critical_pass`/`malformed`) — add them to the per-case output / DB row so the filter can compute the composite. This is the only code change; grader semantics are untouched.
- **FP precision semantics (F + D):** `must_not_flag` must list **every** plausible-safe line, because the matcher only penalizes a flag on a *listed* anchor — a flag on an unlisted safe line goes to `unknown_extras` with `fp=0` and still passes (`matcher.py:170-183`). Pass = no flag on any listed safe line; total silence is **not** required.
- **Rule KB** per case (`kb-rules` artifact, like SEC-011): `LOG-001/002`, `SPEC-001/002`; broad `rule_ids` synonyms (match = right line + defensible reason). F cases cite nothing.
- **Determinism bar (authoring discipline):** every case **unambiguously** a defect or **unambiguously** safe — never style opinion. Logic: one correct fix, exact buggy substring as anchor. FP: defensibly safe with the safety **visible in the diff**. Spec: violated rule **explicit in kb-rules**.
- **Context-fairness (cross-file/multi-hop cases — the confound to guard):** BOTH the taint **source and sink must be visible** in the diff the agent receives. If the source lives in an unshown file, "agent missed the cross-file defect" measures *context availability*, not detection — and a flag is a guess. Every cross-file D case + the anchor must show both ends, like `case-code-review-sqli-very-severe-001` (source in `views/users.py`, sink in `helpers/sql.py`, both in the diff — verified).
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
4. **One bounded harden/clarify iteration**, then a **held-out confirmation run at n=5–8 with a fresh seed** on the (small) KEEP set — divergence must reproduce magnitude-preserving (DoD §1). Concordance is keyed on the tested capability (D = distractor-recall, F = fp-discipline, S = spec-adherence), not the buried CWE.
5. **Ship** the reproduced-divergence set into `method/cases/code-review/` + a **report** (§Report). If none reproduce → ship the report as outcome B + the cost analysis.

## Report (`_cowork_output/status/2026-06-17-p2-filter-results.md`)
Per case × agent (n=3): the per-type composite mean±sd (defect blend for L/S/D, `fp_score` for F — **do not sum composites across types; they live on different ranges**), `critical_pass_rate`, **`malformed_rate` (separate column — so format-divergence isn't mistaken for capability)**, **`total_cost_usd` + `duration` (separate columns — the cost/latency lever; `codex_cli` cost present once the cost-capture task lands)**. Plus: which *capabilities* (distractor-recall / fp-discipline / spec-adherence) separated frontier agents; kept/discarded; held-out (n=5–8) reproduction result; spend.

## Separate task (NOT part of P2): codex_cli cost capture — start now, gate P2 on its result

`codex_cli` cost is **unmeasured** (`codex exec --output-last-message` emits no usage). This is its **own task with its own DoD** — *"`codex_cli` token/cost captured (via `codex exec --json` usage events) and reconciled against a known-cost run"* — and it is **pulled out of P2** because:
- **Value is unconditional, not a hedge:** budget-aware re-rank needs `codex_cli` cost even under outcome A (both at 1.0 ⇒ cost is the *only* differentiator). It serves arbiter's cost features regardless of P2's result, so it must not be gated on case authoring.
- **Data dependency:** the P2 report's `total_cost_usd` column for `codex_cli` is empty today; the capture must land **before** the filter-run for the "cost/latency lever" column to mean anything.
- **It may be the shorter path to Task-4:** if cost *already* separates `claude_code` vs `codex_cli`, the first Task-4 A/B can be built on a **cost re-rank with no hard cases at all** — potentially making most of P2 unnecessary.

**Sequencing:** start cost-capture **now, in parallel**; **glance at its result before fully investing in authoring all 10 cases** (cheap-scout-first). P2 references it but P2's ship-gate does not depend on it. The writing-plans output will make cost-capture Task 0 (parallel) with this early checkpoint.

## Out of scope
- Concurrency/TOCTOU (#4).
- Severity-matching in the matcher (would enable D2-prioritization) — deferred enhancement.
- `output_contract`/json_path, arbiter-side changes, `aider` as a third routable agent.

## Notes
- **Signal shelf-life:** `claude_code` vs `codex_cli` divergence is version-specific; a tool update can collapse/invert it. Re-filter on version change (relevant at P5/Task-4).
