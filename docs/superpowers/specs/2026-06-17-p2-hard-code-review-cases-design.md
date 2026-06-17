# Spec: P2 — harder code-review cases that separate frontier agents

**Status:** approved (brainstormed 2026-06-17)
**Context:** `_cowork_output/status/2026-06-17-r07-matrix-and-spend.md` (the matrix showing routable agents saturate), [[project_r07_phase1]], ADR-007 (taxonomy).
**Scope:** `method/cases/code-review/` (new `code-review-correctness` family), the empirical filter run, a findings report. No new grader infra, no arbiter-side changes.

## Goal

Add code-review cases hard enough that arbiter's **routable** agents (`claude_code`, `codex_cli`) diverge on the `code-review` benchmark. Today the SQLi sweep saturates — both routable agents score 1.0 on every rung — so arbiter's (working, tested) re-rank has zero signal to act on. P2 raises the ceiling by changing the **defect class**, not the concealment.

## Why defect class, not concealment

Frontier models reliably catch known CWEs (SQLi) at any concealment — that is the *easy* axis, and it saturates. The capability frontier for strong reviewers is **subtle correctness**, **false-positive discipline**, and **spec adherence**. So the new cases plant defects in those classes. (Concurrency/TOCTOU is deferred — gradeability + staging cost.)

## Definition of done

After authoring + the empirical filter loop, **≥1 kept case where `claude_code` and `codex_cli` diverge by > 2·sd at n=3** on `critical_pass`. The `code-review` benchmark then contains a case arbiter's re-rank can act on. An honest **no-go** (nothing separates them even after the harden pass) is a valid outcome and is reported as such (→ escalate to concurrency #4 or harder cases).

## The candidate pool (~10 cases)

All are native-style code-review cases: `artifacts` = a `diff` + a `kb-rules` block; grader = `findings_match` (same path as the SQLi family). Python. New family `code-review-correctness`, `task_type: review`.

### Logic / correctness (3) — *looks right, isn't*; anchor = buggy line
- **L1 — off-by-one boundary.** A "keep last N" ring buffer / token-budget check whose comparison is off by one (`>=` vs `>`), dropping or duplicating one valid element. `rule_ids: [LOG-001, off-by-one, boundary-error]`.
- **L2 — inverted predicate.** An access/validity check that grants/serves when it must not — e.g. `if user.is_member or user.is_banned: allow()` (needs `and not banned`), or `is_expired = now < expiry` (inverted). `rule_ids: [LOG-002, logic-error, incorrect-conditional, cwe-697]`.
- **L3 — swallowed error / wrong edge default.** `except (KeyError, ValueError): return 0.0` that hides a real error and returns a valid-looking wrong number used downstream (price/discount), or mishandles empty/None. `rule_ids: [LOG-003, error-handling, swallowed-exception]`.

### False-positive discipline (2) — `expected_findings: []`; gate = `must_not_flag` the scary-but-safe line
- **F1 — looks-like-SQLi, is safe.** `f"... WHERE status = '{Status.ACTIVE.value}'"` where the interpolated value is a hardcoded enum / int-cast, not user input. Over-applying SEC-011 false-flags it.
- **F2 — looks-unsafe, is guarded.** `pickle.loads(blob)` / `subprocess.run(shlex.split(cmd))` / `eval(literal)` where the input is internally generated / trusted / quoted, evident in the diff.

### Spec / contract violation (2) — kb-rules state the contract; code violates it subtly; anchor = violating line
- **S1 — invariant break.** KB `SPEC-001`: "monetary amounts are integer cents." Diff introduces `total = price * 1.0825` / `/ 100` — silently breaks the cents invariant. `rule_ids: [SPEC-001, invariant-violation]`.
- **S2 — policy break.** KB `SPEC-002`: "list endpoints MUST paginate (limit ≤ 100)." Diff adds a list endpoint returning `query.all()` unbounded. Code works; violates policy. `rule_ids: [SPEC-002, missing-pagination]`.

### Distractor-amplified (2) — the modifier
- **D1 — needle in plausible refactor.** ~30–40-line diff of believable refactors (renames, extracted helpers, reordering) with **one** real L- or S-class defect buried; `must_not_flag` several plausible-but-fine lines. Gates recall **and** precision.
- **D2 — prioritization.** Several genuine *minor* issues + **one** critical; `expected_findings` = the critical only (minors neither required nor forbidden). Tests surfacing the critical among noise.

## Grading & determinism

- **Checker:** `findings_match` (anchor + `rule_ids` + `severity`, plus `must_not_flag`). No `output_contract`/`json_path` (that is req-extraction). The existing shims already emit findings JSON.
- **Rule KB** per case (in the `kb-rules` artifact, like SEC-011): `LOG-001/002/003`, `SPEC-001/002`. Broad `rule_ids` synonyms so the gate matches *right line + defensible reason*, not exact wording. FP cases cite nothing.
- **Precision:** defect-bearing cases (esp. D1) also set `must_not_flag` on plausible-but-fine lines, so over-flagging fails.
- **Determinism bar (authoring discipline):** every case is **unambiguously** a defect or **unambiguously** safe — never a style opinion.
  - Logic bugs: a real bug with one correct fix; exact buggy substring as anchor.
  - FP cases: the scary code is *defensibly* safe and the safety is **visible in the diff**; a competent reviewer would not flag it.
  - Spec cases: the violated rule is **explicit in kb-rules** → a contract check, not judgment.
  - Anything genuinely debatable is discarded as degenerate.
- Each authored case ships with a `findings_match` determinism proof (good→pass / bad→fail, identical across runs; FP cases: a correct no-flag passes), mirroring the existing code-review/req-extraction determinism tests.

## Structure

- Family `code-review-correctness`, `task_type: review` → rolls into `benchmark_id: code-review` (taxonomy already maps `review → code-review`).
- `capability` per case: `correctness` (L1–3, S1, D1/D2), `safety_compliance` (F1–2, S2).
- `axis_level` tentative per case (`mild`…`very_severe`); the empirical filter refines the true ordering.
- `tags`: `review` + a class tag (`logic_error` / `false_positive` / `spec_violation` / `distractor`). Tags follow the controlled-vocab pattern `^[a-z0-9]+(?:_[a-z0-9]+)*$`.
- Non-gating rubric kept (cites rule / severity / fix), like the SQLi cases.

## Empirical filter loop

1. **Author** the 10 candidates + determinism proofs.
2. **Run** `claude_code`, `codex_cli`, `deepseek` on the 10, **n=3**, into `_cowork_output/r07-pipecheck/p2-filter.db` via `run_pipe_check` (`--case-dir method/cases/code-review --agents claude_code,codex_cli,deepseek`, repeated 3× per the multi-invocation pattern, since `--runs>1` grades only `runs[0]`).
3. **Classify each case by the routable pair** (`claude_code`, `codex_cli`):
   - **KEEP** — gap between routable agents > 2·sd (divergence), *or* both routable agents stably < 1.0 (ceiling raised).
   - **DISCARD-saturated** — both routable agents = 1.0 → mark to **harden**.
   - **DISCARD-degenerate** — all agents ≈ 0.0 or high-variance/ambiguous → mark to **clarify**.
4. **One bounded iteration:** harden the saturated, clarify the degenerate, re-run once.
5. **Ship** the kept set into `method/cases/code-review/` + a **report** (`_cowork_output/status/2026-06-17-p2-filter-results.md`): which defect classes separated frontier agents, the kept/discarded breakdown, and spend.

## Out of scope
- Concurrency / TOCTOU (#4) — deferred.
- New grader infrastructure, `output_contract`, json_path.
- Arbiter-side changes (the Task-4 A/B is downstream, once a separating case exists).
- `aider` as a third routable agent (separate harness work).

## Open decisions
- **Final difficulty mix if the first pass under/over-shoots:** if >half saturate, the harden pass leans into L2/F-class subtlety; if >half are degenerate, simplify anchors and tighten kb-rules. Resolved empirically in step 4.
- **n for the filter:** n=3 (matches the established noise-guard). Bump only if a borderline case sits within 2·sd and the decision hinges on it.
