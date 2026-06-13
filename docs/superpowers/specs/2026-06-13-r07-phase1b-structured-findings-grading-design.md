# R-07 Phase-1b #1 — structured findings + deterministic grading (design)

> Date: 2026-06-13. Status: design (approved in brainstorm). Scope: axis **#1** of the
> Phase-1b design review. Axes #2 (correctness family) and #4 (language) are separate
> specs that reuse the evaluator built here. #3 (linter usage) is explicitly out.

## Problem

The Phase-1 code-review eval grades the `critical_check` with an **LLM judge** reading
**prose** (`AgentEvalCaseEvaluator._evaluate_critical` always runs a synthetic `llm_eval`,
ignoring `grader.type`). The pipe-check (2026-06-13) showed the symptom: `moderate=0` on a
single run is non-deterministic — could be the agent missing the defect OR the qwen judge
mis-grading the NL check. A routing signal that an A/B will act on must be **reproducible**.

## Decisions (locked in brainstorm)

1. **Hybrid grading.** The `critical_check` (the routing-relevant hard gate) becomes
   **deterministic code**; the **rubric stays `model_graded`** for semantic review quality
   (clarity, prioritization, fix correctness) that code can't judge and that does **not**
   gate the score.
2. **Reusable evaluator (approach C).** A new `findings_match` evaluator in
   `atp/evaluators/findings/`, shared by the methodology (`agent-eval-case`) and native
   (`test_suite`) formats — not a methodology-only patch and not a native-suite-only path.
3. **Anchor + synonym matching (not line numbers).** Matching keys on a code **anchor**
   (the offending snippet) + a **synonym set** of acceptable rule ids, case-insensitive.
   Line numbers are too fragile for LLM output; rule ids vary (claude emitted
   `sql-injection / CWE-89`, not the internal `SEC-011`).

## Architecture

### 1. `findings_match` evaluator (`atp/evaluators/findings/`)
A pure core function + a thin Evaluator wrapper registered as type `findings_match`.

- **Input:** the agent's structured findings (parsed from the response artifact),
  the ground truth (`expected_findings` + `must_not_flag`), and match params.
- **Output:** `recall`, `precision`, `critical_pass`, and the matched / missed /
  false-positive lists (for diagnostics).
- **No dependency** on either suite format — both wirings call the same core.

### 2. Agent output contract (strict JSON)
The agent emits **only** a JSON array:
```json
[{"rule_id": "...", "file": "app.py", "anchor": "query = f\"SELECT", "severity": "critical", "fix": "..."}]
```
- `anchor` = a substring of the offending code (NOT a line number).
- Parsing is tolerant: strip markdown code fences before `json.loads` (mirrors the
  req-extraction checkers). Unparseable → `critical_pass=false` (see Error handling).
- The `claude_code_shim.py` `REVIEW_ENVELOPE` is updated to demand exactly this JSON
  ("output only the JSON array"). The envelope stays pinned (guardrail #1).

### 3. Ground truth (machine-readable, per case)
Today `grader.gold` is a prose string. Add a structured ground truth:
```yaml
expected_findings:
  - rule_ids: [SEC-011, sql-injection, cwe-89]   # synonyms, case-insensitive
    anchor: 'f"SELECT'                            # must appear in the matched finding's anchor
    severity: critical
must_not_flag:                                    # compliant lines = false-positive trap
  - anchor: 'cursor.execute(query, (user_id,))'
  - anchor: 'logger.debug'
```
**A finding matches an `expected_findings` entry** iff its `rule_id` ∈ `rule_ids`
(case-insensitive) **AND** the ground-truth `anchor` is a whitespace-normalized substring
of (or contains) the finding's `anchor`. **A false-positive** = a finding whose anchor
matches a `must_not_flag` anchor.

### 4. Wiring into both formats
- **Native suite:** assertion `type: findings_match`, `critical: true`, config carries
  `expected_findings` / `must_not_flag`. Works immediately (the runner already supports
  `critical` hard-gating).
- **Methodology:** extend `AgentEvalCaseEvaluator._evaluate_critical` — when
  `grader.type == "programmatic"`, call the shared `findings_match` core instead of the
  LLM judge, reading new `agent-eval-case` schema fields `grader.expected_findings` /
  `grader.must_not_flag`. The `rubric` continues through `model_graded` (hybrid).

### 5. Scoring / hard gate
- `critical_pass` = **all** `severity: critical` expected findings matched **AND** zero
  `must_not_flag` hits.
- `score_components` carries deterministic `recall` and `precision` → surfaced in the
  `report_benchmark` payload (a discriminating signal, not a flat score).
- The rubric (LLM) is a separate quality component; it does **not** affect the gate.

### 6. Error handling
- Agent findings non-JSON / unparseable after fence-stripping → `critical_pass=false`
  with message "unparseable findings — cannot verify" (honest fail, never a crash).
- A case with `grader.type: programmatic` but no `expected_findings` → config/load error.

### 7. Testing (TDD)
- Matcher unit tests: anchor hit; synonym match (`cwe-89` ↔ `SEC-011`); false-positive on
  `must_not_flag`; line-number-independence (match via anchor regardless of line); malformed
  JSON → fail; empty findings on a compliant case → pass.
- Convert the two existing code-review cases to structured + `programmatic` (rubric stays
  `model_graded`).
- Offline shim test with a structured-JSON fake `claude` output.
- All tests under `tests/` so CI's `pytest tests/` runs them (lesson from #169).

## Out of scope (this spec)
codex_cli/aider shims, the full 5-level sweep, #2 correctness family, #4 language axis,
the arbiter reader/A-B. They reuse the `findings_match` evaluator and structured contract
built here.

## Success criteria
- `findings_match` grades a structured response deterministically (same input → same
  `critical_pass`/`recall`/`precision`, zero LLM).
- Both formats (native suite + methodology `programmatic`) drive the same matcher.
- The two existing cases run hybrid: deterministic gate + LLM rubric; a real run reproduces
  the same `critical_pass` across repeats (the non-determinism the pipe-check exposed is gone
  for the gate).
