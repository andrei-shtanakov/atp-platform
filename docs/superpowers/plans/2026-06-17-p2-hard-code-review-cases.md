# P2 — Harder Code-Review Cases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add code-review cases hard enough that arbiter's routable agents (`claude_code`, `codex_cli`) diverge on the `code-review` benchmark — by changing the defect *class* (logic / FP-discipline / spec-violation / distractor), then empirically filtering to the cases that demonstrably separate them.

**Architecture:** New `code-review-correctness` case family graded by the existing `findings_match` checker (no grader-semantics change). A small harness tweak surfaces per-case recall/precision/fp_count for a continuous divergence metric. A separate parallel task captures `codex_cli` token/cost (early-gate scout). An empirical filter loop (n=3 → keep divergent → n=5–8 held-out) decides keepers; honest no-go is valid.

**Tech Stack:** Python 3.12, `uv`, pytest, the agent-eval-case methodology (`packages/atp-method`), `findings_match` (`atp/evaluators/findings/`), `method/run_pipe_check.py`, Ollama/Anthropic/codex/deepseek spawner shims.

**Spec:** `docs/superpowers/specs/2026-06-17-p2-hard-code-review-cases-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `method/spawners/codex_cli_shim.py` | codex_cli agent | **Task 0:** capture token usage via `codex exec --json` |
| `tests/unit/method_spawners/test_codex_cli_shim.py` + `fixtures/fake_codex.py` | codex shim tests | **Task 0:** assert tokens parsed |
| `method/run_pipe_check.py` | harness | **Task 1:** surface per-case `recall`/`precision`/`fp_count`; write `case_details_<agent>.jsonl` |
| `tests/unit/method_spawners/test_run_pipe_check.py` | harness test | **Task 1:** assert the new per-case keys |
| `method/cases/code-review/RULES-correctness.md` | rule KB reference (LOG-*/SPEC-*) | **Task 2:** create (doc) |
| `method/cases/code-review/case-code-review-correctness-*.yaml` | the 10 cases | **Tasks 2–4:** author + determinism proofs |
| `packages/atp-method/tests/test_p2_correctness_determinism.py` | determinism proofs | **Tasks 2–4:** good/bad/near-miss/malformed per case |
| `_cowork_output/r07-pipecheck/p2-filter.db` + `.../2026-06-17-p2-filter-results.md` | filter data + report | **Tasks 5–6** |

Tags follow `^[a-z0-9]+(?:_[a-z0-9]+)*$`. All cases: `task_type: review`, family `code-review-correctness`.

---

## Task 0 (PARALLEL — start now, early gate): codex_cli token/cost capture

This is a standalone task with its own DoD (*"`codex_cli` usage captured + reconciled"*). It gates how much we invest in P2 authoring (if cost already separates the routable agents, Task-4 may not need hard cases).

**Files:**
- Modify: `method/spawners/codex_cli_shim.py`
- Modify: `tests/unit/method_spawners/fixtures/fake_codex.py`, `tests/unit/method_spawners/test_codex_cli_shim.py`

- [ ] **Step 1: Write the failing test** (add to `test_codex_cli_shim.py`)

The fake must, when `--json` is passed, print a JSONL usage event to stdout in addition to writing the last-message file. Update `fixtures/fake_codex.py` to emit a usage line and the test to assert tokens land in metrics:

```python
def test_shim_captures_tokens_from_json_events() -> None:
    resp = _run_fake_codex(  # existing helper that runs the shim with CODEX_BIN=fake
        request={"task_id": "t1", "task": {"description": "review"}},
    )
    assert resp["status"] == "completed"
    assert resp["metrics"]["total_tokens"] == 1500  # 1100 in + 400 out from the fake
    assert resp["metrics"]["input_tokens"] == 1100
    assert resp["metrics"]["output_tokens"] == 400
```

In `fixtures/fake_codex.py`, when `--json` is in argv, also `print(json.dumps({"type": "token_count", "info": {"input_tokens": 1100, "output_tokens": 400}}))` to stdout before writing the message file. (Match the real event shape — confirm against a live `codex exec --json` run in Step 3.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/method_spawners/test_codex_cli_shim.py::test_shim_captures_tokens_from_json_events -q`
Expected: FAIL (shim doesn't pass `--json` / doesn't parse usage; tokens are None).

- [ ] **Step 3: Confirm the real event shape, then implement**

First, one live probe to learn the exact usage-event key: `codex exec --json --skip-git-repo-check --sandbox read-only --output-last-message /tmp/cx.txt "Reply OK"` and inspect the JSONL on stdout for the token-usage event (Codex emits a `token_count`/usage event). Then in `codex_cli_shim.py`:
- add `"--json"` to `argv` (before the prompt).
- capture `proc.stdout`, parse it line-by-line as JSON, find the usage event, extract input/output tokens.
- set `metrics.total_tokens = in+out`, `input_tokens`, `output_tokens` (None-safe: if no usage event found, leave all None — never fabricate). `cost_usd` stays None unless an event carries a cost field.

```python
def _parse_usage(stdout: str) -> tuple[int | None, int | None]:
    """Pull (input_tokens, output_tokens) from codex --json JSONL; None if absent."""
    in_tok = out_tok = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        info = ev.get("info") or ev.get("usage") or ev
        if "input_tokens" in info or "output_tokens" in info:
            in_tok = info.get("input_tokens", in_tok)
            out_tok = info.get("output_tokens", out_tok)
    return in_tok, out_tok
```
Wire it into the success path: `in_tok, out_tok = _parse_usage(proc.stdout.decode(errors="replace"))`, then `total = (in_tok or 0)+(out_tok or 0) if (in_tok is not None or out_tok is not None) else None`. Keep the existing `--output-last-message` capture for the review text.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/method_spawners/test_codex_cli_shim.py -q`
Expected: PASS (existing + new). Then a live smoke: run the harness on one code-review case with `--agents codex_cli` and confirm `tokens=` is non-zero (not 0) in the summary line.

- [ ] **Step 5: Commit**

```bash
uv run ruff format method/spawners/codex_cli_shim.py tests/unit/method_spawners/test_codex_cli_shim.py tests/unit/method_spawners/fixtures/fake_codex.py
uv run ruff check method/spawners/codex_cli_shim.py && uv run pyrefly check method/spawners/codex_cli_shim.py
git add method/spawners/codex_cli_shim.py tests/unit/method_spawners/
git commit -m "feat(R-07): capture codex_cli token usage via codex exec --json"
```

- [ ] **Step 6: EARLY GATE — cost/token comparison (informs P2 scope)**

Run claude_code vs codex_cli on the EXISTING code-review cases (already in main), n=3, into a scratch DB; compare tokens + cost + duration:

```bash
DB=/tmp/p2_costscout.db; for i in 1 2 3; do uv run python method/run_pipe_check.py --case-dir method/cases/code-review --task-type review --agents claude_code,codex_cli --db "$DB" --out-dir /tmp/costscout-$i 2>&1 | grep -E "cost=|tokens="; done
sqlite3 -header -column "$DB" "SELECT agent_id, ROUND(AVG(total_tokens),0) tok, ROUND(AVG(COALESCE(total_cost_usd,0)),4) cost, ROUND(AVG(duration_seconds),1) dur FROM benchmark_runs GROUP BY agent_id;"
```
**Decision:** record the result in the filter report (Task 6). If cost/tokens *already* materially separate the two routable agents, note that the first arbiter Task-4 A/B could ride a **cost re-rank** and flag to the user that full P2 case-authoring may be lower priority. Either way, continue (authoring Tasks 1–4 are independent), but surface this finding.

---

## Task 1: Harness — surface per-case recall/precision/fp_count

**Files:**
- Modify: `method/run_pipe_check.py` (`_grade_case`, `_run_agent`)
- Test: `tests/unit/method_spawners/test_run_pipe_check.py`

- [ ] **Step 1: Write the failing test**

Add to `test_run_pipe_check.py` a test that `_grade_case` copies the CaseVerdict continuous fields out of the critical-check details. Use a stub evaluator returning a known `EvalCheck`:

```python
import anyio
from atp.evaluators.base import EvalCheck, EvalResult
from method.run_pipe_check import _grade_case


class _StubEval:
    async def evaluate(self, test_def, response, events, assertion):
        return EvalResult(
            evaluator="stub",
            checks=[EvalCheck(
                name="critical_check", passed=True, score=1.0,
                details={"malformed": False, "recall": 0.5,
                         "precision": 0.75, "fp_count": 1},
            )],
        )


def test_grade_case_surfaces_continuous_metrics(monkeypatch):
    # minimal fake test_result with one completed run + one critical assertion
    tr = _make_fake_test_result_completed()  # helper in this test file
    base = anyio.run(_grade_case, _StubEval(), tr, "moderate", False)
    assert base["recall"] == 0.5
    assert base["precision"] == 0.75
    assert base["fp_count"] == 1
```

(If `_make_fake_test_result_completed` doesn't exist, build it from the loader: load one code-review case, wrap a completed `ATPResponse` with one `ArtifactFile` whose content is a findings JSON. Keep it in the test file.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/method_spawners/test_run_pipe_check.py::test_grade_case_surfaces_continuous_metrics -q`
Expected: FAIL (`KeyError: 'recall'`).

- [ ] **Step 3: Implement in `_grade_case`**

Add `recall`/`precision`/`fp_count` to the `base` dict defaults (None), and populate them from the critical-check details alongside `malformed`:

```python
    base: dict[str, Any] = {
        ...,                       # existing keys unchanged
        "recall": None,
        "precision": None,
        "fp_count": None,
    }
```
and in the assertion loop, where `malformed` is read:
```python
        if assertion.type == METHOD_CRITICAL_CHECK:
            res = await evaluator.evaluate(test_def, response, run.events, assertion)
            check = res.checks[0]
            base["critical_pass"] = bool(check.passed)
            d = check.details or {}
            base["malformed"] = bool(d.get("malformed", False))
            base["recall"] = d.get("recall")
            base["precision"] = d.get("precision")
            base["fp_count"] = d.get("fp_count")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/method_spawners/test_run_pipe_check.py -q`
Expected: PASS.

- [ ] **Step 5: Write per-case detail JSONL in `_run_agent`**

After `case_results` is built (before/after building `payload`), write them to a per-agent JSONL in `out_dir` so the filter can read continuous metrics without touching the `report_benchmark` contract. Add a parameter `out_dir: Path` to `_run_agent` (it's available at the call site in `_main_async`) or write from `_main_async` where `out_dir` already exists. Simplest: return `case_results` from `_run_agent` alongside the payload, and in `_main_async` write `out_dir / f"case_details_{agent_id}.jsonl"` (one JSON object per case_result). Add a test asserting the file is written with the new keys when `_main_async` runs with a fake agent (or assert the writer helper directly).

```python
def _write_case_details(path: Path, case_results: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(c) for c in case_results))
```

- [ ] **Step 6: Run + commit**

Run: `uv run pytest tests/unit/method_spawners/ -q` (expect all pass).
```bash
uv run ruff format method/run_pipe_check.py tests/unit/method_spawners/test_run_pipe_check.py
uv run ruff check method/run_pipe_check.py && uv run pyrefly check method/run_pipe_check.py
git add method/run_pipe_check.py tests/unit/method_spawners/test_run_pipe_check.py
git commit -m "feat(R-07): surface per-case recall/precision/fp_count + case_details jsonl"
```

---

## Task 2: Rule KB + first worked case (L1) + determinism-proof pattern

This establishes the authoring + proof template the remaining cases follow.

**Files:**
- Create: `method/cases/code-review/RULES-correctness.md` (reference for authors; the rules also live inline in each case's `kb-rules` artifact)
- Create: `method/cases/code-review/case-code-review-correctness-logic-offbyone-001.yaml`
- Create: `packages/atp-method/tests/test_p2_correctness_determinism.py`

- [ ] **Step 1: Write the rule KB doc**

`RULES-correctness.md`: list `LOG-001` (off-by-one/boundary), `LOG-002` (incorrect conditional/predicate), `SPEC-001` (monetary amounts are integer cents), `SPEC-002` (list endpoints paginate, limit ≤ 100), each one sentence. Note: FP cases (F*) cite no rule.

- [ ] **Step 2: Author the L1 case** (the worked example; mirrors `case-code-review-sqli-very-severe-001.yaml` structure)

```yaml
id: case-code-review-correctness-logic-offbyone-001
version: 1
family: code-review-correctness
status: active
suite_type: probe
capability: correctness
construction_axis: output_structure
axis_level: moderate
task_type: review
language: python
tags: [review, logic_error]
instruction: >
  Review the attached diff against the team coding rules (provided). Report each
  issue with: rule_id, file:line, severity, and a concrete fix. Do not invent issues.
artifacts:
  - id: diff
    type: text
    content: |
      --- ring.py ---
      +10: def keep_last(events, n):
      +11:     if len(events) >= n:          # off-by-one: drops one valid event at == n
      +12:         return events[-(n - 1):]
      +13:     return events
  - id: kb-rules
    type: text
    content: "LOG-001: boundary/size comparisons must keep exactly the intended count."
environment:
  tools: [none]
  side_effects: none
constraints:
  - cite rule_id for every issue
  - do not flag compliant lines
expected_failure_mode: >
  Agent reads keep_last as correct and misses that '>= n' plus '-(n-1)' drops one
  event when len == n (keeps n-1, not n).
grader:
  type: programmatic
  checker: findings_match
  expected_findings:
    - rule_ids: [LOG-001, off-by-one, boundary-error]
      anchor: 'return events[-(n - 1):]'
      severity: critical
  must_not_flag:
    - anchor: 'return events'
  rubric:
    - {criterion: identifies the off-by-one, weight: 0.5}
    - {criterion: fix keeps exactly n, weight: 0.5}
  critical_check: >
    The off-by-one in keep_last (drops one event at len == n) is reported.
  scoring: "Fail if the critical findings_match check fails; else weighted rubric sum."
provenance:
  author: andrei
  created: "2026-06-17"
  source: hand-authored + LOG-001
```

- [ ] **Step 3: Write the determinism proof (the TEMPLATE the rest reuse)**

Create `test_p2_correctness_determinism.py`. The proof has four checks per case: good→pass, bad→fail, near-miss-anchor (a wrong-but-overlapping anchor must NOT be accepted as the expected finding), malformed (a well-formed correct answer is not malformed). Use the checker directly:

```python
import json
from pathlib import Path
import yaml
from atp.evaluators.checkers import get_checker
from atp_method.schema import AgentEvalCase

ROOT = Path(__file__).resolve().parents[3]
CHECKER = get_checker("findings_match")
assert CHECKER is not None


def _cfg(case_path: Path) -> dict:
    case = AgentEvalCase.model_validate(yaml.safe_load(case_path.read_text()))
    return {
        "expected_findings": [f.model_dump() for f in (case.grader.expected_findings or [])],
        "must_not_flag": [m.model_dump() for m in (case.grader.must_not_flag or [])],
    }


L1 = ROOT / "method/cases/code-review/case-code-review-correctness-logic-offbyone-001.yaml"


def test_l1_good_passes() -> None:
    cfg = _cfg(L1)
    good = json.dumps([{"rule_id": "LOG-001", "anchor": "return events[-(n - 1):]",
                        "severity": "critical"}])
    v = CHECKER(cfg, good)
    assert v.critical_pass is True and v.malformed is False


def test_l1_miss_fails() -> None:
    v = CHECKER(_cfg(L1), json.dumps([]))   # found nothing
    assert v.critical_pass is False and v.malformed is False


def test_l1_near_miss_anchor_not_accepted() -> None:
    # flagging the *safe* return line must not satisfy the expected finding
    cfg = _cfg(L1)
    near = json.dumps([{"rule_id": "LOG-001", "anchor": "return events",
                        "severity": "critical"}])
    v = CHECKER(cfg, near)
    assert v.critical_pass is False  # safe-line flag is an FP, not the defect


def test_l1_wellformed_not_malformed() -> None:
    cfg = _cfg(L1)
    v = CHECKER(cfg, json.dumps([{"rule_id": "LOG-001",
                                  "anchor": "return events[-(n - 1):]",
                                  "severity": "critical"}]))
    assert v.malformed is False
```

- [ ] **Step 4: Run the proof**

Run: `uv run pytest packages/atp-method/tests/test_p2_correctness_determinism.py -q`
Expected: PASS (4). If `near_miss` fails (the safe-line flag is wrongly accepted), the anchors overlap — tighten the anchor strings until distinct (this is the anchor-collision guard from the spec).

- [ ] **Step 5: Validate the case loads + JSON-schema-valid**

Run: `uv run pytest packages/atp-method/tests/test_cases_load.py -q` (the code-review glob now includes the new case — if `test_cases_present` hard-codes a count, update it to the new total).

- [ ] **Step 6: Commit**

```bash
uv run ruff format packages/atp-method/tests/test_p2_correctness_determinism.py
git add method/cases/code-review/RULES-correctness.md method/cases/code-review/case-code-review-correctness-logic-offbyone-001.yaml packages/atp-method/tests/test_p2_correctness_determinism.py
git commit -m "feat(R-07 P2): rule KB + L1 case + determinism-proof template"
```

---

## Task 3: Author L2 + F1–F3 (logic control + FP-discipline)

**Files:** `case-code-review-correctness-{logic-predicate-001, fp-sqli-001, fp-unsafe-001, fp-floor-001}.yaml`; extend `test_p2_correctness_determinism.py`.

For each case: author the YAML per the spec sketch (§Candidate pool), then add the 4-check proof block (good/bad/near-miss/malformed) following the Task-2 template.

**Two binding authoring rules (from Task-2 review):**
1. **No answer-leak** — no diff comment may name/hint the defect (no "off-by-one"/"unsafe"/rule-id); comments describe intent only. A giveaway comment lets an agent parrot it and inflates the signal.
2. **Distinct non-substring anchors** — `expected_findings` anchor and every `must_not_flag` anchor must be mutually non-substring (the matcher uses bidirectional substring, `matcher.py:55-59`); use diff line-prefixes (`+13:     ...`) or unique sub-slices to disambiguate, and the near-miss proof check must confirm it.

- [ ] **Step 1: L2 — inverted predicate.** Diff: `def can_view(u): return u.is_member or u.is_banned` (should be `and not u.is_banned`); kb-rule `LOG-002`. `expected_findings` anchor = `return u.is_member or u.is_banned`, `rule_ids: [LOG-002, logic-error, incorrect-conditional, cwe-697]`; `must_not_flag` a benign adjacent line. Proof block.
- [ ] **Step 2: F1 — looks-like-SQLi, safe.** Diff: `q = f"SELECT * FROM orders WHERE status = '{Status.ACTIVE.value}'"` with a comment that `Status` is a server-side enum. `expected_findings: []`; `must_not_flag` **every** plausible-safe line (the f-string line + any other "scary" line in the diff). Proof: good = `[]` (no findings) → pass; bad = flagging the f-string → fail (FP); plus malformed check.
- [ ] **Step 3: F2 — looks-unsafe, guarded.** Diff: `data = pickle.loads(cache.get(key))` where a comment/adjacent code shows `cache` holds only server-serialized trusted blobs; or `subprocess.run(shlex.split(cmd))` with `cmd` a constant. `expected_findings: []`; `must_not_flag` every scary line. Proof.
- [ ] **Step 4: F3 — looks-like-a-bug, correct.** Diff: an intentional `total // count` integer floor-division with a comment "cents, floor is intended", or a deliberate early `return None` that looks like a missing branch but is correct per a documented contract. `expected_findings: []`; `must_not_flag` the suspicious line. Proof.
- [ ] **Step 5: Run proofs + case-load**

Run: `uv run pytest packages/atp-method/tests/test_p2_correctness_determinism.py packages/atp-method/tests/test_cases_load.py -q`
Expected: PASS. **FP-proof note:** for F cases, `bad` = an output that flags a `must_not_flag` line must yield `critical_pass=False`; `good` = `[]` yields `critical_pass=True`. Confirm the proof encodes precision-on-trap-lines, not total silence.

- [ ] **Step 6: Commit**

```bash
git add method/cases/code-review/case-code-review-correctness-logic-predicate-001.yaml method/cases/code-review/case-code-review-correctness-fp-*.yaml packages/atp-method/tests/test_p2_correctness_determinism.py
git commit -m "feat(R-07 P2): L2 control + F1-F3 FP-discipline cases + proofs"
```

---

## Task 4: Author S1–S2 + D1–D3 (spec-violation + distractors)

**Files:** `case-code-review-correctness-{spec-cents-001, spec-pagination-001, distractor-logic-001, distractor-spec-001, distractor-mixed-001}.yaml`; extend the proof file.

- [ ] **Step 1: S1 — cents invariant.** kb-rule `SPEC-001`. Diff introduces `total = subtotal * 1.0825` (float tax on integer cents). anchor = `total = subtotal * 1.0825`, `rule_ids: [SPEC-001, invariant-violation]`. Proof.
- [ ] **Step 2: S2 — pagination policy.** kb-rule `SPEC-002`. Diff adds `return session.query(User).all()` (unbounded list endpoint). anchor = `return session.query(User).all()`, `rule_ids: [SPEC-002, missing-pagination]`. Proof.
**Targeting (triage finding):** ≥2 of the 3 D cases bury a **cross-file / multi-hop** defect (the axis the triage located — claude_code misses it, codex catches it). **Context-fairness is mandatory:** both the taint **source and sink must be in the provided diff** (like `case-code-review-sqli-very-severe-001`: source in `views/users.py`, sink in `helpers/sql.py`) — else the case tests context availability, not detection.

- [ ] **Step 3: D1 — cross-file buried defect (primary axis).** ~30–40 line diff across 2 files of believable refactors + ONE buried cross-file taint defect (a value flowing through a helper across a file boundary into an unsafe sink), BOTH ends shown. `expected_findings` = the sink anchor; `must_not_flag` = **every** plausible-but-fine refactored line. `tags: [review, distractor]`, `capability: correctness`.
- [ ] **Step 4: D2 — second cross-file buried defect (different shape).** Same cross-file/context-fair discipline; a different taint path (e.g. value through two helpers, or a different sink class). `must_not_flag` all distractor lines.
- [ ] **Step 5: D3 — single-file buried defect (contrast).** ~30–40 line diff, ONE buried single-file L- or S-class defect (no cross-file hop) — the contrast case showing whether the separation is specific to cross-file reasoning. `must_not_flag` all distractors.
- [ ] **Step 6: Proofs** — for each D case the proof additionally checks: recall>0 only when the buried anchor is flagged; precision drops when a `must_not_flag` line is flagged (build a `bad` output that flags two distractor lines and assert `fp_count==2`, `precision<1`). This is the continuous-metric proof.

Run: `uv run pytest packages/atp-method/tests/test_p2_correctness_determinism.py packages/atp-method/tests/test_cases_load.py packages/atp-method/tests/ -q`
Expected: PASS (all method tests; ~10 new cases load + validate).

- [ ] **Step 7: Commit**

```bash
git add method/cases/code-review/case-code-review-correctness-spec-*.yaml method/cases/code-review/case-code-review-correctness-distractor-*.yaml packages/atp-method/tests/test_p2_correctness_determinism.py
git commit -m "feat(R-07 P2): S1-S2 spec-violation + D1-D3 distractor cases + proofs"
```

---

## Task 5: Empirical filter run + classify + harden + held-out

Procedural (paid runs + analysis), not TDD.

- [ ] **Step 1: First filter run (n=3)**

```bash
DB=_cowork_output/r07-pipecheck/p2-filter.db; rm -f "$DB"
AG=claude_code,codex_cli,deepseek
for i in 1 2 3; do
  uv run python method/run_pipe_check.py --case-dir method/cases/code-review \
    --task-type review --agents "$AG" --db "$DB" \
    --out-dir _cowork_output/r07-pipecheck/p2-run-$i 2>&1 | grep -E "critical_pass_rate|SKIP|Traceback" || true
done
```
The per-case continuous metrics are in `_cowork_output/r07-pipecheck/p2-run-*/case_details_*.jsonl` (recall/precision/fp_count per case per agent per run). The filter runs over `method/cases/code-review/` which includes the **seeded anchor** `case-code-review-sqli-very-severe-001` (cross-file) — it participates in the `distractor-recall`/cross-file concordance group but, per the spec, is a selection-side signal that still must reproduce on held-out to count.

- [ ] **Step 2: Compute the per-type composite per case × routable agent**

Aggregate the JSONL: for L/S/D cases composite = `0.5*recall + 0.5*precision`; for F cases `fp_score = 1 - fp_count/N_safe` (N_safe = #must_not_flag in that case). Per case, take mean±sd over the 3 runs for `claude_code` and `codex_cli`. (A short throwaway Python script reading the JSONLs; do NOT aggregate across types.)

- [ ] **Step 3: Classify each case** (routable pair):
  - **KEEP-candidate** — composite gap between `claude_code` and `codex_cli` is directionally clear.
  - **SATURATED** — both ≈ tie → mark to harden (add noise / shift class; **NOT** make the defect debatable — integrity guard).
  - **DEGENERATE** — all ≈ 0 or wild variance → mark to clarify (tighten anchors / kb-rules).

- [ ] **Step 4: One bounded harden/clarify iteration**, re-run those cases (n=3), re-classify.

- [ ] **Step 5: Held-out confirmation (n=5–8, fresh)** on the KEEP-candidate set only:

```bash
DB=_cowork_output/r07-pipecheck/p2-heldout.db; rm -f "$DB"
for i in $(seq 1 6); do
  uv run python method/run_pipe_check.py --case-dir <KEEP_SUBSET_DIR> \
    --task-type review --agents claude_code,codex_cli --db "$DB" \
    --out-dir _cowork_output/r07-pipecheck/p2-heldout-$i 2>&1 | grep critical_pass_rate || true
done
```
A case is **kept** only if, on held-out, the routable gap preserves direction AND ≥ 0.34 absolute (DoD). Apply the DoD: either ≥2 kept cases where the same agent leads the same capability (distractor-recall / fp-discipline / spec-adherence), OR a single case with gap >3·sd AND ≥0.34 AND reproduced.

- [ ] **Step 6: No commit** (data lives under `_cowork_output/`, which is outside the repo tree). Proceed to Task 6.

---

## Task 6: Ship keepers + report

- [ ] **Step 1: Write the report** `_cowork_output/status/2026-06-17-p2-filter-results.md`

Per case × agent: per-type composite mean±sd (defect blend vs `fp_score` — labeled, not summed across types), `critical_pass_rate`, `malformed_rate` (separate), `total_cost_usd` + `duration` (separate; codex_cli cost now populated from Task 0). Plus: the Task-0 early-gate cost comparison; which **capabilities** separated the frontier agents; kept/discarded list; held-out reproduction outcome; total spend. State the outcome explicitly: **A (separation found)** with the kept cases, or **B (no-go — frontier quality-equivalent)** → recommend the cost-routing pivot (Task-0 data).

- [ ] **Step 2: Keep or revert the cases**

- If outcome A: the kept cases stay in `method/cases/code-review/` (they're already committed). DELETE the discarded candidate YAMLs + their proof blocks so only validated cases ship:
  ```bash
  git rm method/cases/code-review/case-code-review-correctness-<discarded>.yaml
  # remove their proof tests; keep kept-case proofs
  git commit -m "feat(R-07 P2): keep <N> code-review cases that separate frontier agents"
  ```
- If outcome B: keep the L/F/S/D cases as a *harder probe suite* (still valid eval content) but mark in the report that none separate the routable agents; do not claim a routing signal. Commit the report reference.

- [ ] **Step 3: Finish the branch** — use `superpowers:finishing-a-development-branch` (push + PR). The PR body states the outcome (A or B), the kept cases (if any), and the Task-0 cost finding.

---

## Self-Review

**Spec coverage:**
- Defect-class pivot, ~10-case pool (2L+3F+2S+3D) → Tasks 2–4. ✓
- `findings_match` reuse, rule KB, FP precision semantics (must_not_flag every safe line) → Tasks 2–4. ✓
- Per-type continuous metric (defect blend / `fp_score = 1 − fp_count/N_safe`), no cross-type aggregation → Task 5 Step 2, Task 6. ✓
- DoD: continuous metric + held-out n=5–8 + absolute floor 0.34 + concordance keyed on capability OR reproduced >3·sd → Task 5 Step 5. ✓
- Integrity guard (no weakening unambiguity to mine separation) → Task 5 Step 3 harden note. ✓
- Determinism proofs incl. near-miss-anchor + malformed → Task 2 Step 3 template, Tasks 3–4. ✓
- Harness surfaces per-case recall/precision/fp_count → Task 1. ✓
- codex_cli cost-capture as a SEPARATE parallel task with an early gate → Task 0. ✓
- Report with malformed_rate + cost/duration separate columns → Task 6. ✓
- Honest no-go (outcome B) handling → Task 6 Step 2. ✓
- D cases share `distractor-recall` capability, vary buried CWE → Task 4 Steps 3–5. ✓

**Placeholder scan:** code steps have concrete code; case-authoring steps reference the spec sketch + the worked L1 example + the repeated 4-check proof template (good/bad/near-miss/malformed) — the pattern is fully shown in Task 2, not deferred. The `<KEEP_SUBSET_DIR>` and discarded-file names in Tasks 5–6 are run-time values (the kept set is determined empirically), which is correct, not a placeholder.

**Type consistency:** `findings_match` checker signature `(config, text) -> CaseVerdict`; `CaseVerdict.recall/precision/fp_count` used consistently in Task 1 (surfacing) and Task 5 (composite). `fp_score = 1 − fp_count/N_safe` defined once, used in Task 5/6. Case naming `case-code-review-correctness-*` consistent across Tasks 2–4.

**Out of scope (parked):** concurrency (#4), matcher severity-matching (would enable D2-prioritization), `output_contract`/json_path, arbiter-side changes, `aider`.
