# R-07 Phase-1b #1 — structured findings + findings_match grading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the code-review `critical_check` deterministic — a reusable `findings_match` evaluator grades structured agent findings against per-case ground truth (anchor + rule-synonym matching), gating the score by code while the LLM rubric stays for quality only.

**Architecture:** A pure matcher core (`atp/evaluators/findings/matcher.py`) + an `Evaluator` wrapper registered as type `findings_match`, callable from native suites (assertion `type: findings_match`) AND the methodology (`AgentEvalCaseEvaluator._evaluate_critical` dispatches to it when `grader.type == "programmatic"`). The agent emits strict JSON findings; ground truth lives in the case/assertion config.

**Tech Stack:** Python 3.12, pydantic, pytest (anyio), the atp evaluator framework, atp-method plugin.

**Spec:** `docs/superpowers/specs/2026-06-13-r07-phase1b-structured-findings-grading-design.md`

---

## File Structure

- `atp/evaluators/findings/__init__.py` — exports `match_findings`, `parse_findings`, `MatchResult`, `FindingsMatchEvaluator`.
- `atp/evaluators/findings/matcher.py` — pure core: `parse_findings`, `match_findings`, `MatchResult` (no ATP deps).
- `atp/evaluators/findings/evaluator.py` — `FindingsMatchEvaluator(Evaluator)` wrapper (reads response, calls core, returns EvalResult).
- `atp/evaluators/registry.py` — register `findings_match`.
- `packages/atp-method/atp_method/schema.py` — add `expected_findings`/`must_not_flag` to `Grader`.
- `packages/atp-method/atp_method/loader.py` — pass the new ground-truth fields into the critical assertion config.
- `packages/atp-method/atp_method/evaluators/case_evaluator.py` — programmatic dispatch in `_evaluate_critical`.
- `method/spawners/claude_code_shim.py` — `REVIEW_ENVELOPE` demands strict JSON findings.
- Tests under `tests/` (CI-visible): `tests/unit/evaluators/findings/…`.

---

## Task 1: Pure matcher core (`matcher.py`)

**Files:**
- Create: `atp/evaluators/findings/__init__.py`
- Create: `atp/evaluators/findings/matcher.py`
- Create: `tests/unit/evaluators/findings/__init__.py`
- Create: `tests/unit/evaluators/findings/test_matcher.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/evaluators/findings/test_matcher.py`:
```python
"""Tests for the deterministic findings matcher (R-07 Phase-1b #1)."""
from atp.evaluators.findings.matcher import match_findings, parse_findings

EXPECTED = [
    {"rule_ids": ["SEC-011", "sql-injection", "cwe-89"], "anchor": 'f"SELECT', "severity": "critical"}
]
MUST_NOT = [{"anchor": "cursor.execute(query, (user_id,))"}, {"anchor": "logger.debug"}]


def test_anchor_hit_with_synonym_ruleid() -> None:
    # agent named it "sql-injection / CWE-89", not the internal SEC-011
    findings = [{"rule_id": "cwe-89", "file": "app.py", "anchor": 'query = f"SELECT * FROM users'}]
    r = match_findings(findings, EXPECTED, MUST_NOT)
    assert r.critical_pass is True
    assert r.recall == 1.0
    assert r.false_positives == []


def test_line_number_independence() -> None:
    # no line numbers anywhere; match is purely by anchor + rule synonym
    findings = [{"rule_id": "SQL-Injection", "anchor": 'x = f"SELECT 1"'}]
    r = match_findings(findings, EXPECTED, MUST_NOT)
    assert r.critical_pass is True


def test_false_positive_on_compliant_line_fails_gate() -> None:
    findings = [
        {"rule_id": "sql-injection", "anchor": 'f"SELECT'},
        {"rule_id": "SEC-011", "anchor": "logger.debug('x')"},  # flags a must_not_flag line
    ]
    r = match_findings(findings, EXPECTED, MUST_NOT)
    assert r.false_positives  # the logger.debug hit
    assert r.critical_pass is False


def test_missed_critical_fails_gate() -> None:
    findings = [{"rule_id": "style-1", "anchor": "return jsonify(rows)"}]
    r = match_findings(findings, EXPECTED, MUST_NOT)
    assert r.recall == 0.0
    assert r.critical_pass is False


def test_compliant_case_empty_findings_passes() -> None:
    # no expected findings (compliant diff) + agent reports nothing -> pass
    r = match_findings([], [], MUST_NOT)
    assert r.critical_pass is True
    assert r.recall == 1.0


def test_compliant_case_false_positive_fails() -> None:
    findings = [{"rule_id": "SEC-011", "anchor": "cursor.execute(query, (user_id,))"}]
    r = match_findings(findings, [], MUST_NOT)
    assert r.critical_pass is False


def test_parse_findings_strips_code_fence() -> None:
    text = '```json\n[{"rule_id": "x", "anchor": "y"}]\n```'
    assert parse_findings(text) == [{"rule_id": "x", "anchor": "y"}]


def test_parse_findings_unparseable_returns_none() -> None:
    assert parse_findings("I think there is a SQL injection somewhere.") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/unit/evaluators/findings/test_matcher.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the matcher**

`atp/evaluators/findings/matcher.py`:
```python
"""Deterministic matcher for structured code-review findings (R-07 Phase-1b #1).

Pure functions, no ATP dependencies. Matching keys on a code ANCHOR (a substring of
the offending code, whitespace-normalized) + a synonym set of acceptable rule ids
(case-insensitive) — NOT line numbers, which are too fragile for LLM output.
"""
import json
import re
from typing import Any

from pydantic import BaseModel


class MatchResult(BaseModel):
    """Outcome of matching agent findings against ground truth."""

    critical_pass: bool
    recall: float
    precision: float
    matched: list[str]
    missed: list[str]
    false_positives: list[str]
    unknown_extras: list[str]


def _norm(s: str) -> str:
    """Whitespace-collapsed, lowercased — the normalization for anchor/id compares."""
    return " ".join(str(s).split()).lower()


def _anchor_overlap(a: str, b: str) -> bool:
    """True if either normalized anchor is a substring of the other (bidirectional:
    the agent may quote a shorter or longer snippet than the ground-truth anchor)."""
    na, nb = _norm(a), _norm(b)
    return bool(na) and bool(nb) and (na in nb or nb in na)


def parse_findings(text: str) -> list[dict[str, Any]] | None:
    """Parse a findings JSON array from the agent's response text, tolerating a
    surrounding markdown code fence. Returns None if it is not a JSON array."""
    if text is None:
        return None
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    try:
        data = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, list) else None


def _finding_matches_expected(finding: dict[str, Any], expected: dict[str, Any]) -> bool:
    rule = _norm(finding.get("rule_id", ""))
    syns = {_norm(r) for r in expected.get("rule_ids", [])}
    if syns and rule not in syns:
        return False
    return _anchor_overlap(finding.get("anchor", ""), expected.get("anchor", ""))


def match_findings(
    findings: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    must_not_flag: list[dict[str, Any]],
) -> MatchResult:
    """Match findings against expected (planted defects) + must_not_flag (compliant
    lines). critical_pass = every critical-severity expected matched AND zero
    must_not_flag hits."""
    matched_keys: list[str] = []
    missed_keys: list[str] = []
    critical_ok = True
    for exp in expected:
        key = f"{exp.get('rule_ids', ['?'])[0]}@{exp.get('anchor', '')}"
        if any(_finding_matches_expected(f, exp) for f in findings):
            matched_keys.append(key)
        else:
            missed_keys.append(key)
            if exp.get("severity") == "critical":
                critical_ok = False

    false_positives: list[str] = []
    for mnf in must_not_flag:
        for f in findings:
            if _anchor_overlap(f.get("anchor", ""), mnf.get("anchor", "")):
                false_positives.append(f"{f.get('rule_id', '?')}@{mnf.get('anchor', '')}")
                break

    # findings that matched neither an expected nor a must_not_flag anchor
    unknown_extras = [
        f.get("anchor", "")
        for f in findings
        if not any(_finding_matches_expected(f, e) for e in expected)
        and not any(_anchor_overlap(f.get("anchor", ""), m.get("anchor", "")) for m in must_not_flag)
    ]

    tp = len(matched_keys)
    fp = len(false_positives)
    recall = 1.0 if not expected else tp / len(expected)
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    critical_pass = critical_ok and fp == 0
    return MatchResult(
        critical_pass=critical_pass,
        recall=round(recall, 6),
        precision=round(precision, 6),
        matched=matched_keys,
        missed=missed_keys,
        false_positives=false_positives,
        unknown_extras=unknown_extras,
    )
```

- [ ] **Step 4: Create the package `__init__` files**

`atp/evaluators/findings/__init__.py`:
```python
"""Deterministic structured-findings matching for code-review evals."""

from atp.evaluators.findings.matcher import MatchResult, match_findings, parse_findings

__all__ = ["MatchResult", "match_findings", "parse_findings"]
```
`tests/unit/evaluators/findings/__init__.py`: empty file.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/unit/evaluators/findings/test_matcher.py -q`
Expected: PASS (8 tests).

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check atp/evaluators/findings/ tests/unit/evaluators/findings/ --fix
uv run ruff format atp/evaluators/findings/ tests/unit/evaluators/findings/
git add atp/evaluators/findings/ tests/unit/evaluators/findings/
git commit -m "feat(evaluators): deterministic findings matcher core (anchor + rule synonyms)"
```

---

## Task 2: `FindingsMatchEvaluator` + registry

**Files:**
- Create: `atp/evaluators/findings/evaluator.py`
- Modify: `atp/evaluators/findings/__init__.py` (export the evaluator)
- Modify: `atp/evaluators/registry.py`
- Create: `tests/unit/evaluators/findings/test_evaluator.py`

- [ ] **Step 1: Read the existing patterns first**

Run: `sed -n '22,120p' atp/evaluators/base.py` and `sed -n '1,60p' atp/evaluators/artifact.py`
Note: `Evaluator` is an ABC with a `name` property and `async def evaluate(self, task, response, trace, assertion) -> EvalResult`. There is a `_create_check(...)` helper — read its exact signature and use it (do not hand-build `EvalCheck` if the helper exists). Findings text comes from `response.artifacts[*].content`. `assertion.config` is a `dict`.

- [ ] **Step 2: Write the failing test**

`tests/unit/evaluators/findings/test_evaluator.py`:
```python
"""Tests for FindingsMatchEvaluator (wraps the matcher as an ATP evaluator)."""
import pytest

from atp.evaluators.findings.evaluator import FindingsMatchEvaluator
from atp.loader.models import Assertion, TestDefinition
from atp.protocol.models import ArtifactFile, ATPResponse, ResponseStatus

pytestmark = pytest.mark.anyio


def _response(findings_json: str) -> ATPResponse:
    return ATPResponse(
        version="1.0",
        task_id="t1",
        status=ResponseStatus.COMPLETED,
        artifacts=[ArtifactFile(path="review.json", content=findings_json)],
        metrics=None,
    )


def _assertion() -> Assertion:
    return Assertion(
        type="findings_match",
        critical=True,
        config={
            "expected_findings": [
                {"rule_ids": ["SEC-011", "cwe-89"], "anchor": 'f"SELECT', "severity": "critical"}
            ],
            "must_not_flag": [{"anchor": "cursor.execute(query, (user_id,))"}],
        },
    )


async def test_evaluator_passes_when_defect_found() -> None:
    ev = FindingsMatchEvaluator()
    resp = _response('[{"rule_id": "cwe-89", "anchor": "q = f\\"SELECT 1\\""}]')
    result = await ev.evaluate(TestDefinition(id="t1", task={"description": "x"}), resp, [], _assertion())
    assert result.passed is True
    assert result.checks[0].details["recall"] == 1.0


async def test_evaluator_fails_on_unparseable_findings() -> None:
    ev = FindingsMatchEvaluator()
    resp = _response("there might be an injection")
    result = await ev.evaluate(TestDefinition(id="t1", task={"description": "x"}), resp, [], _assertion())
    assert result.passed is False
    assert "unparseable" in (result.checks[0].message or "").lower()
```
> NOTE: the exact `Assertion` / `TestDefinition` / `ArtifactFile` constructor args must
> match the real models — adjust the test's construction to whatever those pydantic models
> require (read `atp/loader/models.py` and `packages/atp-core/atp/protocol/models.py`).
> The behavioral asserts (passed / recall / "unparseable") are the contract.

- [ ] **Step 3: Run to verify it fails**

Run: `uv run --no-sync pytest tests/unit/evaluators/findings/test_evaluator.py -q`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement the evaluator**

`atp/evaluators/findings/evaluator.py`:
```python
"""ATP evaluator wrapping the deterministic findings matcher."""
from atp.core.results import EvalCheck, EvalResult
from atp.evaluators.base import Evaluator
from atp.evaluators.findings.matcher import match_findings, parse_findings
from atp.loader.models import Assertion, TestDefinition
from atp.protocol.models import ATPEvent, ATPResponse


class FindingsMatchEvaluator(Evaluator):
    """Grade structured agent findings against ground truth, deterministically."""

    @property
    def name(self) -> str:
        return "findings_match"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        cfg = assertion.config
        expected = cfg.get("expected_findings", [])
        must_not = cfg.get("must_not_flag", [])

        text = next(
            (a.content for a in response.artifacts if getattr(a, "content", None)),
            None,
        )
        findings = parse_findings(text) if text is not None else None
        if findings is None:
            check = EvalCheck(
                name="findings_match",
                passed=False,
                score=0.0,
                message="unparseable findings — cannot verify (expected a JSON array)",
                details={"recall": 0.0, "precision": 0.0},
            )
            return EvalResult(evaluator=self.name, checks=[check], critical=assertion.critical)

        r = match_findings(findings, expected, must_not)
        check = EvalCheck(
            name="findings_match",
            passed=r.critical_pass,
            score=1.0 if r.critical_pass else 0.0,
            message=(
                f"recall={r.recall} precision={r.precision} "
                f"missed={r.missed} false_positives={r.false_positives}"
            ),
            details=r.model_dump(),
        )
        return EvalResult(evaluator=self.name, checks=[check], critical=assertion.critical)
```
> If `_create_check` (from base.py) is the established way to build an `EvalCheck`, use it
> instead of constructing `EvalCheck` directly — match the codebase pattern you saw in Step 1.

- [ ] **Step 5: Export + register**

Append to `atp/evaluators/findings/__init__.py`:
```python
from atp.evaluators.findings.evaluator import FindingsMatchEvaluator  # noqa: E402

__all__.append("FindingsMatchEvaluator")
```
In `atp/evaluators/registry.py`, add an import and (in `EvaluatorRegistry.__init__`, beside the others):
```python
self.register("findings_match", FindingsMatchEvaluator)
```

- [ ] **Step 6: Run tests, lint, commit**

```bash
uv run --no-sync pytest tests/unit/evaluators/findings/ -q
uv run ruff check atp/evaluators/findings/ atp/evaluators/registry.py --fix && uv run ruff format atp/evaluators/findings/
git add atp/evaluators/findings/ atp/evaluators/registry.py tests/unit/evaluators/findings/test_evaluator.py
git commit -m "feat(evaluators): findings_match evaluator + registry wiring"
```

---

## Task 3: agent-eval-case schema — structured ground truth

**Files:**
- Modify: `packages/atp-method/atp_method/schema.py`
- Modify: `packages/atp-method/atp_method/loader.py`
- Test: `packages/atp-method/tests/test_schema.py` (add a case) — but run via root `tests/` if CI scopes there; otherwise the package test is fine for the plugin (its tests run in the plugin CI). Confirm where atp-method tests run before placing.

- [ ] **Step 1: Read the current `Grader` model + loader critical-assertion build**

Run: `sed -n '60,140p' packages/atp-method/atp_method/schema.py` and `sed -n '60,95p' packages/atp-method/atp_method/loader.py`
Note the `Grader` fields (`type`, `rubric`, `critical_check`, `gold`, validators) and how `_assertions()` builds the `METHOD_CRITICAL_CHECK` config.

- [ ] **Step 2: Add the failing schema test**

In the atp-method schema tests, add:
```python
def test_grader_accepts_structured_ground_truth() -> None:
    from atp_method.schema import Grader
    g = Grader(
        type="programmatic",
        critical_check="flag the SQL injection",
        expected_findings=[
            {"rule_ids": ["SEC-011", "cwe-89"], "anchor": 'f"SELECT', "severity": "critical"}
        ],
        must_not_flag=[{"anchor": "cursor.execute(query, (user_id,))"}],
    )
    assert g.expected_findings[0].anchor == 'f"SELECT'
    assert g.must_not_flag[0].anchor.startswith("cursor")
```

- [ ] **Step 3: Add the models to `schema.py`**

Add near `RubricItem`:
```python
class ExpectedFinding(BaseModel):
    """A planted defect the agent MUST surface (anchor + rule synonyms)."""

    rule_ids: list[str] = Field(..., min_length=1)
    anchor: str = Field(..., min_length=1)
    severity: Literal["critical", "major", "minor"] = "critical"


class ForbiddenAnchor(BaseModel):
    """A compliant line the agent MUST NOT flag (false-positive trap)."""

    anchor: str = Field(..., min_length=1)
```
In `Grader`, add:
```python
    expected_findings: list[ExpectedFinding] | None = None
    must_not_flag: list[ForbiddenAnchor] | None = None
```
And extend the existing grader validator: when `type == "programmatic"`, require `expected_findings` to be non-empty (mirror how `exact` requires `gold`):
```python
        if self.type == "programmatic" and not self.expected_findings:
            raise ValueError("grader type 'programmatic' requires expected_findings")
```

- [ ] **Step 4: Pass ground truth through the loader**

In `loader.py` `_assertions()`, add to the `METHOD_CRITICAL_CHECK` assertion `config` dict:
```python
                "expected_findings": [
                    f.model_dump() for f in (case.grader.expected_findings or [])
                ],
                "must_not_flag": [
                    m.model_dump() for m in (case.grader.must_not_flag or [])
                ],
```

- [ ] **Step 5: Run the schema test, lint, commit**

```bash
uv run --no-sync pytest packages/atp-method/tests/test_schema.py -q
uv run ruff check packages/atp-method/ --fix && uv run ruff format packages/atp-method/
git add packages/atp-method/atp_method/schema.py packages/atp-method/atp_method/loader.py packages/atp-method/tests/test_schema.py
git commit -m "feat(method): structured ground truth (expected_findings/must_not_flag) in grader schema"
```

---

## Task 4: methodology programmatic dispatch

**Files:**
- Modify: `packages/atp-method/atp_method/evaluators/case_evaluator.py`
- Test: `packages/atp-method/tests/test_evaluator.py`

- [ ] **Step 1: Read `_evaluate_critical`**

Run: `sed -n '95,145p' packages/atp-method/atp_method/evaluators/case_evaluator.py`
Note it currently always calls `_judge_score`. The assertion `config` now carries `grader_type`, `expected_findings`, `must_not_flag` (from Task 3).

- [ ] **Step 2: Write the failing test**

In `packages/atp-method/tests/test_evaluator.py`:
```python
async def test_programmatic_critical_uses_matcher_not_judge() -> None:
    # grader_type=programmatic must grade by code, never call the (here, absent) judge
    from atp_method.evaluators.case_evaluator import AgentEvalCaseEvaluator
    from atp.loader.models import Assertion, TestDefinition
    from atp.protocol.models import ArtifactFile, ATPResponse, ResponseStatus

    ev = AgentEvalCaseEvaluator(judge=None)  # judge must NOT be called
    resp = ATPResponse(
        version="1.0", task_id="t", status=ResponseStatus.COMPLETED,
        artifacts=[ArtifactFile(path="r.json", content='[{"rule_id":"cwe-89","anchor":"f\\"SELECT"}]')],
        metrics=None,
    )
    assertion = Assertion(
        type="method_critical_check", critical=True,
        config={
            "check": "flag the injection", "grader_type": "programmatic",
            "expected_findings": [{"rule_ids": ["SEC-011", "cwe-89"], "anchor": 'f"SELECT', "severity": "critical"}],
            "must_not_flag": [],
        },
    )
    result = await ev.evaluate(TestDefinition(id="t", task={"description": "x"}), resp, [], assertion)
    assert result.passed is True
```
> Adjust constructor args to the real models (read them first). The contract: with
> `grader_type=programmatic` and a matching finding, the result passes WITHOUT a judge.

- [ ] **Step 3: Add the dispatch in `_evaluate_critical`**

At the top of `_evaluate_critical`, before the judge call:
```python
        if assertion.config.get("grader_type") == "programmatic":
            from atp.evaluators.findings.matcher import match_findings, parse_findings

            text = next(
                (a.content for a in response.artifacts if getattr(a, "content", None)),
                None,
            )
            findings = parse_findings(text) if text is not None else None
            if findings is None:
                return self._critical_result(
                    passed=False, score=0.0,
                    message="unparseable findings — cannot verify",
                )
            r = match_findings(
                findings,
                assertion.config.get("expected_findings", []),
                assertion.config.get("must_not_flag", []),
            )
            return self._critical_result(
                passed=r.critical_pass,
                score=1.0 if r.critical_pass else 0.0,
                message=f"recall={r.recall} fp={r.false_positives}",
                details=r.model_dump(),
            )
```
> `_critical_result(...)` is shorthand for however `_evaluate_critical` already builds its
> `EvalResult` (read Step 1). If there is no such helper, build the `EvalResult` the same
> way the existing judge path does — match the file's pattern, do not invent a new shape.

- [ ] **Step 4: Run tests, lint, commit**

```bash
uv run --no-sync pytest packages/atp-method/tests/test_evaluator.py -q
uv run ruff check packages/atp-method/ --fix && uv run ruff format packages/atp-method/
git add packages/atp-method/atp_method/evaluators/case_evaluator.py packages/atp-method/tests/test_evaluator.py
git commit -m "feat(method): programmatic critical_check dispatches to findings matcher (no judge)"
```

---

## Task 5: shim emits strict JSON findings

**Files:**
- Modify: `method/spawners/claude_code_shim.py`
- Modify: `tests/unit/method_spawners/fixtures/fake_claude.py`
- Modify: `tests/unit/method_spawners/test_claude_code_shim.py`

- [ ] **Step 1: Update the test fixture to emit structured findings**

In `tests/unit/method_spawners/fixtures/fake_claude.py`, change the `result` field to a JSON array string:
```python
    "result": '[{"rule_id": "sql-injection", "file": "app.py", "anchor": "query = f\\"SELECT", "severity": "critical", "fix": "use a parameterized query"}]',
```

- [ ] **Step 2: Update the shim test assertion**

In `tests/unit/method_spawners/test_claude_code_shim.py`, change the content assertion to parse JSON:
```python
    import json as _json
    findings = _json.loads(arts[0]["content"])
    assert findings[0]["rule_id"] == "sql-injection"
    assert "SELECT" in findings[0]["anchor"]
```

- [ ] **Step 3: Run to verify it fails (envelope still asks for prose)**

Run: `uv run --no-sync pytest tests/unit/method_spawners/ -q`
Expected: PASS actually (the shim passes through whatever `result` the fake emits) — so this task is mostly the ENVELOPE change for the real agent. If the test already passes, proceed to Step 4 (the envelope is what makes a REAL claude emit JSON).

- [ ] **Step 4: Update `REVIEW_ENVELOPE`**

In `method/spawners/claude_code_shim.py`, replace `REVIEW_ENVELOPE` with:
```python
REVIEW_ENVELOPE = (
    "You are a senior code reviewer. Review the material below. Output ONLY a JSON "
    "array of findings (no prose, no markdown fence). Each finding is an object with "
    'keys: "rule_id" (the rule/CWE id), "file", "anchor" (the exact offending code '
    'substring), "severity" (critical|major|minor), "fix". If the code is compliant, '
    "output an empty array [].\n\n{task}"
)
```

- [ ] **Step 5: Run tests, lint, commit**

```bash
uv run --no-sync pytest tests/unit/method_spawners/ -q
uv run ruff check method/spawners/ tests/unit/method_spawners/ --fix && uv run ruff format method/spawners/
git add method/spawners/claude_code_shim.py tests/unit/method_spawners/
git commit -m "feat(method): shim demands strict JSON findings (structured output for findings_match)"
```

---

## Task 6: convert the two code-review cases + offline end-to-end

**Files:**
- Modify: `method/cases/code-review/case-code-review-sqli-clean-001.yaml`
- Modify: `method/cases/code-review/case-code-review-sqli-moderate-001.yaml`
- Test: `tests/unit/evaluators/findings/test_cases_load.py`

- [ ] **Step 1: Convert the moderate case to `programmatic` + structured ground truth**

In `case-code-review-sqli-moderate-001.yaml`, set `grader.type: programmatic`, keep the `rubric` (still model_graded for quality), and replace the inline `gold` with:
```yaml
  expected_findings:
    - rule_ids: [SEC-011, sql-injection, cwe-89]
      anchor: 'f"SELECT * FROM users WHERE id = {user_id}'
      severity: critical
  must_not_flag:
    - anchor: 'user_id = request.args["id"]'
    - anchor: 'logger.debug("looking up user")'
    - anchor: 'return jsonify(rows)'
```

- [ ] **Step 2: Convert the clean case (compliant → empty expected, trap on the parameterized line)**

In `case-code-review-sqli-clean-001.yaml`, set `grader.type: programmatic`, keep the rubric, replace `gold` with:
```yaml
  expected_findings: []
  must_not_flag:
    - anchor: 'cursor.execute(query, (user_id,))'
    - anchor: 'query = "SELECT * FROM users WHERE id = %s"'
```
> NOTE: `expected_findings: []` is valid only if the Task-3 validator requires non-empty
> `expected_findings` ONLY for the trap; a compliant case has none. Adjust the Task-3
> validator to allow an empty list (compliant case) — require the KEY to be present under
> `programmatic`, but allow `[]`. Update the Task-3 validator + its test accordingly.

- [ ] **Step 3: Add a load + offline-grade test**

`tests/unit/evaluators/findings/test_cases_load.py`:
```python
"""The converted code-review cases load and grade deterministically offline."""
import subprocess
import sys


def test_cases_list_via_plugin() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "atp.cli.main", "test", "method/cases/code-review",
         "--list-only", "--adapter=cli"],
        capture_output=True, text=True, timeout=60,
    )
    assert "case-code-review-sqli-clean-001" in out.stdout
    assert "case-code-review-sqli-moderate-001" in out.stdout
```
> If `python -m atp.cli.main` is not the entrypoint, use the installed `atp` console
> script path; confirm how `atp` is invoked (`uv run atp ...`) and mirror it.

- [ ] **Step 4: Verify cases load + commit**

```bash
uv run --no-sync atp test method/cases/code-review --list-only --adapter=cli
uv run --no-sync pytest tests/unit/evaluators/findings/ -q
git add method/cases/code-review/ tests/unit/evaluators/findings/test_cases_load.py
git commit -m "feat(method): convert code-review cases to programmatic findings grading"
```

---

## Task 7: full-suite verification

- [ ] **Step 1: Run the new-code surface + a broad regression**

Run: `uv run --no-sync pytest tests/unit/evaluators/findings/ tests/unit/method_spawners/ packages/atp-method/tests/ -q`
Expected: all pass.

- [ ] **Step 2: ruff + pyrefly clean**

Run: `uv run --no-sync ruff check . && uv run --no-sync pyrefly check`
Expected: All checks pass; 0 errors.

- [ ] **Step 3: Offline end-to-end pipe re-run (fake claude, structured output, no API cost)**

```bash
CLAUDE_BIN="$(command -v python3) tests/unit/method_spawners/fixtures/fake_claude.py" \
uv run --no-sync atp test method/cases/code-review \
  --adapter=cli \
  --adapter-config command="python3 method/spawners/claude_code_shim.py",inherit_environment=true \
  --model=claude_code --runs=1
```
Expected: runs without an LLM judge for the critical_check; the moderate case's gate is
graded deterministically by the matcher (the fake emits the SEC-011 finding → critical_pass;
the rubric still needs a judge — set `ATP_JUDGE_*` or expect the rubric component to no-op/skip).
Record what the gate did (deterministic) vs the rubric (still LLM). This closes spec success criteria.

---

## Self-review notes

- **Spec coverage:** §1 evaluator → Task 1+2; §2 output contract → Task 5; §3 ground truth → Task 3; §4 wiring (native `findings_match` type → Task 2 registry; methodology programmatic → Task 4); §5 scoring/gate → matcher `critical_pass`/recall/precision (Task 1) surfaced in Task 2/4 details; §6 error handling → unparseable→fail (Task 1 parse + Task 2/4); §7 testing → every task is TDD, all under `tests/` (Task 3 plugin tests run in plugin CI — confirmed in Step where placed).
- **Read-first steps (not placeholders):** Task 2 Step 1 (`_create_check`/artifact access), Task 4 Step 1 (`_evaluate_critical` result-building helper), Task 3 Step 1 (Grader validator pattern), Task 6 Step 2 (validator must allow empty `expected_findings` for compliant cases) — each has an explicit read-and-reconcile instruction because the exact helper/shape must match live code.
- **Type consistency:** `MatchResult` fields (`critical_pass`, `recall`, `precision`, `matched`, `missed`, `false_positives`, `unknown_extras`) are used consistently in Tasks 1/2/4. `parse_findings` returns `list|None`; the `None` path is the unparseable-fail in both wirings.
- **YAGNI:** no codex/aider, no language axis, no correctness family here (separate specs reuse this matcher).
