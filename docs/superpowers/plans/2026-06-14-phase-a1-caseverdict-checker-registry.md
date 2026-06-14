# Phase A-1: Uniform CaseVerdict + checker registry (grader spine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-capability `grader.type` enum growth with a uniform `CaseVerdict` + a named **checker registry under `programmatic`**, and close the real `agent-eval-case` JSON-schema↔pydantic divergence — so a new test-type registers a checker instead of forking the schema and the dispatch.

**Architecture:** A deterministic capability check becomes `grader: {type: programmatic, checker: <name>}`. Checkers are pure functions registered by name, each returning a shared `CaseVerdict`. `findings_match` folds from a `grader.type` enum member into the first registered checker. The methodology evaluator dispatches on `checker` via the registry; the LLM-judge path is unchanged. This is the precondition for companion-spec SP-1 (the store persists from `CaseVerdict`).

**Tech Stack:** Python 3.12, uv, pydantic v2, pytest (anyio), jsonschema; packages `atp-core`, `atp-method`, `atp/evaluators`.

**Companion docs:** spec `docs/superpowers/specs/2026-06-14-eval-results-architecture-design.md` (§6), ADR `docs/adr/006-unified-capability-test-types.md` (direction 1). P3 (`MatchResult.malformed`, `grade_findings`) already merged (PR #173) — this plan generalizes it, it does not rebuild it.

**Scope guard (NOT in this plan):** envelope lift to capability level, harness `--family`/`--task-type` parameterization, the `task_type↔benchmark_id↔TaskType` taxonomy registry (those are **Phase A-2**, a sibling plan); the store columns / persistence (SP-1); the native `FindingsMatchEvaluator` for `type: findings_match` native suites (`atp/evaluators/findings/evaluator.py`) stays untouched — only the *methodology* grader changes.

---

## File Structure

- Create `packages/atp-core/atp/core/results.py` → add `CaseVerdict` model (shared verdict; lives beside `EvalCheck`/`EvalResult`). Edited via the `atp/core/results.py` symlink.
- Create `atp/evaluators/checkers/registry.py` → `Checker` type + register/get/list.
- Create `atp/evaluators/checkers/__init__.py` → re-exports + registers built-ins.
- Create `atp/evaluators/findings/checker.py` → `findings_check(config, text) -> CaseVerdict` wrapping `grade_findings`.
- Modify `packages/atp-method/atp_method/schema.py` → drop `findings_match` from `GraderType`, add `Grader.checker`, update validator.
- Modify `method/agent-eval-case.schema.json` → add `checker`/`expected_findings`/`must_not_flag` to grader, add `$defs`, add the `programmatic+checker` conditional.
- Modify `packages/atp-method/atp_method/loader.py` → thread `checker` into the assertion config.
- Modify `packages/atp-method/atp_method/evaluators/case_evaluator.py` → dispatch on `checker` via the registry.
- Modify `method/cases/code-review/case-code-review-sqli-clean-001.yaml` and `...-moderate-001.yaml` → migrate to `type: programmatic, checker: findings_match`.
- Tests: `tests/unit/core/test_case_verdict.py`, `tests/unit/evaluators/checkers/test_registry.py`, `tests/unit/evaluators/findings/test_checker.py`, `packages/atp-method/tests/test_schema.py` (extend), `packages/atp-method/tests/test_schema_contract.py` (new), `packages/atp-method/tests/test_evaluator.py` (extend), `packages/atp-method/tests/test_cases_load.py` (new).

**Test cwd note:** core/`atp` tests run from the repo root (`uv run pytest …`). `atp-method` package tests run from `packages/atp-method` (separate conftest root) — `cd packages/atp-method && uv run pytest …`.

---

## Task 1: `CaseVerdict` shared model

**Files:**
- Modify: `packages/atp-core/atp/core/results.py` (add the model near `EvalCheck`)
- Test: `tests/unit/core/test_case_verdict.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_case_verdict.py
"""Tests for the shared CaseVerdict model (Phase A-1)."""

from atp.core.results import CaseVerdict


def test_caseverdict_defaults_minimal() -> None:
    v = CaseVerdict(critical_pass=True)
    assert v.critical_pass is True
    assert v.malformed is False
    assert v.recall == 0.0
    assert v.fp_count == 0
    assert v.rubric_score == 0.0
    assert v.details == {}
    assert v.grader_version == ""


def test_caseverdict_roundtrips_dump() -> None:
    v = CaseVerdict(
        critical_pass=False,
        malformed=True,
        recall=0.5,
        precision=0.5,
        fp_count=2,
        details={"missed": ["x"]},
        grader_version="findings_match@1",
    )
    d = v.model_dump()
    assert d["malformed"] is True and d["fp_count"] == 2
    assert CaseVerdict(**d) == v
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/core/test_case_verdict.py -v`
Expected: FAIL with `ImportError: cannot import name 'CaseVerdict'`.

- [ ] **Step 3: Add the model**

In `packages/atp-core/atp/core/results.py`, ensure `Any` is imported (`from typing import Any`) and `Field` is imported from pydantic (both are already used in this file), then add after the `EvalCheck` class:

```python
class CaseVerdict(BaseModel):
    """Uniform per-case verdict returned by every deterministic checker.

    Shared across evaluators, the runner, reporters, and dashboard persistence
    (companion spec SP-1 maps these fields straight into result columns). A
    checker selected under ``grader: {type: programmatic, checker: <name>}``
    returns this instead of a checker-specific shape.
    """

    critical_pass: bool
    # Distinct from a missed defect: the agent output was not gradeable
    # (unparseable / failed strict validation). See grade_findings (PR #173).
    malformed: bool = False
    recall: float = 0.0
    precision: float = 0.0
    fp_count: int = 0
    rubric_score: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)
    grader_version: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/core/test_case_verdict.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/atp-core/atp/core/results.py tests/unit/core/test_case_verdict.py
git commit -m "feat(core): CaseVerdict shared per-case verdict model (Phase A-1)"
```

---

## Task 2: Checker registry

**Files:**
- Create: `atp/evaluators/checkers/registry.py`
- Test: `tests/unit/evaluators/checkers/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/evaluators/checkers/test_registry.py
"""Tests for the deterministic checker registry (Phase A-1)."""

import pytest

from atp.core.results import CaseVerdict
from atp.evaluators.checkers.registry import (
    get_checker,
    list_checkers,
    register_checker,
)


def _dummy(config: dict, text: str | None) -> CaseVerdict:
    return CaseVerdict(critical_pass=True, grader_version="dummy@1")


def test_register_and_get() -> None:
    register_checker("dummy", _dummy)
    fn = get_checker("dummy")
    assert fn is not None
    assert fn({}, None).grader_version == "dummy@1"
    assert "dummy" in list_checkers()


def test_unknown_checker_returns_none() -> None:
    assert get_checker("does-not-exist") is None


def test_register_rejects_duplicate() -> None:
    register_checker("dup", _dummy)
    with pytest.raises(ValueError, match="already registered"):
        register_checker("dup", _dummy)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/evaluators/checkers/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'atp.evaluators.checkers'`.

- [ ] **Step 3: Create the registry**

```python
# atp/evaluators/checkers/registry.py
"""Registry of named deterministic checkers (Phase A-1).

A checker is a pure function selected by ``grader: {type: programmatic,
checker: <name>}``. It maps a grader config + the agent's output text to a
uniform :class:`CaseVerdict`. A new capability registers a checker instead of
adding a ``grader.type`` enum value — the core dispatch stays closed.
"""

from collections.abc import Callable

from atp.core.results import CaseVerdict

Checker = Callable[[dict, "str | None"], CaseVerdict]

_CHECKERS: dict[str, Checker] = {}


def register_checker(name: str, fn: Checker) -> None:
    """Register a checker under ``name``. Raises on a duplicate name."""
    if name in _CHECKERS:
        raise ValueError(f"checker '{name}' already registered")
    _CHECKERS[name] = fn


def get_checker(name: str) -> Checker | None:
    """Return the checker registered under ``name``, or None if unknown."""
    return _CHECKERS.get(name)


def list_checkers() -> list[str]:
    """Return the sorted names of registered checkers."""
    return sorted(_CHECKERS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/evaluators/checkers/test_registry.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add atp/evaluators/checkers/registry.py tests/unit/evaluators/checkers/test_registry.py
git commit -m "feat(evaluators): named checker registry under programmatic (Phase A-1)"
```

---

## Task 3: `findings_match` checker + built-in registration

**Files:**
- Create: `atp/evaluators/findings/checker.py`
- Create: `atp/evaluators/checkers/__init__.py`
- Test: `tests/unit/evaluators/findings/test_checker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/evaluators/findings/test_checker.py
"""Tests for the findings_match checker → CaseVerdict mapping (Phase A-1)."""

import json

from atp.evaluators.checkers import get_checker  # triggers built-in registration

EXPECTED = [
    {"rule_ids": ["SEC-011", "cwe-89"], "anchor": 'f"SELECT', "severity": "critical"}
]
MUST_NOT = [{"anchor": "logger.debug"}]


def test_findings_match_registered() -> None:
    assert get_checker("findings_match") is not None


def test_valid_match_verdict() -> None:
    check = get_checker("findings_match")
    text = json.dumps(
        [{"rule_id": "cwe-89", "anchor": 'x = f"SELECT 1', "severity": "critical"}]
    )
    v = check({"expected_findings": EXPECTED, "must_not_flag": MUST_NOT}, text)
    assert v.critical_pass is True
    assert v.malformed is False
    assert v.recall == 1.0
    assert v.fp_count == 0
    assert v.grader_version == "findings_match@1"
    assert v.details["malformed"] is False


def test_malformed_verdict() -> None:
    check = get_checker("findings_match")
    # missing required severity -> malformed (not a silent miss)
    text = json.dumps([{"rule_id": "cwe-89", "anchor": 'f"SELECT'}])
    v = check({"expected_findings": EXPECTED, "must_not_flag": MUST_NOT}, text)
    assert v.malformed is True
    assert v.critical_pass is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/evaluators/findings/test_checker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'atp.evaluators.checkers'` is gone (Task 2 created it) but `atp/evaluators/checkers/__init__.py` does not yet register findings — `get_checker("findings_match")` returns None → first test fails.

- [ ] **Step 3: Write the checker**

```python
# atp/evaluators/findings/checker.py
"""findings_match checker: grade_findings → uniform CaseVerdict (Phase A-1)."""

from typing import Any

from atp.core.results import CaseVerdict
from atp.evaluators.findings.matcher import grade_findings

FINDINGS_CHECKER_VERSION = "findings_match@1"


def findings_check(config: dict[str, Any], text: str | None) -> CaseVerdict:
    """Run the deterministic findings matcher and map it to a CaseVerdict.

    ``config`` carries ``expected_findings`` / ``must_not_flag`` (the grader's
    ground truth). ``text`` is the agent's primary output.
    """
    r = grade_findings(
        text,
        config.get("expected_findings", []),
        config.get("must_not_flag", []),
    )
    return CaseVerdict(
        critical_pass=r.critical_pass,
        malformed=r.malformed,
        recall=r.recall,
        precision=r.precision,
        fp_count=len(r.false_positives),
        rubric_score=0.0,
        details=r.model_dump(),
        grader_version=FINDINGS_CHECKER_VERSION,
    )
```

- [ ] **Step 4: Register it as a built-in**

```python
# atp/evaluators/checkers/__init__.py
"""Deterministic checker registry + built-in registrations (Phase A-1)."""

from atp.evaluators.checkers.registry import (
    Checker,
    get_checker,
    list_checkers,
    register_checker,
)
from atp.evaluators.findings.checker import findings_check

register_checker("findings_match", findings_check)

__all__ = [
    "Checker",
    "get_checker",
    "list_checkers",
    "register_checker",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/evaluators/findings/test_checker.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add atp/evaluators/findings/checker.py atp/evaluators/checkers/__init__.py tests/unit/evaluators/findings/test_checker.py
git commit -m "feat(evaluators): findings_match checker + register as built-in (Phase A-1)"
```

---

## Task 4: pydantic schema — drop `findings_match` type, add `checker`

**Files:**
- Modify: `packages/atp-method/atp_method/schema.py`
- Test: `packages/atp-method/tests/test_schema.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `packages/atp-method/tests/test_schema.py`:

```python
def test_grader_type_findings_match_now_rejected() -> None:
    # findings_match is no longer a grader.type; it is a checker under programmatic
    with pytest.raises(ValidationError):
        Grader(
            type="findings_match",
            critical_check="x",
            scoring="y",
            expected_findings=[],
        )


def test_programmatic_checker_findings_requires_expected_findings() -> None:
    with pytest.raises(ValidationError, match="expected_findings"):
        Grader(
            type="programmatic",
            checker="findings_match",
            critical_check="x",
            scoring="y",
        )


def test_programmatic_checker_findings_accepts_empty_expected() -> None:
    g = Grader(
        type="programmatic",
        checker="findings_match",
        critical_check="x",
        scoring="y",
        expected_findings=[],
    )
    assert g.checker == "findings_match"


def test_checker_requires_programmatic_type() -> None:
    with pytest.raises(ValidationError, match="programmatic"):
        Grader(
            type="rubric",
            checker="findings_match",
            critical_check="x",
            scoring="y",
            rubric=[{"criterion": "c", "weight": 1.0}],
        )
```

Confirm the test file's imports include `pytest`, `ValidationError` (from `pydantic`), and `Grader` (from `atp_method.schema`); add any that are missing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/atp-method && uv run pytest tests/test_schema.py -v -k "findings or checker or programmatic"`
Expected: FAIL — `findings_match` still accepted as a type; `checker` is an unknown field (`extra="forbid"`).

- [ ] **Step 3: Edit the schema model**

In `packages/atp_method/schema.py`, change `GraderType` to drop `findings_match`:

```python
GraderType = Literal[
    "exact",
    "regex",
    "programmatic",
    "rubric",
    "model_graded",
    "human",
]
```

Add the `checker` field to `Grader` (after `type`):

```python
    type: GraderType
    checker: str | None = None
```

Replace the `findings_match` branch in `validate_grader_requirements` with checker-based rules:

```python
        if self.checker is not None and self.type != "programmatic":
            raise ValueError("grader.checker requires type 'programmatic'")
        if self.checker == "findings_match" and self.expected_findings is None:
            raise ValueError(
                "checker 'findings_match' requires expected_findings "
                "(use [] for a compliant case with no planted defect)"
            )
```

(Keep the existing `rubric`/`model_graded`/`exact` rules unchanged. Remove the old `if self.type == "findings_match"` line.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/atp-method && uv run pytest tests/test_schema.py -v`
Expected: PASS, including the 4 new tests. (The old `test_findings_match_grader_*` tests that asserted `type="findings_match"` will now fail — update them in this step to `type="programmatic", checker="findings_match"`, or delete the ones that duplicate the new tests.)

- [ ] **Step 5: Commit**

```bash
git add packages/atp-method/atp_method/schema.py packages/atp-method/tests/test_schema.py
git commit -m "feat(method): grader.checker under programmatic; drop findings_match type (Phase A-1)"
```

---

## Task 5: JSON contract — add `checker`/`expected_findings`/`must_not_flag` + conditional

**Files:**
- Modify: `method/agent-eval-case.schema.json`
- Test: `packages/atp-method/tests/test_schema_contract.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# packages/atp-method/tests/test_schema_contract.py
"""The JSON contract accepts programmatic+checker findings cases (Phase A-1)."""

import json
from pathlib import Path

import jsonschema
import pytest

# repo-root-relative; tests run from packages/atp-method
SCHEMA = json.loads(
    (Path(__file__).resolve().parents[3] / "method" / "agent-eval-case.schema.json")
    .read_text()
)


def _case(grader: dict) -> dict:
    return {
        "id": "c-1",
        "version": 1,
        "family": "f",
        "status": "active",
        "suite_type": "probe",
        "capability": "safety_compliance",
        "construction_axis": "adversarial_environment",
        "axis_level": "moderate",
        "instruction": "review",
        "artifacts": [{"id": "d", "type": "text", "content": "x"}],
        "environment": {"tools": ["file_read"], "side_effects": "none"},
        "expected_failure_mode": "misses it",
        "grader": grader,
        "provenance": {"author": "a", "created": "2026-06-14"},
    }


def test_contract_accepts_programmatic_checker_findings() -> None:
    case = _case({
        "type": "programmatic",
        "checker": "findings_match",
        "expected_findings": [
            {"rule_ids": ["SEC-011"], "anchor": 'f"SELECT', "severity": "critical"}
        ],
        "must_not_flag": [{"anchor": "logger.debug"}],
        "critical_check": "flag it",
        "scoring": "fail if critical fails",
    })
    jsonschema.validate(case, SCHEMA)  # must not raise


def test_contract_rejects_findings_checker_without_expected_findings() -> None:
    case = _case({
        "type": "programmatic",
        "checker": "findings_match",
        "critical_check": "flag it",
        "scoring": "fail if critical fails",
    })
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(case, SCHEMA)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/atp-method && uv run pytest tests/test_schema_contract.py -v`
Expected: FAIL — `additionalProperties` rejects `checker`/`expected_findings`/`must_not_flag` (not yet in the contract).

- [ ] **Step 3: Edit the JSON schema**

In `method/agent-eval-case.schema.json`, under `properties.grader.properties`, add three properties (keep `additionalProperties: false`):

```json
        "checker": {
          "type": "string",
          "description": "Named deterministic checker, used with type 'programmatic' (e.g. findings_match)."
        },
        "expected_findings": {
          "type": "array",
          "description": "Ground-truth planted defects for the findings_match checker.",
          "items": { "$ref": "#/$defs/expected_finding" }
        },
        "must_not_flag": {
          "type": "array",
          "description": "Compliant anchors the agent must not flag (findings_match).",
          "items": { "$ref": "#/$defs/forbidden_anchor" }
        }
```

Add two `$defs` (alongside `artifact`/`rubric_item`/`turn`):

```json
    "expected_finding": {
      "type": "object",
      "additionalProperties": false,
      "required": ["rule_ids", "anchor", "severity"],
      "properties": {
        "rule_ids": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
        "anchor": { "type": "string", "minLength": 1 },
        "severity": { "type": "string", "enum": ["critical", "major", "minor"] }
      }
    },
    "forbidden_anchor": {
      "type": "object",
      "additionalProperties": false,
      "required": ["anchor"],
      "properties": { "anchor": { "type": "string", "minLength": 1 } }
    }
```

Add a conditional to the top-level `allOf` array (a new element):

```json
    {
      "$comment": "findings_match checker requires ground-truth expected_findings.",
      "if": {
        "properties": {
          "grader": {
            "properties": {
              "type": { "const": "programmatic" },
              "checker": { "const": "findings_match" }
            },
            "required": ["checker"]
          }
        }
      },
      "then": {
        "properties": {
          "grader": { "required": ["expected_findings"] }
        }
      }
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/atp-method && uv run pytest tests/test_schema_contract.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add method/agent-eval-case.schema.json packages/atp-method/tests/test_schema_contract.py
git commit -m "feat(method): JSON contract — checker/expected_findings/must_not_flag + conditional (Phase A-1)"
```

---

## Task 6: Loader — thread `checker` into the assertion config

**Files:**
- Modify: `packages/atp-method/atp_method/loader.py` (`_assertions`)
- Test: `packages/atp-method/tests/test_loader.py` (create if absent, else extend)

- [ ] **Step 1: Write the failing test**

```python
# packages/atp-method/tests/test_loader.py  (add this test; create the file if missing)
"""Loader threads grader.checker into the critical-check assertion (Phase A-1)."""

from atp_method.loader import METHOD_CRITICAL_CHECK, case_to_test_definition
from atp_method.schema import AgentEvalCase


def _case_dict() -> dict:
    return {
        "id": "c-1",
        "version": 1,
        "family": "f",
        "status": "active",
        "suite_type": "probe",
        "capability": "safety_compliance",
        "construction_axis": "adversarial_environment",
        "axis_level": "moderate",
        "instruction": "review",
        "artifacts": [{"id": "d", "type": "text", "content": "x"}],
        "environment": {"tools": ["file_read"], "side_effects": "none"},
        "expected_failure_mode": "misses it",
        "grader": {
            "type": "programmatic",
            "checker": "findings_match",
            "expected_findings": [],
            "critical_check": "flag it",
            "scoring": "fail if critical fails",
        },
        "provenance": {"author": "a", "created": "2026-06-14"},
    }


def test_loader_threads_checker_into_critical_config() -> None:
    td = case_to_test_definition(AgentEvalCase.model_validate(_case_dict()))
    crit = next(a for a in td.assertions if a.type == METHOD_CRITICAL_CHECK)
    assert crit.config["checker"] == "findings_match"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/atp-method && uv run pytest tests/test_loader.py::test_loader_threads_checker_into_critical_config -v`
Expected: FAIL with `KeyError: 'checker'`.

- [ ] **Step 3: Add `checker` to both assertion configs**

In `_assertions`, add `"checker": case.grader.checker,` to the `METHOD_CRITICAL_CHECK` config dict and to the `METHOD_RUBRIC` config dict (next to the existing `"grader_type": case.grader.type,` line in each).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/atp-method && uv run pytest tests/test_loader.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-method/atp_method/loader.py packages/atp-method/tests/test_loader.py
git commit -m "feat(method): thread grader.checker into assertion config (Phase A-1)"
```

---

## Task 7: Evaluator — dispatch on `checker` via the registry

**Files:**
- Modify: `packages/atp-method/atp_method/evaluators/case_evaluator.py` (`_evaluate_critical`)
- Test: `packages/atp-method/tests/test_evaluator.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `packages/atp-method/tests/test_evaluator.py` (reuses `BombJudge`, `_task`, `ArtifactFile`, `ResponseStatus`, `ATPResponse`, `Assertion`, `METHOD_CRITICAL_CHECK` already imported in that file):

```python
@pytest.mark.anyio
async def test_checker_dispatch_uses_registry_not_judge() -> None:
    """grader.checker resolves via the checker registry; judge is never called."""
    ev = AgentEvalCaseEvaluator(judge=BombJudge())
    response = ATPResponse(
        task_id="case-x-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[
            ArtifactFile(
                path="findings.json",
                content='[{"rule_id":"cwe-89","anchor":"f\\"SELECT","severity":"critical"}]',
            )
        ],
    )
    assertion = Assertion(
        type=METHOD_CRITICAL_CHECK,
        critical=True,
        config={
            "check": "flag the injection",
            "checker": "findings_match",
            "expected_findings": [
                {"rule_ids": ["cwe-89"], "anchor": 'f"SELECT', "severity": "critical"}
            ],
            "must_not_flag": [],
        },
    )
    result = await ev.evaluate(_task(), response, [], assertion)
    check = result.checks[0]
    assert check.name == "critical_check"
    assert check.passed is True
    assert check.details is not None and check.details["malformed"] is False


@pytest.mark.anyio
async def test_unknown_checker_fails_closed() -> None:
    ev = AgentEvalCaseEvaluator(judge=BombJudge())
    response = ATPResponse(
        task_id="case-x-001", status=ResponseStatus.COMPLETED,
        artifacts=[ArtifactFile(path="f.json", content="[]")],
    )
    assertion = Assertion(
        type=METHOD_CRITICAL_CHECK, critical=True,
        config={"check": "x", "checker": "nope", "expected_findings": []},
    )
    result = await ev.evaluate(_task(), response, [], assertion)
    assert result.checks[0].passed is False
    assert "unknown checker" in (result.checks[0].message or "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/atp-method && uv run pytest tests/test_evaluator.py -v -k "checker"`
Expected: FAIL — the evaluator still dispatches on `grader_type == "findings_match"`, so the `BombJudge` is called (or the path is wrong).

- [ ] **Step 3: Replace the dispatch block**

In `case_evaluator.py` `_evaluate_critical`, replace the `if assertion.config.get("grader_type") == "findings_match":` block with checker-registry dispatch:

```python
        checker_name = assertion.config.get("checker")
        if checker_name:
            from atp.evaluators.checkers import get_checker

            checker = get_checker(checker_name)
            if checker is None:
                return EvalResult(
                    evaluator=self.name,
                    checks=[
                        EvalCheck(
                            name="critical_check",
                            passed=False,
                            score=0.0,
                            message=f"unknown checker: {checker_name}",
                        )
                    ],
                )
            text = next(
                (a.content for a in response.artifacts if getattr(a, "content", None)),
                None,
            )
            verdict = checker(assertion.config, text)
            message = (
                "malformed findings — cannot verify"
                if verdict.malformed
                else f"recall={verdict.recall} fp={verdict.fp_count}"
            )
            return EvalResult(
                evaluator=self.name,
                checks=[
                    EvalCheck(
                        name="critical_check",
                        passed=verdict.critical_pass,
                        score=1.0 if verdict.critical_pass else 0.0,
                        message=message,
                        details=verdict.model_dump(),
                    )
                ],
            )
```

(The LLM-judge fallback below this block is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/atp-method && uv run pytest tests/test_evaluator.py -v`
Expected: PASS. (The old `test_findings_match_critical_uses_matcher_not_judge` / `test_findings_match_malformed_is_distinct_from_missed` tests use `grader_type: findings_match` in their config — update those configs to `"checker": "findings_match"` in this step, since `grader_type`-based dispatch is gone.)

- [ ] **Step 5: Commit**

```bash
git add packages/atp-method/atp_method/evaluators/case_evaluator.py packages/atp-method/tests/test_evaluator.py
git commit -m "feat(method): dispatch critical_check via checker registry (Phase A-1)"
```

---

## Task 8: Migrate the two code-review cases + end-to-end load test

**Files:**
- Modify: `method/cases/code-review/case-code-review-sqli-clean-001.yaml`
- Modify: `method/cases/code-review/case-code-review-sqli-moderate-001.yaml`
- Test: `packages/atp-method/tests/test_cases_load.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# packages/atp-method/tests/test_cases_load.py
"""The shipped code-review cases load + validate under the new grader shape."""

import json
from pathlib import Path

import jsonschema
import yaml

from atp_method.loader import load_case
from atp_method.schema import AgentEvalCase

ROOT = Path(__file__).resolve().parents[3]
CASES = sorted((ROOT / "method" / "cases" / "code-review").glob("*.yaml"))
SCHEMA = json.loads((ROOT / "method" / "agent-eval-case.schema.json").read_text())


def test_cases_present() -> None:
    assert len(CASES) == 2


def test_cases_validate_pydantic_and_contract() -> None:
    for path in CASES:
        doc = yaml.safe_load(path.read_text())
        # pydantic
        case = AgentEvalCase.model_validate(doc)
        assert case.grader.type == "programmatic"
        assert case.grader.checker == "findings_match"
        # JSON contract (the canonical cross-project schema)
        jsonschema.validate(doc, SCHEMA)
        # loader path
        td = load_case(path)
        assert td.assertions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/atp-method && uv run pytest tests/test_cases_load.py -v`
Expected: FAIL — cases still say `type: findings_match` (now invalid in both pydantic and contract).

- [ ] **Step 3: Migrate both cases**

In each of the two YAML files, change the grader header from:

```yaml
grader:
  type: findings_match
```

to:

```yaml
grader:
  type: programmatic
  checker: findings_match
```

Leave `expected_findings`, `must_not_flag`, `rubric`, `critical_check`, `scoring` unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/atp-method && uv run pytest tests/test_cases_load.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add method/cases/code-review/*.yaml packages/atp-method/tests/test_cases_load.py
git commit -m "feat(method): migrate code-review cases to programmatic+checker findings_match (Phase A-1)"
```

---

## Task 9: Full regression + quality gates

**Files:** none (verification only)

- [ ] **Step 1: Run the full method + affected core suites**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"   # repo root
uv run pytest tests/unit/core tests/unit/evaluators -q
cd packages/atp-method && uv run pytest -q
```
Expected: all PASS. If the harness `method/run_pipe_check.py` is exercised by any test, confirm it still drives the migrated cases (its grading goes through the evaluator, which now uses the checker).

- [ ] **Step 2: Lint + types**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"   # repo root
uv run ruff format atp/evaluators/checkers atp/evaluators/findings/checker.py packages/atp-core/atp/core/results.py
uv run ruff check atp/ packages/atp-method
uv run pyrefly check
```
Expected: ruff clean; pyrefly 0 errors.

- [ ] **Step 3: Confirm the divergence is closed**

Run:
```bash
uv run python -c "
import json
from atp_method.schema import GraderType
d=json.load(open('method/agent-eval-case.schema.json'))
enum=d['properties']['grader']['properties']['type']['enum']
import typing
pyd=set(typing.get_args(GraderType))
print('json enum =', enum)
print('pydantic  =', sorted(pyd))
assert 'findings_match' not in enum and 'findings_match' not in pyd, 'divergence not closed'
print('OK: findings_match is a checker, not a grader.type, in both')
"
```
Expected: prints `OK: ...`.

- [ ] **Step 4: Commit any formatting**

```bash
git add -A
git commit -m "chore(phase-a1): formatting + divergence-closed verification" || echo "nothing to commit"
```

---

## Self-Review (completed during authoring)

- **Spec coverage (§6):** checker registry under `programmatic` (Tasks 2–3), uniform `CaseVerdict` (Task 1), close schema↔pydantic divergence (Tasks 4–5), migrate cases (Task 8). Envelope/harness/taxonomy are explicitly Phase A-2 (scope guard).
- **Type consistency:** `CaseVerdict` fields (`critical_pass`, `malformed`, `recall`, `precision`, `fp_count`, `rubric_score`, `details`, `grader_version`) are defined once (Task 1) and consumed identically in the findings checker (Task 3) and the evaluator (Task 7). `Checker = Callable[[dict, str | None], CaseVerdict]` is used consistently. `grader.checker` flows schema (Task 4) → contract (Task 5) → loader config key `"checker"` (Task 6) → evaluator dispatch (Task 7) → case YAML (Task 8).
- **Placeholders:** none — every code/test step carries full content.
- **Known follow-through:** Tasks 4 and 7 update pre-existing tests that asserted the old `type: findings_match` shape; the steps call this out explicitly so the engineer migrates rather than deletes coverage.
