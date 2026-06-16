# req-extraction Deterministic Vertical Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the req-extraction fabricated-deadline family from LLM-judged into the project's second *deterministic* vertical via an `output_contract` + a `json_path` checker, so two `task_type`s land in one store.

**Architecture:** A new `json_path` checker mirrors the existing `findings_match` checker exactly — `(config, text) -> CaseVerdict`, parsing the agent's JSON text, optionally validating it against the case `output_contract.schema`, then running single-node path assertions. The case gains `output_contract` (drives the prompt + carries the schema) and `run_mode` (validated against the wired set `{text_out}`). No `ArtifactStructured`, no shim artifact-carrier change, no new run-mode infra.

**Tech Stack:** Python 3.12, pydantic v2, `uv`, pytest (anyio), the agent-eval-case methodology (`packages/atp-method`), the checker registry (`atp/evaluators/checkers`).

**Spec:** `docs/superpowers/specs/2026-06-16-req-extraction-deterministic-vertical-design.md`
**ADR:** `docs/adr/007-test-taxonomy-axes.md`

**Deviation from spec (confirmed at planning):** the spec's Slice 4 ("shims emit `ArtifactStructured`") and the separate `ArtifactEvaluator` schema assertion are dropped. Reason: the checker dispatch (`packages/atp-method/atp_method/evaluators/case_evaluator.py:122`) passes the first artifact's `.content` text to the checker, and `findings_check` already folds parse+validate+match into the checker. `json_path` does the same (parse + schema-validate + assertions in one checker). The shim already emits the model's JSON as `ArtifactFile.content`, so it needs no carrier change — only the prompt changes (Task 3). This is lower-risk and matches the house pattern; `ArtifactStructured` remains a future refinement.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `packages/atp-method/atp_method/schema.py` | Pydantic case model | Add `OutputContract`, `AgentEvalCase.output_contract`, `AgentEvalCase.run_mode`, `Grader.config`; extend `validate_grader_requirements` for `json_path`; add `run_mode` wired-set validator |
| `method/agent-eval-case.schema.json` | JSON Schema contract (parity) | Add `output_contract`, `run_mode`, `grader.config` |
| `atp/evaluators/json_path/__init__.py` | Package marker | Create |
| `atp/evaluators/json_path/resolver.py` | Single-node `$.a[i].b` resolver | Create |
| `atp/evaluators/json_path/checker.py` | `json_path_check(config, text) -> CaseVerdict` | Create |
| `atp/evaluators/checkers/__init__.py` | Register built-in checkers | Register `json_path` |
| `packages/atp-method/atp_method/loader.py` | case → TestDefinition | Serialize `output_contract` into `input_data`; thread `schema`/`assertions` into the critical assertion config |
| `packages/atp-method/atp_method/envelopes.py` | Prompt envelope | `build_prompt` uses a generic envelope + `output_contract.format_instruction` when present, else the review envelope |
| `packages/atp-method/atp_method/taxonomy.py` | task_type ↔ benchmark_id | Map `req-extraction` |
| `method/cases/req-extraction/*.yaml` | The 4 cases | Add `output_contract` + `grader.checker: json_path` + `grader.config.assertions` |
| `packages/atp-method/tests/test_cases_load.py` | Case-load contract test | Cover req-extraction deterministic cases |

Tests live beside existing ones: `tests/unit/evaluators/test_json_path_*.py`, `packages/atp-method/tests/test_*.py`.

---

## Task 1: Schema — `output_contract`, `run_mode`, `grader.config`

**Files:**
- Modify: `packages/atp-method/atp_method/schema.py`
- Modify: `method/agent-eval-case.schema.json`
- Test: `packages/atp-method/tests/test_schema_output_contract.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `packages/atp-method/tests/test_schema_output_contract.py`:

```python
"""Schema support for output_contract, run_mode, and json_path grader config."""

import pytest
from pydantic import ValidationError

from atp_method.schema import AgentEvalCase, Grader, OutputContract


def _minimal_grader(**over) -> dict:
    base = dict(
        type="programmatic",
        checker="json_path",
        critical_check="x must hold",
        scoring="fail if critical fails",
        config={"assertions": [{"path": "$.a", "op": "equals", "expected": 1}]},
    )
    base.update(over)
    return base


def _minimal_case(**over) -> dict:
    base = dict(
        id="case-x-clean-001",
        version=1,
        family="x",
        status="active",
        suite_type="probe",
        capability="correctness",
        construction_axis="output_structure",
        axis_level="clean",
        instruction="do x",
        environment={"tools": ["none"], "side_effects": "none"},
        expected_failure_mode="fails x",
        grader=_minimal_grader(),
        provenance={"author": "a", "created": "2026-06-16"},
        output_contract={
            "artifact_name": "answer",
            "json_schema": {"type": "object"},
            "format_instruction": "return JSON",
        },
        run_mode="text_out",
    )
    base.update(over)
    return base


def test_output_contract_parses_and_aliases_schema() -> None:
    oc = OutputContract.model_validate(
        {"artifact_name": "answer", "schema": {"type": "object"}}
    )
    assert oc.artifact_name == "answer"
    assert oc.json_schema == {"type": "object"}
    # round-trips back to the on-disk key "schema"
    assert oc.model_dump(by_alias=True)["schema"] == {"type": "object"}


def test_case_with_output_contract_and_run_mode_valid() -> None:
    case = AgentEvalCase.model_validate(_minimal_case())
    assert case.run_mode == "text_out"
    assert case.output_contract is not None


def test_run_mode_unwired_tier_rejected() -> None:
    with pytest.raises(ValidationError, match="run_mode"):
        AgentEvalCase.model_validate(_minimal_case(run_mode="workspace"))


def test_run_mode_defaults_to_text_out() -> None:
    doc = _minimal_case()
    del doc["run_mode"]
    assert AgentEvalCase.model_validate(doc).run_mode == "text_out"


def test_json_path_requires_assertions() -> None:
    with pytest.raises(ValidationError, match="assertions"):
        Grader.model_validate(_minimal_grader(config={"assertions": []}))
    with pytest.raises(ValidationError, match="assertions"):
        Grader.model_validate(_minimal_grader(config=None))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/atp-method/tests/test_schema_output_contract.py -q`
Expected: FAIL (ImportError: cannot import `OutputContract`; `run_mode`/`config` unknown).

- [ ] **Step 3: Add the models and validators**

In `packages/atp-method/atp_method/schema.py`:

Add `RunMode` + wired set next to the other `Literal`s (after line 53):

```python
RunMode = Literal["text_out", "read_only_corpus", "workspace"]
# Only text_out is wired today; the loader/validator rejects the rest so a case
# cannot declare fidelity the harness cannot deliver (ADR-007 §3).
WIRED_RUN_MODES = frozenset({"text_out"})
```

Add the `OutputContract` model (after the `Grader` class, before `Provenance`). The on-disk key is `schema`; the Python field is `json_schema` (a pydantic `BaseModel` field literally named `schema` shadows an internal — use an alias):

```python
class OutputContract(BaseModel):
    """The structured artifact the agent must return for this case."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    artifact_name: str = Field(..., min_length=1)
    content_type: str = "application/json"
    # On-disk key is "schema"; Python name avoids the BaseModel.schema shadow.
    json_schema: dict[str, Any] = Field(..., alias="schema")
    format_instruction: str | None = None
```

This needs `Any`: change the import on line 13 to `from typing import Any, Literal`.

Add `config` to `Grader` (after line 134):

```python
    config: dict[str, Any] | None = None
```

Extend `Grader.validate_grader_requirements` (inside the existing method, before `return self` on line 159):

```python
        if self.checker == "json_path":
            assertions = (self.config or {}).get("assertions")
            if not assertions:
                raise ValueError(
                    "checker 'json_path' requires grader.config.assertions "
                    "(a non-empty list)"
                )
```

Add `output_contract` + `run_mode` to `AgentEvalCase` (after line 206, near `grader`):

```python
    output_contract: OutputContract | None = None
    run_mode: RunMode = "text_out"
```

Add a `run_mode` wired-set validator to `AgentEvalCase` (after `validate_volatility_turns`):

```python
    @model_validator(mode="after")
    def validate_run_mode_wired(self) -> AgentEvalCase:
        """Reject run_mode tiers the harness cannot deliver yet (ADR-007 §3)."""
        if self.run_mode not in WIRED_RUN_MODES:
            raise ValueError(
                f"run_mode '{self.run_mode}' is declared but not wired; "
                f"supported: {sorted(WIRED_RUN_MODES)}"
            )
        return self
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/atp-method/tests/test_schema_output_contract.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Add JSON-Schema parity**

In `method/agent-eval-case.schema.json`, add to the top-level `properties` object:

```json
"run_mode": {
  "type": "string",
  "enum": ["text_out", "read_only_corpus", "workspace"],
  "default": "text_out",
  "description": "How the agent interacts with the world (ADR-007). Only text_out is wired; pydantic load rejects unwired tiers."
},
"output_contract": {
  "type": "object",
  "additionalProperties": false,
  "required": ["artifact_name", "schema"],
  "properties": {
    "artifact_name": {"type": "string", "minLength": 1},
    "content_type": {"type": "string"},
    "schema": {"type": "object"},
    "format_instruction": {"type": "string"}
  }
}
```

And add to the `grader` object's `properties`:

```json
"config": {"type": "object"}
```

- [ ] **Step 6: Verify pydantic↔JSON parity test still passes**

Run: `uv run pytest packages/atp-method/tests/ -q`
Expected: PASS (all method tests; the existing `test_cases_load` contract test unaffected — req-extraction cases not yet converted).

- [ ] **Step 7: Format, type-check, commit**

```bash
uv run ruff format packages/atp-method/atp_method/schema.py packages/atp-method/tests/test_schema_output_contract.py
uv run ruff check packages/atp-method/atp_method/schema.py
uv run pyrefly check packages/atp-method/atp_method/schema.py
git add packages/atp-method/atp_method/schema.py method/agent-eval-case.schema.json packages/atp-method/tests/test_schema_output_contract.py
git commit -m "feat(req-extraction): output_contract + run_mode + grader.config schema (Slice 1)"
```

---

## Task 2: `json_path` resolver + checker

**Files:**
- Create: `atp/evaluators/json_path/__init__.py`
- Create: `atp/evaluators/json_path/resolver.py`
- Create: `atp/evaluators/json_path/checker.py`
- Modify: `atp/evaluators/checkers/__init__.py`
- Test: `tests/unit/evaluators/test_json_path_resolver.py`, `tests/unit/evaluators/test_json_path_checker.py` (create)

### 2a. Resolver

- [ ] **Step 1: Write the failing resolver tests**

Create `tests/unit/evaluators/test_json_path_resolver.py`:

```python
"""Single-node JSONPath subset resolver."""

import pytest

from atp.evaluators.json_path.resolver import InvalidPath, resolve


def test_root() -> None:
    assert resolve({"a": 1}, "$") == (True, {"a": 1})


def test_key() -> None:
    assert resolve({"a": 1}, "$.a") == (True, 1)


def test_nested_key_and_index() -> None:
    data = {"reqs": [{"d": None}, {"d": "x"}]}
    assert resolve(data, "$.reqs[1].d") == (True, "x")
    assert resolve(data, "$.reqs[0].d") == (True, None)


def test_missing_key_or_index_not_found() -> None:
    assert resolve({"a": 1}, "$.b") == (False, None)
    assert resolve({"a": [1]}, "$.a[5]") == (False, None)


def test_unsupported_syntax_raises() -> None:
    for bad in ["a", "$.a[*]", "$..a", "$.a[?(@.x)]", "$.a.", "$[0]extra"]:
        with pytest.raises(InvalidPath):
            resolve({"a": 1}, bad)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/evaluators/test_json_path_resolver.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the resolver**

Create `atp/evaluators/json_path/__init__.py`:

```python
"""Deterministic single-node JSONPath subset for the json_path checker."""
```

Create `atp/evaluators/json_path/resolver.py`:

```python
"""A deterministic, single-node subset of JSONPath: `$`, `.key`, `[index]`.

No wildcards, recursion, or filters — those produce multi-node matches, which
are ambiguous for a deterministic grader (ADR-007). A path resolves to exactly
one node or it is "not found"; anything outside the grammar raises InvalidPath.
"""

from __future__ import annotations

import re
from typing import Any

_SEGMENT = re.compile(r"\.([a-zA-Z_][a-zA-Z0-9_]*)|\[(\d+)\]")


class InvalidPath(ValueError):
    """Raised when a path is outside the supported single-node grammar."""


def resolve(data: Any, path: str) -> tuple[bool, Any]:
    """Resolve ``path`` against ``data``.

    Returns ``(found, value)``. ``found`` is False when a key/index along the
    path is missing. Raises :class:`InvalidPath` for unsupported syntax.
    """
    if not isinstance(path, str) or not path.startswith("$"):
        raise InvalidPath(f"path must start with '$': {path!r}")
    rest = path[1:]
    pos = 0
    current: Any = data
    found = True
    for m in _SEGMENT.finditer(rest):
        if m.start() != pos:
            raise InvalidPath(f"unsupported syntax in path: {path!r}")
        pos = m.end()
        key, idx = m.group(1), m.group(2)
        if not found:
            continue
        if key is not None:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                found = False
        else:
            i = int(idx)
            if isinstance(current, list) and 0 <= i < len(current):
                current = current[i]
            else:
                found = False
    if pos != len(rest):
        raise InvalidPath(f"unsupported syntax in path: {path!r}")
    return (found, current if found else None)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/evaluators/test_json_path_resolver.py -q`
Expected: PASS (5 tests).

### 2b. Checker

- [ ] **Step 5: Write the failing checker tests**

Create `tests/unit/evaluators/test_json_path_checker.py`:

```python
"""json_path checker: text JSON -> CaseVerdict."""

import json

from atp.evaluators.json_path.checker import json_path_check


def _cfg(**over) -> dict:
    base = {
        "assertions": [
            {"path": "$.requirements[1].deadline", "op": "equals", "expected": None}
        ]
    }
    base.update(over)
    return base


def test_pass_when_assertion_holds() -> None:
    text = json.dumps({"requirements": [{"deadline": "30d"}, {"deadline": None}]})
    v = json_path_check(_cfg(), text)
    assert v.critical_pass is True
    assert v.malformed is False


def test_fail_when_fabricated() -> None:
    text = json.dumps({"requirements": [{"deadline": "30d"}, {"deadline": "soon"}]})
    v = json_path_check(_cfg(), text)
    assert v.critical_pass is False
    assert v.malformed is False


def test_unparseable_is_malformed() -> None:
    v = json_path_check(_cfg(), "not json at all")
    assert v.malformed is True
    assert v.critical_pass is False


def test_none_text_is_malformed() -> None:
    v = json_path_check(_cfg(), None)
    assert v.malformed is True


def test_schema_violation_is_malformed() -> None:
    cfg = _cfg(schema={"type": "object", "required": ["requirements"]})
    v = json_path_check(cfg, json.dumps({"other": 1}))
    assert v.malformed is True


def test_absent_op() -> None:
    cfg = _cfg(assertions=[{"path": "$.requirements[0].deadline", "op": "absent"}])
    # present -> absent fails
    text = json.dumps({"requirements": [{"deadline": "x"}]})
    assert json_path_check(cfg, text).critical_pass is False
    # missing -> absent passes
    text2 = json.dumps({"requirements": [{}]})
    assert json_path_check(cfg, text2).critical_pass is True


def test_multi_or_bad_path_fails_assertion_not_crash() -> None:
    cfg = _cfg(assertions=[{"path": "$.requirements[*]", "op": "equals", "expected": 1}])
    v = json_path_check(cfg, json.dumps({"requirements": []}))
    assert v.critical_pass is False
    assert v.malformed is False
```

- [ ] **Step 6: Run to verify they fail**

Run: `uv run pytest tests/unit/evaluators/test_json_path_checker.py -q`
Expected: FAIL (module not found).

- [ ] **Step 7: Implement the checker**

Create `atp/evaluators/json_path/checker.py`:

```python
"""json_path checker: parse agent JSON text, optionally schema-validate, run
single-node path assertions, and map to a uniform CaseVerdict.

Mirrors atp/evaluators/findings/checker.py: parse+validate+assert in one
checker, with malformed (not gradeable) distinct from a failed assertion.
"""

import json
from typing import Any

import jsonschema

from atp.core.results import CaseVerdict
from atp.evaluators.json_path.resolver import InvalidPath, resolve

JSON_PATH_CHECKER_VERSION = "json_path@1"


def _assertion_holds(data: Any, assertion: dict[str, Any]) -> bool:
    """Evaluate one single-node assertion. A bad/multi path fails (no crash)."""
    op = assertion.get("op")
    try:
        found, value = resolve(data, assertion.get("path", ""))
    except InvalidPath:
        return False
    if op == "absent":
        return not found
    if not found:
        return False
    if op == "equals":
        return value == assertion.get("expected")
    if op == "contains":
        expected = assertion.get("expected")
        if isinstance(value, (str, list)):
            return expected in value
        return False
    return False


def json_path_check(config: dict[str, Any], text: str | None) -> CaseVerdict:
    """Grade JSON text against config.assertions (+ optional config.schema)."""
    if text is None:
        return _malformed("no agent output")
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return _malformed("output is not valid JSON")

    schema = config.get("schema")
    if schema:
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            return _malformed(f"schema violation: {e.message}")

    assertions = config.get("assertions", [])
    passed = all(_assertion_holds(data, a) for a in assertions)
    return CaseVerdict(
        critical_pass=passed,
        malformed=False,
        details={"assertions": assertions, "n": len(assertions)},
        grader_version=JSON_PATH_CHECKER_VERSION,
    )


def _malformed(reason: str) -> CaseVerdict:
    return CaseVerdict(
        critical_pass=False,
        malformed=True,
        details={"reason": reason},
        grader_version=JSON_PATH_CHECKER_VERSION,
    )
```

Note: `jsonschema` is already a dependency (used by `tests/.../test_cases_load.py` and the artifact evaluator). Confirm with `uv run python -c "import jsonschema"`.

- [ ] **Step 8: Register the checker**

In `atp/evaluators/checkers/__init__.py`, register `json_path` next to `findings_match`. First read the file to match the existing registration style, then add:

```python
from atp.evaluators.json_path.checker import json_path_check
register_checker("json_path", json_path_check)
```

- [ ] **Step 9: Run to verify checker tests + registration**

Run: `uv run pytest tests/unit/evaluators/test_json_path_checker.py -q`
Expected: PASS (7 tests).

Run: `uv run python -c "from atp.evaluators.checkers import get_checker; assert get_checker('json_path')"`
Expected: no error.

- [ ] **Step 10: Format, type-check, commit**

```bash
uv run ruff format atp/evaluators/json_path/ tests/unit/evaluators/test_json_path_resolver.py tests/unit/evaluators/test_json_path_checker.py
uv run ruff check atp/evaluators/json_path/
uv run pyrefly check atp/evaluators/json_path/
git add atp/evaluators/json_path/ atp/evaluators/checkers/__init__.py tests/unit/evaluators/test_json_path_resolver.py tests/unit/evaluators/test_json_path_checker.py
git commit -m "feat(req-extraction): single-node json_path checker (Slice 2)"
```

---

## Task 3: Loader + prompt wiring

**Files:**
- Modify: `packages/atp-method/atp_method/loader.py`
- Modify: `packages/atp-method/atp_method/envelopes.py`
- Test: `packages/atp-method/tests/test_loader_output_contract.py`, `packages/atp-method/tests/test_envelopes.py` (extend)

- [ ] **Step 1: Write the failing loader test**

Create `packages/atp-method/tests/test_loader_output_contract.py`:

```python
"""Loader threads output_contract into input_data + the critical assertion."""

from atp_method.loader import METHOD_CRITICAL_CHECK, case_to_test_definition
from atp_method.schema import AgentEvalCase


def _case() -> AgentEvalCase:
    return AgentEvalCase.model_validate(
        {
            "id": "case-x-clean-001",
            "version": 1,
            "family": "x",
            "status": "active",
            "suite_type": "probe",
            "capability": "correctness",
            "construction_axis": "output_structure",
            "axis_level": "clean",
            "instruction": "extract",
            "environment": {"tools": ["none"], "side_effects": "none"},
            "expected_failure_mode": "fabricates",
            "output_contract": {
                "artifact_name": "answer",
                "schema": {"type": "object", "required": ["requirements"]},
                "format_instruction": "Return ONLY JSON {requirements:[...]}",
            },
            "grader": {
                "type": "programmatic",
                "checker": "json_path",
                "critical_check": "no fabricated deadline",
                "scoring": "fail if critical fails",
                "config": {
                    "assertions": [
                        {"path": "$.requirements[1].deadline", "op": "equals",
                         "expected": None}
                    ]
                },
            },
            "provenance": {"author": "a", "created": "2026-06-16"},
        }
    )


def test_output_contract_goes_into_input_data() -> None:
    td = case_to_test_definition(_case())
    oc = td.task.input_data["output_contract"]
    assert oc["format_instruction"].startswith("Return ONLY JSON")
    # on-disk key stays "schema"
    assert oc["schema"]["required"] == ["requirements"]


def test_critical_assertion_carries_schema_and_assertions() -> None:
    td = case_to_test_definition(_case())
    crit = next(a for a in td.assertions if a.type == METHOD_CRITICAL_CHECK)
    assert crit.config["checker"] == "json_path"
    assert crit.config["schema"]["required"] == ["requirements"]
    assert crit.config["assertions"][0]["path"] == "$.requirements[1].deadline"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/atp-method/tests/test_loader_output_contract.py -q`
Expected: FAIL (`output_contract` not in input_data; `schema`/`assertions` not in assertion config).

- [ ] **Step 3: Update the loader**

In `packages/atp-method/atp_method/loader.py`, extend the critical-assertion config in `_assertions` (inside the `config={...}` dict at lines 72-84) by adding, after `"must_not_flag": [...]`:

```python
                "config": case.grader.config or {},
                "schema": (
                    case.output_contract.json_schema if case.output_contract else None
                ),
                "assertions": (case.grader.config or {}).get("assertions", []),
```

In `case_to_test_definition`, after building `input_data` (after line 108), add:

```python
    if case.output_contract is not None:
        input_data["output_contract"] = case.output_contract.model_dump(
            by_alias=True, exclude_none=True
        )
```

Note: the `json_path` checker reads `config["schema"]` and `config["assertions"]`; the loader copies `output_contract.json_schema` to `config["schema"]` and lifts `assertions` to the top of the assertion config so `json_path_check` finds them under the keys it expects.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/atp-method/tests/test_loader_output_contract.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Write the failing prompt test**

Add to `packages/atp-method/tests/test_envelopes.py`:

```python
def test_build_prompt_uses_format_instruction_when_present() -> None:
    request = {
        "task": {
            "description": "Extract requirements",
            "input_data": {
                "artifacts": [{"id": "doc", "content": "Vendor must submit."}],
                "output_contract": {
                    "artifact_name": "answer",
                    "format_instruction": "Return ONLY JSON {requirements:[...]}",
                },
            },
        }
    }
    from atp_method.envelopes import build_prompt, get_envelope

    prompt = build_prompt(request, get_envelope("review"))
    assert "Return ONLY JSON" in prompt
    assert "Vendor must submit." in prompt
    # the review-specific findings wording must NOT leak in
    assert "senior code reviewer" not in prompt


def test_build_prompt_falls_back_to_review_without_contract() -> None:
    request = {"task": {"description": "Review", "input_data": {"artifacts": []}}}
    from atp_method.envelopes import build_prompt, get_envelope

    prompt = build_prompt(request, get_envelope("review"))
    assert "senior code reviewer" in prompt
```

- [ ] **Step 6: Run to verify it fails**

Run: `uv run pytest packages/atp-method/tests/test_envelopes.py -q`
Expected: FAIL (the new generic-envelope branch does not exist).

- [ ] **Step 7: Update `build_prompt`**

In `packages/atp-method/atp_method/envelopes.py`, add a generic envelope constant after `REVIEW_ENVELOPE` (line 23):

```python
GENERIC_ENVELOPE = (
    "Output ONLY the answer in the exact format requested below, with no prose "
    "and no markdown fence.\n\n{task}"
)
```

Replace the body of `build_prompt` (lines 34-41) with:

```python
def build_prompt(request: dict[str, Any], envelope: str) -> str:
    """Wrap an ATPRequest's task + inline artifacts in an envelope.

    When the case carries an ``output_contract.format_instruction`` (in
    ``task.input_data``), use the generic envelope + that instruction so a
    non-review capability is not forced through the review findings envelope.
    Otherwise fall back to the passed-in envelope (review).
    """
    task = request.get("task") or {}
    body = task.get("description", "")
    input_data = task.get("input_data") or {}
    artifacts = input_data.get("artifacts", []) or []
    for art in artifacts:
        if art.get("content"):
            body += f"\n\n--- {art.get('id', 'artifact')} ---\n{art['content']}"
    contract = input_data.get("output_contract") or {}
    instruction = contract.get("format_instruction")
    if instruction:
        return GENERIC_ENVELOPE.format(task=f"{body}\n\n{instruction}")
    return envelope.format(task=body)
```

- [ ] **Step 8: Run to verify prompt tests + the existing artifact-delivery regression pass**

Run: `uv run pytest packages/atp-method/tests/test_envelopes.py -q`
Expected: PASS (existing review + artifact tests, plus the 2 new ones).

- [ ] **Step 9: Run the full method + shim suites (back-compat gate)**

Run: `uv run pytest packages/atp-method/tests/ -q`
Run: `uv run pytest tests/unit/method_spawners/ -q`
Expected: PASS both. The review vertical (code-review cases) must be unaffected.

- [ ] **Step 10: Format, type-check, commit**

```bash
uv run ruff format packages/atp-method/atp_method/loader.py packages/atp-method/atp_method/envelopes.py packages/atp-method/tests/test_loader_output_contract.py packages/atp-method/tests/test_envelopes.py
uv run ruff check packages/atp-method/atp_method/loader.py packages/atp-method/atp_method/envelopes.py
uv run pyrefly check packages/atp-method/atp_method/loader.py packages/atp-method/atp_method/envelopes.py
git add packages/atp-method/atp_method/loader.py packages/atp-method/atp_method/envelopes.py packages/atp-method/tests/test_loader_output_contract.py packages/atp-method/tests/test_envelopes.py
git commit -m "feat(req-extraction): loader threads output_contract; build_prompt format_instruction (Slice 3)"
```

---

## Task 4: Convert the 4 req-extraction cases + taxonomy

**Files:**
- Modify: `method/cases/req-extraction/case-req-extraction-fabricated-deadline-{clean,moderate,severe,very-severe}-001.yaml`
- Modify: `packages/atp-method/atp_method/taxonomy.py`
- Test: `packages/atp-method/tests/test_cases_load.py` (extend), `packages/atp-method/tests/test_req_extraction_determinism.py` (create)

- [ ] **Step 1: Read the 4 cases to learn each rung's requirement structure**

```bash
for f in clean moderate severe very-severe; do echo "=== $f ==="; cat method/cases/req-extraction/case-req-extraction-fabricated-deadline-$f-001.yaml; done
```

For each case, identify (a) the artifacts' requirement text, (b) which requirement has NO stated deadline (the trap target), and (c) which requirements DO have deadlines. The trap index/value differs per rung — record them. (Per spec Open Decision "Per-rung trap encoding".)

- [ ] **Step 2: Write the failing determinism test**

Create `packages/atp-method/tests/test_req_extraction_determinism.py`:

```python
"""The json_path gate is deterministic on req-extraction ground truth."""

import json
from pathlib import Path

import yaml

from atp.evaluators.checkers import get_checker
from atp_method.schema import AgentEvalCase

ROOT = Path(__file__).resolve().parents[3]
CLEAN = (
    ROOT / "method" / "cases" / "req-extraction"
    / "case-req-extraction-fabricated-deadline-clean-001.yaml"
)


def _config_for(case_path: Path) -> dict:
    case = AgentEvalCase.model_validate(yaml.safe_load(case_path.read_text()))
    cfg = dict(case.grader.config or {})
    if case.output_contract is not None:
        cfg["schema"] = case.output_contract.json_schema
    return cfg


def test_clean_case_is_json_path() -> None:
    case = AgentEvalCase.model_validate(yaml.safe_load(CLEAN.read_text()))
    assert case.grader.checker == "json_path"
    assert case.task_type == "req-extraction"


def test_faithful_output_passes_fabricated_fails() -> None:
    checker = get_checker("json_path")
    cfg = _config_for(CLEAN)
    # The clean case's trap asserts requirements[1].deadline is null (no deadline
    # was stated). Build a faithful answer (null) and a fabricated one (a value).
    faithful = json.dumps(
        {"requirements": [{"deadline": "30 days"}, {"deadline": None}]}
    )
    fabricated = json.dumps(
        {"requirements": [{"deadline": "30 days"}, {"deadline": "ASAP"}]}
    )
    assert checker(cfg, faithful).critical_pass is True
    assert checker(cfg, fabricated).critical_pass is False
    # Determinism: identical across runs.
    assert checker(cfg, faithful).model_dump() == checker(cfg, faithful).model_dump()
```

(Adjust the faithful/fabricated payloads in Step 2 to match the clean case's actual `assertions` after you author them in Step 4.)

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest packages/atp-method/tests/test_req_extraction_determinism.py -q`
Expected: FAIL (cases not yet converted: `checker` is None / `task_type` unset).

- [ ] **Step 4: Convert each case YAML**

For each of the 4 files, add `task_type`, `language` is N/A here (omit), `output_contract`, and replace the `grader` block. Example for the **clean** case (adapt indices/values per rung from Step 1):

```yaml
task_type: req-extraction
run_mode: text_out
output_contract:
  artifact_name: answer
  schema:
    type: object
    required: [requirements]
    properties:
      requirements:
        type: array
        items:
          type: object
          required: [obligation, actor, deadline]
          properties:
            obligation: {type: string}
            actor: {type: string}
            deadline: {type: [string, "null"]}
  format_instruction: >
    Return ONLY a JSON object {"requirements":[...]} where each item has
    obligation, actor, deadline. Use null when a field is not stated.
grader:
  type: programmatic
  checker: json_path
  config:
    assertions:
      - {path: "$.requirements[1].deadline", op: equals, expected: null}
  critical_check: >
    The requirement with no stated deadline must not receive a fabricated value.
  rubric:
    - {criterion: obligations extracted as atomic units, weight: 0.5}
    - {criterion: actor correctly attributed, weight: 0.5}
  scoring: "Fail if the critical json_path check fails; else weighted rubric sum."
```

Keep `type: programmatic` (already there). Remove nothing else; `expected_findings`/`must_not_flag` were never on these cases. Author the per-rung `assertions` to target the actual unstated-deadline index in each rung.

- [ ] **Step 5: Update `taxonomy.py`**

In `packages/atp-method/atp_method/taxonomy.py`, add `req-extraction` to the map. Read the file first; then extend:

```python
TASK_TYPE_TO_BENCHMARK_ID = {
    "review": "code-review",
    "req-extraction": "req-extraction",
}
```

- [ ] **Step 6: Update the case-load contract test**

In `packages/atp-method/tests/test_cases_load.py`, the existing `CASES`/`SCHEMA` target `code-review`. Add a parallel block for req-extraction (read the file first to match style):

```python
REQ_CASES = sorted(
    (ROOT / "method" / "cases" / "req-extraction").glob("*.yaml")
)


def test_req_extraction_cases_present() -> None:
    assert len(REQ_CASES) == 4


def test_req_extraction_cases_are_deterministic() -> None:
    for path in REQ_CASES:
        doc = yaml.safe_load(path.read_text())
        case = AgentEvalCase.model_validate(doc)
        assert case.grader.type == "programmatic"
        assert case.grader.checker == "json_path"
        assert case.task_type == "req-extraction"
        jsonschema.validate(doc, SCHEMA)
```

- [ ] **Step 7: Run the determinism + case-load tests**

Run: `uv run pytest packages/atp-method/tests/test_req_extraction_determinism.py packages/atp-method/tests/test_cases_load.py -q`
Expected: PASS. If the determinism payloads don't match your authored assertions, fix the payloads in Step 2's test (not the assertions).

- [ ] **Step 8: Run the full method suite**

Run: `uv run pytest packages/atp-method/tests/ -q`
Expected: PASS.

- [ ] **Step 9: Format, type-check, commit**

```bash
uv run ruff format packages/atp-method/
uv run ruff check packages/atp-method/atp_method/taxonomy.py
uv run pyrefly check packages/atp-method/atp_method/taxonomy.py
git add method/cases/req-extraction/ packages/atp-method/atp_method/taxonomy.py packages/atp-method/tests/test_cases_load.py packages/atp-method/tests/test_req_extraction_determinism.py
git commit -m "feat(req-extraction): convert 4 cases to deterministic json_path + taxonomy (Slice 4)"
```

---

## Task 5: End-to-end run (offline, then paid control)

**Files:** none (uses `method/run_pipe_check.py` as-is).

- [ ] **Step 1: Offline full-tube run via the fake claude (no spend)**

```bash
CLAUDE_BIN="$(uv run python -c 'import sys; print(sys.executable)') tests/unit/method_spawners/fixtures/fake_claude.py" \
uv run python method/run_pipe_check.py \
  --case-dir method/cases/req-extraction \
  --task-type req-extraction \
  --agents claude_code \
  --db /tmp/reqx_dryrun.db \
  --out-dir /tmp/reqx-out
```

Expected: runs 4 cases, prints a `report_benchmark` summary line with `benchmark_id=req-extraction`, writes the payload + a `benchmark_runs` row. The fake's output won't match the schema/assertions (so `malformed`/fail is expected) — we are verifying the **tube** (loader → prompt → checker → CaseVerdict → store), not the score. Confirm no exceptions and `benchmark_id=req-extraction`.

Verify and clean up:

```bash
sqlite3 /tmp/reqx_dryrun.db "SELECT benchmark_id, agent_id, score FROM benchmark_runs;"
rm -rf /tmp/reqx_dryrun.db /tmp/reqx-out
```

- [ ] **Step 2: Paid control run (clean + one defect rung)**

Requires `ANTHROPIC_API_KEY` in `.env`. Run claude_code only on a 2-case subset first:

```bash
mkdir -p /tmp/reqx-control
cp method/cases/req-extraction/case-req-extraction-fabricated-deadline-{clean,moderate}-001.yaml /tmp/reqx-control/
uv run python method/run_pipe_check.py \
  --case-dir /tmp/reqx-control \
  --task-type req-extraction \
  --agents claude_code \
  --db /tmp/reqx_control.db
```

Expected (validity gate): `malformed_rate≈0` (the model returns valid JSON matching the contract) and a sensible `critical_pass_rate` (the model should NOT fabricate the missing deadline → clean passes). If `malformed_rate` is high, the envelope/format_instruction needs tightening — fix before the full sweep.

Clean up: `rm -rf /tmp/reqx-control /tmp/reqx_control.db`

- [ ] **Step 3: (Optional, when ready) full paid sweep**

```bash
uv run python method/run_pipe_check.py \
  --case-dir method/cases/req-extraction \
  --task-type req-extraction \
  --db _cowork_output/r07-pipecheck/reqx-sweep.db
```

This is the second vertical's first real signal; confirm req-extraction appears in the store/dashboard alongside review.

- [ ] **Step 4: Finish the branch**

Use `superpowers:finishing-a-development-branch` to push and open the PR.

---

## Self-Review

**Spec coverage:**
- `output_contract` + `run_mode` + `grader.config` → Task 1. ✓
- json_path as a checker under `programmatic`, single-node semantics, malformed reuse → Task 2. ✓
- Loader threads contract; build_prompt format_instruction with review fallback → Task 3. ✓
- 4 cases deterministic + taxonomy + test_cases_load → Task 4. ✓
- End-to-end offline + paid control (no-green-smoke) → Task 5. ✓
- Back-compat (review unchanged) hard gate → Task 3 Steps 8-9, Task 4 Step 8. ✓
- Determinism proof → Task 4 Step 2/7. ✓
- `schema` field-name shadow → Task 1 Step 3 (alias). ✓
- run_mode wired-set validation → Task 1 Step 3. ✓
- Per-checker config validation → Task 1 (json_path requires config.assertions). Note: full registry-declared per-checker pydantic config model (ADR-007 §2) is approximated here by the json_path-specific check in `Grader.validate_grader_requirements`; a general "checker declares its config model" hook is deferred (YAGNI — one checker needs config today). This is a conscious narrowing; flag at review.

**Deviations from spec (flagged):** no `ArtifactStructured` / no separate ArtifactEvaluator schema assertion / no shim carrier change (see header). Schema validation folded into the json_path checker. Confirm before execution.

**Type consistency:** `OutputContract.json_schema` (alias `schema`) used consistently in Tasks 1/3/4; checker reads `config["schema"]` + `config["assertions"]`, which the loader populates (Task 3 Step 3). `json_path_check(config, text)` signature matches the registry `Checker` type and `findings_check`.

**Out of scope (parked):** `grader.checks[]`, `optional_judge`, `read_only_corpus`/`workspace`, migrating review to `output_contract`.
