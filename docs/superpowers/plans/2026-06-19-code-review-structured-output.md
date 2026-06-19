# Code-review structured-output migration (#1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate code-review findings to the spec-faithful structured-output contract — agent returns an object `{"findings": [...]}` validated against `output_contract.schema` (malformed gate), graded by `findings_match` — matching how req-extraction already works via `json_path`.

**Architecture:** The codebase's "structured output" is **JSON text + jsonschema validation against `output_contract.schema`**, not `ArtifactStructured.data` end-to-end (agents emit text; `json_path_check` is the reference). This migration makes the findings path object-aware and adds the same schema gate `json_path` already has, then declares `output_contract` on the 15 code-review cases (which flips them to `GENERIC_ENVELOPE` + the case's `format_instruction` via the existing `build_prompt`).

**Tech Stack:** Python 3.12, pydantic, jsonschema, pytest + anyio, uv, YAML cases.

**Spec:** `spec/structured-method-output.md` (reconciled draft).

## Global Constraints

- `uv` only (never pip); run tools via `uv run`.
- Type hints on all code; `uv run pyrefly check` exits 0 (project baseline suppresses pre-existing; add no NEW errors).
- `uv run ruff format .` + `uv run ruff check .` clean; line length 88.
- Async tests use `anyio` (not asyncio).
- Branch: `r07/code-review-structured-output` (already created; never work on `main`).
- Top-level structured outputs are **objects, not arrays** (`{"findings": [...]}`) — spec rule.
- `output_contract.schema.required` must align with the strict `Finding` model: `rule_id`, `anchor`, `severity` (severity enum `critical|major|minor`); `file`/`fix` are optional. A schema stricter than `Finding` would reject outputs the matcher accepts, and vice versa.
- **Non-goal:** do NOT wire `ArtifactStructured.data` through the adapters/shims. The agent keeps emitting JSON text; only the method grading layer changes. (Full structured-artifact transport is deferred — out of scope.)
- **Non-goal:** do NOT change `REVIEW_ENVELOPE`. Cases with `output_contract` already route through `GENERIC_ENVELOPE` + `format_instruction` in `build_prompt`; `REVIEW_ENVELOPE` stays as the legacy fallback.
- Legacy bare-array findings output must still parse (migration safety) — object form is preferred, array is fallback.
- The live re-baseline of the routing signal happens at the planned weekend pipe-check run; this plan's tests are deterministic and call no agents.

---

### Task 1: Findings matcher — object-form unwrap + schema gate

**Files:**
- Modify: `atp/evaluators/findings/matcher.py` (`parse_findings`, `grade_findings`)
- Modify: `atp/evaluators/findings/checker.py` (`findings_check`)
- Test: `tests/unit/evaluators/` (findings matcher/checker tests — create `test_findings_structured.py` if no single obvious home; otherwise extend the existing findings test module)

**Interfaces:**
- Produces: `parse_findings(text: str | None) -> list[dict] | None` now also unwraps `{"findings": [...]}` (object → inner list), keeping bare-array as fallback.
- Produces: `grade_findings(text, expected, must_not_flag, schema: dict[str, Any] | None = None) -> MatchResult` — when `schema` is given, the parsed JSON is `jsonschema.validate`-d against it and a violation yields `malformed=True` before matching.
- Consumes (by `findings_check`): `config.get("schema")` (the loader already places `output_contract.json_schema` there — see `loader.py:84`).

- [ ] **Step 1: Write the failing tests**

Create/extend the findings test module:

```python
from atp.evaluators.findings.checker import findings_check
from atp.evaluators.findings.matcher import grade_findings, parse_findings

_OBJ_SCHEMA = {
    "type": "object",
    "required": ["findings"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["rule_id", "anchor", "severity"],
                "properties": {
                    "rule_id": {"type": "string"},
                    "file": {"type": "string"},
                    "anchor": {"type": "string"},
                    "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                    "fix": {"type": "string"},
                },
            },
        }
    },
}
_EXPECTED = [{"rule_ids": ["SEC-011"], "anchor": "f\"...{user_id}\"", "severity": "critical"}]


def test_parse_findings_unwraps_object_form() -> None:
    out = parse_findings('{"findings": [{"rule_id": "SEC-011", "anchor": "a", "severity": "critical"}]}')
    assert out == [{"rule_id": "SEC-011", "anchor": "a", "severity": "critical"}]


def test_parse_findings_still_accepts_legacy_array() -> None:
    out = parse_findings('[{"rule_id": "SEC-011", "anchor": "a", "severity": "critical"}]')
    assert out == [{"rule_id": "SEC-011", "anchor": "a", "severity": "critical"}]


def test_grade_findings_schema_violation_is_malformed() -> None:
    # findings present but as a bare array → violates the object schema
    r = grade_findings(
        '[{"rule_id": "SEC-011", "anchor": "a", "severity": "critical"}]',
        _EXPECTED, [], schema=_OBJ_SCHEMA,
    )
    assert r.malformed is True
    assert r.critical_pass is False


def test_grade_findings_valid_object_matches() -> None:
    r = grade_findings(
        '{"findings": [{"rule_id": "SEC-011", "anchor": "f\\"...{user_id}\\"", "severity": "critical"}]}',
        _EXPECTED, [], schema=_OBJ_SCHEMA,
    )
    assert r.malformed is False


def test_findings_check_threads_schema_from_config() -> None:
    v = findings_check(
        {"expected_findings": _EXPECTED, "must_not_flag": [], "schema": _OBJ_SCHEMA},
        '[{"rule_id": "SEC-011", "anchor": "a", "severity": "critical"}]',
    )
    assert v.malformed is True  # bare array rejected by the object schema
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/evaluators/test_findings_structured.py -v`
Expected: FAIL — `parse_findings` returns `None` for the object form; `grade_findings` has no `schema` parameter (TypeError).

- [ ] **Step 3: Implement the matcher changes**

In `atp/evaluators/findings/matcher.py`, add a raw loader and make `parse_findings` object-aware:

```python
def _load_json(text: str | None) -> Any:
    """Parse JSON from agent text, tolerating one surrounding markdown fence.

    Returns the parsed value (object/array/...) or None if it is not JSON.
    """
    if text is None:
        return None
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    try:
        return json.loads(stripped)
    except (ValueError, TypeError):
        return None


def parse_findings(text: str | None) -> list[dict[str, Any]] | None:
    """Findings array from agent text. Accepts the structured object form
    ``{"findings": [...]}`` (preferred) and a bare array (legacy fallback)."""
    data = _load_json(text)
    if isinstance(data, dict) and isinstance(data.get("findings"), list):
        return data["findings"]
    return data if isinstance(data, list) else None
```

Then thread the schema gate through `grade_findings` (add `import jsonschema` at the top of the module):

```python
def grade_findings(
    text: str | None,
    expected: list[dict[str, Any]],
    must_not_flag: list[dict[str, Any]],
    schema: dict[str, Any] | None = None,
) -> MatchResult:
    """Parse, optionally schema-validate, strictly validate, and match in one pass.

    When ``schema`` is given (the case's ``output_contract.schema``), the parsed
    JSON is validated against it first; a violation is ``malformed`` — the
    contract gate, mirroring ``json_path_check``.
    """
    raw = _load_json(text)
    if raw is None:
        return _malformed_result()
    if schema is not None:
        try:
            jsonschema.validate(raw, schema)
        except (jsonschema.ValidationError, jsonschema.SchemaError):
            return _malformed_result()
    parsed = raw["findings"] if isinstance(raw, dict) else raw
    if not isinstance(parsed, list):
        return _malformed_result()
    findings = validate_findings(parsed)
    if findings is None:
        return _malformed_result()
    return match_findings(findings, expected, must_not_flag)
```

In `atp/evaluators/findings/checker.py`, pass the schema through:

```python
def findings_check(config: dict[str, Any], text: str | None) -> CaseVerdict:
    """Run the deterministic findings matcher and map it to a CaseVerdict.

    ``config`` carries ``expected_findings`` / ``must_not_flag`` (ground truth)
    and optional ``schema`` (the output contract). ``text`` is the agent output.
    """
    r = grade_findings(
        text,
        config.get("expected_findings", []),
        config.get("must_not_flag", []),
        schema=config.get("schema"),
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/evaluators/test_findings_structured.py -v`
Expected: all pass.

- [ ] **Step 5: Run the existing findings + method suites for no regression**

Run: `uv run pytest tests/unit/evaluators -k findings -q && uv run pytest packages/atp-method/tests -q`
Expected: all pass (legacy bare-array tests unaffected; `schema=None` path unchanged).

- [ ] **Step 6: Commit**

```bash
git add atp/evaluators/findings/matcher.py atp/evaluators/findings/checker.py tests/unit/evaluators/test_findings_structured.py
git commit -m "feat(findings): object-form findings + output_contract schema gate"
```

---

### Task 2: Declare `output_contract` on the 15 code-review cases

**Files:**
- Modify: all 15 `method/cases/code-review/*.yaml`
- Test: `packages/atp-method/tests/test_cases_load.py` (loads), plus a new prompt/grade test (`packages/atp-method/tests/test_code_review_structured.py`)

**Interfaces:**
- Consumes: object-aware `grade_findings` + schema gate (Task 1); the existing loader behavior (`loader.py:84` puts `output_contract.json_schema` into the critical-check config; `loader.py:117` puts `output_contract` into `input_data`); `build_prompt` (`envelopes.py:60`) uses `GENERIC_ENVELOPE` + `format_instruction` when `output_contract` is present.
- Produces: every code-review case carries an `output_contract` whose schema requires the object findings shape.

- [ ] **Step 1: Write the failing tests**

Add `packages/atp-method/tests/test_code_review_structured.py`:

```python
from pathlib import Path

from atp_method.envelopes import build_prompt, get_envelope
from atp_method.loader import load_case

_CASES = sorted(
    (Path(__file__).resolve().parents[2] / "method" / "cases" / "code-review").glob(
        "*.yaml"
    )
)


def test_every_code_review_case_declares_object_output_contract() -> None:
    from atp_method.schema import AgentEvalCase
    import yaml

    for path in _CASES:
        case = AgentEvalCase.model_validate(yaml.safe_load(path.read_text()))
        assert case.output_contract is not None, path.name
        schema = case.output_contract.json_schema
        assert schema.get("type") == "object", path.name
        assert "findings" in schema.get("required", []), path.name


def test_code_review_prompt_uses_object_format_instruction() -> None:
    # With output_contract present, build_prompt switches to the generic
    # envelope + the case's object format_instruction (not the array envelope).
    td = load_case(_CASES[0])
    request = {
        "task": {
            "description": td.task.description,
            "input_data": td.task.input_data,
        }
    }
    prompt = build_prompt(request, get_envelope("review"))
    assert '"findings"' in prompt
    assert "JSON array of findings" not in prompt  # the old review envelope text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/atp-method/tests/test_code_review_structured.py -v`
Expected: FAIL — cases have no `output_contract` yet; the prompt still uses the array envelope.

- [ ] **Step 3: Add the `output_contract` block to every code-review case**

Append this block to each `method/cases/code-review/*.yaml` (top-level key, same indentation as `grader:`). The schema and instruction are identical across the 15 cases (the findings shape is shared); keep them inline per case to match the self-contained req-extraction case pattern:

```yaml
output_contract:
  artifact_name: findings
  content_type: application/json
  schema:
    type: object
    required: [findings]
    properties:
      findings:
        type: array
        items:
          type: object
          required: [rule_id, anchor, severity]
          properties:
            rule_id: {type: string}
            file: {type: string}
            anchor: {type: string}
            severity: {type: string, enum: [critical, major, minor]}
            fix: {type: string}
  format_instruction: >
    Return ONLY a JSON object with a single key "findings" whose value is an
    array. Each finding is an object with keys: "rule_id" (the rule/CWE id),
    "file", "anchor" (the exact offending code substring), "severity"
    (critical|major|minor), "fix". If the code is compliant, return
    {"findings": []}. No prose, no markdown fence.
```

Do NOT remove or change each case's existing `grader:` block (`checker: findings_match`, `expected_findings`, `must_not_flag` stay as-is — the schema gate is additive).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/atp-method/tests/test_code_review_structured.py packages/atp-method/tests/test_cases_load.py -v`
Expected: all pass.

- [ ] **Step 5: Grade an object-form output end-to-end through the case config**

Add to `test_code_review_structured.py` (proves a compliant object output passes and a bare array is now malformed for a real case's schema):

```python
def test_object_output_grades_through_case_schema() -> None:
    import yaml

    from atp_method.schema import AgentEvalCase

    from atp.evaluators.findings.matcher import grade_findings

    # Pick the moderate SQLi case (one seeded SEC-011 defect).
    path = next(p for p in _CASES if "sqli-moderate" in p.name)
    case = AgentEvalCase.model_validate(yaml.safe_load(path.read_text()))
    schema = case.output_contract.json_schema
    expected = [f.model_dump() for f in (case.grader.expected_findings or [])]
    must_not = [m.model_dump() for m in (case.grader.must_not_flag or [])]

    anchor = expected[0]["anchor"]
    good = (
        '{"findings": [{"rule_id": "SEC-011", "anchor": '
        + f'{anchor!r}'.replace("'", '"')
        + ', "severity": "critical"}]}'
    )
    r_ok = grade_findings(good, expected, must_not, schema=schema)
    assert r_ok.malformed is False
    assert r_ok.critical_pass is True

    # The pre-migration bare-array form now violates the object contract.
    r_bad = grade_findings(good.removeprefix('{"findings": ').removesuffix("}"),
                           expected, must_not, schema=schema)
    assert r_bad.malformed is True
```

Run: `uv run pytest packages/atp-method/tests/test_code_review_structured.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add method/cases/code-review/*.yaml packages/atp-method/tests/test_code_review_structured.py
git commit -m "feat(cases): declare object output_contract on code-review cases"
```

---

### Task 3: Full re-validation + status sync

**Files:**
- Modify: `TODO.md` (mark #1 done), `spec/structured-method-output.md` (check the relevant Implementation Tasks)

- [ ] **Step 1: Format, lint, type-check**

```bash
uv run ruff format .
uv run ruff check atp/evaluators/findings packages/atp-method method/cases/code-review
uv run pyrefly check
```
Expected: format clean; ruff "All checks passed!"; pyrefly exits 0 (no NEW errors vs baseline).

- [ ] **Step 2: Run all touched suites**

```bash
uv run pytest tests/unit/evaluators -k "findings or json_path" -q
uv run pytest packages/atp-method/tests -q
```
Expected: all pass — object form graded, legacy array still parses, req-extraction `json_path` untouched.

- [ ] **Step 3: Sync TODO + spec status**

In `TODO.md`, change the `#1` line from `[~]` to `[x]` and note: code-review migrated to object `output_contract` + `findings_match` schema gate (2026-06-19); structured-output now uniform across both verticals. In `spec/structured-method-output.md`, tick the Implementation Tasks now satisfied (`output_contract` emits schema; `findings_match` prefers structured/object data with legacy fallback; programmatic checker hard-gates).

- [ ] **Step 4: Commit**

```bash
git add TODO.md spec/structured-method-output.md
git commit -m "docs: mark #1 structured-output migration done (code-review)"
```

---

## Self-Review

- **Spec coverage:** "Emit a schema assertion from output_contract" → already in loader, exercised by Task 1's schema gate + Task 2 cases. "findings_match prefers structured artifact data, legacy text fallback" → Task 1 (`parse_findings` object form + bare-array fallback); note our "structured" = object-form JSON text, not `ArtifactStructured.data` (stated non-goal). "Prompt output shape from output_contract" → Task 2 (`build_prompt` already threads `format_instruction`). "Top-level objects not arrays" → Task 2 schema. "Keep json_path semantics" → untouched. "critical_pass from programmatic checker" → unchanged (findings_match already hard-gates). Migration tests (legacy still loads/runs) → Task 1 bare-array tests + Task 2 grader block preserved.
- **Placeholder scan:** none — every step carries real code/schema/commands.
- **Type consistency:** `grade_findings(text, expected, must_not_flag, schema=None) -> MatchResult` defined in Task 1, consumed by `findings_check` (Task 1) and the Task 2 grade test with the same signature; `parse_findings(text) -> list|None` consistent; `output_contract.schema` object shape in Task 2 matches the `_OBJ_SCHEMA` used in Task 1 and aligns with the strict `Finding` model (`rule_id`/`anchor`/`severity`, severity enum).
