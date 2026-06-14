# Spec: structured outputs and deterministic scoring for `atp-method`

**Status:** draft
**Created:** 2026-06-14
**Scope:** `method/`, `packages/atp-method/`, ATP protocol/evaluator integration,
and method case authoring workflow.

## Overview

`atp-method` currently runs methodology cases by presenting an instruction to an
agent and scoring the returned `ATPResponse`. The ATP envelope is structured, but
the answer under test is often plain text inside `artifacts[].content`. Some
flows ask the model to emit JSON, but that JSON is still treated as text until a
checker parses it. This makes the LLM judge too central for cases that should be
deterministically verifiable.

The target workflow is: every method case defines an explicit structured output
contract, the agent response includes a structured artifact that satisfies that
contract, and the primary evaluator checks that artifact deterministically.
LLM-as-judge remains available only for optional, non-gating quality criteria
that cannot be expressed as schema or field-level checks.

The design keeps `atp-method` as the owner of method-specific case semantics,
while reusing ATP primitives where they already fit. `ArtifactStructured` is the
preferred response carrier. The existing `artifact` evaluator already validates
structured artifacts against JSON Schema. The existing checker registry and
`findings_match` checker remain the right path for code-review finding matching.
The main new contract is an `output_contract` plus deterministic checks in the
agent-eval-case format.

## Specification

| Component | Signature | Description |
|-----------|-----------|-------------|
| `AgentEvalCase.output_contract` | `OutputContract | None` | Declares the structured artifact the agent must return for this case. Required for new active method cases after migration. |
| `OutputContract` | `artifact_name: str`, `content_type: str`, `schema: dict`, `format_instruction: str | None` | Names the expected structured artifact and the JSON Schema for its `data`. |
| `Grader.checks` | `list[DeterministicCheck]` | Ordered deterministic checks that hard-gate the case unless marked non-critical. Replaces prose-only `critical_check` as the source of truth. |
| `DeterministicCheck` | `type: str`, `critical: bool = True`, `config: dict` | Machine-readable check definition. Supported initial types: `schema`, `findings_match`, `json_path_equals`, `json_path_absent`, `json_path_contains`. |
| `Grader.optional_judge` | `list[RubricItem] | None` | Optional LLM-judge rubric for subjective aspects. Non-critical by default. |
| `atp_method.loader._assertions` | `def _assertions(case: AgentEvalCase) -> list[Assertion]` | Emits deterministic assertions first, all critical by default, followed by optional rubric assertions. |
| `ArtifactEvaluator` reuse | assertion type `schema` | Validates `ArtifactStructured.data` against `output_contract.schema`. |
| Checker registry reuse | `get_checker(checker_name)` | Runs domain-specific deterministic checks such as `findings_match`. |
| `StructuredOutputEvaluator` | `async def evaluate(task, response, trace, assertion) -> EvalResult` | Provides generic JSON-path field checks if no existing evaluator covers them. |
| Spawner output contract | `ATPResponse.artifacts[]: ArtifactStructured` | Spawners should emit structured artifacts directly instead of JSON encoded as markdown/text where the test expects structured output. |

## Data Models

### Method case output contract

```yaml
output_contract:
  artifact_name: answer
  content_type: application/json
  schema:
    type: object
    required: [requirements]
    properties:
      requirements:
        type: array
        items:
          type: object
          required: [obligation, actor, condition, deadline]
          properties:
            obligation: {type: string}
            actor: {type: string}
            condition: {type: [string, "null"]}
            deadline: {type: [string, "null"]}
  format_instruction: >
    Return one structured artifact named answer whose data matches this schema.
```

`output_contract.schema` is JSON Schema and validates the `data` field of an
`ArtifactStructured` response artifact. It does not validate the surrounding
`ATPResponse` envelope.

### Deterministic grader checks

```yaml
grader:
  type: programmatic
  critical_check: >
    Requirement 2 has no stated deadline and must not receive a fabricated value.
  checks:
    - type: schema
      critical: true
      config:
        artifact_name: answer
    - type: json_path_equals
      critical: true
      config:
        artifact_name: answer
        path: $.requirements[1].deadline
        expected: null
  optional_judge:
    - criterion: requirement wording is concise and faithful to the source
      weight: 1.0
  scoring: >
    Fail if any critical deterministic check fails. Otherwise score equals the
    deterministic pass score, with optional judge rubric reported separately.
```

`critical_check` remains as human-readable methodology context. `grader.checks`
becomes the machine-readable source of truth for pass/fail.

### Code-review findings contract

The code-review family should stop treating a JSON array as markdown text. The
target response artifact is:

```json
{
  "type": "structured",
  "name": "findings",
  "content_type": "application/json",
  "data": {
    "findings": [
      {
        "rule_id": "SEC-011",
        "file": "diff",
        "anchor": "f\"SELECT * FROM users WHERE id = {user_id}\"",
        "severity": "critical",
        "fix": "Use a parameterized query."
      }
    ]
  }
}
```

The schema should require `findings` as an array. `findings_match` should compare
`data.findings` against `expected_findings` and `must_not_flag`. During migration,
the checker may keep accepting legacy JSON text from `ArtifactFile.content`, but
new spawners and cases should use `ArtifactStructured`.

## Business Rules

- New regression or held-out method cases must define `output_contract`.
- Probe cases may temporarily omit `output_contract`, but they must not be
  promoted to regression until deterministic checks exist.
- The primary pass/fail result must come from deterministic checks, not from an
  LLM judge.
- All deterministic checks are critical unless they explicitly set
  `critical: false`.
- A malformed structured output is a deterministic failure and must set details
  that reporters can aggregate as `malformed=true`.
- Optional judge rubrics cannot make a failed deterministic case pass.
- Optional judge rubrics should be reported as secondary score components, not
  as the benchmark routing score.
- `critical_check` remains required for methodology readability, but it must
  semantically describe the same failure guarded by `grader.checks`.
- Structured artifacts are preferred over JSON encoded inside a markdown or text
  file artifact.
- A checker may support legacy text artifacts during migration, but the target
  contract is structured artifact input.

## Functional Requirements

- [ ] Extend `method/agent-eval-case.schema.json` with `output_contract`,
      `grader.checks`, and `grader.optional_judge`.
- [ ] Extend `packages/atp-method/atp_method/schema.py` with matching Pydantic
      models and validators.
- [ ] Validate that `output_contract.artifact_name` is referenced by every
      deterministic check that needs an artifact.
- [ ] Validate that `schema` checks require an `output_contract`.
- [ ] Validate that `findings_match` checks require `expected_findings`; compliant
      cases may use an empty list.
- [ ] Update `atp_method.loader` to emit a critical `schema` assertion for
      `output_contract.schema`.
- [ ] Update `atp_method.loader` to emit deterministic check assertions before
      optional rubric assertions.
- [ ] Add a generic structured-output evaluator only for field-level checks that
      are not covered by the existing artifact evaluator or checker registry.
- [ ] Update `AgentEvalCaseEvaluator` so the LLM judge is not used for
      `critical_check` when deterministic checks are present.
- [ ] Preserve backward compatibility for existing cases that only define the
      current `critical_check` and optional `rubric`.
- [ ] Update method spawner prompt envelopes to instruct agents to return the
      named structured artifact shape.
- [ ] Update code-review spawners to emit `ArtifactStructured(name="findings")`
      when possible.
- [ ] Update `findings_match` to read structured findings from
      `ArtifactStructured.data.findings`, with legacy text parsing as fallback.
- [ ] Update benchmark reporting so `score` remains deterministic pass rate and
      optional judge scores are separate components.
- [ ] Update `CASE_GENERATOR.md` to require output contracts and deterministic
      checks for generated cases.

## Implementation Tasks

- [ ] Add schema-contract tests for `output_contract`, deterministic checks,
      optional judge rubric, and invalid cross-field combinations.
- [ ] Add Pydantic model tests for the same cases in `packages/atp-method/tests`.
- [ ] Add loader tests showing a method case becomes:
      `artifact_exists` or `schema` assertions, deterministic field/checker
      assertions, then optional `method_rubric`.
- [ ] Add evaluator tests for malformed structured output, missing structured
      artifact, schema failure, field mismatch, and successful deterministic pass.
- [ ] Add migration tests proving existing YAML cases still load and run.
- [ ] Update code-review sample cases to define `output_contract` for `findings`.
- [ ] Update req-extraction sample cases to define `output_contract` for
      extracted requirements.
- [ ] Update `method/spawners/claude_code_shim.py` and
      `method/spawners/anthropic_api_shim.py` to normalize model output into the
      structured artifact contract.
- [ ] Update `method/run_pipe_check.py` to consume deterministic structured
      result details, including `malformed`.
- [ ] Run targeted tests for `packages/atp-method`, evaluator checkers, and
      benchmark reporting.

## Migration Strategy

Phase 1 keeps the current case format valid and adds the new fields as optional.
The loader uses deterministic checks when present; otherwise it falls back to the
existing `method_critical_check` behavior. This keeps current examples runnable.

Phase 2 migrates the existing `code-review` and `req-extraction` families. These
families become the reference examples for generated cases. Code-review uses the
existing `findings_match` checker. Requirement extraction uses schema validation
plus JSON-path equality or absence checks.

Phase 3 changes authoring policy: new active cases must include
`output_contract` and deterministic checks. LLM judge rubrics remain allowed, but
only as optional quality signals.

Phase 4 deprecates prose-only critical grading for method cases. The field
`critical_check` remains in the methodology schema as human-readable context, but
it no longer drives pass/fail when structured checks are available.

## Open Decisions

- JSON path syntax should be standardized before implementation. Use one
  library-backed syntax consistently rather than ad hoc path parsing.
- Decide whether `schema` assertions should require an existing artifact check
  or whether schema failure on a missing artifact is sufficient.
- Decide whether `ArtifactStructured.data` should allow top-level arrays. It is
  currently typed as `dict[str, Any]`, so array outputs should be wrapped, for
  example `{findings: [...]}`.
- Decide whether optional judge scores affect per-task display score or remain
  purely diagnostic. The benchmark routing score should remain deterministic.
