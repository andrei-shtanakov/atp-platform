# Spec: structured outputs and deterministic scoring for `atp-method`

**Status:** reconciled draft
**Created:** 2026-06-14
**Updated:** 2026-06-16
**Disposition:** this draft is reconciled with the #188 direction described in
ADR-007 and the req-extraction deterministic vertical design. It should not
introduce a second grading taxonomy.

## Overview

`atp-method` should make deterministic grading the default path for methodology
cases. The agent response should contain structured artifacts, those artifacts
should be validated against a declared output contract, and domain-specific
checks should run through the existing programmatic checker registry.

This draft originally proposed a generic `grader.checks[]` array. The review
correctly identified that as a competing grading taxonomy next to ADR-006's
closed `GraderType` plus `grader.checker` registry model. The reconciled design
keeps the existing model: one case selects a named programmatic checker, and
that checker owns a registry-declared, load-validated config model. A future
case that truly needs heterogeneous checkers should get its own ADR before
adding a `checks[]` collection.

JSON Schema validation is also not a checker. It is already an ATP artifact
assertion handled by `ArtifactEvaluator` assertion type `schema`. The output
contract should drive that schema assertion, while field-value checks should be
handled by named programmatic checkers such as `json_path` and domain checkers
such as `findings_match`.

LLM-as-judge remains available through the existing `grader.rubric` field. It is
non-gating for method cases and must not duplicate into a second
`optional_judge` field.

## Specification

| Component | Signature | Description |
|-----------|-----------|-------------|
| `AgentEvalCase.output_contract` | `OutputContract | None` | Declares the structured artifact the agent must return. Required for new deterministic verticals after migration. |
| `OutputContract` | `artifact_name: str`, `content_type: str`, `schema: dict`, `format_instruction: str | None` | Names the expected `ArtifactStructured` and JSON Schema for its `data`. |
| `Grader.checker` | `str | None` | Existing selector for deterministic programmatic grading. Examples: `findings_match`, `json_path`. |
| checker config model | `BaseModel` owned by checker registry | Each checker declares and validates its own config shape at case load. |
| `ArtifactEvaluator` | assertion type `schema` | Validates `ArtifactStructured.data` against `output_contract.schema`. |
| `grader.rubric` | `list[RubricItem] | None` | Existing optional LLM-judge rubric. Non-gating and secondary to deterministic critical pass/fail. |
| `critical_check` | `str` | Human-readable methodology explanation of the critical guard. It must describe the deterministic gate but not be judge-graded when structured checks exist. |

## Data Models

### Output contract

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

`ArtifactStructured.data` is currently `dict[str, Any]`, so top-level arrays
must be wrapped. Use `{findings: [...]}` or `{requirements: [...]}`, not a bare
array.

`format_instruction` is the single source for the output-shape clause that is
threaded into the task prompt. Prompt envelopes must not hand-code a second,
different schema description.

### Programmatic checker config

Requirement extraction should use `grader.checker: json_path` or another named
checker with a load-validated config model:

```yaml
grader:
  type: programmatic
  checker: json_path
  checker_config:
    artifact_name: answer
    assertions:
      - path: $.requirements[1].deadline
        op: equals
        expected: null
  rubric:
    - criterion: requirement wording is concise and faithful to the source
      weight: 1.0
  critical_check: >
    Requirement 2 has no stated deadline and must not receive a fabricated value.
  scoring: >
    Fail if the programmatic checker fails. Rubric is non-gating.
```

`json_path` paths must resolve deterministically:

- `equals`, `contains`, and similar value operations require exactly one node.
- `absent` requires zero nodes.
- multi-match paths fail rather than choosing one match implicitly.

### Findings output

Code-review findings should be structured, but still use the existing
`findings_match` checker pattern:

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

During migration, `findings_match` may keep parsing legacy JSON text from
`ArtifactFile.content`, but the target input is `ArtifactStructured.data`.

## Business Rules

- Deterministic pass/fail must come from `type: programmatic` and the named
  checker selected by `grader.checker`.
- `output_contract.schema` drives an ATP artifact `schema` assertion; it is not
  modeled as a checker.
- `grader.rubric` remains the only LLM-judge rubric field and is non-gating.
- `critical_check` remains required prose, but deterministic structured cases
  must re-point `critical_pass` to the programmatic checker result.
- `malformed` and `malformed_rate` are existing result/reporting concepts and
  must be reused rather than redefined.
- Checker configs must be validated during case load.
- Prompt output-shape instructions must be derived from `output_contract` to
  avoid schema/prompt drift.
- Top-level structured outputs must be objects, not arrays.

## Functional Requirements

- [ ] Extend `method/agent-eval-case.schema.json` with `output_contract`.
- [ ] Extend `packages/atp-method/atp_method/schema.py` with `OutputContract`.
- [ ] Add checker-side config validation for named programmatic checkers.
- [ ] Add `checker_config` or equivalent config plumbing under `grader` without
      introducing `grader.checks[]`.
- [ ] Emit an ATP `schema` assertion from `output_contract.schema`.
- [ ] Ensure the method evaluator sources `critical_pass` from the programmatic
      checker when `grader.checker` is present.
- [ ] Keep `grader.rubric` as non-gating secondary scoring.
- [ ] Update prompt-envelope generation so output shape comes from
      `output_contract.format_instruction` and schema metadata.
- [ ] Update `findings_match` to prefer structured artifact data and keep legacy
      text parsing as fallback.
- [ ] Add a `json_path` checker with single-node-required semantics.

## Implementation Tasks

- [ ] Add schema contract tests for `output_contract` and checker config shape.
- [ ] Add Pydantic tests proving invalid checker configs fail at load time.
- [ ] Add loader tests showing `output_contract` emits a `schema` assertion.
- [ ] Add evaluator tests proving programmatic checker failure hard-gates the
      case and rubric cannot rescue it.
- [ ] Add `json_path` tests for exact-one-node, absent-zero-node, missing node,
      and multi-match failure behavior.
- [ ] Add migration tests proving legacy cases without `output_contract` still
      load and run through the existing path.

## Migration Strategy

Phase 1 adds `output_contract` and checker config validation while preserving
legacy cases.

Phase 2 migrates req-extraction to structured artifacts plus the named
programmatic checker model carried by the #188 req-extraction deterministic
vertical.

Phase 3 migrates code-review findings to structured artifacts while keeping
`findings_match` as the named checker.

Phase 4 deprecates judge-graded `critical_check` for method cases that declare
structured output and programmatic checkers. `critical_check` remains as
methodology prose.

## Deferred

- `grader.checks[]` is deferred. It should only be introduced by a future ADR if
  a real case needs multiple heterogeneous deterministic checkers in one case.
- `read_only_corpus` grounding is specified separately in
  `spec/artifact-corpus-grounding.md` and depends on this structured-output path.
