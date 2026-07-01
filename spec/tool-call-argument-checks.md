# Spec: Tool-call argument checks

**Status:** proposed first slice
**Created:** 2026-06-21
**Scope:** ATP `BehaviorEvaluator`, `ATPEvent(event_type="tool_call")` traces,
normal ATP assertions, and method-case follow-up integration.

## Overview

ATP already records tool-call behavior in run traces and has coarse behavior
assertions for required tools, forbidden tools, tool-call counts, and errors.
That is enough to check that a tool was used, but not enough to verify an
agentic workflow such as "save approved data to DB" versus "email a manager
when source data is invalid." Those workflows need parameter-aware tool-call
checks.

The first slice extends `BehaviorEvaluator` so normal ATP assertions can verify
tool-call names, status, input arguments, output values, forbidden calls, and
optional call order. The evaluator should use `ATPEvent` trace data as the
canonical evidence source. Mock-tool recorder entries are useful diagnostics,
but are not the evaluator input unless they are surfaced as trace events by the
runner or adapter.

This feature intentionally starts with normal assertions, not method
`grader.checker`. Normal assertions already receive `(task, response, trace,
assertion)` and are the natural place for behavior checks. Method
`grader.checker` currently receives only checker config and response text; using
it for tool calls would require checker API changes or special-case evaluator
plumbing. A later method-loader slice may translate method YAML behavior fields
into normal ATP assertions.

## Specification

| Component | Signature | Description |
|-----------|-----------|-------------|
| `BehaviorEvaluator._evaluate_behavior_config` | `def _evaluate_behavior_config(trace: list[ATPEvent], response: ATPResponse, config: dict[str, Any]) -> list[EvalCheck]` | Add parameter-aware tool-call checks while preserving existing config keys. |
| `BehaviorEvaluator._check_expected_tool_calls` | `def _check_expected_tool_calls(trace: list[ATPEvent], config: dict[str, Any]) -> EvalCheck` | Verify each expected tool-call pattern matches at least one trace event. |
| `BehaviorEvaluator._check_forbidden_tool_calls` | `def _check_forbidden_tool_calls(trace: list[ATPEvent], config: dict[str, Any]) -> EvalCheck` | Fail if any forbidden tool-call pattern matches a trace event. |
| `ToolCallMatcher.matches` | `def matches(event: ATPEvent) -> ToolCallMatchResult` | Match one `tool_call` event against tool name, status, input, and output expectations. |
| `json_path.resolve` reuse | `resolve(data: Any, path: str) -> tuple[bool, Any]` | Reuse the deterministic single-node JSONPath subset for argument paths. |

## Assertion Shape

The first slice adds new keys under the existing `behavior` assertion config:

```yaml
assertions:
  - type: behavior
    critical: true
    config:
      expected_tool_calls:
        - tool: db_save
          status: success
          input_matches:
            - path: $.table
              equals: document_results
            - path: $.record.status
              equals: approved
            - path: $.record.source_ids
              exists: true
      forbidden_tool_calls:
        - tool: send_email
      tool_call_order: expected
```

The negative branch can use the same shape:

```yaml
assertions:
  - type: behavior
    critical: true
    config:
      expected_tool_calls:
        - tool: send_email
          input_matches:
            - path: $.to
              equals: manager@example.com
            - path: $.reason
              equals: data_mismatch
      forbidden_tool_calls:
        - tool: db_save
```

Existing behavior config keys remain valid:

```yaml
config:
  must_use_tools: [file_read]
  forbidden_tools: [delete_file]
  min_tool_calls: 1
  max_tool_calls: 5
  no_errors: true
```

## Data Models

### `ToolCallExpectation`

```python
class ToolCallExpectation(BaseModel):
    tool: str
    status: str | None = None
    input_matches: list[PayloadMatch] = Field(default_factory=list)
    output_matches: list[PayloadMatch] = Field(default_factory=list)
```

`tool` is required. `status`, `input_matches`, and `output_matches` are
optional filters. An expectation matches a trace event only when every supplied
filter matches.

### `PayloadMatch`

```python
class PayloadMatch(BaseModel):
    path: str
    equals: Any | None = None
    exists: bool | None = None
    absent: bool | None = None
```

`path` uses the existing deterministic JSONPath subset: `$`, `.key`, and
`[index]`. Exactly one of `equals`, `exists`, or `absent` must be supplied.

### `ToolCallMatchResult`

```python
class ToolCallMatchResult(BaseModel):
    matched: bool
    reason: str | None = None
    event_sequence: int | None = None
```

This internal result supports clear failure messages without exposing verbose
trace payloads by default.

## Trace Payload Rules

The canonical event is:

```json
{
  "event_type": "tool_call",
  "sequence": 2,
  "payload": {
    "tool": "db_save",
    "input": {"table": "document_results"},
    "output": {"id": "row-123"},
    "status": "success"
  }
}
```

Business rules:

- Only `event_type == "tool_call"` events are considered.
- `payload.tool` is required for matching.
- `payload.input` is the canonical input object.
- `payload.args` may be accepted as a backward-compatible alias when
  `payload.input` is absent.
- `payload.output` is optional and only checked when `output_matches` is
  configured.
- Missing `status` fails only expectations that specify `status`.
- Non-object input or output fails path-based matches except `absent`.

## Matching Rules

- `expected_tool_calls` is existential: each expectation must match at least one
  tool-call event.
- The same event may satisfy more than one expectation unless
  `tool_call_order: expected` is configured.
- `forbidden_tool_calls` fails if any configured forbidden expectation matches
  any event.
- `input_matches` and `output_matches` are conjunctive: every payload match must
  pass.
- `equals` uses normal Python value equality after JSONPath resolution.
- `exists: true` passes when the path resolves.
- `absent: true` passes when the path does not resolve.
- Invalid JSONPath syntax is a configuration error surfaced as a failed check,
  not an agent failure.

### Ordering

`tool_call_order` is optional:

| Value | Behavior |
|-------|----------|
| omitted / `any` | Expectations may match in any order. |
| `expected` | Expectations must match in the order listed, using increasing event sequence numbers. |

No first-slice support is required for exact full trace equality, repeated-call
cardinality, or interleaving constraints.

## EvalCheck Output

Expected-call success:

```python
EvalCheck(
    name="expected_tool_calls",
    passed=True,
    message="All expected tool calls matched",
    details={
        "expected": 2,
        "matched": 2,
        "matches": [
            {"tool": "file_read", "event_sequence": 1},
            {"tool": "db_save", "event_sequence": 3},
        ],
    },
)
```

Expected-call failure:

```python
EvalCheck(
    name="expected_tool_calls",
    passed=False,
    message="Missing expected tool call: db_save",
    details={
        "missing": [
            {
                "tool": "db_save",
                "reason": "$.record.status expected 'approved', got 'rejected'",
            }
        ],
        "observed_tools": ["file_read", "send_email"],
    },
)
```

Forbidden-call failure:

```python
EvalCheck(
    name="forbidden_tool_calls",
    passed=False,
    message="Forbidden tool call matched: send_email",
    details={
        "violations": [
            {"tool": "send_email", "event_sequence": 4}
        ],
    },
)
```

## Business Rules

- Existing behavior checks must remain backward compatible.
- Parameter-aware checks must use trace events, not response artifact content.
- The evaluator must not call external tools, replay tool calls, or inspect
  provider-specific logs.
- Parameter-aware checks must be deterministic and side-effect free.
- Tool-call payload matching must be partial by default: extra input/output
  fields are allowed.
- Sensitive payload values may appear in traces; compact reporters should show
  concise mismatch paths and values, not full payload dumps by default.
- Method `grader.checker` integration is deferred. The first supported method
  integration should be loader-level translation into normal ATP assertions.

## Functional Requirements

- [ ] Extend `BehaviorEvaluator` to support `expected_tool_calls`.
- [ ] Extend `BehaviorEvaluator` to support `forbidden_tool_calls`.
- [ ] Support matching by tool name and optional status.
- [ ] Support `input_matches` and `output_matches` using deterministic JSONPath.
- [ ] Support `equals`, `exists`, and `absent` payload match operators.
- [ ] Support optional `tool_call_order: expected`.
- [ ] Preserve existing `must_use_tools`, `forbidden_tools`, `min_tool_calls`,
      `max_tool_calls`, and `no_errors` behavior.
- [ ] Accept `payload.args` as a compatibility alias when `payload.input` is
      absent.
- [ ] Return concise, structured `EvalCheck.details` for missing expected calls
      and forbidden-call violations.
- [ ] Document the new assertion shape in the test-format/configuration docs.

## Implementation Tasks

- [ ] Add unit tests in `tests/unit/evaluators/test_behavior.py` for expected
      tool-call matches, mismatches, forbidden parameterized calls, missing
      payload paths, invalid JSONPath config, and ordered matching.
- [ ] Add helper functions or small internal models for tool-call extraction and
      payload matching.
- [ ] Reuse `atp.evaluators.json_path.resolver.resolve` for path resolution.
- [ ] Extend `_evaluate_behavior_config` with the new config keys.
- [ ] Add direct assertion handling only if a new assertion type is introduced;
      otherwise keep the first slice under `type: behavior`.
- [ ] Add integration coverage proving runner-collected `ATPEvent` traces feed
      the new behavior checks.
- [ ] Update `docs/reference/test-format.md`,
      `docs/reference/configuration.md`, and any evaluator docs that list
      behavior assertion capabilities.

## Deferred

- Method `agent-eval-case` schema fields for behavior checks.
- Loader translation from method case behavior fields to normal ATP assertions.
- Changing deterministic checker signatures to receive trace data.
- Full tool-call trace equality assertions.
- Cardinality per expectation such as exactly N matching calls.
- Regex, numeric comparison, subset, or schema-based payload operators.
- Automatic conversion of mock-tool recorder entries into trace events.
