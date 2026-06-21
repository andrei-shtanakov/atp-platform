# Tool-Call Argument Checks

## Overview

Extend `BehaviorEvaluator` so normal ATP behavior assertions can verify tool
call names, statuses, input arguments, output values, forbidden calls, and
optional expected ordering from `ATPEvent(event_type="tool_call")` traces.

## Architecture Alignment

This change belongs in `atp/evaluators/behavior.py` because behavior assertions
already receive the full evaluation context: `task`, `response`, `trace`, and
`assertion`. The first slice should preserve the existing behavior assertion
keys and add parameter-aware checks inside the existing `type: behavior` config.

The matcher should reuse `atp.evaluators.json_path.resolver.resolve` and its
`InvalidPath` error type instead of adding a second path parser. The existing
`tests/unit/evaluators/test_behavior.py` file is the right unit-test home, but
it must be edited through the required isolated `$unit-tester` flow.

Method `grader.checker` integration remains out of scope for this plan. A later
method-loader change can translate method YAML behavior fields into normal ATP
assertions once this evaluator behavior is stable.

## Phase 1: Expected Tool Calls

### Tasks

- [ ] Add internal matcher data structures - `atp/evaluators/behavior.py` (MODIFIED)
  - `ToolCallExpectation`
  - `PayloadMatch`
  - `ToolCallMatchResult`
- [ ] Add trace event extraction helpers - `atp/evaluators/behavior.py` (MODIFIED)
  - `def _iter_tool_call_events(trace: list[ATPEvent]) -> list[ATPEvent]`
  - `def _tool_call_input(payload: dict[str, Any]) -> Any`
  - `def _tool_call_output(payload: dict[str, Any]) -> Any`
- [ ] Add payload match evaluation - `atp/evaluators/behavior.py` (MODIFIED)
  - `def _match_payload(data: Any, matches: list[PayloadMatch]) -> ToolCallMatchResult`
  - `def _match_payload_rule(data: Any, rule: PayloadMatch) -> ToolCallMatchResult`
- [ ] Add expected call checking - `atp/evaluators/behavior.py` (MODIFIED)
  - `def _check_expected_tool_calls(trace: list[ATPEvent], config: dict[str, Any]) -> EvalCheck`
- [ ] Wire `expected_tool_calls` into `_evaluate_behavior_config` without changing
      existing behavior keys - `atp/evaluators/behavior.py` (MODIFIED)

### Tests First

- Test file: `tests/unit/evaluators/test_behavior.py`
- Use `$unit-tester` after running `touch .Codex/.test-edit-mode`; remove the
  marker with `rm -f .Codex/.test-edit-mode` after the isolated test-writing
  step.
- Key test cases:
  - Expected tool call passes when tool, status, and all `input_matches` match.
  - Expected tool call fails when no event has the expected tool name.
  - Expected tool call fails when the tool exists but an `equals` rule differs.
  - `payload.args` is accepted when `payload.input` is absent.
  - `output_matches` checks `payload.output` only when configured.
  - Invalid JSONPath creates a failed `expected_tool_calls` check with a clear
    configuration-oriented message.

### Acceptance Criteria

- [ ] Existing behavior assertions still pass with unchanged config shapes.
- [ ] `expected_tool_calls` is existential: each expectation must match at least
      one tool-call event.
- [ ] Extra fields in tool input or output do not cause failures.
- [ ] Missing `status` fails only expectations that specify `status`.
- [ ] Targeted unit tests pass:

```bash
uv run pytest tests/unit/evaluators/test_behavior.py -v
```

## Phase 2: Forbidden Tool Call Patterns

### Tasks

- [ ] Add forbidden expectation matching - `atp/evaluators/behavior.py` (MODIFIED)
  - `def _check_forbidden_tool_calls(trace: list[ATPEvent], config: dict[str, Any]) -> EvalCheck`
- [ ] Wire `forbidden_tool_calls` into `_evaluate_behavior_config` independently
      from existing `forbidden_tools` - `atp/evaluators/behavior.py` (MODIFIED)
- [ ] Keep existing `forbidden_tools` behavior as the coarse backwards-compatible
      tool-name check - `atp/evaluators/behavior.py` (MODIFIED)

### Tests First

- Test file: `tests/unit/evaluators/test_behavior.py`
- Use the isolated `$unit-tester` flow before implementation.
- Key test cases:
  - Forbidden call fails when a matching tool event appears.
  - Forbidden call can match only by tool name.
  - Forbidden call can match by tool name plus input argument rules.
  - Forbidden call passes when the same tool appears with non-matching arguments.
  - `forbidden_tools` and `forbidden_tool_calls` can both be present and produce
    separate checks.

### Acceptance Criteria

- [ ] `forbidden_tool_calls` fails only when a full forbidden pattern matches.
- [ ] Failure details include the violating tool and event sequence.
- [ ] Existing `forbidden_tools` tests remain unchanged and passing.
- [ ] Targeted unit tests pass:

```bash
uv run pytest tests/unit/evaluators/test_behavior.py -v
```

## Phase 3: Ordering

### Tasks

- [ ] Add ordered expected-call matching - `atp/evaluators/behavior.py` (MODIFIED)
  - Reuse the same matcher but require increasing `event.sequence` when
    `tool_call_order: expected` is configured.
- [ ] Treat missing or `any` ordering as the existing unordered matching behavior
      - `atp/evaluators/behavior.py` (MODIFIED)
- [ ] Return a clear failed check for unsupported `tool_call_order` values -
      `atp/evaluators/behavior.py` (MODIFIED)

### Tests First

- Test file: `tests/unit/evaluators/test_behavior.py`
- Use the isolated `$unit-tester` flow before implementation.
- Key test cases:
  - Ordered expectations pass when matching events appear in listed order.
  - Ordered expectations fail when matching events appear out of order.
  - Unordered expectations pass even when events appear in a different order.
  - Unsupported `tool_call_order` value fails with a configuration message.

### Acceptance Criteria

- [ ] `tool_call_order: expected` uses increasing trace event sequence numbers.
- [ ] The same event is not reused for multiple ordered expectations.
- [ ] Omitted or `any` ordering remains unordered.
- [ ] Targeted unit tests pass:

```bash
uv run pytest tests/unit/evaluators/test_behavior.py -v
```

## Phase 4: Documentation And Examples

### Tasks

- [ ] Update behavior assertion docs or add a compact example near existing
      evaluator documentation - documentation file to be selected during
      implementation after checking current docs structure (MODIFIED)
- [ ] Add one positive branch and one negative branch YAML example using
      `expected_tool_calls` and `forbidden_tool_calls` (MODIFIED)
- [ ] Mention that method `grader.checker` support is deferred and normal ATP
      assertions are the supported first path (MODIFIED)

### Tests First

- No test-file edits are expected for documentation-only changes.
- Run release/doc checks only if the touched docs are covered by existing
  validation.

### Acceptance Criteria

- [ ] Users can see how to assert "valid document -> DB save" and "invalid
      document -> manager email" workflows.
- [ ] Documentation names the supported payload fields: `tool`, `status`,
      `input`, `args`, and `output`.
- [ ] Documentation states that checks are partial and extra payload fields are
      allowed.

## Phase 5: Verification And Cleanup

### Tasks

- [ ] Run formatter and lint for touched Python files.
- [ ] Run targeted behavior evaluator tests.
- [ ] Run the broader evaluator test set if matcher code is shared beyond
      `BehaviorEvaluator`.
- [ ] Inspect the final diff for accidental unrelated changes.

### Acceptance Criteria

- [ ] Formatting passes:

```bash
uv run ruff format atp/evaluators/behavior.py
uv run ruff check atp/evaluators/behavior.py
```

- [ ] Targeted tests pass:

```bash
uv run pytest tests/unit/evaluators/test_behavior.py -v
```

- [ ] Broader evaluator tests pass when practical:

```bash
uv run pytest tests/unit/evaluators -v
```

- [ ] `git diff --check` reports no whitespace issues.

## Risks And Dependencies

- Test edits are protected and must go through the isolated `$unit-tester` flow.
- Invalid configuration handling should be explicit but small; do not introduce
  a new config validation framework for this first slice.
- The trace payload contract depends on adapters or runners emitting
  `EventType.TOOL_CALL` events with usable payloads. This evaluator should not
  consume mock-tool recorder files directly.
- Equality uses normal Python value equality after JSONPath resolution. If users
  later need regex, numeric comparisons, schemas, or call cardinality, those
  should be separate follow-up features.
