# Spec: Compact Reporting

**Status:** proposed first slice
**Created:** 2026-06-21
**Scope:** compact console/JSON reporting for ATP suite results, with structured
failure output for one evaluator path: `agent_eval_case` +
`citation_grounding`.

## Overview

ATP already has detailed JSON/HTML reporting for debugging. The next reporting
slice should add a compact view optimized for local iteration, CI logs, and MR
evidence. The compact view answers only the operational questions after a run:
whether the suite passed, aggregate counts, which cases failed, and the shortest
actionable reason for each failure.

The compact report should derive from the existing `SuiteReport` and
`TestReport` models. It must not introduce a second result pipeline, re-run
evaluators, parse raw artifacts, or depend on provider-specific logs.

For the first implementation, structured `expected` / `received` output is
required only for the `agent_eval_case` deterministic checker path when the
check details come from `citation_grounding@1`. Other evaluators and checkers
may fall back to a concise message. This keeps the report contract extensible
without forcing a broad checker retrofit.

## Specification

| Component | Signature | Description |
|-----------|-----------|-------------|
| `CompactSuiteSummary.from_report` | `@classmethod def from_report(cls, report: SuiteReport, *, include_passed: bool = False, max_failures: int | None = None) -> CompactSuiteSummary` | Build the compact suite summary from an existing report. |
| `CompactTestSummary.from_test` | `@classmethod def from_test(cls, test: TestReport) -> CompactTestSummary` | Build compact per-test status, score, duration, and first failure. |
| `CompactFailureExtractor.extract` | `def extract(test: TestReport) -> CompactFailure | None` | Select the most actionable failure from one test. |
| `CitationGroundingFailureExtractor.extract` | `def extract(evaluator: str, check: EvalCheck) -> CompactFailure | None` | Convert `citation_grounding@1` checker details into structured failure output. |
| `SummaryReporter.report` | `def report(self, report: SuiteReport) -> None` | Render compact summary as console text or JSON. |
| `ReporterRegistry.create` | `summary` reporter type | Construct `SummaryReporter` from reporter config. |

## Data Models

### `CompactSuiteSummary`

```python
class CompactSuiteSummary(BaseModel):
    version: Literal["compact-summary-v1"]
    suite_name: str
    agent_name: str
    success: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    malformed_tests: int
    errored_tests: int
    success_rate: float
    duration_seconds: float | None
    runs_per_test: int
    failures: list[CompactTestSummary]
    passed: list[CompactTestSummary] | None = None
    top_failure_reasons: list[FailureReasonCount]
    truncated_failures: int = 0
    error: str | None = None
```

### `CompactTestSummary`

```python
class CompactTestSummary(BaseModel):
    test_id: str
    test_name: str
    status: Literal["passed", "failed", "malformed", "error"]
    score: float | None
    duration_seconds: float | None
    failure: CompactFailure | None = None
```

### `CompactFailure`

```python
class CompactFailure(BaseModel):
    kind: Literal[
        "execution_error",
        "malformed_output",
        "value_mismatch",
        "missing_value",
        "forbidden_value",
        "critical_check_failed",
        "scored_failure",
        "unknown_failure",
    ]
    message: str
    evaluator: str | None = None
    check: str | None = None
    path: str | None = None
    expected: Any | None = None
    received: Any | None = None
```

### `FailureReasonCount`

```python
class FailureReasonCount(BaseModel):
    kind: str
    count: int
```

## Status Rules

Per-test status must be derived deterministically:

1. If `test.error` exists, status is `error`.
2. Else if the selected failure has `kind == "malformed_output"`, status is
   `malformed`.
3. Else if `test.success is False`, status is `failed`.
4. Else status is `passed`.

`failed_tests` remains the existing suite failed count. `malformed_tests` and
`errored_tests` are additional breakdown counts for compact reporting.

## Failure Extraction Rules

The first implementation should select one failure per test. Selection order:

1. Execution error from `test.error`.
2. First failed `EvalCheck` whose details indicate malformed output.
3. First failed `EvalCheck` supported by a structured extractor.
4. First failed `EvalCheck.message`.
5. Fallback to `unknown_failure`.

Only one structured extractor is in scope for the first slice:

```text
evaluator == "agent_eval_case"
check.name == "critical_check"
check.details["grader_version"] == "citation_grounding@1"
```

All other evaluators must still produce a compact failure with `kind`,
`message`, `evaluator`, and `check`, but may leave `path`, `expected`, and
`received` as `None`.

## `citation_grounding@1` Mapping

The source details are already attached to `EvalCheck.details` as a serialized
`CaseVerdict`. The compact extractor should read:

```python
check.details["malformed"]
check.details["details"]["reason"]
check.details["details"]["results"]
```

For malformed verdicts, produce:

```json
{
  "kind": "malformed_output",
  "message": "<details.reason>",
  "evaluator": "agent_eval_case",
  "check": "critical_check"
}
```

For value mismatches, the extractor should map known `reason` strings into
standard test-style output:

```json
{
  "kind": "value_mismatch",
  "message": "citation source mismatch",
  "path": "$.requirements[0].citations.deadline.path",
  "expected": "policy-current.md",
  "received": "archive/policy-2023.md",
  "evaluator": "agent_eval_case",
  "check": "critical_check"
}
```

Supported first-slice structured mappings:

| Checker reason | Failure kind | Path | Expected | Received |
|----------------|--------------|------|----------|----------|
| `expected source <expected>, got <received>` | `value_mismatch` | `<output_path>.path` | expected source | received source |
| `citation page does not match expected page` | `value_mismatch` | `<output_path>.page` | expectation page | citation page |
| `citation line range does not match expected range` | `value_mismatch` | `<output_path>.line_start` / `<output_path>.line_end` as one compact range path | expected range | received range |
| `output_path not found: <path>` | `missing_value` | output path | `citation object` | `missing` |
| `forbidden source cited: <source_path>` | `forbidden_value` | `$.**.path` | `not <source_path>` | source path |

If a reason is not recognized, preserve it as:

```json
{
  "kind": "critical_check_failed",
  "message": "<reason>",
  "evaluator": "agent_eval_case",
  "check": "critical_check"
}
```

## JSON Output

The JSON summary should be stable and suitable for CI parsing:

```json
{
  "version": "compact-summary-v1",
  "suite_name": "req-extraction",
  "agent_name": "anthropic_api",
  "success": false,
  "total_tests": 4,
  "passed_tests": 3,
  "failed_tests": 1,
  "malformed_tests": 0,
  "errored_tests": 0,
  "success_rate": 0.75,
  "duration_seconds": 18.42,
  "runs_per_test": 1,
  "top_failure_reasons": [
    {"kind": "value_mismatch", "count": 1}
  ],
  "failures": [
    {
      "test_id": "case-req-extraction-fabricated-deadline-corpus-clean-001",
      "test_name": "Fabricated deadline corpus clean",
      "status": "failed",
      "score": 0.0,
      "duration_seconds": 4.12,
      "failure": {
        "kind": "value_mismatch",
        "message": "citation source mismatch",
        "evaluator": "agent_eval_case",
        "check": "critical_check",
        "path": "$.requirements[0].citations.deadline.path",
        "expected": "policy-current.md",
        "received": "archive/policy-2023.md"
      }
    }
  ],
  "passed": null,
  "truncated_failures": 0,
  "error": null
}
```

## Console Output

Console output should use standard test-style expected/received lines when
available:

```text
ATP Summary
Suite: req-extraction
Agent: anthropic_api

Result: FAILED
Tests: 3 passed, 1 failed, 0 malformed, 0 error
Success rate: 75.0%
Duration: 18.4s

Failures:
  x case-req-extraction-fabricated-deadline-corpus-clean-001
    score: 0.0/100
    reason: value_mismatch
    check: agent_eval_case:critical_check
    path: $.requirements[0].citations.deadline.path
    expected: policy-current.md
    received: archive/policy-2023.md
```

By default, console output includes only failures, malformed cases, and errors.
Passed cases are included only when configured with `include_passed=True`.

## Business Rules

- Compact reporting must never include full artifacts, prompts, traces, model
  output, or corpus file contents.
- `expected` and `received` should be included only for bounded scalar values or
  small structured values.
- Large values must be omitted or truncated before rendering.
- The first slice must support structured `expected` / `received` only for
  `citation_grounding@1`.
- Unsupported evaluators/checkers must fall back to concise message output.
- JSON output must be stable; console output may change formatting as long as it
  remains readable.
- Detailed reporters must remain unchanged.

## Functional Requirements

- [ ] Add compact summary models.
- [ ] Build compact summaries from existing `SuiteReport`.
- [ ] Add failure extraction with one supported structured path:
      `agent_eval_case` + `citation_grounding@1`.
- [ ] Render structured failures with `expected` and `received` when available.
- [ ] Fall back to concise messages for all other evaluator failures.
- [ ] Add JSON summary output.
- [ ] Add console summary output.
- [ ] Register reporter type `summary`.
- [ ] Support config:
      `format`, `output_file`, `max_failures`, `include_passed`.
- [ ] Exclude artifacts, prompts, traces, model output, and corpus contents from
      compact reports.

## Implementation Tasks

- [ ] Add `atp/reporters/summary.py`.
- [ ] Add `CompactSuiteSummary`, `CompactTestSummary`, `CompactFailure`, and
      `FailureReasonCount`.
- [ ] Add `CompactFailureExtractor`.
- [ ] Add `CitationGroundingFailureExtractor`.
- [ ] Register `summary` in `atp/reporters/registry.py`.
- [ ] Add unit tests in `tests/unit/reporters/test_summary_reporter.py`.
- [ ] Add reporter documentation with console and JSON examples.

## Acceptance Criteria

- [ ] A fully passing suite renders a short success summary.
- [ ] A failed `citation_grounding@1` source mismatch renders
      `expected` / `received`.
- [ ] A failed `citation_grounding@1` missing output path renders
      `missing_value`.
- [ ] A malformed `citation_grounding@1` verdict renders `malformed_output`.
- [ ] A non-`citation_grounding` failed check renders a concise fallback message.
- [ ] Compact JSON excludes full artifacts and raw model output.
- [ ] Console output defaults to failed/malformed/error cases only.

## Deferred

- Structured `expected` / `received` extraction for additional evaluators.
- Multiple failures per test.
- Grouping by family, axis, run mode, or checker.
- Trend comparison across runs.
- HTML compact dashboard widgets.
