# Compact Reporting Implementation Plan

**Source spec:** [`spec/compact-reporting.md`](../../spec/compact-reporting.md)

## Overview

Add a compact `summary` reporter that derives from existing `SuiteReport` and
`TestReport` data, with structured `expected` / `received` failures for the
first supported path: `agent_eval_case` + `citation_grounding@1`.

## Assumptions

- The first slice registers one new reporter type: `summary`.
- The reporter supports console output by default and JSON output through a
  reporter config option, not by changing the existing detailed `json` reporter.
- Only one structured extractor is in scope: failed `critical_check` results
  from `agent_eval_case` where `check.details["grader_version"]` is
  `citation_grounding@1`.
- Existing detailed reporters remain unchanged.
- Because page and line-range mismatches need received values, the
  `citation_grounding` checker may add bounded diagnostic fields to its existing
  `CaseVerdict.details["results"]` entries. This is still scoped to one
  checker/evaluator path.

## TDD Workflow

For every phase that touches `test_*.py` files, follow the repository hook:

1. Run `touch .Codex/.test-edit-mode` from the repo root.
2. Use an isolated `$unit-tester` run for the test-writing step.
3. Run the targeted tests and confirm they fail for the intended reason.
4. Run `rm -f .Codex/.test-edit-mode`.
5. Implement production code.
6. Run the same targeted tests until green.

Do not edit tests directly from the implementation context.

## Architecture Alignment

- Reuse `SuiteReport`, `TestReport`, `EvalResult`, and `EvalCheck` from
  `atp.core.results`; do not add a second result pipeline.
- Keep compact summary construction in reporter code, because it is a
  presentation concern and should not change runner or scoring semantics.
- Register `summary` through the existing `ReporterRegistry`, matching the
  built-in reporter pattern used by `console`, `json`, `html`, and `junit`.
- Keep deterministic checker diagnostics in `CaseVerdict.details`, preserving
  the current `AgentEvalCaseEvaluator` behavior of attaching
  `verdict.model_dump()` to `EvalCheck.details`.
- Prefer focused unit tests around model construction and rendering. Add CLI
  coverage only if current CLI tests already exercise `output_format` choices
  cheaply.

## Phase 1: Checker Diagnostics For Received Values

**Goal:** make `citation_grounding@1` details sufficient for structured
`expected` / `received` reporting without parsing raw agent output in the
reporter.

### Tasks

- [ ] Extend `atp/evaluators/citation_grounding/checker.py` (MODIFIED) so failed
  expected citation checks include bounded diagnostic fields in each result:
  `path`, `expected`, `received`, and optionally `field`.
- [ ] Preserve the existing `reason`, `expected`, `ok`, `malformed`, and
  `grader_version` behavior so detailed reports remain compatible.
- [ ] Keep forbidden-source diagnostics bounded and source-path only.

### Public Interfaces

```python
def citation_grounding_check(
    config: dict[str, Any],
    text: str | None,
) -> CaseVerdict: ...
```

`CaseVerdict.details["results"][i]` may include:

```python
{
    "path": str,
    "field": str | None,
    "expected_value": Any,
    "received_value": Any,
}
```

### Tests First

- Test file: existing `citation_grounding` checker unit test file.
- Key cases:
  - source mismatch includes expected source and received source
  - page mismatch includes expected page and received page
  - line-range mismatch includes expected range and received range
  - missing output path stays bounded and does not include raw output
  - malformed output shape is unchanged

### Acceptance Criteria

- [ ] Targeted citation-grounding tests pass.
- [ ] Existing checker consumers still receive a valid `CaseVerdict`.
- [ ] No raw artifact body is added to checker details.

## Phase 2: Compact Summary Models And Failure Extraction

**Goal:** create the reusable compact summary model and one scoped structured
extractor.

### Tasks

- [ ] Add `atp/reporters/summary_models.py` (NEW) with:
  - `CompactSuiteSummary`
  - `CompactTestSummary`
  - `CompactFailure`
  - `FailureReasonCount`
- [ ] Add `atp/reporters/summary_extractor.py` (NEW) with:
  - `CompactFailureExtractor`
  - `CitationGroundingFailureExtractor`
- [ ] Implement deterministic status rules from the spec:
  error, malformed, failed, passed.
- [ ] Implement one-failure-per-test selection order from the spec.
- [ ] Implement fallback failures for unsupported evaluators/checks.
- [ ] Implement truncation/bounding for `expected`, `received`, and `message`
  values before they reach output models.

### Public Interfaces

```python
class CompactSuiteSummary(BaseModel):
    @classmethod
    def from_report(
        cls,
        report: SuiteReport,
        *,
        include_passed: bool = False,
        max_failures: int | None = None,
    ) -> CompactSuiteSummary: ...


class CompactTestSummary(BaseModel):
    @classmethod
    def from_test(cls, test: TestReport) -> CompactTestSummary: ...


class CompactFailureExtractor:
    def extract(self, test: TestReport) -> CompactFailure | None: ...


class CitationGroundingFailureExtractor:
    def extract(
        self,
        evaluator: str,
        check: EvalCheck,
    ) -> CompactFailure | None: ...
```

### Tests First

- Test file: `tests/unit/reporters/test_summary_models.py`
- Test file: `tests/unit/reporters/test_summary_extractor.py`
- Key cases:
  - passing suite produces no failures and success is true
  - failed source mismatch maps to `value_mismatch`
  - page mismatch and line-range mismatch expose standard
    `expected` / `received` values
  - missing output path maps to `missing_value`
  - forbidden source maps to `forbidden_value`
  - malformed `CaseVerdict` maps to `malformed_output`
  - unsupported failed check falls back to concise message
  - execution error wins over failed checks
  - `max_failures` truncates failures and sets `truncated_failures`

### Acceptance Criteria

- [ ] Targeted summary model and extractor tests pass.
- [ ] Compact JSON model has stable field names matching the spec.
- [ ] Unsupported evaluators still produce actionable compact failures.

## Phase 3: Summary Reporter Rendering

**Goal:** expose the compact model through a reporter that can render console
text or JSON.

### Tasks

- [ ] Add `atp/reporters/summary_reporter.py` (NEW).
- [ ] Implement `SummaryReporter.report(report: SuiteReport) -> None`.
- [ ] Support reporter config:
  - `output_file`
  - `output`
  - `format`: `console` or `json`
  - `indent`
  - `include_passed`
  - `max_failures`
  - `use_colors`
- [ ] Render console output with `expected:` and `received:` lines when present.
- [ ] Render JSON by serializing `CompactSuiteSummary`, not by reusing the
  detailed `JSONReporter` shape.
- [ ] Avoid including artifacts, prompts, traces, raw model output, or corpus
  file contents.

### Public Interfaces

```python
class SummaryReporter(Reporter):
    def __init__(
        self,
        output_file: Path | str | None = None,
        output: TextIO | None = None,
        format: Literal["console", "json"] = "console",
        indent: int | None = 2,
        include_passed: bool = False,
        max_failures: int | None = None,
        use_colors: bool = True,
    ) -> None: ...

    @property
    def name(self) -> str: ...

    def report(self, report: SuiteReport) -> None: ...
```

### Tests First

- Test file: `tests/unit/reporters/test_summary_reporter.py`
- Key cases:
  - console passing suite shows compact counts
  - console failing suite shows only failures by default
  - console value mismatch includes `expected:` and `received:`
  - JSON output matches `compact-summary-v1`
  - file output creates parent directories
  - `include_passed=True` includes passed test summaries
  - large values are truncated or omitted

### Acceptance Criteria

- [ ] Targeted reporter tests pass.
- [ ] Console output is readable in CI logs without detailed artifacts.
- [ ] JSON output is parseable and stable.

## Phase 4: Registry And CLI Wiring

**Goal:** make the compact reporter selectable through existing reporter
creation paths.

### Tasks

- [ ] Update `atp/reporters/registry.py` (MODIFIED) to register `summary`.
- [ ] Update `atp/reporters/__init__.py` (MODIFIED) to export
  `SummaryReporter`.
- [ ] Update CLI output format choices in `atp/cli/main.py` (MODIFIED) if the
  CLI currently restricts formats to known reporter names.
- [ ] Pass summary reporter config from existing CLI options. Add only the
  smallest needed option if JSON summary cannot be selected otherwise.
- [ ] Update user-facing docs or help text only where the reporter list is
  documented.

### Public Interfaces

```python
ReporterRegistry().create("summary", config) -> SummaryReporter
```

### Tests First

- Test file: `tests/unit/reporters/test_registry.py`
- Optional CLI test file: existing CLI test covering `run` output formats.
- Key cases:
  - registry lists `summary`
  - registry creates `SummaryReporter`
  - registry passes summary config
  - CLI accepts `--output-format summary` if output formats are constrained

### Acceptance Criteria

- [ ] Registry tests pass.
- [ ] CLI can select the reporter by name.
- [ ] Existing reporter registrations are unchanged.

## Phase 5: Verification And Documentation Sync

**Goal:** verify the slice end to end and keep docs aligned with the delivered
behavior.

### Tasks

- [ ] Update `spec/compact-reporting.md` (MODIFIED) only if implementation
  requires a scoped contract adjustment.
- [ ] Update README or method docs only if reporter usage is documented there.
- [ ] Run focused test suite:
  - `uv run pytest tests/unit/reporters -q`
  - citation-grounding checker tests
- [ ] Run quality checks if the change is ready for merge:
  - `uv run ruff format .`
  - `uv run ruff check .`
  - `uv run pyrefly check`

### Acceptance Criteria

- [ ] Focused unit tests pass.
- [ ] Formatting and lint checks pass.
- [ ] Type checking passes or any pre-existing unrelated failures are recorded.
- [ ] The compact report does not expose raw artifacts, prompts, traces, model
  output, or corpus contents.

## Risks & Dependencies

- The spec expects `received` values for page and line-range mismatches, but the
  current checker details do not expose the actual citation object. Phase 1
  addresses this inside the one in-scope checker instead of making reporters
  parse raw output.
- CLI format wiring may already be generic. If so, Phase 4 should remain limited
  to registry/export tests and no CLI changes.
- The compact summary models should stay reporter-local. Moving them into core
  would increase the blast radius without a current dashboard or persistence
  requirement.
- Multiple failures per test are explicitly deferred. Do not build grouping or
  full failure arrays in the first slice.
