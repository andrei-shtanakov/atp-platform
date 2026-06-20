# Artifact Corpus Grounding Implementation Plan

**Source spec:** [`spec/artifact-corpus-grounding.md`](../../spec/artifact-corpus-grounding.md)

## Overview

Add the first `read_only_corpus` run-mode slice for method cases: a case declares
a text/markdown corpus, ATP verifies it by SHA-256 manifest, materializes it into
a per-run workspace, serves `file_read` through `Context.tools_endpoint`, and
grades JSON-text citations with a deterministic `citation_grounding` checker.

## Assumptions

- The first slice supports only UTF-8 text and markdown files with LF newline
  normalization.
- The first tool-capable method spawner should be `anthropic_api`, because it
  owns the provider message loop and can map provider-native tool calls to the
  HTTP `Context.tools_endpoint`. Product CLI shims can follow once their tool
  integration is explicit.
- The agent prompt may list corpus-relative file paths, but must not inline file
  contents. If true discovery without a path list is required, add a separate
  `file_list` tool in a follow-up; do not overload directory reads.
- `ArtifactStructured.data` transport remains out of scope. Checkers grade the
  primary JSON text artifact, matching the current `json_path` and
  `findings_match` path.
- `read_only_corpus` remains outside `WIRED_RUN_MODES` until schema, corpus
  prep, file serving, one tool-capable spawner, and citation grading all have
  passing tests.

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

- Keep the ADR-007 model: `run_mode` is the interaction tier, and deterministic
  checks stay named programmatic checkers selected by `grader.checker`.
- Reuse `packages/atp-method/atp_method/schema.py` and
  `method/agent-eval-case.schema.json` as the mirrored case contract.
- Reuse `Context.tools_endpoint` and the existing `atp.mock_tools` FastAPI seam
  instead of adding a separate file-delivery mechanism.
- Reuse the existing `CaseVerdict` result shape and `malformed` semantics.
- Add a generic runner request-preparation hook so `atp-method` can prepare
  corpus-backed runs without hard-coding method YAML into adapters.
- Keep checker config validation in the current Pydantic validator style for
  this slice. Registry-declared checker config models remain a later cleanup.

## Current State

| Area | Current behavior | Required change |
| --- | --- | --- |
| Method schema | `RunMode` includes `read_only_corpus`, but `WIRED_RUN_MODES = {"text_out"}` rejects it | Add `artifact_corpus` models while preserving the guard |
| Loader | Emits inline artifacts, constraints, and method assertions | Thread corpus metadata and checker config into `TestDefinition` |
| Runner | Builds `Context(workspace_path=...)` only for sandbox runs | Add request-preparer hook to attach tools endpoint/workspace per run |
| Mock tools | Static response-rule tools at `/tools/call` | Add directory-backed `file_read` handler |
| Spawners | Prompt-only JSON text responses | Add one spawner path that can call `file_read` through tools endpoint |
| Checkers | `findings_match`, `json_path` parse primary JSON text | Add `citation_grounding` with schema gate and line-range validation |

## Phase 1: Schema Models And Guarded Loading

**Goal:** model `artifact_corpus` in JSON Schema and Pydantic without enabling
`read_only_corpus`.

### Tasks

- [ ] Add `artifact_corpus` to `method/agent-eval-case.schema.json` (MODIFIED).
- [ ] Add Pydantic models in `packages/atp-method/atp_method/schema.py`
  (MODIFIED):
  - `CorpusDigest`
  - `ArtifactCorpus`
  - `CorpusFileMetadata`
  - `CorpusMetadata`
- [ ] Add cross-field validation:
  - `artifact_corpus` requires `run_mode: read_only_corpus`
  - `run_mode: read_only_corpus` requires `artifact_corpus`
  - absolute paths, `..`, `~`, null bytes, and empty path segments are rejected
  - `digest.algorithm == "sha256"`
  - `digest.normalization == "lf"`
- [ ] Keep `read_only_corpus` rejected by `validate_run_mode_wired`.
- [ ] Extend `Grader` validation for `checker: citation_grounding` shape without
  registering the checker yet.

### Public Interfaces

```python
class CorpusDigest(BaseModel):
    algorithm: Literal["sha256"]
    manifest_path: str
    normalization: Literal["lf"]


class ArtifactCorpus(BaseModel):
    id: str
    root: str
    include: list[str]
    exclude: list[str] = Field(default_factory=list)
    digest: CorpusDigest
    metadata_path: str | None = None
```

### Tests First

- Test file: `packages/atp-method/tests/test_schema_artifact_corpus.py`
- Key cases:
  - corpus model accepts the spec example paths
  - `artifact_corpus` under `text_out` is rejected
  - `read_only_corpus` without `artifact_corpus` is rejected
  - `read_only_corpus` with a valid corpus still fails the wired-mode guard
  - absolute paths, traversal, `~`, null bytes, duplicate tools, and unsupported
    digest settings are rejected
  - `citation_grounding` config requires non-empty `expected`

### Acceptance Criteria

- [ ] `uv run pytest packages/atp-method/tests/test_schema_artifact_corpus.py -q`
  passes after implementation.
- [ ] Existing schema tests still pass.
- [ ] No corpus-backed case can run yet because `WIRED_RUN_MODES` is unchanged.

## Phase 2: Corpus Resolution, Verification, And Materialization

**Goal:** implement deterministic corpus preparation primitives independent of
the runner.

### Tasks

- [ ] Add `packages/atp-method/atp_method/corpus.py` (NEW).
- [ ] Add `packages/atp-method/atp_method/corpus_manifest.py` (NEW) for a
  documented manifest generation helper.
- [ ] Resolve include/exclude patterns relative to the case file and corpus root.
- [ ] Parse `manifest.sha256`, reject duplicates and unsafe paths.
- [ ] Hash LF-normalized text content and compare selected files exactly against
  manifest entries.
- [ ] Load optional `corpus.meta.yaml`; treat it as semantic metadata, not
  identity.
- [ ] Materialize selected files into a fresh run workspace while preserving
  relative paths.
- [ ] Build a one-based line index over the same normalized content used for
  hashing.

### Public Interfaces

```python
@dataclass(frozen=True)
class ResolvedCorpus:
    corpus_id: str
    root: Path
    files: tuple[ResolvedCorpusFile, ...]
    manifest_path: Path
    metadata_path: Path | None


class CorpusResolver:
    def resolve(self, case_path: Path, corpus: ArtifactCorpus) -> ResolvedCorpus: ...


class CorpusIntegrityVerifier:
    def verify(self, resolved: ResolvedCorpus) -> CorpusVerificationResult: ...


class CorpusMaterializer:
    def materialize(
        self, resolved: ResolvedCorpus, workspace: Path
    ) -> MaterializedCorpus: ...
```

### Tests First

- Test file: `packages/atp-method/tests/test_corpus.py`
- Key cases:
  - include/exclude expansion is canonical and sorted
  - selected files must exactly match manifest paths
  - duplicate manifest paths fail
  - missing selected file, extra selected file, and hash mismatch fail
  - CRLF source content hashes the same normalized LF representation used for
    line indexing
  - metadata is optional unless later checker config references it
  - materialization copies files into an isolated workspace and rejects symlinks
    or resolved paths escaping the corpus root

### Acceptance Criteria

- [ ] `uv run pytest packages/atp-method/tests/test_corpus.py -q` passes.
- [ ] Manifest helper can generate stable `manifest.sha256` output for a fixture
  corpus.
- [ ] No runtime path is enabled yet.

## Phase 3: Directory-Backed `file_read`

**Goal:** serve materialized corpus files through the existing mock-tools HTTP
surface.

### Tasks

- [ ] Add dynamic handler support to `atp/mock_tools/server.py` (MODIFIED) while
  preserving existing static `MockTool` behavior.
- [ ] Add `atp/mock_tools/file_tools.py` (NEW) with a directory-backed
  `file_read` handler.
- [ ] Add `atp/mock_tools/runtime.py` (NEW) with a small async context manager
  that starts `create_mock_app(...)` on an ephemeral localhost port for real
  spawner calls.
- [ ] Ensure valid `file_read` returns normalized text plus basic metadata.
- [ ] Reject unknown files, traversal attempts, absolute paths, null bytes,
  directory reads, and non-text files.
- [ ] Record calls through the existing `CallRecorder`.

### Public Interfaces

```python
class ToolHandler(Protocol):
    async def __call__(self, call: ToolCall) -> MockResponse: ...


class DirectoryFileRead:
    def __init__(self, root: Path, *, allowed_paths: set[str]) -> None: ...
    async def __call__(self, call: ToolCall) -> MockResponse: ...


@asynccontextmanager
async def serve_mock_tools(server: MockToolServer) -> AsyncIterator[str]: ...
```

### Tests First

- Test files:
  - `tests/unit/mock_tools/test_directory_file_read.py`
  - `tests/unit/mock_tools/test_server.py` (extend only for handler dispatch)
- Key cases:
  - `file_read` valid path returns file content
  - missing path returns error
  - traversal and absolute paths return error
  - directory reads return error
  - static mock tools still work
  - handler calls are recorded with tool name, input, output/error, and status

### Acceptance Criteria

- [ ] `uv run pytest tests/unit/mock_tools/test_directory_file_read.py tests/unit/mock_tools/test_server.py -q`
  passes.
- [ ] Existing mock tool YAML tests still pass.

## Phase 4: Request Preparation Hook And Method Corpus Preparer

**Goal:** add a generic runner seam and an `atp-method` implementation that
prepares corpus runs before adapter execution.

### Tasks

- [ ] Add `atp/runner/preparation.py` (NEW) with a small registry for named
  request preparers.
- [ ] Extend `TestOrchestrator` in `atp/runner/orchestrator.py` (MODIFIED) to
  apply a registered preparer after `_create_request` and before adapter
  execution, with cleanup in a `finally` block.
- [ ] Add `packages/atp-method/atp_method/runtime.py` (NEW) containing
  `CorpusRunPreparer`.
- [ ] Update `packages/atp-method/atp_method/plugin.py` to register the method
  corpus preparer.
- [ ] Update `packages/atp-method/atp_method/loader.py` to preserve method case
  metadata in `task.input_data`, including `case_path`, `run_mode`,
  `artifact_corpus`, and the requested preparer name for corpus cases.
- [ ] Ensure the prepared request has:
  - `Context.workspace_path`
  - `Context.tools_endpoint`
  - `constraints.allowed_tools == ["file_read"]`
  - no inline corpus file contents in `task.input_data["artifacts"]`

### Public Interfaces

```python
@dataclass(frozen=True)
class PreparedRequest:
    request: ATPRequest
    cleanup: Callable[[], Awaitable[None]] | None = None


class RequestPreparer(Protocol):
    async def prepare(
        self, test: TestDefinition, request: ATPRequest
    ) -> PreparedRequest: ...


class CorpusRunPreparer:
    async def prepare(
        self, test: TestDefinition, request: ATPRequest
    ) -> PreparedRequest: ...
```

### Tests First

- Test files:
  - `tests/unit/runner/test_request_preparation.py`
  - `packages/atp-method/tests/test_loader_artifact_corpus.py`
  - `packages/atp-method/tests/test_runtime_corpus_preparer.py`
- Key cases:
  - runner applies identity behavior when no preparer is requested
  - runner applies registered preparer before adapter execution
  - cleanup runs on success and adapter failure
  - loader preserves case path and corpus metadata without exposing contents
  - corpus preparer resolves, verifies, materializes, starts a tool endpoint, and
    updates request context
  - preparation failure becomes a failed run before the agent is invoked

### Acceptance Criteria

- [ ] Targeted runner and method runtime tests pass.
- [ ] Existing orchestrator tests pass.
- [ ] `read_only_corpus` is still not in `WIRED_RUN_MODES`.

## Phase 5: First Tool-Capable Method Spawner

**Goal:** make one method spawner actually call `file_read` through
`Context.tools_endpoint`.

### Tasks

- [ ] Add `method/spawners/_tool_client.py` (NEW) for HTTP calls to
  `POST /tools/call`.
- [ ] Update `method/spawners/anthropic_api_shim.py` (MODIFIED) to expose a
  provider-native `file_read` tool when the ATP request has:
  - `constraints.allowed_tools` containing `file_read`
  - `context.tools_endpoint`
- [ ] Run a bounded tool loop:
  - send initial prompt and tool definitions
  - when the model emits `tool_use` for `file_read`, call the ATP tools endpoint
  - send tool results back to the model
  - stop on final text or max tool iterations
- [ ] Emit `tool_call` events on stderr JSONL so traces show file access.
- [ ] Update `packages/atp-method/atp_method/envelopes.py` to add a
  corpus-aware instruction block that names the corpus id and relative paths but
  never inlines corpus content.
- [ ] Preserve current prompt-only behavior for `text_out` cases and all other
  shims.

### Tests First

- Test files:
  - `tests/unit/method_spawners/test_tool_client.py`
  - `tests/unit/method_spawners/test_anthropic_api_shim.py`
  - `packages/atp-method/tests/test_envelopes.py`
- Key cases:
  - shim does not expose tools when request lacks `context.tools_endpoint`
  - fake Anthropic SDK emits `tool_use`, shim calls `file_read`, and final
    artifact contains the model's JSON text
  - tool endpoint error is returned to the model, not raised as shim crash
  - max tool iterations fails with a contract-shaped ATPResponse
  - prompt includes corpus id and relative file paths, but not file contents

### Acceptance Criteria

- [ ] Targeted spawner tests pass offline with fake SDK and fake tool endpoint.
- [ ] Existing `anthropic_api` tests still pass.
- [ ] No product CLI shim behavior changes.

## Phase 6: `citation_grounding` Checker

**Goal:** add deterministic grading for citations against corpus files and
metadata.

### Tasks

- [ ] Add `atp/evaluators/citation_grounding/checker.py` (NEW).
- [ ] Register `citation_grounding` in `atp/evaluators/checkers/__init__.py`
  (MODIFIED).
- [ ] Reuse `atp.evaluators.json_path.resolver.resolve` for exact-one-node
  output paths.
- [ ] Parse the primary JSON text artifact and validate it against
  `output_contract.schema`.
- [ ] Select the primary text artifact by `grader.config.artifact_name` or
  `output_contract.artifact_name`; preserve first-content fallback for existing
  checkers.
- [ ] Validate each expected citation:
  - output path resolves to exactly one node
  - citation path is a selected corpus file
  - page is `null` for first slice
  - line range is one-based, inclusive, and within file bounds
  - citation line range matches configured expected source location
  - metadata fields such as `status` and `role` match when referenced
- [ ] Validate forbidden sources so obsolete or disallowed files fail the gate.
- [ ] Treat JSON parse failures and schema violations as
  `malformed=True, critical_pass=False`.

### Public Interfaces

```python
CITATION_GROUNDING_CHECKER_VERSION = "citation_grounding@1"


def citation_grounding_check(
    config: dict[str, Any], text: str | None
) -> CaseVerdict: ...
```

### Tests First

- Test files:
  - `tests/unit/evaluators/test_citation_grounding_checker.py`
  - `packages/atp-method/tests/test_evaluator.py` (artifact selector only)
- Key cases:
  - valid citation passes
  - missing file fails, not malformed
  - invalid line range fails, not malformed
  - obsolete-source citation fails when forbidden
  - metadata-sensitive expectation requires metadata entry
  - malformed citation object fails as malformed
  - bad JSON and schema violation are malformed
  - unsupported page value fails for first slice
  - missing, invalid, or multi-match `output_path` fails deterministically

### Acceptance Criteria

- [ ] `uv run pytest tests/unit/evaluators/test_citation_grounding_checker.py packages/atp-method/tests/test_evaluator.py -q`
  passes.
- [ ] `json_path` and `findings_match` tests still pass.

## Phase 7: Enable `read_only_corpus` And Add A Corpus Fixture

**Goal:** flip the mode on only after the end-to-end path is covered.

### Tasks

- [ ] Add `read_only_corpus` to `WIRED_RUN_MODES` in
  `packages/atp-method/atp_method/schema.py` (MODIFIED).
- [ ] Add corpus fixture under
  `method/cases/req-extraction/assets/fabricated-deadline-clean-corpus-001/`
  (NEW):
  - `policy-current.md`
  - `vendor-addendum.md`
  - `archive/policy-2023.md`
  - `manifest.sha256`
  - `corpus.meta.yaml`
- [ ] Add one corpus-backed req-extraction case YAML under
  `method/cases/req-extraction/` (NEW).
- [ ] Ensure the case uses:
  - `run_mode: read_only_corpus`
  - `environment.tools: [file_read]`
  - `grader.checker: citation_grounding`
  - `output_contract.schema`
- [ ] Update `method/CASE_GENERATOR.md` with manifest and corpus layout
  guidance.
- [ ] Update `packages/atp-method/README.md` status for `read_only_corpus`.
- [ ] Update `method/run_pipe_check.py` if needed so corpus-backed req-extraction
  sweeps can select `anthropic_api` as the first supported tool-capable harness.

### Tests First

- Test files:
  - `packages/atp-method/tests/test_cases_load.py`
  - `packages/atp-method/tests/test_req_extraction_corpus_grounding.py`
  - `tests/unit/method_spawners/test_run_pipe_check.py`
- Key cases:
  - the new corpus-backed case loads after `WIRED_RUN_MODES` is updated
  - the same case fails if manifest is corrupted
  - prompt body contains no corpus file contents
  - simulated agent output with valid citations passes `citation_grounding`
  - simulated obsolete citation fails
  - dry-run pipe-check recognizes the corpus-backed case and supported harness

### Acceptance Criteria

- [ ] `uv run pytest packages/atp-method/tests/test_cases_load.py packages/atp-method/tests/test_req_extraction_corpus_grounding.py tests/unit/method_spawners/test_run_pipe_check.py -q`
  passes.
- [ ] A local offline end-to-end test with fake `anthropic` and fake
  `file_read` proves the request contains a working tools endpoint.
- [ ] `read_only_corpus` is enabled only in this phase.

## Phase 8: Full Verification And Documentation

**Goal:** prove the slice is stable and leave clear operating docs.

### Tasks

- [ ] Run targeted suites:
  - `uv run pytest packages/atp-method/tests -q`
  - `uv run pytest tests/unit/mock_tools tests/unit/evaluators tests/unit/method_spawners tests/unit/runner -q`
- [ ] Run broader non-slow tests if the targeted suites are green:
  - `uv run pytest tests -m "not slow" -q`
- [ ] Run lint/type checks used by the repo:
  - `uv run ruff check .`
  - `uv run pyrefly check`
- [ ] Add or update docs:
  - `method/README.md`
  - `method/METHODOLOGY.md`
  - `packages/atp-method/README.md`
- [ ] Add a changelog note if this branch uses changelog entries for feature
  slices.

### Acceptance Criteria

- [ ] Existing `text_out` method cases still load and run.
- [ ] Corpus-backed method case loads, prepares, executes with tool access, and
  grades with `citation_grounding`.
- [ ] No inline corpus content reaches the prompt.
- [ ] Manifest mismatch fails before the agent is invoked.
- [ ] Citation failures are distinguishable from malformed JSON/schema failures.

## Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Request-preparer hook grows into a plugin framework | Medium | Keep the interface minimal: name lookup, prepare, cleanup. No lifecycle features beyond this slice. |
| Tool server lifecycle leaks background tasks | High | Use an async context manager, explicit cleanup tests, and per-run ephemeral port. |
| Prompt path list weakens "discover files" fidelity | Medium | Accept for first slice because only `file_read` is specified. Add `file_list` later if stricter discovery is required. |
| `citation_grounding` re-verifies corpus and duplicates pre-run work | Low | Reuse the same corpus helpers; correctness is preferable to sharing mutable run state through evaluator plumbing. |
| Product CLI shims cannot use `Context.tools_endpoint` yet | Medium | Start with `anthropic_api`; keep CLI shims explicitly unsupported for `read_only_corpus` until they have real tool wiring. |
| Enabling `read_only_corpus` too early creates false-fidelity runs | High | The plan keeps the wired-mode flip as the final implementation phase. |

## Definition Of Done

- [ ] `read_only_corpus` cases remain rejected until Phase 7.
- [ ] `artifact_corpus` is schema-validated in both JSON Schema and Pydantic.
- [ ] Corpus selection, manifest verification, LF normalization, and
  materialization are deterministic and unit-tested.
- [ ] `file_read` is served through `Context.tools_endpoint`.
- [ ] At least one method spawner actually calls `file_read`.
- [ ] `citation_grounding` validates JSON text against `output_contract.schema`
  and checks path/line/metadata grounding.
- [ ] One text/markdown req-extraction corpus fixture runs through the complete
  path without inline corpus contents.
- [ ] Existing `text_out` method behavior remains unchanged.

## Explicitly Deferred

- `ArtifactStructured.data` transport through adapters.
- PDF/DOCX normalization, page segmentation, and citation line mapping.
- Mounting large corpora instead of copying selected files per run.
- Product CLI tool integration for `claude_code`, `codex_cli`, `ollama`, and
  other shims.
- Heterogeneous `grader.checks[]`.
- Registry-declared checker config models for all checkers.
