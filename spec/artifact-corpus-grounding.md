# Spec: `read_only_corpus` grounding for method cases

**Status:** reconciled draft - implementation requires runtime wiring first
**Created:** 2026-06-14
**Updated:** 2026-06-20
**Scope:** `method/`, `packages/atp-method/`, `read_only_corpus` run mode,
mock-tools-backed file serving, corpus asset layout, and citation grounding.

## Overview

Inline method cases are useful for focused single-call checks, but they do not
test agentic behavior. A requirement-extraction agent should sometimes have to
discover files, read relevant documents, ignore distractors, compare sources,
and cite where each extracted field came from.

This feature defines the next run-mode tier: `read_only_corpus`. A case points
to a corpus folder, the corpus is verified by SHA-256 manifest, selected files
are copied into a per-run workspace, and the agent reads them through the
existing tools path. The natural serving seam is the existing
`Context.tools_endpoint` plus mock tool server, not a new file-delivery
mechanism.

This feature depends on structured output. The current method implementation
uses JSON text plus `output_contract.schema` validation inside named
programmatic checkers; it does not yet transport `ArtifactStructured.data`
through adapters. The first citation-grounding slice must follow that current
contract: parse the agent's primary JSON text artifact, validate it against the
case `output_contract.schema`, then run deterministic citation checks. Typed
structured-artifact transport remains out of scope.

The current codebase also names `read_only_corpus` in the method schema, but
rejects it at load time because only `text_out` is wired. That guard is correct
until corpus preparation, mock file serving, and spawner/tool wiring exist. Do
not flip `read_only_corpus` into the wired run-mode set until a corpus-backed
case can be loaded, materialized, served through `Context.tools_endpoint`, run
through at least one method spawner that can call `file_read`, and graded by
`citation_grounding`.

The first slice should support text and markdown only; PDF/DOCX page-line
normalization is a separate sub-project.

## Specification

| Component | Signature | Description |
|-----------|-----------|-------------|
| `run_mode` | `"read_only_corpus"` | Taxonomy coordinate for cases where the agent reads a materialized corpus through tools. |
| `AgentEvalCase.artifact_corpus` | `ArtifactCorpus | None` | Declares a source folder to expose as read-only corpus files. |
| `ArtifactCorpus` | `id: str`, `root: str`, `include: list[str]`, `exclude: list[str]`, `digest: CorpusDigest`, `metadata_path: str | None` | Describes corpus selection and integrity. |
| `CorpusDigest` | `algorithm: "sha256"`, `manifest_path: str`, `normalization: "lf"` | Points to the manifest used to verify normalized file content. |
| `CorpusMetadata` | `files: dict[str, CorpusFileMetadata]` | Optional semantic metadata for selected files; never used as content identity. |
| `CorpusResolver` | `def resolve(case_path: Path, corpus: ArtifactCorpus) -> ResolvedCorpus` | Expands include/exclude patterns under the corpus root with canonical sorted paths. |
| `CorpusIntegrityVerifier` | `def verify(resolved: ResolvedCorpus) -> CorpusVerificationResult` | Computes SHA-256 over the normalized content and compares it to the manifest. |
| `CorpusMaterializer` | `def materialize(resolved: ResolvedCorpus, workspace: Path) -> MaterializedCorpus` | Copies selected files into a per-run workspace. |
| `CorpusRunPreparer` | `def prepare(case, run_workspace) -> PreparedCorpusRun` | Resolves, verifies, materializes, and starts/registers the file tool endpoint before agent execution. |
| directory-backed mock file tool | `Context.tools_endpoint` | Serves deterministic `file_read` access to the materialized corpus, instead of static canned responses. |
| method spawner file tools | `file_read` via `Context.tools_endpoint` | Lets the target agent call the mock file tool. A prompt-only shim is not enough for `read_only_corpus`. |
| `citation_grounding` checker | `Callable[[dict, str | None], CaseVerdict]` | Parses primary JSON text output, validates schema, then verifies citation path, line range, file status, and expected source location. |

## Data Models

### Case YAML

```yaml
run_mode: read_only_corpus
artifact_corpus:
  id: fabricated-deadline-clean-corpus
  root: method/cases/req-extraction/assets/fabricated-deadline-clean-001
  include: ["**/*.md", "**/*.txt"]
  exclude: ["README.md"]
  digest:
    algorithm: sha256
    normalization: lf
    manifest_path: manifest.sha256
  metadata_path: corpus.meta.yaml
environment:
  tools: [file_read]
  side_effects: none
```

`root`, `manifest_path`, `metadata_path`, and all citation paths use the same
safe relative-path rules already enforced by ATP protocol path validators:
absolute paths, `..`, `~`, and null bytes are invalid.

### Hash manifest

```text
8b4f0e3b...  policy-current.md
53aac201...  vendor-addendum.md
f91c5a8d...  archive/policy-2023.md
```

The manifest contains one SHA-256 digest and one relative file path per line.
Paths are relative to `artifact_corpus.root`. The selected file set after
`include`/`exclude` expansion must match the manifest exactly.

Hashes must be computed over the same normalized representation used for line
indexing. For the first slice this means text/markdown content with LF newline
normalization. This avoids a split where raw bytes hash one way but citations
refer to a different normalized line view.

### Optional metadata

```yaml
files:
  policy-current.md:
    role: source
    status: current
    document_id: policy-current
  vendor-addendum.md:
    role: source
    status: current
    document_id: vendor-addendum
  archive/policy-2023.md:
    role: distractor
    status: obsolete
    document_id: policy-archive-2023
```

Metadata is semantic. If a deterministic check references metadata fields such
as `status`, the metadata file and the referenced entry are required. If no
check references metadata, missing metadata is a no-op.

### Structured output with citations

For the first implementation, the agent returns this object as JSON text in the
primary response artifact, matching the landed structured-output path. The
checker may later accept `ArtifactStructured.data`, but it must not require that
transport in the first slice.

```json
{
  "requirements": [
    {
      "obligation": "submit a security attestation",
      "actor": "vendor",
      "deadline": "within 30 days of onboarding",
      "citations": {
        "deadline": {
          "path": "policy-current.md",
          "page": null,
          "line_start": 14,
          "line_end": 14,
          "field": "deadline",
          "quote": "within 30 days of onboarding"
        }
      }
    }
  ]
}
```

Line ranges are one-based and inclusive. `quote` is optional diagnostic text and
is not the primary grounding key.

### Citation-grounding checker config

The citation-grounding check should remain a named programmatic checker, not an
entry in a generic `checks[]` array:

```yaml
grader:
  type: programmatic
  checker: citation_grounding
  config:
    artifact_name: answer
    corpus_id: fabricated-deadline-clean-corpus
    expected:
      - output_path: $.requirements[0].citations.deadline
        source_path: policy-current.md
        page: null
        line_start: 14
        line_end: 14
        status: current
    forbidden:
      - source_path: archive/policy-2023.md
        status: obsolete
  critical_check: >
    Each extracted deadline is grounded in the current policy corpus and does
    not cite obsolete distractor documents.
```

`output_path` follows the structured-output spec's deterministic JSON-path
semantics: the path must resolve to exactly one node.

The checker receives the same `schema` key that `json_path` and `findings_match`
already receive from `output_contract.schema`. Schema failure is `malformed=True`
and `critical_pass=False`, not a rubric failure. The checker then resolves each
configured `output_path` into the parsed JSON object and validates the cited
source file and line range against the prepared corpus manifest/metadata made
available in the assertion config.

### Runtime preparation

`read_only_corpus` needs an execution preparation step before the adapter/spawner
runs:

1. Load the case and reject `artifact_corpus` unless `run_mode` is
   `read_only_corpus`.
2. Resolve include/exclude paths relative to the case file and corpus root.
3. Verify the selected set against `manifest.sha256` using LF-normalized text.
4. Copy selected files into a fresh per-run workspace.
5. Build a directory-backed mock `file_read` tool rooted at the materialized
   corpus directory.
6. Expose that tool through an HTTP `Context.tools_endpoint`.
7. Pass `workspace_path`, `tools_endpoint`, and `allowed_tools=["file_read"]` to
   the agent request.
8. Ensure the prompt names the corpus and expected tool use, but does not inline
   the corpus file contents.

This is the runtime seam that makes the mode agentic. A case that merely inlines
the same files into the prompt is still `text_out`, not `read_only_corpus`.

## Business Rules

- `read_only_corpus` cases depend on structured-output support.
- The first implementation uses JSON text output plus `output_contract.schema`
  validation, matching the current method pipeline. It must not depend on
  `ArtifactStructured.data` transport.
- Corpus content reproducibility is based on SHA-256 hashes over normalized
  content, not on listing source content in case YAML.
- `include`/`exclude` expansion must be path-normalized and canonically sorted.
- The selected file set must match the manifest exactly. Missing files, extra
  selected files, duplicate manifest paths, and hash mismatches are pre-run
  failures.
- Manifest paths must be relative and must pass existing ATP-safe path rules.
- Corpus files are copied into a per-run workspace for isolation and
  determinism. Mounting is deferred.
- File access should be served through the existing mock-tools / tools endpoint
  seam, but the mock tool must be directory-backed for this mode rather than
  static response-rule based.
- `read_only_corpus` must remain rejected by the method schema until corpus
  preparation and at least one method spawner can execute a `file_read` call
  through `Context.tools_endpoint`.
- Prompt-only shims that ignore `Context.tools_endpoint` are insufficient for
  `read_only_corpus`. They may still run `text_out` cases.
- Metadata is optional unless a checker references metadata fields.
- A citation path must reference a selected corpus file.
- A citation must not cite an obsolete file unless the checker explicitly
  allows it.
- The first implementation supports text and markdown only.
- PDF/DOCX page-line citation support requires stable normalized text/page
  artifacts and is out of scope for the first slice.

## Functional Requirements

- [ ] Extend the method case schema with `run_mode: read_only_corpus` and
      `artifact_corpus`.
- [ ] Extend the Pydantic method schema with corpus models and path validators
      that reuse ATP protocol path-safety behavior.
- [ ] Keep `read_only_corpus` outside `WIRED_RUN_MODES` until runtime
      preparation, file serving, spawner tool use, and citation grading are all
      covered by tests.
- [ ] Add deterministic include/exclude expansion with canonical sorted paths.
- [ ] Add SHA-256 manifest parsing and normalized-content verification.
- [ ] Fail pre-run preparation when selected files and manifest entries differ.
- [ ] Copy selected corpus files into a per-run workspace.
- [ ] Add a directory-backed `file_read` implementation to the mock tool server
      and serve it through `Context.tools_endpoint`.
- [ ] Add a method run-preparation layer that resolves/verifies/materializes the
      corpus and attaches `workspace_path` plus `tools_endpoint` to the
      `ATPRequest`.
- [ ] Update at least one method spawner path to call `file_read` through
      `Context.tools_endpoint` when `allowed_tools` includes `file_read`.
- [ ] Ensure the task prompt references the corpus root and does not inline file
      content.
- [ ] Add a named `citation_grounding` programmatic checker with a
      load-validated config model.
- [ ] Make `citation_grounding` parse the primary JSON text artifact and apply
      `output_contract.schema`, following the landed `json_path` /
      `findings_match` pattern.
- [ ] Implement line-range validation for text and markdown files.
- [ ] Reuse existing malformed/critical-pass result fields when structured
      citations are missing or malformed.
- [ ] Update case-generation guidance to create corpus folders and
      `manifest.sha256` files for agentic document cases.

## Implementation Tasks

- [ ] Add schema contract tests for valid and invalid `artifact_corpus` blocks,
      including rejection when `artifact_corpus` appears without
      `run_mode: read_only_corpus`.
- [ ] Add a test proving `read_only_corpus` still fails while it is not in
      `WIRED_RUN_MODES`, then update that test only when the runtime path is
      complete.
- [ ] Add tests for absolute paths, traversal, duplicate manifest entries,
      missing files, extra files, and hash mismatches.
- [ ] Add tests proving hash input and citation line indexing use the same LF
      normalized content.
- [ ] Add tests for absent metadata when no metadata-sensitive check exists.
- [ ] Add tests requiring metadata when a checker references `status` or `role`.
- [ ] Add workspace materialization tests proving files are copied into an
      isolated run directory.
- [ ] Add directory-backed mock-tools `file_read` tests for the materialized
      corpus: valid path, missing path, traversal attempt, and directory read.
- [ ] Add method run-preparation tests proving the resulting `ATPRequest`
      contains `Context.tools_endpoint`, `Context.workspace_path`, and no inline
      corpus file contents.
- [ ] Add spawner/tool-use tests for the first supported method spawner, proving
      a `file_read` call is sent to `Context.tools_endpoint`.
- [ ] Add citation-grounding checker tests for valid citations, missing files,
      invalid line ranges, obsolete-source citations, malformed citations, and
      multi-match JSON paths.
- [ ] Add citation-grounding tests proving JSON text schema violations are
      reported as `malformed=True` and `critical_pass=False`.
- [ ] Add a documented helper or CLI command to generate `manifest.sha256`.
- [ ] Add one small text/markdown req-extraction corpus fixture with source,
      supporting, and obsolete distractor documents.

## Migration Strategy

Phase 1 uses the landed structured-output path as-is: JSON text output,
`output_contract.schema`, and named programmatic checkers. Do not wait for
`ArtifactStructured.data` transport.

Phase 2 adds corpus schema/models, resolver, verifier, materializer, and
directory-backed mock `file_read`, while keeping `read_only_corpus` rejected as
unwired.

Phase 3 adds method run preparation and one spawner path that can actually use
`Context.tools_endpoint`. Only after this path is covered by tests should
`read_only_corpus` be added to `WIRED_RUN_MODES`.

Phase 4 adds `citation_grounding` and one corpus-backed req-extraction fixture,
then runs the corpus-backed sweep while preserving the inline suite. The inline
suite remains the single-call signal; the corpus suite becomes the agentic
file-grounded signal.

Phase 5 evaluates whether PDF/DOCX normalized page-line artifacts are worth a
separate design.

## Deferred

- Mounting large corpora instead of copying per run.
- PDF/DOCX rendering, page segmentation, line indexing, and cache invalidation.
- Heterogeneous `grader.checks[]`; citation grounding remains a named checker
  until a separate ADR says otherwise.
