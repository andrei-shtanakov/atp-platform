# Spec: artifact corpus grounding for method cases

**Status:** draft
**Created:** 2026-06-14
**Scope:** `method/`, `packages/atp-method/`, corpus asset layout, loader
materialization, and deterministic citation grounding.

## Overview

Method cases can currently embed all relevant information directly in YAML
`artifacts[].content`. That is useful for focused single-call model checks, but
it does not exercise agentic behavior: the agent does not need to discover
documents, read files, compare sources, ignore distractors, or cite evidence.
For document-heavy cases such as requirement extraction, the methodology should
support a corpus-folder presentation mode.

In this mode, a case points to a folder of source assets instead of listing every
file inline. The folder is materialized into the agent workspace, the prompt
references the corpus root, and the agent must inspect files through allowed
tools such as `file_read`. Reproducibility is guaranteed by a SHA-256 manifest:
the case lists the corpus root and selection rules, while the manifest records
the exact bytes of the selected files.

Grounding is verified through structured citations. The agent output must cite
source `path`, `page`, and `line_start`/`line_end` for each extracted field. Quote
text may be included for debugging, but deterministic checks should rely on
stable file identity plus page/line ranges and expected source locations, not
free-form quote comparison.

This feature complements `spec/structured-method-output.md`: structured outputs
define the answer shape, while corpus grounding defines how source files are
provided and how citations are verified.

## Specification

| Component | Signature | Description |
|-----------|-----------|-------------|
| `AgentEvalCase.artifact_corpus` | `ArtifactCorpus | None` | Declares a source folder to expose as workspace files for the case. |
| `ArtifactCorpus` | `id: str`, `root: str`, `include: list[str]`, `exclude: list[str]`, `digest: CorpusDigest`, `metadata_path: str | None`, `presentation: "workspace_files"` | Describes corpus selection, integrity, and presentation mode. |
| `CorpusDigest` | `algorithm: "sha256"`, `manifest_path: str` | Points to the file containing path-to-hash entries for selected corpus files. |
| `CorpusMetadata` | `files: dict[str, CorpusFileMetadata]` | Optional semantic metadata for selected files; never used as content identity. |
| `CorpusFileMetadata` | `role: "source" | "distractor" | "supporting"`, `status: "current" | "obsolete" | "unknown"`, `document_id: str | None` | Labels file semantics for deterministic checks and analysis. |
| `Citation` | `path: str`, `page: int | None`, `line_start: int`, `line_end: int`, `field: str | None`, `quote: str | None` | Structured reference emitted by the agent for an extracted field. |
| `CorpusResolver` | `def resolve(case_path: Path, corpus: ArtifactCorpus) -> ResolvedCorpus` | Expands include/exclude patterns under the corpus root using repo-relative, safe paths. |
| `CorpusIntegrityVerifier` | `def verify(resolved: ResolvedCorpus) -> CorpusVerificationResult` | Computes SHA-256 for selected files and compares them to the manifest. |
| `CorpusMaterializer` | `def materialize(resolved: ResolvedCorpus, workspace: Path) -> MaterializedCorpus` | Copies or mounts selected files into the per-run workspace and returns the exposed root path. |
| `CitationGroundingEvaluator` | `async def evaluate(task, response, trace, assertion) -> EvalResult` | Verifies citations against corpus path, status, page, and line constraints. |

## Data Models

### Case YAML

```yaml
artifact_corpus:
  id: fabricated-deadline-clean-corpus
  root: method/cases/req-extraction/assets/fabricated-deadline-clean-001
  include: ["**/*.md", "**/*.txt", "**/*.pdf"]
  exclude: ["README.md"]
  presentation: workspace_files
  digest:
    algorithm: sha256
    manifest_path: manifest.sha256
  metadata_path: corpus.meta.yaml
```

`root`, `manifest_path`, and `metadata_path` are repository-relative or
case-file-relative paths. They must not be absolute and must not escape the
repository or case asset root.

### Hash manifest

```text
8b4f0e3b...  policy-current.md
53aac201...  vendor-addendum.md
f91c5a8d...  archive/policy-2023.md
```

The manifest contains one SHA-256 digest and one relative file path per line.
Paths are relative to `artifact_corpus.root`. The selected file set after
`include`/`exclude` expansion must match the manifest exactly.

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

Metadata is semantic. It can identify distractors, obsolete documents, and
current source documents, but it does not prove content identity. Content
identity comes only from the digest manifest.

### Structured output with citations

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

`quote` is optional and diagnostic. The deterministic evaluator should not rely
on it as the primary grounding key.

### Expected source locations

Cases that need strict grounding should define expected source locations in
their deterministic checks:

```yaml
grader:
  checks:
    - type: citation_grounding
      critical: true
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
```

## Business Rules

- Corpus content reproducibility is based on SHA-256 hashes, not on listing
  source content in case YAML.
- `artifact_corpus.root` must resolve to a directory inside the repository or an
  approved case asset root.
- `include`/`exclude` expansion must be deterministic and path-normalized.
- The selected file set must match `manifest.sha256` exactly. Missing files,
  extra selected files, duplicate manifest paths, and hash mismatches are
  pre-run failures.
- Manifest paths must be relative paths. Absolute paths and `..` traversal are
  invalid.
- Corpus metadata is optional and cannot replace hash verification.
- A citation path must reference a selected corpus file.
- A citation must not cite a file whose metadata marks it `status: obsolete`
  unless the check explicitly allows obsolete sources.
- `line_start` and `line_end` are required for citations. `page` is required for
  paginated normalized artifacts and may be `null` for plain text files.
- For plain text and markdown, line numbers refer to the checked-in file bytes
  after normal newline normalization.
- For PDF, DOCX, and other rendered formats, the corpus must include or generate
  a stable normalized text view with page and line indexes before citations can
  be verified.
- Citation checks verify location and allowed source status. Field-value support
  should be checked by deterministic expected-output checks where feasible.

## Functional Requirements

- [ ] Extend `method/agent-eval-case.schema.json` with `artifact_corpus`.
- [ ] Extend `packages/atp-method/atp_method/schema.py` with `ArtifactCorpus`,
      `CorpusDigest`, and validators for safe relative paths.
- [ ] Add corpus resolution that expands `include` and `exclude` patterns
      deterministically under `artifact_corpus.root`.
- [ ] Add SHA-256 manifest parsing and integrity verification.
- [ ] Fail case loading or pre-run preparation when corpus integrity does not
      match the manifest.
- [ ] Add optional corpus metadata parsing for file role/status/document ID.
- [ ] Materialize selected corpus files into the agent workspace for
      `presentation: workspace_files`.
- [ ] Ensure the task prompt references the exposed corpus root, not the full
      file contents.
- [ ] Add a citation schema fragment reusable by structured output contracts.
- [ ] Add deterministic citation-grounding checks for path, page, line range,
      file existence, and allowed metadata status.
- [ ] Support text/markdown line checks first.
- [ ] Define normalized text-page artifacts before enabling strict PDF/DOCX page
      and line grounding.
- [ ] Update `CASE_GENERATOR.md` so generated agentic cases create corpus
      folders, hash manifests, and citation requirements.
- [ ] Update at least one `req-extraction` case family to use corpus-folder
      presentation as a reference example.

## Implementation Tasks

- [ ] Add schema contract tests for valid and invalid `artifact_corpus` blocks.
- [ ] Add tests for unsafe corpus paths, absolute paths, traversal, duplicate
      manifest entries, missing files, extra files, and hash mismatches.
- [ ] Add Pydantic tests for corpus model validation in `packages/atp-method`.
- [ ] Add loader tests showing corpus metadata appears in `TestDefinition`
      input data without inlining file contents.
- [ ] Add workspace materialization tests for selected files.
- [ ] Add citation-grounding evaluator tests for valid citations, missing files,
      invalid line ranges, obsolete-source citations, and malformed citations.
- [ ] Add a helper command or documented script for generating `manifest.sha256`
      from a corpus folder.
- [ ] Migrate a small `req-extraction` fixture corpus with current, supporting,
      and obsolete/distractor documents.
- [ ] Run targeted tests for `packages/atp-method` and citation evaluator
      behavior.

## Migration Strategy

Phase 1 adds `artifact_corpus` as optional. Existing inline-artifact cases keep
working unchanged.

Phase 2 adds one corpus-backed `req-extraction` sweep while preserving the
existing inline sweep. This gives separate signal for single-call extraction and
agentic file-grounded extraction.

Phase 3 updates generated method-case guidance. New document-heavy cases should
prefer corpus-folder presentation and structured citations. Inline artifacts
remain valid for small, focused, non-agentic checks.

Phase 4 makes corpus integrity verification part of normal method case loading
for any case that declares `artifact_corpus`.

## Open Decisions

- Decide whether hash manifests are generated manually by a helper script or by
  a formal `atp method corpus hash` CLI command.
- Decide whether corpus paths are repo-relative only, case-file-relative only,
  or support both with explicit `root_base`.
- Decide where normalized text/page views for PDFs and DOCX should live:
  alongside source assets, in a generated cache, or as checked-in derived
  artifacts.
- Decide whether citation line ranges should be one-based and inclusive. The
  recommendation is one-based inclusive ranges for human readability.
- Decide whether the runner should copy corpus files into the workspace or
  mount/read them in place. Copying is simpler and isolates the run; mounting
  avoids duplication for large corpora.
