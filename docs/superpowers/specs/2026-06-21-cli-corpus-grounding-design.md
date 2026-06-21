# CLI corpus grounding — Path A (materialized directory + native tools)

**Date:** 2026-06-21
**Status:** proposal (forward design — not yet planned/executed)
**Owner:** ATP / R-07
**Builds on:** `spec/artifact-corpus-grounding.md`, `docs/plans/artifact-corpus-grounding.md` (the `read_only_corpus` slice, PR #203), `docs/superpowers/specs/2026-06-20-agent-roster-tier2-design.md` (CLI shims).
**Ecosystem refs:** `../_cowork_output/contracts/add-new-agent-runbook.md` (this stays ATP-side; the affected agents are non-routable benchmark rows).

---

## 1. Why this matters (the development goal)

R-07 evaluates agents on capability axes that feed arbiter routing. PR #203
added a genuinely *agentic* axis — **grounding / source recency** — via the new
`read_only_corpus` run mode: an agent must discover files, read them, ignore
distractors, and cite where each extracted field came from. This is a high-value
signal (it separates agents that verify sources from agents that hallucinate
plausible answers), and it maps to a real routing dimension.

But the first slice is wired for exactly **one** spawner — `anthropic_api`. Every
agent we actually benchmark and that arbiter cares about runs through a **CLI
shim** (`claude_code`, `codex_cli`, `pi`, `opencode`, …) — and none of them can
run a corpus case today. So our most agentic signal currently covers none of the
roster. **Closing that gap is the development task this document specifies.**

## 2. Background — the `read_only_corpus` case (self-contained recap)

A corpus case (`run_mode: read_only_corpus`) declares an `artifact_corpus`
pointing at a folder of text/markdown source documents:

```yaml
run_mode: read_only_corpus
instruction: >
  Use file_read to inspect the available corpus files. Extract atomic
  current-policy requirements into JSON. ... Use only current source documents;
  do not cite obsolete archive documents.
artifact_corpus:
  id: fabricated-deadline-clean-corpus
  root: assets/fabricated-deadline-clean-corpus-001
  include: ["**/*.md"]
  digest: { algorithm: sha256, normalization: lf, manifest_path: manifest.sha256 }
  metadata_path: corpus.meta.yaml
environment: { tools: [file_read], side_effects: none }
output_contract:
  artifact_name: answer
  schema: { ... requirements[].citations.deadline{ path, page, line_start, line_end, field } ... }
grader:
  type: programmatic
  checker: citation_grounding
  config:
    expected:
      - output_path: "$.requirements[0].citations.deadline"
        source_path: policy-current.md
        line_start: 3
        line_end: 3
        status: current
    forbidden:
      - source_path: archive/policy-2023.md   # the obsolete distractor
        status: obsolete
  critical_check: cite the current policy, not the obsolete archive
```

The reference fixture (`fabricated-deadline-*`) is a **recency trap**: a
`policy-current.md` with the real deadline plus an `archive/policy-2023.md`
distractor carrying a plausible-but-obsolete deadline. The critical check fails
if the agent cites the archive.

**Runtime pipeline today (PR #203):**

1. **Resolve** (`atp_method/corpus.py::CorpusResolver`) — expand include/exclude
   under the corpus root, canonical-sort, reject symlinks / escapes / non-text.
2. **Verify** (`CorpusIntegrityVerifier`) — SHA-256 over **LF-normalized**
   content, exact match against `manifest.sha256`; selected set must equal the
   manifest set.
3. **Materialize** (`CorpusMaterializer`) — copy verified files into a per-run
   workspace `.atp-runs/<task_id>/<corpus_id>/`.
4. **Serve** (`atp_method/runtime.py` + `atp/mock_tools/file_tools.py`) — start a
   mock tool server whose `file_read` handler (`DirectoryFileRead`) serves only
   the allowed relative paths; its URL goes into `Context.tools_endpoint`.
5. **Prepare** (`atp/runner/preparation.py::RequestPreparer`) — a pluggable
   pre-adapter hook; the corpus preparer sets `workspace_path`, `tools_endpoint`,
   `allowed_tools=["file_read"]`, injects per-file `line_count`/metadata into the
   `citation_grounding` assertion config, and returns an async `cleanup`.
6. **Execute** (`anthropic_api_shim.py`) — a **bounded tool loop**
   (`MAX_TOOL_ITERATIONS=8`): the model emits `file_read` `tool_use`, the shim
   POSTs to `<tools_endpoint>/tools/call` (`_tool_client.call_tool`), feeds back
   `tool_result`, until the model returns the final JSON.
7. **Grade** (`atp/evaluators/citation_grounding/checker.py`) — parse the final
   JSON, validate against `output_contract.schema` (→ `malformed` on violation),
   then check each citation's `source_path` / line range / page-null / file
   length / metadata, and verify no `forbidden` source was cited.

## 3. The architectural asymmetry (root of the gap)

`anthropic_api_shim` **is its own harness**: it drives the raw Messages API and
runs the tool loop itself, so it can route every file access through ATP's
`tools_endpoint`. ATP therefore controls exactly which files exist and records
every read.

CLI shims are the opposite: they launch a **product CLI as a subprocess** that
has **its own closed tool system** (its own file-read/grep/bash over the real
filesystem). The shim only passes a prompt on argv and reads stdout; it does
**not** mediate the agent's tool calls. The CLI knows nothing about ATP's
`tools_endpoint`, so the corpus — which is only reachable through that endpoint —
is invisible to it. Hence "prompt-only": a CLI corpus case has no file source
wired in.

"tools-endpoint support" means closing this gap: making the agent inside the CLI
read the **verified corpus** (and nothing else), and making its citation paths
line up with what the grader expects.

## 4. Path A — native CLI tools over the materialized directory

**Key leverage:** the corpus is **already materialized to a real, verified
directory** (`workspace_path = materialized.root`). Product coding CLIs are built
to operate on a working directory with their own tools. So instead of forcing the
CLI through ATP's HTTP tool, we point the CLI's native tools at the corpus
directory and confine it there.

- Run the CLI with **`cwd = workspace_path`** and **confine it to that directory**
  (sandbox / allowed-dir / read-only).
- The HTTP `tools_endpoint` is **not used** for CLI agents — the corpus
  *directory itself* is the surface.
- The shim needs **no tool loop** — the CLI's own agent loop discovers and reads
  the files.

This mirrors why the HTTP-tool model exists in `anthropic_api` at all: that shim
had **no** filesystem harness and had to synthesize one. CLI products already
have one — we just aim it at the corpus and fence it in.

### 4.1 Trade-off vs. the HTTP/MCP-mediated model (Path B)

Path A trades ATP's *hard* guarantee ("the agent could physically only see corpus
files, via a mediator that logged every read") for **directory confinement**
(cwd + sandbox + read-only mount). That is a weaker but usually sufficient
guarantee, and it depends on each CLI's sandbox actually confining reads. A CLI
without real sandboxing either falls back to Path B (wrap ATP's `file_read` as an
MCP server, register it as the *only* tool, disable native fs tools) or is
excluded from corpus runs. Path B is deferred — it is more CLI-specific and
brittle (not every CLI lets you disable built-in tools or make yours exclusive).
We capture it here only as the fallback for un-sandboxable CLIs.

## 5. Components & changes

### 5.1 Split `CorpusRunPreparer` (shared materialize, optional serve)

Today `CorpusRunPreparer.prepare()` **always** starts the HTTP mock server and
sets `tools_endpoint`. Factor it so the **materialize+verify** core is shared and
the **serve-over-HTTP** step is optional:

- `anthropic_api` (and future raw-API shims): materialize **+** serve HTTP (as
  today).
- CLI family: materialize **only** — set `Context.workspace_path` and
  `constraints.allowed_tools`, **skip** the HTTP server, return the same
  `citation_grounding` `files` config (per-file `line_count` + metadata) so the
  grader is unchanged. Cleanup just removes the run workspace.

Selection is by spawner family. Either branch inside the preparer on a request
flag, or register two named preparers (`corpus_http`, `corpus_dir`) via the
existing `register_request_preparer` seam and pick per adapter.

### 5.2 Per-shim corpus mode

Each CLI shim must, when `context.workspace_path` is present and
`allowed_tools` contains a read tool:

1. set the subprocess **cwd** to `workspace_path`;
2. add the CLI's **confinement flags** (work only here, read-only — §5.4);
3. keep the existing stdout parsing (text + tokens) unchanged;
4. (optional) parse the CLI's JSON event stream for read/tool events (§5.5).

The shims currently ignore `context` entirely; this is the new branch. Put the
shared "is this a corpus run? compute cwd + confine args" logic in
`method/spawners/_cli_common.py` (the Tier-2 shared runner) so each thin shim only
supplies its own confinement flags.

### 5.3 Citation paths must stay corpus-relative

`citation_grounding` keys on **corpus-relative** paths (`policy-current.md`,
`archive/policy-2023.md`) — see `_files(config)` and `_check_expected`. When the
CLI reads from `cwd = corpus root`, the agent naturally sees these relative
paths, which matches. Guard the failure mode where a CLI emits **absolute** paths
in citations:

- instruct relative paths in the (tool-agnostic) instruction, and/or
- normalize absolute paths back to corpus-relative in the shim before emitting
  the artifact (strip the `workspace_path` prefix). Reject paths that escape the
  corpus root.

### 5.4 Per-CLI confinement (the non-uniform part)

Each CLI has a different "work here / read-only / limited tools" knob. Exact
flags must be confirmed per tool at implementation time (CLI docs drift); the
shape:

| CLI | Working dir | Confine / read-only | Tool limiting |
|-----|-------------|---------------------|---------------|
| `claude_code` | run in cwd | `--add-dir` to corpus, restrictive permission-mode, no network | limit to read tools |
| `codex_cli` | `-C/--cd <dir>` | `--sandbox read-only` | profile/config |
| `pi` | cwd | own permission/agent config (already `--no-prompt-templates`) | answer-only / read-only mode |
| `opencode` | cwd | own permission config | restrict tools |

This is **per-CLI work, not a shared refactor** — exactly why PR #203 wired only
`anthropic_api` first. Sequence one CLI at a time.

### 5.5 (Optional) read recording

The HTTP endpoint gave a free log of which files were read, in what order. Under
Path A that is lost. It is **not** required for grading (the grader only reads the
final JSON citations), but it is valuable for analysis ("did it actually read the
current doc, or guess?"). `opencode`/`pi` already emit JSONL we parse for
text/tokens; extend the parser to extract tool/read events where available. Mark
this optional and CLI-dependent.

### 5.6 Tool-agnostic instruction

The case instruction says "Use file_read…". For native-fs CLIs the tool is named
differently (Read/cat/grep). Make the instruction neutral ("read the files in
your working directory"); the `output_contract.format_instruction` (the JSON
shape) is unchanged. Consider a per-run-mode instruction variant so the inline
and corpus phrasings stay separate.

## 6. Security / confinement model

The corpus is verified + materialized into an isolated workspace, so integrity is
already guaranteed before the CLI starts. The CLI must not be able to:

- read outside the corpus root (cwd confinement + sandbox/read-only; reject
  absolute/`..`/symlink-escaping citation paths in the shim — mirror the existing
  guards in `corpus.py` / `file_tools.py`);
- mutate the corpus (read-only mount / don't grant write);
- reach the network or run arbitrary commands (disable where the CLI allows it).

A CLI that cannot enforce read-only confinement is **not eligible** for Path A —
route it to Path B or exclude it. State that exclusion explicitly in the roster
notes rather than silently shipping a weaker guarantee.

## 7. Testing

- **Offline unit (per shim):** a fake binary that "reads" a file under cwd and
  emits the CLI's JSON; assert the shim sets cwd + confine flags, parses output,
  and (if implemented) normalizes absolute → relative citation paths.
- **Confinement test:** a corpus run where the agent attempts to read outside the
  corpus root must fail closed (no out-of-corpus content reaches the artifact).
- **Grader compatibility:** an end-to-end-ish test that the `citation_grounding`
  `files` config is populated for CLI runs exactly as for `anthropic_api`.
- **Live smoke (the gate, per Tier-2 precedent):** one real corpus case per CLI →
  `status:"completed"`, a parsed citation, and the recency trap actually
  exercised (a strong agent cites `policy-current.md`, not the archive). Only
  CLIs that smoke green enter a paid sweep. ruff + pyrefly clean; suites green.

## 8. Sequencing

1. Split `CorpusRunPreparer` (materialize-only variant) + shared corpus-mode
   helper in `_cli_common.py`. Tests.
2. Wire **`claude_code` first** — it has the clearest `--add-dir` / permission
   confinement. Offline tests + confinement test + live smoke.
3. Then `codex_cli` (`--sandbox read-only` + `-C`).
4. Then `pi` / `opencode` as their confinement configs are confirmed; any CLI
   that can't confine reads → defer to Path B or exclude (note it).
5. Add corpus cases to a paid sweep (fresh `--out-dir` + `--dashboard-replace`)
   alongside the rest of the roster.

## 9. Non-goals

- No Path B (MCP-mediated tool) build now — only documented as the fallback for
  un-sandboxable CLIs.
- No PDF/DOCX page-line normalization (separate sub-project, already out of scope
  in the corpus spec).
- No typed `ArtifactStructured.data` transport — keep the JSON-text + schema
  contract.
- No Maestro/arbiter changes: the affected agents are non-routable benchmark
  rows; this is ATP-side only (runbook B2-style, no `agents.toml` / spawner).
- No change to `citation_grounding`, the manifest/verify pipeline, or the
  `<harness>@<model>` convention.

## 10. Open questions

- Which roster CLIs can actually enforce read-only directory confinement
  end-to-end? (Probe each before committing it to Path A.)
- Do we need read-recording in v1, or is final-JSON grading enough for the first
  signal? (Lean: optional, add where the JSON stream makes it cheap.)
- Should the corpus instruction variant live in the case YAML (per run_mode) or
  be derived by the shim? (Lean: a run_mode-keyed instruction so authoring stays
  declarative.)
