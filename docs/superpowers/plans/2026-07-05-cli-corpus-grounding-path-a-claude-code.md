# CLI Corpus Grounding Path A — claude_code slice — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `claude_code` run `read_only_corpus` cases by pointing the CLI's
native tools at the already-materialized corpus directory, confined read-only.

**Architecture:** The corpus preparer already materializes the verified corpus
and sets `Context.workspace_path` on every corpus run — we do NOT split it in
this slice (the extra local HTTP server is harmless; split deferred). The work
is: (1) shared corpus-mode helpers in `_cli_common.py`, (2) a corpus branch in
`claude_code_shim.py` (cwd + `--allowed-tools Read,Glob,Grep` + citation-path
normalization), (3) tool-agnostic corpus phrasing in the shared envelope,
(4) harness unskip for corpus-capable harnesses + preparer registration,
(5) a paid live smoke as the merge gate.

**Tech Stack:** Python 3.12, stdlib subprocess shims, pytest (+anyio), uv.

**Spec:** `docs/superpowers/specs/2026-06-21-cli-corpus-grounding-design.md`

## Global Constraints

- Prompt envelope is a single shared source (`atp_method/envelopes.py`) — all
  shims must keep drawing from it; per-shim phrasing forks are forbidden.
- Grader (`citation_grounding`), corpus verify/materialize pipeline, and the
  `<harness>@<model>` convention are unchanged (spec §9 non-goals).
- Citation paths are corpus-relative (`policy-current.md`,
  `archive/policy-2023.md`) — the grader keys on them (spec §5.3).
- Shims never crash: any error → `status:"failed"` ATPResponse on stdout.
- `uv run ruff format/check` + `uv run pyrefly check` clean after every task.
- Line length 88; type hints required.

---

### Task 1: Corpus-mode helpers in `_cli_common.py`

**Files:**
- Modify: `method/spawners/_cli_common.py` (append two functions)
- Test: `tests/unit/method_spawners/test_cli_common_corpus.py` (new)

**Interfaces:**
- Produces: `corpus_workspace(request: dict) -> str | None` — the materialized
  corpus root when this is a corpus run, else None.
- Produces: `normalize_citation_paths(text: str, workspace: str) -> str` —
  strip the absolute workspace prefix from citation paths in the model's
  output text (string-level, format-agnostic).

- [ ] **Step 1: Write the failing tests**

```python
"""Corpus-mode helpers shared by CLI spawner shims (Path A)."""

from method.spawners._cli_common import corpus_workspace, normalize_citation_paths


def test_corpus_workspace_returns_root_for_corpus_run() -> None:
    request = {
        "task": {"input_data": {"run_mode": "read_only_corpus"}},
        "context": {"workspace_path": "/tmp/x/.atp-runs/t1/corpus-1"},
    }
    assert corpus_workspace(request) == "/tmp/x/.atp-runs/t1/corpus-1"


def test_corpus_workspace_none_without_run_mode() -> None:
    # workspace_path alone is not enough — only read_only_corpus runs
    # switch the shim into corpus mode.
    request = {
        "task": {"input_data": {}},
        "context": {"workspace_path": "/tmp/x"},
    }
    assert corpus_workspace(request) is None


def test_corpus_workspace_none_without_workspace() -> None:
    request = {
        "task": {"input_data": {"run_mode": "read_only_corpus"}},
        "context": {},
    }
    assert corpus_workspace(request) is None


def test_corpus_workspace_none_on_missing_keys() -> None:
    assert corpus_workspace({}) is None


def test_normalize_strips_workspace_prefix() -> None:
    text = '{"path": "/ws/root/policy-current.md", "n": 1}'
    assert normalize_citation_paths(text, "/ws/root") == (
        '{"path": "policy-current.md", "n": 1}'
    )


def test_normalize_strips_nested_and_multiple() -> None:
    text = '"/ws/root/archive/policy-2023.md" and "/ws/root/policy-current.md"'
    out = normalize_citation_paths(text, "/ws/root")
    assert out == '"archive/policy-2023.md" and "policy-current.md"'


def test_normalize_handles_trailing_slash_workspace() -> None:
    assert (
        normalize_citation_paths('"/ws/root/a.md"', "/ws/root/")
        == '"a.md"'
    )


def test_normalize_leaves_relative_paths_alone() -> None:
    text = '{"path": "policy-current.md"}'
    assert normalize_citation_paths(text, "/ws/root") == text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/method_spawners/test_cli_common_corpus.py -q`
Expected: FAIL — `ImportError: cannot import name 'corpus_workspace'`

- [ ] **Step 3: Implement the helpers**

Append to `method/spawners/_cli_common.py`:

```python
def corpus_workspace(request: dict) -> str | None:
    """Materialized corpus root for a read_only_corpus run, else None.

    Path A (CLI corpus grounding): the corpus preparer materializes the
    verified corpus and sets ``context.workspace_path``. A native-tools CLI
    runs with cwd at that root instead of ATP's HTTP file_read endpoint.
    Both markers must be present — workspace_path alone may mean any
    future workspace-carrying run mode.
    """
    task = request.get("task") or {}
    input_data = task.get("input_data") or {}
    if input_data.get("run_mode") != "read_only_corpus":
        return None
    context = request.get("context") or {}
    workspace = context.get("workspace_path")
    return str(workspace) if workspace else None


def normalize_citation_paths(text: str, workspace: str) -> str:
    """Rewrite absolute corpus paths in model output to corpus-relative.

    The citation_grounding grader keys on corpus-relative paths
    (``policy-current.md``); a CLI reading files under cwd may cite the
    absolute path instead. String-level replace keeps this format-agnostic
    (works inside JSON strings without parsing the whole artifact).
    """
    prefix = workspace.rstrip("/") + "/"
    return text.replace(prefix, "")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/method_spawners/test_cli_common_corpus.py -q`
Expected: 8 passed

- [ ] **Step 5: Lint + type-check + commit**

```bash
uv run ruff format method/spawners/_cli_common.py tests/unit/method_spawners/test_cli_common_corpus.py
uv run ruff check method/spawners/ tests/unit/method_spawners/ && uv run pyrefly check
git add method/spawners/_cli_common.py tests/unit/method_spawners/test_cli_common_corpus.py
git commit -m "feat(method): corpus-mode helpers for CLI shims (Path A)"
```

---

### Task 2: claude_code_shim corpus branch

**Files:**
- Modify: `method/spawners/claude_code_shim.py`
- Test: `tests/unit/method_spawners/test_claude_code_shim.py` (extend)
- Test fixture: `tests/unit/method_spawners/fixtures/fake_claude.py` (extend)

**Interfaces:**
- Consumes: `corpus_workspace`, `normalize_citation_paths` from Task 1.
- Produces: corpus runs invoke `claude` with `cwd=<workspace>` and
  `--allowed-tools Read,Glob,Grep`; non-corpus invocations are byte-identical
  to today.

Notes for the implementer:
- The shim imports from `_cli_common` via a sibling import. The existing shim
  does `from atp_method.envelopes import ...` (workspace package) but has no
  sibling import yet; `method/spawners/` has no `__init__.py` — import via
  `sys.path` bootstrap exactly as `pi_shim.py`/`opencode_shim.py` do (check
  their header: they append `Path(__file__).parent` and
  `import _cli_common`). Mirror that pattern.
- Confinement rationale (spec §5.4/§6): whitelist `Read,Glob,Grep` — in
  non-interactive `-p` mode every other tool (Write/Edit/Bash/WebSearch…)
  is denied without a prompt. cwd = corpus root gives natural relative paths.

- [ ] **Step 1: Write the failing tests** (append to
  `tests/unit/method_spawners/test_claude_code_shim.py`; follow the file's
  existing fake-binary invocation pattern — it runs the shim as a subprocess
  with `CLAUDE_BIN="python .../fake_claude.py"`)

```python
def _corpus_request(workspace: str) -> dict:
    return {
        "version": "1.0",
        "task_id": "corpus-1",
        "task": {
            "description": "Extract requirements with citations.",
            "input_data": {
                "run_mode": "read_only_corpus",
                "artifact_corpus": {
                    "id": "c1",
                    "files": ["policy-current.md", "archive/policy-2023.md"],
                },
            },
        },
        "context": {"workspace_path": workspace},
    }


def test_corpus_run_sets_cwd_and_confinement_flags(tmp_path) -> None:
    # fake_claude (in corpus mode, see fixture) records its argv and cwd to
    # ATP_FAKE_CLAUDE_LOG and emits a citation with an ABSOLUTE path.
    workspace = tmp_path / "corpus"
    workspace.mkdir()
    (workspace / "policy-current.md").write_text("deadline: 2026-08-01\n")
    log_path = tmp_path / "invocation.json"

    response = _run_shim(
        _corpus_request(str(workspace)),
        extra_env={"ATP_FAKE_CLAUDE_LOG": str(log_path)},
    )

    invocation = json.loads(log_path.read_text())
    assert invocation["cwd"] == str(workspace)
    argv = invocation["argv"]
    assert "--allowed-tools" in argv
    assert argv[argv.index("--allowed-tools") + 1] == "Read,Glob,Grep"
    assert response["status"] == "completed"


def test_corpus_run_normalizes_absolute_citation_paths(tmp_path) -> None:
    workspace = tmp_path / "corpus"
    workspace.mkdir()
    log_path = tmp_path / "invocation.json"

    response = _run_shim(
        _corpus_request(str(workspace)),
        extra_env={"ATP_FAKE_CLAUDE_LOG": str(log_path)},
    )

    content = response["artifacts"][0]["content"]
    # fake_claude cites <workspace>/policy-current.md absolutely; the shim
    # must strip the prefix so the grader sees a corpus-relative path.
    assert str(workspace) not in content
    assert "policy-current.md" in content


def test_non_corpus_run_has_no_confinement_flags(tmp_path) -> None:
    log_path = tmp_path / "invocation.json"
    response = _run_shim(
        _plain_request(),  # the file's existing minimal request helper
        extra_env={"ATP_FAKE_CLAUDE_LOG": str(log_path)},
    )
    invocation = json.loads(log_path.read_text())
    assert "--allowed-tools" not in invocation["argv"]
    assert invocation["cwd"] != ""  # recorded, but just the test cwd
    assert response["status"] == "completed"
```

Extend `fixtures/fake_claude.py`: when `ATP_FAKE_CLAUDE_LOG` is set, write
`{"argv": sys.argv[1:], "cwd": os.getcwd()}` to that file; when the prompt
mentions `read_only_corpus`/corpus paths, emit
`"result": "{\"citations\": [{\"path\": \"" + os.getcwd() + "/policy-current.md\"}]}"`
(absolute path) so normalization is observable. Keep the existing usage block
so token tests stay green.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/method_spawners/test_claude_code_shim.py -q`
Expected: new tests FAIL (no `--allowed-tools`, absolute path leaks through)

- [ ] **Step 3: Implement the corpus branch in `claude_code_shim.py`**

```python
# header (after existing imports):
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli_common import corpus_workspace, normalize_citation_paths  # noqa: E402

# inside main(), replacing the fixed cmd/run/result flow:
    workspace = corpus_workspace(request)
    cmd = shlex.split(CLAUDE_BIN) + [
        "-p",
        prompt,
        "--model",
        MODEL,
        "--output-format",
        "json",
    ]
    if workspace:
        # Path A corpus confinement: run inside the materialized corpus with
        # read-only native tools. Whitelisting Read/Glob/Grep means every
        # other tool (Write/Edit/Bash/Web*) is denied in -p mode.
        cmd += ["--allowed-tools", "Read,Glob,Grep"]
    proc = subprocess.run(
        cmd, capture_output=True, timeout=600, cwd=workspace or None
    )
    ...
    result_text = out.get("result", "")
    if workspace:
        result_text = normalize_citation_paths(result_text, workspace)
    # use result_text in the artifact content
```

- [ ] **Step 4: Run the full shim test file**

Run: `uv run pytest tests/unit/method_spawners/test_claude_code_shim.py -q`
Expected: all pass (old + 3 new)

- [ ] **Step 5: Lint + type-check + commit**

```bash
uv run ruff format method/spawners/ tests/unit/method_spawners/
uv run ruff check method/spawners/ tests/unit/method_spawners/ && uv run pyrefly check
git add method/spawners/claude_code_shim.py tests/unit/method_spawners/
git commit -m "feat(method): claude_code corpus mode — cwd confinement + citation path normalization"
```

---

### Task 3: Tool-agnostic corpus phrasing in the shared envelope

**Files:**
- Modify: `packages/atp-method/atp_method/envelopes.py` (the corpus block in
  `build_prompt`)
- Test: `tests/unit/method_plugin/test_envelopes.py` (or wherever
  `build_prompt` corpus tests live — locate with
  `grep -rn "file_read" tests/unit --include="*.py" -l`)

**Interfaces:**
- Consumes: nothing new.
- Produces: corpus prompt block no longer names `file_read`; instructs
  relative-path citations. anthropic_api (HTTP tool loop) and native CLIs
  read the SAME text.

- [ ] **Step 1: Update the failing expectation first** — find the existing
  test asserting the corpus block text (`grep -rn "Read-only corpus" tests/`)
  and change it to the new phrasing:

```python
    assert "Read-only corpus files are available to your file-reading tool" in prompt
    assert "Cite source paths relative to the corpus root" in prompt
    assert "file_read" not in prompt_block_for_cli_case  # tool name gone
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit -k envelope -q`
Expected: FAIL on old phrasing

- [ ] **Step 3: Change the block in `build_prompt`**

```python
    if input_data.get("run_mode") == "read_only_corpus" and corpus:
        corpus_id = corpus.get("id", "corpus")
        paths = corpus.get("files") or []
        if paths:
            path_list = "\n".join(f"- {path}" for path in paths)
            body += (
                "\n\nRead-only corpus files are available to your "
                "file-reading tool. Cite source paths relative to the "
                f"corpus root. Corpus id: {corpus_id}. Available paths:\n"
                f"{path_list}"
            )
```

- [ ] **Step 4: Run the envelope + shim + evaluator test set**

Run: `uv run pytest tests/unit -k "envelope or shim or citation" -q`
Expected: pass. (Case YAML untouched → `SUITE.lock.toml` hashes unchanged —
verify with `uv run python method/run_pipe_check.py --case-dir method/cases/req-extraction --agents claude_code@claude-sonnet-4-6 --dry-run` exiting 0.)

- [ ] **Step 5: Commit**

```bash
git add packages/atp-method/atp_method/envelopes.py tests/
git commit -m "feat(method): tool-agnostic corpus prompt block (Path A prerequisite)"
```

---

### Task 4: Harness unskip + preparer registration for corpus-capable harnesses

**Files:**
- Modify: `method/run_pipe_check.py` — `CORPUS_CAPABLE_HARNESSES`, per-agent
  corpus filtering, preparer registration
- Test: `tests/unit/method_spawners/test_run_pipe_check.py` (extend the #217
  regression tests)

**Interfaces:**
- Consumes: Task 2's shim behavior (workspace-driven corpus mode).
- Produces: `CORPUS_CAPABLE_HARNESSES: frozenset[str] = frozenset({"claude_code"})`;
  `_corpus_case_ids(case_dir)` unchanged; the SKIP of corpus cases becomes
  per-agent: skipped only when `harness not in CORPUS_CAPABLE_HARNESSES`.

Implementation notes:
- Today the skip happens once at suite load (`_axis_by_id` excludes corpus
  cases; the run loop also reports `skipped_corpus`). Restructure minimally:
  load the FULL suite once; inside the per-agent loop, drop corpus cases
  only for non-capable harnesses (harness = `agent_id.split("@", 1)[0]`).
- Preparer registration: the orchestrator looks up
  `input_data["request_preparer"] == "corpus"` per test; `atp test` registers
  it via the plugin entry point, but run_pipe_check drives `TestOrchestrator`
  directly — register once at harness startup:

```python
def _register_corpus_preparer() -> None:
    """Corpus cases need the materialize/verify preparer; atp's plugin
    entry point registers it under `atp test`, but this harness drives
    TestOrchestrator directly, so register it ourselves. Idempotent."""
    from atp.runner.preparation import register_request_preparer

    from atp_method.runtime import CorpusRunPreparer

    register_request_preparer("corpus", CorpusRunPreparer())
```

Call it in `main()` before the agent loop (unconditionally — cheap, and
corpus cases may appear in any case dir).

- [ ] **Step 1: Write failing tests** (extend `test_run_pipe_check.py`)

```python
def test_corpus_cases_kept_for_capable_harness(tmp_path: Path) -> None:
    from method.run_pipe_check import CORPUS_CAPABLE_HARNESSES, _cases_for_harness

    (tmp_path / "inline.yaml").write_text(
        "id: case-inline-001\naxis_level: severe\ntask_type: req-extraction\n"
    )
    (tmp_path / "corpus.yaml").write_text(
        "id: case-corpus-001\naxis_level: clean\nrun_mode: read_only_corpus\n"
        "task_type: req-extraction\n"
    )
    assert "claude_code" in CORPUS_CAPABLE_HARNESSES
    kept = _cases_for_harness(tmp_path, "claude_code")
    assert kept == {"case-inline-001", "case-corpus-001"}
    kept_non = _cases_for_harness(tmp_path, "deepseek")
    assert kept_non == {"case-inline-001"}
```

(Adapt the exact helper name/shape to the current structure of the skip
logic around `_axis_by_id`/`_corpus_case_ids` — keep `_corpus_case_ids`
itself untouched so the #217 tests stay valid. If the cleanest seam differs,
preserve the observable contract in this test.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/method_spawners/test_run_pipe_check.py -q`
Expected: FAIL — helper/constant missing

- [ ] **Step 3: Implement** — `CORPUS_CAPABLE_HARNESSES` constant, per-agent
  filtering in the run loop, `_register_corpus_preparer()` called from
  `main()`, and update the `[skip]` notice to only count cases actually
  skipped for that agent.

- [ ] **Step 4: Run the whole harness test file**

Run: `uv run pytest tests/unit/method_spawners/test_run_pipe_check.py -q`
Expected: all pass (incl. existing #217 tests)

- [ ] **Step 5: Lint + type-check + commit**

```bash
uv run ruff format method/ tests/unit/method_spawners/
uv run ruff check method/ tests/ && uv run pyrefly check
git add method/run_pipe_check.py tests/unit/method_spawners/test_run_pipe_check.py
git commit -m "feat(method): run corpus cases for corpus-capable harnesses (claude_code first)"
```

---

### Task 5: Live smoke — the merge gate (paid, 1 case)

**Files:** none (execution only; results land in `_bench_output/`)

- [ ] **Step 1: Run the corpus case live for claude_code**

```bash
set -a; source .env; set +a
uv run python method/run_pipe_check.py \
  --case-dir method/cases/req-extraction \
  --agents claude_code@claude-sonnet-4-6 \
  --only-case case-req-extraction-fabricated-deadline-corpus-clean-001 \
  --out-dir _bench_output/r07-pipecheck/corpus-smoke-2026-07-05
```

(If `--only-case` does not exist, run the full req-extraction dir — 13 inline
cases are cheap and known-perfect for claude; the corpus case is the new
signal. Check `--help` first.)

- [ ] **Step 2: Verify the gate criteria (spec §7)**

1. Corpus case `status: "completed"` (not failed/skipped).
2. `critical_pass == true` — the citation parsed and points at
   `policy-current.md`, NOT `archive/policy-2023.md` (recency trap exercised).
3. Raw artifacts under `raw/` show the CLI actually read corpus files
   (spot-check stdout for Read tool usage).

- [ ] **Step 3: Record the result** in the PR description (score + tokens +
  cost). If the smoke fails on confinement or parsing — STOP, diagnose with
  `ATP_SHIM_RAW_DIR` output before any re-run (paid).

---

### Deferred (explicitly out of this slice)

- `CorpusRunPreparer` split / `serve_http=False` variant (spec §5.1) — the
  always-on local HTTP server is harmless; split when it hurts.
- `codex_cli` (`--sandbox read-only` + `-C`), `pi`, `opencode` corpus modes
  (spec §8 items 3-4) — one CLI per slice.
- Read-event recording (spec §5.5) — optional, CLI-dependent.
- Corpus cases in the paid full-roster sweep (spec §8 item 5).
