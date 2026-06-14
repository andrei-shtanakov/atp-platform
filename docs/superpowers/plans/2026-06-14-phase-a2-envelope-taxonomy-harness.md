# Phase A-2: capability envelope + taxonomy registry + harness parameterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish hardening the spine: lift the output envelope out of the spawner into a shared capability-level module, stand up the `task_type ↔ benchmark_id` taxonomy registry, and parameterize the run harness by `--task-type` (drop the hard-coded `benchmark_id`).

**Architecture:** The envelope and taxonomy are methodology-level concerns, so they live in the installed `atp_method` package. The spawner shims run via the venv interpreter, so they can `import atp_method.*` directly — both shims import the shared envelope, killing the current `anthropic_api_shim → claude_code_shim` cross-import (the N×M drift the ADR flags). The harness derives `benchmark_id` from `--task-type` via the registry instead of a constant.

**Tech Stack:** Python 3.12, uv, pydantic, pytest; packages `atp-method`; `method/spawners/*` shims; `method/run_pipe_check.py`.

**Companion docs:** ADR-006 `docs/adr/006-unified-capability-test-types.md` (seams #2, #3; resolved Q1/Q3), spec `docs/superpowers/specs/2026-06-14-eval-results-architecture-design.md` (§6, §7). Phase A-1 (CaseVerdict + checker registry) is merged on `main`.

**Scope guard (NOT in this plan):** a `grader`/case schema field for per-capability envelope selection (YAGNI until a 2nd capability needs a different envelope — the registry keys on capability now, default `review`); SP-1 store columns/persistence; dashboard. Selection stays single (`review`) — only the *location* and *coupling* change.

---

## File Structure

- Create `packages/atp-method/atp_method/envelopes.py` — `DEFAULT_MODEL`, `REVIEW_ENVELOPE`, `build_prompt(request, envelope)`, `get_envelope(capability="review")`. The shared, spawner-agnostic prompt contract.
- Create `packages/atp-method/atp_method/taxonomy.py` — `TASK_TYPE_TO_BENCHMARK_ID`, `benchmark_id_for(task_type)`. The one taxonomy registry (store/CLI speak `task_type`; `benchmark_id` is derived for the arbiter export).
- Modify `method/spawners/claude_code_shim.py` — import `DEFAULT_MODEL`, `build_prompt`, `get_envelope` from `atp_method.envelopes`; drop the local `REVIEW_ENVELOPE`/`_build_prompt`; `MODEL` reads `CLAUDE_MODEL` defaulting to `DEFAULT_MODEL`.
- Modify `method/spawners/anthropic_api_shim.py` — import from `atp_method.envelopes` (NOT from `claude_code_shim`); same `MODEL` default source.
- Modify `method/run_pipe_check.py` — add `--task-type` (default `review`); derive `benchmark_id` via `benchmark_id_for`; drop the `BENCHMARK_ID` constant; thread `benchmark_id` into `_run_agent`.
- Tests: `packages/atp-method/tests/test_envelopes.py`, `packages/atp-method/tests/test_taxonomy.py` (new). Existing `tests/unit/method_spawners/test_*_shim.py` must stay green unchanged.

**Test cwd note:** `atp-method` package tests run from `packages/atp-method` (`cd packages/atp-method && uv run pytest …`). Shim tests live under the repo-root `tests/` tree (`uv run pytest tests/unit/method_spawners -q`).

**Import-feasibility note (verified):** the CLI adapter runs `command=sys.executable, args=[shim]`, i.e. the venv interpreter, whose site-packages contain the editable `atp_method`. So `from atp_method.envelopes import …` resolves inside the shim subprocess regardless of the adapter's filtered env (site-packages come from the interpreter, not `PYTHONPATH`). The offline shim tests also invoke `sys.executable`, so they keep working.

---

## Task 1: capability envelope module

**Files:**
- Create: `packages/atp-method/atp_method/envelopes.py`
- Test: `packages/atp-method/tests/test_envelopes.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/atp-method/tests/test_envelopes.py
"""Tests for the shared capability envelope module (Phase A-2)."""

import pytest

from atp_method.envelopes import (
    DEFAULT_MODEL,
    REVIEW_ENVELOPE,
    build_prompt,
    get_envelope,
)


def test_default_model_is_pinned() -> None:
    assert DEFAULT_MODEL == "claude-opus-4-8"


def test_get_envelope_review() -> None:
    assert get_envelope("review") is REVIEW_ENVELOPE
    assert "{task}" in get_envelope("review")


def test_get_envelope_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_envelope("nope")


def test_build_prompt_inlines_task_and_artifacts() -> None:
    request = {
        "task": {"description": "Review the diff"},
        "context": {"artifacts": [{"id": "diff", "content": "x = 1"}]},
    }
    prompt = build_prompt(request, get_envelope("review"))
    assert "Review the diff" in prompt
    assert "--- diff ---" in prompt
    assert "x = 1" in prompt


def test_build_prompt_tolerates_missing_fields() -> None:
    assert isinstance(build_prompt({}, "{task}"), str)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd packages/atp-method && uv run pytest tests/test_envelopes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'atp_method.envelopes'`.

- [ ] **Step 3: Create the module**

```python
# packages/atp-method/atp_method/envelopes.py
"""Shared, spawner-agnostic output envelopes for the agent-eval-case methodology.

The envelope is the output contract handed to a spawner — it belongs to the
capability, NOT to any one spawner (ADR-006 seam #2). Keeping it here, in the
installed package, lets every shim import the SAME envelope (the shim subprocess
runs under the venv interpreter, so atp_method is importable) instead of one shim
importing another — which removes the N×M drift and keeps the API-vs-CLI ablation
equivalent by construction.
"""

# Pinned model for the code-review vertical (override per shim via CLAUDE_MODEL).
# Shared so both spawners pin the SAME model (the ablation's equivalence guard).
DEFAULT_MODEL = "claude-opus-4-8"

REVIEW_ENVELOPE = (
    "You are a senior code reviewer. Review the material below. Output ONLY a JSON "
    "array of findings (no prose, no markdown fence). Each finding is an object with "
    'keys: "rule_id" (the rule/CWE id), "file", "anchor" (the exact offending code '
    'substring), "severity" (critical|major|minor), "fix". If the code is compliant, '
    "output an empty array [].\n\n{task}"
)

# capability -> envelope. One entry today; a new capability adds one here.
_ENVELOPES: dict[str, str] = {"review": REVIEW_ENVELOPE}


def get_envelope(capability: str = "review") -> str:
    """Return the output envelope for a capability. Raises KeyError if unknown."""
    return _ENVELOPES[capability]


def build_prompt(request: dict, envelope: str) -> str:
    """Wrap an ATPRequest's task + inline artifacts in the given envelope."""
    task = request.get("task") or {}
    body = task.get("description", "")
    for art in (request.get("context") or {}).get("artifacts", []) or []:
        if art.get("content"):
            body += f"\n\n--- {art.get('id', 'artifact')} ---\n{art['content']}"
    return envelope.format(task=body)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd packages/atp-method && uv run pytest tests/test_envelopes.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-method && uv run pyrefly check
git add packages/atp-method/atp_method/envelopes.py packages/atp-method/tests/test_envelopes.py
git commit -m "feat(method): shared capability envelope module (Phase A-2)"
```

---

## Task 2: decouple claude_code_shim from the local envelope

**Files:**
- Modify: `method/spawners/claude_code_shim.py`
- Test: `tests/unit/method_spawners/test_claude_code_shim.py` (must stay green, unchanged)

- [ ] **Step 1: Confirm the existing shim test passes (baseline)**

Run: `uv run pytest tests/unit/method_spawners/test_claude_code_shim.py -q`
Expected: PASS (1 passed).

- [ ] **Step 2: Edit `method/spawners/claude_code_shim.py`**

Replace the model constant + the local `REVIEW_ENVELOPE` block + `_build_prompt` function. After the edit, the top of the module reads:

```python
from atp_method.envelopes import DEFAULT_MODEL, build_prompt, get_envelope

MODEL = os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
```

Delete the local `REVIEW_ENVELOPE = (...)` assignment and the local `def _build_prompt(request: dict) -> str: ...` function entirely.

In `main()`, replace the call `prompt = _build_prompt(request)` with:

```python
    prompt = build_prompt(request, get_envelope("review"))
```

(Keep everything else — the `claude -p` invocation, output normalization, error handling — unchanged. Ensure `os` is still imported.)

- [ ] **Step 3: Run the shim test to verify it still passes**

Run: `uv run pytest tests/unit/method_spawners/test_claude_code_shim.py -q`
Expected: PASS (1 passed) — the fake `claude` ignores the prompt, so identical output proves the envelope move is transparent.

- [ ] **Step 4: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check method/spawners/claude_code_shim.py && uv run pyrefly check
git add method/spawners/claude_code_shim.py
git commit -m "refactor(method): claude_code_shim uses shared envelope module (Phase A-2)"
```

---

## Task 3: decouple anthropic_api_shim from claude_code_shim

**Files:**
- Modify: `method/spawners/anthropic_api_shim.py`
- Test: `tests/unit/method_spawners/test_anthropic_api_shim.py` (must stay green, unchanged)

- [ ] **Step 1: Confirm the existing shim test passes (baseline)**

Run: `uv run pytest tests/unit/method_spawners/test_anthropic_api_shim.py -q`
Expected: PASS (4 passed).

- [ ] **Step 2: Edit `method/spawners/anthropic_api_shim.py`**

Replace the cross-shim import:

```python
from claude_code_shim import MODEL, _build_prompt
```

with:

```python
from atp_method.envelopes import DEFAULT_MODEL, build_prompt, get_envelope

MODEL = os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)
```

(Place the `MODEL = ...` line next to the existing `MAX_TOKENS = ...` near the top; ensure `os` is imported — it already is.)

In `main()`, replace `prompt = _build_prompt(request)` with:

```python
    prompt = build_prompt(request, get_envelope("review"))
```

Update the module docstring line that says "the shared REVIEW_ENVELOPE" to reference `atp_method.envelopes` (it already imports the same envelope — now from the shared module, not from the CLI shim).

- [ ] **Step 3: Run the shim test to verify it still passes**

Run: `uv run pytest tests/unit/method_spawners/test_anthropic_api_shim.py -q`
Expected: PASS (4 passed) — the fake SDK ignores the prompt, so identical behavior proves the decoupling is transparent.

- [ ] **Step 4: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check method/spawners/anthropic_api_shim.py && uv run pyrefly check
git add method/spawners/anthropic_api_shim.py
git commit -m "refactor(method): anthropic_api_shim imports shared envelope, not the CLI shim (Phase A-2)"
```

---

## Task 4: taxonomy registry (`task_type ↔ benchmark_id`)

**Files:**
- Create: `packages/atp-method/atp_method/taxonomy.py`
- Test: `packages/atp-method/tests/test_taxonomy.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/atp-method/tests/test_taxonomy.py
"""Tests for the task_type <-> benchmark_id taxonomy registry (Phase A-2)."""

import pytest

from atp_method.taxonomy import TASK_TYPE_TO_BENCHMARK_ID, benchmark_id_for


def test_review_maps_to_code_review() -> None:
    assert benchmark_id_for("review") == "code-review"


def test_registry_is_the_source() -> None:
    assert TASK_TYPE_TO_BENCHMARK_ID["review"] == "code-review"


def test_unknown_task_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown task_type"):
        benchmark_id_for("does-not-exist")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd packages/atp-method && uv run pytest tests/test_taxonomy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'atp_method.taxonomy'`.

- [ ] **Step 3: Create the module**

```python
# packages/atp-method/atp_method/taxonomy.py
"""The one taxonomy registry mapping internal `task_type` to the arbiter export
`benchmark_id` (ADR-006 direction #3).

The internal store / CLI / dashboard speak `task_type` (the arbiter `TaskType`
canon, e.g. "review"); `benchmark_id` (e.g. "code-review") exists only on the
`report_benchmark-v1` export to arbiter and is derived here at the sink. Keeping
the map in one place stops "benchmark" leaking back onto the internal store.
"""

# task_type (internal) -> benchmark_id (arbiter export key).
TASK_TYPE_TO_BENCHMARK_ID: dict[str, str] = {
    "review": "code-review",  # arbiter TaskType::Review (ordinal 5)
}


def benchmark_id_for(task_type: str) -> str:
    """Return the arbiter export benchmark_id for an internal task_type."""
    try:
        return TASK_TYPE_TO_BENCHMARK_ID[task_type]
    except KeyError as exc:
        known = sorted(TASK_TYPE_TO_BENCHMARK_ID)
        raise ValueError(
            f"unknown task_type {task_type!r}; known: {known}"
        ) from exc
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd packages/atp-method && uv run pytest tests/test_taxonomy.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-method && uv run pyrefly check
git add packages/atp-method/atp_method/taxonomy.py packages/atp-method/tests/test_taxonomy.py
git commit -m "feat(method): task_type<->benchmark_id taxonomy registry (Phase A-2)"
```

---

## Task 5: parameterize the harness by `--task-type`

**Files:**
- Modify: `method/run_pipe_check.py`

- [ ] **Step 1: Baseline — confirm the offline harness still runs**

Run (from repo root):
```bash
FAKE="$(git rev-parse --show-toplevel)/tests/unit/method_spawners/fixtures/fake_claude.py"
CLAUDE_BIN="python $FAKE" uv run python method/run_pipe_check.py --agents claude_code --dry-run
```
Expected: prints the plan and `[dry-run] Would run (PAID): ['claude_code']`.

- [ ] **Step 2: Edit `method/run_pipe_check.py`**

(a) Add the import next to the other `atp_method` imports:
```python
from atp_method.taxonomy import benchmark_id_for
```
(b) Delete the module-level constant `BENCHMARK_ID = "code-review"`.
(c) Change `_run_agent`'s signature to accept `benchmark_id` and use it. Replace:
```python
async def _run_agent(
    agent_id: str,
    case_dir: Path,
    axis_by_id: dict[str, str],
    runs: int,
    with_rubric: bool,
    timeout_s: float,
) -> dict[str, Any]:
```
with (add `benchmark_id: str`):
```python
async def _run_agent(
    agent_id: str,
    case_dir: Path,
    axis_by_id: dict[str, str],
    runs: int,
    with_rubric: bool,
    timeout_s: float,
    benchmark_id: str,
) -> dict[str, Any]:
```
and in its `build_report_benchmark_payload(...)` call change `benchmark_id=BENCHMARK_ID,` to `benchmark_id=benchmark_id,`.
(d) In `_main_async`, after computing `agents`, derive the benchmark_id once:
```python
    benchmark_id = benchmark_id_for(args.task_type)
```
and pass it through the `_run_agent(...)` call:
```python
        payload = await _run_agent(
            agent_id,
            case_dir,
            axis_by_id,
            args.runs,
            args.with_rubric,
            args.timeout,
            benchmark_id,
        )
```
(e) In `main()`, add the argument (next to `--case-dir`):
```python
    p.add_argument(
        "--task-type",
        default="review",
        help="internal task_type; benchmark_id is derived for the arbiter export",
    )
```
(f) In `_main_async`'s header print, surface it:
```python
    print(f"Pipe-check: {n_cases} case(s) in {case_dir} | task_type={args.task_type}")
```
(replace the existing `print(f"Pipe-check: {n_cases} case(s) in {case_dir}")` line).

- [ ] **Step 3: Verify — dry-run with a known and an unknown task_type**

Run (from repo root):
```bash
FAKE="$(git rev-parse --show-toplevel)/tests/unit/method_spawners/fixtures/fake_claude.py"
CLAUDE_BIN="python $FAKE" uv run python method/run_pipe_check.py --agents claude_code --task-type review --dry-run
CLAUDE_BIN="python $FAKE" uv run python method/run_pipe_check.py --agents claude_code --task-type bogus --dry-run; echo "exit=$?"
```
Expected: the `review` run prints the plan (`task_type=review`); the `bogus` run exits non-zero with a `ValueError: unknown task_type 'bogus'` traceback (the registry rejects it).

- [ ] **Step 4: Full offline E2E — payload benchmark_id is derived**

Run (from repo root):
```bash
FAKE="$(git rev-parse --show-toplevel)/tests/unit/method_spawners/fixtures/fake_claude.py"
rm -rf /tmp/pa2 && CLAUDE_BIN="python $FAKE" uv run python method/run_pipe_check.py --agents claude_code --task-type review --out-dir /tmp/pa2
uv run python -c "import json; print('benchmark_id =', json.load(open('/tmp/pa2/report_benchmark_claude_code.json'))['benchmark_id'])"
```
Expected: `benchmark_id = code-review`.

- [ ] **Step 5: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff format method/run_pipe_check.py
uv run ruff check method/run_pipe_check.py && uv run pyrefly check
git add method/run_pipe_check.py
git commit -m "feat(method): harness --task-type derives benchmark_id via taxonomy (Phase A-2)"
```

---

## Task 6: full regression + quality gates

**Files:** none (verification only)

- [ ] **Step 1: Run the shim + method suites**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
uv run pytest tests/unit/method_spawners tests/unit/reporters -q
cd packages/atp-method && uv run pytest -q
```
Expected: all PASS (shim tests green proves the envelope/decoupling is transparent; method package green incl. new envelopes/taxonomy tests).

- [ ] **Step 2: Confirm the cross-shim import is gone**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
grep -rn "from claude_code_shim" method/ && echo "STILL COUPLED" || echo "OK: no shim imports another shim"
grep -rn "BENCHMARK_ID =" method/run_pipe_check.py && echo "CONSTANT STILL PRESENT" || echo "OK: hard-coded benchmark_id removed"
```
Expected: both print the `OK:` line.

- [ ] **Step 3: Lint + types (whole touched surface)**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check atp/ packages/atp-method method/
uv run pyrefly check
```
Expected: ruff clean; pyrefly 0 errors.

- [ ] **Step 4: Commit any formatting**

```bash
cd "$(git rev-parse --show-toplevel)"
git add -A && git commit -m "chore(phase-a2): formatting" || echo "nothing to commit"
```

---

## Self-Review (completed during authoring)

- **Spec/ADR coverage:** seam #2 envelope decoupling (Tasks 1–3), seam #3 generic run path + taxonomy registry (Tasks 4–5). Envelope selection kept single (`review`) per scope guard; capability keying is in place for the next vertical.
- **Type consistency:** `build_prompt(request, envelope)` and `get_envelope(capability)` (Task 1) are imported and called identically in both shims (Tasks 2–3). `benchmark_id_for(task_type)` (Task 4) returns the `benchmark_id` threaded into `_run_agent(..., benchmark_id)` and `build_report_benchmark_payload(benchmark_id=...)` (Task 5). `DEFAULT_MODEL` is the single model pin both shims read via `CLAUDE_MODEL`.
- **Placeholders:** none — every code/test step carries full content.
- **Risk watch:** the shims now depend on `atp_method` being importable in the subprocess — verified true (venv interpreter → site-packages). The offline shim tests exercise exactly this path and must stay green (Tasks 2/3 Step 3, Task 6 Step 1).
