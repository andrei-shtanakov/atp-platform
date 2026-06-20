# Agent roster Tier-2 (pi, opencode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pi@gpt-5` and `opencode@glm-5.1` as non-routable CLI agents in the pipe-check roster, via two new spawner shims over a shared CLI runner.

**Architecture:** A shared `method/spawners/_cli_common.py` holds the subprocess+timeout+contract boilerplate (mirrors how `_openai_compat.py` is shared for API shims). Thin `opencode_shim.py` / `pi_shim.py` supply their argv template + a JSONL parser. The registry adds both harnesses; both are non-routable (no Maestro/arbiter changes).

**Tech Stack:** Python 3.12 stdlib (subprocess, json), pytest, uv. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-20-agent-roster-tier2-design.md`.

## Global Constraints

- `uv` only; run via `uv run`. Type hints; whole-project `uv run pyrefly check` exits 0 (the sibling `from _cli_common import ...` carries `# pyrefly: ignore[missing-import]`, like the existing thin shims). `uv run ruff format/check` clean; line length 88.
- Branch: `r07/agent-roster-tier2` (already created; spec committed). Never work on `main`.
- Shims are stdlib-only, emit the ATPResponse contract on stdout, and turn ANY error (bad stdin, missing model, non-zero exit, timeout, empty output) into a `status:"failed"` response — never crash.
- `agent_id` is slash-free (`pi@gpt-5`, `opencode@glm-5.1`); the provider prefix (`openai/`, `opencode/`) is added by the shim, never in the id.
- Both agents are **non-routable** — no Maestro spawner / `AgentType`, no arbiter `config/agents.toml`.
- pi may hang in agentic mode → the shared runner enforces a hard timeout; pi's viability is gated on the live smoke (Task 4). If pi fails the smoke, drop `("pi","gpt-5")`.

---

### Task 1: Shared CLI runner + opencode shim

**Files:**
- Create: `method/spawners/_cli_common.py`, `method/spawners/opencode_shim.py`
- Test: `tests/unit/method_spawners/test_cli_shims.py`

**Interfaces:**
- Produces: `_cli_common.fail(task_id, error) -> int`; `_cli_common.build_response(task_id, text, in_tok, out_tok) -> dict`; `_cli_common.model_arg(model, default_provider) -> str`; `_cli_common.run(*, bin_env, default_bin, model_env, default_provider, argv, parse_output) -> int` where `argv(binary_tokens: list[str], model: str, prompt: str) -> list[str]` and `parse_output(stdout: str) -> tuple[str, int|None, int|None]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/method_spawners/test_cli_shims.py`:

```python
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

_SPAWNERS = Path(__file__).resolve().parents[3] / "method" / "spawners"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SPAWNERS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_model_arg_prefixes_provider_when_bare() -> None:
    cli = _load("_cli_common")
    assert cli.model_arg("gpt-5", "openai") == "openai/gpt-5"
    assert cli.model_arg("glm-5.1", "opencode") == "opencode/glm-5.1"
    # already provider-qualified → unchanged
    assert cli.model_arg("openai/gpt-5", "openai") == "openai/gpt-5"


def test_build_response_shape_and_token_total() -> None:
    cli = _load("_cli_common")
    r = cli.build_response("t1", "[]", 10, 4)
    assert r["status"] == "completed"
    assert r["task_id"] == "t1"
    assert r["artifacts"][0]["content"] == "[]"
    assert r["metrics"]["total_tokens"] == 14
    assert r["metrics"]["cost_usd"] is None
    r2 = cli.build_response("t", "x", None, 4)  # partial → total None
    assert r2["metrics"]["total_tokens"] is None


def test_opencode_parse_output_text_and_tokens() -> None:
    oc = _load("opencode_shim")
    stdout = (
        '{"type":"text","part":{"text":"["}}\n'
        '{"type":"text","part":{"text":"]"}}\n'
        '{"type":"step_finish","part":{"tokens":{"total":20,"input":15,"output":5}}}\n'
    )
    text, in_tok, out_tok = oc._parse(stdout)
    assert text == "[]"
    assert (in_tok, out_tok) == (15, 5)


def _run_shim(shim: str, env_extra: dict[str, str]) -> dict:
    # Force the binary to a non-existent command so the subprocess errors out
    # without any network/CLI dependency.
    env = dict(os.environ)
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(_SPAWNERS / f"{shim}.py")],
        input=json.dumps({"task_id": "t", "task": {"description": "x", "input_data": {}}}).encode(),
        capture_output=True,
        env=env,
        timeout=30,
    )
    return json.loads(proc.stdout.decode())


def test_opencode_shim_fails_when_binary_missing() -> None:
    out = _run_shim("opencode_shim", {"OPENCODE_BIN": "definitely-not-a-real-bin-xyz", "OPENCODE_MODEL": "glm-5.1"})
    assert out["status"] == "failed"
    assert "invocation error" in out["error"] or "failed" in out["error"]


def test_opencode_shim_fails_without_model() -> None:
    out = _run_shim("opencode_shim", {"OPENCODE_MODEL": ""})
    assert out["status"] == "failed"
    assert "OPENCODE_MODEL not set" in out["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/method_spawners/test_cli_shims.py -v`
Expected: FAIL — `_cli_common` / `opencode_shim` don't exist.

- [ ] **Step 3: Create `method/spawners/_cli_common.py`**

```python
#!/usr/bin/env python3
"""Shared CLI-spawner logic for ATP's CLI adapter (Tier-2 agents).

A thin per-tool shim supplies an argv template + a JSONL parser; this module
runs the subprocess with a hard timeout and normalizes the result into the
ATPResponse contract on stdout. Any error (bad stdin, missing model, non-zero
exit, timeout, empty output) becomes a status:"failed" response — never a crash.
Mirrors codex_cli_shim.py; stdlib only.
"""

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable

from atp_method.envelopes import build_prompt, get_envelope

REQUEST_TIMEOUT_S = 600.0


def fail(task_id: str, error: str) -> int:
    """Emit a status=failed ATPResponse (the adapter reads it off stdout)."""
    sys.stdout.write(
        json.dumps(
            {
                "version": "1.0",
                "task_id": task_id,
                "status": "failed",
                "artifacts": [],
                "metrics": {},
                "error": error[:2000],
            }
        )
    )
    return 0


def build_response(
    task_id: str, text: str, in_tok: int | None, out_tok: int | None
) -> dict:
    """Normalize a CLI run into a completed ATPResponse dict."""
    total = in_tok + out_tok if in_tok is not None and out_tok is not None else None
    return {
        "version": "1.0",
        "task_id": task_id,
        "status": "completed",
        "artifacts": [
            {
                "type": "file",
                "path": "review.md",
                "content": text,
                "content_type": "text/markdown",
            }
        ],
        "metrics": {
            "total_tokens": total,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": None,
        },
    }


def model_arg(model: str, default_provider: str) -> str:
    """A bare model id gets the tool's provider prefix; an already
    provider-qualified id (contains '/') passes through unchanged."""
    return model if "/" in model else f"{default_provider}/{model}"


def run(
    *,
    bin_env: str,
    default_bin: str,
    model_env: str,
    default_provider: str,
    argv: Callable[[list[str], str, str], list[str]],
    parse_output: Callable[[str], tuple[str, int | None, int | None]],
) -> int:
    """Drive one CLI tool. ``argv(binary_tokens, model, prompt) -> list[str]``;
    ``parse_output(stdout) -> (text, input_tokens, output_tokens)``."""
    raw_in = sys.stdin.read()
    try:
        request = json.loads(raw_in)
    except (ValueError, TypeError) as exc:
        return fail("", f"invalid ATPRequest JSON on stdin: {exc}")
    task_id = request.get("task_id", "")

    model = os.environ.get(model_env)
    if not model:
        return fail(task_id, f"{model_env} not set")
    binary = shlex.split(os.environ.get(bin_env, default_bin)) or [default_bin]

    prompt = build_prompt(request, get_envelope("review"))
    cmd = argv(binary, model_arg(model, default_provider), prompt)
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=REQUEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return fail(task_id, f"{binary[0]} timed out after {REQUEST_TIMEOUT_S}s")
    except (OSError, subprocess.SubprocessError) as exc:
        return fail(task_id, f"{binary[0]} invocation error: {exc}")

    if proc.returncode != 0:
        return fail(
            task_id,
            f"{binary[0]} failed (rc={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')[:2000]}",
        )
    text, in_tok, out_tok = parse_output(proc.stdout.decode(errors="replace"))
    if not text.strip():
        return fail(task_id, f"{binary[0]} produced no output text")

    sys.stdout.write(json.dumps(build_response(task_id, text, in_tok, out_tok)))
    return 0
```

- [ ] **Step 4: Create `method/spawners/opencode_shim.py`**

```python
#!/usr/bin/env python3
"""opencode spawner shim — `opencode run --format json` (GLM via opencode).

Non-routable API/CLI baseline row. Reads OPENCODE_MODEL (e.g. glm-5.1; the
shim prefixes the `opencode/` provider) and OPENCODE_BIN (default "opencode").
Auth is opencode's own (operator's OPENCODE_GLM_API_KEY / `opencode auth`).
"""

import json

from _cli_common import run  # pyrefly: ignore[missing-import]


def _argv(binary: list[str], model: str, prompt: str) -> list[str]:
    return [*binary, "run", "--format", "json", "-m", model, prompt]


def _parse(stdout: str) -> tuple[str, int | None, int | None]:
    """Concat `type:text` part.text; tokens from the `step_finish` event."""
    text = ""
    in_tok: int | None = None
    out_tok: int | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if event.get("type") == "text":
            text += (event.get("part") or {}).get("text", "")
        elif event.get("type") == "step_finish":
            tokens = (event.get("part") or {}).get("tokens") or {}
            in_tok = tokens.get("input")
            out_tok = tokens.get("output")
    return text, in_tok, out_tok


if __name__ == "__main__":
    raise SystemExit(
        run(
            bin_env="OPENCODE_BIN",
            default_bin="opencode",
            model_env="OPENCODE_MODEL",
            default_provider="opencode",
            argv=_argv,
            parse_output=_parse,
        )
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/method_spawners/test_cli_shims.py -v`
Expected: all pass (model_arg, build_response, opencode parse, binary-missing, no-model).

- [ ] **Step 6: Commit**

```bash
git add method/spawners/_cli_common.py method/spawners/opencode_shim.py tests/unit/method_spawners/test_cli_shims.py
git commit -m "feat(spawners): shared CLI runner + opencode shim"
```

---

### Task 2: pi shim

**Files:**
- Create: `method/spawners/pi_shim.py`
- Test: `tests/unit/method_spawners/test_cli_shims.py` (extend)

**Interfaces:**
- Consumes: `_cli_common.run` (Task 1).
- Produces: `pi_shim._parse(stdout) -> tuple[str, int|None, int|None]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/method_spawners/test_cli_shims.py`:

```python
def test_pi_parse_output_assistant_text_and_usage() -> None:
    pi = _load("pi_shim")
    stdout = (
        '{"type":"message_start","message":{"role":"assistant","content":[],'
        '"usage":{"input":0,"output":0,"totalTokens":0}}}\n'
        '{"type":"message_end","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"[]"}],'
        '"usage":{"input":12,"output":3,"totalTokens":15}}}\n'
    )
    text, in_tok, out_tok = pi._parse(stdout)
    assert text == "[]"
    assert (in_tok, out_tok) == (12, 3)


def test_pi_shim_fails_when_binary_missing() -> None:
    out = _run_shim("pi_shim", {"PI_BIN": "definitely-not-a-real-bin-xyz", "PI_MODEL": "gpt-5"})
    assert out["status"] == "failed"
    assert "invocation error" in out["error"] or "failed" in out["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/method_spawners/test_cli_shims.py -k pi -v`
Expected: FAIL — `pi_shim` doesn't exist.

- [ ] **Step 3: Create `method/spawners/pi_shim.py`**

```python
#!/usr/bin/env python3
"""pi (earendil-works) spawner shim — `pi -p --mode json`.

Non-routable CLI row. Reads PI_MODEL (e.g. gpt-5; the shim prefixes the
`openai/` provider, since a bare id routes pi to an unauthed azure provider)
and PI_BIN (default "pi"). pi authenticates via its own session (no key here).
`--no-prompt-templates` keeps the run lean; the shared runner's hard timeout
guards pi's agentic behavior (it can otherwise not terminate).
"""

import json

from _cli_common import run  # pyrefly: ignore[missing-import]


def _argv(binary: list[str], model: str, prompt: str) -> list[str]:
    return [*binary, "-p", "--mode", "json", "--no-prompt-templates",
            "--model", model, prompt]


def _parse(stdout: str) -> tuple[str, int | None, int | None]:
    """Last non-empty assistant message's content text + usage."""
    text = ""
    in_tok: int | None = None
    out_tok: int | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            continue
        parts = message.get("content") or []
        joined = "".join(
            p.get("text", "")
            for p in parts
            if isinstance(p, dict) and p.get("type") == "text"
        )
        if joined:
            text = joined  # message_end carries the full content
        usage = message.get("usage") or {}
        if usage.get("input") is not None:
            in_tok = usage.get("input")
        if usage.get("output") is not None:
            out_tok = usage.get("output")
    return text, in_tok, out_tok


if __name__ == "__main__":
    raise SystemExit(
        run(
            bin_env="PI_BIN",
            default_bin="pi",
            model_env="PI_MODEL",
            default_provider="openai",
            argv=_argv,
            parse_output=_parse,
        )
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/method_spawners/test_cli_shims.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add method/spawners/pi_shim.py tests/unit/method_spawners/test_cli_shims.py
git commit -m "feat(spawners): pi shim (openai/ provider prefix + timeout guard)"
```

---

### Task 3: Registry wiring (pi, opencode)

**Files:**
- Modify: `method/run_pipe_check.py` (`HARNESSES`, `AGENT_MODELS`, `ALLOWED_ENV`, `_preflight`)
- Test: `tests/unit/method_spawners/test_run_pipe_check.py`

**Interfaces:**
- Consumes: the shim paths from Tasks 1–2.
- Produces: `pi@gpt-5`, `opencode@glm-5.1` in `AGENTS`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/method_spawners/test_run_pipe_check.py`:

```python
def test_registry_has_pi_and_opencode() -> None:
    from method.run_pipe_check import AGENTS

    assert "pi@gpt-5" in AGENTS
    assert "opencode@glm-5.1" in AGENTS
    assert AGENTS["pi@gpt-5"]["model_env"] == "PI_MODEL"
    assert AGENTS["pi@gpt-5"]["shim"].endswith("pi_shim.py")
    assert AGENTS["opencode@glm-5.1"]["shim"].endswith("opencode_shim.py")


def test_preflight_skips_pi_opencode_without_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from method.run_pipe_check import _preflight

    monkeypatch.setenv("PI_BIN", "definitely-not-a-real-bin-xyz")
    monkeypatch.setenv("OPENCODE_BIN", "definitely-not-a-real-bin-xyz")
    assert "pi binary not found" in (_preflight("pi@gpt-5") or "")
    assert "opencode binary not found" in (_preflight("opencode@glm-5.1") or "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/method_spawners/test_run_pipe_check.py -k "pi_and_opencode or pi_opencode" -v`
Expected: FAIL — pi/opencode absent from the registry and `_preflight`.

- [ ] **Step 3: Add to `HARNESSES`**

In `method/run_pipe_check.py`, append to `HARNESSES`:

```python
    "ollama": ("method/spawners/ollama_shim.py", "OLLAMA_MODEL"),
    "mimo": ("method/spawners/mimo_shim.py", "MIMO_MODEL"),
    "qwen": ("method/spawners/qwen_shim.py", "QWEN_MODEL"),
    "pi": ("method/spawners/pi_shim.py", "PI_MODEL"),
    "opencode": ("method/spawners/opencode_shim.py", "OPENCODE_MODEL"),
}
```

- [ ] **Step 4: Add to `AGENT_MODELS`**

Append the two Tier-2 rows (after the ollama rows):

```python
    ("ollama", "qwen2.5:14b"),
    ("pi", "gpt-5"),
    ("opencode", "glm-5.1"),
]
```

- [ ] **Step 5: Allowlist the env vars**

Append to `ALLOWED_ENV`:

```python
    "QWEN_MODEL",
    "PI_BIN",
    "PI_MODEL",
    "OPENCODE_BIN",
    "OPENCODE_MODEL",
    "OPENCODE_GLM_API_KEY",
]
```

- [ ] **Step 6: Add preflight binary checks**

In `_preflight`, after the qwen branch and before the ollama branch:

```python
    if harness == "pi":
        binary = os.environ.get("PI_BIN", "pi")
        parts = shlex.split(binary) if binary else ["pi"]
        head = parts[0] if parts else "pi"
        if shutil.which(head) is None and not Path(head).exists():
            return f"pi binary not found (PI_BIN={binary!r})"
    if harness == "opencode":
        binary = os.environ.get("OPENCODE_BIN", "opencode")
        parts = shlex.split(binary) if binary else ["opencode"]
        head = parts[0] if parts else "opencode"
        if shutil.which(head) is None and not Path(head).exists():
            return f"opencode binary not found (OPENCODE_BIN={binary!r})"
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/unit/method_spawners/test_run_pipe_check.py -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add method/run_pipe_check.py tests/unit/method_spawners/test_run_pipe_check.py
git commit -m "feat(R-07): register pi + opencode (Tier-2, non-routable)"
```

---

### Task 4: Verification + live smoke (the gate)

**Files:** none (verification + operational).

- [ ] **Step 1: Format, lint, type-check**

```bash
uv run ruff format .
uv run ruff check method tests
uv run pyrefly check
```
Expected: clean; pyrefly exits 0.

- [ ] **Step 2: Run the touched suites**

```bash
uv run pytest tests/unit/method_spawners -q
```
Expected: all pass.

- [ ] **Step 3: Dry-run smoke — new ids present**

```bash
uv run python method/run_pipe_check.py --dry-run
```
Expected: the agent list includes `pi@gpt-5` and `opencode@glm-5.1`; exit 0. If the `pi`/`opencode` binaries are absent they SKIP with a clear reason.

- [ ] **Step 4: LIVE smoke — one code-review case per agent (the Tier-2 gate)**

```bash
uv run python method/run_pipe_check.py --agents opencode@glm-5.1 --case-dir method/cases/code-review --runs 1
uv run python method/run_pipe_check.py --agents pi@gpt-5 --case-dir method/cases/code-review --runs 1
```
Expected: each prints a `critical_pass_rate` line with `error_class` NOT `failed`/`timeout` on its cases (i.e. the agent actually ran). **opencode** should pass. **pi** is the risk: if it times out / produces no output (agentic hang), it is NOT viable — remove the `("pi","gpt-5")` row from `AGENT_MODELS` (keep the shim + registry harness for later) and note pi as deferred. Record the per-agent outcome.

- [ ] **Step 5: Sync TODO + commit**

In `TODO.md` under the R-07 roster section, record: Tier-2 landed 2026-06-20 — opencode@glm-5.1 (+ pi@gpt-5 if it smoked green); generic `_cli_common` CLI runner; both non-routable (no Maestro/arbiter). Note the live-smoke outcome for pi.

```bash
git add TODO.md
git commit -m "docs: record Tier-2 roster (opencode + pi) smoke outcome"
```

---

## Self-Review

- **Spec coverage:** shared CLI runner + opencode shim (Task 1); pi shim with provider-prefix + timeout (Task 2); registry HARNESSES/AGENT_MODELS/ALLOWED_ENV/_preflight (Task 3); live smoke gate + pi-drop contingency (Task 4). Runbook Case-B/ATP-only + non-routable → no Maestro/arbiter tasks (correct, by design). ADR HARNESSES-declarative refactor — not in scope (deferred), no task (correct). Provider-prefix as shim launch detail — Task 1 `model_arg` + Tasks 2/3 (slash-free ids).
- **Placeholder scan:** none — full code for `_cli_common`, both shims, registry edits, and tests. pi's "probe for an extra answer-only flag" lives in the spec as smoke-gated, not a code placeholder; the plan ships `--no-prompt-templates` + the hard timeout as the concrete guard.
- **Type consistency:** `run(*, bin_env, default_bin, model_env, default_provider, argv, parse_output)` defined in Task 1, consumed verbatim by both thin shims (Tasks 2/3 via the registry); `parse_output(stdout) -> tuple[str, int|None, int|None]` consistent across opencode/pi `_parse`; `model_arg(model, default_provider)` used by `run`; `HARNESSES` entries `(shim, model_env)` match the existing tuple shape; ids `pi@gpt-5` / `opencode@glm-5.1` consistent across AGENT_MODELS, tests, and smoke.
```
