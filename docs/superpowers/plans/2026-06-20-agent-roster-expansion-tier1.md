# Agent roster expansion — Tier-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Tier-1 weekend-sweep agents — claude on Sonnet 4.6 (CLI + API baseline), mimo and qwen (OpenAI-compatible APIs) — and retire opus, keeping the `<harness>@<model>` convention.

**Architecture:** A new generic OpenAI-compatible shim (`_openai_compat.py`) holds the chat-completions logic (mirrors `deepseek_shim.py`); thin per-provider shims (`mimo_shim.py`, `qwen_shim.py`) bake in their env prefix + default host. `run_pipe_check.py`'s registry swaps opus→sonnet and adds mimo/qwen rows; `deepseek_shim.py` is untouched.

**Tech Stack:** Python 3.12 stdlib (`urllib`), pytest, uv. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-20-agent-roster-expansion-design.md`.

## Global Constraints

- `uv` only; run tools via `uv run`. Type hints; whole-project `uv run pyrefly check` exits 0 (no NEW errors vs baseline). `uv run ruff format/check` clean; line length 88.
- Branch: `r07/agent-roster-expansion` (already created; spec + `.env.example` already committed). Never work on `main`.
- `agent_id = "<harness>@<model>"`; faithful model id. Sonnet model id is exactly `claude-sonnet-4-6`.
- Shims are stdlib-only and emit the ATPResponse JSON contract on stdout; a missing key / bad input yields a `status:"failed"` response, never a crash.
- Tier-2 (`pi`, `opencode`) is OUT of this plan — separate fast-follow.
- Non-goal: do NOT refactor `deepseek_shim.py` onto the shared helper (don't churn working Tier-1 code).

---

### Task 1: Generic OpenAI-compat shim + mimo/qwen thin shims

**Files:**
- Create: `method/spawners/_openai_compat.py`, `method/spawners/mimo_shim.py`, `method/spawners/qwen_shim.py`
- Test: `tests/unit/method_spawners/test_openai_compat_shim.py`

**Interfaces:**
- Produces: `_openai_compat.run(prefix: str, default_host: str) -> int` (reads `{prefix}_API_KEY`/`{prefix}_HOST`/`{prefix}_MODEL`, calls `{host}/v1/chat/completions`, writes an ATPResponse to stdout); `_openai_compat.build_response(request: dict, raw: dict) -> dict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/method_spawners/test_openai_compat_shim.py`:

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


def test_build_response_normalizes_openai_compat_payload() -> None:
    mod = _load("_openai_compat")
    raw = {
        "choices": [{"message": {"content": "[]"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }
    resp = mod.build_response({"task_id": "t1"}, raw)
    assert resp["status"] == "completed"
    assert resp["task_id"] == "t1"
    assert resp["artifacts"][0]["content"] == "[]"
    assert resp["metrics"]["total_tokens"] == 12
    assert resp["metrics"]["cost_usd"] is None


def test_build_response_missing_usage_is_none() -> None:
    mod = _load("_openai_compat")
    resp = mod.build_response({"task_id": "t"}, {"choices": [{"message": {"content": "x"}}]})
    assert resp["metrics"]["total_tokens"] is None


def _run_shim(shim: str, env_extra: dict[str, str]) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(_SPAWNERS / f"{shim}.py")],
        input=json.dumps({"task_id": "t"}).encode(),
        capture_output=True,
        env=env,
        timeout=30,
    )
    return json.loads(proc.stdout.decode())


def test_mimo_shim_fails_clearly_without_key() -> None:
    out = _run_shim("mimo_shim", {})
    assert out["status"] == "failed"
    assert "MIMO_API_KEY not set" in out["error"]


def test_qwen_shim_fails_clearly_without_key() -> None:
    out = _run_shim("qwen_shim", {})
    assert out["status"] == "failed"
    assert "QWEN_API_KEY not set" in out["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/method_spawners/test_openai_compat_shim.py -v`
Expected: FAIL — the shim files don't exist (`FileNotFoundError`/import error).

- [ ] **Step 3: Create `method/spawners/_openai_compat.py`**

```python
#!/usr/bin/env python3
"""Shared OpenAI-compatible spawner logic for ATP's CLI adapter.

Provider-agnostic core: read an ATPRequest JSON from stdin, call an
OpenAI-compatible ``/v1/chat/completions`` endpoint, normalize the result into
an ATPResponse JSON on stdout. A thin per-provider shim calls ``run(prefix,
default_host)`` — the prefix selects the ``{prefix}_API_KEY`` / ``{prefix}_HOST``
/ ``{prefix}_MODEL`` env vars. Mirrors ``deepseek_shim.py`` (kept separate so
working Tier-1 code is not churned). Stdlib only — no new dependency.
"""

import json
import os
import sys
import urllib.error
import urllib.request

from atp_method.envelopes import build_prompt, get_envelope

REQUEST_TIMEOUT_S = 300.0


def _fail(task_id: str, error: str) -> int:
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


def _call(host: str, key: str, model: str, prompt: str) -> dict:
    """POST a single user turn and return the parsed JSON (errors propagate)."""
    url = f"{host.rstrip('/')}/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode())


def build_response(request: dict, raw: dict) -> dict:
    """Normalize an OpenAI-compatible chat-completions payload into ATPResponse."""
    task_id = request.get("task_id", "")
    text = raw["choices"][0]["message"]["content"]
    usage = raw.get("usage") or {}
    in_tok = usage.get("prompt_tokens")
    out_tok = usage.get("completion_tokens")
    total = (
        (in_tok or 0) + (out_tok or 0)
        if in_tok is not None or out_tok is not None
        else None
    )
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


def run(prefix: str, default_host: str) -> int:
    """Drive one OpenAI-compatible provider selected by ``prefix`` env vars."""
    raw_in = sys.stdin.read()
    try:
        request = json.loads(raw_in)
    except (ValueError, TypeError) as exc:
        return _fail("", f"invalid ATPRequest JSON on stdin: {exc}")
    task_id = request.get("task_id", "")

    key = os.environ.get(f"{prefix}_API_KEY")
    if not key:
        return _fail(task_id, f"{prefix}_API_KEY not set")
    model = os.environ.get(f"{prefix}_MODEL")
    if not model:
        return _fail(task_id, f"{prefix}_MODEL not set")
    host = os.environ.get(f"{prefix}_HOST", default_host)

    prompt = build_prompt(request, get_envelope("review"))
    try:
        raw = _call(host, key, model, prompt)
        response = build_response(request, raw)
    except (urllib.error.URLError, OSError) as exc:
        return _fail(task_id, f"{prefix.lower()} request error: {exc}")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return _fail(task_id, f"{prefix.lower()} response error: {exc}")

    sys.stdout.write(json.dumps(response))
    return 0
```

- [ ] **Step 4: Create the thin shims**

`method/spawners/mimo_shim.py`:

```python
#!/usr/bin/env python3
"""mimo (Xiaomi MiMo) spawner shim — OpenAI-compatible endpoint.

API baseline row (like deepseek): never substitutes a CLI agent in routing.
Reads MIMO_API_KEY / MIMO_HOST / MIMO_MODEL.
"""

from _openai_compat import run  # pyrefly: ignore[missing-import]

if __name__ == "__main__":
    raise SystemExit(run("MIMO", "https://token-plan-sgp.xiaomimimo.com/v1"))
```

`method/spawners/qwen_shim.py`:

```python
#!/usr/bin/env python3
"""qwen (DashScope) spawner shim — OpenAI-compatible endpoint.

API baseline row: never substitutes a CLI agent in routing. Reads
QWEN_API_KEY / QWEN_HOST / QWEN_MODEL.
"""

from _openai_compat import run  # pyrefly: ignore[missing-import]

if __name__ == "__main__":
    raise SystemExit(run("QWEN", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/method_spawners/test_openai_compat_shim.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add method/spawners/_openai_compat.py method/spawners/mimo_shim.py method/spawners/qwen_shim.py tests/unit/method_spawners/test_openai_compat_shim.py
git commit -m "feat(spawners): generic OpenAI-compat shim + mimo/qwen thin shims"
```

---

### Task 2: Registry — sonnet swap + mimo/qwen wiring

**Files:**
- Modify: `method/run_pipe_check.py` (`HARNESSES`, `AGENT_MODELS`, `ALLOWED_ENV`, `_preflight`)
- Test: `tests/unit/method_spawners/test_run_pipe_check.py`

**Interfaces:**
- Consumes: the shim paths from Task 1 (`method/spawners/mimo_shim.py`, `qwen_shim.py`).
- Produces: new agent_ids in `AGENTS` (`claude_code@claude-sonnet-4-6`, `anthropic_api@claude-sonnet-4-6`, `mimo@MiMo-V2.5-Pro`, `qwen@qwen3.6-plus`); no opus ids.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/method_spawners/test_run_pipe_check.py`:

```python
def test_registry_has_sonnet_and_new_api_agents_no_opus() -> None:
    from method.run_pipe_check import AGENTS

    assert "claude_code@claude-sonnet-4-6" in AGENTS
    assert "anthropic_api@claude-sonnet-4-6" in AGENTS
    assert "mimo@MiMo-V2.5-Pro" in AGENTS
    assert "qwen@qwen3.6-plus" in AGENTS
    # opus fully retired
    assert not any("claude-opus-4-8" in a for a in AGENTS)
    assert AGENTS["mimo@MiMo-V2.5-Pro"]["model_env"] == "MIMO_MODEL"
    assert AGENTS["mimo@MiMo-V2.5-Pro"]["shim"].endswith("mimo_shim.py")
    assert AGENTS["qwen@qwen3.6-plus"]["shim"].endswith("qwen_shim.py")


def test_preflight_skips_mimo_qwen_without_key(monkeypatch: "pytest.MonkeyPatch") -> None:
    from method.run_pipe_check import _preflight

    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    assert _preflight("mimo@MiMo-V2.5-Pro") == "MIMO_API_KEY not set"
    assert _preflight("qwen@qwen3.6-plus") == "QWEN_API_KEY not set"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/method_spawners/test_run_pipe_check.py -k "sonnet or mimo_qwen" -v`
Expected: FAIL — opus ids present, mimo/qwen absent; `_preflight` has no mimo/qwen branch.

- [ ] **Step 3: Add mimo/qwen to `HARNESSES`**

In `method/run_pipe_check.py`, add two entries to `HARNESSES`:

```python
    "ollama": ("method/spawners/ollama_shim.py", "OLLAMA_MODEL"),
    "mimo": ("method/spawners/mimo_shim.py", "MIMO_MODEL"),
    "qwen": ("method/spawners/qwen_shim.py", "QWEN_MODEL"),
}
```

- [ ] **Step 4: Swap opus→sonnet and add mimo/qwen in `AGENT_MODELS`**

Replace the two opus lines and add the two API rows:

```python
AGENT_MODELS: list[tuple[str, str]] = [
    ("claude_code", "claude-sonnet-4-6"),
    ("anthropic_api", "claude-sonnet-4-6"),
    ("deepseek", "deepseek-chat"),
    ("mimo", "MiMo-V2.5-Pro"),
    ("qwen", "qwen3.6-plus"),
    ("ollama", "llama3.2:1b"),
    ("ollama", "llama3.2:3b"),
    ("ollama", "qwen2.5:3b"),
    ("ollama", "qwen2.5:7b"),
    ("ollama", "qwen2.5:14b"),
]
```

- [ ] **Step 5: Allowlist the new env vars**

Append to `ALLOWED_ENV`:

```python
    "DEEPSEEK_HOST",
    "MIMO_API_KEY",
    "MIMO_HOST",
    "MIMO_MODEL",
    "QWEN_API_KEY",
    "QWEN_HOST",
    "QWEN_MODEL",
]
```

- [ ] **Step 6: Add mimo/qwen preflight branches**

In `_preflight`, after the deepseek branch and before the ollama branch:

```python
    if harness == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        return "DEEPSEEK_API_KEY not set"
    if harness == "mimo" and not os.environ.get("MIMO_API_KEY"):
        return "MIMO_API_KEY not set"
    if harness == "qwen" and not os.environ.get("QWEN_API_KEY"):
        return "QWEN_API_KEY not set"
    if harness == "ollama":
        return _preflight_ollama(spec["model"])
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/unit/method_spawners/test_run_pipe_check.py -v`
Expected: all pass (new + existing; the existing `claude_code@claude-opus-4-8` registry test from PR #200 was about the old default — update it: change its assertions to `claude_code@claude-sonnet-4-6` and the `test_unknown_task_type_exits_2_with_stderr` `--agents claude_code@claude-opus-4-8` to `claude_code@claude-sonnet-4-6`).

- [ ] **Step 8: Commit**

```bash
git add method/run_pipe_check.py tests/unit/method_spawners/test_run_pipe_check.py
git commit -m "feat(R-07): retire opus → sonnet-4-6; register mimo/qwen OpenAI-compat agents"
```

---

### Task 3: Verification + status sync

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Format, lint, type-check**

```bash
uv run ruff format .
uv run ruff check method tests
uv run pyrefly check
```
Expected: format clean; ruff "All checks passed!"; pyrefly exits 0 (the thin-shim `# pyrefly: ignore[missing-import]` keeps the sibling import from adding an error).

- [ ] **Step 2: Run the touched suites**

```bash
uv run pytest tests/unit/method_spawners -q
```
Expected: all pass.

- [ ] **Step 3: Dry-run smoke — new roster, no opus**

```bash
uv run python method/run_pipe_check.py --dry-run
```
Expected: the agent list shows `claude_code@claude-sonnet-4-6`, `anthropic_api@claude-sonnet-4-6`, `mimo@MiMo-V2.5-Pro`, `qwen@qwen3.6-plus`, deepseek + ollama; NO `claude-opus-4-8`; mimo/qwen SKIP with "MIMO_API_KEY/QWEN_API_KEY not set" if those keys are absent from the environment; exit 0.

- [ ] **Step 4: Sync TODO**

In `TODO.md` under the R-07 section, add: weekend roster (Tier-1) registered 2026-06-20 — claude→sonnet-4-6 (CLI + API baseline), mimo/qwen via generic OpenAI-compat shim, opus retired; codex (`gpt-5-codex`) added by operator; pi/opencode are Tier-2 fast-follow. Note arbiter re-keys `config/agents.toml` (claude→sonnet, +codex, −aider).

- [ ] **Step 5: Commit**

```bash
git add TODO.md
git commit -m "docs: record Tier-1 weekend roster (sonnet + mimo/qwen; opus retired)"
```

---

## Self-Review

- **Spec coverage:** generic OpenAI-compat shim + mimo/qwen thin shims (Task 1); registry sonnet-swap + mimo/qwen + ALLOWED_ENV + preflight (Task 2); opus fully retired (Task 2 AGENT_MODELS + test asserts no opus); `.env.example` (already committed on the branch). Tier-2 (pi/opencode) explicitly excluded. arbiter coordination is their side (noted in TODO). codex stays operator-added (unchanged from PR #200).
- **Placeholder scan:** none — full code for all three shims, the registry edits, and tests. The only provisional value (`OPENCODE_GLM_API_KEY`) belongs to Tier-2, not this plan.
- **Type consistency:** `run(prefix, default_host) -> int` and `build_response(request, raw) -> dict` defined in Task 1, used by the thin shims + tests; `HARNESSES` entries `(shim_path, model_env)` for mimo/qwen match the existing tuple shape; `AGENTS[id]` keys `shim`/`model_env`/`model`/`harness` consumed in the Task 2 tests as in PR #200; sonnet id `claude-sonnet-4-6` consistent across AGENT_MODELS and tests.
