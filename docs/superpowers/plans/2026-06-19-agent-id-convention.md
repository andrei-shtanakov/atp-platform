# Agent-id convention `<harness>@<model>` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every pipe-check `agent_id` uniform and model-explicit — `<harness>@<model>` with the faithful provider model id — replacing the inconsistent harness-only (cloud) vs harness+model (ollama) naming.

**Architecture:** A data-driven harness registry in `run_pipe_check.py` builds `agent_id = f"{harness}@{model}"` from `(harness, model)` pairs and injects each model into its harness's env var. A `safe_agent_id()` helper renders ids filesystem-safe for output filenames only; the faithful id stays in the payload/dashboard/arbiter key. The importer parses the model out of `agent_id` for the dashboard `model` column.

**Tech Stack:** Python 3.12, ATP CLI adapter, pytest + anyio, uv.

**Spec:** `docs/superpowers/specs/2026-06-19-agent-id-convention-design.md` (arbiter ACKED 2026-06-19).

## Global Constraints

- `uv` only; run tools via `uv run`.
- Type hints on all code; whole-project `uv run pyrefly check` exits 0 (no NEW errors vs baseline).
- `uv run ruff format .` + `uv run ruff check` clean; line length 88.
- Async tests use `anyio`.
- Branch: `r07/agent-id-convention` (already created; never work on `main`).
- `agent_id = "<harness>@<model>"`; `@` is the single harness/model separator; the model portion is the faithful provider id and may contain `:`/`.`/`-`.
- Faithful `agent_id` is preserved in the payload, dashboard `agent_name`, DB, and arbiter key. `safe_agent_id()` is used ONLY for output filenames.
- **codex_cli is NOT in the default matrix** — it has no pinned model, and a faithful `@model` id cannot be guessed. The operator adds `("codex_cli", "<model>")` to `AGENT_MODELS` when the model is known (part of weekend model registration). Do not invent a codex model id.
- Non-goal: no separate `model` field in the payload (model lives in `agent_id`); no grading/signal-logic changes; `report_benchmark-v1` schema unchanged.

---

### Task 1: Harness registry + `<harness>@<model>` ids + safe filenames (`run_pipe_check.py`)

**Files:**
- Modify: `method/run_pipe_check.py` (registry, `_preflight`, `_run_agent`, `main`, `_main_async` filename derivation)
- Test: `tests/unit/method_spawners/test_run_pipe_check.py`

**Interfaces:**
- Produces: `HARNESSES: dict[str, tuple[str, str]]` (harness → (shim_path, model_env_var)); `AGENT_MODELS: list[tuple[str, str]]` (the default matrix); `AGENTS: dict[str, dict]` (agent_id → `{"shim", "model_env", "model", "harness"}`); `safe_agent_id(agent_id: str) -> str`.
- Removes: `SHIMS`, `OLLAMA_MODELS` (replaced by the above).

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/method_spawners/test_run_pipe_check.py` (the file already does `from method.run_pipe_check import ...`):

```python
def test_agents_registry_builds_harness_at_model_ids() -> None:
    from method.run_pipe_check import AGENTS

    assert "claude_code@claude-opus-4-8" in AGENTS
    assert "anthropic_api@claude-opus-4-8" in AGENTS
    assert "deepseek@deepseek-chat" in AGENTS
    assert "ollama@qwen2.5:14b" in AGENTS
    # codex is NOT in the default matrix (no pinned model)
    assert not any(a.startswith("codex_cli@") for a in AGENTS)
    spec = AGENTS["ollama@qwen2.5:14b"]
    assert spec["model"] == "qwen2.5:14b"
    assert spec["model_env"] == "OLLAMA_MODEL"
    assert spec["harness"] == "ollama"
    assert spec["shim"].endswith("ollama_shim.py")


def test_safe_agent_id_renders_filesystem_safe() -> None:
    from method.run_pipe_check import safe_agent_id

    assert safe_agent_id("ollama@qwen2.5:14b") == "ollama_qwen2_5_14b"
    assert safe_agent_id("claude_code@claude-opus-4-8") == "claude_code_claude-opus-4-8"
```

Update the existing `test_dashboard_replace_without_to_dashboard_exits_2` and any test passing `--agents claude_code` to use a valid new id where the agent must resolve (the dashboard-replace test hits the `main()` guard before agent validation, so it can keep `--agents claude_code`, but add an explicit unknown-id assertion):

```python
def test_legacy_bare_harness_id_is_unknown() -> None:
    # The old harness-only id no longer resolves; must exit 2.
    proc = _run(["--agents", "claude_code", "--task-type", "review", "--dry-run"])
    assert proc.returncode == 2
    assert "Unknown agent" in proc.stderr.decode()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/method_spawners/test_run_pipe_check.py -k "registry or safe_agent_id or legacy_bare" -v`
Expected: FAIL — `AGENTS`/`safe_agent_id` don't exist; `claude_code` still resolves (old SHIMS).

- [ ] **Step 3: Replace the registry**

In `method/run_pipe_check.py`, ensure `import re` is present at the top (add if missing). Replace the `_OLLAMA_SHIM` / `SHIMS` / `OLLAMA_MODELS` block with:

```python
# harness -> (shim path relative to repo root, env var that pins the model).
HARNESSES: dict[str, tuple[str, str]] = {
    "claude_code": ("method/spawners/claude_code_shim.py", "CLAUDE_MODEL"),
    "codex_cli": ("method/spawners/codex_cli_shim.py", "CODEX_MODEL"),
    "anthropic_api": ("method/spawners/anthropic_api_shim.py", "CLAUDE_MODEL"),
    "deepseek": ("method/spawners/deepseek_shim.py", "DEEPSEEK_MODEL"),
    "ollama": ("method/spawners/ollama_shim.py", "OLLAMA_MODEL"),
}

# Default (harness, model) matrix. agent_id = f"{harness}@{model}". The model is
# the faithful provider id. codex_cli is intentionally absent: it has no pinned
# model, so the operator adds ("codex_cli", "<model>") here when it is known.
AGENT_MODELS: list[tuple[str, str]] = [
    ("claude_code", "claude-opus-4-8"),
    ("anthropic_api", "claude-opus-4-8"),
    ("deepseek", "deepseek-chat"),
    ("ollama", "llama3.2:1b"),
    ("ollama", "llama3.2:3b"),
    ("ollama", "qwen2.5:3b"),
    ("ollama", "qwen2.5:7b"),
    ("ollama", "qwen2.5:14b"),
]

# agent_id -> resolved spec. The id is the routing key (faithful, with '@').
AGENTS: dict[str, dict[str, str]] = {
    f"{harness}@{model}": {
        "shim": HARNESSES[harness][0],
        "model_env": HARNESSES[harness][1],
        "model": model,
        "harness": harness,
    }
    for harness, model in AGENT_MODELS
}


def safe_agent_id(agent_id: str) -> str:
    """Filesystem-safe rendering of an agent_id for output file names.

    The faithful id (with '@', ':', '.') stays in the payload/dashboard/key;
    only file names use this form.
    """
    return re.sub(r"[@:.]", "_", agent_id)
```

Keep `ALLOWED_ENV` unchanged (it already allowlists CLAUDE_MODEL/CODEX_MODEL/DEEPSEEK_MODEL/OLLAMA_MODEL).

- [ ] **Step 4: Rewrite `_preflight` to key on the harness**

Replace the `_preflight` body's literal `agent_id == "..."` checks:

```python
def _preflight(agent_id: str) -> str | None:
    """Return a skip-reason if the agent can't run here, else None."""
    spec = AGENTS.get(agent_id)
    if spec is None:
        return f"unknown agent: {agent_id}"
    harness = spec["harness"]
    if harness == "claude_code":
        claude_bin = os.environ.get("CLAUDE_BIN", "claude")
        parts = shlex.split(claude_bin) if claude_bin else ["claude"]
        binary = parts[0] if parts else "claude"
        if shutil.which(binary) is None and not Path(binary).exists():
            return f"claude binary not found (CLAUDE_BIN={claude_bin!r})"
    if harness == "codex_cli":
        codex_bin = os.environ.get("CODEX_BIN", "codex")
        parts = shlex.split(codex_bin) if codex_bin else ["codex"]
        binary = parts[0] if parts else "codex"
        if shutil.which(binary) is None and not Path(binary).exists():
            return f"codex binary not found (CODEX_BIN={codex_bin!r})"
    if harness == "anthropic_api" and not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY not set"
    if harness == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        return "DEEPSEEK_API_KEY not set"
    if harness == "ollama":
        return _preflight_ollama(spec["model"])
    return None
```

- [ ] **Step 5: Rewrite `_run_agent` env injection + shim lookup**

Replace the `adapter_env` block and the `SHIMS[agent_id]` reference:

```python
    spec = AGENTS[agent_id]
    adapter_env: dict[str, str] = {spec["model_env"]: spec["model"]}
    adapter = create_adapter(
        "cli",
        {
            "command": sys.executable,
            "args": [str(REPO_ROOT / spec["shim"])],
            "inherit_environment": True,
            "allowed_env_vars": ALLOWED_ENV,
            "environment": adapter_env,
            "timeout_seconds": timeout_s,
        },
    )
```

- [ ] **Step 6: Safe filenames in `_main_async` + default/validation against `AGENTS`**

In `_main_async`, the per-agent output paths:

```python
        safe = safe_agent_id(agent_id)
        out_file = out_dir / f"report_benchmark_{safe}.json"
        out_file.write_text(json.dumps(payload, indent=2))
        _write_case_details(out_dir / f"case_details_{safe}.jsonl", case_results)
```

The unknown-agent check:

```python
    unknown = [a for a in agents if a not in AGENTS]
    if unknown:
        print(f"Unknown agent(s): {unknown}. Known: {list(AGENTS)}", file=sys.stderr)
        return 2
```

In `main`, the default:

```python
    p.add_argument("--agents", default=",".join(AGENTS))
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/unit/method_spawners/test_run_pipe_check.py -v`
Expected: all pass (registry/safe/legacy + the existing dry-run/grade/export/write tests).

- [ ] **Step 8: Commit**

```bash
git add method/run_pipe_check.py tests/unit/method_spawners/test_run_pipe_check.py
git commit -m "feat(R-07): <harness>@<model> agent ids via data-driven registry + safe filenames"
```

---

### Task 2: Importer parses model from `agent_id` + dashboard route survives `@`/`:`

**Files:**
- Modify: `method/import_pipecheck_to_dashboard.py` (`import_reports` — model column)
- Test: `tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py`, `tests/integration/dashboard/test_ui_eval_run_detail.py`

**Interfaces:**
- Consumes: the new `agent_id` format from Task 1 (faithful `harness@model` in the report payload's `agent_id`).
- Produces: `SuiteExecution.model` = the model parsed from `agent_id` (falls back to the full id when there is no `@`); `agent_name` stays the full `agent_id`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py`:

```python
@pytest.mark.anyio
async def test_import_sets_model_from_agent_id(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    _write_report(
        tmp_path / "report_benchmark_claude_code_claude-opus-4-8.json",
        run_id="run-x",
        agent_id="claude_code@claude-opus-4-8",
    )
    await imp.import_reports(imp.discover_reports(tmp_path), db_url=db_url)

    from atp.dashboard import init_database
    from atp.dashboard.models import SuiteExecution
    from sqlalchemy import select

    db = await init_database(url=db_url)
    async with db.session() as session:
        row = (await session.execute(select(SuiteExecution))).scalars().one()
        assert row.agent_name == "claude_code@claude-opus-4-8"
        assert row.model == "claude-opus-4-8"


@pytest.mark.anyio
async def test_import_model_falls_back_to_agent_name_without_at(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    _write_report(
        tmp_path / "report_benchmark_legacy.json", run_id="run-l", agent_id="legacy_agent"
    )
    await imp.import_reports(imp.discover_reports(tmp_path), db_url=db_url)

    from atp.dashboard import init_database
    from atp.dashboard.models import SuiteExecution
    from sqlalchemy import select

    db = await init_database(url=db_url)
    async with db.session() as session:
        row = (await session.execute(select(SuiteExecution))).scalars().one()
        assert row.model == "legacy_agent"
```

Add to `tests/integration/dashboard/test_ui_eval_run_detail.py` (verifies `@`/`:` in the URL path resolve):

```python
@pytest.mark.anyio
async def test_eval_run_detail_resolves_at_model_agent_id(fresh_app: tuple) -> None:
    from atp.dashboard.storage import ResultStorage

    app, db = fresh_app
    async with db.session() as session:
        storage = ResultStorage(session)
        ex = await storage.create_suite_execution_by_name(
            suite_name="code-review",
            agent_name="ollama@qwen2.5:14b",
            started_at=datetime.now(),
            adapter="pipe-check",
        )
        await storage.update_suite_execution(ex, status="completed")
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/eval-run/code-review/ollama@qwen2.5:14b")
    assert resp.status_code == 200
    assert "ollama@qwen2.5:14b" in resp.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py -k "model_from_agent_id or falls_back" -v`
Expected: FAIL — `model` is currently set to `agent_name` (the full id), so `row.model == "claude_code@claude-opus-4-8"`, not `"claude-opus-4-8"`.
Run: `uv run pytest tests/integration/dashboard/test_ui_eval_run_detail.py -k at_model -v`
Expected: PASS already if the route handles `@`/`:` (this test guards against a regression; if it fails the route needs a path-converter fix — note it).

- [ ] **Step 3: Parse the model in `import_reports`**

In `method/import_pipecheck_to_dashboard.py`, where the parent execution is created (`create_suite_execution_by_name(..., model=r.agent_name)`), replace the `model=` argument:

```python
            # model lives inside the agent_id (<harness>@<model>); fall back to
            # the full id for legacy ids without '@'.
            model_label = r.agent_name.partition("@")[2] or r.agent_name
            execution = await storage.create_suite_execution_by_name(
                suite_name=r.suite_name,
                agent_name=r.agent_name,
                started_at=r.started_at,
                adapter="pipe-check",
                model=model_label,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py tests/integration/dashboard/test_ui_eval_run_detail.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add method/import_pipecheck_to_dashboard.py tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py tests/integration/dashboard/test_ui_eval_run_detail.py
git commit -m "feat(import): parse model from <harness>@<model> agent_id into SuiteExecution.model"
```

---

### Task 3: Full verification + status sync

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Format, lint, type-check**

```bash
uv run ruff format .
uv run ruff check method packages/atp-method packages/atp-dashboard tests
uv run pyrefly check
```
Expected: format clean; ruff "All checks passed!"; pyrefly exits 0.

- [ ] **Step 2: Run all touched suites (separately — distinct test roots)**

```bash
uv run pytest tests/unit/method_spawners -q
uv run pytest tests/integration/dashboard/test_ui_eval_run_detail.py -q
```
Expected: all pass.

- [ ] **Step 3: Dry-run smoke shows the new default ids**

```bash
uv run python method/run_pipe_check.py --dry-run
```
Expected: the plan lists agents like `claude_code@claude-opus-4-8`, `ollama@qwen2.5:14b`, etc.; no `codex_cli@...`; exit 0.

- [ ] **Step 4: Sync TODO**

In `TODO.md`, under the R-07 dashboard/visualization section, add a line: agent_id convention `<harness>@<model>` landed 2026-06-19 (arbiter acked); model now explicit per agent + recorded in `SuiteExecution.model`; codex_cli + new models added to `AGENT_MODELS` at run-config time; weekend run re-bases on the new ids.

- [ ] **Step 5: Commit**

```bash
git add TODO.md
git commit -m "docs: record <harness>@<model> agent_id convention landing"
```

---

## Self-Review

- **Spec coverage:** `<harness>@<model>` format + faithful id (Task 1 registry); harness registry restructure + per-harness model env injection (Task 1 Steps 3-5); codex declared-in-registry / absent-from-default (Task 1 constraint + AGENT_MODELS comment); `safe_agent_id` for filenames only (Task 1 Step 3+6); importer parses model into `SuiteExecution.model`, agent_name = full id (Task 2); URL survives `@`/`:` (Task 2 integration test); data migration via weekend `--dashboard-replace` (no code — noted in spec/TODO); arbiter notice (already written, acked). Non-goals (no payload model field, no schema/grading change) — respected.
- **Placeholder scan:** none — every step has real code/commands. The only deliberately-unfilled value (codex model id) is documented as operator config, not a code placeholder.
- **Type consistency:** `AGENTS[agent_id]` is `dict[str,str]` with keys `shim`/`model_env`/`model`/`harness` — used identically in `_preflight` and `_run_agent`; `safe_agent_id(str)->str` used in both filename sites; importer `model_label = agent_name.partition("@")[2] or agent_name` matches the `<harness>@<model>` format from Task 1.
