# Agent roster expansion for the weekend sweep — design

**Date:** 2026-06-20
**Status:** approved (brainstorm)
**Branch:** `r07/agent-roster-expansion`
**Spec context:** builds on `2026-06-19-agent-id-convention-design.md` (`<harness>@<model>` ids, shipped PR #200).

## Goal

Expand the pipe-check agent roster for the weekend paid sweep (feeds the ~2026-06-25 demo): retire opus, move claude to Sonnet 4.6, and add new API + CLI agents — while keeping the `<harness>@<model>` convention and the arbiter join intact.

## Roster (final)

| agent_id | harness | shim | model env / value | key (`.env`) | tier |
|---|---|---|---|---|---|
| `claude_code@claude-sonnet-4-6` | claude_code (CLI) | existing | `CLAUDE_MODEL=claude-sonnet-4-6` | session (none) | 1 |
| `anthropic_api@claude-sonnet-4-6` | anthropic_api | existing | `CLAUDE_MODEL=claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | 1 |
| `codex_cli@gpt-5-codex` | codex_cli (CLI) | existing | `CODEX_MODEL=gpt-5-codex` | `OPENAI_API_KEY` | 1 |
| `deepseek@deepseek-chat` | deepseek (API) | existing | `DEEPSEEK_MODEL=deepseek-chat` | `DEEPSEEK_API_KEY` | 1 |
| `ollama@llama3.2:1b`, `ollama@llama3.2:3b`, `ollama@qwen2.5:3b`, `ollama@qwen2.5:7b`, `ollama@qwen2.5:14b` | ollama (local) | existing | per-id | — | 1 |
| `mimo@MiMo-V2.5-Pro` | mimo (OpenAI-compat API) | **new generic** | `MIMO_MODEL=MiMo-V2.5-Pro` | `MIMO_API_KEY` | 1 |
| `qwen@qwen3.6-plus` | qwen (OpenAI-compat API) | **new generic** | `QWEN_MODEL=qwen3.6-plus` | `QWEN_API_KEY` | 1 |
| `pi@gpt-5.4` | pi (CLI) | **new CLI** | `--model gpt-5.4` | session (none) | 2 |
| `opencode@GLM-5.1` | opencode (CLI) | **new CLI** | `--model …/GLM-5.1` | `OPENCODE_GLM_API_KEY` (provisional) | 2 |

**Opus is removed entirely** — no `claude-opus-4-8` in `AGENT_MODELS`, no paid opus run. Existing pre-convention opus data is bare-id and retires; the weekend `--dashboard-replace` re-bases the dashboard on the new `@`-ids.

## Tiering (risk/timing)

- **Tier-1** (this weekend's run): all rows above except pi/opencode. These reuse existing shims or one thin new generic API shim — config + low-risk code.
- **Tier-2** (fast-follow, live-smoked before any paid inclusion): `pi`, `opencode` — brand-new CLI subprocess integrations (binary must be installed, output parsing). They debut only after a standalone smoke, not in the demo-feeding run by default.

## Components

### 1. Registry additions (`method/run_pipe_check.py`)
- `HARNESSES` **stays the existing `(shim_path, model_env)` tuple** — no restructure. Each new provider gets its own shim path, so the OpenAI-compat prefix and default host are baked into that thin shim (component 2), not injected via the registry. Add `mimo → ("method/spawners/mimo_shim.py", "MIMO_MODEL")`, `qwen → ("method/spawners/qwen_shim.py", "QWEN_MODEL")` (Tier-1) and later `pi`/`opencode` (Tier-2). `_run_agent` keeps injecting `{model_env: model}` unchanged.
- `AGENT_MODELS`: replace the two opus entries with the two sonnet entries; add `("mimo","MiMo-V2.5-Pro")`, `("qwen","qwen3.6-plus")`; keep deepseek + ollama. codex stays operator-added (`("codex_cli","gpt-5-codex")`).
- `ALLOWED_ENV`: add `MIMO_API_KEY`, `MIMO_HOST`, `QWEN_API_KEY`, `QWEN_HOST` (Tier-1) and `OPENCODE_GLM_API_KEY` (Tier-2). `CLAUDE_MODEL` is already allowlisted (claude moves model, not env).

### 2. Generic OpenAI-compatible shim (Tier-1: mimo, qwen)
- New shared helper `method/spawners/_openai_compat.py`: `run(prefix: str, default_host: str)` reads `{prefix}_API_KEY` / `{prefix}_HOST` (default `default_host`) / `{prefix}_MODEL` (required, no default — registry-injected), POSTs `{host}/chat/completions` (Bearer; the default host already carries the version segment, e.g. mimo `…/v1`, qwen `…/compatible-mode/v1`), normalizes to the ATPResponse JSON contract — mirroring `deepseek_shim.py` (which stays as-is; DRY-migrating deepseek onto this helper is a deferred follow-up to avoid churning working Tier-1 code).
- Thin per-provider shims that set the prefix + default host:
  - `method/spawners/mimo_shim.py` → `run("MIMO", "https://token-plan-sgp.xiaomimimo.com/v1")`
  - `method/spawners/qwen_shim.py` → `run("QWEN", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")`
- Preflight: a generic "API key not set" check keyed on the harness (mirrors deepseek's) → skip with a reason, not a crash.

### 3. Tier-2 CLI shims (pi, opencode) — outline only
- `pi_shim.py`: `pi -p <prompt> --model gpt-5.4` (pi authenticates via its own session — no key). Parse stdout to the ATPResponse contract.
- `opencode_shim.py`: `opencode run --model <provider>/GLM-5.1 …`. opencode manages provider auth itself; whether it reads `OPENCODE_GLM_API_KEY` or `opencode auth login` is verified when this shim is built — the env name is provisional until then.
- Both require the binary present + a standalone live smoke before inclusion in any paid sweep.

### 4. `.env.example` (done on this branch)
Added templates: `MIMO_API_KEY`, `QWEN_API_KEY`, `OPENCODE_GLM_API_KEY` (+ commented `MIMO_HOST`/`QWEN_HOST` for region override). Operator mirrors into `.env`.

## arbiter coordination (their side, not ATP)
arbiter edits `config/agents.toml`: change the routable claude key to `claude_code@claude-sonnet-4-6` (opus retired), keep `codex_cli@gpt-5-codex`, remove `aider`. Other ATP agents (`anthropic_api@…`, `deepseek@…`, `ollama@…`, `mimo@…`, `qwen@…`, `pi@…`, `opencode@…`) land in `benchmark_runs` as opaque keys but are non-routable by design. The routable keys must match ATP-emitted ids byte-for-byte (silent-None trap). This is a one-line-per-agent TOML edit on arbiter's side.

## Non-goals
- No DRY-migration of `deepseek_shim` onto the shared helper now (deferred — don't churn working Tier-1 code before the run).
- No change to grading/signal logic, the `report_benchmark` schema, or the `<harness>@<model>` convention.
- pi/opencode are NOT in the default `AGENT_MODELS` for the weekend run unless their shims land + smoke green in time.

## Testing
- Registry: `AGENTS` builds the expected Tier-1 ids; `_run_agent` injects `{model_env, **harness env}` (incl. `OPENAI_COMPAT_PREFIX`-free design — prefix is baked into the thin shim, so the generic helper reads `{prefix}_*`).
- Generic shim: unit-test `_openai_compat.run` parsing/normalization with a stubbed HTTP response (no live call); preflight skip when the key is unset.
- Dry-run smoke: `run_pipe_check --dry-run` lists the new Tier-1 ids; mimo/qwen skip cleanly without keys.
- No new pyrefly/ruff errors; existing suites green.

## Sequencing
1. **Now:** Tier-1 — registry + generic shim + `.env.example` (done) + tests → PR.
2. **Operator (parallel):** fill `.env` keys; arbiter edits `config/agents.toml`.
3. **Weekend run:** Tier-1 sweep with `--to-dashboard --dashboard-replace`.
4. **Fast-follow:** Tier-2 pi/opencode shims + standalone smoke; add to `AGENT_MODELS` once green.
