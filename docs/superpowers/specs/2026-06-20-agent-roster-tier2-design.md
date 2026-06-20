# Agent roster Tier-2 (pi, opencode) — design

**Date:** 2026-06-20
**Status:** approved (brainstorm)
**Branch:** `r07/agent-roster-tier2`
**Builds on:** `2026-06-20-agent-roster-expansion-design.md` (Tier-1), `2026-06-19-agent-id-convention-design.md`.
**Ecosystem refs:** `../_cowork_output/contracts/add-new-agent-runbook.md`, `../_cowork_output/decisions/2026-06-20-adr-harness-model-management.md`.

## Goal

Add the two Tier-2 CLI agents to the pipe-check roster: `pi@gpt-5` and `opencode@glm-5.1`. Both are **new CLI harnesses** requiring new spawner shims, both **non-routable** (benchmark-only).

## Ecosystem placement (per the runbook)

These are **Case B (new harness/CLI)**, but **non-routable** — arbiter does not route them (like deepseek/mimo/qwen/anthropic_api). So only the **ATP side (runbook step B2)** applies: `HARNESSES` + shim + `ALLOWED_ENV` + `AGENT_MODELS`. **No Maestro spawner / `AgentType`, no arbiter `config/agents.toml`** (the runbook: "протестированные, но не-routable → не добавлять" in arbiter). If either is ever made routable, the full Case B (Maestro spawner + `AgentType` + arbiter section, in that order) is required first — out of scope here.

## Verified facts (live probes, 2026-06-20)

- **opencode** — `opencode run --format json -m opencode/glm-5.1 "<prompt>"` works cleanly and fast. Output is JSONL events: agent text in `{"type":"text", "part":{"text": ...}}` (concatenate across events); tokens in `{"type":"step_finish", "part":{"tokens":{"total","input","output"}}}`. Auth comes from opencode's own config (the operator's `OPENCODE_GLM_API_KEY` / `opencode auth`). Model id `glm-5.1`; the provider prefix is `opencode/`.
- **pi** — `pi -p --mode json --model openai/gpt-5 "<prompt>"` works ONLY with the explicit `openai/` provider prefix (a bare `gpt-5`/`gpt-5.4` routes to `azure-openai-responses`, which is unauthed → fails). pi authenticates via its own session (no key). Output is JSONL: assistant `message_*` events; final text in the assistant message `content[].text` (type `text`); usage in the assistant message `usage:{input,output,totalTokens}` (+ `cost`).
  - **⚠ pi hang risk:** in full non-interactive mode pi is an *agentic* coding CLI and did NOT terminate on a trivial prompt (it kept running / explored). The shim MUST run pi non-agentically and with a hard timeout (see Components). pi's viability is gated on the live smoke — if it still hangs, **drop pi** and ship opencode alone.

## Components

### 1. Two new CLI shims (mirror `codex_cli_shim.py`)
Structure: read ATPRequest from stdin → `build_prompt(request, get_envelope("review"))` → `subprocess.run(argv, capture_output=True, timeout=REQUEST_TIMEOUT_S)` → parse the CLI's JSON(L) → emit the ATPResponse contract on stdout; any error / non-zero exit / empty output / timeout → a `status:"failed"` response via `_fail` (never crash). Stdlib only.

- **`method/spawners/opencode_shim.py`** — argv `["opencode","run","--format","json","-m", f"opencode/{model}", prompt]`. Parse: concat `part.text` from `type:"text"` events; tokens from the `step_finish` event's `part.tokens`. `cost_usd` = null (unknown).
- **`method/spawners/pi_shim.py`** — argv `["pi","-p","--mode","json","--no-prompt-templates","--model", f"openai/{model}", prompt]` (provider-prefix `openai/` when `model` has no `/`). Parse the assistant message `content[].text` + `usage`. Apply a hard `REQUEST_TIMEOUT_S`; on `subprocess.TimeoutExpired` → `_fail(task_id, "pi timed out")`. During implementation, probe pi for any additional "answer-only / no tools" flag and add it if it reduces the hang risk; the live smoke is the acceptance gate.
- **Provider-prefix is a launch detail of the shim**, NOT part of `agent_id` (a `/` in `agent_id` would break the dashboard route `/ui/eval-run/{suite}/{agent}` and filenames). The faithful model in `AGENT_MODELS` is slash-free (`gpt-5`, `glm-5.1`); the shim prepends the provider. (Relevant to a future ADR-ECO-002 D3 harness-descriptor, which would capture launch+provider; not formalized now.)

### 2. Registry (`method/run_pipe_check.py`)
- `HARNESSES`: `"pi": ("method/spawners/pi_shim.py", "PI_MODEL")`, `"opencode": ("method/spawners/opencode_shim.py", "OPENCODE_MODEL")`.
- `AGENT_MODELS`: `("pi", "gpt-5")`, `("opencode", "glm-5.1")`.
- `ALLOWED_ENV`: `PI_BIN`, `PI_MODEL`, `OPENCODE_BIN`, `OPENCODE_MODEL`, `OPENCODE_GLM_API_KEY`. (Implementation note: `PI_BIN`/`OPENCODE_BIN` overrides were added during build — the shared `_cli_common` reads the binary via `bin_env`, so a `_BIN` override is uniform across CLI shims, mirroring `CLAUDE_BIN`/`CODEX_BIN`.)
- `_preflight`: pi/opencode — skip with a reason if the binary is absent, honoring the `PI_BIN`/`OPENCODE_BIN` override when set, else `shutil.which("pi")` / `shutil.which("opencode")`.

### 3. Live smoke (the Tier-2 gate)
Before any paid inclusion: run each agent against a SINGLE code-review case and confirm `status:"completed"` with a parsed finding/text (not `failed`/timeout). opencode is expected to pass; pi is the risk. Only the agents that smoke green go into a paid sweep.

## Non-goals
- No Maestro / arbiter changes (non-routable).
- No declarative `HARNESSES` config refactor (ADR-ECO-002 Action Item #4 — deferred until the ≥2-custom-harness trigger; pi/opencode being non-routable do not trigger it).
- No change to grading/signal logic, the `report_benchmark` schema, or the `<harness>@<model>` convention.

## Testing
- Unit (`tests/unit/method_spawners/`): opencode/pi shim — binary-missing/non-zero → `_fail` (subprocess with a fake/missing binary); provider-prefix logic (`gpt-5` → `openai/gpt-5`, `glm-5.1` → `opencode/glm-5.1`, an already-prefixed value passes through); registry builds `pi@gpt-5` / `opencode@glm-5.1`; preflight skips when the binary is absent.
- Live smoke (manual, the gate): one code-review case per agent → completed.
- ruff + pyrefly clean; existing suites green.

## Sequencing
1. opencode + pi shims + registry + tests → PR.
2. Live smoke both. opencode → keep. pi → keep only if it smokes green within the timeout; otherwise drop the `("pi","gpt-5")` row (ship opencode alone) and note it.
3. Add to a paid sweep (fresh `--out-dir` + `--dashboard-replace`) with the rest of the roster when next running.
