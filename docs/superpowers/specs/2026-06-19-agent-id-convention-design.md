# Agent-id convention: `<harness>@<model>` — design

**Date:** 2026-06-19
**Status:** approved (brainstorm); **arbiter ACKED 2026-06-19** (implementing their side + informing Maestro) → implementation unblocked
**Branch:** `r07/agent-id-convention`
**Cross-project:** changes the `report_benchmark` `agent_id` (arbiter's routing key) — coordination notice: `../_cowork_output/contracts/2026-06-19-agent-id-convention-change.md`

## Problem

`agent_id` today conflates two axes inconsistently. Cloud/CLI agents are named by **harness only** (`claude_code`, `codex_cli`, `anthropic_api`, `deepseek`) with the model left implicit (`DEFAULT_MODEL = claude-opus-4-8` / per-shim env). Local agents fuse **harness + model** (`ollama_qwen25_14b`). Consequences:

- Running one harness with two models (e.g. `claude_code` on opus vs sonnet) collides on a single `agent_id` → the dashboard leaderboard ("latest run per agent") overwrites, and arbiter's routing key merges two distinct models.
- The model — a first-class dimension of "what agent was evaluated" — is absent from the key for every agent except ollama.
- Smell: the importer writes `SuiteExecution.model = agent_name` (the model column duplicates the harness id) because the payload carries no model.

## Decision

`agent_id = "<harness>@<model>"`, with the **faithful provider model id** on the right (option B from the brainstorm — chosen over the abbreviated ollama-style extension for unambiguous, exact model identity). `@` is the single harness/model separator; the model portion may contain `:` / `.` / `-` (e.g. ollama tags).

| harness | example `agent_id` |
|---|---|
| claude_code | `claude_code@claude-opus-4-8` |
| anthropic_api | `anthropic_api@claude-opus-4-8` |
| codex_cli | `codex_cli@gpt-5-codex` (model declared in the registry) |
| deepseek | `deepseek@deepseek-chat` |
| ollama | `ollama@qwen2.5:14b`, `ollama@llama3.2:1b`, … |

The id stays a valid **opaque key** (arbiter need not parse it); parsing on the first `@` to recover `(harness, model)` is available to any consumer that wants the model as a dimension.

## Components & changes

### 1. Harness registry (`method/run_pipe_check.py`)
Replace `SHIMS` (agent_id→shim) + `OLLAMA_MODELS` (agent_id→model) with one data-driven registry built from `(harness, model)` pairs:
- `HARNESSES: dict[str, tuple[shim_path, model_env_var]]` — e.g. `claude_code → ("…claude_code_shim.py", "CLAUDE_MODEL")`, `codex_cli → (…, "CODEX_MODEL")`, `anthropic_api → (…, "CLAUDE_MODEL")`, `deepseek → (…, "DEEPSEEK_MODEL")`, `ollama → ("…ollama_shim.py", "OLLAMA_MODEL")`.
- `AGENT_MODELS: list[tuple[harness, model]]` — the matrix to run (the existing 5 ollama models + the 4 cloud/CLI defaults; new models are added here).
- Derived `AGENTS: dict[agent_id, {shim, model_env, model}]` where `agent_id = f"{harness}@{model}"`.
- `_run_agent` injects `model` into `model_env_var` for every agent (today only `OLLAMA_MODEL` is set; generalize so each harness gets its model from the registry).
- codex declares its model(s) explicitly in `AGENT_MODELS` (symmetric with ollama). The shim's "don't hardcode a global default" note still holds — the model is per-agent registry data, not a global constant.

### 2. Filename / path safety
`agent_id` flows into filenames (`report_benchmark_<id>.json`, `case_details_<id>.jsonl`) and the drill-down URL (`/ui/eval-run/{suite}/{agent}`). `@`/`:`/`.` are filesystem- and URL-awkward. Add a shared `safe_agent_id(agent_id) -> str` (replace `@`,`:`,`.` → `_`) used **only** for file/path derivation, in the harness and the importer (sibling `case_details` lookup). The faithful `agent_id` is preserved in the payload, the dashboard `agent_name`, the DB, and the arbiter key.

### 3. Importer (`method/import_pipecheck_to_dashboard.py`)
- Parse `harness, _, model = agent_id.partition("@")`; write the real `model` into `SuiteExecution.model` (instead of `agent_name`). `agent_name` stays the full `agent_id`.
- Derive the sibling `case_details` path via `safe_agent_id`.

### 4. Data migration (existing runs)
- **Our dashboard:** the weekend pipe-check run with `--dashboard-replace` re-bases the pipe-check rows on the new ids — no separate migration required.
- **Optional one-off renamer:** a script to rewrite the 81 historical `report_benchmark_*.json` `agent_id`s to the new format (only if we want the old sweep re-imported under new ids before the weekend run).
- **arbiter:** their side — the notice asks them to update `AgentConfig` entries and migrate historical keys / stats.

### 5. arbiter coordination notice
`../_cowork_output/contracts/2026-06-19-agent-id-convention-change.md`: states the format change, the `@` semantics, the before/after id table, what arbiter must update (their opaque `AgentConfig` registry keyed by `agent_id`, plus historical `benchmark_runs`/stats keys), and that the id remains a valid opaque key (parsing optional). It is a **proposal pending their ack** — implementation does not start until they agree (their feedback could change the format, e.g. if they prefer model as a separate feature over `@`-fusion).

## Non-goals
- No separate `model` field in the `report_benchmark` payload — the model lives in `agent_id` (the chosen "fuse into id" option); consumers recover it by splitting on `@`.
- No change to grading / signal logic, evaluators, or the `report_benchmark` schema beyond the `agent_id` value format.
- No change to the abbreviated style of ollama tags' *meaning* — but ollama ids DO change form (`ollama_qwen25_14b` → `ollama@qwen2.5:14b`), which is the churn the user accepted (script or re-run).

## Testing (for the eventual plan, post-ack)
- Registry: `AGENTS` builds expected `agent_id`s from `(harness, model)` pairs; `_run_agent` sets the correct model env var per harness.
- `safe_agent_id`: `@`/`:`/`.` → `_`; round-trips into a valid filename; importer sibling-path uses it.
- Importer: `agent_id` `harness@model` → `SuiteExecution.model == model`, `agent_name == agent_id`.
- Unknown-agent / preflight paths still exit 2 with the new ids.

## Sequencing
1. **Now:** spec (this doc) + arbiter notice. Hold plan + code.
2. **On arbiter ack:** writing-plans → subagent-driven implementation → PR.
3. **Before/at the weekend run:** new ids live; `--dashboard-replace` re-bases; new models registered in `AGENT_MODELS`.
