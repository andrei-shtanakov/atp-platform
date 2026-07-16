# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active Work & Roadmap

- **Current task list:** `./TODO.md` — read it at the start of every session (ecosystem section at top)
- **Ecosystem roadmap (strategic):** `../prograph-vault/authored/notes/ecosystem-roadmap.md` — R-01…R-16 across Maestro / arbiter / ATP / spec-runner
- **Latest weekly status:** `../prograph-vault/authored/notes/status/2026-04-10-status.md`
- **Sibling projects** (reference only): `../Maestro/`, `../arbiter/`, `../spec-runner/`, `../proctor/`

ATP's role in the ecosystem: task validation for Maestro (`validation_cmd` — see `docs/maestro-integration.md`) and eval-driven learning for arbiter (ATP guardrails relate to arbiter invariants as separate lifecycle phases — see `../arbiter/docs/guardrails-atp-mapping.md`). R-06a and R-13 closed 2026-04-25; deep integration (R-06b, R-07) remains blocked on Maestro R-03.

## `../_cowork_output/` is dev-only — never a code/runtime resource

`../_cowork_output/` (the polyrepo **sibling** workspace; this repo's local sweep outputs live in the separately-named, gitignored `_bench_output/` directory) is the development-time coordination area (cross-team ADRs, status notes, contract drafts, PM/dev tooling). Users and teams installing or cloning this project do NOT have it. Rules:

- Shipped/runtime code must never read, import, or resolve paths under `../_cowork_output/`.
- Canonical shippable facts live inside the owning repo: the ecosystem agents-catalog SSOT is **`method/agents-catalog.toml` in this repo** (ADR-ECO-003, canon confirmed 2026-07-03); arbiter vendors a byte-identical copy (`config/agents-catalog.toml`), and `../_cowork_output/contracts/` holds a communication mirror only.
- Vendoring a pinned copy INTO a repo is the correct pattern; referencing OUT to `../_cowork_output/` from shipped code is the antipattern.
- Only workspace-local dev tooling (e.g. the conformance check in `../_cowork_output/devtools/`) and documentation may reference it.

## Project Overview

ATP (Agent Test Platform) is a framework-agnostic platform for testing and evaluating AI agents. It provides a unified protocol and infrastructure for testing agents regardless of their implementation framework (LangGraph, CrewAI, AutoGen, custom, etc.).

**Key principle**: Agent = black box with a contract (input → output + events via ATP Protocol).

## Development Commands

```bash
# Package management (ONLY use uv, never pip)
uv sync --group dev                 # Install all deps including dev tools
uv add <package>                    # Add a new package
uv run <tool>                       # Run tool
uv add --dev <package> --upgrade-package <package>  # Upgrade dev package

# Testing
uv run pytest tests/ -v --cov=atp --cov-report=term-missing  # All tests
uv run pytest tests/unit -v                                   # Unit tests only
uv run pytest tests/ -v -m "not slow"                        # Fast tests

# Code quality
uv run ruff format .               # Format code
uv run ruff check .                # Lint check
uv run ruff check . --fix          # Auto-fix lint issues
uv run pyrefly check               # Type checking (run after every change)

# Task management (parses spec/tasks.md)
python task.py list                # List all tasks
python task.py next                # Show ready tasks
python task.py start TASK-001      # Start a task
python task.py done TASK-001       # Complete a task
python task.py show TASK-001       # Show task details
python task.py stats               # Task statistics
python task.py graph               # Dependency graph

# Task executor (automated via Claude CLI)
python executor.py run             # Execute next task
python executor.py run --task=TASK-001  # Execute specific task
python executor.py status          # Check execution status
python executor.py retry           # Retry failed task
python executor.py logs            # View execution logs

# CLI commands
uv run atp test suite.yaml --adapter=cli  # Run tests
uv run atp run suite.yaml --adapter=cli   # Alias for test
uv run atp list suite.yaml         # List tests in a suite
uv run atp validate --suite=suite.yaml    # Validate test suite
uv run atp baseline save/compare   # Baseline management
uv run atp list-agents             # List available adapters
uv run atp dashboard               # Start web dashboard
uv run atp tui                     # Start terminal UI (requires [tui] extra)
uv run atp init                    # Initialize ATP project
uv run atp quickstart              # Quick project scaffolding
uv run atp generate                # Generate test suites
uv run atp benchmark               # Run benchmarks
uv run atp budget                  # Budget management
uv run atp experiment              # Run experiments
uv run atp plugins                 # Manage plugins
uv run atp game                    # Game-theoretic evaluation
uv run atp catalog                 # Browse and run tests from the catalog
uv run atp models init             # Create a starter model catalog (~/.config/atp/)
uv run atp models list             # List models in the resolved catalog
uv run atp compare                 # Multi-model comparison
uv run atp estimate                # Cost estimation
uv run atp traces                  # Trace management
uv run atp replay                  # Replay agent traces
uv run atp trend                   # Cross-run trend analysis
uv run atp version                 # Show version info

# Suite sync commands (push/pull/sync YAML suites to/from remote server)
uv run atp push suite.yaml --server=https://atp.example.com  # Upload YAML to server
uv run atp pull --server=https://atp.example.com             # Download suites from server
uv run atp sync                    # Sync local YAML suites with remote server
```

## Architecture

### Core Components

1. **Protocol** (`atp/protocol/`) - ATP Request/Response/Event models defining the contract
2. **Adapters** (`atp/adapters/`) - Translate between ATP Protocol and agent types (HTTP, Container, CLI, LangGraph, CrewAI, AutoGen, MCP, Bedrock, Vertex, Azure OpenAI, SDK)
3. **Runner** (`atp/runner/`) - Orchestrates test execution, manages sandboxes
4. **Evaluators** (`atp/evaluators/`) - Assess agent results. Registered pipeline evaluators (`atp/evaluators/registry.py`): artifact, behavior, llm_judge, code_exec, security, factuality, performance, style, filesystem, composite, findings_match. Deterministic **checkers** are a separate registry (`atp/evaluators/checkers/`) selected via `grader: {type: programmatic, checker: <name>}` — currently `citation_grounding`, `findings_match`, `json_path`, and `receipt_chain` (verifies open-prose `receipts.jsonl` hash chains — vendored contract in `method/contract/openprose/`). (`git_commit.py`, `guardrails.py`, `container.py` exist as modules but are not registered pipeline evaluators; container is the isolation runtime — see component 20.)
5. **Reporters** (`atp/reporters/`) - Format output; registry (`atp/reporters/registry.py`): `console`, `html`, `json`, `junit`, `summary`, `report_benchmark`. (`GameReporter` exists in `atp/reporters/game_reporter.py` but is used directly by the `game` command, not registered in the reporter registry; the CLI `--output` option accepts `console`/`json`/`junit`/`summary`.)
6. **Benchmark API** (`atp/dashboard/benchmark/`) - REST API for pull-model benchmarks with leaderboard (`/api/v1/benchmarks`, `/api/v1/runs`)
7. **Tournament API** (`atp/dashboard/tournament/`) - REST API for game-theoretic tournaments (`/api/v1/tournaments`)
8. **SDK** (`packages/atp-sdk/`) - Python SDK v2.0.0 for benchmark participants (`AsyncATPClient` + sync `ATPClient` wrapper, `BenchmarkRun` async/sync iteration, `next_batch(n)`, sync methods `submit_sync()`/`status_sync()`/`cancel_sync()`/`leaderboard_sync()`/`next_batch_sync()`/`emit_sync()`, `emit()` for event streaming, exponential-backoff retry). PyPI package name: `atp-platform-sdk`
9. **Auth** (`atp/dashboard/auth/`) - Authentication system with GitHub OAuth (OIDC), Device Flow for CLI login, JWT tokens
10. **RBAC** (`atp/dashboard/rbac/`) - Role-based access control with auto-admin for first user
11. **Dashboard UI** (`atp/dashboard/v2/routes/ui.py`) - HTMX + Pico CSS frontend served at `/ui/` (home activity feed, login, register, about, agents, tokens, invites, admin, tournaments, benchmarks, games, runs + run detail `/ui/runs/{id}`, leaderboard, suites, analytics)
12. **Token Self-Service** (`atp/dashboard/tokens.py`, `atp/dashboard/v2/routes/token_api.py`) - APIToken and Invite ORM models; API endpoints `POST/GET/DELETE /api/v1/tokens`, invite-gated registration; agent-scoped tokens (prefix `atp_a_`) and user-level tokens (prefix `atp_u_`)
13. **Agent Ownership** (`atp/dashboard/v2/routes/agent_management_api.py`) - Agent CRUD with ownership checks; `POST/GET/DELETE /api/v1/agents`; per-user benchmark-agent quota enforced via `ATP_MAX_BENCHMARK_AGENTS_PER_USER` (the older `ATP_MAX_AGENTS_PER_USER` is deprecated)
14. **Invite System** (`atp/dashboard/v2/routes/invite_api.py`) - Admin-only invite code management; `POST/GET/DELETE /api/v1/invites`; controls registration when `ATP_REGISTRATION_MODE=invite`
15. **YAML Upload** (`POST /api/suite-definitions/upload`) - Upload YAML test suites to the server; used by `atp push`
16. **Trend Analysis** (`atp/analytics/trend.py`) - Cross-run trend analysis detecting gradual success_rate drift via OLS slope
17. **Rate Limiting** (`atp/dashboard/v2/rate_limit.py`) - Per-endpoint HTTP rate limiting via slowapi, keyed by JWT user_id or client IP, configurable via `ATP_RATE_LIMIT_*` env vars
18. **Webhooks** (`atp/dashboard/webhook.py`) - Webhook delivery for benchmark run notifications with SSRF protection, retry with backoff; `webhook_url` field on Benchmark model
19. **Event Streaming** (`POST /api/v1/runs/{id}/events`) - Append events to running benchmark runs; SDK `emit()`/`emit_sync()` methods; max 1000 events per run
20. **Container Isolation** (`atp/evaluators/container.py`) - Docker/Podman container runtime for isolated code execution with resource limits (memory, CPU)
21. **Auth State Store** (`atp/dashboard/auth/state_store.py`) - Unified transient auth state store (InMemory, protocol-based) replacing per-module session dicts; used by SSO, SAML, DeviceFlow
22. **Post-Auth Pipeline** (`atp/dashboard/auth/post_auth.py`) - Shared `complete_auth()` pipeline for user provisioning, role assignment, and token issuance
23. **Shared Result Models** (`atp/core/results.py`) - EvalCheck, EvalResult, TestResult etc. shared across evaluators, runner, reporters, and dashboard storage
24. **MCP Tournament Server** (`atp/dashboard/mcp/`) - FastMCP server mounted at `/mcp` exposing tournament tools (`join_tournament`, `make_move`, `get_current_state`, `list_tournaments`, `get_tournament`, `get_history`, `leave_tournament`) via SSE. Auth via `MCPAuthMiddleware` (rejects 401 without `user_id` in request state)
25. **Agent-Eval-Case Methodology** (`packages/atp-method/`, `method/`) - Plugin for structured agent-eval-case tests. `AgentEvalCase` schema (`output_contract`, `run_mode`, `grader.checker`) → `AgentEvalCaseEvaluator` (deterministic critical-check gate + non-gating rubric). Shared output envelopes (`atp_method/envelopes.py`) + deterministic checkers (`citation_grounding`, `findings_match`, `json_path`, `receipt_chain`). Cases in `method/cases/` (code-review, req-extraction), CLI-adapter spawner shims in `method/spawners/` (`claude_code`, `anthropic_api`, `codex_cli`, `deepseek`, `mimo`, `ollama`, `opencode`, `pi`, `qwen`), batch harness `method/run_pipe_check.py` emitting `report_benchmark-v1` for arbiter. The harness loads its agent roster from `method/agents-catalog.toml` — the **canonical ecosystem SSOT** (ADR-ECO-003; canon confirmed 2026-07-03): edited here first, vendored byte-identically into `arbiter/config/agents-catalog.toml`, with `../_cowork_output/contracts/` keeping a dev communication mirror; it also emits a `rank_score`/`bp_ordinal` tiebreaker in `score_components` for arbiter re-rank. Each case suite is pinned by a `SUITE.lock.toml` golden-suite lock (case ids + sha256, ADR-ECO-003a D3): the harness refuses a paid run on suite drift and regenerates the lock via `run_pipe_check.py --write-suite-lock`. Cases with `run_mode: read_only_corpus` need a `file_read` tool + mounted corpus, which the CLI-adapter pipe-check does not provide — run them via a tool-capable adapter. Failure diagnostics (post glm-5.1 empty-run incident, #221/#222): shims persist raw per-case stdout/stderr via `ATP_SHIM_RAW_DIR` (the harness wires it to `<out-dir>/raw/<agent>`), empty-output failures keep the stderr tail, shim infra-failure text is classified onto the v1 `error_class` enum's `timeout`/`crash` values (the full enum is `timeout`/`crash`/`test_failure`/`other`; empty-output runs — agent ran, produced nothing — intentionally stay `test_failure` via status normalization), and `score_components` carries `infra_error_rate` so capability and reliability stay separable; the opencode shim additionally isolates its data dir per invocation (temp `XDG_DATA_HOME` seeded with auth.json) to prevent SQLite lock contention between concurrent runs. **Cloud-`$` pricing view (ADR-ECO-003d, surface A):** a cache-aware LiteLLM pricer (`packages/atp-core/atp/cost/cloud_pricer.py`, gated behind the `[pricing]` extra) derives honest cloud-`$` from stored per-class token usage; `method/price_reports.py` reads saved `report_benchmark_*.json`, resolves the model from `agent_id` (`harness@model`), prices per case (cache-split; measured-only — estimated-fallback deferred), and emits a numeric reliability block + `cost_view.json` sidecar — derived-not-stored (a price change re-derives without a re-sweep). Token semantics are normalized at the shim edge to the `usage_contract = "cloud_pricing_usage_v1"` contract (`input_tokens` = billable uncached; `cache_*` additive) stamped top-level in the payload; reports without the stamp are flagged, never silently priced. Local models are excluded (`pricing_scope=local_excluded`, 003c D4); open-tail prices come from `method/price_overrides.toml` (provenance-required, folds into the 003b catalog contour later). **Shippable model catalog (ADR-003b SP-A):** the loader/schema/inert-template live in `atp/model_catalog/` (exposed via `atp models init`/`atp models list`); resolution is `$ATP_CATALOG` → XDG (`~/.config/atp/agents-catalog.toml`) → fail-loud, while the active `method/agents-catalog.toml` above stays the dev-SSOT and is not shipped in the wheel. The pipe-check harness itself now reads that SSOT catalog through the shared `atp.model_catalog.load_catalog` (ADR-003b SP-E): the `harnesses`/`agents` planes are typed (`HarnessEntry`/`AgentEntry`) with a referential-integrity validator, while the tested filter and sweep projection remain in `run_pipe_check.py`. The evaluator's default model now resolves through the catalog's `[defaults]` plane via `resolve_default_model()` (ADR-003b SP-C), tolerant of a missing/broken optional catalog. See `docs/adr/006-unified-capability-test-types.md` + `docs/adr/007-test-taxonomy-axes.md`.

### Data Flow

```
Test Definition (YAML) → Loader → Runner → Adapter → Agent → Response → Evaluators → Score Aggregator → Report
```

### ATP Protocol Messages

- **ATPRequest**: task description, constraints (max_steps, timeout, allowed_tools), context
- **ATPResponse**: status, artifacts, metrics (tokens, steps, cost)
- **ATPEvent**: streaming events (tool_call, llm_request, reasoning, error)

## Project Structure (Monorepo)

The project uses implicit namespace packages (PEP 420) with uv workspaces. Core modules live in `packages/` with symlinks in `atp/` for import compatibility.

```
packages/                    # Extracted packages (uv workspace members)
├── atp-core/                # Protocol, core, loader, chaos, cost, scoring, statistics, streaming
├── atp-adapters/            # All agent adapters (HTTP, CLI, Container, cloud, MCP, SDK)
├── atp-dashboard/           # Web dashboard + analytics + benchmark/tournament API
├── atp-sdk/                 # Python SDK for benchmark platform participants (pull-model)
└── atp-method/              # agent-eval-case methodology plugin (atp test method/cases/...)

atp/                         # Namespace package (symlinks to packages/ + local modules)
├── cli/           # CLI entry point (atp test, validate, baseline, dashboard, etc.)
├── runner/        # Test orchestration, sandbox, progress
├── evaluators/    # Result evaluation (artifact, behavior, LLM-judge, code-exec, security, etc.)
├── reporters/     # Output formatting (console, JSON, HTML, JUnit, summary, report_benchmark; GameReporter used directly by the game command)
├── baseline/      # Baseline storage, regression detection (Welch's t-test)
├── mock_tools/    # Mock tool server for deterministic testing
├── performance/   # Profiling, caching, memory tracking
├── benchmarks/    # Benchmark suites
├── generator/     # Test suite generation
├── test_catalog/  # Test catalog (browse, run, publish curated/community test suites)
├── plugins/       # Plugin ecosystem management
├── sdk/           # Python SDK for programmatic test execution
├── tracing/       # Agent replay and trace management
├── tui/           # Terminal user interface (optional, requires [tui] extra)
├── protocol/ → packages/atp-core/     # (symlink)
├── core/     → packages/atp-core/     # (symlink)
├── loader/   → packages/atp-core/     # (symlink)
├── cost/     → packages/atp-core/     # (symlink)
├── chaos/    → packages/atp-core/     # (symlink)
├── scoring/  → packages/atp-core/     # (symlink)
├── statistics/→ packages/atp-core/    # (symlink)
├── streaming/→ packages/atp-core/     # (symlink)
├── adapters/ → packages/atp-adapters/ # (symlink)
├── dashboard/→ packages/atp-dashboard/# (symlink)
└── analytics/→ packages/atp-dashboard/# (symlink)

game-environments/           # Standalone game theory library (8 games, 25+ strategies)
atp-games/                   # ATP plugin for game-theoretic evaluation
demo/                        # Demo agents (code writer) for functional testing
demo-game/                   # LLM game agents (GPT-4o-mini plays Prisoner's Dilemma)

spec/              # Task specifications and requirements
docs/              # Architecture documentation, guides, API reference
tests/             # Test suite (unit, integration, contract, e2e)
examples/          # Sample test suites, CI templates, example agents
```

## Environment Variables

Key environment variables for the dashboard and SDK:

- `ATP_SECRET_KEY` - JWT signing secret for auth tokens (required in production)
- `ATP_DATABASE_URL` - Database connection string (default: SQLite)
- `ATP_DATABASE_ECHO` - Echo SQL statements for debugging (default: false)
- `ATP_DEBUG` - Enable debug mode (default: false)
- `ATP_HOST` - Server host address (default: "127.0.0.1")
- `ATP_PORT` - Server port (default: 8080)
- `ATP_DISABLE_AUTH` - Disable authentication (default: false, dev only!)
- `ATP_GITHUB_CLIENT_ID` - GitHub OAuth App client ID (required for GitHub login)
- `ATP_GITHUB_CLIENT_SECRET` - GitHub OAuth App client secret (required for GitHub login)
- `ATP_TOKEN_EXPIRE_MINUTES` - JWT token expiration in minutes for regular users (default: 60)
- `ATP_ADMIN_TOKEN_EXPIRE_MINUTES` - JWT expiration in minutes for admin users (default: 720 = 12 h). Applied at token issuance when `User.is_admin=True`; non-admins keep `ATP_TOKEN_EXPIRE_MINUTES`. Set higher so admins can monitor multi-hour tournaments without re-authenticating.
- `ATP_CORS_ORIGINS` - Comma-separated CORS origins (default: empty)
- `ATP_RATE_LIMIT_ENABLED` - Enable HTTP rate limiting (default: true)
- `ATP_RATE_LIMIT_DEFAULT` - Default rate limit (default: "60/minute")
- `ATP_RATE_LIMIT_AUTH` - Auth endpoint rate limit (default: "5/minute")
- `ATP_RATE_LIMIT_API` - Benchmark API rate limit (default: "120/minute")
- `ATP_RATE_LIMIT_UPLOAD` - Upload endpoint rate limit (default: "10/minute")
- `ATP_RATE_LIMIT_STORAGE` - Rate limit storage URI (default: "memory://", supports "redis://host:port")
- `ATP_BATCH_MAX_SIZE` - Max batch size for next-task endpoint (default: 10)
- `ATP_UPLOAD_MAX_SIZE_MB` - Max YAML upload file size in MB (default: 1)
- `ATP_REGISTRATION_MODE` - Registration mode: `invite` (invite code required) or `open` (default: "invite")
- `ATP_MAX_AGENTS_PER_USER` - **Deprecated** (LABS-TSA PR-2); retained for backward compatibility so existing config keeps loading. Use the two vars below instead.
- `ATP_MAX_BENCHMARK_AGENTS_PER_USER` - Maximum benchmark agents per user (default: 10)
- `ATP_MAX_TOURNAMENT_AGENTS_PER_USER` - Maximum tournament agents per user (default: 5)
- `ATP_MAX_CONCURRENT_PRIVATE_TOURNAMENTS_PER_USER` - Maximum pending+active private tournaments per user (default: 1)
- `ATP_MAX_TOKENS_PER_AGENT` - Maximum active API tokens per agent (default: 3)
- `ATP_MAX_USER_TOKENS` - Maximum user-level API tokens (default: 5)
- `ATP_DEFAULT_TOKEN_DAYS` - Default token expiry in days (default: 30)
- `ATP_MAX_TOKEN_DAYS` - Maximum allowed token expiry in days; 0 = allow "never" (default: 365)
- `ATP_TOURNAMENT_PENDING_MAX_WAIT_S` - Max seconds to wait for tournament participants before timing out (default: 300)

## Code Style

- Python 3.12+
- Type hints required for all code
- Line length: 88 characters
- Use pydantic for data models
- Docstrings for public APIs
- PEP 8 naming (snake_case functions, PascalCase classes, UPPER_SNAKE_CASE constants)

## Testing Requirements

- Unit test coverage ≥80% for new code
- Use `anyio` for async testing, not `asyncio`
- Test pyramid: ~70% unit, ~20% integration, ~10% e2e
- Test fixtures in `tests/fixtures/`

## Deploy Workflow

Production is deployed via `.github/workflows/deploy.yml` using SSH to a VPS (Namecheap). Trigger options:

- **Automatic**: every push to `main` deploys (so each merged PR ships to prod). No `[deploy]` marker is required — that older convention was dropped because PR merge commits don't carry it.
- **Manual**: `workflow_dispatch` from GitHub Actions UI

The workflow SSHes into the VPS, pulls latest code, rebuilds the Docker image, and restarts the container via `docker compose up -d platform`. Requires `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY` secrets in GitHub.

## Task Workflow

Tasks are defined in `spec/tasks.md` with dependencies and checklists. The `task.py` CLI manages task status, and `executor.py` can auto-execute tasks via Claude CLI.

Task status: `todo` → `in_progress` → `done` (or `blocked`)

## Repo scope & boundaries

- **Этот репо:** `atp-platform` — git-корень `all_ai_orchestrators/atp-platform/`, remote `git@github.com:andrei-shtanakov/atp-platform.git`.
- **Соседи (READ-ONLY reference):** `../arbiter/`, `../deployer/`, `../dispatcher/`, `../Maestro/`, `../open-prose/`, `../proctor/`, `../prograph/`, `../prograph-vault/`, `../robin-runtime/`, `../robin-toolkit/`, `../spec-runner/`, `../spec-runner-vscode/`, `../steward/` — их код не редактировать.
- Нужна правка у соседа → **стоп**: запиши handoff в `../prograph-vault/authored/notes/`
  (кросс-проектное) или `../_cowork_output/` (черновик), не трогай его файлы.
- Кросс-репные контракты — **вендорить пиненой копией внутрь**, не ссылаться наружу.
- Полное правило (SSOT): `../prograph-vault/authored/rules/repo-boundaries.md`.

## Git workflow (у репо есть remote)

- Ветка `<type>/<slug>` → push → `gh pr create`. **Прямые коммиты в `main` запрещены.**
- После открытия PR — прочитать ревью **GitHub Copilot**: валидные замечания исправлять
  новыми коммитами в ту же ветку; невалидные — ответить с обоснованием, **не применять
  вслепую**; итерировать, пока не останется открытых замечаний.
- **Не мержить.** Мерж делает пользователь.
- После мержа пользователем: `git switch main && git pull --ff-only`, затем удалить
  влитую ветку (`git branch -d <branch>`) и `git fetch --prune`; убрать прочие влитые ветки.
- Никогда не делать force-push в общие ветки; не трогать другие репо (см. scope выше).
- Полное правило (SSOT): `../prograph-vault/authored/rules/git-workflow.md`.
