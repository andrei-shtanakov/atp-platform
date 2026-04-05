# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
4. **Evaluators** (`atp/evaluators/`) - Assess agent results (artifact, behavior, LLM-judge, code-exec, security, factuality, filesystem, style, performance, composite, git-commit, guardrails, container)
5. **Reporters** (`atp/reporters/`) - Format output (console, JSON, JUnit, HTML, game)
6. **Benchmark API** (`atp/dashboard/benchmark/`) - REST API for pull-model benchmarks with leaderboard (`/api/v1/benchmarks`, `/api/v1/runs`)
7. **Tournament API** (`atp/dashboard/tournament/`) - REST API for game-theoretic tournaments (`/api/v1/tournaments`)
8. **SDK** (`packages/atp-sdk/`) - Python SDK v2.0.0 for benchmark participants (`AsyncATPClient` + sync `ATPClient` wrapper, `BenchmarkRun` async/sync iteration, `next_batch(n)`, sync methods `submit_sync()`/`status_sync()`/`cancel_sync()`/`leaderboard_sync()`/`next_batch_sync()`/`emit_sync()`, `emit()` for event streaming, exponential-backoff retry). PyPI package name: `atp-platform-sdk`
9. **Auth** (`atp/dashboard/auth/`) - Authentication system with GitHub OAuth (OIDC), Device Flow for CLI login, JWT tokens
10. **RBAC** (`atp/dashboard/rbac/`) - Role-based access control with auto-admin for first user
11. **Dashboard UI** (`atp/dashboard/v2/routes/ui.py`) - HTMX + Pico CSS frontend served at `/ui/` (login, benchmarks, games, runs + run detail `/ui/runs/{id}`, leaderboard, suites, analytics)
12. **YAML Upload** (`POST /api/suite-definitions/upload`) - Upload YAML test suites to the server; used by `atp push`
13. **Trend Analysis** (`atp/analytics/trend.py`) - Cross-run trend analysis detecting gradual success_rate drift via OLS slope
14. **Rate Limiting** (`atp/dashboard/v2/rate_limit.py`) - Per-endpoint HTTP rate limiting via slowapi, keyed by JWT user_id or client IP, configurable via `ATP_RATE_LIMIT_*` env vars
15. **Webhooks** (`atp/dashboard/webhook.py`) - Webhook delivery for benchmark run notifications with SSRF protection, retry with backoff; `webhook_url` field on Benchmark model
16. **Event Streaming** (`POST /api/v1/runs/{id}/events`) - Append events to running benchmark runs; SDK `emit()`/`emit_sync()` methods; max 1000 events per run
17. **Container Isolation** (`atp/evaluators/container.py`) - Docker/Podman container runtime for isolated code execution with resource limits (memory, CPU)
18. **Auth State Store** (`atp/dashboard/auth/state_store.py`) - Unified transient auth state store (InMemory, protocol-based) replacing per-module session dicts; used by SSO, SAML, DeviceFlow
19. **Post-Auth Pipeline** (`atp/dashboard/auth/post_auth.py`) - Shared `complete_auth()` pipeline for user provisioning, role assignment, and token issuance
20. **Shared Result Models** (`atp/core/results.py`) - EvalCheck, EvalResult, TestResult etc. shared across evaluators, runner, reporters, and dashboard storage

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
└── atp-sdk/                 # Python SDK for benchmark platform participants (pull-model)

atp/                         # Namespace package (symlinks to packages/ + local modules)
├── cli/           # CLI entry point (atp test, validate, baseline, dashboard, etc.)
├── runner/        # Test orchestration, sandbox, progress
├── evaluators/    # Result evaluation (artifact, behavior, LLM-judge, code-exec, security, etc.)
├── reporters/     # Output formatting (console, JSON, HTML, JUnit, game)
├── baseline/      # Baseline storage, regression detection (Welch's t-test)
├── mock_tools/    # Mock tool server for deterministic testing
├── performance/   # Profiling, caching, memory tracking
├── benchmarks/    # Benchmark suites
├── generator/     # Test suite generation
├── catalog/       # Test catalog (browse, run, publish curated/community test suites)
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
- `ATP_DEBUG` - Enable debug mode (default: false)
- `ATP_HOST` - Server host address (default: "127.0.0.1")
- `ATP_PORT` - Server port (default: 8080)
- `ATP_DISABLE_AUTH` - Disable authentication (default: false, dev only!)
- `ATP_GITHUB_CLIENT_ID` - GitHub OAuth App client ID (required for GitHub login)
- `ATP_GITHUB_CLIENT_SECRET` - GitHub OAuth App client secret (required for GitHub login)
- `ATP_TOKEN_EXPIRE_MINUTES` - JWT token expiration in minutes (default: 60)
- `ATP_CORS_ORIGINS` - Comma-separated CORS origins (default: empty)
- `ATP_RATE_LIMIT_ENABLED` - Enable HTTP rate limiting (default: true)
- `ATP_RATE_LIMIT_DEFAULT` - Default rate limit (default: "60/minute")
- `ATP_RATE_LIMIT_AUTH` - Auth endpoint rate limit (default: "5/minute")
- `ATP_RATE_LIMIT_API` - Benchmark API rate limit (default: "120/minute")
- `ATP_RATE_LIMIT_UPLOAD` - Upload endpoint rate limit (default: "10/minute")
- `ATP_RATE_LIMIT_STORAGE` - Rate limit storage URI (default: "memory://", supports "redis://host:port")
- `ATP_BATCH_MAX_SIZE` - Max batch size for next-task endpoint (default: 10)
- `ATP_UPLOAD_MAX_SIZE_MB` - Max YAML upload file size in MB (default: 1)

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

- **Automatic**: push to `main` with commit message containing `[deploy]`
- **Manual**: `workflow_dispatch` from GitHub Actions UI

The workflow SSHes into the VPS, pulls latest code, rebuilds the Docker image, and restarts the container via `docker compose up -d platform`. Requires `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY` secrets in GitHub.

## Task Workflow

Tasks are defined in `spec/tasks.md` with dependencies and checklists. The `task.py` CLI manages task status, and `executor.py` can auto-execute tasks via Claude CLI.

Task status: `todo` → `in_progress` → `done` (or `blocked`)
