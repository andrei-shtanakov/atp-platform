# AGENTS.md

## Package Management

ONLY use `uv`, NEVER `pip`. Installation: `uv add <pkg>`. Running tools: `uv run <tool>`. Forbidden: `uv pip install`, `@latest` syntax.

## Quick Reference

```bash
# Setup
uv sync --all-extras --group dev    # Install everything (workspace + dev deps)

# Code quality (run in this order before committing)
uv run ruff format .                # Format
uv run ruff check .                 # Lint (auto-fix: --fix)
uv run pyrefly check                # Type check

# Testing
uv run pytest tests/ -v -m "not slow" --cov=atp --cov-report=term-missing --cov-fail-under=80  # CI default
uv run pytest tests/unit -v         # Unit only
uv run pytest tests/unit/dashboard/tournament -v                        # Tournament unit only
uv run pytest tests/integration/dashboard/tournament -v                 # Tournament integration
uv run pytest tests/ -v -m slow     # Slow tests (needs real uvicorn+FastMCP SSE or container image)
uv run pytest tests/ -k <pattern>   # Filter by name

# Makefile shortcuts
make test-unit / make test-fast / make lint / make format / make check

# Dashboard database migrations
uv run alembic upgrade head          # Apply migrations
```

## Pre-commit Hooks

Run `uv run pre-commit install` to activate. Hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-json, check-toml, check-merge-conflict, check-added-large-files, debug-statements, ruff (with --fix), ruff-format, pyrefly. CI enforces the same checks.

## Monorepo Layout

uv workspace (`[tool.uv.workspace]` in root `pyproject.toml`):

```
packages/atp-core/       → atp-core (protocol, core, loader, chaos, cost, scoring, statistics, streaming)
packages/atp-adapters/   → atp-adapters (HTTP, CLI, Container, LangGraph, CrewAI, AutoGen, MCP, cloud, SDK)
packages/atp-dashboard/ → atp-dashboard (FastAPI web UI, analytics, benchmark/tournament API, auth, DB)
packages/atp-sdk/        → atp-platform-sdk (Python SDK v2.0.0 for benchmark participants)
packages/atp-method/     → atp-method (agent-eval-case methodology plugin)
game-environments/       → game-environments (standalone game theory library, 8 games, 25+ strategies)
atp-games/               → atp-games (ATP plugin for game-theoretic evaluation)
atp/                     → Main package (CLI, runner, evaluators, reporters, baseline, catalog, etc.)
```

**Symlink gotcha**: Directories in `atp/` that are symlinks (adapters, analytics, chaos, core, cost, dashboard, loader, protocol, scoring, statistics, streaming) point to `packages/`. When editing files under `atp/adapters/`, the real files live in `packages/atp-adapters/atp/adapters/`. Same for all symlinked packages.

**Entry points** define the plugin system — evaluators, reporters, adapters, and method plugin are registered in `pyproject.toml` `[project.entry-points]` sections (root for evaluators/reporters, `atp-adapters` for adapters, `atp-method` for plugins).

## Key Architecture

- **Protocol**: `atp/protocol/` → `packages/atp-core/` — ATPRequest/ATPResponse/ATPEvent models
- **Adapters**: `atp/adapters/` → `packages/atp-adapters/` — Translate between ATP Protocol and agent types
- **Runner**: `atp/runner/` — Test orchestration
- **Evaluators**: `atp/evaluators/` — Pipeline evaluators via entry points; deterministic **checkers** (`atp/evaluators/checkers/`) are a separate registry selected via `grader: {type: programmatic, checker: <name>}`
- **Reporters**: `atp/reporters/` — Console, JSON, JUnit, HTML, game
- **CLI**: `atp/cli/main.py` → `atp` command
- **Dashboard**: `atp/dashboard/` → `packages/atp-dashboard/` — FastAPI + HTMX + Pico CSS at `/ui/`; DB migrations via Alembic in `migrations/`
- **Auth**: GitHub OAuth (OIDC), Device Flow for CLI, JWT tokens, RBAC
- **SDK**: `packages/atp-sdk/` — `AsyncATPClient` + sync `ATPClient`, PyPI package `atp-platform-sdk`
- **MCP Tournament Server**: `atp/dashboard/mcp/` — FastMCP SSE server at `/mcp`
- **Method Plugin**: `packages/atp-method/` — Agent-eval-case methodology, registered via `atp.plugins` entry point
- **Data flow**: YAML → Loader → Runner → Adapter → Agent → Response → Evaluators → Report

## Code Style

- Python 3.12+, type hints required
- Line length: 88 chars (ruff enforces)
- pydantic for data models; `anyio` for async (never `asyncio` directly)
- Naming: `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- Docstrings required for public APIs
- `pyrefly` for type checking (not mypy) — config in `[tool.pyrefly]` in root `pyproject.toml`

## Testing

- Framework: pytest + pytest-anyio + pytest-cov + pytest-mock
- Fixtures: `tests/conftest.py` (paths), `tests/fixtures/` (data, mock tools, YAML suites, test site)
- `@pytest.mark.slow` — excluded from CI; these need real uvicorn+FastMCP SSE or container images, flaky in shared runners
- Coverage gate: 80% minimum on `atp` package; separate 80% gate for tournament package in CI
- Test dirs: `tests/unit/`, `tests/integration/`, `tests/contract/`, `tests/e2e/`, `tests/ci/`
- See `TESTING.md` for detailed fixture docs and test-writing patterns

## CI Pipeline

`.github/workflows/ci.yml` runs on push to `main`/`master`/`develop`/`task/*` and PRs to those branches:

1. **test job**: `uv sync --all-extras --dev` → `ruff format --check` → `ruff check` → `pyrefly check` → `pytest -m "not slow"` with 80% coverage gate
2. **tournament-unit → tournament-integration → tournament-e2e**: separate pipeline for `atp.dashboard.tournament` with its own 80% coverage gate; e2e only runs on push (not PRs) unless labeled `tournament-e2e`
3. **lint job**: format + lint checks (lighter install, no test deps)
4. **docker-build job**: builds Docker image, starts container, health-checks `/api/health` endpoint

## Deploy

Auto-deploys on every push to `main` via `.github/workflows/deploy.yml` using SSH to VPS. No `[deploy]` marker needed — that convention was dropped because PR merge commits don't carry it. Manual trigger via `workflow_dispatch`. Docker compose config is in `deploy/docker-compose.yml`. Requires `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY` secrets.

## Environment Variables

Dashboard requires `.env` (copy from `.env.example`). Key vars:
- `ATP_SECRET_KEY` — JWT signing (required in production)
- `ATP_DATABASE_URL` — DB connection (default: SQLite at `~/.atp/dashboard.db`)
- `ATP_DISABLE_AUTH` — Disable auth (dev only)
- `ATP_GITHUB_CLIENT_ID/SECRET` — GitHub OAuth
- `ATP_REGISTRATION_MODE` — `invite` (default) or `open`
- LLM keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) needed for LLM-judge evaluator and test generation

Full list in `.env.example` and `CLAUDE.md`.

## Task Management

Tasks in `spec/tasks.md`. CLI: `uv run python task.py list/next/start/done/show` or `make task-list/task-next/task-start ID=TASK-001`.
