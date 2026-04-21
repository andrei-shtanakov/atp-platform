# Repository Guidelines

## Project Structure & Module Organization
Core runtime code for the main package lives in `atp/` (CLI, evaluators, reporters, runner, generator, SDK, tracing, TUI). Workspace packages under `packages/` split shared concerns: `packages/atp-adapters/` contains adapter implementations, `packages/atp-core/` contains shared core modules, `packages/atp-dashboard/` contains the web dashboard plus analytics, and `packages/atp-sdk/` contains the Python SDK. Game-specific logic is split between `game-environments/` and `atp-games/` for reuse outside the main package. Docs and specs reside in `docs/` and `spec/`, while ready-to-run suites, demos, and CI templates sit in `examples/`. Automated checks live in `tests/`, grouped by `unit/`, `integration/`, `contract/`, and `e2e/`, with shared fixtures in `tests/fixtures/`. Runtime configs such as `atp.config.yaml` and `executor.config.yaml` are versioned in the repo, while task automation is driven by `task.py` and `executor.py`.

## Build, Test, and Development Commands
Use `make install` (or `uv sync --all-extras`) to install dependencies, and `make dev` to set up pre-commit hooks. `make test`, `make test-unit`, `make test-integration`, and `make test-e2e` all wrap `uv run pytest` with the right paths and coverage flags. `make lint` runs `ruff check` plus `pyrefly`, `make format` applies the formatter + autofix, and `make build` invokes `uv build` for packaging. Run `make task-list` or `make exec` when coordinating with the task automation pipeline defined in `task.py` and `executor.py`.

## Coding Style & Naming Conventions
Target Python 3.12 with full type hints on every public function. Format via `uv run ruff format .`, keep line length at 88 characters, and ensure `ruff check` stays clean before commit. Docstrings are required for user-facing APIs. Stick to `snake_case` for functions and module-level values, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Prefer descriptive file names (e.g., `test_loader_config.py`) and mirror adapter/evaluator names between `atp/` and `tests/`.

## Testing Guidelines
Pytest plus `pytest-anyio`, `pytest-cov`, and shared fixtures (`tests/conftest.py`) are the standard stack. Maintain ≥80% coverage; CI enforces this gate. Name tests after the behavior under check (`test_adapter_handles_retry`). Use `pytest -k pattern`, `pytest --lf`, or `make test-fast` (`-m "not slow"`) for focused runs. Generate reports with `make coverage` and inspect `htmlcov/index.html`. Every new adapter, evaluator, or game needs both positive and failure-path assertions, ideally with sample suites from `examples/test_suites/`.

## Commit & Pull Request Guidelines
Git history follows concise prefixes such as `feat:`, `docs:`, `fix:`, or `refactor:` (see `git log`). Work off `main`, keep branches scoped to one feature, and squash trivial fixups. Before opening a PR: run `make format lint test`, ensure new files are documented, and attach evidence (logs, HTML reports, or screenshots for dashboards). Describe why the change matters, link related issues, and call out config migrations or breaking API shifts explicitly so reviewers can plan release notes.

## Security & Configuration Tips
Configuration files often include credentials or endpoints, so prefer environment variables referenced from `atp.config.yaml` rather than committed secrets. When testing adapters that touch external services, prefer mock endpoints from `tests/fixtures/mock_tools/` to avoid leaking data. Dashboard code lives under `packages/atp-dashboard/atp/dashboard/`; keep local SQLite artifacts such as `atp_dashboard.db` out of PR diffs unless the change is intentional, and mention any required migration steps in the PR description.
