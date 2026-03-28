# Repository Guidelines

## Project Structure & Module Organization
Core runtime code lives in `atp/` (CLI, adapters, evaluators, reporters, analytics). Game-specific logic is split between `game-environments/` and `atp-games/` for reuse outside the main package. Docs and specs reside in `docs/` and `spec/`, while ready-to-run suites and CI templates sit in `examples/`. All automated checks belong in `tests/`, grouped by `unit/`, `integration/`, `contract/`, and `e2e/`, with fixtures in `tests/fixtures/`. Declarative configs such as `atp.config.yaml`, `executor.config.yaml`, and `task.py` keep execution settings and task workflows under version control.

## Build, Test, and Development Commands
Use `make install` (or `uv sync --all-extras`) to install dependencies, and `make dev` to set up pre-commit hooks. `make test`, `make test-unit`, `make test-integration`, and `make test-e2e` all wrap `uv run pytest` with the right paths and coverage flags. `make lint` runs `ruff check` plus `pyrefly`, `make format` applies the formatter + autofix, and `make build` invokes `uv build` for packaging. Run `make task-list` or `make exec` when coordinating with the task automation pipeline defined in `task.py` and `executor.py`.

## Coding Style & Naming Conventions
Target Python 3.12 with full type hints on every public function. Format via `uv run ruff format .`, keep line length at 88 characters, and ensure `ruff check` stays clean before commit. Docstrings are required for user-facing APIs. Stick to `snake_case` for functions and module-level values, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Prefer descriptive file names (e.g., `test_loader_config.py`) and mirror adapter/evaluator names between `atp/` and `tests/`.

## Testing Guidelines
Pytest plus `pytest-anyio`, `pytest-cov`, and shared fixtures (`tests/conftest.py`) are the standard stack. Maintain ≥80% coverage; CI enforces this gate. Name tests after the behavior under check (`test_adapter_handles_retry`). Use `pytest -k pattern`, `pytest --lf`, or `make test-fast` (`-m "not slow"`) for focused runs. Generate reports with `make coverage` and inspect `htmlcov/index.html`. Every new adapter, evaluator, or game needs both positive and failure-path assertions, ideally with sample suites from `examples/test_suites/`.

## Commit & Pull Request Guidelines
Git history follows concise prefixes such as `feat:`, `docs:`, `fix:`, or `refactor:` (see `git log`). Work off `main`, keep branches scoped to one feature, and squash trivial fixups. Before opening a PR: run `make format lint test`, ensure new files are documented, and attach evidence (logs, HTML reports, or screenshots for dashboards). Describe why the change matters, link related issues, and call out config migrations or breaking API shifts explicitly so reviewers can plan release notes.

## Security & Configuration Tips
Configuration files often include credentials or endpoints—store sensitive values in environment variables consumed by `atp.config.yaml` templates rather than committing secrets. When testing adapters that touch external services, prefer mock endpoints from `tests/fixtures/mock_tools/` to avoid leaking data. For local dashboards (`atp/dashboard/`), keep SQLite files like `atp_dashboard.db` out of PR diffs unless schema changes are intentional, and mention any migration steps in the PR description.
