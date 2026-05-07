# Contributing to ATP

Contributions are welcome. This guide covers everything you need to get started.

## Development Setup

```bash
git clone https://github.com/andrei-shtanakov/atp-platform.git
cd atp-platform

# Install all dependencies (including dev tools)
uv sync --group dev

# Verify the setup
uv run pytest tests/ -v -m "not slow"
```

### Code Quality Tools

```bash
uv run ruff format .       # format code
uv run ruff check .        # lint
uv run ruff check . --fix  # auto-fix lint issues
uv run pyrefly check       # type checking (run after every change)
```

## What to Contribute

### Add an Adapter

Adapters connect ATP to a new agent type (HTTP endpoint, framework, cloud service, etc.).

1. Create `packages/atp-adapters/atp/adapters/your_adapter.py` — implement the `AgentAdapter` interface
2. Register it in `packages/atp-adapters/atp/adapters/__init__.py`
3. Add adapter config docs to `docs/reference/adapters.md`
4. Write unit tests in `tests/unit/adapters/test_your_adapter.py`

Existing adapters in `packages/atp-adapters/atp/adapters/` are good reference: `cli.py`, `http.py`, `langgraph.py`.

### Add an Evaluator

Evaluators assess agent results against assertions in a test suite.

1. Create `atp/evaluators/your_evaluator.py` — implement the `Evaluator` base class
2. Register it in `atp/evaluators/registry.py`
3. Write unit tests in `tests/unit/evaluators/test_your_evaluator.py`

Existing evaluators: `artifact.py`, `behavior.py`, `llm_judge.py`, `security/`, `factuality.py`, `style.py`, `performance.py`.

### Add a Game

Games live in the standalone `game-environments` package (zero ATP dependency).

1. Create `game-environments/game_envs/games/your_game.py` — implement `BaseGame`
2. Register it in `game-environments/game_envs/__init__.py` and `GameRegistry`
3. Add built-in strategies in `game-environments/game_envs/strategies/`
4. Write tests in `game-environments/tests/`
5. Optionally add a built-in ATP-games YAML suite in `atp-games/atp_games/suites/builtin/`

See `game-environments/game_envs/games/prisoners_dilemma.py` as a reference.

### Add Tests to the Catalog

The test catalog (`atp catalog`) provides curated and community test suites.

1. Create a YAML file with a `catalog:` metadata section and standard `test_suite:` format
2. Include: `category`, `slug`, `name`, `description`, `author`, `difficulty`, `tags`
3. Publish locally: `atp catalog publish your-suite.yaml`
4. For builtin tests: add YAML to `atp/catalog/builtin/<category>/`

See `atp/catalog/builtin/coding/file-operations.yaml` as a reference.

## Code Standards

- **Python 3.12+** with full type hints on all functions and methods
- **Line length**: 88 characters (enforced by ruff)
- **Formatter**: ruff (`uv run ruff format .`)
- **Linter**: ruff (`uv run ruff check .`)
- **Type checker**: pyrefly (`uv run pyrefly check`) — fix all errors before submitting
- **Test coverage**: 80% minimum for new code
- **Async testing**: use `anyio`, not `asyncio` directly
- **Data models**: use Pydantic
- **Naming**: `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- **Docstrings**: required on all public APIs

## Pull Request Process

1. Fork the repository and create a feature branch off `main`
2. Write tests for new functionality (bug fixes need regression tests)
3. Run the full quality check before pushing:
   ```bash
   uv run ruff format .
   uv run ruff check .
   uv run pyrefly check
   uv run pytest tests/ -v -m "not slow"
   ```
4. Open a pull request against `main` with a clear description of what and why
5. Address review feedback; all CI checks must pass before merge

For significant changes (new adapter type, new game, protocol changes), open an issue first to discuss the design.

## Releases & Changelog

### Changelog rule

Every PR labelled `breaking` or `feat` must add a bullet under
`## [Unreleased]` in `CHANGELOG.md`. Bug fixes are optional but encouraged.

The `changelog` GitHub Actions workflow (`.github/workflows/changelog.yml`)
enforces this on label change. If you remove the `feat`/`breaking` label
the gate goes away — this is intentional. The goal is a reminder, not a
hard wall. If a stricter check is ever needed, the next layer is parsing
PR titles for conventional-commit prefixes; that's a follow-up, not in
this workflow.

### Routing — which CHANGELOG?

- Changes that affect the platform / CLI / dashboard / engines (El Farol,
  PD, etc.) → root `CHANGELOG.md`.
- Changes that affect a specific distributable package
  (`atp-platform-sdk`, `game-environments`, `atp-adapters`, `atp-core`,
  `atp-dashboard`) → `packages/<name>/CHANGELOG.md`.
- When in doubt, write the bullet in both — the root entry can just point
  to the package one.

Sub-packages are versioned independently. The root `pyproject.toml`'s
`version` describes the platform / CLI; each `packages/atp-*/pyproject.toml`
tracks its own release cadence.

### Cutting a release

1. Open a release PR. Replace `## [Unreleased]` with
   `## [X.Y.Z] - <today's date>` and add a fresh empty `## [Unreleased]`
   above it. Update linkbacks at the bottom of the file.
2. Bump `version` in the relevant `pyproject.toml`. Root for platform /
   CLI releases; sub-package `pyproject.toml` for SDK / package releases.
3. Merge the release PR.
4. Tag the merge commit:
   ```bash
   git tag -a vX.Y.Z <merge-commit-sha> -m "Release X.Y.Z"
   git push origin vX.Y.Z
   ```
5. Open the GitHub Release UI for the new tag. Title: `vX.Y.Z`. Body:
   paste the contents of the `## [X.Y.Z]` section from `CHANGELOG.md`
   directly. Do **not** rely on GitHub's auto-generated release notes —
   they group by PR title and label and ignore CHANGELOG entirely, so
   the result drifts from the actual changelog.
6. **PyPI publish is not automatic.** Tagging produces only the GitHub
   Release. Publishing to PyPI requires a separate workflow run (or
   manual `uv build` + `uv publish` if the workflow is missing). After
   publishing, smoke-test the release with `pip install <package>==X.Y.Z`
   from a clean venv.
