# Contributing to ATP

Contributions are welcome. This guide covers everything you need to get started.

## Development Setup

```bash
git clone https://github.com/yourusername/atp-platform.git
cd atp-platform

# Install all dependencies (including dev extras)
uv sync --all-extras

# Verify the setup
uv run pytest tests/ -v -m "not slow"
```

### Code Quality Tools

```bash
uv run ruff format .       # format code
uv run ruff check .        # lint
uv run ruff check . --fix  # auto-fix lint issues
pyrefly check              # type checking (run after every change)
```

## What to Contribute

### Add an Adapter

Adapters connect ATP to a new agent type (HTTP endpoint, framework, cloud service, etc.).

1. Create `atp/adapters/your_adapter.py` — implement the `BaseAdapter` interface
2. Register it in `atp/adapters/__init__.py`
3. Add adapter config docs to `docs/reference/adapters.md`
4. Write unit tests in `tests/unit/adapters/test_your_adapter.py`

Existing adapters in `atp/adapters/` are good reference: `cli.py`, `http.py`, `langgraph.py`.

### Add an Evaluator

Evaluators assess agent results against assertions in a test suite.

1. Create `atp/evaluators/your_evaluator.py` — implement `BaseEvaluator`
2. Register it in `atp/evaluators/__init__.py`
3. Write unit tests in `tests/unit/evaluators/test_your_evaluator.py`

Existing evaluators: `artifact.py`, `behavior.py`, `llm_judge.py`, `security.py`.

### Add a Game

Games live in the standalone `game-environments` package (zero ATP dependency).

1. Create `game-environments/game_envs/games/your_game.py` — implement `BaseGame`
2. Register it in `game-environments/game_envs/__init__.py` and `GameRegistry`
3. Add built-in strategies in `game-environments/game_envs/strategies/`
4. Write tests in `game-environments/tests/`
5. Optionally add a built-in ATP-games YAML suite in `atp-games/atp_games/suites/builtin/`

See `game-environments/game_envs/games/prisoners_dilemma.py` as a reference.

## Code Standards

- **Python 3.12+** with full type hints on all functions and methods
- **Line length**: 88 characters (enforced by ruff)
- **Formatter**: ruff (`uv run ruff format .`)
- **Linter**: ruff (`uv run ruff check .`)
- **Type checker**: pyrefly (`pyrefly check`) — fix all errors before submitting
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
   pyrefly check
   uv run pytest tests/ -v -m "not slow"
   ```
4. Open a pull request against `main` with a clear description of what and why
5. Address review feedback; all CI checks must pass before merge

For significant changes (new adapter type, new game, protocol changes), open an issue first to discuss the design.
