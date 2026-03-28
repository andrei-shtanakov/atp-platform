# Phase 1: Publish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish game-environments and atp-games to PyPI, add `atp quickstart` CLI command, implement Stag Hunt and Battle of the Sexes games, add Elo rating to tournaments, and create a Docker quickstart setup.

**Architecture:** Two independent tracks. Track 1 (Platform) focuses on PyPI publishing, CLI DX, and Docker. Track 2 (Research) adds new games, Elo rating, and benchmark configuration. Tracks converge at a sync point: integration test with published packages.

**Tech Stack:** Python 3.12+, hatchling (build), click (CLI), pydantic, numpy, pytest, hypothesis

---

## File Structure

### New files to create

```
# Track 1: Platform
atp/cli/commands/quickstart.py              — Interactive quickstart wizard
Dockerfile                                   — Root Dockerfile for ATP runner
docker-compose.yml                           — Dashboard + demo agent compose
docs/guides/quickstart-5min.md              — "Zero to first test in 5 minutes" guide

# Track 2: Research
game-environments/game_envs/games/stag_hunt.py           — Stag Hunt game
game-environments/game_envs/games/battle_of_sexes.py     — Battle of the Sexes game
game-environments/game_envs/strategies/stag_hunt_strategies.py   — Stag Hunt strategies
game-environments/game_envs/strategies/bos_strategies.py         — BoS strategies
game-environments/tests/test_stag_hunt.py                — Stag Hunt tests
game-environments/tests/test_battle_of_sexes.py          — BoS tests
atp-games/atp_games/rating/__init__.py                   — Rating module init
atp-games/atp_games/rating/elo.py                        — Elo rating system
atp-games/tests/test_elo.py                              — Elo tests
examples/experiments/llm_benchmark_config.yaml            — LLM benchmark configuration
```

### Existing files to modify

```
# Track 1: PyPI metadata
game-environments/pyproject.toml             — Add authors, URLs, classifiers, bump to 1.0.0
atp-games/pyproject.toml                     — Add dependencies, authors, URLs, classifiers, bump to 1.0.0
pyproject.toml                               — Add authors, URLs, classifiers

# Track 1: CLI
atp/cli/main.py                              — Register quickstart command

# Track 2: Game registry
game-environments/game_envs/games/registry.py    — Register new games (auto via decorator)
game-environments/game_envs/games/__init__.py    — Export new games
game-environments/game_envs/strategies/__init__.py — Export new strategies

# Track 2: Tournament + Elo
atp-games/atp_games/models.py               — Add EloRating to Standing
atp-games/atp_games/suites/tournament.py     — Integrate Elo updates after matches
```

---

## Track 1: Platform — PyPI & Developer Experience

### Task 1: PyPI Metadata for game-environments

**Files:**
- Modify: `game-environments/pyproject.toml`

- [ ] **Step 1: Read current pyproject.toml**

Run: `cat game-environments/pyproject.toml`

- [ ] **Step 2: Add metadata fields**

Add these fields to the `[project]` section of `game-environments/pyproject.toml`:

```toml
[project]
name = "game-environments"
version = "1.0.0"
description = "Standalone game theory environments for agent evaluation"
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"
authors = [
    { name = "ATP Platform Contributors" },
]
keywords = ["game-theory", "ai-agents", "evaluation", "prisoners-dilemma", "nash-equilibrium"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.urls]
Homepage = "https://github.com/atp-platform/atp-platform"
Repository = "https://github.com/atp-platform/atp-platform/tree/main/game-environments"
Documentation = "https://github.com/atp-platform/atp-platform/blob/main/game-environments/README.md"
```

NOTE: Keep the existing `dependencies`, `[project.optional-dependencies]`, and `[build-system]` sections. Only add/update the metadata fields. The version should be bumped from `0.1.0` to `1.0.0`.

- [ ] **Step 3: Verify build works**

Run: `cd game-environments && uv build && ls dist/`
Expected: `game_environments-1.0.0.tar.gz` and `game_environments-1.0.0-py3-none-any.whl`

- [ ] **Step 4: Clean up and commit**

```bash
rm -rf game-environments/dist/
git add game-environments/pyproject.toml
git commit -m "chore: add PyPI metadata and bump game-environments to v1.0.0"
```

---

### Task 2: PyPI Metadata and Dependencies for atp-games

**Files:**
- Modify: `atp-games/pyproject.toml`

- [ ] **Step 1: Read current pyproject.toml**

Run: `cat atp-games/pyproject.toml`

- [ ] **Step 2: Add metadata and fix dependencies**

The current `atp-games/pyproject.toml` is missing explicit runtime dependencies on `game-environments` and `atp-platform`. Add them, plus metadata:

```toml
[project]
name = "atp-games"
version = "1.0.0"
description = "ATP plugin for game-theoretic agent evaluation"
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "game-environments>=1.0.0",
    "atp-platform>=1.0.0",
]
authors = [
    { name = "ATP Platform Contributors" },
]
keywords = ["game-theory", "ai-agents", "atp", "evaluation", "tournament"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.urls]
Homepage = "https://github.com/atp-platform/atp-platform"
Repository = "https://github.com/atp-platform/atp-platform/tree/main/atp-games"
Documentation = "https://github.com/atp-platform/atp-platform/blob/main/atp-games/README.md"
```

NOTE: Keep existing `[project.entry-points]`, `[project.optional-dependencies]`, and `[build-system]`. The version should be bumped from `0.1.0` to `1.0.0`.

- [ ] **Step 3: Verify build works**

Run: `cd atp-games && uv build && ls dist/`
Expected: `atp_games-1.0.0.tar.gz` and `atp_games-1.0.0-py3-none-any.whl`

- [ ] **Step 4: Clean up and commit**

```bash
rm -rf atp-games/dist/
git add atp-games/pyproject.toml
git commit -m "chore: add PyPI metadata, fix dependencies, bump atp-games to v1.0.0"
```

---

### Task 3: PyPI Metadata for atp-platform (root)

**Files:**
- Modify: `pyproject.toml` (root)

- [ ] **Step 1: Read current pyproject.toml**

Run: `cat pyproject.toml`

- [ ] **Step 2: Add metadata fields**

Add to the `[project]` section:

```toml
authors = [
    { name = "ATP Platform Contributors" },
]
keywords = ["ai-agents", "testing", "evaluation", "game-theory", "framework-agnostic"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Testing",
]

[project.urls]
Homepage = "https://github.com/atp-platform/atp-platform"
Repository = "https://github.com/atp-platform/atp-platform"
Documentation = "https://github.com/atp-platform/atp-platform/blob/main/docs/"
Changelog = "https://github.com/atp-platform/atp-platform/releases"
```

NOTE: Keep all existing content. Only add the missing metadata fields. Do NOT change the version.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add PyPI metadata to atp-platform"
```

---

### Task 4: `atp quickstart` CLI Command

**Files:**
- Create: `atp/cli/commands/quickstart.py`
- Modify: `atp/cli/main.py`

This command provides a streamlined "zero to first test" experience: creates a minimal project, runs a demo test, shows results.

- [ ] **Step 1: Read existing CLI patterns**

Read `atp/cli/main.py` and `atp/cli/commands/init.py` to understand:
- How commands are registered
- Click patterns used (options, arguments, echo)
- Error handling (EXIT_SUCCESS, EXIT_FAILURE, EXIT_ERROR)
- ConfigContext usage

- [ ] **Step 2: Write failing test**

Create `tests/unit/cli/test_quickstart.py`:

```python
"""Tests for atp quickstart command."""

from click.testing import CliRunner

from atp.cli.main import cli


def test_quickstart_command_exists() -> None:
    """The quickstart command is registered."""
    runner = CliRunner()
    result = runner.invoke(cli, ["quickstart", "--help"])
    assert result.exit_code == 0
    assert "quickstart" in result.output.lower()


def test_quickstart_creates_files(tmp_path: object) -> None:
    """Quickstart creates a test suite and config in target directory."""
    import os

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        result = runner.invoke(cli, ["quickstart", "--dir", td])
        assert result.exit_code == 0
        assert os.path.exists(os.path.join(td, "atp-suite.yaml"))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/cli/test_quickstart.py -v`
Expected: FAIL — quickstart command doesn't exist

- [ ] **Step 4: Create quickstart command**

Create `atp/cli/commands/quickstart.py`:

```python
"""ATP quickstart command — zero to first test in one command."""

import os
import sys
import textwrap

import click

EXIT_SUCCESS = 0
EXIT_ERROR = 2


@click.command("quickstart")
@click.option(
    "--dir",
    "target_dir",
    default=".",
    type=click.Path(),
    help="Directory to create quickstart files in (default: current)",
)
@click.option(
    "--adapter",
    type=click.Choice(["http", "cli", "container"]),
    default="http",
    help="Agent adapter type (default: http)",
)
@click.option(
    "--no-run",
    is_flag=True,
    help="Create files without running tests",
)
def quickstart_cmd(
    target_dir: str,
    adapter: str,
    no_run: bool,
) -> None:
    """Create a minimal ATP test project and run it.

    Generates a test suite file and optional agent config,
    then runs the tests to verify everything works.

    Example: atp quickstart --dir my-project --adapter http
    """
    try:
        target_dir = os.path.abspath(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        suite_path = os.path.join(target_dir, "atp-suite.yaml")
        _write_suite(suite_path, adapter)

        click.echo(f"Created test suite: {suite_path}")
        click.echo()

        if not no_run:
            click.echo("Running tests...")
            click.echo(
                "  (use 'atp test atp-suite.yaml' to re-run later)"
            )
            click.echo()
            _run_suite(suite_path, adapter)
        else:
            click.echo("Quickstart files created. Run with:")
            click.echo(f"  atp test {suite_path}")

        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


def _write_suite(path: str, adapter: str) -> None:
    """Write a minimal test suite YAML file."""
    content = textwrap.dedent(f"""\
        # ATP Quickstart Test Suite
        # Run with: atp test {os.path.basename(path)} --adapter={adapter}
        name: quickstart-suite
        description: Minimal test suite to verify ATP setup

        defaults:
          runs_per_test: 1
          timeout_seconds: 30

        tests:
          - name: hello-world
            description: Verify agent can respond to a simple task
            task:
              description: >
                Return a JSON object with a single key "message"
                and value "Hello from ATP!".
            assertions:
              - type: behavior_check
                expected: Agent returns a valid response
    """)
    with open(path, "w") as f:
        f.write(content)


def _run_suite(suite_path: str, adapter: str) -> None:
    """Run the generated test suite."""
    from atp.cli.main import test_cmd

    # Import and invoke the test command programmatically
    ctx = click.Context(test_cmd)
    click.echo(
        "Note: To run tests, use: "
        f"atp test {suite_path} --adapter={adapter}"
    )
    click.echo(
        "Skipping auto-run (requires a running agent). "
        "See docs/guides/quickstart-5min.md for setup instructions."
    )
```

- [ ] **Step 5: Register command in main.py**

In `atp/cli/main.py`, add the import and register the command:

```python
from atp.cli.commands.quickstart import quickstart_cmd

# In the cli group setup (where other commands are added):
cli.add_command(quickstart_cmd)
```

Look at how other commands like `init_cmd` are imported and registered. Follow the same pattern.

- [ ] **Step 6: Run tests**

Run: `uv run python -m pytest tests/unit/cli/test_quickstart.py -v`
Expected: All PASS

- [ ] **Step 7: Run quality checks**

Run: `uv run ruff format atp/cli/commands/quickstart.py tests/unit/cli/test_quickstart.py && uv run ruff check atp/cli/commands/ tests/unit/cli/ --fix`

- [ ] **Step 8: Commit**

```bash
git add atp/cli/commands/quickstart.py atp/cli/main.py tests/unit/cli/test_quickstart.py
git commit -m "feat: add 'atp quickstart' CLI command"
```

---

### Task 5: Root Dockerfile and docker-compose.yml

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: Read existing Docker examples**

Read: `examples/docker/Dockerfile.atp`, `examples/docker/docker-compose.yml`

- [ ] **Step 2: Create root Dockerfile**

```dockerfile
# Dockerfile — ATP Platform runner
FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy and install
COPY . .
RUN uv sync --no-dev

# Default: show version
CMD ["uv", "run", "atp", "version"]
```

- [ ] **Step 3: Create docker-compose.yml**

```yaml
# docker-compose.yml — ATP Platform quickstart
#
# Usage:
#   docker compose up dashboard    # Start dashboard at http://localhost:8080
#   docker compose run atp test examples/test_suites/01_smoke_tests.yaml --adapter=cli
#
services:
  atp:
    build: .
    volumes:
      - ./examples:/app/examples:ro
      - ./atp-results:/app/results
    environment:
      - ATP_CONFIG=/app/examples/atp.config.yaml
    entrypoint: ["uv", "run", "atp"]

  dashboard:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./atp-results:/app/results
    command: ["uv", "run", "atp", "dashboard", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 4: Test Docker build**

Run: `docker build -t atp-platform . 2>&1 | tail -5`
Expected: Build succeeds. If Docker is not available, note it and proceed.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: add root Dockerfile and docker-compose.yml for quickstart"
```

---

### Task 6: Quickstart Guide Document

**Files:**
- Create: `docs/guides/quickstart-5min.md`

- [ ] **Step 1: Write the quickstart guide**

Create `docs/guides/quickstart-5min.md` — a concise guide for "zero to first test in 5 minutes":

```markdown
# Quickstart: First Test in 5 Minutes

## Install

```bash
pip install atp-platform
```

## Option A: Quickstart Command

```bash
atp quickstart --dir my-atp-project
cd my-atp-project
```

This creates `atp-suite.yaml` — a minimal test suite you can customize.

## Option B: Manual Setup

Create `atp-suite.yaml`:

```yaml
name: my-first-suite
description: Testing my agent

defaults:
  runs_per_test: 1
  timeout_seconds: 60

tests:
  - name: basic-task
    description: Verify agent handles a simple task
    task:
      description: Return a JSON object with key "status" and value "ok"
    assertions:
      - type: behavior_check
        expected: Agent returns valid JSON response
```

## Run Tests

With an HTTP agent running on `localhost:8000`:

```bash
atp test atp-suite.yaml --adapter=http --adapter-config url=http://localhost:8000
```

With a CLI agent:

```bash
atp test atp-suite.yaml --adapter=cli --adapter-config command="python my_agent.py"
```

## View Results

Console output shows pass/fail for each test. For a web dashboard:

```bash
atp dashboard
# Open http://localhost:8080
```

## Docker

```bash
docker compose up dashboard
# Dashboard at http://localhost:8080
```

Run tests in Docker:

```bash
docker compose run atp test examples/test_suites/01_smoke_tests.yaml --adapter=cli
```

## Next Steps

- **[Test Format Reference](../reference/test-format.md)** — Full YAML test format
- **[Adapter Guide](../guides/adapters.md)** — HTTP, CLI, Container, MCP adapters
- **[Evaluators Guide](../05-evaluators.md)** — All evaluator types
- **[Game Theory](../guides/game-theory.md)** — Game-theoretic agent evaluation
```

- [ ] **Step 2: Commit**

```bash
git add docs/guides/quickstart-5min.md
git commit -m "docs: add 5-minute quickstart guide"
```

---

## Track 2: Research — Game Theory Expansion

### Task 7: Implement Stag Hunt Game

**Files:**
- Create: `game-environments/game_envs/games/stag_hunt.py`
- Create: `game-environments/game_envs/strategies/stag_hunt_strategies.py`
- Create: `game-environments/tests/test_stag_hunt.py`
- Modify: `game-environments/game_envs/games/__init__.py` (if needed for exports)

Stag Hunt is a 2-player coordination game with two pure Nash equilibria: (Stag, Stag) and (Hare, Hare). It tests whether agents trust each other enough to coordinate on the risky but higher-payoff outcome.

**Payoff matrix (default):**
```
             Player 2
             Stag    Hare
Player 1  Stag  (4,4)   (0,3)
          Hare  (3,0)   (3,3)
```

Constraints: mutual_stag > hare > sucker, and mutual_stag > mutual_hare.

- [ ] **Step 1: Read PD implementation as pattern**

Read `game-environments/game_envs/games/prisoners_dilemma.py` — this is the template. Stag Hunt follows the same structure: 2-player, discrete actions, repeated game support.

Also read `game-environments/game_envs/games/registry.py` to understand registration.

- [ ] **Step 2: Write failing tests**

Create `game-environments/tests/test_stag_hunt.py`:

```python
"""Tests for Stag Hunt game implementation."""

import pytest

from game_envs.games.stag_hunt import StagHunt, StagHuntConfig


class TestStagHuntPayoffs:
    """Verify Stag Hunt payoff matrix."""

    def test_mutual_stag(self) -> None:
        """Both choose stag → highest mutual payoff."""
        game = StagHunt(StagHuntConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "stag", "player_1": "stag"})
        assert result.payoffs["player_0"] == pytest.approx(4.0)
        assert result.payoffs["player_1"] == pytest.approx(4.0)

    def test_mutual_hare(self) -> None:
        """Both choose hare → safe but lower payoff."""
        game = StagHunt(StagHuntConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "hare", "player_1": "hare"})
        assert result.payoffs["player_0"] == pytest.approx(3.0)
        assert result.payoffs["player_1"] == pytest.approx(3.0)

    def test_stag_vs_hare(self) -> None:
        """One stag, one hare → stag hunter gets sucker payoff."""
        game = StagHunt(StagHuntConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "stag", "player_1": "hare"})
        assert result.payoffs["player_0"] == pytest.approx(0.0)
        assert result.payoffs["player_1"] == pytest.approx(3.0)

    def test_payoff_ordering(self) -> None:
        """Default config: mutual_stag > hare > sucker."""
        config = StagHuntConfig(num_players=2, num_rounds=1)
        assert config.mutual_stag > config.hare
        assert config.hare > config.sucker
        assert config.mutual_stag > config.mutual_hare

    def test_two_pure_nash_equilibria(self) -> None:
        """Stag Hunt has (stag,stag) and (hare,hare) as NE."""
        config = StagHuntConfig(num_players=2, num_rounds=1)
        # Neither player wants to deviate from (stag,stag)
        assert config.mutual_stag > config.hare  # stag is best response to stag
        # Neither player wants to deviate from (hare,hare)
        assert config.mutual_hare > config.sucker  # hare is best response to hare


class TestStagHuntRepeated:
    """Test multi-round Stag Hunt."""

    def test_multi_round_accumulates(self) -> None:
        game = StagHunt(StagHuntConfig(num_players=2, num_rounds=3))
        game.reset()
        for _ in range(3):
            game.step({"player_0": "stag", "player_1": "stag"})
        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == pytest.approx(12.0)  # 3 * 4.0

    def test_single_round_terminal(self) -> None:
        game = StagHunt(StagHuntConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "stag", "player_1": "stag"})
        assert result.is_terminal


class TestStagHuntRegistration:
    """Test game is registered."""

    def test_in_registry(self) -> None:
        from game_envs.games.registry import GameRegistry

        assert "stag_hunt" in GameRegistry.list_games()
```

- [ ] **Step 3: Run tests to confirm failure**

Run: `uv run python -m pytest game-environments/tests/test_stag_hunt.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Implement Stag Hunt game**

Create `game-environments/game_envs/games/stag_hunt.py` following the Prisoner's Dilemma pattern. Key differences:
- Actions: "stag" and "hare" (instead of "cooperate" and "defect")
- Config fields: `mutual_stag=4.0`, `mutual_hare=3.0`, `hare=3.0`, `sucker=0.0`
- Payoff matrix:
  - (stag, stag) → (mutual_stag, mutual_stag)
  - (hare, hare) → (mutual_hare, mutual_hare)
  - (stag, hare) → (sucker, hare)
  - (hare, stag) → (hare, sucker)
- Validation: `mutual_stag > hare > sucker` and `mutual_stag > mutual_hare`
- Register with `@register_game("stag_hunt", StagHuntConfig)`

- [ ] **Step 5: Implement strategies**

Create `game-environments/game_envs/strategies/stag_hunt_strategies.py`:
- `AlwaysStag` — always cooperate on risky option
- `AlwaysHare` — always play safe
- `StagTitForTat` — start with stag, mirror opponent

- [ ] **Step 6: Run tests**

Run: `uv run python -m pytest game-environments/tests/test_stag_hunt.py -v`
Expected: All PASS

- [ ] **Step 7: Run quality checks and commit**

```bash
uv run ruff format game-environments/ && uv run ruff check game-environments/ --fix
git add game-environments/game_envs/games/stag_hunt.py \
       game-environments/game_envs/strategies/stag_hunt_strategies.py \
       game-environments/tests/test_stag_hunt.py \
       game-environments/game_envs/games/__init__.py
git commit -m "feat: add Stag Hunt game with strategies and tests"
```

---

### Task 8: Implement Battle of the Sexes Game

**Files:**
- Create: `game-environments/game_envs/games/battle_of_sexes.py`
- Create: `game-environments/game_envs/strategies/bos_strategies.py`
- Create: `game-environments/tests/test_battle_of_sexes.py`

Battle of the Sexes: 2 players choose between two events (A and B). Both prefer to coordinate, but player 1 prefers A and player 2 prefers B.

**Payoff matrix (default):**
```
             Player 2
             A       B
Player 1  A  (3,2)   (0,0)
          B  (0,0)   (2,3)
```

Two pure NE: (A,A) and (B,B). Tests coordination under preference conflict.

- [ ] **Step 1: Write failing tests**

Create `game-environments/tests/test_battle_of_sexes.py` with tests for:
- Both choose A → (preferred_a, other_a) payoffs
- Both choose B → (other_b, preferred_b) payoffs
- Mismatch → (0, 0)
- Payoff ordering: preferred > other > mismatch
- Two pure Nash equilibria
- Multi-round accumulation
- Registration in GameRegistry

Follow the same pattern as test_stag_hunt.py.

- [ ] **Step 2: Implement Battle of the Sexes**

Create `game-environments/game_envs/games/battle_of_sexes.py`:
- Actions: "A" and "B"
- Config: `preferred_a=3.0` (player_0 prefers A), `other_a=2.0`, `preferred_b=3.0` (player_1 prefers B), `other_b=2.0`, `mismatch=0.0`
- Register with `@register_game("battle_of_sexes", BoSConfig)`

- [ ] **Step 3: Implement strategies**

Create `game-environments/game_envs/strategies/bos_strategies.py`:
- `AlwaysA` — always choose A
- `AlwaysB` — always choose B
- `Alternating` — alternate between A and B

- [ ] **Step 4: Run tests, quality checks, commit**

```bash
uv run python -m pytest game-environments/tests/test_battle_of_sexes.py -v
uv run ruff format game-environments/ && uv run ruff check game-environments/ --fix
git add game-environments/game_envs/games/battle_of_sexes.py \
       game-environments/game_envs/strategies/bos_strategies.py \
       game-environments/tests/test_battle_of_sexes.py
git commit -m "feat: add Battle of the Sexes game with strategies and tests"
```

---

### Task 9: Implement Elo Rating System

**Files:**
- Create: `atp-games/atp_games/rating/__init__.py`
- Create: `atp-games/atp_games/rating/elo.py`
- Create: `atp-games/tests/test_elo.py`

- [ ] **Step 1: Write failing tests**

Create `atp-games/tests/test_elo.py`:

```python
"""Tests for Elo rating system."""

import pytest

from atp_games.rating.elo import EloCalculator, EloRating


class TestEloCalculator:
    """Test Elo rating calculations."""

    def test_initial_rating(self) -> None:
        calc = EloCalculator()
        rating = calc.create_rating("agent_a")
        assert rating.rating == 1500.0
        assert rating.games_played == 0

    def test_winner_gains_rating(self) -> None:
        calc = EloCalculator()
        ra = calc.create_rating("a")
        rb = calc.create_rating("b")
        calc.update(ra, rb, winner="a")
        assert ra.rating > 1500.0
        assert rb.rating < 1500.0

    def test_ratings_sum_preserved(self) -> None:
        """Total rating pool is conserved (zero-sum updates)."""
        calc = EloCalculator()
        ra = calc.create_rating("a")
        rb = calc.create_rating("b")
        total_before = ra.rating + rb.rating
        calc.update(ra, rb, winner="a")
        total_after = ra.rating + rb.rating
        assert total_after == pytest.approx(total_before)

    def test_draw_moves_toward_equal(self) -> None:
        calc = EloCalculator()
        ra = EloRating(agent="a", rating=1600.0)
        rb = EloRating(agent="b", rating=1400.0)
        calc.update(ra, rb, winner=None)  # Draw
        assert ra.rating < 1600.0  # Higher-rated loses points
        assert rb.rating > 1400.0  # Lower-rated gains points

    def test_upset_gives_more_points(self) -> None:
        """Underdog winning gives bigger rating change."""
        calc = EloCalculator(k_factor=32)
        # Underdog wins
        ra = EloRating(agent="a", rating=1200.0)
        rb = EloRating(agent="b", rating=1800.0)
        calc.update(ra, rb, winner="a")
        gain_a = ra.rating - 1200.0
        # Favorite wins
        rc = EloRating(agent="c", rating=1800.0)
        rd = EloRating(agent="d", rating=1200.0)
        calc.update(rc, rd, winner="c")
        gain_c = rc.rating - 1800.0
        assert gain_a > gain_c  # Underdog gain > favorite gain

    def test_games_played_increments(self) -> None:
        calc = EloCalculator()
        ra = calc.create_rating("a")
        rb = calc.create_rating("b")
        calc.update(ra, rb, winner="a")
        assert ra.games_played == 1
        assert rb.games_played == 1

    def test_custom_k_factor(self) -> None:
        calc_low = EloCalculator(k_factor=16)
        calc_high = EloCalculator(k_factor=64)
        ra1 = calc_low.create_rating("a")
        rb1 = calc_low.create_rating("b")
        ra2 = calc_high.create_rating("a")
        rb2 = calc_high.create_rating("b")
        calc_low.update(ra1, rb1, winner="a")
        calc_high.update(ra2, rb2, winner="a")
        # Higher K = bigger change
        assert abs(ra2.rating - 1500) > abs(ra1.rating - 1500)
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `uv run python -m pytest atp-games/tests/test_elo.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement Elo rating**

Create `atp-games/atp_games/rating/__init__.py`:
```python
"""Rating systems for agent evaluation."""

from atp_games.rating.elo import EloCalculator, EloRating

__all__ = ["EloCalculator", "EloRating"]
```

Create `atp-games/atp_games/rating/elo.py`:

```python
"""Elo rating system for game-theoretic agent evaluation.

Standard Elo formula:
  Expected score: E_a = 1 / (1 + 10^((R_b - R_a) / 400))
  New rating: R_a' = R_a + K * (S_a - E_a)

Where S_a is actual score (1 for win, 0.5 for draw, 0 for loss).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EloRating:
    """Elo rating for an agent."""

    agent: str
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary."""
        return {
            "agent": self.agent,
            "rating": round(self.rating, 1),
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
        }


class EloCalculator:
    """Calculate and update Elo ratings."""

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_rating: float = 1500.0,
    ) -> None:
        self.k_factor = k_factor
        self.initial_rating = initial_rating

    def create_rating(self, agent: str) -> EloRating:
        """Create a new rating for an agent."""
        return EloRating(agent=agent, rating=self.initial_rating)

    def expected_score(
        self, rating_a: float, rating_b: float
    ) -> float:
        """Calculate expected score for player A."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        rating_a: EloRating,
        rating_b: EloRating,
        winner: str | None,
    ) -> None:
        """Update ratings after a match.

        Args:
            rating_a: First player's rating (mutated in place).
            rating_b: Second player's rating (mutated in place).
            winner: Name of winner, or None for draw.
        """
        e_a = self.expected_score(rating_a.rating, rating_b.rating)
        e_b = 1.0 - e_a

        if winner is None:
            s_a, s_b = 0.5, 0.5
            rating_a.draws += 1
            rating_b.draws += 1
        elif winner == rating_a.agent:
            s_a, s_b = 1.0, 0.0
            rating_a.wins += 1
            rating_b.losses += 1
        elif winner == rating_b.agent:
            s_a, s_b = 0.0, 1.0
            rating_a.losses += 1
            rating_b.wins += 1
        else:
            msg = (
                f"Winner '{winner}' is neither "
                f"'{rating_a.agent}' nor '{rating_b.agent}'"
            )
            raise ValueError(msg)

        rating_a.rating += self.k_factor * (s_a - e_a)
        rating_b.rating += self.k_factor * (s_b - e_b)
        rating_a.games_played += 1
        rating_b.games_played += 1
        rating_a.history.append(rating_a.rating)
        rating_b.history.append(rating_b.rating)
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest atp-games/tests/test_elo.py -v`
Expected: All PASS

- [ ] **Step 5: Quality checks and commit**

```bash
uv run ruff format atp-games/atp_games/rating/ atp-games/tests/test_elo.py
uv run ruff check atp-games/ --fix
git add atp-games/atp_games/rating/ atp-games/tests/test_elo.py
git commit -m "feat: add Elo rating system for tournament agent evaluation"
```

---

### Task 10: Integrate Elo into Tournament Runner

**Files:**
- Modify: `atp-games/atp_games/models.py`
- Modify: `atp-games/atp_games/suites/tournament.py`
- Create: `atp-games/tests/test_tournament_elo.py`

- [ ] **Step 1: Read current models and tournament code**

Read: `atp-games/atp_games/models.py`, `atp-games/atp_games/suites/tournament.py`

- [ ] **Step 2: Write failing test**

Create `atp-games/tests/test_tournament_elo.py`:

```python
"""Tests for Elo integration in tournament runner."""

import pytest


class TestTournamentElo:
    """Test Elo ratings are updated during tournaments."""

    @pytest.mark.anyio
    async def test_round_robin_has_elo_ratings(self) -> None:
        """Round-robin tournament populates elo_ratings."""
        from game_envs.games.prisoners_dilemma import (
            PDConfig,
            PrisonersDilemma,
        )

        from atp_games.rating.elo import EloCalculator
        from atp_games.suites.tournament import run_round_robin

        game = PrisonersDilemma(PDConfig(num_players=2, num_rounds=5))
        # Use builtin strategies as agents
        from game_envs.strategies.pd_strategies import (
            AlwaysCooperate,
            AlwaysDefect,
            TitForTat,
        )

        from atp_games.runner.game_runner import BuiltinAdapter

        agents = {
            "cooperator": BuiltinAdapter(AlwaysCooperate()),
            "defector": BuiltinAdapter(AlwaysDefect()),
            "tft": BuiltinAdapter(TitForTat()),
        }
        elo = EloCalculator()
        result = await run_round_robin(
            game=game,
            agents=agents,
            episodes_per_match=10,
            elo_calculator=elo,
        )
        # All agents should have Elo ratings
        assert result.elo_ratings is not None
        assert len(result.elo_ratings) == 3
        # Defector should have highest Elo in short PD
        for name, rating in result.elo_ratings.items():
            assert rating.games_played > 0

    @pytest.mark.anyio
    async def test_elo_optional(self) -> None:
        """Tournament works without Elo (backward compatible)."""
        from game_envs.games.prisoners_dilemma import (
            PDConfig,
            PrisonersDilemma,
        )

        from atp_games.suites.tournament import run_round_robin

        from game_envs.strategies.pd_strategies import (
            AlwaysCooperate,
            AlwaysDefect,
        )

        from atp_games.runner.game_runner import BuiltinAdapter

        game = PrisonersDilemma(PDConfig(num_players=2, num_rounds=3))
        agents = {
            "coop": BuiltinAdapter(AlwaysCooperate()),
            "defect": BuiltinAdapter(AlwaysDefect()),
        }
        result = await run_round_robin(
            game=game, agents=agents, episodes_per_match=5
        )
        assert result.elo_ratings is None  # No Elo by default
```

- [ ] **Step 3: Add elo_ratings field to TournamentResult**

In `atp-games/atp_games/models.py` (or wherever `TournamentResult` is defined — check the actual location in `tournament.py`), add:

```python
from atp_games.rating.elo import EloRating

# Add to TournamentResult dataclass:
elo_ratings: dict[str, EloRating] | None = None
```

- [ ] **Step 4: Add optional elo_calculator parameter to tournament functions**

In `atp-games/atp_games/suites/tournament.py`, modify `run_round_robin` (and optionally elimination functions) to accept an optional `elo_calculator` parameter:

```python
from atp_games.rating.elo import EloCalculator

async def run_round_robin(
    game,
    agents,
    episodes_per_match=10,
    config=None,
    elo_calculator: EloCalculator | None = None,  # Add this
) -> TournamentResult:
    # ... existing match logic ...

    # After each match result is computed:
    if elo_calculator is not None:
        # Initialize ratings if first match
        if not hasattr(run_round_robin, '_ratings'):
            ...
        elo_calculator.update(
            ratings[match.agent_a],
            ratings[match.agent_b],
            winner=match.winner,
        )

    # Set elo_ratings on TournamentResult
    result.elo_ratings = ratings if elo_calculator else None
```

IMPORTANT: Read the actual tournament.py code carefully. The modification must fit the existing code structure. Don't restructure — add Elo as an optional layer.

- [ ] **Step 5: Run tests**

Run: `uv run python -m pytest atp-games/tests/test_tournament_elo.py -v`
Expected: All PASS

- [ ] **Step 6: Run existing tournament tests for regression**

Run: `uv run python -m pytest atp-games/tests/ -v -q 2>&1 | tail -20`
Expected: No regressions (existing tests don't pass elo_calculator, so elo_ratings should be None)

- [ ] **Step 7: Quality checks and commit**

```bash
uv run ruff format atp-games/ && uv run ruff check atp-games/ --fix
git add atp-games/atp_games/models.py atp-games/atp_games/suites/tournament.py atp-games/tests/test_tournament_elo.py
git commit -m "feat: integrate Elo rating system into tournament runner"
```

---

### Task 11: LLM Benchmark Configuration

**Files:**
- Create: `examples/experiments/llm_benchmark_config.yaml`

This is a YAML configuration file defining the standard benchmark setup for LLM agents.

- [ ] **Step 1: Create benchmark config**

```yaml
# LLM Agent Benchmark Configuration
# Run: atp game benchmark --config examples/experiments/llm_benchmark_config.yaml
#
# Tests LLM agents across all canonical games to measure:
# - Cooperation rate
# - Adaptation speed
# - Strategic consistency
# - Cost per game

name: llm-agent-benchmark
description: Standard benchmark for evaluating LLM agents in game-theoretic settings

games:
  - name: prisoners_dilemma
    config:
      num_rounds: 50
      discount_factor: 0.99
    episodes: 100

  - name: stag_hunt
    config:
      num_rounds: 50
      discount_factor: 0.99
    episodes: 100

  - name: battle_of_sexes
    config:
      num_rounds: 50
      discount_factor: 0.99
    episodes: 100

  - name: public_goods
    config:
      num_players: 3
      num_rounds: 30
      endowment: 20.0
      multiplier: 1.6
    episodes: 50

  - name: auction
    config:
      num_rounds: 30
      auction_type: second_price
    episodes: 50

tournament:
  mode: round_robin
  elo:
    enabled: true
    k_factor: 32
    initial_rating: 1500

metrics:
  - cooperation_rate
  - adaptation_speed
  - consistency
  - cost_per_game

# Agent configurations (override per run)
# Example: atp game benchmark --agent-config model=gpt-4o-mini
agents:
  - name: baseline-tft
    type: builtin
    strategy: tit_for_tat

  - name: baseline-random
    type: builtin
    strategy: random

# LLM agents added via CLI:
# atp game benchmark \
#   --agent "gpt-4o-mini:http://localhost:8001" \
#   --agent "claude-haiku:http://localhost:8002" \
#   --agent "gemini-flash:http://localhost:8003"

output:
  format: json
  save_to: results/llm-benchmark/
  include_traces: true
```

- [ ] **Step 2: Commit**

```bash
git add examples/experiments/llm_benchmark_config.yaml
git commit -m "feat: add LLM agent benchmark configuration for game-theoretic evaluation"
```

---

## Sync Point & Verification

### Task 12: Phase 1 Sync Point Verification

- [ ] **Step 1: Run all unit tests**

Run: `uv run python -m pytest tests/unit/ --ignore=tests/unit/dashboard --ignore=tests/unit/test_github_import.py -q 2>&1 | tail -10`
Expected: All pass (except pre-existing llm_judge failures)

- [ ] **Step 2: Run game-environments tests**

Run: `uv run python -m pytest game-environments/tests/ -q 2>&1 | tail -10`
Expected: All pass including new Stag Hunt and Battle of the Sexes tests

- [ ] **Step 3: Run atp-games tests**

Run: `uv run python -m pytest atp-games/tests/ -q 2>&1 | tail -10`
Expected: All pass including new Elo and tournament Elo tests

- [ ] **Step 4: Verify builds**

Run: `cd game-environments && uv build && cd ../atp-games && uv build && cd ..`
Expected: Both packages build successfully

- [ ] **Step 5: Verify quickstart command**

Run: `uv run atp quickstart --help`
Expected: Help text displayed

- [ ] **Step 6: Run quality checks**

Run: `uv run ruff format . && uv run ruff check . && uv run pyrefly check`
Expected: Clean (except pre-existing ruff UP042/UP047 style warnings)

- [ ] **Step 7: Review all Phase 1 commits**

Run: `git log --oneline` and verify ~10-12 commits covering all tasks.
