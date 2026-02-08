# game-environments

> Standalone game theory environments for evaluating AI agents in strategic interactions

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`game-environments` is a lightweight, framework-agnostic library for creating game-theoretic environments where AI agents interact strategically. It provides core abstractions for defining games, action spaces, observations, and baseline strategies -- everything needed to evaluate how agents behave in multi-agent strategic settings.

**Key design principles:**

- **Zero ATP dependency** -- usable standalone in Jupyter, research scripts, or any Python project
- **LLM-first observations** -- `Observation.to_prompt()` generates human-readable text for LLM agents
- **Analytical verifiability** -- games have known Nash equilibria for rigorous evaluation
- **Progressive complexity** -- from one-shot 2-player games to repeated N-player stochastic games

## Installation

```bash
# From the game-environments directory
uv sync

# Or install as a dependency in another project
uv add game-environments --path ./game-environments
```

### Optional dependencies

```bash
# NumPy support for numerical analysis
uv add game-environments[numpy]
```

## Quick Start

### Define a simple game

```python
from game_envs import (
    Game, GameConfig, GameType, MoveOrder,
    DiscreteActionSpace, GameState, Observation, StepResult,
)

class PrisonersDilemma(Game):
    """Classic Prisoner's Dilemma."""

    PAYOFFS = {
        ("cooperate", "cooperate"): (3, 3),
        ("cooperate", "defect"): (0, 5),
        ("defect", "cooperate"): (5, 0),
        ("defect", "defect"): (1, 1),
    }

    @property
    def name(self) -> str:
        return "Prisoner's Dilemma"

    @property
    def game_type(self) -> GameType:
        return GameType.NORMAL_FORM

    @property
    def move_order(self) -> MoveOrder:
        return MoveOrder.SIMULTANEOUS

    @property
    def player_ids(self) -> list[str]:
        return ["player_0", "player_1"]

    def action_space(self, player_id: str) -> DiscreteActionSpace:
        return DiscreteActionSpace(["cooperate", "defect"])

    def reset(self) -> StepResult:
        self._current_round = 0
        self._history.clear()
        self._cumulative = {"player_0": 0.0, "player_1": 0.0}
        state = GameState(
            round_number=0,
            player_states={},
            public_state={"game": self.name},
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state, observations=obs,
            payoffs={"player_0": 0, "player_1": 0},
            is_terminal=False,
        )

    def step(self, actions: dict[str, str]) -> StepResult:
        a0 = actions["player_0"]
        a1 = actions["player_1"]
        p0, p1 = self.PAYOFFS[(a0, a1)]

        self._current_round += 1
        self._cumulative["player_0"] += p0
        self._cumulative["player_1"] += p1

        from game_envs import RoundResult
        self._history.add_round(RoundResult(
            round_number=self._current_round,
            actions=actions,
            payoffs={"player_0": p0, "player_1": p1},
        ))

        terminal = self._current_round >= self.config.num_rounds
        state = GameState(
            round_number=self._current_round,
            player_states={},
            public_state={"last_actions": actions},
            is_terminal=terminal,
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state, observations=obs,
            payoffs={"player_0": p0, "player_1": p1},
            is_terminal=terminal,
        )

    def get_payoffs(self) -> dict[str, float]:
        return dict(self._cumulative)

    @property
    def is_terminal(self) -> bool:
        return self._current_round >= self.config.num_rounds
```

### Run a game

```python
# One-shot game
game = PrisonersDilemma()
result = game.reset()

# Each player sees their observation
obs = result.observations["player_0"]
print(obs.to_prompt())
# You are player 'player_0'.
# Round 0 of 1.
# Available actions:
#   - cooperate
#   - defect

# Play a round
result = game.step({
    "player_0": "cooperate",
    "player_1": "defect",
})
print(result.payoffs)  # {'player_0': 0, 'player_1': 5}
print(result.is_terminal)  # True
```

### Repeated games

```python
# 10-round iterated game with trembling-hand noise
config = GameConfig(num_rounds=10, noise=0.05, seed=42)
game = PrisonersDilemma(config)
result = game.reset()

while not result.is_terminal:
    result = game.step({
        "player_0": "cooperate",
        "player_1": "cooperate",
    })

print(game.get_payoffs())
# {'player_0': ~30.0, 'player_1': ~30.0}
```

### Define a baseline strategy

```python
from game_envs import Strategy, Observation

class TitForTat(Strategy):
    """Start cooperating, then mirror opponent's last move."""

    def __init__(self) -> None:
        self._last_opponent_action: str | None = None

    @property
    def name(self) -> str:
        return "tit_for_tat"

    def choose_action(self, observation: Observation) -> str:
        if not observation.history:
            return "cooperate"
        last_round = observation.history[-1]
        # Find opponent's action
        for pid, action in last_round.actions.items():
            if pid != observation.player_id:
                return action
        return "cooperate"

    def reset(self) -> None:
        self._last_opponent_action = None
```

### Use with LLM agents

Observations produce LLM-friendly prompts:

```python
obs = game.observe("player_0")
prompt = obs.to_prompt()
# You are player 'player_0'.
# Round 3 of 10.
#
# Current state:
#   last_actions: {'player_0': 'cooperate', 'player_1': 'defect'}
#
# Available actions:
#   - cooperate
#   - defect
#
# History:
#   Round 1: actions=[player_0=cooperate, player_1=cooperate] payoffs=[player_0=3, player_1=3]
#   Round 2: actions=[player_0=cooperate, player_1=defect] payoffs=[player_0=0, player_1=5]

# Send prompt to an LLM and parse its response as an action
llm_response = call_llm(prompt)  # Returns "defect"
```

## Core Concepts

### Game

The `Game` ABC defines the lifecycle of a strategic interaction:

```
reset() → step(actions) → step(actions) → ... → is_terminal == True → get_payoffs()
```

Every game implements:

| Property/Method | Description |
|---|---|
| `name` | Human-readable game name |
| `game_type` | `NORMAL_FORM`, `EXTENSIVE_FORM`, `REPEATED`, `STOCHASTIC` |
| `move_order` | `SIMULTANEOUS` or `SEQUENTIAL` |
| `player_ids` | List of player identifiers |
| `action_space(player_id)` | Action space for a specific player |
| `reset()` | Initialize/reset, returns `StepResult` |
| `step(actions)` | Process one round, returns `StepResult` |
| `get_payoffs()` | Cumulative payoffs for all players |
| `is_terminal` | Whether the game has ended |
| `observe(player_id)` | Player-specific observation (overrideable) |

### GameConfig

Immutable configuration with validation:

```python
config = GameConfig(
    num_players=2,       # Number of players (>= 1)
    num_rounds=10,       # Rounds per episode (>= 1)
    discount_factor=0.95,# Future payoff discount [0, 1]
    noise=0.05,          # Trembling-hand probability [0, 1]
    communication=True,  # Enable pre-action messaging
    seed=42,             # Reproducibility seed
)
```

### Action Spaces

Three types of action spaces, all with LLM-friendly descriptions:

| Type | Use Case | Example |
|---|---|---|
| `DiscreteActionSpace` | Finite choices | `["cooperate", "defect"]` |
| `ContinuousActionSpace` | Numeric range | Bid in `[0.0, 100.0]` |
| `StructuredActionSpace` | Allocation vectors | Blotto troop deployment |

```python
# Discrete
space = DiscreteActionSpace(["cooperate", "defect"])
space.to_description()  # "Choose one of: cooperate, defect"

# Continuous
space = ContinuousActionSpace(0.0, 100.0, "Your bid amount")
space.contains(50.0)  # True

# Structured (allocation)
space = StructuredActionSpace(
    schema={"fields": ["A", "B", "C"], "total": 100, "min_value": 0},
    description="Allocate 100 troops across 3 battlefields",
)
space.contains({"A": 40, "B": 30, "C": 30})  # True
```

### Observation

What a player sees each round. Includes `to_prompt()` for LLM agents and `to_dict()` / `from_dict()` for serialization:

```python
obs = game.observe("player_0")

# For LLM agents
prompt = obs.to_prompt()

# For ATP protocol integration
data = obs.to_dict()
obs2 = Observation.from_dict(data)
```

### GameHistory

Tracks all rounds with per-player filtering for partial observability:

```python
history = game.history

# Full history
all_rounds = history.rounds

# Filtered view (only show own + opponent's actions)
player_view = history.for_player("player_0", visible_players=["player_0", "player_1"])

# Per-player stats
actions = history.get_player_actions("player_0")  # ["cooperate", "defect", ...]
payoffs = history.get_player_payoffs("player_0")   # [3.0, 0.0, ...]
total = history.total_payoff("player_0")            # 15.0
```

### Strategy

Baseline agents for comparison and exploitability analysis:

```python
class AlwaysCooperate(Strategy):
    @property
    def name(self) -> str:
        return "always_cooperate"

    def choose_action(self, observation: Observation) -> str:
        return "cooperate"
```

## Serialization

All core data structures support `to_dict()` / `from_dict()` for JSON serialization:

```python
# Serialize
data = step_result.to_dict()
json_str = json.dumps(data)

# Deserialize
step_result = StepResult.from_dict(json.loads(json_str))
```

Serializable types: `Message`, `RoundResult`, `GameState`, `Observation`, `StepResult`, `GameHistory`.

## Architecture

```
game_envs/
├── core/
│   ├── action.py     # ActionSpace ABC + Discrete/Continuous/Structured
│   ├── game.py       # Game ABC, GameConfig, GameType, MoveOrder
│   ├── history.py    # GameHistory with per-player filtering
│   ├── state.py      # GameState, Observation, StepResult, Message, RoundResult
│   └── strategy.py   # Strategy ABC for baseline agents
└── __init__.py       # Public API exports
```

## Integration with ATP

While `game-environments` is standalone, it integrates with the [ATP Platform](https://github.com/yourusername/atp-platform-ru) via the `atp-games` plugin:

```
game-environments (standalone)     ATP Platform
┌──────────────────────────┐      ┌────────────────────────────┐
│  Game ABC                │      │  GameRunner                │
│  ActionSpace variants    │◄─────│  Game Evaluators           │
│  Observation.to_prompt() │      │  Game Suite Loader (YAML)  │
│  Strategy baselines      │      │  Game Reporter (JSON/HTML) │
│  GameHistory             │      │  Dashboard integration     │
└──────────────────────────┘      └────────────────────────────┘
```

- **Observations** map to `ATPRequest.context`
- **Actions** map to `ATPResponse.artifacts`
- **Payoffs/metrics** map to game-theoretic evaluators

## Development

```bash
cd game-environments

# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/ -v --cov=game_envs

# Format and lint
uv run ruff format .
uv run ruff check .

# Type checking
pyrefly check
```

### Testing

Tests use `pytest` with `hypothesis` for property-based testing of action space invariants:

```bash
uv run pytest tests/ -v --cov=game_envs --cov-report=term-missing
```

## License

MIT License -- see the parent project's [LICENSE](../LICENSE) for details.
