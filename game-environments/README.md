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

## Built-in Games

Five canonical games are included, each registered via `@register_game` and instantiable through `GameRegistry`:

| Game | Players | Action Space | Key Feature |
|---|---|---|---|
| **Prisoner's Dilemma** | 2 | Discrete (`cooperate`/`defect`) | Configurable payoff matrix, noise, repeated play |
| **Public Goods** | 2-20 | Continuous (`[0, endowment]`) | Multiplier, optional punishment mechanism |
| **Auction** | 2+ | Continuous (`[min_bid, max_bid]`) | First-price & second-price (Vickrey), private values |
| **Colonel Blotto** | 2 | Structured (allocation vector) | Multiple battlefields, troop allocation |
| **Congestion** | 2-50 | Discrete (route choice) | Network routing, latency functions |

### Using built-in games

```python
from game_envs import GameRegistry, GameConfig

# List available games
print(GameRegistry.list_games())
# ['prisoners_dilemma', 'public_goods', 'auction', 'colonel_blotto', 'congestion']

# Create via registry
game = GameRegistry.create("prisoners_dilemma", {
    "num_rounds": 10,
    "noise": 0.05,
})

# Or import directly
from game_envs import PrisonersDilemma, PDConfig
game = PrisonersDilemma(PDConfig(
    R=3, S=0, T=5, P=1,  # payoff parameters
    num_rounds=10,
))
```

### Game-specific configs

```python
from game_envs import (
    PDConfig, PGConfig, AuctionConfig,
    BlottoConfig, CongestionConfig, RouteDefinition,
)

# Prisoner's Dilemma with custom payoffs
pd = PDConfig(R=3, S=0, T=5, P=1, num_rounds=50)

# Public Goods with punishment
pg = PGConfig(
    endowment=10.0, multiplier=1.6,
    punishment_cost=1.0, punishment_effect=3.0,
    num_players=4, num_rounds=20,
)

# Sealed-bid auction
auction = AuctionConfig(
    auction_type="second_price",
    min_bid=0.0, max_bid=100.0,
    num_players=3,
)

# Colonel Blotto
blotto = BlottoConfig(
    num_battlefields=5, total_troops=100,
)

# Congestion game with custom routes
congestion = CongestionConfig(
    routes=[
        RouteDefinition(name="highway", base_cost=5.0, coefficient=2.0),
        RouteDefinition(name="backroad", base_cost=10.0, coefficient=0.5),
    ],
    num_players=10,
)
```

## Built-in Strategies

Baseline strategies are provided for each game:

| Game | Strategies |
|---|---|
| **Prisoner's Dilemma** | `TitForTat`, `AlwaysCooperate`, `AlwaysDefect`, `GrimTrigger`, `Pavlov`, `RandomStrategy` |
| **Public Goods** | `FullContributor`, `FreeRider`, `ConditionalCooperator`, `Punisher` |
| **Auction** | `TruthfulBidder`, `ShadeBidder(factor)`, `RandomBidder` |
| **Colonel Blotto** | `UniformAllocation`, `ConcentratedAllocation`, `NashMixed` |
| **Congestion** | `SelfishRouter`, `SocialOptimum`, `EpsilonGreedy` |

```python
from game_envs import StrategyRegistry, TitForTat

# Look up by name
strategy_cls = StrategyRegistry.get("tit_for_tat")
strategy = strategy_cls()

# Or import directly
tft = TitForTat()
action = tft.choose_action(observation)
```

## Analysis Tools

### Nash Equilibrium Solver

Compute Nash equilibria for 2-player bimatrix games:

```python
import numpy as np
from game_envs import NashSolver

# Prisoner's Dilemma payoff matrices
payoff_1 = np.array([[3, 0], [5, 1]])  # row player
payoff_2 = np.array([[3, 5], [0, 1]])  # column player

# Find all Nash equilibria
equilibria = NashSolver.solve_2player(payoff_1, payoff_2)
for ne in equilibria:
    print(f"Strategies: {ne.strategies}")
    print(f"Payoffs: {ne.payoffs}")
    print(f"Type: {'pure' if ne.is_pure() else 'mixed'}")
```

Available methods: `support_enumeration`, `lemke_howson`, `fictitious_play`, `replicator_dynamics`.

### Exploitability Calculator

Measure how exploitable a strategy is:

```python
from game_envs import compute_exploitability, EmpiricalStrategy

# Extract empirical strategy from game history
emp = EmpiricalStrategy.from_history(history, "player_0")

# Compute exploitability (best-response payoff gap)
result = compute_exploitability(
    payoff_1, payoff_2,
    strategy_1, strategy_2,
    action_names_1=["cooperate", "defect"],
    action_names_2=["cooperate", "defect"],
)
print(f"Total exploitability: {result.total:.4f}")
print(f"Per player: {result.per_player}")
# exploitability ~ 0 means playing near Nash equilibrium
```

### Cooperation Metrics

Analyze cooperative behavior in repeated games:

```python
from game_envs import cooperation_rate, conditional_cooperation, reciprocity_index

# Cooperation rate (fraction of cooperative actions)
rate = cooperation_rate(history, "player_0")

# Conditional cooperation: P(C|C) and P(C|D)
cond = conditional_cooperation(history, "player_0", "player_1")
print(f"P(C|C) = {cond['prob_c_given_c']:.2f}")
print(f"P(C|D) = {cond['prob_c_given_d']:.2f}")

# Reciprocity index (correlation between players)
recip = reciprocity_index(history, "player_0", "player_1")
# +1 = perfect reciprocity, -1 = anti-reciprocity
```

### Fairness Metrics

Evaluate outcome fairness:

```python
from game_envs import gini_coefficient, envy_freeness, proportionality, utilitarian_welfare

payoffs = {"player_0": 15.0, "player_1": 25.0, "player_2": 10.0}

# Gini coefficient (0 = perfect equality, 1 = perfect inequality)
gini = gini_coefficient(list(payoffs.values()))

# Envy-freeness check
is_envy_free = envy_freeness(payoffs)

# Utilitarian welfare (sum of payoffs)
welfare = utilitarian_welfare(list(payoffs.values()))
```

### Population Dynamics

Simulate evolutionary strategy dynamics:

```python
from game_envs import PopulationSimulator, ReplicatorDynamics, MoranProcess, is_ess

# Replicator dynamics (continuous-time)
rd = ReplicatorDynamics()
result = rd.simulate(
    payoff_matrix=np.array([[3, 0], [5, 1]]),
    initial_freqs=np.array([0.5, 0.5]),
    generations=100,
)
# result.snapshots contains strategy frequencies over time

# Moran process (stochastic finite population)
mp = MoranProcess(population_size=50)
result = mp.simulate(
    payoff_matrix=np.array([[3, 0], [5, 1]]),
    initial_counts=np.array([25, 25]),
    generations=200,
)

# Check evolutionarily stable strategy
stable = is_ess(strategy_index=0, payoff_matrix=payoff_matrix)

# Full population simulation
sim = PopulationSimulator(
    game=game,
    strategies=[TitForTat(), AlwaysDefect()],
    population_size=100,
    mutation_rate=0.01,
)
result = sim.run(generations=500)
```

## Game Development Guide

### Creating a custom game

Implement the `Game` ABC:

```python
from game_envs import (
    Game, GameConfig, GameType, MoveOrder,
    DiscreteActionSpace, GameState, Observation,
    StepResult, RoundResult,
)
from game_envs.games.registry import register_game

@register_game("matching_pennies")
class MatchingPennies(Game):
    """Two-player zero-sum game."""

    @property
    def name(self) -> str:
        return "Matching Pennies"

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
        return DiscreteActionSpace(["heads", "tails"])

    def reset(self) -> StepResult:
        self._current_round = 0
        self._history.clear()
        self._cumulative = {pid: 0.0 for pid in self.player_ids}
        state = GameState(
            round_number=0, player_states={},
            public_state={"game": self.name},
        )
        return StepResult(
            state=state,
            observations={pid: self.observe(pid) for pid in self.player_ids},
            payoffs={pid: 0.0 for pid in self.player_ids},
            is_terminal=False,
        )

    def step(self, actions: dict[str, str]) -> StepResult:
        a0, a1 = actions["player_0"], actions["player_1"]
        match = a0 == a1
        p0, p1 = (1.0, -1.0) if match else (-1.0, 1.0)

        self._current_round += 1
        self._cumulative["player_0"] += p0
        self._cumulative["player_1"] += p1
        self._history.add_round(RoundResult(
            round_number=self._current_round,
            actions=actions,
            payoffs={"player_0": p0, "player_1": p1},
        ))

        terminal = self._current_round >= self.config.num_rounds
        state = GameState(
            round_number=self._current_round, player_states={},
            public_state={"last_actions": actions},
            is_terminal=terminal,
        )
        return StepResult(
            state=state,
            observations={pid: self.observe(pid) for pid in self.player_ids},
            payoffs={"player_0": p0, "player_1": p1},
            is_terminal=terminal,
        )

    def get_payoffs(self) -> dict[str, float]:
        return dict(self._cumulative)

    @property
    def is_terminal(self) -> bool:
        return self._current_round >= self.config.num_rounds
```

After registration, the game is available via `GameRegistry.create("matching_pennies", config)`.

### Creating a custom strategy

```python
from game_envs import Strategy, Observation, StrategyRegistry

class AdaptiveStrategy(Strategy):
    """Adapts based on opponent cooperation rate."""

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "adaptive"

    def choose_action(self, observation: Observation) -> str:
        if not observation.history:
            return "cooperate"
        # Count opponent's cooperative actions
        opp_id = [p for p in observation.history[0].actions
                  if p != observation.player_id][0]
        coop_count = sum(
            1 for r in observation.history
            if r.actions.get(opp_id) == "cooperate"
        )
        rate = coop_count / len(observation.history)
        return "cooperate" if rate >= self._threshold else "defect"

    def reset(self) -> None:
        pass  # No internal state to reset

# Register for use in YAML suites
StrategyRegistry.register("adaptive", AdaptiveStrategy)
```

## Architecture

```
game_envs/
├── core/
│   ├── action.py          # ActionSpace ABC + Discrete/Continuous/Structured
│   ├── game.py            # Game ABC, GameConfig, GameType, MoveOrder
│   ├── history.py         # GameHistory with per-player filtering
│   ├── state.py           # GameState, Observation, StepResult, Message, RoundResult
│   ├── strategy.py        # Strategy ABC for baseline agents
│   └── communication.py   # CommunicationChannel, InformationSet
├── games/
│   ├── registry.py        # GameRegistry with @register_game decorator
│   ├── prisoners_dilemma.py
│   ├── public_goods.py
│   ├── auction.py
│   ├── colonel_blotto.py
│   └── congestion.py
├── strategies/            # Built-in baseline strategies per game
├── analysis/
│   ├── nash_solver.py     # NashSolver (support enum, Lemke-Howson, etc.)
│   ├── exploitability.py  # Best response, exploitability computation
│   ├── cooperation.py     # Cooperation rate, reciprocity, conditional coop
│   ├── fairness.py        # Gini, envy-freeness, proportionality
│   ├── population.py      # Replicator dynamics, Moran process, ESS
│   └── models.py          # NashEquilibrium data model
└── __init__.py            # Public API exports
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
