# atp-games

> ATP plugin for game-theoretic agent evaluation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`atp-games` bridges the standalone [`game-environments`](../game-environments/) library with the [ATP Platform](../), enabling game-theoretic evaluation of AI agents through the standard ATP testing pipeline. It provides:

- **GameRunner** -- orchestrates multi-agent game execution via ATP protocol
- **Protocol mapping** -- converts game observations to ATP requests and responses back to game actions
- **Game-theoretic evaluators** -- payoff analysis, exploitability, cooperation metrics, equilibrium distance
- **YAML game suites** -- declarative game evaluation definitions
- **Tournament & cross-play** -- round-robin, elimination brackets, agent comparison matrices

## Installation

```bash
cd atp-games
uv sync

# Or install as a dependency
uv add atp-games --path ./atp-games
```

### Dependencies

- `atp-platform` (parent package)
- `game-environments` (game library)
- `numpy` (for Nash solver and exploitability analysis)

## Quick Start

### Run a built-in game suite

```bash
# Evaluate two strategies on Prisoner's Dilemma
uv run atp test --suite=game:prisoners_dilemma.yaml
```

### Programmatic usage

```python
import asyncio
from game_envs import PrisonersDilemma, PDConfig, TitForTat, AlwaysDefect
from atp_games import (
    GameRunner, GameRunConfig, BuiltinAdapter,
)

async def main():
    # Create game
    game = PrisonersDilemma(PDConfig(num_rounds=50))

    # Wrap strategies as ATP-compatible adapters
    agents = {
        "player_0": BuiltinAdapter(TitForTat()),
        "player_1": BuiltinAdapter(AlwaysDefect()),
    }

    # Run evaluation
    runner = GameRunner()
    result = await runner.run_game(
        game=game,
        agents=agents,
        config=GameRunConfig(episodes=20, base_seed=42),
    )

    # Analyze results
    print(f"Episodes: {result.num_episodes}")
    print(f"Average payoffs: {result.average_payoffs}")

    for stat in result.player_statistics():
        print(
            f"  {stat.player_id}: "
            f"mean={stat.mean:.2f} "
            f"95% CI=[{stat.ci_lower:.2f}, {stat.ci_upper:.2f}]"
        )

    # Compare agents (Welch's t-test)
    for cmp in result.agent_comparisons():
        print(
            f"  {cmp.player_a} vs {cmp.player_b}: "
            f"p={cmp.p_value:.4f} "
            f"{'significant' if cmp.is_significant else 'not significant'}"
        )

asyncio.run(main())
```

## Game Suite YAML Format

Game suites define complete evaluation scenarios in YAML:

```yaml
type: game_suite
name: PD Cooperation Test
version: "1.0"

game:
  type: prisoners_dilemma
  variant: repeated          # "one_shot" or "repeated"
  config:
    num_rounds: 100
    noise: 0.0
    discount_factor: 1.0

agents:
  - name: my_agent
    adapter: http
    endpoint: ${AGENT_ENDPOINT}   # Variable substitution for CI

  - name: baseline_tft
    adapter: builtin
    strategy: tit_for_tat

evaluation:
  episodes: 50
  metrics:
    - type: average_payoff
      weight: 1.0
    - type: exploitability
      weight: 0.5
      config:
        epsilon: 0.15
    - type: cooperation
      weight: 0.5
  thresholds:
    average_payoff:
      min: 1.0

reporting:
  strategy_profile: true
  payoff_matrix: true
  round_by_round: true
  export_formats:
    - json
    - csv
```

### YAML Reference

#### `game` section

| Field | Type | Description |
|---|---|---|
| `type` | string | Game name from registry (`prisoners_dilemma`, `auction`, `colonel_blotto`, `congestion`, `public_goods`) |
| `variant` | string | `"one_shot"` or `"repeated"` |
| `config` | dict | Game-specific config (passed to game constructor) |

#### `agents` section

Each agent entry:

| Field | Type | Description |
|---|---|---|
| `name` | string | Display name |
| `adapter` | string | `"builtin"`, `"http"`, `"cli"`, `"docker"` |
| `strategy` | string | For `builtin` adapter: strategy name from registry |
| `endpoint` | string | For `http` adapter: URL |
| `config` | dict | Additional adapter configuration |

#### `evaluation` section

| Field | Type | Description |
|---|---|---|
| `episodes` | int | Number of game episodes to run |
| `metrics` | list | Evaluator metrics to compute |
| `thresholds` | dict | Pass/fail thresholds per metric |

Metric types: `average_payoff`, `exploitability`, `cooperation`, `equilibrium`.

#### Variable substitution

Use `${VAR_NAME}` for environment variable substitution (useful for CI):

```yaml
agents:
  - name: my_agent
    adapter: http
    endpoint: ${AGENT_ENDPOINT}
```

#### Suite inheritance

Extend a base suite:

```yaml
extends: base_pd.yaml

evaluation:
  episodes: 100  # Override episode count
```

## Evaluators

Four game-theoretic evaluators integrate with the ATP scoring pipeline:

### PayoffEvaluator

Evaluates game outcomes based on payoff metrics.

**Checks:**
- Average payoff per player (with min/max thresholds)
- Payoff distribution (min, max, median, percentiles)
- Social welfare (sum of average payoffs)
- Pareto efficiency

```yaml
metrics:
  - type: average_payoff
    weight: 1.0
    config:
      min_payoff:
        player_0: 2.0
      min_social_welfare: 4.0
      pareto_check: true
```

### ExploitabilityEvaluator

Measures how exploitable an agent's strategy is.

**Checks:**
- Per-player exploitability (best-response payoff gap)
- Total exploitability
- Empirical strategy extraction

```yaml
metrics:
  - type: exploitability
    weight: 0.5
    config:
      epsilon: 0.15     # Max exploitability for pass
      payoff_matrix_1: [[3, 0], [5, 1]]
      payoff_matrix_2: [[3, 5], [0, 1]]
      action_names_1: ["cooperate", "defect"]
      action_names_2: ["cooperate", "defect"]
```

A Nash equilibrium strategy has exploitability ~ 0. A dominated strategy (e.g., AlwaysCooperate in PD) has high exploitability.

### CooperationEvaluator

Measures cooperative behavior patterns.

**Checks:**
- Cooperation rate per player (with thresholds)
- Conditional cooperation: P(C|C) and P(C|D)
- Reciprocity index (cooperation correlation between players)

```yaml
metrics:
  - type: cooperation
    weight: 0.5
    config:
      min_cooperation_rate:
        player_0: 0.6
      min_reciprocity: 0.3
```

### EquilibriumEvaluator

Measures proximity to Nash equilibrium.

**Checks:**
- L1 distance to nearest Nash equilibrium
- Equilibrium classification (pure/mixed)
- Convergence detection over time

```yaml
metrics:
  - type: equilibrium
    weight: 0.5
    config:
      max_nash_distance: 0.5
      convergence_window: 20
      convergence_threshold: 0.1
      payoff_matrix_1: [[3, 0], [5, 1]]
      payoff_matrix_2: [[3, 5], [0, 1]]
```

## Tournament Mode

### Round-Robin

Every agent plays every other agent:

```python
from atp_games import run_round_robin

result = await run_round_robin(
    game=game,
    agents={"tft": tft_adapter, "allc": allc_adapter, "alld": alld_adapter},
    config=GameRunConfig(episodes=20),
)
print(result.standings)  # Sorted by total payoff
```

### Single Elimination

```python
from atp_games import run_single_elimination

result = await run_single_elimination(
    game=game,
    agents=agents,
    config=config,
)
print(result.bracket)
print(result.winner)
```

### Double Elimination

```python
from atp_games import run_double_elimination

result = await run_double_elimination(
    game=game,
    agents=agents,
    config=config,
)
```

### Cross-Play Matrix

Run every agent pair (including self-play) and generate a payoff heatmap:

```python
from atp_games import run_cross_play

result = await run_cross_play(
    game=game,
    agents=agents,
    config=config,
)
# result contains per-pair payoff statistics
```

### Stress Testing

Test agent robustness against best-response oracles:

```python
from atp_games import run_stress_test

result = await run_stress_test(
    game=game,
    agent=agent_adapter,
    config=config,
)
print(f"Exploitability under stress: {result.exploitability}")
```

## Architecture

```
atp_games/
├── models.py              # GameResult, EpisodeResult, PlayerStats, comparisons
├── plugin.py              # ATP plugin registration
├── mapping/
│   ├── observation_mapper.py  # Observation → ATPRequest
│   └── action_mapper.py      # ATPResponse → GameAction
├── runner/
│   ├── game_runner.py     # GameRunner orchestrator
│   ├── action_validator.py # Validation with retry logic
│   └── builtin_adapter.py # Wraps Strategy as ATP adapter
├── evaluators/
│   ├── payoff_evaluator.py
│   ├── exploitability_evaluator.py
│   ├── cooperation_evaluator.py
│   └── equilibrium_evaluator.py
└── suites/
    ├── models.py          # GameSuiteConfig, GameAgentConfig
    ├── game_suite_loader.py  # YAML parser with inheritance
    ├── schema.py          # JSON Schema validation
    ├── tournament.py      # Round-robin, elimination
    ├── cross_play.py      # Agent comparison matrix
    ├── stress_test.py     # Adversarial testing
    └── builtin/           # Built-in suite YAMLs
        ├── prisoners_dilemma.yaml
        └── auction_battery.yaml
```

### Data Flow

```
YAML Suite → GameSuiteLoader → Game + Agents (from registries)
                                       ↓
                                  GameRunner.run_game()
                                       ↓
                              Per-Episode Loop:
                                Game.observe() → Observation
                                ObservationMapper → ATPRequest
                                AgentAdapter.execute() → ATPResponse
                                ActionMapper → GameAction
                                ActionValidator → validated action
                                Game.step(actions) → StepResult
                                       ↓
                              GameResult (aggregated)
                                       ↓
                              Evaluators → EvalResult → Score
```

## Development

```bash
cd atp-games

# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/ -v --cov=atp_games

# Format and lint
uv run ruff format .
uv run ruff check .
```

## License

MIT License -- see the parent project's [LICENSE](../LICENSE) for details.
