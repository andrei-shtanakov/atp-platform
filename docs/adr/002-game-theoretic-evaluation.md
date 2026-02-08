# ADR-002: Game-Theoretic Evaluation Architecture

**Status**: Accepted
**Date**: 2026-02-07
**Decision Makers**: Architecture Team
**Supersedes**: —
**Related**: ADR-001 (Framework Agnostic Design)

## Context

ATP Platform (Phases 1–3 complete, Phase 4 in progress) covers single-agent testing and basic multi-agent modes (comparison, collaboration, handoff). However, there is a growing need to evaluate agents in **strategic interaction** scenarios where:

- Multiple agents make decisions that **directly affect each other's payoffs**
- Agent quality is measured not against a ground truth, but against **game-theoretic solution concepts** (Nash equilibrium, exploitability, Pareto optimality)
- Evaluation reveals emergent properties: cooperation dynamics, bias, fairness, strategic reasoning capability

This is driven by several research directions:
- MARL formalization as stochastic games [1][2]
- LLM-agents as pseudo-players in strategic games (Alympics, FAIRGAME) [6][7][8]
- Game-theoretic evaluation as a benchmarking methodology [10]
- Agent-based modeling combined with game theory for norm emergence and mechanism design [3][9]

### Problem with Current Architecture

The existing `MultiAgentOrchestrator` (TASK-601–604) handles agents working **together or in comparison**. It does not support:

1. **Simultaneous/sequential strategic moves** with payoff dependencies
2. **Game rules** as a first-class concept (payoff matrices, action spaces, information sets)
3. **Equilibrium-based metrics** (Nash distance, exploitability, best response analysis)
4. **Population dynamics** (evolutionary stability, strategy distribution over time)
5. **Repeated game mechanics** (history-dependent strategies, discount factors)

## Decision

We adopt a **two-package architecture**:

1. **`game-environments`** — standalone library of game environments and analysis tools, with no dependency on ATP
2. **`atp-games`** — ATP plugin that bridges game-environments into the ATP evaluation pipeline

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ATP Platform                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    atp-games plugin                       │   │
│  │                                                           │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │   │
│  │  │ GameRunner   │  │   Game       │  │  Game Suite    │  │   │
│  │  │ (extends     │  │  Evaluators  │  │  Loader        │  │   │
│  │  │  Runner)     │  │  (new type)  │  │  (YAML ext)    │  │   │
│  │  └──────┬───────┘  └──────┬───────┘  └───────┬────────┘  │   │
│  │         │                 │                   │           │   │
│  └─────────┼─────────────────┼───────────────────┼───────────┘   │
│            │                 │                   │               │
│  ┌─────────▼─────────────────▼───────────────────▼───────────┐   │
│  │              ATP Core (Protocol, Adapters, Reporters)      │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │
         │  uses (import, no ATP dependency)
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     game-environments                            │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────┐  │
│  │    Games       │  │    Core       │  │    Analysis         │  │
│  │               │  │               │  │                     │  │
│  │ prisoners_    │  │ base_game.py  │  │ nash_solver.py      │  │
│  │  dilemma.py  │  │ action_space  │  │ exploitability.py   │  │
│  │ public_      │  │ payoff.py     │  │ cooperation.py      │  │
│  │  goods.py    │  │ history.py    │  │ fairness.py         │  │
│  │ auction.py   │  │ info_set.py   │  │ evolutionary.py     │  │
│  │ blotto.py    │  │ comm_channel  │  │ population.py       │  │
│  │ congestion   │  │               │  │                     │  │
│  │  .py         │  │               │  │                     │  │
│  └───────────────┘  └───────────────┘  └─────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### 1. Game as Environment, not Test

A game is **not** a test case — it's an **environment** in which agents interact. The test is the combination of game + agents + evaluation criteria + number of episodes. This distinction matters because:

- The same game can be used with different evaluation criteria
- Games can be composed (tournament = sequence of games)
- Games have their own lifecycle (setup → rounds → termination)

```python
# game-environments: pure game logic
class Game(ABC):
    @abstractmethod
    def reset(self) -> GameState: ...

    @abstractmethod
    def step(self, actions: dict[str, Action]) -> GameState: ...

    @abstractmethod
    def get_payoffs(self) -> dict[str, float]: ...

    @property
    @abstractmethod
    def action_space(self) -> dict[str, ActionSpace]: ...

    @property
    @abstractmethod
    def is_terminal(self) -> bool: ...
```

```python
# atp-games: bridges Game into ATP
class GameRunner(BaseRunner):
    """Extends ATP Runner with game loop orchestration."""

    async def run_game(
        self,
        game: Game,
        agents: dict[str, AgentAdapter],
        config: GameConfig,
    ) -> GameResult: ...
```

#### 2. Agent Interface for Games

Agents interact with games through a **minimal extension** of the ATP Protocol:

```python
@dataclass
class GameObservation:
    """What agent sees at each step."""
    game_state: dict          # Observable state
    available_actions: list   # Legal actions
    history: list[Round]      # Past rounds (if observable)
    message: str | None       # Communication from other agents

@dataclass
class GameAction:
    """What agent returns."""
    action: str | int | dict  # Chosen action
    message: str | None       # Optional communication
    reasoning: str | None     # Optional explanation (for analysis)
```

This maps cleanly to ATP Request/Response:
- `GameObservation` → enriched `ATPRequest.context`
- `GameAction` → structured `ATPResponse.artifacts`

#### 3. Separation of Concerns for Metrics

| Layer | Responsibility | Examples |
|-------|---------------|----------|
| `game-environments` | Game-theoretic calculations | Nash equilibrium solver, payoff analysis, exploitability computation |
| `atp-games` evaluators | Evaluation against criteria | "Is agent within ε of Nash?", "Cooperation rate > 60%?", "No bias detected?" |
| ATP reporters | Presentation | Payoff matrices in HTML report, strategy distribution charts |

#### 4. Game Suite YAML Format

```yaml
# suites/game-theoretic/prisoners-dilemma.yaml
type: game_suite
name: "Prisoner's Dilemma Evaluation"
version: "1.0"

game:
  type: prisoners_dilemma
  variant: repeated
  config:
    rounds: 100
    noise: 0.05              # Action noise probability
    discount_factor: 0.95
    communication: false

agents:
  - name: agent-under-test
    adapter: http
    endpoint: "http://localhost:8001"
  - name: baseline-tft
    adapter: builtin
    strategy: tit_for_tat    # Built-in baseline strategy

evaluation:
  episodes: 50               # Statistical significance
  metrics:
    - type: average_payoff
      weight: 0.3
    - type: cooperation_rate
      weight: 0.2
    - type: exploitability
      weight: 0.3
      config:
        method: best_response
    - type: nash_distance
      weight: 0.2

  thresholds:
    cooperation_rate:
      min: 0.4
      warn: 0.6
    exploitability:
      max: 0.15

reporting:
  include_strategy_profile: true
  include_payoff_matrix: true
  include_round_by_round: false
```

#### 5. Built-in Baseline Strategies

`game-environments` ships with canonical strategies for each game, serving as baselines:

| Game | Baselines |
|------|-----------|
| Prisoner's Dilemma | Tit-for-Tat, Always Cooperate, Always Defect, Grim Trigger, Pavlov, Random |
| Public Goods | Full Contributor, Free Rider, Conditional Cooperator, Punisher |
| Auction | Truthful Bidder, Shade-50%, Random Bidder |
| Colonel Blotto | Uniform, Concentrated, Nash-mixed |
| Congestion | Selfish Router, Social Optimum, ε-greedy |

These are essential for exploitability calculation (best response) and for providing meaningful comparison baselines.

## Consequences

### Positive

- **Modularity**: `game-environments` is reusable beyond ATP (research, MARL training, education)
- **Natural extension**: ATP plugin system already supports new evaluators and runners
- **Backward compatible**: Existing tests, adapters, and reports are unaffected
- **Principled evaluation**: Game-theoretic metrics are mathematically grounded, not ad-hoc
- **Scalable game library**: New games are added to `game-environments` without touching ATP
- **Different lifecycles**: Game library evolves independently from ATP core

### Negative

- **Two packages to maintain**: Increased maintenance overhead
- **Nash computation complexity**: Exact Nash for general-sum games is PPAD-complete; need approximation algorithms
- **LLM agent latency**: Game loops with LLM agents are slow (seconds per round vs microseconds for classical agents)
- **Stochastic evaluation**: Games with randomized strategies require many episodes for statistical validity

### Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Nash solver fails for complex games | High | Use approximate methods (support / Lemke-Howson for 2-player, fictitious play for n-player) |
| LLM agents don't follow game rules | Medium | Validate actions against legal action space; retry with clarification |
| Evaluation is too slow | Medium | Parallel episode execution; caching; async game loops |
| Game library grows unwieldy | Low | Categorization (classic, auction, negotiation, social dilemma); community contributions with review |

## Alternatives Considered

### 1. Monolithic: Everything Inside ATP

Add games, solvers, and evaluators directly into `atp/` module.

**Pros**: Single codebase, simpler dependency management
**Cons**: Game logic tightly coupled to ATP; can't use games without ATP; different change rates create friction

**Rejected**: Violates single responsibility; limits reusability.

### 2. Fully Separate Platform

Build an independent "Game-Theoretic Agent Benchmark" platform.

**Pros**: Clean slate, optimized for games
**Cons**: Duplicates ATP infrastructure (adapters, reporters, CLI, dashboard, CI/CD); two systems to learn

**Rejected**: ATP already provides 80% of needed infrastructure.

### 3. Fork Existing MARL Framework

Use PettingZoo / OpenSpiel as game environment layer.

**Pros**: Battle-tested; large game library
**Cons**: Heavy dependencies (JAX, TensorFlow); designed for RL training, not evaluation; poor LLM agent support; Python-specific action spaces

**Considered partially**: We adopt PettingZoo's `AECEnv` interface pattern but implement our own lightweight version focused on evaluation use cases and LLM-compatible observation/action formats.

## Implementation Plan

See updated `docs/07-roadmap.md` → Phase 5: Game-Theoretic Evaluation.

## References

- [1] Game-Theoretic Multiagent Reinforcement Learning — https://arxiv.org/abs/2011.00583
- [2] Multi-Agent Reinforcement Learning in Games — https://pmc.ncbi.nlm.nih.gov/articles/PMC12190516/
- [3] Agent-Based Modeling and Game Theory — https://smythos.com/managers/legal/agent-based-modeling-and-game-theory/
- [6] FAIRGAME: Framework for AI Agents Bias Recognition — https://arxiv.org/abs/2504.14325
- [7] ALYMPICS: LLM Agents Meet Game Theory — https://arxiv.org/abs/2311.03220
- [8] Alympics v3 — https://arxiv.org/html/2311.03220v3
- [9] Agent-Based Modeling Simulating — https://smythos.com/managers/legal/agent-based-modeling-and-game-theory/
- [10] Game Theory Approaches for Autonomy — https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2022.880706/full
- PettingZoo AEC API — https://pettingzoo.farama.org/api/aec/
- OpenSpiel — https://github.com/google-deepmind/open_spiel
