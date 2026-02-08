# Phase 5: Game-Theoretic Evaluation — Technical Design

> Architecture and detailed design for game-environments and atp-games packages
> Per ADR-002: Two-package architecture

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Status | Draft |
| Created | 2026-02-07 |
| Related | ADR-002, Phase 5 Requirements, Phase 5 Roadmap |

---

## 1. Design Principles

### DESIGN-P5-001: Game as First-Class Environment

Игра — это не тест. Игра — это **среда**, в которой агенты взаимодействуют. Тест — это комбинация: game + agents + evaluation criteria + N episodes.

Следствия:
- Game отделена от evaluation logic
- Одна игра используется с разными evaluators
- Games composable: tournament = sequence of games

### DESIGN-P5-002: LLM-First Observation/Action Format

Observations и actions проектируются для LLM-агентов в первую очередь: текстовые описания, JSON-serializable, human-readable.

Следствия:
- Observation включает текстовое описание состояния
- Action space описывается текстом + schema
- Error messages при invalid actions understandable for LLM retry

### DESIGN-P5-003: Analytical Verifiability

Каждая игра и каждый solver должны быть верифицируемы аналитически на известных примерах.

Следствия:
- Payoff matrices проверяемы вручную
- Nash solver тестируется на играх с известными решениями
- Property-based tests для инвариантов

### DESIGN-P5-004: Progressive Complexity

Минимальная конфигурация для быстрого старта, расширяемость для сложных сценариев.

```python
# Minimal: 3 lines to run a game
game = PrisonersDilemma()
result = play(game, [AlwaysCooperate(), AlwaysDefect()])
print(result.payoffs)

# Full: tournament with LLM agents, metrics, reporting
# ... (via YAML suite + ATP CLI)
```

---

## 2. Package Architecture

### 2.1 game-environments Package

```
game-environments/
├── pyproject.toml
├── README.md
├── docs/
│   ├── api-reference.md
│   ├── game-development-guide.md
│   └── examples/
│       ├── basic_usage.ipynb
│       ├── custom_game.ipynb
│       └── population_dynamics.ipynb
├── game_envs/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── game.py             # Game ABC, GameConfig
│   │   ├── state.py            # GameState, Observation, Round
│   │   ├── action.py           # Action, ActionSpace variants
│   │   ├── payoff.py           # PayoffMatrix, PayoffVector
│   │   ├── history.py          # GameHistory, InformationSet
│   │   ├── communication.py    # CommChannel, Message
│   │   ├── player.py           # Player, PlayerRole
│   │   └── strategy.py         # Strategy ABC, built-in strategies
│   ├── games/
│   │   ├── __init__.py
│   │   ├── registry.py         # Game registry (name → class)
│   │   ├── prisoners_dilemma.py
│   │   ├── public_goods.py
│   │   ├── auction.py          # First-price, Second-price
│   │   ├── blotto.py           # Colonel Blotto
│   │   └── congestion.py       # Congestion / routing game
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── pd_strategies.py    # TFT, AllC, AllD, Grim, Pavlov
│   │   ├── pg_strategies.py    # FullContributor, FreeRider, etc.
│   │   ├── auction_strategies.py
│   │   ├── blotto_strategies.py
│   │   └── congestion_strategies.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── nash_solver.py      # Lemke-Howson, support enum, fictitious play
│   │   ├── exploitability.py   # Best response, exploitability score
│   │   ├── cooperation.py      # Cooperation rate, reciprocity, conditional coop
│   │   ├── fairness.py         # Gini, envy-freeness, proportionality
│   │   ├── evolutionary.py     # Replicator dynamics, ESS, Moran process
│   │   └── population.py       # Population simulator
│   └── utils/
│       ├── __init__.py
│       ├── play.py             # High-level play() function
│       ├── serialization.py    # JSON/dict serialization
│       └── visualization.py    # Matplotlib helpers (optional dep)
└── tests/
    ├── conftest.py
    ├── test_core/
    ├── test_games/
    ├── test_strategies/
    ├── test_analysis/
    └── test_integration/
```

### 2.2 atp-games Plugin

```
atp-games/
├── pyproject.toml
├── README.md
├── atp_games/
│   ├── __init__.py
│   ├── plugin.py              # ATP plugin registration
│   ├── runner/
│   │   ├── __init__.py
│   │   ├── game_runner.py     # GameRunner extends BaseRunner
│   │   ├── game_loop.py       # Core game loop logic
│   │   ├── action_validator.py # Validate + retry for LLM agents
│   │   └── builtin_adapter.py # Adapter for built-in strategies
│   ├── mapping/
│   │   ├── __init__.py
│   │   ├── observation_mapper.py  # GameObservation → ATPRequest
│   │   └── action_mapper.py      # ATPResponse → GameAction
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── payoff_evaluator.py
│   │   ├── exploitability_evaluator.py
│   │   ├── cooperation_evaluator.py
│   │   ├── fairness_evaluator.py
│   │   └── equilibrium_evaluator.py
│   ├── suites/
│   │   ├── __init__.py
│   │   ├── game_suite_loader.py   # YAML parser for game suites
│   │   ├── tournament.py          # Round-robin, elimination
│   │   ├── cross_play.py          # All-vs-all matrix
│   │   └── builtin/               # Pre-built YAML suites
│   │       ├── prisoners_dilemma.yaml
│   │       ├── auction_battery.yaml
│   │       └── alympics_lite.yaml
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── game_reporter.py       # Extends ATP reporter
│   │   ├── payoff_matrix_view.py
│   │   ├── strategy_chart.py
│   │   └── heatmap.py
│   └── dashboard/
│       ├── __init__.py
│       ├── routes.py              # New dashboard routes
│       └── templates/
│           ├── game_results.html
│           ├── tournament.html
│           └── components/
│               ├── payoff_matrix.html
│               └── strategy_timeline.html
└── tests/
```

---

## 3. Core Data Models

### 3.1 game-environments Models

```python
# game_envs/core/game.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

class GameType(Enum):
    NORMAL_FORM = "normal_form"
    EXTENSIVE_FORM = "extensive_form"
    REPEATED = "repeated"
    STOCHASTIC = "stochastic"

class MoveOrder(Enum):
    SIMULTANEOUS = "simultaneous"
    SEQUENTIAL = "sequential"

@dataclass(frozen=True)
class GameConfig:
    """Immutable game configuration."""
    num_players: int = 2
    num_rounds: int = 1               # 1 = one-shot
    discount_factor: float = 1.0      # For repeated games
    noise: float = 0.0                # Action noise probability
    communication: bool = False
    seed: int | None = None

class Game(ABC):
    """Abstract base for all games."""

    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self._rng = numpy.random.default_rng(self.config.seed)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def game_type(self) -> GameType: ...

    @property
    @abstractmethod
    def move_order(self) -> MoveOrder: ...

    @property
    @abstractmethod
    def player_ids(self) -> list[str]: ...

    @abstractmethod
    def action_space(self, player_id: str) -> ActionSpace: ...

    @abstractmethod
    def reset(self) -> GameState: ...

    @abstractmethod
    def step(self, actions: dict[str, Action]) -> StepResult: ...

    @abstractmethod
    def get_payoffs(self) -> dict[str, float]: ...

    @property
    @abstractmethod
    def is_terminal(self) -> bool: ...

    def observe(self, player_id: str) -> Observation:
        """Default: full observability. Override for partial."""
        return Observation(
            player_id=player_id,
            game_state=self._state.to_dict(),
            available_actions=self.action_space(player_id).to_list(),
            history=self._history.for_player(player_id),
            round_number=self._current_round,
            total_rounds=self.config.num_rounds,
        )
```

```python
# game_envs/core/state.py
@dataclass
class GameState:
    """Full game state (god's eye view)."""
    round_number: int
    player_states: dict[str, dict[str, Any]]
    public_state: dict[str, Any]
    is_terminal: bool = False

@dataclass
class Observation:
    """What a single player observes."""
    player_id: str
    game_state: dict[str, Any]       # Visible state only
    available_actions: list[Any]
    history: list[RoundResult]
    round_number: int
    total_rounds: int
    messages: list[Message] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Human-readable description for LLM agents."""
        ...

    def to_dict(self) -> dict:
        """JSON-serializable for ATP protocol."""
        ...

@dataclass
class RoundResult:
    """Result of a single round."""
    round_number: int
    actions: dict[str, Any]          # Visible actions only
    payoffs: dict[str, float]
    messages: list[Message] = field(default_factory=list)

@dataclass
class StepResult:
    """What step() returns."""
    state: GameState
    observations: dict[str, Observation]
    payoffs: dict[str, float]        # This round's payoffs
    is_terminal: bool
    info: dict[str, Any] = field(default_factory=dict)
```

```python
# game_envs/core/action.py
from abc import ABC, abstractmethod

class ActionSpace(ABC):
    @abstractmethod
    def contains(self, action: Any) -> bool: ...
    @abstractmethod
    def sample(self, rng=None) -> Any: ...
    @abstractmethod
    def to_list(self) -> list: ...
    @abstractmethod
    def to_description(self) -> str:
        """Human-readable for LLM agents."""
        ...

class DiscreteActionSpace(ActionSpace):
    def __init__(self, actions: list[str]):
        self.actions = actions

    def contains(self, action) -> bool:
        return action in self.actions

    def sample(self, rng=None) -> str:
        return (rng or numpy.random.default_rng()).choice(self.actions)

    def to_list(self) -> list[str]:
        return self.actions

    def to_description(self) -> str:
        return f"Choose one of: {', '.join(self.actions)}"

class ContinuousActionSpace(ActionSpace):
    def __init__(self, low: float, high: float, description: str = ""):
        self.low = low
        self.high = high
        self._description = description

class StructuredActionSpace(ActionSpace):
    """For games like Blotto where action is an allocation vector."""
    def __init__(self, schema: dict, description: str = ""):
        self.schema = schema
        self._description = description
```

```python
# game_envs/core/strategy.py
class Strategy(ABC):
    """Built-in strategy for baseline comparisons."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def choose_action(self, observation: Observation) -> Any: ...

    def reset(self) -> None:
        """Reset internal state between episodes."""
        pass
```

### 3.2 atp-games Models

```python
# atp_games/mapping/observation_mapper.py
from atp.protocol.models import ATPRequest, ATPContext
from game_envs.core.state import Observation

class ObservationMapper:
    """Maps GameObservation to ATPRequest."""

    def to_atp_request(
        self,
        observation: Observation,
        game_name: str,
        episode: int,
    ) -> ATPRequest:
        return ATPRequest(
            task=self._build_task_prompt(observation, game_name),
            context=ATPContext(
                environment={
                    "game": game_name,
                    "round": observation.round_number,
                    "total_rounds": observation.total_rounds,
                    "player_id": observation.player_id,
                },
                data={
                    "game_state": observation.game_state,
                    "available_actions": observation.available_actions,
                    "history": [r.__dict__ for r in observation.history],
                    "messages": [m.__dict__ for m in observation.messages],
                },
            ),
            constraints={
                "response_format": {
                    "action": "required",
                    "message": "optional",
                    "reasoning": "optional",
                },
            },
            metadata={
                "game_type": "game_theoretic",
                "episode": episode,
            },
        )

    def _build_task_prompt(self, obs: Observation, game_name: str) -> str:
        """Build LLM-friendly task description."""
        return (
            f"You are playing {game_name}. "
            f"Round {obs.round_number}/{obs.total_rounds}.\n\n"
            f"{obs.to_prompt()}\n\n"
            f"Respond with a JSON object: "
            f'{{"action": <your choice>, "message": "<optional>", '
            f'"reasoning": "<optional>"}}'
        )
```

```python
# atp_games/runner/game_runner.py
from atp.runner.base import BaseRunner
from game_envs.core.game import Game
from game_envs.analysis import exploitability, cooperation

class GameRunner(BaseRunner):
    """Orchestrates multi-agent game execution through ATP pipeline."""

    async def run_game(
        self,
        game: Game,
        agents: dict[str, AgentAdapter],
        config: GameRunConfig,
    ) -> GameResult:
        results = []

        for episode in range(config.episodes):
            state = game.reset()
            episode_history = []

            while not game.is_terminal:
                # Get observations for each player
                observations = {
                    pid: game.observe(pid) for pid in game.player_ids
                }

                # Convert to ATP requests and send to agents
                actions = {}
                if game.move_order == MoveOrder.SIMULTANEOUS:
                    actions = await self._parallel_moves(
                        observations, agents, game, episode
                    )
                else:
                    actions = await self._sequential_moves(
                        observations, agents, game, episode
                    )

                # Step the game
                step_result = game.step(actions)
                episode_history.append(step_result)

            results.append(EpisodeResult(
                episode=episode,
                payoffs=game.get_payoffs(),
                history=episode_history,
            ))

        return GameResult(
            game_name=game.name,
            config=config,
            episodes=results,
        )

    async def _parallel_moves(self, observations, agents, game, episode):
        """Send requests to all agents in parallel."""
        tasks = {}
        for pid, obs in observations.items():
            request = self.mapper.to_atp_request(obs, game.name, episode)
            tasks[pid] = self._get_validated_action(
                agents[pid], request, game.action_space(pid)
            )
        return await asyncio.gather_dict(tasks)

    async def _get_validated_action(self, agent, request, action_space):
        """Get action from agent with validation and retry."""
        for attempt in range(self.max_retries + 1):
            response = await agent.execute(request)
            action = self.action_mapper.from_atp_response(response)

            if action_space.contains(action.action):
                return action

            if attempt < self.max_retries:
                request = self._add_error_context(
                    request,
                    f"Invalid action '{action.action}'. "
                    f"Valid: {action_space.to_description()}"
                )

        # Fallback to default
        return GameAction(action=action_space.sample(game._rng))
```

---

## 4. Game Implementations

### 4.1 Prisoner's Dilemma

```python
# game_envs/games/prisoners_dilemma.py

@dataclass(frozen=True)
class PDConfig(GameConfig):
    """Prisoner's Dilemma specific config."""
    # Payoff matrix: (R, S, T, P) where T > R > P > S
    reward: float = 3.0       # R: mutual cooperation
    sucker: float = 0.0       # S: cooperate vs defect
    temptation: float = 5.0   # T: defect vs cooperate
    punishment: float = 1.0   # P: mutual defection

class PrisonersDilemma(Game):
    """Classic Prisoner's Dilemma (one-shot or repeated)."""

    COOPERATE = "cooperate"
    DEFECT = "defect"

    @property
    def name(self) -> str:
        suffix = f" (repeated x{self.config.num_rounds})" if self.config.num_rounds > 1 else ""
        return f"Prisoner's Dilemma{suffix}"

    def action_space(self, player_id: str) -> DiscreteActionSpace:
        return DiscreteActionSpace([self.COOPERATE, self.DEFECT])

    def step(self, actions: dict[str, str]) -> StepResult:
        a1, a2 = actions[self.player_ids[0]], actions[self.player_ids[1]]

        # Apply noise
        if self.config.noise > 0:
            if self._rng.random() < self.config.noise:
                a1 = self.DEFECT if a1 == self.COOPERATE else self.COOPERATE
            if self._rng.random() < self.config.noise:
                a2 = self.DEFECT if a2 == self.COOPERATE else self.COOPERATE

        payoffs = self._compute_payoffs(a1, a2)
        ...

    def _compute_payoffs(self, a1: str, a2: str) -> dict[str, float]:
        c = self.config
        matrix = {
            ("cooperate", "cooperate"): (c.reward, c.reward),
            ("cooperate", "defect"):    (c.sucker, c.temptation),
            ("defect", "cooperate"):    (c.temptation, c.sucker),
            ("defect", "defect"):       (c.punishment, c.punishment),
        }
        p1, p2 = matrix[(a1, a2)]
        return {self.player_ids[0]: p1, self.player_ids[1]: p2}
```

### 4.2 Auction (First-Price / Second-Price)

```python
# game_envs/games/auction.py

class AuctionType(Enum):
    FIRST_PRICE = "first_price"
    SECOND_PRICE = "second_price"   # Vickrey

@dataclass(frozen=True)
class AuctionConfig(GameConfig):
    auction_type: AuctionType = AuctionType.SECOND_PRICE
    num_items: int = 1
    min_bid: float = 0.0
    max_bid: float = 100.0
    reserve_price: float = 0.0
    value_distribution: str = "uniform"  # How private values are drawn
    value_min: float = 0.0
    value_max: float = 100.0

class Auction(Game):
    """Sealed-bid auction with private values."""

    def reset(self) -> GameState:
        # Draw private values for each player
        self._values = {
            pid: self._rng.uniform(
                self.config.value_min, self.config.value_max
            )
            for pid in self.player_ids
        }
        ...

    def observe(self, player_id: str) -> Observation:
        """Player sees own value but not others'."""
        return Observation(
            player_id=player_id,
            game_state={
                "your_value": self._values[player_id],
                "num_bidders": self.config.num_players,
                "auction_type": self.config.auction_type.value,
                "reserve_price": self.config.reserve_price,
            },
            available_actions=[f"bid between {self.config.min_bid} and {self.config.max_bid}"],
            ...
        )

    def action_space(self, player_id: str) -> ContinuousActionSpace:
        return ContinuousActionSpace(
            self.config.min_bid, self.config.max_bid,
            description=f"Submit a bid between {self.config.min_bid} and {self.config.max_bid}"
        )
```

### 4.3 Colonel Blotto, Public Goods, Congestion — аналогичная структура

Каждая игра: своя конфигурация, action space, payoff logic, observe() с partial observability.

---

## 5. Analysis Components

### 5.1 Nash Solver

```python
# game_envs/analysis/nash_solver.py

class NashSolver:
    """Compute Nash equilibria for normal-form games."""

    @staticmethod
    def solve_2player(
        payoff_matrix_1: np.ndarray,  # (m x n)
        payoff_matrix_2: np.ndarray,  # (m x n)
        method: str = "support_enumeration",
    ) -> list[NashEquilibrium]:
        """
        Find Nash equilibria for 2-player game.

        Methods:
        - support_enumeration: Exact, finds all NE for small games
        - lemke_howson: Exact, finds one NE efficiently
        - vertex_enumeration: Exact, for degenerate games
        """
        ...

    @staticmethod
    def solve_nplayer(
        payoff_tensors: list[np.ndarray],
        method: str = "fictitious_play",
        max_iterations: int = 10000,
        epsilon: float = 0.01,
    ) -> NashEquilibrium:
        """
        Approximate Nash for n-player games.

        Methods:
        - fictitious_play: Iterate best responses to empirical frequencies
        - replicator_dynamics: Continuous-time evolutionary dynamics
        """
        ...

@dataclass
class NashEquilibrium:
    """Represents a Nash equilibrium."""
    strategies: dict[str, np.ndarray]   # Player → mixed strategy
    payoffs: dict[str, float]           # Expected payoffs
    support: dict[str, list[int]]       # Actions with positive probability
    epsilon: float = 0.0                # ε-Nash: how approximate
```

### 5.2 Exploitability

```python
# game_envs/analysis/exploitability.py

def compute_exploitability(
    game: Game,
    empirical_strategy: dict[str, EmpiricalStrategy],
) -> ExploitabilityResult:
    """
    Compute exploitability of an empirical strategy profile.

    For each player:
    1. Fix all other players to their empirical strategies
    2. Compute best response for this player
    3. Exploitability = BR payoff - current payoff

    Total exploitability = sum over players.
    """
    ...

@dataclass
class EmpiricalStrategy:
    """Strategy derived from observed play history."""
    action_frequencies: dict[str, float]  # Action → frequency

    @classmethod
    def from_history(cls, history: list[RoundResult], player_id: str) -> "EmpiricalStrategy":
        ...

@dataclass
class ExploitabilityResult:
    per_player: dict[str, float]       # Exploitability per player
    total: float                        # Sum
    best_responses: dict[str, Any]     # What BR would do
```

---

## 6. Game Suite YAML Schema

```yaml
# JSON Schema for game_suite YAML files
type: object
required: [type, name, game, agents, evaluation]
properties:
  type:
    const: game_suite
  name:
    type: string
  version:
    type: string

  game:
    type: object
    required: [type]
    properties:
      type:
        enum: [prisoners_dilemma, public_goods, auction, blotto, congestion]
      variant:
        enum: [one_shot, repeated]
        default: one_shot
      config:
        type: object
        properties:
          num_rounds:    { type: integer, default: 1 }
          noise:         { type: number, default: 0.0 }
          discount_factor: { type: number, default: 1.0 }
          communication: { type: boolean, default: false }
          seed:          { type: integer }
          # Game-specific config merged here

  agents:
    type: array
    minItems: 2
    items:
      type: object
      required: [name]
      properties:
        name:     { type: string }
        adapter:  { enum: [http, docker, cli, builtin] }
        endpoint: { type: string }
        strategy: { type: string }  # For builtin adapter
        config:   { type: object }

  evaluation:
    type: object
    properties:
      episodes:
        type: integer
        default: 50
      metrics:
        type: array
        items:
          type: object
          required: [type]
          properties:
            type:
              enum: [average_payoff, cooperation_rate, exploitability,
                     nash_distance, fairness, equilibrium_type]
            weight: { type: number, default: 1.0 }
            config: { type: object }
      thresholds:
        type: object
        additionalProperties:
          type: object
          properties:
            min:  { type: number }
            max:  { type: number }
            warn: { type: number }

  reporting:
    type: object
    properties:
      include_strategy_profile: { type: boolean, default: true }
      include_payoff_matrix:    { type: boolean, default: true }
      include_round_by_round:   { type: boolean, default: false }
      export_formats:           { type: array, items: { type: string } }
```

---

## 7. Integration Points with ATP

### 7.1 Plugin Registration

```python
# atp_games/plugin.py
from atp.plugins import PluginRegistry

def register():
    """Called by ATP plugin discovery."""
    PluginRegistry.register_runner("game", GameRunner)
    PluginRegistry.register_suite_loader("game_suite", GameSuiteLoader)
    PluginRegistry.register_evaluator("payoff", PayoffEvaluator)
    PluginRegistry.register_evaluator("exploitability", ExploitabilityEvaluator)
    PluginRegistry.register_evaluator("cooperation", CooperationEvaluator)
    PluginRegistry.register_evaluator("fairness", FairnessEvaluator)
    PluginRegistry.register_evaluator("equilibrium", EquilibriumEvaluator)
    PluginRegistry.register_reporter("game", GameReporter)
    PluginRegistry.register_dashboard_routes("game", game_dashboard_routes)
```

### 7.2 CLI Extensions

```bash
# New commands (additive)
atp game list                                    # List available games
atp game info prisoners_dilemma                  # Show game details
atp game play prisoners_dilemma --agents a,b     # Quick play
atp tournament --suite=pd_tournament.yaml        # Run tournament
atp crossplay --suite=auction_crossplay.yaml     # Cross-play matrix

# Existing commands work with game suites
atp test --suite=game:prisoners_dilemma.yaml     # Standard test flow
atp compare --suites=game:pd.yaml --agents=a,b,c # Comparison with game metrics
```

### 7.3 Dashboard Routes

| Route | View |
|-------|------|
| `/games/` | Game results overview |
| `/games/{result_id}` | Detailed game result |
| `/games/{result_id}/payoff-matrix` | Payoff matrix visualization |
| `/games/{result_id}/strategy-timeline` | Strategy changes over rounds |
| `/tournaments/{id}` | Tournament standings and brackets |
| `/crossplay/{id}` | Cross-play heatmap |

---

## 8. Data Flow

```
User: atp test --suite=game:prisoners_dilemma.yaml
                    │
                    ▼
            ┌───────────────┐
            │  CLI Parser   │  Detects type: game_suite
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │ GameSuiteLoader│  Parses YAML, creates Game + AgentConfigs
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │  GameRunner   │  Orchestrates game loop
            └───────┬───────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │ Game   │ │Observe │ │Observe │   game-environments
    │.reset()│ │Player 1│ │Player 2│
    └───┬────┘ └───┬────┘ └───┬────┘
        │          │          │
        │          ▼          ▼
        │    ┌─────────┐ ┌─────────┐
        │    │Obs→ATP  │ │Obs→ATP  │   atp-games mapper
        │    │Request  │ │Request  │
        │    └────┬────┘ └────┬────┘
        │         │           │
        │         ▼           ▼
        │    ┌─────────┐ ┌─────────┐
        │    │ HTTP     │ │ Builtin │   ATP adapters
        │    │ Adapter  │ │ Adapter │
        │    └────┬────┘ └────┬────┘
        │         │           │
        │         ▼           ▼
        │    ┌─────────┐ ┌─────────┐
        │    │ LLM     │ │  TFT    │   Agents
        │    │ Agent   │ │Strategy │
        │    └────┬────┘ └────┬────┘
        │         │           │
        │         ▼           ▼
        │    ┌─────────┐ ┌─────────┐
        │    │ATP Resp │ │ Action  │   Validate actions
        │    │→Action  │ │validate │
        │    └────┬────┘ └────┬────┘
        │         │           │
        │    ┌────┴───────────┘
        │    │
        ▼    ▼
    ┌────────────┐
    │ Game.step  │  Execute round, compute payoffs
    │ (actions)  │
    └────┬───────┘
         │
         │  Repeat until terminal
         ▼
    ┌────────────┐
    │ GameResult │  All episodes complete
    └────┬───────┘
         │
    ┌────┴──────────────────────┐
    │        Evaluators         │
    ├───────────────────────────┤
    │ PayoffEvaluator           │
    │ ExploitabilityEvaluator   │
    │ CooperationEvaluator      │
    │ FairnessEvaluator         │
    │ EquilibriumEvaluator      │
    └────┬──────────────────────┘
         │
         ▼
    ┌────────────┐
    │  ATP       │  Standard reporting pipeline
    │ Reporters  │  JSON + HTML + Dashboard
    └────────────┘
```

---

## 9. Migration & Backward Compatibility

### Impact on Existing Code

| Component | Change | Risk |
|-----------|--------|------|
| Protocol models | No change | None |
| Adapters | No change (game requests use existing format) | None |
| Runner | New GameRunner alongside existing | None |
| Evaluators | New evaluators registered via plugin | None |
| Suite loader | New type `game_suite` added | None |
| CLI | New commands added | None |
| Dashboard | New routes added | None (existing routes unchanged) |

### Plugin Discovery

```toml
# atp-games/pyproject.toml
[project.entry-points."atp.plugins"]
game = "atp_games.plugin:register"
```

ATP discovers plugins via entry points. No changes to ATP core required.

---

## 10. Testing Strategy

### game-environments Tests

| Category | What | How |
|----------|------|-----|
| Unit | Game logic, payoff computation | pytest, known payoff matrices |
| Analytical | Nash solver correctness | Compare with known equilibria |
| Property-based | Invariants (payoffs sum, action containment) | Hypothesis |
| Integration | Full game play with strategies | End-to-end game episodes |

### atp-games Tests

| Category | What | How |
|----------|------|-----|
| Unit | Mappers, validators, evaluators | pytest, mock game results |
| Integration | GameRunner with mock agents | pytest-asyncio |
| E2E | `atp test --suite=game:...` | subprocess, real CLI |
| Contract | YAML schema validation | jsonschema |

---

## 11. Open Design Questions

1. **Concurrent game instances**: Should GameRunner support running multiple independent game instances simultaneously for different matchups in a tournament?
   - **Proposed**: Yes, via asyncio task pool with configurable concurrency.

2. **LLM prompt format**: Should observation-to-prompt conversion be customizable?
   - **Proposed**: Default prompt builder with override via YAML config (`prompt_template`).

3. **State persistence**: Should game states be saved to DB for replay?
   - **Proposed**: Phase 5.1 — file-based traces (JSON). Phase 5.2 — DB storage in dashboard.

4. **Mixed agent types**: Can one game have LLM agent + coded strategy + RL agent?
   - **Proposed**: Yes. `builtin` adapter wraps Strategy, HTTP/Docker/CLI handle external agents.
