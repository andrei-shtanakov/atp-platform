---
name: generate-game-tests
description: Generate game theory test scenarios — YAML game suites and pytest tests for game-environments and atp-games. Use when asked to create game tests, game scenarios, game suites, test game strategies, or evaluate game-theoretic properties. Triggers on "game test", "game scenario", "game suite", "test game", "тесты для игры", "игровой сценарий", "тестовый сценарий для теории игр", "prisoners dilemma test", "auction test", "blotto test".
---

# Generate Game Theory Test Scenarios

Generate YAML game suites for atp-games and pytest tests for game-environments, covering all 5 canonical games and 25+ strategies.

## Invocation

- **With argument:** `/generate-game-tests prisoners_dilemma` — generates suite + tests for the specified game
- **Without argument:** asks which game, strategies, and metrics to test

## Supported Games

| Game | Actions | Key Properties |
|------|---------|----------------|
| `prisoners_dilemma` | cooperate, defect | T > R > P > S, 2R > T+S |
| `public_goods` | contribution (0.0-1.0) | Free-rider advantage, alpha multiplier |
| `colonel_blotto` | allocation per battlefield | Sum=1.0 constraint, majority wins |
| `auction` | bid (continuous) | First/second price, truthful bidding |
| `congestion` | route selection | Load-dependent latency, Nash routing |

## Workflow

### Step 1: Determine Parameters

If game type is provided as argument, use defaults below. Otherwise ask:

1. "Which game?" — offer the 5 games above
2. "Which strategies to pit against each other?" — suggest relevant combos:
   - PD: tit_for_tat vs always_defect, pavlov vs grim_trigger
   - Public Goods: full_contributor vs free_rider
   - Auction: truthful_bidder vs shade_bidder
   - Blotto: uniform_allocation vs concentrated_allocation
   - Congestion: selfish_router vs social_optimum
3. "What metrics matter?" — payoff, exploitability, cooperation, fairness, equilibrium

### Step 2: Generate YAML Game Suite

Create file at `atp-games/suites/generated/{game_type}_{timestamp}.yaml`

#### Template: Prisoner's Dilemma
```yaml
type: game_suite
name: "Prisoner's Dilemma — {strategy_a} vs {strategy_b}"
version: "1.0"

game:
  type: prisoners_dilemma
  variant: repeated
  config:
    num_rounds: 100
    noise: 0.0
    discount_factor: 1.0

agents:
  - name: "{strategy_a}"
    adapter: builtin
    strategy: "{strategy_a}"

  - name: "{strategy_b}"
    adapter: builtin
    strategy: "{strategy_b}"

evaluation:
  episodes: 50
  metrics:
    - type: average_payoff
      weight: 1.0
    - type: cooperation_rate
      weight: 0.5
    - type: exploitability
      weight: 0.3
      config:
        epsilon: 0.15
  thresholds:
    average_payoff:
      min: 1.0
```

#### Template: Public Goods
```yaml
type: game_suite
name: "Public Goods — {n_players} players"
version: "1.0"

game:
  type: public_goods
  variant: repeated
  config:
    num_players: {n_players}
    num_rounds: 50
    initial_endowment: 10.0
    multiplier: 1.6

agents:
  - name: "contributor"
    adapter: builtin
    strategy: full_contributor
  - name: "free_rider"
    adapter: builtin
    strategy: free_rider

evaluation:
  episodes: 30
  metrics:
    - type: average_payoff
      weight: 1.0
    - type: fairness
      weight: 0.5
      config:
        metric: gini_coefficient
```

#### Template: Auction
```yaml
type: game_suite
name: "Auction — {auction_type} price"
version: "1.0"

game:
  type: auction
  variant: one_shot
  config:
    auction_type: {auction_type}  # first_price or second_price
    num_rounds: 100
    value_distribution: uniform

agents:
  - name: "truthful"
    adapter: builtin
    strategy: truthful_bidder
  - name: "strategic"
    adapter: builtin
    strategy: shade_bidder

evaluation:
  episodes: 100
  metrics:
    - type: average_payoff
      weight: 1.0
    - type: exploitability
      weight: 0.5
```

#### Template: Colonel Blotto
```yaml
type: game_suite
name: "Colonel Blotto — {n_battlefields} battlefields"
version: "1.0"

game:
  type: colonel_blotto
  variant: one_shot
  config:
    num_battlefields: {n_battlefields}
    num_rounds: 50
    tie_breaking: split

agents:
  - name: "uniform"
    adapter: builtin
    strategy: uniform_allocation
  - name: "concentrated"
    adapter: builtin
    strategy: concentrated_allocation

evaluation:
  episodes: 100
  metrics:
    - type: average_payoff
      weight: 1.0
```

#### Template: Congestion
```yaml
type: game_suite
name: "Congestion Game — {n_routes} routes"
version: "1.0"

game:
  type: congestion
  variant: repeated
  config:
    num_routes: {n_routes}
    num_rounds: 50
    latency_function: linear

agents:
  - name: "selfish"
    adapter: builtin
    strategy: selfish_router
  - name: "social"
    adapter: builtin
    strategy: social_optimum

evaluation:
  episodes: 50
  metrics:
    - type: average_payoff
      weight: 1.0
    - type: equilibrium
      weight: 0.5
```

### Step 3: Generate Pytest Tests

Create file at `tests/unit/games/test_{game_type}_scenarios.py`

#### Payoff Correctness Tests
```python
"""Tests for {game_type} game correctness and scenarios."""

from __future__ import annotations

import pytest

from game_envs.games.{game_module} import {GameClass}, {ConfigClass}


class Test{GameClass}PayoffStructure:
    """Verify payoff structure matches game theory."""

    def test_payoff_ordering(self) -> None:
        """Verify canonical payoff ordering holds."""
        cfg = {ConfigClass}()
        # Game-specific assertions, e.g. for PD:
        # assert cfg.temptation > cfg.reward > cfg.punishment > cfg.sucker
        ...

    def test_mutual_best_outcome(self) -> None:
        """Verify mutual cooperation/optimal outcome payoff."""
        game = {GameClass}()
        game.reset()
        result = game.step({mutual_best_actions})
        for player in result.payoffs:
            assert result.payoffs[player] == pytest.approx({expected})

    def test_mutual_worst_outcome(self) -> None:
        """Verify mutual defection/worst outcome payoff."""
        game = {GameClass}()
        game.reset()
        result = game.step({mutual_worst_actions})
        for player in result.payoffs:
            assert result.payoffs[player] == pytest.approx({expected})

    def test_asymmetric_outcome(self) -> None:
        """Verify exploiter vs exploited payoffs."""
        game = {GameClass}()
        game.reset()
        result = game.step({asymmetric_actions})
        assert result.payoffs["player_0"] > result.payoffs["player_1"]
```

#### Strategy Behavior Tests
```python
from game_envs.strategies.{strategy_module} import {StrategyClass}
from game_envs.core.base import Observation


class Test{StrategyClass}Behavior:
    """Verify strategy follows its documented behavior."""

    @pytest.fixture
    def strategy(self) -> {StrategyClass}:
        return {StrategyClass}()

    def test_first_move(self, strategy: {StrategyClass}) -> None:
        """Verify strategy's opening move."""
        obs = Observation(round_number=0, history=[])
        action = strategy.choose_action(obs)
        assert action == {expected_first_move}

    def test_response_to_cooperation(
        self, strategy: {StrategyClass}
    ) -> None:
        """Verify response when opponent cooperated."""
        obs = Observation(
            round_number=1,
            history=[{{"opponent": "cooperate"}}],
        )
        action = strategy.choose_action(obs)
        assert action == {expected_response_to_coop}

    def test_response_to_defection(
        self, strategy: {StrategyClass}
    ) -> None:
        """Verify response when opponent defected."""
        obs = Observation(
            round_number=1,
            history=[{{"opponent": "defect"}}],
        )
        action = strategy.choose_action(obs)
        assert action == {expected_response_to_defect}
```

#### Equilibrium Property Tests
```python
class Test{GameClass}EquilibriumProperties:
    """Verify game-theoretic equilibrium properties."""

    def test_nash_equilibrium_is_stable(self) -> None:
        """Verify no player benefits from unilateral deviation at NE."""
        game = {GameClass}()
        game.reset()
        # Play Nash equilibrium actions
        ne_result = game.step({nash_actions})
        # Try deviations for each player
        for player in game.players:
            for alt_action in game.action_space.actions:
                deviated = {{**{nash_actions}, player: alt_action}}
                game.reset()
                dev_result = game.step(deviated)
                assert dev_result.payoffs[player] <= ne_result.payoffs[player]

    def test_pareto_efficiency(self) -> None:
        """Verify Pareto-optimal outcomes exist."""
        game = {GameClass}()
        # Check that mutual cooperation Pareto-dominates mutual defection
        game.reset()
        coop = game.step({mutual_best_actions})
        game.reset()
        defect = game.step({mutual_worst_actions})
        total_coop = sum(coop.payoffs.values())
        total_defect = sum(defect.payoffs.values())
        assert total_coop > total_defect
```

#### Multi-Round Convergence Tests
```python
class Test{GameClass}MultiRound:
    """Verify multi-round game dynamics."""

    def test_strategy_convergence(self) -> None:
        """Verify strategies converge to expected behavior over rounds."""
        game = {GameClass}(config={ConfigClass}(num_rounds=100))
        game.reset()
        strategies = {{
            "player_0": {StrategyA}(),
            "player_1": {StrategyB}(),
        }}
        payoffs: dict[str, list[float]] = {{"player_0": [], "player_1": []}}
        for _ in range(100):
            obs = game.observe()
            actions = {{
                p: s.choose_action(obs[p])
                for p, s in strategies.items()
            }}
            result = game.step(actions)
            for p in payoffs:
                payoffs[p].append(result.payoffs[p])

        # Verify average payoff in expected range
        avg_0 = sum(payoffs["player_0"]) / len(payoffs["player_0"])
        assert {expected_range_low} <= avg_0 <= {expected_range_high}

    def test_noise_affects_outcomes(self) -> None:
        """Verify noise parameter creates variation in outcomes."""
        game_clean = {GameClass}(
            config={ConfigClass}(num_rounds=50, noise=0.0)
        )
        game_noisy = {GameClass}(
            config={ConfigClass}(num_rounds=50, noise=0.1)
        )
        # Run both and compare variance
        ...
```

### Step 4: Game-Specific Validation Rules

When generating tests, apply these domain-specific checks:

**Prisoner's Dilemma:**
- Payoff ordering: T(5.0) > R(3.0) > P(1.0) > S(0.0)
- Cooperation sustainability: 2R > T + S
- TitForTat must cooperate first, then mirror
- AlwaysDefect must always defect regardless of history

**Public Goods:**
- Free-rider advantage: free_rider payoff > contributor payoff (per round)
- Social optimum: all contribute → max total welfare
- Alpha constraint: multiplier > 1 for cooperation incentive

**Auction:**
- Second-price: truthful bidding is dominant strategy
- First-price: shade bidding should outperform truthful
- Revenue equivalence (asymptotic, many bidders)

**Colonel Blotto:**
- Budget constraint: sum of allocations = total resources
- Concentrated beats uniform on fewer battlefields
- Random has non-zero win probability against any deterministic

**Congestion:**
- Nash routing: no player benefits from switching routes
- Price of anarchy: Nash welfare < social optimal welfare
- Adding routes can decrease welfare (Braess's paradox)

### Step 5: Post-Generation

After writing files:
1. Run `uv run ruff format {test_file}`
2. Run `uv run ruff check {test_file} --fix`
3. Run `uv run pyrefly check`
4. Run `uv run pytest {test_file} -v` to verify tests pass
5. If game suite YAML was generated, validate with `uv run atp validate --suite={suite_file}`

## Rules

- **Line length:** 88 characters max
- **Type hints:** required on all test methods (return `-> None`)
- **Float comparison:** always use `pytest.approx()` for payoffs
- **Parametrize:** use for testing multiple strategy combinations
- **Fixtures:** use game/strategy fixtures from `game-environments/tests/conftest.py`
- **anyio not asyncio:** for any async game runner tests
- **Seed determinism:** always set seed in config for reproducible tests
- **Theoretical grounding:** every test must reference the game-theoretic property it verifies
