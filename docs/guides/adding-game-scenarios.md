# Adding Game Scenarios for Game-Theoretic Testing

This guide explains how to add new game-theoretic scenarios to the ATP Platform. Games live in the standalone `game-environments` package and are used by `atp-games` for agent evaluation.

## Overview

Each game consists of three parts:

1. **Game class** — implements the game rules, payoff logic, and state management
2. **Strategies** — baseline agents for benchmarking (e.g., always-cooperate, tit-for-tat)
3. **Tests** — verify correctness of payoff matrices and edge cases

## Quick Start

To add a game called "My Game":

```
game-environments/
  game_envs/
    games/my_game.py              # Game + config
    strategies/my_game_strategies.py  # Baseline strategies
  tests/test_my_game.py           # Tests
```

## Step 1: Define the Config

Create `game-environments/game_envs/games/my_game.py`. Start with a frozen dataclass that extends `GameConfig`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from game_envs.core.action import DiscreteActionSpace
from game_envs.core.game import Game, GameConfig, GameType, MoveOrder
from game_envs.core.state import GameState, Observation, RoundResult, StepResult
from game_envs.games.registry import register_game


@dataclass(frozen=True)
class MyGameConfig(GameConfig):
    """Configuration for My Game.

    Constraints: reward_high > reward_low > 0
    """

    reward_high: float = 4.0
    reward_low: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (self.reward_high > self.reward_low > 0):
            msg = "Must have reward_high > reward_low > 0"
            raise ValueError(msg)
```

**Inherited fields from `GameConfig`** (available on every config):
- `num_players` (default 2)
- `num_rounds` (default 1)
- `discount_factor` (default 1.0)
- `noise` (default 0.0 — probability of random action flip)
- `seed` (optional — for reproducibility)

## Step 2: Implement the Game

```python
@register_game("my_game", MyGameConfig)
class MyGame(Game):
    """My Game — a description of what it tests."""

    def __init__(self, config: MyGameConfig | None = None) -> None:
        super().__init__(config or MyGameConfig())
        self._terminal = False
        self._cumulative: dict[str, float] = {}

    @property
    def _cfg(self) -> MyGameConfig:
        return self.config  # type: ignore[return-value]

    # --- Required properties ---

    @property
    def name(self) -> str:
        rounds = self._cfg.num_rounds
        if rounds > 1:
            return f"My Game (repeated x{rounds})"
        return "My Game"

    @property
    def game_type(self) -> GameType:
        if self._cfg.num_rounds > 1:
            return GameType.REPEATED
        return GameType.NORMAL_FORM

    @property
    def move_order(self) -> MoveOrder:
        return MoveOrder.SIMULTANEOUS

    @property
    def player_ids(self) -> list[str]:
        return [f"player_{i}" for i in range(self._cfg.num_players)]

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    # --- Required methods ---

    def action_space(self, player_id: str) -> DiscreteActionSpace:
        return DiscreteActionSpace(["action_a", "action_b"])

    def reset(self) -> StepResult:
        self._reset_base()
        self._terminal = False
        self._current_round = 0
        self._cumulative = {pid: 0.0 for pid in self.player_ids}
        state = GameState(
            round_number=0,
            player_states={},
            public_state={"game": self.name},
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs={pid: 0.0 for pid in self.player_ids},
            is_terminal=False,
        )

    def step(self, actions: dict[str, Any]) -> StepResult:
        if self._terminal:
            msg = "Game is already terminal"
            raise RuntimeError(msg)

        p0, p1 = self.player_ids[0], self.player_ids[1]
        a0, a1 = actions[p0], actions[p1]

        # Apply noise (trembling hand)
        if self._cfg.noise > 0:
            if self._rng.random() < self._cfg.noise:
                a0 = "action_b" if a0 == "action_a" else "action_a"
            if self._rng.random() < self._cfg.noise:
                a1 = "action_b" if a1 == "action_a" else "action_a"

        # Compute payoffs (your game logic here)
        if a0 == "action_a" and a1 == "action_a":
            payoffs = {p0: self._cfg.reward_high, p1: self._cfg.reward_high}
        else:
            payoffs = {p0: self._cfg.reward_low, p1: self._cfg.reward_low}

        # Apply discount factor for repeated games
        discount = self._cfg.discount_factor ** self._current_round
        for pid in self.player_ids:
            self._cumulative[pid] += payoffs[pid] * discount

        # Record round in history
        self._history.add_round(
            RoundResult(
                round_number=self._current_round,
                actions={p0: a0, p1: a1},
                payoffs=payoffs,
            )
        )

        self._current_round += 1
        if self._current_round >= self._cfg.num_rounds:
            self._terminal = True

        state = GameState(
            round_number=self._current_round,
            player_states={},
            public_state={"game": self.name},
            is_terminal=self._terminal,
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs=payoffs,
            is_terminal=self._terminal,
        )

    def get_payoffs(self) -> dict[str, float]:
        return dict(self._cumulative)

    def observe(self, player_id: str) -> Observation:
        return Observation(
            player_id=player_id,
            game_state={
                "game": self.name,
                "your_role": player_id,
            },
            available_actions=self.action_space(player_id).to_list(),
            history=self._history.for_player(player_id),
            round_number=self._current_round,
            total_rounds=self._cfg.num_rounds,
            messages=self._get_pending_messages(player_id),
        )
```

**Key points:**
- `@register_game("my_game", MyGameConfig)` registers the game in the global registry
- `step()` returns **per-round** payoffs in `StepResult.payoffs`, not cumulative
- `get_payoffs()` returns **cumulative** payoffs (with discount factor applied)
- `self._history` and `self._rng` are provided by the base `Game` class
- `self._reset_base()` must be called in `reset()` to clear history

## Step 3: Add Strategies

Create `game-environments/game_envs/strategies/my_game_strategies.py`:

```python
from __future__ import annotations

from typing import Any

from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy


class AlwaysA(Strategy):
    """Always choose action_a."""

    @property
    def name(self) -> str:
        return "always_a"

    def choose_action(self, observation: Observation) -> Any:
        return "action_a"

    def reset(self) -> None:
        pass


class AlwaysB(Strategy):
    """Always choose action_b."""

    @property
    def name(self) -> str:
        return "always_b"

    def choose_action(self, observation: Observation) -> Any:
        return "action_b"

    def reset(self) -> None:
        pass


class ReactiveStrategy(Strategy):
    """Mirror the opponent's last action (start with action_a)."""

    @property
    def name(self) -> str:
        return "reactive"

    def choose_action(self, observation: Observation) -> Any:
        if not observation.history:
            return "action_a"
        last_round = observation.history[-1]
        # Find opponent's action
        for pid, action in last_round.actions.items():
            if pid != observation.player_id:
                return action
        return "action_a"

    def reset(self) -> None:
        pass
```

## Step 4: Register Exports

Add to `game-environments/game_envs/games/__init__.py`:

```python
from game_envs.games.my_game import MyGame, MyGameConfig

# Add to __all__:
__all__ = [
    # ... existing exports ...
    "MyGame",
    "MyGameConfig",
]
```

Add to `game-environments/game_envs/strategies/__init__.py`:

```python
from game_envs.strategies.my_game_strategies import AlwaysA, AlwaysB, ReactiveStrategy

# Add to __all__:
__all__ = [
    # ... existing exports ...
    "AlwaysA",
    "AlwaysB",
    "ReactiveStrategy",
]
```

## Step 5: Write Tests

Create `game-environments/tests/test_my_game.py`:

```python
"""Tests for My Game implementation."""

import pytest

from game_envs.games.my_game import MyGame, MyGameConfig


class TestMyGamePayoffs:
    """Verify payoff matrix matches theory."""

    def test_both_action_a(self) -> None:
        game = MyGame(MyGameConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "action_a", "player_1": "action_a"})
        assert result.payoffs["player_0"] == pytest.approx(4.0)
        assert result.payoffs["player_1"] == pytest.approx(4.0)

    def test_mixed_actions(self) -> None:
        game = MyGame(MyGameConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "action_a", "player_1": "action_b"})
        assert result.payoffs["player_0"] == pytest.approx(1.0)

    def test_config_validation(self) -> None:
        with pytest.raises(ValueError):
            MyGameConfig(reward_high=1.0, reward_low=5.0)


class TestMyGameRepeated:
    """Test multi-round behavior."""

    def test_multi_round_accumulates(self) -> None:
        game = MyGame(MyGameConfig(num_players=2, num_rounds=3))
        game.reset()
        for _ in range(3):
            game.step({"player_0": "action_a", "player_1": "action_a"})
        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == pytest.approx(12.0)  # 3 * 4.0

    def test_single_round_terminal(self) -> None:
        game = MyGame(MyGameConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "action_a", "player_1": "action_a"})
        assert result.is_terminal


class TestMyGameRegistry:
    """Test registry integration."""

    def test_in_registry(self) -> None:
        from game_envs.games.registry import GameRegistry

        assert "my_game" in GameRegistry.list_games()
```

## Step 6: Update Registry Test Count

In `game-environments/tests/test_registry.py`, update the game count assertion:

```python
# Find the line: assert len(result) == 7
# Change to:     assert len(result) == 8
```

## Step 7: Run Tests and Quality Checks

```bash
# Run your tests
uv run python -m pytest game-environments/tests/test_my_game.py -v

# Run all game-environments tests
uv run python -m pytest game-environments/tests/ -v

# Code quality
uv run ruff format game-environments/
uv run ruff check game-environments/ --fix
```

## Action Space Types

Choose the appropriate action space for your game:

| Type | Use Case | Example |
|------|----------|---------|
| `DiscreteActionSpace(["a", "b"])` | Finite named actions | Cooperate/defect, rock/paper/scissors |
| `ContinuousActionSpace(0.0, 100.0)` | Numeric range | Bid amount, contribution level |
| `StructuredActionSpace(schema)` | Allocation vectors | Troop deployment, resource distribution |

## Existing Games as Reference

| Game | File | Action Type | Players | Key Feature |
|------|------|-------------|---------|-------------|
| Prisoner's Dilemma | `prisoners_dilemma.py` | Discrete | 2 | Cooperation vs defection |
| Stag Hunt | `stag_hunt.py` | Discrete | 2 | Trust vs safety (simplest template) |
| Battle of the Sexes | `battle_of_sexes.py` | Discrete | 2 | Asymmetric payoffs |
| Public Goods | `public_goods.py` | Continuous | 2-20 | N-player, contribution amounts |
| Auction | `auction.py` | Continuous | 2+ | Private values, partial observability |
| Colonel Blotto | `colonel_blotto.py` | Structured | 2 | Allocation with sum constraint |
| Congestion | `congestion.py` | Discrete | 2-50 | Route selection, negative payoffs |

**Recommended starting template:** `stag_hunt.py` — it's the simplest 2-player discrete game.

## Checklist

- [ ] Config class (`@dataclass(frozen=True)`, extends `GameConfig`, validation in `__post_init__`)
- [ ] Game class (`@register_game`, all abstract methods implemented)
- [ ] Strategies (at least 2-3 baselines, each extends `Strategy`)
- [ ] Exports in `games/__init__.py` and `strategies/__init__.py`
- [ ] Tests: payoff correctness, config validation, multi-round, terminal state, registry
- [ ] Registry test count updated
- [ ] `ruff format` and `ruff check` pass
