"""Shared fixtures for game-environments tests."""

from __future__ import annotations

import random
from typing import Any

import pytest

from game_envs.core.action import (
    ContinuousActionSpace,
    DiscreteActionSpace,
    StructuredActionSpace,
)
from game_envs.core.game import (
    Game,
    GameConfig,
    GameType,
    MoveOrder,
)
from game_envs.core.state import (
    GameState,
    Message,
    Observation,
    RoundResult,
    StepResult,
)
from game_envs.core.strategy import Strategy


class StubGame(Game):
    """Minimal concrete Game for testing."""

    def __init__(
        self,
        config: GameConfig | None = None,
        num_players: int = 2,
    ) -> None:
        super().__init__(config)
        self._num_players = num_players
        self._terminal = False
        self._cumulative_payoffs: dict[str, float] = {
            pid: 0.0 for pid in self.player_ids
        }

    @property
    def name(self) -> str:
        return "stub_game"

    @property
    def game_type(self) -> GameType:
        return GameType.NORMAL_FORM

    @property
    def move_order(self) -> MoveOrder:
        return MoveOrder.SIMULTANEOUS

    @property
    def player_ids(self) -> list[str]:
        return [f"player_{i}" for i in range(self._num_players)]

    def action_space(self, player_id: str) -> DiscreteActionSpace:
        return DiscreteActionSpace(["A", "B"])

    def reset(self) -> StepResult:
        self._reset_base()
        self._terminal = False
        self._cumulative_payoffs = {pid: 0.0 for pid in self.player_ids}
        state = GameState(
            round_number=0,
            player_states={},
            public_state={},
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs={pid: 0.0 for pid in self.player_ids},
            is_terminal=False,
        )

    def step(self, actions: dict[str, Any]) -> StepResult:
        self._current_round += 1
        payoffs = {pid: 1.0 for pid in self.player_ids}
        for pid, p in payoffs.items():
            self._cumulative_payoffs[pid] += p
        rr = RoundResult(
            round_number=self._current_round,
            actions=actions,
            payoffs=payoffs,
        )
        self._history.add_round(rr)
        if self._current_round >= self.config.num_rounds:
            self._terminal = True
        state = GameState(
            round_number=self._current_round,
            player_states={},
            public_state={},
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
        return dict(self._cumulative_payoffs)

    @property
    def is_terminal(self) -> bool:
        return self._terminal


class StubStrategy(Strategy):
    """Always picks the first action."""

    @property
    def name(self) -> str:
        return "stub_strategy"

    def choose_action(self, observation: Observation) -> Any:
        if observation.available_actions:
            return observation.available_actions[0]
        return None


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


@pytest.fixture
def game_config() -> GameConfig:
    return GameConfig(num_players=2, num_rounds=3, seed=42)


@pytest.fixture
def stub_game(game_config: GameConfig) -> StubGame:
    return StubGame(config=game_config)


@pytest.fixture
def stub_strategy() -> StubStrategy:
    return StubStrategy()


@pytest.fixture
def discrete_space() -> DiscreteActionSpace:
    return DiscreteActionSpace(["cooperate", "defect"])


@pytest.fixture
def continuous_space() -> ContinuousActionSpace:
    return ContinuousActionSpace(0.0, 100.0)


@pytest.fixture
def structured_space() -> StructuredActionSpace:
    return StructuredActionSpace(
        schema={
            "type": "allocation",
            "fields": ["field_a", "field_b", "field_c"],
            "total": 100,
            "min_value": 0,
        }
    )


@pytest.fixture
def sample_round_result() -> RoundResult:
    return RoundResult(
        round_number=1,
        actions={"p1": "cooperate", "p2": "defect"},
        payoffs={"p1": 0.0, "p2": 5.0},
        messages=[
            Message(
                sender="p1",
                content="hello",
                round_number=1,
            )
        ],
    )


@pytest.fixture
def sample_observation(
    sample_round_result: RoundResult,
) -> Observation:
    return Observation(
        player_id="p1",
        game_state={"score": 10},
        available_actions=["cooperate", "defect"],
        history=[sample_round_result],
        round_number=2,
        total_rounds=5,
        messages=[
            Message(
                sender="p2",
                content="hi",
                round_number=2,
            )
        ],
    )
