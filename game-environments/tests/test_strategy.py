"""Tests for Strategy ABC."""

from __future__ import annotations

import pytest

from game_envs.core.state import Observation, RoundResult
from game_envs.core.strategy import Strategy
from game_envs.strategies import (
    AlwaysCooperate,
    AlwaysDefect,
    ConcentratedAllocation,
    ConditionalCooperator,
    EpsilonGreedy,
    FreeRider,
    FullContributor,
    GrimTrigger,
    NashMixed,
    Pavlov,
    Punisher,
    RandomBidder,
    RandomStrategy,
    SelfishRouter,
    ShadeBidder,
    SocialOptimum,
    TitForTat,
    TruthfulBidder,
    UniformAllocation,
)


class TestStubStrategy:
    def test_name(self, stub_strategy) -> None:  # type: ignore[no-untyped-def]
        assert stub_strategy.name == "stub_strategy"

    def test_choose_action(self, stub_strategy) -> None:  # type: ignore[no-untyped-def]
        obs = Observation(
            player_id="p1",
            game_state={},
            available_actions=["cooperate", "defect"],
            history=[],
            round_number=1,
            total_rounds=1,
        )
        action = stub_strategy.choose_action(obs)
        assert action == "cooperate"

    def test_choose_action_empty(self, stub_strategy) -> None:  # type: ignore[no-untyped-def]
        obs = Observation(
            player_id="p1",
            game_state={},
            available_actions=[],
            history=[],
            round_number=1,
            total_rounds=1,
        )
        action = stub_strategy.choose_action(obs)
        assert action is None

    def test_reset(self, stub_strategy) -> None:  # type: ignore[no-untyped-def]
        # Reset should not raise
        stub_strategy.reset()

    def test_repr(self, stub_strategy) -> None:  # type: ignore[no-untyped-def]
        r = repr(stub_strategy)
        assert "StubStrategy" in r
        assert "stub_strategy" in r


ALL_STRATEGIES: list[Strategy] = [
    AlwaysCooperate(),
    AlwaysDefect(),
    TitForTat(),
    GrimTrigger(),
    Pavlov(),
    RandomStrategy(seed=42),
    FullContributor(),
    FreeRider(),
    ConditionalCooperator(),
    Punisher(),
    TruthfulBidder(),
    ShadeBidder(factor=0.5),
    RandomBidder(seed=42),
    UniformAllocation(),
    ConcentratedAllocation(),
    NashMixed(seed=42),
    SelfishRouter(),
    SocialOptimum(),
    EpsilonGreedy(epsilon=0.1, seed=42),
]


class TestAllStrategiesABC:
    """Verify all strategies implement the Strategy ABC."""

    @pytest.mark.parametrize(
        "strategy",
        ALL_STRATEGIES,
        ids=lambda s: s.name,
    )
    def test_is_strategy_subclass(self, strategy: Strategy) -> None:
        assert isinstance(strategy, Strategy)

    @pytest.mark.parametrize(
        "strategy",
        ALL_STRATEGIES,
        ids=lambda s: s.name,
    )
    def test_has_name(self, strategy: Strategy) -> None:
        assert isinstance(strategy.name, str)
        assert len(strategy.name) > 0

    @pytest.mark.parametrize(
        "strategy",
        ALL_STRATEGIES,
        ids=lambda s: s.name,
    )
    def test_reset_does_not_raise(self, strategy: Strategy) -> None:
        strategy.reset()


class TestResetClearsState:
    """Verify reset() clears stateful strategies."""

    def test_grim_trigger_reset(self) -> None:
        s = GrimTrigger()
        obs = Observation(
            player_id="player_0",
            game_state={},
            available_actions=["cooperate", "defect"],
            history=[
                RoundResult(
                    round_number=0,
                    actions={"player_0": "cooperate", "player_1": "defect"},
                    payoffs={"player_0": 0.0, "player_1": 5.0},
                )
            ],
            round_number=1,
            total_rounds=10,
        )
        # Trigger the grim state
        assert s.choose_action(obs) == "defect"
        # Reset should clear
        s.reset()
        fresh_obs = Observation(
            player_id="player_0",
            game_state={},
            available_actions=["cooperate", "defect"],
            history=[],
            round_number=0,
            total_rounds=10,
        )
        assert s.choose_action(fresh_obs) == "cooperate"
