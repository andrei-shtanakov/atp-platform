"""Tests for StrategyRegistry."""

from __future__ import annotations

import pytest

from game_envs.core.strategy import Strategy
from game_envs.strategies.registry import StrategyRegistry


class TestStrategyRegistry:
    def setup_method(self) -> None:
        """Save and restore registry state."""
        self._saved = dict(StrategyRegistry._registry)

    def teardown_method(self) -> None:
        StrategyRegistry._registry = self._saved

    def test_register_and_get(self) -> None:
        from game_envs.strategies.pd_strategies import (
            AlwaysCooperate,
        )

        StrategyRegistry.clear()
        StrategyRegistry.register("test_ac", AlwaysCooperate)
        assert StrategyRegistry.get("test_ac") is AlwaysCooperate

    def test_register_duplicate_raises(self) -> None:
        from game_envs.strategies.pd_strategies import (
            AlwaysCooperate,
        )

        StrategyRegistry.clear()
        StrategyRegistry.register("dup", AlwaysCooperate)
        with pytest.raises(ValueError, match="already registered"):
            StrategyRegistry.register("dup", AlwaysCooperate)

    def test_get_unknown_raises(self) -> None:
        StrategyRegistry.clear()
        with pytest.raises(KeyError, match="Unknown strategy"):
            StrategyRegistry.get("nonexistent")

    def test_create_instance(self) -> None:
        from game_envs.strategies.pd_strategies import (
            AlwaysCooperate,
        )

        StrategyRegistry.clear()
        StrategyRegistry.register("create_test", AlwaysCooperate)
        instance = StrategyRegistry.create("create_test")
        assert isinstance(instance, Strategy)
        assert instance.name == "always_cooperate"

    def test_create_with_kwargs(self) -> None:
        from game_envs.strategies.pd_strategies import (
            RandomStrategy,
        )

        StrategyRegistry.clear()
        StrategyRegistry.register("rand_test", RandomStrategy)
        instance = StrategyRegistry.create("rand_test", seed=42)
        assert isinstance(instance, RandomStrategy)

    def test_list_strategies(self) -> None:
        from game_envs.strategies.pd_strategies import (
            AlwaysCooperate,
            AlwaysDefect,
        )

        StrategyRegistry.clear()
        StrategyRegistry.register("b_strat", AlwaysDefect)
        StrategyRegistry.register("a_strat", AlwaysCooperate)
        result = StrategyRegistry.list_strategies()
        assert result == ["a_strat", "b_strat"]

    def test_clear(self) -> None:
        from game_envs.strategies.pd_strategies import (
            AlwaysCooperate,
        )

        StrategyRegistry.clear()
        StrategyRegistry.register("clear_test", AlwaysCooperate)
        assert len(StrategyRegistry.list_strategies()) == 1
        StrategyRegistry.clear()
        assert len(StrategyRegistry.list_strategies()) == 0

    def test_builtin_strategies_registered(self) -> None:
        """Verify all built-in strategies are registered at
        import time."""
        # Re-import to trigger registration
        import game_envs  # noqa: F401

        expected = [
            "always_cooperate",
            "always_defect",
            "concentrated_allocation",
            "conditional_cooperator",
            "epsilon_greedy",
            "free_rider",
            "full_contributor",
            "grim_trigger",
            "nash_mixed",
            "pavlov",
            "punisher",
            "random",
            "random_bidder",
            "selfish_router",
            "shade_bidder",
            "social_optimum",
            "tit_for_tat",
            "truthful_bidder",
            "uniform_allocation",
        ]
        registered = StrategyRegistry.list_strategies()
        for name in expected:
            assert name in registered, f"Strategy '{name}' not registered"
