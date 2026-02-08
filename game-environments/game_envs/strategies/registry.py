"""Strategy registry for name-based strategy lookup."""

from __future__ import annotations

from typing import Any

from game_envs.core.strategy import Strategy


class StrategyRegistry:
    """Registry mapping strategy names to strategy classes.

    Provides centralized lookup for creating strategy instances
    by name, useful for YAML-based suite configuration where
    strategies are referenced by string.
    """

    _registry: dict[str, type[Strategy]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        strategy_class: type[Strategy],
    ) -> None:
        """Register a strategy class under a name.

        Args:
            name: Lookup key for the strategy.
            strategy_class: The Strategy subclass to register.

        Raises:
            ValueError: If name is already registered.
        """
        if name in cls._registry:
            raise ValueError(f"Strategy '{name}' is already registered")
        cls._registry[name] = strategy_class

    @classmethod
    def get(cls, name: str) -> type[Strategy]:
        """Look up a strategy class by name.

        Args:
            name: The registered strategy name.

        Returns:
            The Strategy subclass.

        Raises:
            KeyError: If name is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(f"Unknown strategy '{name}'. Available: {available}")
        return cls._registry[name]

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,
    ) -> Strategy:
        """Create a strategy instance by name.

        Args:
            name: The registered strategy name.
            **kwargs: Arguments passed to the strategy
                constructor.

        Returns:
            A new Strategy instance.
        """
        strategy_class = cls.get(name)
        return strategy_class(**kwargs)

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names."""
        return sorted(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Remove all registered strategies (for testing)."""
        cls._registry.clear()
