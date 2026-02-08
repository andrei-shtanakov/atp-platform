"""Game registry for name-based game lookup and factory."""

from __future__ import annotations

import dataclasses
from typing import Any

from game_envs.core.game import Game, GameConfig


class GameRegistry:
    """Registry mapping game names to game classes.

    Provides a centralized lookup for creating game instances
    by name, useful for YAML-based suite configuration.

    Supports:
        - Name-based registration and lookup
        - Factory creation from GameConfig or raw dict
        - Auto-registration via ``@register_game`` decorator
        - Metadata introspection via ``game_info``
    """

    _registry: dict[str, type[Game]] = {}
    _config_classes: dict[str, type[GameConfig]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        game_class: type[Game],
        config_class: type[GameConfig] | None = None,
    ) -> None:
        """Register a game class under a name.

        Args:
            name: Lookup key for the game.
            game_class: The Game subclass to register.
            config_class: Optional config class for dict-based
                creation. If not provided, uses ``GameConfig``.

        Raises:
            ValueError: If name is already registered.
        """
        if name in cls._registry:
            raise ValueError(f"Game '{name}' is already registered")
        cls._registry[name] = game_class
        cls._config_classes[name] = config_class or GameConfig

    @classmethod
    def get(cls, name: str) -> type[Game]:
        """Look up a game class by name.

        Args:
            name: The registered game name.

        Returns:
            The Game subclass.

        Raises:
            KeyError: If name is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(f"Unknown game '{name}'. Available: {available}")
        return cls._registry[name]

    @classmethod
    def create(
        cls,
        name: str,
        config: GameConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Game:
        """Create a game instance by name.

        Args:
            name: The registered game name.
            config: Game configuration as a ``GameConfig``
                instance or a dict (for YAML deserialization).
                Dict values are passed to the registered
                config class constructor.
            **kwargs: Additional keyword arguments for the
                game constructor.

        Returns:
            A new Game instance.
        """
        game_class = cls.get(name)
        if isinstance(config, dict):
            config_cls = cls._config_classes[name]
            config = config_cls(**config)
        return game_class(config=config, **kwargs)

    @classmethod
    def list_games(cls) -> list[str]:
        """List all registered game names."""
        return sorted(cls._registry)

    @classmethod
    def game_info(cls, name: str) -> dict[str, Any]:
        """Return metadata about a registered game.

        Creates a temporary instance with default config to
        extract description, action spaces, and config schema.

        Args:
            name: The registered game name.

        Returns:
            Dict with keys: name, description, game_type,
            move_order, player_ids, action_spaces, and
            config_schema.

        Raises:
            KeyError: If name is not registered.
        """
        game_class = cls.get(name)
        config_cls = cls._config_classes[name]

        # Create a temporary instance for metadata
        game = game_class()

        # Build action space descriptions per player
        action_spaces: dict[str, str] = {}
        for pid in game.player_ids:
            action_spaces[pid] = game.action_space(pid).to_description()

        # Build config schema from dataclass fields
        config_schema: dict[str, Any] = {}
        for field in dataclasses.fields(config_cls):
            config_schema[field.name] = {
                "type": (
                    field.type
                    if isinstance(field.type, str)
                    else getattr(field.type, "__name__", str(field.type))
                ),
                "default": (
                    field.default if field.default is not dataclasses.MISSING else None
                ),
            }

        return {
            "name": game.name,
            "description": (game_class.__doc__ or "").strip(),
            "game_type": str(game.game_type),
            "move_order": str(game.move_order),
            "player_ids": game.player_ids,
            "action_spaces": action_spaces,
            "config_schema": config_schema,
        }

    @classmethod
    def clear(cls) -> None:
        """Remove all registered games (for testing)."""
        cls._registry.clear()
        cls._config_classes.clear()


def register_game(
    name: str,
    config_class: type[GameConfig] | None = None,
) -> Any:
    """Decorator for auto-registering a game class.

    Usage::

        @register_game("prisoners_dilemma", PDConfig)
        class PrisonersDilemma(Game):
            ...

    Args:
        name: Registry lookup key for the game.
        config_class: Optional config class for dict-based
            creation.

    Returns:
        A class decorator that registers the game and returns
        the class unchanged.
    """

    def decorator(cls: type[Game]) -> type[Game]:
        GameRegistry.register(name, cls, config_class)
        return cls

    return decorator
