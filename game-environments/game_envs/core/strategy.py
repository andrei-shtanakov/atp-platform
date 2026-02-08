"""Strategy abstraction for baseline agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from game_envs.core.state import Observation


class Strategy(ABC):
    """Abstract base for built-in game strategies.

    Strategies are deterministic or stochastic policies that
    map observations to actions. Used as baselines for
    evaluating LLM agents.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this strategy."""
        ...

    @abstractmethod
    def choose_action(self, observation: Observation) -> Any:
        """Choose an action given the current observation.

        Args:
            observation: What the player currently sees.

        Returns:
            An action valid for the game's action space.
        """
        ...

    def reset(self) -> None:
        """Reset internal state between episodes."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
