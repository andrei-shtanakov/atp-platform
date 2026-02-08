"""Data models for Nash equilibrium analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class NashEquilibrium:
    """Represents a Nash equilibrium of a game.

    Attributes:
        strategies: Player ID to mixed strategy (probability
            distribution over actions).
        payoffs: Expected payoff for each player at this
            equilibrium.
        support: Player ID to list of action indices with
            positive probability.
        epsilon: Tolerance for approximate equilibria.
            0.0 means exact Nash equilibrium.
    """

    strategies: dict[str, np.ndarray]
    payoffs: dict[str, float]
    support: dict[str, list[int]]
    epsilon: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "strategies": {k: v.tolist() for k, v in self.strategies.items()},
            "payoffs": dict(self.payoffs),
            "support": {k: list(v) for k, v in self.support.items()},
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NashEquilibrium:
        """Deserialize from dictionary."""
        return cls(
            strategies={k: np.array(v) for k, v in data["strategies"].items()},
            payoffs=data["payoffs"],
            support={k: list(v) for k, v in data["support"].items()},
            epsilon=data.get("epsilon", 0.0),
        )

    def is_pure(self) -> bool:
        """Check if this is a pure strategy Nash equilibrium."""
        for strategy in self.strategies.values():
            if np.count_nonzero(strategy) != 1:
                return False
        return True

    def is_mixed(self) -> bool:
        """Check if any player uses a mixed strategy."""
        return not self.is_pure()

    def __repr__(self) -> str:
        strats = {k: np.round(v, 4).tolist() for k, v in self.strategies.items()}
        return (
            f"NashEquilibrium(strategies={strats}, "
            f"payoffs={self.payoffs}, epsilon={self.epsilon})"
        )
