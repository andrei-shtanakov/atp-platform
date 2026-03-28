"""Battle of the Sexes baseline strategies."""

from __future__ import annotations

from typing import Any

from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy

ACTION_A = "A"
ACTION_B = "B"


class AlwaysA(Strategy):
    """Always chooses event A regardless of opponent's actions."""

    @property
    def name(self) -> str:
        return "always_a"

    def choose_action(self, observation: Observation) -> Any:
        return ACTION_A


class AlwaysB(Strategy):
    """Always chooses event B regardless of opponent's actions."""

    @property
    def name(self) -> str:
        return "always_b"

    def choose_action(self, observation: Observation) -> Any:
        return ACTION_B


class Alternating(Strategy):
    """Alternates between A and B starting with A.

    Plays A on even rounds (0, 2, 4, ...) and B on odd rounds
    (1, 3, 5, ...) based on the current round number.
    """

    @property
    def name(self) -> str:
        return "alternating"

    def choose_action(self, observation: Observation) -> Any:
        round_number = len(observation.history)
        return ACTION_A if round_number % 2 == 0 else ACTION_B
