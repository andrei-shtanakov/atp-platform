"""Prisoner's Dilemma baseline strategies."""

from __future__ import annotations

import random
from typing import Any

from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy

COOPERATE = "cooperate"
DEFECT = "defect"


class AlwaysCooperate(Strategy):
    """Always cooperates regardless of opponent's actions."""

    @property
    def name(self) -> str:
        return "always_cooperate"

    def choose_action(self, observation: Observation) -> Any:
        return COOPERATE


class AlwaysDefect(Strategy):
    """Always defects regardless of opponent's actions."""

    @property
    def name(self) -> str:
        return "always_defect"

    def choose_action(self, observation: Observation) -> Any:
        return DEFECT


class TitForTat(Strategy):
    """Cooperates first, then copies opponent's last action.

    One of the most successful strategies in repeated PD
    tournaments (Axelrod, 1984). It is nice (starts with
    cooperation), retaliatory (punishes defection), forgiving
    (returns to cooperation), and clear (easy to understand).
    """

    @property
    def name(self) -> str:
        return "tit_for_tat"

    def choose_action(self, observation: Observation) -> Any:
        if not observation.history:
            return COOPERATE
        last_round = observation.history[-1]
        # Find opponent's last action
        for pid, action in last_round.actions.items():
            if pid != observation.player_id:
                return action
        return COOPERATE


class GrimTrigger(Strategy):
    """Cooperates until opponent defects, then defects forever.

    Also known as "Grim" or "Trigger". Maximally retaliatory:
    a single defection triggers permanent punishment.
    """

    def __init__(self) -> None:
        self._triggered = False

    @property
    def name(self) -> str:
        return "grim_trigger"

    def choose_action(self, observation: Observation) -> Any:
        if self._triggered:
            return DEFECT
        for rr in observation.history:
            for pid, action in rr.actions.items():
                if pid != observation.player_id and action == DEFECT:
                    self._triggered = True
                    return DEFECT
        return COOPERATE

    def reset(self) -> None:
        self._triggered = False


class Pavlov(Strategy):
    """Win-stay, lose-shift strategy.

    Cooperates on the first move. Thereafter, repeats the
    previous action if it got a high payoff (R or T), and
    switches if it got a low payoff (P or S).

    In practice: cooperates if both players chose the same
    action last round, defects otherwise.
    """

    @property
    def name(self) -> str:
        return "pavlov"

    def choose_action(self, observation: Observation) -> Any:
        if not observation.history:
            return COOPERATE
        last_round = observation.history[-1]
        my_action = last_round.actions.get(observation.player_id)
        opponent_action = None
        for pid, action in last_round.actions.items():
            if pid != observation.player_id:
                opponent_action = action
                break
        # Win-stay, lose-shift: cooperate if both same
        if my_action == opponent_action:
            return COOPERATE
        return DEFECT


class RandomStrategy(Strategy):
    """Cooperates or defects with equal probability.

    Uses a seeded RNG for reproducibility. Useful as a
    baseline for measuring exploitability.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random"

    def choose_action(self, observation: Observation) -> Any:
        return self._rng.choice([COOPERATE, DEFECT])

    def reset(self) -> None:
        pass
