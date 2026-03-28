"""Stag Hunt baseline strategies."""

from __future__ import annotations

from typing import Any

from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy

STAG = "stag"
HARE = "hare"


class AlwaysStag(Strategy):
    """Always hunts stag regardless of opponent's actions."""

    @property
    def name(self) -> str:
        return "always_stag"

    def choose_action(self, observation: Observation) -> Any:
        return STAG


class AlwaysHare(Strategy):
    """Always hunts hare regardless of opponent's actions."""

    @property
    def name(self) -> str:
        return "always_hare"

    def choose_action(self, observation: Observation) -> Any:
        return HARE


class StagTitForTat(Strategy):
    """Starts with stag, then mirrors opponent's last action.

    A coordination-friendly strategy: begins optimistically by
    hunting stag, then matches whatever the opponent did last
    round. Converges to mutual stag when paired with itself
    or other cooperative strategies.
    """

    @property
    def name(self) -> str:
        return "stag_tit_for_tat"

    def choose_action(self, observation: Observation) -> Any:
        if not observation.history:
            return STAG
        last_round = observation.history[-1]
        for pid, action in last_round.actions.items():
            if pid != observation.player_id:
                return action
        return STAG
