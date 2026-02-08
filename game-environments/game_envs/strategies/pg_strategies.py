"""Public Goods Game baseline strategies."""

from __future__ import annotations

from typing import Any

from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy


class FullContributor(Strategy):
    """Contributes entire endowment to the public pool.

    Maximizes group welfare but is exploitable by free riders.
    """

    @property
    def name(self) -> str:
        return "full_contributor"

    def choose_action(self, observation: Observation) -> Any:
        endowment = observation.game_state.get("endowment", 20.0)
        return float(endowment)


class FreeRider(Strategy):
    """Contributes nothing to the public pool.

    Nash equilibrium in one-shot public goods games.
    Maximizes individual payoff when others contribute.
    """

    @property
    def name(self) -> str:
        return "free_rider"

    def choose_action(self, observation: Observation) -> Any:
        return 0.0


class ConditionalCooperator(Strategy):
    """Matches the average contribution of others.

    Contributes the average amount contributed by other
    players in the previous round. Contributes the full
    endowment in the first round.
    """

    @property
    def name(self) -> str:
        return "conditional_cooperator"

    def choose_action(self, observation: Observation) -> Any:
        endowment = observation.game_state.get("endowment", 20.0)
        if not observation.history:
            return float(endowment)
        last_round = observation.history[-1]
        others = []
        for pid, action in last_round.actions.items():
            if pid != observation.player_id:
                # Handle both simple float and dict actions
                if isinstance(action, dict):
                    others.append(float(action.get("contribution", 0)))
                else:
                    others.append(float(action))
        if not others:
            return float(endowment)
        avg = sum(others) / len(others)
        return min(float(endowment), max(0.0, avg))


class Punisher(Strategy):
    """Contributes fully but punishes free riders.

    Contributes the full endowment in the contribution stage.
    In the punishment stage, punishes players who contributed
    less than 50% of the endowment.

    In games without punishment, behaves like FullContributor.
    """

    @property
    def name(self) -> str:
        return "punisher"

    def choose_action(self, observation: Observation) -> Any:
        endowment = observation.game_state.get("endowment", 20.0)
        stage = observation.game_state.get("stage")
        if stage == "punish":
            # Punish: spend proportional to free-riding
            contributions = observation.game_state.get("contributions", {})
            total_punishment = 0.0
            for pid, contrib in contributions.items():
                if pid != observation.player_id:
                    c = float(contrib)
                    if c < endowment * 0.5:
                        total_punishment += endowment * 0.5 - c
            return min(float(endowment), total_punishment)
        return float(endowment)
