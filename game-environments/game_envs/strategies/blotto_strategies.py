"""Colonel Blotto baseline strategies."""

from __future__ import annotations

import random
from typing import Any

from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy


class UniformAllocation(Strategy):
    """Spreads resources equally across all battlefields.

    A defensive strategy that avoids leaving any battlefield
    uncontested. Performs well against concentrated strategies
    but is exploitable by uneven allocations.
    """

    @property
    def name(self) -> str:
        return "uniform_allocation"

    def choose_action(self, observation: Observation) -> Any:
        fields = observation.game_state.get("fields", [])
        total = observation.game_state.get("total_units", 100.0)
        n = len(fields)
        if n == 0:
            return {}
        per_field = total / n
        return {f: per_field for f in fields}


class ConcentratedAllocation(Strategy):
    """Concentrates resources on a majority of battlefields.

    Places all resources on the first ceil(n/2) + 1 fields,
    attempting to win a majority of battlefields while
    conceding the rest. Deterministic and predictable.
    """

    @property
    def name(self) -> str:
        return "concentrated_allocation"

    def choose_action(self, observation: Observation) -> Any:
        fields = observation.game_state.get("fields", [])
        total = observation.game_state.get("total_units", 100.0)
        n = len(fields)
        if n == 0:
            return {}
        # Concentrate on majority of fields
        target_count = n // 2 + 1
        per_target = total / target_count
        result: dict[str, float] = {}
        for i, f in enumerate(fields):
            if i < target_count:
                result[f] = per_target
            else:
                result[f] = 0.0
        return result


class NashMixed(Strategy):
    """Approximate Nash equilibrium mixed strategy for Blotto.

    For symmetric Colonel Blotto, the Nash equilibrium
    involves randomizing allocations. This approximation
    uses random Dirichlet-like sampling: generate n random
    values and normalize to sum to the total budget.

    This produces a different random allocation each round,
    making the strategy hard to exploit.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "nash_mixed"

    def choose_action(self, observation: Observation) -> Any:
        fields = observation.game_state.get("fields", [])
        total = observation.game_state.get("total_units", 100.0)
        n = len(fields)
        if n == 0:
            return {}
        # Approximate Nash: Dirichlet-like random allocation
        # using exponential draws (equivalent to symmetric
        # Dirichlet with alpha=1)
        weights = [self._rng.expovariate(1.0) for _ in range(n)]
        weight_sum = sum(weights)
        result: dict[str, float] = {}
        for i, f in enumerate(fields):
            result[f] = weights[i] / weight_sum * total
        return result

    def reset(self) -> None:
        pass
