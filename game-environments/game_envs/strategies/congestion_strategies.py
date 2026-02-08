"""Congestion Game baseline strategies."""

from __future__ import annotations

import random
from typing import Any

from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy


class SelfishRouter(Strategy):
    """Always chooses the route with lowest expected cost.

    Selects the route with the minimum base cost from the
    observation. Does not account for congestion effects
    from other players choosing the same route.
    """

    @property
    def name(self) -> str:
        return "selfish_router"

    def choose_action(self, observation: Observation) -> Any:
        routes = observation.game_state.get("routes", {})
        if not routes:
            if observation.available_actions:
                return observation.available_actions[0]
            return None
        # Pick route with lowest base cost
        best_route = None
        best_cost = float("inf")
        for route, info in routes.items():
            if isinstance(info, dict):
                cost = float(info.get("base_cost", float("inf")))
            else:
                cost = float(info)
            if cost < best_cost:
                best_cost = cost
                best_route = route
        return best_route


class SocialOptimum(Strategy):
    """Chooses route to minimize total system cost.

    Considers congestion: picks the route with the highest
    capacity or lowest congestion factor, spreading load
    across the network. Falls back to the route with the
    highest capacity if congestion info is available.
    """

    @property
    def name(self) -> str:
        return "social_optimum"

    def choose_action(self, observation: Observation) -> Any:
        routes = observation.game_state.get("routes", {})
        if not routes:
            if observation.available_actions:
                return observation.available_actions[0]
            return None
        # Consider historical load: pick least-used route
        route_usage: dict[str, int] = {}
        for route in routes:
            route_usage[route] = 0
        for rr in observation.history:
            for pid, action in rr.actions.items():
                if action in route_usage:
                    route_usage[action] += 1
        # Pick least congested route
        best_route = min(route_usage, key=lambda r: route_usage[r])
        return best_route


class EpsilonGreedy(Strategy):
    """Epsilon-greedy route selection.

    With probability (1 - epsilon), picks the best route
    (lowest recent average cost). With probability epsilon,
    explores by choosing a random route.

    Args:
        epsilon: Exploration probability (0.0 to 1.0).
            Default 0.1.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self._epsilon = epsilon
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return f"epsilon_greedy_{self._epsilon}"

    def choose_action(self, observation: Observation) -> Any:
        routes = observation.game_state.get("routes", {})
        route_names = list(routes.keys()) if routes else []
        if not route_names:
            if observation.available_actions:
                route_names = list(observation.available_actions)
            else:
                return None

        # Explore
        if self._rng.random() < self._epsilon:
            return self._rng.choice(route_names)

        # Exploit: pick route with best recent payoff
        if not observation.history:
            return self._rng.choice(route_names)

        # Compute average payoff per route from history
        route_payoffs: dict[str, list[float]] = {r: [] for r in route_names}
        for rr in observation.history:
            my_action = rr.actions.get(observation.player_id)
            my_payoff = rr.payoffs.get(observation.player_id, 0.0)
            if isinstance(my_action, str) and my_action in route_payoffs:
                route_payoffs[my_action].append(my_payoff)

        # Pick route with highest average payoff (or lowest
        # cost, depending on game convention)
        best_route = route_names[0]
        best_avg = float("-inf")
        for route, payoffs in route_payoffs.items():
            if payoffs:
                avg = sum(payoffs) / len(payoffs)
                if avg > best_avg:
                    best_avg = avg
                    best_route = route
            elif not route_payoffs[best_route]:
                # Prefer unexplored routes
                best_route = route

        return best_route

    def reset(self) -> None:
        pass
