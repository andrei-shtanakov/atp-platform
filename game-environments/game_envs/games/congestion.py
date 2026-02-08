"""Congestion Game implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from game_envs.core.action import DiscreteActionSpace
from game_envs.core.game import Game, GameConfig, GameType, MoveOrder
from game_envs.core.state import (
    GameState,
    Observation,
    RoundResult,
    StepResult,
)


@dataclass(frozen=True)
class RouteDefinition:
    """Definition of a route in the congestion game.

    Attributes:
        name: Unique identifier for this route.
        base_cost: Fixed cost of using this route with no
            congestion.
        coefficient: How much each additional user increases
            the latency.
    """

    name: str
    base_cost: float
    coefficient: float

    def __post_init__(self) -> None:
        if self.base_cost < 0:
            raise ValueError(f"base_cost must be >= 0, got {self.base_cost}")
        if self.coefficient < 0:
            raise ValueError(f"coefficient must be >= 0, got {self.coefficient}")


@dataclass(frozen=True)
class CongestionConfig(GameConfig):
    """Congestion Game configuration.

    Attributes:
        routes: List of route definitions. Each route has a
            name, base_cost, and coefficient for the latency
            function: latency(r) = base_cost + coefficient *
            num_users_on_route.
    """

    routes: tuple[RouteDefinition, ...] = field(
        default_factory=lambda: (
            RouteDefinition(name="route_A", base_cost=1.0, coefficient=1.0),
            RouteDefinition(name="route_B", base_cost=2.0, coefficient=0.5),
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.routes) < 2:
            raise ValueError(f"need at least 2 routes, got {len(self.routes)}")
        names = [r.name for r in self.routes]
        if len(names) != len(set(names)):
            raise ValueError("route names must be unique")


class CongestionGame(Game):
    """N-player Congestion Game (routing game).

    Players simultaneously choose a route from source to
    destination. The cost (latency) of each route depends on
    how many players choose it:

        latency(route) = base_cost + coefficient * num_users

    Payoff = negative latency (lower latency is better).

    This game models network routing, traffic congestion, and
    resource selection problems. It can demonstrate Braess's
    paradox where adding a route worsens outcomes for all.

    Features:
        - Supports 2 to 50 players
        - Configurable routes with linear latency functions
        - Discount factor for repeated games
    """

    def __init__(self, config: CongestionConfig | None = None) -> None:
        super().__init__(config or CongestionConfig())
        n = self.config.num_players
        if n < 2 or n > 50:
            raise ValueError(f"num_players must be between 2 and 50, got {n}")
        self._terminal = False
        self._cumulative: dict[str, float] = {pid: 0.0 for pid in self.player_ids}

    @property
    def _cg_config(self) -> CongestionConfig:
        """Typed access to the congestion-specific config."""
        return self.config  # type: ignore[return-value]

    @property
    def name(self) -> str:
        n_routes = len(self._cg_config.routes)
        base = f"Congestion Game ({n_routes} routes)"
        if self.config.num_rounds > 1:
            base += f" (repeated x{self.config.num_rounds})"
        return base

    @property
    def game_type(self) -> GameType:
        if self.config.num_rounds > 1:
            return GameType.REPEATED
        return GameType.NORMAL_FORM

    @property
    def move_order(self) -> MoveOrder:
        return MoveOrder.SIMULTANEOUS

    @property
    def player_ids(self) -> list[str]:
        return [f"player_{i}" for i in range(self.config.num_players)]

    def action_space(self, player_id: str) -> DiscreteActionSpace:
        """Each player chooses one route."""
        route_names = [r.name for r in self._cg_config.routes]
        return DiscreteActionSpace(route_names)

    def reset(self) -> StepResult:
        """Reset game to initial state."""
        self._current_round = 0
        self._terminal = False
        self._history.clear()
        self._cumulative = {pid: 0.0 for pid in self.player_ids}
        state = GameState(
            round_number=0,
            player_states={},
            public_state={"game": self.name},
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs={pid: 0.0 for pid in self.player_ids},
            is_terminal=False,
        )

    def step(self, actions: dict[str, Any]) -> StepResult:
        """Process one round of route choices.

        Args:
            actions: Mapping of player_id to route name.

        Returns:
            StepResult with payoffs = negative latency.
        """
        if self._terminal:
            raise RuntimeError("Game is already terminal")

        # Count users per route
        route_counts: dict[str, int] = {r.name: 0 for r in self._cg_config.routes}
        for pid in self.player_ids:
            route = actions[pid]
            route_counts[route] += 1

        # Build route lookup
        route_map = {r.name: r for r in self._cg_config.routes}

        # Compute latency per route
        route_latency: dict[str, float] = {}
        for rname, count in route_counts.items():
            rd = route_map[rname]
            route_latency[rname] = rd.base_cost + rd.coefficient * count

        # Payoff = negative latency
        payoffs: dict[str, float] = {}
        for pid in self.player_ids:
            route = actions[pid]
            payoffs[pid] = -route_latency[route]

        return self._finalize_round(dict(actions), payoffs)

    def _finalize_round(
        self,
        actions: dict[str, Any],
        payoffs: dict[str, float],
    ) -> StepResult:
        """Apply discount, record round, check termination."""
        discount = self.config.discount_factor**self._current_round
        discounted = {pid: p * discount for pid, p in payoffs.items()}
        for pid in self.player_ids:
            self._cumulative[pid] += discounted[pid]

        self._current_round += 1

        rr = RoundResult(
            round_number=self._current_round,
            actions=actions,
            payoffs=payoffs,
        )
        self._history.add_round(rr)

        if self._current_round >= self.config.num_rounds:
            self._terminal = True

        state = GameState(
            round_number=self._current_round,
            player_states={},
            public_state={"game": self.name},
            is_terminal=self._terminal,
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs=payoffs,
            is_terminal=self._terminal,
        )

    def observe(self, player_id: str) -> Observation:
        """Get observation with game-specific state info."""
        c = self._cg_config
        routes_info: dict[str, dict[str, float]] = {}
        for r in c.routes:
            routes_info[r.name] = {
                "base_cost": r.base_cost,
                "coefficient": r.coefficient,
            }

        game_state: dict[str, Any] = {
            "game": self.name,
            "your_role": player_id,
            "num_players": c.num_players,
            "routes": routes_info,
            "latency_formula": (
                "latency(route) = base_cost + coefficient * num_users_on_route"
            ),
            "payoff_rule": ("payoff = -latency (lower latency is better)"),
        }

        return Observation(
            player_id=player_id,
            game_state=game_state,
            available_actions=self.action_space(player_id).to_list(),
            history=self._history.for_player(player_id),
            round_number=self._current_round,
            total_rounds=self.config.num_rounds,
        )

    def get_payoffs(self) -> dict[str, float]:
        """Get cumulative (discounted) payoffs."""
        return dict(self._cumulative)

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    def compute_latency(
        self,
        route_name: str,
        num_users: int,
    ) -> float:
        """Compute latency for a route given user count.

        Utility method for analysis and testing.

        Args:
            route_name: Name of the route.
            num_users: Number of users on the route.

        Returns:
            The latency value.
        """
        route_map = {r.name: r for r in self._cg_config.routes}
        rd = route_map[route_name]
        return rd.base_cost + rd.coefficient * num_users

    def compute_total_latency(
        self,
        actions: dict[str, str],
    ) -> float:
        """Compute total system latency for a routing profile.

        Args:
            actions: Mapping of player_id to route name.

        Returns:
            Sum of all players' latencies.
        """
        route_counts: dict[str, int] = {r.name: 0 for r in self._cg_config.routes}
        for route in actions.values():
            route_counts[route] += 1

        route_map = {r.name: r for r in self._cg_config.routes}
        total = 0.0
        for rname, count in route_counts.items():
            if count > 0:
                rd = route_map[rname]
                latency = rd.base_cost + rd.coefficient * count
                total += latency * count
        return total
