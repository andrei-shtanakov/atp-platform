"""Tests for Congestion Game strategies."""

from __future__ import annotations

from game_envs.core.state import Observation, RoundResult
from game_envs.strategies.congestion_strategies import (
    EpsilonGreedy,
    SelfishRouter,
    SocialOptimum,
)


def _make_obs(
    player_id: str = "player_0",
    routes: dict | None = None,
    history: list[RoundResult] | None = None,
    available_actions: list | None = None,
) -> Observation:
    """Create a minimal congestion game observation."""
    if routes is None:
        routes = {
            "route_a": {"base_cost": 10.0, "capacity": 5},
            "route_b": {"base_cost": 20.0, "capacity": 10},
            "route_c": {"base_cost": 15.0, "capacity": 7},
        }
    if available_actions is None:
        available_actions = list(routes.keys())
    return Observation(
        player_id=player_id,
        game_state={
            "game": "Congestion Game",
            "your_role": player_id,
            "routes": routes,
        },
        available_actions=available_actions,
        history=history or [],
        round_number=len(history) if history else 0,
        total_rounds=10,
    )


class TestSelfishRouter:
    def test_picks_lowest_cost(self) -> None:
        s = SelfishRouter()
        obs = _make_obs()
        assert s.choose_action(obs) == "route_a"

    def test_different_costs(self) -> None:
        s = SelfishRouter()
        routes = {
            "x": {"base_cost": 50.0},
            "y": {"base_cost": 5.0},
            "z": {"base_cost": 30.0},
        }
        obs = _make_obs(routes=routes)
        assert s.choose_action(obs) == "y"

    def test_fallback_to_available_actions(self) -> None:
        s = SelfishRouter()
        obs = _make_obs(
            routes={},
            available_actions=["path_1", "path_2"],
        )
        assert s.choose_action(obs) == "path_1"

    def test_name(self) -> None:
        assert SelfishRouter().name == "selfish_router"


class TestSocialOptimum:
    def test_picks_any_on_first_round(self) -> None:
        s = SocialOptimum()
        obs = _make_obs()
        action = s.choose_action(obs)
        assert action in ("route_a", "route_b", "route_c")

    def test_picks_least_used(self) -> None:
        s = SocialOptimum()
        history = [
            RoundResult(
                round_number=1,
                actions={
                    "player_0": "route_a",
                    "player_1": "route_a",
                },
                payoffs={"player_0": -10.0, "player_1": -10.0},
            ),
            RoundResult(
                round_number=2,
                actions={
                    "player_0": "route_b",
                    "player_1": "route_a",
                },
                payoffs={"player_0": -20.0, "player_1": -10.0},
            ),
        ]
        obs = _make_obs(history=history)
        # route_a used 3 times, route_b 1 time, route_c 0
        assert s.choose_action(obs) == "route_c"

    def test_name(self) -> None:
        assert SocialOptimum().name == "social_optimum"


class TestEpsilonGreedy:
    def test_explores_randomly(self) -> None:
        """With epsilon=1.0, always explores."""
        s = EpsilonGreedy(epsilon=1.0, seed=42)
        obs = _make_obs()
        actions = {s.choose_action(obs) for _ in range(30)}
        # Should eventually pick multiple routes
        assert len(actions) >= 2

    def test_exploits_best(self) -> None:
        """With epsilon=0.0, always exploits."""
        s = EpsilonGreedy(epsilon=0.0, seed=42)
        history = [
            RoundResult(
                round_number=1,
                actions={"player_0": "route_a"},
                payoffs={"player_0": -10.0},
            ),
            RoundResult(
                round_number=2,
                actions={"player_0": "route_b"},
                payoffs={"player_0": -5.0},
            ),
            RoundResult(
                round_number=3,
                actions={"player_0": "route_c"},
                payoffs={"player_0": -15.0},
            ),
        ]
        obs = _make_obs(history=history)
        # route_b has highest payoff (-5 > -10 > -15)
        assert s.choose_action(obs) == "route_b"

    def test_invalid_epsilon(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="epsilon must be"):
            EpsilonGreedy(epsilon=1.5)

    def test_name_includes_epsilon(self) -> None:
        assert EpsilonGreedy(epsilon=0.1).name == "epsilon_greedy_0.1"

    def test_first_round_random(self) -> None:
        """With no history and epsilon=0, picks randomly."""
        s = EpsilonGreedy(epsilon=0.0, seed=42)
        obs = _make_obs()
        action = s.choose_action(obs)
        assert action in ("route_a", "route_b", "route_c")
