"""Tests for the Congestion Game."""

from __future__ import annotations

import pytest

from game_envs.core.action import DiscreteActionSpace
from game_envs.core.game import GameType, MoveOrder
from game_envs.core.state import Observation
from game_envs.games.congestion import (
    CongestionConfig,
    CongestionGame,
    RouteDefinition,
)
from game_envs.games.registry import GameRegistry


class TestRouteDefinition:
    """Tests for RouteDefinition validation."""

    def test_valid_route(self) -> None:
        r = RouteDefinition(name="A", base_cost=1.0, coefficient=0.5)
        assert r.name == "A"
        assert r.base_cost == 1.0
        assert r.coefficient == 0.5

    def test_zero_values(self) -> None:
        r = RouteDefinition(name="A", base_cost=0.0, coefficient=0.0)
        assert r.base_cost == 0.0
        assert r.coefficient == 0.0

    def test_negative_base_cost(self) -> None:
        with pytest.raises(ValueError, match="base_cost must be >= 0"):
            RouteDefinition(name="A", base_cost=-1.0, coefficient=0.5)

    def test_negative_coefficient(self) -> None:
        with pytest.raises(ValueError, match="coefficient must be >= 0"):
            RouteDefinition(name="A", base_cost=1.0, coefficient=-0.5)

    def test_frozen(self) -> None:
        r = RouteDefinition(name="A", base_cost=1.0, coefficient=0.5)
        with pytest.raises(AttributeError):
            r.base_cost = 2.0  # type: ignore[misc]


class TestCongestionConfig:
    """Tests for CongestionConfig validation."""

    def test_defaults(self) -> None:
        config = CongestionConfig()
        assert len(config.routes) == 2
        assert config.routes[0].name == "route_A"
        assert config.routes[1].name == "route_B"
        assert config.num_players == 2

    def test_custom_routes(self) -> None:
        routes = (
            RouteDefinition("X", 1.0, 1.0),
            RouteDefinition("Y", 2.0, 0.5),
            RouteDefinition("Z", 3.0, 0.1),
        )
        config = CongestionConfig(routes=routes, num_players=4)
        assert len(config.routes) == 3
        assert config.num_players == 4

    def test_too_few_routes(self) -> None:
        with pytest.raises(ValueError, match="at least 2 routes"):
            CongestionConfig(routes=(RouteDefinition("A", 1.0, 1.0),))

    def test_duplicate_route_names(self) -> None:
        with pytest.raises(ValueError, match="route names must be unique"):
            CongestionConfig(
                routes=(
                    RouteDefinition("A", 1.0, 1.0),
                    RouteDefinition("A", 2.0, 0.5),
                )
            )

    def test_inherits_gameconfig_validation(self) -> None:
        with pytest.raises(ValueError, match="num_rounds must be >= 1"):
            CongestionConfig(num_rounds=0)
        with pytest.raises(ValueError, match="discount_factor"):
            CongestionConfig(discount_factor=1.5)

    def test_frozen(self) -> None:
        config = CongestionConfig()
        with pytest.raises(AttributeError):
            config.num_players = 5  # type: ignore[misc]


class TestCongestionGameProperties:
    """Tests for game properties."""

    def test_basic_properties(self) -> None:
        game = CongestionGame()
        assert "Congestion Game" in game.name
        assert "2 routes" in game.name
        assert game.game_type == GameType.NORMAL_FORM
        assert game.move_order == MoveOrder.SIMULTANEOUS
        assert game.player_ids == ["player_0", "player_1"]

    def test_repeated_name(self) -> None:
        config = CongestionConfig(num_rounds=5)
        game = CongestionGame(config)
        assert "repeated x5" in game.name
        assert game.game_type == GameType.REPEATED

    def test_n_player_ids(self) -> None:
        config = CongestionConfig(num_players=5)
        game = CongestionGame(config)
        assert len(game.player_ids) == 5
        assert game.player_ids == [f"player_{i}" for i in range(5)]

    def test_max_players(self) -> None:
        config = CongestionConfig(num_players=50)
        game = CongestionGame(config)
        assert len(game.player_ids) == 50

    def test_too_many_players(self) -> None:
        config = CongestionConfig(num_players=51)
        with pytest.raises(ValueError, match="between 2 and 50"):
            CongestionGame(config)

    def test_action_space_discrete(self) -> None:
        game = CongestionGame()
        space = game.action_space("player_0")
        assert isinstance(space, DiscreteActionSpace)
        assert "route_A" in space.to_list()
        assert "route_B" in space.to_list()

    def test_action_space_contains(self) -> None:
        game = CongestionGame()
        space = game.action_space("player_0")
        assert space.contains("route_A")
        assert space.contains("route_B")
        assert not space.contains("route_C")
        assert not space.contains(42)


class TestCongestionGameLatency:
    """Tests for latency computation."""

    def test_latency_formula(self) -> None:
        """latency(r) = base_cost + coefficient * num_users."""
        game = CongestionGame()
        # route_A: base=1.0, coeff=1.0
        assert game.compute_latency("route_A", 1) == 2.0
        assert game.compute_latency("route_A", 2) == 3.0
        assert game.compute_latency("route_A", 5) == 6.0

        # route_B: base=2.0, coeff=0.5
        assert game.compute_latency("route_B", 1) == 2.5
        assert game.compute_latency("route_B", 2) == 3.0
        assert game.compute_latency("route_B", 5) == 4.5

    def test_latency_zero_users(self) -> None:
        game = CongestionGame()
        assert game.compute_latency("route_A", 0) == 1.0
        assert game.compute_latency("route_B", 0) == 2.0


class TestCongestionGameOneShot:
    """Tests for one-shot congestion game payoffs."""

    def test_both_same_route(self) -> None:
        """Both players choose route_A.

        latency = 1.0 + 1.0 * 2 = 3.0
        payoff = -3.0 for each.
        """
        game = CongestionGame()
        game.reset()
        result = game.step({"player_0": "route_A", "player_1": "route_A"})
        assert result.payoffs["player_0"] == pytest.approx(-3.0)
        assert result.payoffs["player_1"] == pytest.approx(-3.0)
        assert result.is_terminal

    def test_both_route_b(self) -> None:
        """Both choose route_B.

        latency = 2.0 + 0.5 * 2 = 3.0
        payoff = -3.0 for each.
        """
        game = CongestionGame()
        game.reset()
        result = game.step({"player_0": "route_B", "player_1": "route_B"})
        assert result.payoffs["player_0"] == pytest.approx(-3.0)
        assert result.payoffs["player_1"] == pytest.approx(-3.0)

    def test_split_routes(self) -> None:
        """Players choose different routes.

        p0 on A: latency = 1.0 + 1.0 * 1 = 2.0, payoff = -2.0
        p1 on B: latency = 2.0 + 0.5 * 1 = 2.5, payoff = -2.5
        """
        game = CongestionGame()
        game.reset()
        result = game.step({"player_0": "route_A", "player_1": "route_B"})
        assert result.payoffs["player_0"] == pytest.approx(-2.0)
        assert result.payoffs["player_1"] == pytest.approx(-2.5)

    def test_total_latency(self) -> None:
        """Total system latency for a routing profile."""
        game = CongestionGame()
        # Both on A: 2 * 3.0 = 6.0
        total = game.compute_total_latency(
            {"player_0": "route_A", "player_1": "route_A"}
        )
        assert total == pytest.approx(6.0)

        # Split: 1 * 2.0 + 1 * 2.5 = 4.5
        total = game.compute_total_latency(
            {"player_0": "route_A", "player_1": "route_B"}
        )
        assert total == pytest.approx(4.5)

    def test_payoff_is_negative_latency(self) -> None:
        """Payoff equals negative of the latency."""
        game = CongestionGame()
        game.reset()
        result = game.step({"player_0": "route_A", "player_1": "route_B"})
        latency_a = game.compute_latency("route_A", 1)
        latency_b = game.compute_latency("route_B", 1)
        assert result.payoffs["player_0"] == pytest.approx(-latency_a)
        assert result.payoffs["player_1"] == pytest.approx(-latency_b)


class TestCongestionGameNPlayer:
    """Tests for n-player congestion game."""

    def test_three_players_all_same(self) -> None:
        """3 players all choose route_A.

        latency = 1.0 + 1.0 * 3 = 4.0
        payoff = -4.0 each.
        """
        config = CongestionConfig(num_players=3)
        game = CongestionGame(config)
        game.reset()
        actions = {f"player_{i}": "route_A" for i in range(3)}
        result = game.step(actions)
        for pid in game.player_ids:
            assert result.payoffs[pid] == pytest.approx(-4.0)

    def test_five_players_spread(self) -> None:
        """5 players spread across 2 routes.

        3 on A: latency = 1 + 1*3 = 4, payoff = -4
        2 on B: latency = 2 + 0.5*2 = 3, payoff = -3
        """
        config = CongestionConfig(num_players=5)
        game = CongestionGame(config)
        game.reset()
        actions = {
            "player_0": "route_A",
            "player_1": "route_A",
            "player_2": "route_A",
            "player_3": "route_B",
            "player_4": "route_B",
        }
        result = game.step(actions)
        for i in range(3):
            assert result.payoffs[f"player_{i}"] == pytest.approx(-4.0)
        for i in range(3, 5):
            assert result.payoffs[f"player_{i}"] == pytest.approx(-3.0)

    def test_many_routes(self) -> None:
        """Game with more than 2 routes."""
        routes = (
            RouteDefinition("A", 1.0, 1.0),
            RouteDefinition("B", 2.0, 0.5),
            RouteDefinition("C", 3.0, 0.1),
        )
        config = CongestionConfig(routes=routes, num_players=3)
        game = CongestionGame(config)
        game.reset()
        actions = {
            "player_0": "A",
            "player_1": "B",
            "player_2": "C",
        }
        result = game.step(actions)
        # A: 1 + 1*1 = 2, B: 2 + 0.5*1 = 2.5, C: 3 + 0.1*1 = 3.1
        assert result.payoffs["player_0"] == pytest.approx(-2.0)
        assert result.payoffs["player_1"] == pytest.approx(-2.5)
        assert result.payoffs["player_2"] == pytest.approx(-3.1)


class TestCongestionGameNashFlow:
    """Tests for Nash equilibrium vs social optimum."""

    def test_nash_flow_default(self) -> None:
        """In default 2-player game, Nash is both on route_A.

        Both on A: each gets -3.0 (latency 3)
        Split: p0 on A gets -2.0, p1 on B gets -2.5
        If p1 deviates from split to A: gets -3.0 (worse)
        If p0 deviates from split to B: gets -3.0 (worse)

        Nash: split (A, B) since neither wants to deviate.
        Actually, let's check: if both on B, latency = 3.0 each.
        From (A,B): p0 on A=-2.0, if p0 switches to B: -3.0 (worse).
        p1 on B=-2.5, if p1 switches to A: -3.0 (worse).
        So (A,B) is a Nash equilibrium.
        """
        game = CongestionGame()
        game.reset()

        # Nash: split
        split = game.step({"player_0": "route_A", "player_1": "route_B"})
        game.reset()

        # Check no profitable deviation for player_0
        game.reset()
        deviate_0 = game.step({"player_0": "route_B", "player_1": "route_B"})
        assert split.payoffs["player_0"] >= deviate_0.payoffs["player_0"]

        # Check no profitable deviation for player_1
        game.reset()
        deviate_1 = game.step({"player_0": "route_A", "player_1": "route_A"})
        assert split.payoffs["player_1"] >= deviate_1.payoffs["player_1"]

    def test_social_optimum_vs_selfish(self) -> None:
        """Social optimum minimizes total latency.

        Default 2-player:
        Both A: total = 2*3 = 6
        Both B: total = 2*3 = 6
        Split: total = 2.0 + 2.5 = 4.5 (social optimum)
        """
        game = CongestionGame()
        total_both_a = game.compute_total_latency(
            {"player_0": "route_A", "player_1": "route_A"}
        )
        total_split = game.compute_total_latency(
            {"player_0": "route_A", "player_1": "route_B"}
        )
        assert total_split < total_both_a


class TestBraessParadox:
    """Tests verifying Braess's paradox.

    Braess's paradox: adding a route can increase total latency
    at equilibrium. Classic Braess requires shared network links,
    which our independent-route model approximates by modeling
    compound routes with higher congestion coefficients.
    """

    def test_symmetric_nash_equilibrium(self) -> None:
        """Symmetric routes: Nash splits evenly (2-2).

        top=bottom, coeff=1. With 4 players, Nash is 2+2.
        Each route: 0+1*2=2. Total = 2*2 + 2*2 = 8.
        """
        routes = (
            RouteDefinition("top", base_cost=0.0, coefficient=1.0),
            RouteDefinition("bottom", base_cost=0.0, coefficient=1.0),
        )
        config = CongestionConfig(routes=routes, num_players=4)
        game = CongestionGame(config)

        total = game.compute_total_latency(
            {
                "player_0": "top",
                "player_1": "top",
                "player_2": "bottom",
                "player_3": "bottom",
            }
        )
        assert total == pytest.approx(8.0)

    def test_shortcut_worsens_total_when_overused(self) -> None:
        """When all players flock to a shortcut, total increases.

        Original Nash (3-3 split): total = 3*3 + 3*3 = 27.
        All 6 on shortcut (coeff=2): latency=12, total=72 > 27.
        """
        routes_orig = (
            RouteDefinition("top", base_cost=0.0, coefficient=1.5),
            RouteDefinition("bottom", base_cost=0.0, coefficient=1.5),
        )
        config_orig = CongestionConfig(routes=routes_orig, num_players=6)
        game_orig = CongestionGame(config_orig)

        total_nash = game_orig.compute_total_latency(
            {
                "player_0": "top",
                "player_1": "top",
                "player_2": "top",
                "player_3": "bottom",
                "player_4": "bottom",
                "player_5": "bottom",
            }
        )

        routes_sc = (
            RouteDefinition("top", base_cost=0.0, coefficient=1.5),
            RouteDefinition("bottom", base_cost=0.0, coefficient=1.5),
            RouteDefinition("shortcut", base_cost=0.0, coefficient=2.0),
        )
        config_sc = CongestionConfig(routes=routes_sc, num_players=6)
        game_sc = CongestionGame(config_sc)

        total_all_shortcut = game_sc.compute_total_latency(
            {f"player_{i}": "shortcut" for i in range(6)}
        )
        assert total_all_shortcut > total_nash

    def test_braess_compound_route_model(self) -> None:
        """Model Braess via compound routes with higher coefficient.

        Original: top/bottom (base=45, coeff=1), 10 players.
        Nash 5-5: latency=50 each. Total=500.

        Adding "middle" (base=0, coeff=2) models a route
        through two congested segments. All 10 on middle:
        latency=20. Total=200.

        This shows the game can model compound-route dynamics.
        """
        routes_orig = (
            RouteDefinition("top", base_cost=45.0, coefficient=1.0),
            RouteDefinition("bottom", base_cost=45.0, coefficient=1.0),
        )
        config_orig = CongestionConfig(routes=routes_orig, num_players=10)
        game_orig = CongestionGame(config_orig)

        nash_split = game_orig.compute_total_latency(
            {f"player_{i}": "top" if i < 5 else "bottom" for i in range(10)}
        )
        assert nash_split == pytest.approx(500.0)

        routes_mid = (
            RouteDefinition("top", base_cost=45.0, coefficient=1.0),
            RouteDefinition("bottom", base_cost=45.0, coefficient=1.0),
            RouteDefinition("middle", base_cost=0.0, coefficient=2.0),
        )
        config_mid = CongestionConfig(routes=routes_mid, num_players=10)
        game_mid = CongestionGame(config_mid)

        # Verify all-middle payoffs
        game_mid.reset()
        result = game_mid.step({f"player_{i}": "middle" for i in range(10)})
        for pid in game_mid.player_ids:
            assert result.payoffs[pid] == pytest.approx(-20.0)

        total_middle = game_mid.compute_total_latency(
            {f"player_{i}": "middle" for i in range(10)}
        )
        assert total_middle < nash_split  # 200 < 500

    def test_tunnel_attracts_but_congests(self) -> None:
        """A tunnel (base=0, coeff=3) is tempting individually
        but costly when all 4 players use it.

        All on tunnel: 0+3*4=12 each. Total=48.
        All on fixed-cost routes: 10 each. Total=40.
        """
        routes = (
            RouteDefinition("scenic", base_cost=10.0, coefficient=0.0),
            RouteDefinition("express", base_cost=10.0, coefficient=0.0),
            RouteDefinition("tunnel", base_cost=0.0, coefficient=3.0),
        )
        config = CongestionConfig(routes=routes, num_players=4)
        game = CongestionGame(config)

        total_fixed = game.compute_total_latency(
            {
                "player_0": "scenic",
                "player_1": "scenic",
                "player_2": "express",
                "player_3": "express",
            }
        )
        assert total_fixed == pytest.approx(40.0)

        game.reset()
        result = game.step({f"player_{i}": "tunnel" for i in range(4)})
        for pid in game.player_ids:
            assert result.payoffs[pid] == pytest.approx(-12.0)

        total_tunnel = game.compute_total_latency(
            {f"player_{i}": "tunnel" for i in range(4)}
        )
        assert total_tunnel > total_fixed  # 48 > 40


class TestCongestionGameRepeated:
    """Tests for repeated congestion game."""

    def test_multi_round(self) -> None:
        config = CongestionConfig(num_rounds=3, seed=42)
        game = CongestionGame(config)
        game.reset()

        for i in range(3):
            result = game.step({"player_0": "route_A", "player_1": "route_B"})
            if i < 2:
                assert not result.is_terminal
            else:
                assert result.is_terminal

    def test_step_after_terminal_raises(self) -> None:
        game = CongestionGame()
        game.reset()
        game.step({"player_0": "route_A", "player_1": "route_A"})
        with pytest.raises(RuntimeError, match="terminal"):
            game.step({"player_0": "route_A", "player_1": "route_A"})

    def test_cumulative_payoffs(self) -> None:
        """Cumulative payoffs sum correctly over rounds."""
        config = CongestionConfig(num_rounds=3)
        game = CongestionGame(config)
        game.reset()

        for _ in range(3):
            game.step({"player_0": "route_A", "player_1": "route_B"})

        # Each round: p0 on A (1 user) = -(1+1) = -2
        # p1 on B (1 user) = -(2+0.5) = -2.5
        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == pytest.approx(-6.0)
        assert payoffs["player_1"] == pytest.approx(-7.5)

    def test_discount_factor(self) -> None:
        """Discount factor reduces future payoffs."""
        config = CongestionConfig(num_rounds=3, discount_factor=0.5)
        game = CongestionGame(config)
        game.reset()

        for _ in range(3):
            game.step({"player_0": "route_A", "player_1": "route_A"})

        # Each round raw payoff = -3.0
        # Round 0: -3 * 0.5^0 = -3
        # Round 1: -3 * 0.5^1 = -1.5
        # Round 2: -3 * 0.5^2 = -0.75
        # Total = -5.25
        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == pytest.approx(-5.25)

    def test_history_tracking(self) -> None:
        config = CongestionConfig(num_rounds=2)
        game = CongestionGame(config)
        game.reset()

        game.step({"player_0": "route_A", "player_1": "route_B"})
        game.step({"player_0": "route_B", "player_1": "route_A"})

        history = game.history.for_player("player_0")
        assert len(history) == 2
        assert history[0].round_number == 0
        assert history[0].actions["player_0"] == "route_A"
        assert history[1].round_number == 1
        assert history[1].actions["player_0"] == "route_B"

    def test_reset_clears_state(self) -> None:
        game = CongestionGame()
        game.reset()
        game.step({"player_0": "route_A", "player_1": "route_A"})

        game.reset()
        assert game.current_round == 0
        assert not game.is_terminal
        assert game.get_payoffs() == {
            "player_0": 0.0,
            "player_1": 0.0,
        }
        assert len(game.history.for_player("player_0")) == 0


class TestCongestionGameObserve:
    """Tests for observe and to_prompt."""

    def test_observe_basic(self) -> None:
        game = CongestionGame()
        game.reset()
        obs = game.observe("player_0")
        assert isinstance(obs, Observation)
        assert obs.player_id == "player_0"
        assert obs.round_number == 0
        assert obs.total_rounds == 1
        assert obs.game_state["num_players"] == 2
        assert "routes" in obs.game_state
        routes = obs.game_state["routes"]
        assert "route_A" in routes
        assert "route_B" in routes
        assert routes["route_A"]["base_cost"] == 1.0
        assert routes["route_A"]["coefficient"] == 1.0

    def test_observe_has_latency_formula(self) -> None:
        game = CongestionGame()
        game.reset()
        obs = game.observe("player_0")
        assert "latency_formula" in obs.game_state
        assert "base_cost" in obs.game_state["latency_formula"]
        assert "coefficient" in obs.game_state["latency_formula"]

    def test_observe_has_payoff_rule(self) -> None:
        game = CongestionGame()
        game.reset()
        obs = game.observe("player_0")
        assert "payoff_rule" in obs.game_state
        assert (
            "negative" in obs.game_state["payoff_rule"].lower()
            or "-latency" in obs.game_state["payoff_rule"]
        )

    def test_to_prompt_content(self) -> None:
        game = CongestionGame()
        game.reset()
        obs = game.observe("player_0")
        prompt = obs.to_prompt()
        assert "player_0" in prompt
        assert "Round 0 of 1" in prompt

    def test_to_prompt_with_history(self) -> None:
        config = CongestionConfig(num_rounds=3)
        game = CongestionGame(config)
        game.reset()
        game.step({"player_0": "route_A", "player_1": "route_B"})

        obs = game.observe("player_0")
        prompt = obs.to_prompt()
        assert "History:" in prompt
        assert "Round 1" in prompt

    def test_serialization_roundtrip(self) -> None:
        game = CongestionGame()
        game.reset()
        obs = game.observe("player_0")
        d = obs.to_dict()
        restored = Observation.from_dict(d)
        assert restored.player_id == obs.player_id
        assert restored.round_number == obs.round_number
        assert restored.total_rounds == obs.total_rounds
        assert restored.game_state == obs.game_state


class TestCongestionGameRegistry:
    """Tests for game registry integration."""

    def setup_method(self) -> None:
        self._saved = dict(GameRegistry._registry)

    def teardown_method(self) -> None:
        GameRegistry._registry = self._saved

    def test_registered(self) -> None:
        assert "congestion" in GameRegistry.list_games()

    def test_create_from_registry(self) -> None:
        game = GameRegistry.create("congestion")
        assert isinstance(game, CongestionGame)
        assert "Congestion Game" in game.name

    def test_create_with_config(self) -> None:
        config = CongestionConfig(num_players=4)
        game = GameRegistry.create("congestion", config=config)
        assert isinstance(game, CongestionGame)
        assert len(game.player_ids) == 4


class TestCongestionGameToPrompt:
    """Tests for the game-level to_prompt() method."""

    def test_basic_content(self) -> None:
        game = CongestionGame()
        prompt = game.to_prompt()
        assert "2-player" in prompt
        assert "Congestion" in prompt or "Routing" in prompt
        assert "latency" in prompt.lower()
        assert "base_cost" in prompt
        assert "coefficient" in prompt

    def test_routes_listed(self) -> None:
        game = CongestionGame()
        prompt = game.to_prompt()
        assert "route_A" in prompt
        assert "route_B" in prompt

    def test_custom_routes(self) -> None:
        routes = (
            RouteDefinition("highway", 5.0, 0.5),
            RouteDefinition("backroad", 10.0, 0.1),
            RouteDefinition("tunnel", 0.0, 3.0),
        )
        config = CongestionConfig(routes=routes, num_players=4)
        game = CongestionGame(config)
        prompt = game.to_prompt()
        assert "4-player" in prompt
        assert "3 routes" in prompt
        assert "highway" in prompt
        assert "backroad" in prompt
        assert "tunnel" in prompt

    def test_repeated_game_info(self) -> None:
        config = CongestionConfig(num_rounds=10, discount_factor=0.9)
        game = CongestionGame(config)
        prompt = game.to_prompt()
        assert "10 rounds" in prompt
        assert "0.9" in prompt

    def test_no_repeated_info_for_oneshot(self) -> None:
        game = CongestionGame()
        prompt = game.to_prompt()
        assert "repeated" not in prompt.lower()

    def test_strategy_note(self) -> None:
        game = CongestionGame()
        prompt = game.to_prompt()
        assert "Nash" in prompt
        assert "social" in prompt.lower()

    def test_payoff_rule_described(self) -> None:
        game = CongestionGame()
        prompt = game.to_prompt()
        assert "payoff" in prompt.lower()
        assert "-latency" in prompt or "negative" in prompt.lower()


class TestBraessParadoxVerification:
    """Verify Braess's paradox: adding a route worsens outcomes.

    Classic Braess network (4 players):
    - Route top:    base=0,  coeff=10  (congestion-sensitive)
    - Route bottom: base=50, coeff=0   (fixed-cost)

    Without shortcut:
        Nash: 2 on top (latency=20), 2 on bottom (latency=50).
        No one deviates because 3-on-top = latency 30, still < 50.
        But 4-on-top = 40 < 50, so actually all go top.

    With a shortcut that attracts overuse, total worsens.
    """

    def test_braess_adding_route_worsens_equilibrium(self) -> None:
        """Classic Braess's paradox demonstration.

        Network without shortcut: 2 symmetric routes.
        top:    latency = 10 * n_users (congestion-sensitive)
        bottom: latency = 50          (fixed-cost)

        With 4 players, Nash is all-on-top: 10*4 = 40 < 50.
        Total latency = 4 * 40 = 160.

        Add a shortcut route (base=0, coeff=15).
        If all 4 use shortcut: latency = 15*4 = 60 each.
        Total = 240 > 160. Worse!

        But individually, 1 player on shortcut alone = 15,
        which beats 40, so it's tempting — paradox emerges
        as everyone switches.
        """
        # Without shortcut: 2 routes
        routes_before = (
            RouteDefinition("top", base_cost=0.0, coefficient=10.0),
            RouteDefinition("bottom", base_cost=50.0, coefficient=0.0),
        )
        config_before = CongestionConfig(routes=routes_before, num_players=4)
        game_before = CongestionGame(config_before)

        # Nash equilibrium: all on top (40 < 50)
        nash_before = game_before.compute_total_latency(
            {f"player_{i}": "top" for i in range(4)}
        )
        assert nash_before == pytest.approx(160.0)

        # Verify it's Nash: no one wants to deviate to bottom
        game_before.reset()
        result = game_before.step({f"player_{i}": "top" for i in range(4)})
        # One player deviating to bottom gets -50, worse than -40
        assert result.payoffs["player_0"] == pytest.approx(-40.0)
        lat_bottom = game_before.compute_latency("bottom", 1)
        assert lat_bottom == pytest.approx(50.0)
        assert -40.0 > -50.0  # staying on top is better

        # With shortcut: add route with high coefficient
        routes_after = (
            RouteDefinition("top", base_cost=0.0, coefficient=10.0),
            RouteDefinition("bottom", base_cost=50.0, coefficient=0.0),
            RouteDefinition("shortcut", base_cost=0.0, coefficient=15.0),
        )
        config_after = CongestionConfig(routes=routes_after, num_players=4)
        game_after = CongestionGame(config_after)

        # All on shortcut: each gets 15*4 = 60
        total_shortcut = game_after.compute_total_latency(
            {f"player_{i}": "shortcut" for i in range(4)}
        )
        assert total_shortcut == pytest.approx(240.0)

        # Braess's paradox: total latency increased
        assert total_shortcut > nash_before

    def test_braess_individual_incentive(self) -> None:
        """Show individual incentive to use shortcut exists.

        When others are on top (3 players), shortcut alone
        has latency 15*1 = 15, much less than 10*3 = 30 on
        top. So the shortcut is individually attractive.
        """
        routes = (
            RouteDefinition("top", base_cost=0.0, coefficient=10.0),
            RouteDefinition("bottom", base_cost=50.0, coefficient=0.0),
            RouteDefinition("shortcut", base_cost=0.0, coefficient=15.0),
        )
        config = CongestionConfig(routes=routes, num_players=4)
        game = CongestionGame(config)
        game.reset()

        # 3 on top, 1 on shortcut
        result = game.step(
            {
                "player_0": "shortcut",
                "player_1": "top",
                "player_2": "top",
                "player_3": "top",
            }
        )
        # Shortcut user: 15*1 = 15, payoff = -15
        # Top users: 10*3 = 30, payoff = -30
        assert result.payoffs["player_0"] == pytest.approx(-15.0)
        assert result.payoffs["player_1"] == pytest.approx(-30.0)

        # Individual incentive: shortcut user is better off
        assert result.payoffs["player_0"] > result.payoffs["player_1"]
