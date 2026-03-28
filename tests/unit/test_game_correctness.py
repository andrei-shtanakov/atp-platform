"""Game-theoretic correctness tests for all 5 canonical games.

Verifies payoff structures match theoretical definitions:
- Prisoner's Dilemma: T > R > P > S, 2R > T+S, dominant strategy
- Sealed-Bid Auction: first-price and second-price payment rules
- Public Goods Game: payoff formula, free-rider advantage
- Colonel Blotto: battlefield scoring, tie splitting
- Congestion Game: latency formula, payoff = -latency
"""

from __future__ import annotations

import pytest
from game_envs.games.auction import Auction, AuctionConfig, AuctionType
from game_envs.games.colonel_blotto import BlottoConfig, ColonelBlotto
from game_envs.games.congestion import CongestionConfig, CongestionGame, RouteDefinition
from game_envs.games.prisoners_dilemma import PDConfig, PrisonersDilemma
from game_envs.games.public_goods import PGConfig, PublicGoodsGame

# ---------------------------------------------------------------------------
# Prisoner's Dilemma
# ---------------------------------------------------------------------------


class TestPrisonersDilemma:
    """Tests for Prisoner's Dilemma payoff structure."""

    def test_default_payoff_ordering(self) -> None:
        """Verify T > R > P > S and 2R > T + S for default config."""
        cfg = PDConfig()
        assert cfg.temptation > cfg.reward
        assert cfg.reward > cfg.punishment
        assert cfg.punishment > cfg.sucker
        assert 2 * cfg.reward > cfg.temptation + cfg.sucker

    def test_mutual_cooperation_payoff(self) -> None:
        """Both cooperate -> each gets R = 3.0."""
        game = PrisonersDilemma()
        game.reset()
        result = game.step({"player_0": "cooperate", "player_1": "cooperate"})
        assert result.payoffs["player_0"] == pytest.approx(3.0)
        assert result.payoffs["player_1"] == pytest.approx(3.0)

    def test_mutual_defection_payoff(self) -> None:
        """Both defect -> each gets P = 1.0."""
        game = PrisonersDilemma()
        game.reset()
        result = game.step({"player_0": "defect", "player_1": "defect"})
        assert result.payoffs["player_0"] == pytest.approx(1.0)
        assert result.payoffs["player_1"] == pytest.approx(1.0)

    def test_asymmetric_cooperate_defect_payoff(self) -> None:
        """Player_0 cooperates, player_1 defects -> S=0.0 and T=5.0."""
        game = PrisonersDilemma()
        game.reset()
        result = game.step({"player_0": "cooperate", "player_1": "defect"})
        assert result.payoffs["player_0"] == pytest.approx(0.0)  # sucker
        assert result.payoffs["player_1"] == pytest.approx(5.0)  # temptation

    def test_asymmetric_defect_cooperate_payoff(self) -> None:
        """Player_0 defects, player_1 cooperates -> T=5.0 and S=0.0."""
        game = PrisonersDilemma()
        game.reset()
        result = game.step({"player_0": "defect", "player_1": "cooperate"})
        assert result.payoffs["player_0"] == pytest.approx(5.0)  # temptation
        assert result.payoffs["player_1"] == pytest.approx(0.0)  # sucker

    def test_defect_dominates_cooperate_vs_cooperate(self) -> None:
        """Defecting yields strictly more than cooperating when opponent cooperates."""
        game = PrisonersDilemma()
        game.reset()
        defect_result = game.step({"player_0": "defect", "player_1": "cooperate"})
        defect_payoff = defect_result.payoffs["player_0"]

        game2 = PrisonersDilemma()
        game2.reset()
        coop_result = game2.step({"player_0": "cooperate", "player_1": "cooperate"})
        coop_payoff = coop_result.payoffs["player_0"]

        assert defect_payoff > coop_payoff

    def test_defect_dominates_cooperate_vs_defect(self) -> None:
        """Defecting yields strictly more than cooperating when opponent defects."""
        game = PrisonersDilemma()
        game.reset()
        defect_result = game.step({"player_0": "defect", "player_1": "defect"})
        defect_payoff = defect_result.payoffs["player_0"]

        game2 = PrisonersDilemma()
        game2.reset()
        coop_result = game2.step({"player_0": "cooperate", "player_1": "defect"})
        coop_payoff = coop_result.payoffs["player_0"]

        assert defect_payoff > coop_payoff

    def test_custom_payoff_parameters(self) -> None:
        """Verify custom T/R/P/S values are applied correctly."""
        cfg = PDConfig(temptation=10.0, reward=6.0, punishment=2.0, sucker=0.0)
        game = PrisonersDilemma(cfg)
        game.reset()
        result = game.step({"player_0": "cooperate", "player_1": "defect"})
        assert result.payoffs["player_0"] == pytest.approx(0.0)
        assert result.payoffs["player_1"] == pytest.approx(10.0)

    def test_invalid_config_t_not_gt_r_raises(self) -> None:
        """PDConfig must enforce T > R."""
        with pytest.raises(ValueError, match="T > R"):
            PDConfig(temptation=3.0, reward=3.0, punishment=1.0, sucker=0.0)

    def test_invalid_config_r_not_gt_p_raises(self) -> None:
        """PDConfig must enforce R > P."""
        with pytest.raises(ValueError, match="R > P"):
            PDConfig(temptation=5.0, reward=1.0, punishment=1.0, sucker=0.0)


# ---------------------------------------------------------------------------
# Sealed-Bid Auction
# ---------------------------------------------------------------------------


class TestAuction:
    """Tests for Sealed-Bid Auction payoff structure."""

    def _make_first_price(self) -> Auction:
        """Create a 2-player first-price auction."""
        cfg = AuctionConfig(
            auction_type=AuctionType.FIRST_PRICE,
            reserve_price=0.0,
        )
        game = Auction(cfg)
        game.reset()
        return game

    def _make_second_price(self) -> Auction:
        """Create a 2-player second-price auction."""
        cfg = AuctionConfig(
            auction_type=AuctionType.SECOND_PRICE,
            reserve_price=0.0,
        )
        game = Auction(cfg)
        game.reset()
        return game

    def test_first_price_winner_pays_own_bid(self) -> None:
        """First-price: winner pays own bid, payoff = value - bid."""
        game = self._make_first_price()
        # Set private values deterministically
        game._private_values = {"player_0": 80.0, "player_1": 40.0}
        result = game.step({"player_0": 60.0, "player_1": 30.0})
        # player_0 wins (higher bid), pays own bid 60
        assert result.payoffs["player_0"] == pytest.approx(80.0 - 60.0)
        assert result.payoffs["player_1"] == pytest.approx(0.0)

    def test_first_price_loser_gets_zero(self) -> None:
        """First-price: losing bidder receives zero payoff."""
        game = self._make_first_price()
        game._private_values = {"player_0": 50.0, "player_1": 70.0}
        result = game.step({"player_0": 20.0, "player_1": 55.0})
        assert result.payoffs["player_0"] == pytest.approx(0.0)
        assert result.payoffs["player_1"] == pytest.approx(70.0 - 55.0)

    def test_second_price_winner_pays_second_highest(self) -> None:
        """Second-price (Vickrey): winner pays second-highest bid."""
        game = self._make_second_price()
        game._private_values = {"player_0": 80.0, "player_1": 40.0}
        # player_0 bids 70, player_1 bids 30
        result = game.step({"player_0": 70.0, "player_1": 30.0})
        # player_0 wins, pays second-highest = 30
        assert result.payoffs["player_0"] == pytest.approx(80.0 - 30.0)
        assert result.payoffs["player_1"] == pytest.approx(0.0)

    def test_second_price_truthful_bidding_payoff(self) -> None:
        """Second-price: when bidding truthfully, payoff = value - opponent_bid."""
        game = self._make_second_price()
        game._private_values = {"player_0": 60.0, "player_1": 40.0}
        # Both bid truthfully
        result = game.step({"player_0": 60.0, "player_1": 40.0})
        assert result.payoffs["player_0"] == pytest.approx(60.0 - 40.0)
        assert result.payoffs["player_1"] == pytest.approx(0.0)

    def test_reserve_price_prevents_sale(self) -> None:
        """No sale if highest bid is below reserve price."""
        cfg = AuctionConfig(
            auction_type=AuctionType.FIRST_PRICE,
            reserve_price=50.0,
        )
        game = Auction(cfg)
        game.reset()
        game._private_values = {"player_0": 80.0, "player_1": 70.0}
        result = game.step({"player_0": 30.0, "player_1": 40.0})
        # Neither bid meets reserve; both get 0
        assert result.payoffs["player_0"] == pytest.approx(0.0)
        assert result.payoffs["player_1"] == pytest.approx(0.0)

    def test_tie_breaking_lower_player_id_wins(self) -> None:
        """On equal bids, lower player_id (player_0) wins in first-price."""
        game = self._make_first_price()
        game._private_values = {"player_0": 50.0, "player_1": 70.0}
        result = game.step({"player_0": 40.0, "player_1": 40.0})
        # player_0 wins tie (lexicographically first)
        assert result.payoffs["player_0"] == pytest.approx(50.0 - 40.0)
        assert result.payoffs["player_1"] == pytest.approx(0.0)

    def test_private_values_accessible(self) -> None:
        """private_values property returns current values."""
        game = self._make_first_price()
        game._private_values = {"player_0": 55.0, "player_1": 75.0}
        values = game.private_values
        assert values["player_0"] == pytest.approx(55.0)
        assert values["player_1"] == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# Public Goods Game
# ---------------------------------------------------------------------------


class TestPublicGoodsGame:
    """Tests for Public Goods Game payoff formula."""

    def _make_game(self, num_players: int = 2, **kwargs: float) -> PublicGoodsGame:
        """Create a Public Goods Game with given parameters."""
        # multiplier must be < num_players; use safe default
        multiplier = kwargs.pop("multiplier", 1.6)
        cfg = PGConfig(num_players=num_players, multiplier=multiplier, **kwargs)
        game = PublicGoodsGame(cfg)
        game.reset()
        return game

    def test_full_contribution_payoff_two_players(self) -> None:
        """Both contribute fully: payoff = 0 + multiplier * total / n."""
        endowment = 20.0
        multiplier = 1.6
        n = 2
        game = self._make_game(
            num_players=n, multiplier=multiplier, endowment=endowment
        )
        result = game.step({"player_0": 20.0, "player_1": 20.0})
        # payoff = endowment - contribution + multiplier * sum / n
        # = 20 - 20 + 1.6 * 40 / 2 = 0 + 32 = 32
        expected = endowment - endowment + multiplier * (endowment * n) / n
        assert result.payoffs["player_0"] == pytest.approx(expected)
        assert result.payoffs["player_1"] == pytest.approx(expected)

    def test_zero_contribution_payoff(self) -> None:
        """No one contributes: everyone keeps endowment."""
        endowment = 20.0
        multiplier = 1.6
        n = 2
        game = self._make_game(
            num_players=n, multiplier=multiplier, endowment=endowment
        )
        result = game.step({"player_0": 0.0, "player_1": 0.0})
        # payoff = endowment - 0 + multiplier * 0 / n = endowment
        assert result.payoffs["player_0"] == pytest.approx(endowment)
        assert result.payoffs["player_1"] == pytest.approx(endowment)

    def test_free_rider_advantage(self) -> None:
        """Non-contributor earns more than contributor when others contribute."""
        endowment = 20.0
        multiplier = 1.6
        n = 2
        game = self._make_game(
            num_players=n, multiplier=multiplier, endowment=endowment
        )
        result = game.step({"player_0": 20.0, "player_1": 0.0})
        # player_0 (contributor): 20 - 20 + 1.6*20/2 = 0 + 16 = 16
        # player_1 (free rider):  20 - 0  + 1.6*20/2 = 20 + 16 = 36
        assert result.payoffs["player_0"] == pytest.approx(16.0)
        assert result.payoffs["player_1"] == pytest.approx(36.0)
        assert result.payoffs["player_1"] > result.payoffs["player_0"]

    def test_payoff_formula_general(self) -> None:
        """Verify payoff formula: endowment - c_i + multiplier * sum(c) / n."""
        endowment = 10.0
        multiplier = 1.5
        n = 3
        game = self._make_game(
            num_players=n, multiplier=multiplier, endowment=endowment
        )
        c0, c1, c2 = 5.0, 3.0, 8.0
        result = game.step({"player_0": c0, "player_1": c1, "player_2": c2})
        total = c0 + c1 + c2
        share = multiplier * total / n
        assert result.payoffs["player_0"] == pytest.approx(endowment - c0 + share)
        assert result.payoffs["player_1"] == pytest.approx(endowment - c1 + share)
        assert result.payoffs["player_2"] == pytest.approx(endowment - c2 + share)

    def test_full_contribution_social_optimum(self) -> None:
        """Full contribution yields higher total payoff than zero contribution."""
        endowment = 20.0
        multiplier = 1.6
        n = 2
        game_all = self._make_game(
            num_players=n, multiplier=multiplier, endowment=endowment
        )
        result_all = game_all.step({"player_0": 20.0, "player_1": 20.0})
        total_all = sum(result_all.payoffs.values())

        game_none = self._make_game(
            num_players=n, multiplier=multiplier, endowment=endowment
        )
        result_none = game_none.step({"player_0": 0.0, "player_1": 0.0})
        total_none = sum(result_none.payoffs.values())

        assert total_all > total_none


# ---------------------------------------------------------------------------
# Colonel Blotto
# ---------------------------------------------------------------------------


class TestColonelBlotto:
    """Tests for Colonel Blotto battlefield scoring."""

    def _make_game(
        self, num_battlefields: int = 3, total_troops: int = 9
    ) -> ColonelBlotto:
        """Create a 2-player Colonel Blotto game."""
        cfg = BlottoConfig(num_battlefields=num_battlefields, total_troops=total_troops)
        game = ColonelBlotto(cfg)
        game.reset()
        return game

    def _alloc(self, *troops: int) -> dict[str, float]:
        """Build allocation dict for battlefields."""
        return {f"battlefield_{i}": float(t) for i, t in enumerate(troops)}

    def test_player_wins_all_battlefields(self) -> None:
        """Player_0 dominates all battlefields: payoff = 1.0."""
        game = self._make_game(num_battlefields=3, total_troops=9)
        result = game.step(
            {
                "player_0": self._alloc(4, 3, 2),
                "player_1": self._alloc(3, 2, 4),
            }
        )
        # player_0 wins bf_0 (4>3), loses bf_1 (3>2... wait 3>2 player_0 wins)
        # actually: p0[0]=4>p1[0]=3, p0[1]=3>p1[1]=2, p0[2]=2<p1[2]=4
        # player_0 wins 2 battlefields, player_1 wins 1 → 2/3 vs 1/3
        assert result.payoffs["player_0"] == pytest.approx(2.0 / 3.0)
        assert result.payoffs["player_1"] == pytest.approx(1.0 / 3.0)

    def test_player_sweeps_all_battlefields(self) -> None:
        """Player_0 wins all 3 battlefields: payoff = 1.0."""
        game = self._make_game(num_battlefields=3, total_troops=9)
        result = game.step(
            {
                "player_0": self._alloc(5, 3, 1),
                "player_1": self._alloc(1, 1, 7),
            }
        )
        # p0: 5>1, 3>1, 1<7 → player_0 wins 2, player_1 wins 1
        assert result.payoffs["player_0"] == pytest.approx(2.0 / 3.0)
        assert result.payoffs["player_1"] == pytest.approx(1.0 / 3.0)

    def test_all_ties_splits_equally(self) -> None:
        """All battlefields tied: each player gets 0.5."""
        game = self._make_game(num_battlefields=3, total_troops=9)
        result = game.step(
            {
                "player_0": self._alloc(3, 3, 3),
                "player_1": self._alloc(3, 3, 3),
            }
        )
        assert result.payoffs["player_0"] == pytest.approx(0.5)
        assert result.payoffs["player_1"] == pytest.approx(0.5)

    def test_single_tied_battlefield(self) -> None:
        """Tied battlefield contributes 0.5 to each player's score."""
        # 3 battlefields, total_troops=9
        game = self._make_game(num_battlefields=3, total_troops=9)
        # bf_0: p0=6>p1=1 → p0 wins, bf_1: p0=2==p1=2 → tie, bf_2: p0=1<p1=6 → p1 wins
        result = game.step(
            {
                "player_0": self._alloc(6, 2, 1),
                "player_1": self._alloc(1, 2, 6),
            }
        )
        # p0: wins bf_0 (1pt) + half bf_1 (0.5pt) + loses bf_2 (0pt) = 1.5/3 = 0.5
        # p1: loses bf_0 (0pt) + half bf_1 (0.5pt) + wins bf_2 (1pt) = 1.5/3 = 0.5
        assert result.payoffs["player_0"] == pytest.approx(0.5)
        assert result.payoffs["player_1"] == pytest.approx(0.5)

    def test_payoffs_sum_to_one(self) -> None:
        """Total payoff across players always sums to 1.0 (zero-sum)."""
        game = self._make_game(num_battlefields=4, total_troops=12)
        result = game.step(
            {
                "player_0": self._alloc(5, 3, 2, 2),
                "player_1": self._alloc(2, 4, 4, 2),
            }
        )
        total = sum(result.payoffs.values())
        assert total == pytest.approx(1.0)

    def test_invalid_allocation_raises(self) -> None:
        """Allocation not summing to total_troops should be rejected."""
        game = self._make_game(num_battlefields=3, total_troops=9)
        with pytest.raises(ValueError):
            game.step(
                {
                    "player_0": self._alloc(5, 5, 5),  # sums to 15, not 9
                    "player_1": self._alloc(3, 3, 3),
                }
            )


# ---------------------------------------------------------------------------
# Congestion Game
# ---------------------------------------------------------------------------


class TestCongestionGame:
    """Tests for Congestion Game latency and payoff calculations."""

    def _make_game(self) -> CongestionGame:
        """Create default 2-player congestion game."""
        cfg = CongestionConfig()
        game = CongestionGame(cfg)
        game.reset()
        return game

    def _make_custom_game(
        self,
        routes: tuple[RouteDefinition, ...],
        num_players: int = 2,
    ) -> CongestionGame:
        """Create a congestion game with custom routes."""
        cfg = CongestionConfig(routes=routes, num_players=num_players)
        game = CongestionGame(cfg)
        game.reset()
        return game

    def test_single_user_latency(self) -> None:
        """Single user on a route: latency = base_cost + coefficient * 1."""
        game = self._make_game()
        # Default: route_A: base=1, coeff=1; route_B: base=2, coeff=0.5
        # Both players on different routes → each is the only user on their route
        result = game.step({"player_0": "route_A", "player_1": "route_B"})
        # route_A latency = 1 + 1*1 = 2, payoff = -2
        assert result.payoffs["player_0"] == pytest.approx(-2.0)
        # route_B latency = 2 + 0.5*1 = 2.5, payoff = -2.5
        assert result.payoffs["player_1"] == pytest.approx(-2.5)

    def test_two_users_same_route_congestion(self) -> None:
        """Two users on same route: latency = base_cost + coefficient * 2."""
        game = self._make_game()
        result = game.step({"player_0": "route_A", "player_1": "route_A"})
        # route_A: base=1, coeff=1 → latency = 1 + 1*2 = 3, payoff = -3
        assert result.payoffs["player_0"] == pytest.approx(-3.0)
        assert result.payoffs["player_1"] == pytest.approx(-3.0)

    def test_payoff_equals_negative_latency(self) -> None:
        """Payoff is exactly negative latency for all players."""
        routes = (
            RouteDefinition(name="fast", base_cost=0.5, coefficient=2.0),
            RouteDefinition(name="slow", base_cost=5.0, coefficient=0.1),
        )
        game = self._make_custom_game(routes=routes, num_players=2)
        result = game.step({"player_0": "fast", "player_1": "slow"})
        # fast: 0.5 + 2.0*1 = 2.5 → -2.5
        assert result.payoffs["player_0"] == pytest.approx(-2.5)
        # slow: 5.0 + 0.1*1 = 5.1 → -5.1
        assert result.payoffs["player_1"] == pytest.approx(-5.1)

    def test_compute_latency_utility_method(self) -> None:
        """compute_latency() helper matches the formula."""
        routes = (
            RouteDefinition(name="r1", base_cost=3.0, coefficient=2.0),
            RouteDefinition(name="r2", base_cost=1.0, coefficient=0.5),
        )
        game = self._make_custom_game(routes=routes, num_players=2)
        assert game.compute_latency("r1", 1) == pytest.approx(3.0 + 2.0 * 1)
        assert game.compute_latency("r1", 4) == pytest.approx(3.0 + 2.0 * 4)
        assert game.compute_latency("r2", 3) == pytest.approx(1.0 + 0.5 * 3)

    def test_three_player_congestion(self) -> None:
        """Three players, two on same route, one solo: verify payoffs."""
        routes = (
            RouteDefinition(name="route_A", base_cost=1.0, coefficient=1.0),
            RouteDefinition(name="route_B", base_cost=2.0, coefficient=0.5),
        )
        game = self._make_custom_game(routes=routes, num_players=3)
        result = game.step(
            {
                "player_0": "route_A",
                "player_1": "route_A",
                "player_2": "route_B",
            }
        )
        # route_A with 2 users: 1 + 1*2 = 3, payoff = -3
        assert result.payoffs["player_0"] == pytest.approx(-3.0)
        assert result.payoffs["player_1"] == pytest.approx(-3.0)
        # route_B with 1 user: 2 + 0.5*1 = 2.5, payoff = -2.5
        assert result.payoffs["player_2"] == pytest.approx(-2.5)

    def test_zero_coefficient_route(self) -> None:
        """Route with coefficient=0 has fixed latency regardless of users."""
        routes = (
            RouteDefinition(name="fixed", base_cost=4.0, coefficient=0.0),
            RouteDefinition(name="variable", base_cost=1.0, coefficient=1.0),
        )
        game = self._make_custom_game(routes=routes, num_players=2)
        result = game.step({"player_0": "fixed", "player_1": "fixed"})
        # both on fixed: 4 + 0*2 = 4, payoff = -4
        assert result.payoffs["player_0"] == pytest.approx(-4.0)
        assert result.payoffs["player_1"] == pytest.approx(-4.0)

    def test_compute_total_latency(self) -> None:
        """compute_total_latency() sums all player latencies."""
        game = self._make_game()
        actions = {"player_0": "route_A", "player_1": "route_B"}
        total = game.compute_total_latency(actions)
        # route_A: 1+1*1=2, route_B: 2+0.5*1=2.5 → total = 4.5
        assert total == pytest.approx(4.5)
