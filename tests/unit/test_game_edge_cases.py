"""Edge case tests for the game-environments library.

Covers boundary conditions, invalid inputs, terminal state behavior,
and game lifecycle edge cases for all 5 canonical games.
"""

from __future__ import annotations

import pytest
from game_envs.games.auction import Auction, AuctionConfig, AuctionType
from game_envs.games.colonel_blotto import BlottoConfig, ColonelBlotto
from game_envs.games.congestion import CongestionConfig, CongestionGame, RouteDefinition
from game_envs.games.prisoners_dilemma import PDConfig, PrisonersDilemma
from game_envs.games.public_goods import PGConfig, PublicGoodsGame

# ---------------------------------------------------------------------------
# Prisoner's Dilemma edge cases
# ---------------------------------------------------------------------------


class TestPDEdgeCases:
    """Edge cases for Prisoner's Dilemma."""

    def test_single_round_terminates_after_one_step(self) -> None:
        """Game with num_rounds=1 is terminal after a single step."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        game.reset()
        result = game.step({"player_0": "cooperate", "player_1": "cooperate"})
        assert result.is_terminal
        assert game.is_terminal

    def test_multi_round_not_terminal_until_last_round(self) -> None:
        """Game with num_rounds=3 is not terminal after first two steps."""
        game = PrisonersDilemma(PDConfig(num_rounds=3))
        game.reset()
        r1 = game.step({"player_0": "cooperate", "player_1": "cooperate"})
        assert not r1.is_terminal
        r2 = game.step({"player_0": "defect", "player_1": "defect"})
        assert not r2.is_terminal
        r3 = game.step({"player_0": "cooperate", "player_1": "defect"})
        assert r3.is_terminal

    def test_multi_round_cumulative_payoffs_accumulate(self) -> None:
        """Cumulative payoffs sum across all rounds (discount_factor=1.0)."""
        game = PrisonersDilemma(PDConfig(num_rounds=3))
        game.reset()
        # round 0: both cooperate → 3, 3
        game.step({"player_0": "cooperate", "player_1": "cooperate"})
        # round 1: both defect → 1, 1
        game.step({"player_0": "defect", "player_1": "defect"})
        # round 2: p0 defects, p1 cooperates → 5, 0
        game.step({"player_0": "defect", "player_1": "cooperate"})
        cumulative = game.get_payoffs()
        assert cumulative["player_0"] == pytest.approx(3.0 + 1.0 + 5.0)
        assert cumulative["player_1"] == pytest.approx(3.0 + 1.0 + 0.0)

    def test_invalid_action_raises_value_error(self) -> None:
        """Invalid action string raises ValueError."""
        game = PrisonersDilemma()
        game.reset()
        with pytest.raises(ValueError, match="Invalid action"):
            game.step({"player_0": "neither", "player_1": "cooperate"})

    def test_none_action_raises_value_error(self) -> None:
        """None action raises ValueError."""
        game = PrisonersDilemma()
        game.reset()
        with pytest.raises(ValueError, match="Invalid action"):
            game.step({"player_0": None, "player_1": "cooperate"})

    def test_step_after_terminal_raises_runtime_error(self) -> None:
        """Stepping a terminal game raises RuntimeError."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        game.reset()
        game.step({"player_0": "defect", "player_1": "defect"})
        assert game.is_terminal
        with pytest.raises(RuntimeError, match="already terminal"):
            game.step({"player_0": "cooperate", "player_1": "cooperate"})

    def test_reset_clears_terminal_flag(self) -> None:
        """After reset, a terminated game can be played again."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        game.reset()
        game.step({"player_0": "cooperate", "player_1": "cooperate"})
        assert game.is_terminal
        game.reset()
        assert not game.is_terminal
        result = game.step({"player_0": "defect", "player_1": "defect"})
        assert result.payoffs["player_0"] == pytest.approx(1.0)

    def test_reset_clears_cumulative_payoffs(self) -> None:
        """Cumulative payoffs are zeroed after reset."""
        game = PrisonersDilemma(PDConfig(num_rounds=2))
        game.reset()
        game.step({"player_0": "defect", "player_1": "cooperate"})
        game.step({"player_0": "defect", "player_1": "cooperate"})
        assert game.get_payoffs()["player_0"] > 0.0
        game.reset()
        assert game.get_payoffs()["player_0"] == pytest.approx(0.0)
        assert game.get_payoffs()["player_1"] == pytest.approx(0.0)

    def test_invalid_config_2r_not_gt_t_plus_s(self) -> None:
        """PDConfig must enforce 2R > T + S."""
        with pytest.raises(ValueError, match="2R > T"):
            # T=9, R=4, P=2, S=0: T>R, R>P, P>S all hold,
            # but 2*4=8 is not > 9+0=9
            PDConfig(temptation=9.0, reward=4.0, punishment=2.0, sucker=0.0)


# ---------------------------------------------------------------------------
# Public Goods Game edge cases
# ---------------------------------------------------------------------------


class TestPublicGoodsEdgeCases:
    """Edge cases for the Public Goods Game."""

    def _make_game(self, num_players: int = 2, **kwargs: float) -> PublicGoodsGame:
        """Create a Public Goods Game."""
        multiplier = kwargs.pop("multiplier", 1.6)
        cfg = PGConfig(num_players=num_players, multiplier=multiplier, **kwargs)
        game = PublicGoodsGame(cfg)
        game.reset()
        return game

    def test_contribution_at_zero_boundary(self) -> None:
        """Contribution of exactly 0.0 is accepted and formula applies."""
        game = self._make_game(num_players=2, endowment=20.0)
        result = game.step({"player_0": 0.0, "player_1": 0.0})
        # payoff = 20 - 0 + 1.6 * 0 / 2 = 20
        assert result.payoffs["player_0"] == pytest.approx(20.0)

    def test_contribution_at_endowment_boundary(self) -> None:
        """Contribution of exactly endowment is accepted and formula applies."""
        endowment = 20.0
        game = self._make_game(num_players=2, endowment=endowment)
        result = game.step({"player_0": endowment, "player_1": endowment})
        # payoff = 20 - 20 + 1.6 * 40 / 2 = 0 + 32 = 32
        assert result.payoffs["player_0"] == pytest.approx(32.0)

    def test_minimum_two_players_allowed(self) -> None:
        """Two-player Public Goods Game is the minimum valid configuration."""
        game = self._make_game(num_players=2)
        result = game.step({"player_0": 5.0, "player_1": 10.0})
        assert "player_0" in result.payoffs
        assert "player_1" in result.payoffs

    def test_one_player_raises(self) -> None:
        """Fewer than 2 players raises ValueError."""
        with pytest.raises(ValueError):
            PublicGoodsGame(PGConfig(num_players=1, multiplier=1.5))

    def test_negative_contribution_accepted_formula_applies(self) -> None:
        """Negative contribution is not validated; formula applies as-is."""
        endowment = 20.0
        multiplier = 1.6
        n = 2
        game = self._make_game(num_players=n, endowment=endowment)
        # No validation in step(); negative contribution is processed
        result = game.step({"player_0": -5.0, "player_1": 10.0})
        total = -5.0 + 10.0
        share = multiplier * total / n
        expected_p0 = endowment - (-5.0) + share
        expected_p1 = endowment - 10.0 + share
        assert result.payoffs["player_0"] == pytest.approx(expected_p0)
        assert result.payoffs["player_1"] == pytest.approx(expected_p1)

    def test_contribution_exceeding_endowment_accepted(self) -> None:
        """Contribution above endowment is not validated; formula applies."""
        endowment = 20.0
        multiplier = 1.6
        n = 2
        game = self._make_game(num_players=n, endowment=endowment)
        # Contribute more than endowment — game accepts it
        over = endowment + 10.0
        result = game.step({"player_0": over, "player_1": over})
        total = over + over
        share = multiplier * total / n
        expected = endowment - over + share
        assert result.payoffs["player_0"] == pytest.approx(expected)

    def test_multiplier_ge_num_players_raises(self) -> None:
        """multiplier >= num_players violates social dilemma constraint."""
        with pytest.raises(ValueError, match="social dilemma"):
            PGConfig(num_players=2, multiplier=2.0)

    def test_endowment_zero_raises(self) -> None:
        """endowment=0 is rejected by PGConfig."""
        with pytest.raises(ValueError, match="endowment must be positive"):
            PGConfig(num_players=2, multiplier=1.5, endowment=0.0)

    def test_step_after_terminal_raises(self) -> None:
        """Stepping a terminal PG game raises RuntimeError."""
        game = self._make_game(num_players=2)
        game.step({"player_0": 5.0, "player_1": 5.0})
        assert game.is_terminal
        with pytest.raises(RuntimeError, match="already terminal"):
            game.step({"player_0": 5.0, "player_1": 5.0})


# ---------------------------------------------------------------------------
# Auction edge cases
# ---------------------------------------------------------------------------


class TestAuctionEdgeCases:
    """Edge cases for Sealed-Bid Auction."""

    def _make_game(
        self,
        auction_type: str = AuctionType.FIRST_PRICE,
        num_players: int = 2,
        reserve_price: float = 0.0,
        min_bid: float = 0.0,
        max_bid: float = 100.0,
    ) -> Auction:
        """Create an auction game."""
        cfg = AuctionConfig(
            auction_type=auction_type,
            num_players=num_players,
            reserve_price=reserve_price,
            min_bid=min_bid,
            max_bid=max_bid,
        )
        game = Auction(cfg)
        game.reset()
        return game

    def test_bid_at_min_boundary_accepted(self) -> None:
        """Bid exactly at min_bid is processed without error."""
        game = self._make_game(min_bid=10.0, max_bid=100.0)
        game._private_values = {"player_0": 50.0, "player_1": 30.0}
        result = game.step({"player_0": 10.0, "player_1": 10.0})
        # Both bid at min; player_0 wins tie (lower id)
        assert result.payoffs["player_0"] == pytest.approx(50.0 - 10.0)

    def test_bid_at_max_boundary_accepted(self) -> None:
        """Bid exactly at max_bid is processed without error."""
        game = self._make_game(min_bid=0.0, max_bid=100.0)
        game._private_values = {"player_0": 80.0, "player_1": 60.0}
        result = game.step({"player_0": 100.0, "player_1": 50.0})
        assert result.payoffs["player_0"] == pytest.approx(80.0 - 100.0)

    def test_bid_above_max_not_validated(self) -> None:
        """step() does not validate bid range; above max_bid is accepted."""
        game = self._make_game(max_bid=100.0)
        game._private_values = {"player_0": 80.0, "player_1": 60.0}
        # No validation: bid 200 is accepted and formula applies
        result = game.step({"player_0": 200.0, "player_1": 50.0})
        assert result.payoffs["player_0"] == pytest.approx(80.0 - 200.0)

    def test_single_bidder_config_raises(self) -> None:
        """AuctionConfig with num_players=1 raises ValueError."""
        with pytest.raises(ValueError, match="num_players must be >= 2"):
            AuctionConfig(num_players=1)

    def test_equal_bids_tie_broken_by_lower_player_id(self) -> None:
        """Equal bids: player with lexicographically lower id wins."""
        game = self._make_game(num_players=3)
        game._private_values = {
            "player_0": 50.0,
            "player_1": 60.0,
            "player_2": 70.0,
        }
        # All bid 40; player_0 wins (lowest id)
        result = game.step({"player_0": 40.0, "player_1": 40.0, "player_2": 40.0})
        assert result.payoffs["player_0"] == pytest.approx(50.0 - 40.0)
        assert result.payoffs["player_1"] == pytest.approx(0.0)
        assert result.payoffs["player_2"] == pytest.approx(0.0)

    def test_reserve_price_exactly_met(self) -> None:
        """Highest bid exactly equal to reserve price triggers a sale."""
        game = self._make_game(
            auction_type=AuctionType.FIRST_PRICE,
            reserve_price=50.0,
        )
        game._private_values = {"player_0": 80.0, "player_1": 40.0}
        result = game.step({"player_0": 50.0, "player_1": 30.0})
        # Bid == reserve price: sale occurs
        assert result.payoffs["player_0"] == pytest.approx(80.0 - 50.0)
        assert result.payoffs["player_1"] == pytest.approx(0.0)

    def test_second_price_single_bidder_pays_reserve(self) -> None:
        """Second-price with one winning bidder: pays max(second_bid, reserve)."""
        game = self._make_game(
            auction_type=AuctionType.SECOND_PRICE,
            reserve_price=10.0,
        )
        game._private_values = {"player_0": 80.0, "player_1": 40.0}
        # Only player_0 meets reserve; second bid is 5 but reserve is 10
        result = game.step({"player_0": 70.0, "player_1": 5.0})
        # second_bid=5, reserve=10 → price=max(5,10)=10
        assert result.payoffs["player_0"] == pytest.approx(80.0 - 10.0)

    def test_step_after_terminal_raises(self) -> None:
        """Stepping a terminal auction raises RuntimeError."""
        game = self._make_game(num_players=2)
        game._private_values = {"player_0": 50.0, "player_1": 30.0}
        game.step({"player_0": 40.0, "player_1": 20.0})
        assert game.is_terminal
        with pytest.raises(RuntimeError, match="already terminal"):
            game.step({"player_0": 40.0, "player_1": 20.0})


# ---------------------------------------------------------------------------
# Colonel Blotto edge cases
# ---------------------------------------------------------------------------


class TestBlottoEdgeCases:
    """Edge cases for Colonel Blotto."""

    def _make_game(
        self,
        num_battlefields: int = 3,
        total_troops: int = 100,
    ) -> ColonelBlotto:
        """Create a 2-player Colonel Blotto game."""
        cfg = BlottoConfig(num_battlefields=num_battlefields, total_troops=total_troops)
        game = ColonelBlotto(cfg)
        game.reset()
        return game

    def _alloc(self, *troops: int) -> dict[str, float]:
        """Build allocation dict for battlefields."""
        return {f"battlefield_{i}": float(t) for i, t in enumerate(troops)}

    def test_all_troops_on_one_battlefield_valid(self) -> None:
        """Concentrating all troops on one battlefield is a valid move."""
        game = self._make_game(num_battlefields=3, total_troops=9)
        # player_0 puts all troops on bf_0
        result = game.step(
            {
                "player_0": self._alloc(9, 0, 0),
                "player_1": self._alloc(3, 3, 3),
            }
        )
        # player_0 wins bf_0 (9>3); player_1 wins bf_1 (3>0) and bf_2 (3>0)
        assert result.payoffs["player_0"] == pytest.approx(1.0 / 3.0)
        assert result.payoffs["player_1"] == pytest.approx(2.0 / 3.0)

    def test_all_troops_one_battlefield_wins_that_field(self) -> None:
        """Player with all troops on one battlefield wins it outright."""
        game = self._make_game(num_battlefields=2, total_troops=10)
        # p0: 10,0; p1: 5,5 → p0 wins bf_0 (10>5), p1 wins bf_1 (5>0)
        result = game.step(
            {
                "player_0": self._alloc(10, 0),
                "player_1": self._alloc(5, 5),
            }
        )
        assert result.payoffs["player_0"] == pytest.approx(0.5)
        assert result.payoffs["player_1"] == pytest.approx(0.5)

    def test_invalid_allocation_wrong_sum_raises(self) -> None:
        """Allocation with wrong total raises ValueError."""
        game = self._make_game(num_battlefields=3, total_troops=9)
        with pytest.raises(ValueError):
            game.step(
                {
                    "player_0": self._alloc(5, 5, 5),  # sums to 15, not 9
                    "player_1": self._alloc(3, 3, 3),
                }
            )

    def test_negative_troop_allocation_raises(self) -> None:
        """Negative troop values violate min_value=0 constraint."""
        game = self._make_game(num_battlefields=3, total_troops=9)
        with pytest.raises(ValueError):
            game.step(
                {
                    "player_0": {
                        "battlefield_0": -1.0,
                        "battlefield_1": 5.0,
                        "battlefield_2": 5.0,
                    },
                    "player_1": self._alloc(3, 3, 3),
                }
            )

    def test_missing_battlefield_field_raises(self) -> None:
        """Allocation missing a battlefield field raises ValueError."""
        game = self._make_game(num_battlefields=3, total_troops=9)
        with pytest.raises(ValueError):
            game.step(
                {
                    "player_0": {
                        "battlefield_0": 5.0,
                        "battlefield_1": 4.0,
                        # battlefield_2 missing
                    },
                    "player_1": self._alloc(3, 3, 3),
                }
            )

    def test_zero_total_troops_raises(self) -> None:
        """BlottoConfig with total_troops=0 is invalid."""
        with pytest.raises(ValueError, match="total_troops must be >= 1"):
            BlottoConfig(num_battlefields=2, total_troops=0)

    def test_step_after_terminal_raises(self) -> None:
        """Stepping a terminal Blotto game raises RuntimeError."""
        game = self._make_game(num_battlefields=2, total_troops=4)
        game.step(
            {
                "player_0": self._alloc(3, 1),
                "player_1": self._alloc(2, 2),
            }
        )
        assert game.is_terminal
        with pytest.raises(RuntimeError, match="already terminal"):
            game.step(
                {
                    "player_0": self._alloc(3, 1),
                    "player_1": self._alloc(2, 2),
                }
            )

    def test_num_battlefields_one_raises(self) -> None:
        """BlottoConfig with num_battlefields=1 is invalid."""
        with pytest.raises(ValueError, match="num_battlefields must be >= 2"):
            BlottoConfig(num_battlefields=1, total_troops=10)


# ---------------------------------------------------------------------------
# Congestion Game edge cases
# ---------------------------------------------------------------------------


class TestCongestionEdgeCases:
    """Edge cases for the Congestion Game."""

    def _make_game(
        self,
        routes: tuple[RouteDefinition, ...] | None = None,
        num_players: int = 2,
    ) -> CongestionGame:
        """Create a Congestion Game."""
        if routes is None:
            routes = (
                RouteDefinition(name="route_A", base_cost=1.0, coefficient=1.0),
                RouteDefinition(name="route_B", base_cost=2.0, coefficient=0.5),
            )
        cfg = CongestionConfig(routes=routes, num_players=num_players)
        game = CongestionGame(cfg)
        game.reset()
        return game

    def test_all_players_same_route_maximum_congestion(self) -> None:
        """All players on same route: everyone gets maximum congestion payoff."""
        n = 4
        game = self._make_game(num_players=n)
        # All 4 players choose route_A: latency = 1 + 1*4 = 5
        actions = {f"player_{i}": "route_A" for i in range(n)}
        result = game.step(actions)
        for pid in game.player_ids:
            assert result.payoffs[pid] == pytest.approx(-5.0)

    def test_single_player_on_route_no_congestion(self) -> None:
        """With 2 players on different routes, each is alone (min congestion)."""
        game = self._make_game(num_players=2)
        result = game.step({"player_0": "route_A", "player_1": "route_B"})
        # route_A alone: 1 + 1*1 = 2 → payoff -2
        assert result.payoffs["player_0"] == pytest.approx(-2.0)
        # route_B alone: 2 + 0.5*1 = 2.5 → payoff -2.5
        assert result.payoffs["player_1"] == pytest.approx(-2.5)

    def test_invalid_route_name_raises_key_error(self) -> None:
        """Unknown route name causes KeyError during step."""
        game = self._make_game(num_players=2)
        with pytest.raises(KeyError):
            game.step({"player_0": "nonexistent_route", "player_1": "route_A"})

    def test_minimum_two_players_allowed(self) -> None:
        """Two-player Congestion Game is the minimum valid configuration."""
        game = self._make_game(num_players=2)
        result = game.step({"player_0": "route_A", "player_1": "route_A"})
        assert len(result.payoffs) == 2

    def test_one_player_raises(self) -> None:
        """num_players=1 raises ValueError."""
        with pytest.raises(ValueError, match="num_players must be between"):
            CongestionGame(
                CongestionConfig(
                    routes=(
                        RouteDefinition("r1", 1.0, 1.0),
                        RouteDefinition("r2", 2.0, 0.5),
                    ),
                    num_players=1,
                )
            )

    def test_all_players_same_route_congestion_scales_linearly(self) -> None:
        """Congestion penalty scales linearly with number of users."""
        routes = (
            RouteDefinition(name="route_A", base_cost=0.0, coefficient=2.0),
            RouteDefinition(name="route_B", base_cost=10.0, coefficient=0.0),
        )
        game = self._make_game(routes=routes, num_players=3)
        actions = {"player_0": "route_A", "player_1": "route_A", "player_2": "route_A"}
        result = game.step(actions)
        # route_A with 3 users: 0 + 2*3 = 6 → payoff -6
        for pid in game.player_ids:
            assert result.payoffs[pid] == pytest.approx(-6.0)

    def test_one_route_config_raises(self) -> None:
        """CongestionConfig with fewer than 2 routes raises ValueError."""
        with pytest.raises(ValueError, match="need at least 2 routes"):
            CongestionConfig(
                routes=(
                    RouteDefinition(name="only_route", base_cost=1.0, coefficient=1.0),
                )
            )

    def test_duplicate_route_names_raises(self) -> None:
        """CongestionConfig with duplicate route names raises ValueError."""
        with pytest.raises(ValueError, match="route names must be unique"):
            CongestionConfig(
                routes=(
                    RouteDefinition(name="route_A", base_cost=1.0, coefficient=1.0),
                    RouteDefinition(name="route_A", base_cost=2.0, coefficient=0.5),
                )
            )

    def test_step_after_terminal_raises(self) -> None:
        """Stepping a terminal Congestion game raises RuntimeError."""
        game = self._make_game(num_players=2)
        game.step({"player_0": "route_A", "player_1": "route_B"})
        assert game.is_terminal
        with pytest.raises(RuntimeError, match="already terminal"):
            game.step({"player_0": "route_A", "player_1": "route_B"})

    def test_negative_base_cost_raises(self) -> None:
        """RouteDefinition with negative base_cost raises ValueError."""
        with pytest.raises(ValueError, match="base_cost must be >= 0"):
            RouteDefinition(name="bad_route", base_cost=-1.0, coefficient=1.0)

    def test_negative_coefficient_raises(self) -> None:
        """RouteDefinition with negative coefficient raises ValueError."""
        with pytest.raises(ValueError, match="coefficient must be >= 0"):
            RouteDefinition(name="bad_route", base_cost=1.0, coefficient=-0.5)
