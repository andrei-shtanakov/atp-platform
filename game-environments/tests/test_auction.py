"""Tests for sealed-bid auction game implementation."""

from __future__ import annotations

import pytest

from game_envs.core.game import GameType, MoveOrder
from game_envs.core.state import Observation
from game_envs.games.auction import (
    Auction,
    AuctionConfig,
    AuctionType,
    ValueDistribution,
)
from game_envs.games.registry import GameRegistry


class TestAuctionConfig:
    """Tests for AuctionConfig validation."""

    def test_defaults(self) -> None:
        c = AuctionConfig()
        assert c.auction_type == AuctionType.FIRST_PRICE
        assert c.min_bid == 0.0
        assert c.max_bid == 100.0
        assert c.reserve_price == 0.0
        assert c.value_distribution == ValueDistribution.UNIFORM
        assert c.value_min == 0.0
        assert c.value_max == 100.0
        assert c.num_players == 2
        assert c.num_rounds == 1

    def test_frozen(self) -> None:
        c = AuctionConfig()
        with pytest.raises(AttributeError):
            c.min_bid = 10.0  # type: ignore[misc]

    def test_inherits_game_config_validation(self) -> None:
        with pytest.raises(ValueError, match="num_rounds"):
            AuctionConfig(num_rounds=0)

    def test_invalid_auction_type(self) -> None:
        with pytest.raises(ValueError, match="auction_type"):
            AuctionConfig(auction_type="dutch")

    def test_negative_min_bid(self) -> None:
        with pytest.raises(ValueError, match="min_bid"):
            AuctionConfig(min_bid=-1.0)

    def test_max_bid_not_greater_than_min_bid(self) -> None:
        with pytest.raises(ValueError, match="max_bid"):
            AuctionConfig(min_bid=50.0, max_bid=50.0)

    def test_negative_reserve_price(self) -> None:
        with pytest.raises(ValueError, match="reserve_price"):
            AuctionConfig(reserve_price=-1.0)

    def test_invalid_value_distribution(self) -> None:
        with pytest.raises(ValueError, match="value_distribution"):
            AuctionConfig(value_distribution="exponential")

    def test_value_max_not_greater_than_value_min(self) -> None:
        with pytest.raises(ValueError, match="value_max"):
            AuctionConfig(value_min=100.0, value_max=50.0)

    def test_num_players_less_than_two(self) -> None:
        with pytest.raises(ValueError, match="num_players"):
            AuctionConfig(num_players=1)

    def test_valid_second_price_config(self) -> None:
        c = AuctionConfig(
            auction_type=AuctionType.SECOND_PRICE,
            num_players=5,
            reserve_price=10.0,
        )
        assert c.auction_type == AuctionType.SECOND_PRICE
        assert c.num_players == 5
        assert c.reserve_price == 10.0


class TestAuctionProperties:
    """Tests for Auction game properties."""

    def test_first_price_name(self) -> None:
        game = Auction()
        assert "First-Price" in game.name
        assert "Sealed-Bid Auction" in game.name

    def test_second_price_name(self) -> None:
        config = AuctionConfig(auction_type=AuctionType.SECOND_PRICE)
        game = Auction(config)
        assert "Second-Price (Vickrey)" in game.name

    def test_repeated_name(self) -> None:
        config = AuctionConfig(num_rounds=5)
        game = Auction(config)
        assert "repeated x5" in game.name

    def test_game_type_one_shot(self) -> None:
        game = Auction()
        assert game.game_type == GameType.NORMAL_FORM

    def test_game_type_repeated(self) -> None:
        config = AuctionConfig(num_rounds=3)
        game = Auction(config)
        assert game.game_type == GameType.REPEATED

    def test_move_order(self) -> None:
        game = Auction()
        assert game.move_order == MoveOrder.SIMULTANEOUS

    def test_player_ids(self) -> None:
        game = Auction()
        assert game.player_ids == ["player_0", "player_1"]

    def test_player_ids_n_players(self) -> None:
        config = AuctionConfig(num_players=4)
        game = Auction(config)
        assert game.player_ids == [
            "player_0",
            "player_1",
            "player_2",
            "player_3",
        ]

    def test_action_space(self) -> None:
        config = AuctionConfig(min_bid=5.0, max_bid=50.0)
        game = Auction(config)
        space = game.action_space("player_0")
        assert space.contains(5.0)
        assert space.contains(50.0)
        assert space.contains(25.0)
        assert not space.contains(4.9)
        assert not space.contains(50.1)


class TestAuctionReset:
    """Tests for reset and private value draws."""

    def test_reset_not_terminal(self) -> None:
        game = Auction(AuctionConfig(seed=42))
        result = game.reset()
        assert not result.is_terminal
        assert game.current_round == 0

    def test_reset_zero_payoffs(self) -> None:
        game = Auction(AuctionConfig(seed=42))
        result = game.reset()
        for pid in game.player_ids:
            assert result.payoffs[pid] == 0.0

    def test_reset_draws_private_values(self) -> None:
        game = Auction(AuctionConfig(seed=42))
        game.reset()
        values = game.private_values
        assert len(values) == 2
        for v in values.values():
            assert 0.0 <= v <= 100.0

    def test_uniform_value_draw(self) -> None:
        """Values should be within [value_min, value_max]."""
        config = AuctionConfig(
            value_min=10.0,
            value_max=90.0,
            seed=42,
            num_rounds=100,
        )
        game = Auction(config)
        game.reset()

        # Check initial values
        for v in game.private_values.values():
            assert 10.0 <= v <= 90.0

        # Check across many rounds
        for _ in range(99):
            game.step({pid: 50.0 for pid in game.player_ids})
            for v in game.private_values.values():
                assert 10.0 <= v <= 90.0

    def test_normal_value_draw_clamped(self) -> None:
        """Normal draws should be clamped to bounds."""
        config = AuctionConfig(
            value_distribution=ValueDistribution.NORMAL,
            value_min=0.0,
            value_max=100.0,
            seed=42,
            num_rounds=100,
        )
        game = Auction(config)
        game.reset()

        for v in game.private_values.values():
            assert 0.0 <= v <= 100.0

    def test_reset_clears_state(self) -> None:
        config = AuctionConfig(num_rounds=3, seed=42)
        game = Auction(config)
        game.reset()
        game.step({pid: 50.0 for pid in game.player_ids})
        game.reset()
        assert game.current_round == 0
        assert not game.is_terminal
        assert len(game.history) == 0
        assert game.get_payoffs() == {
            "player_0": 0.0,
            "player_1": 0.0,
        }

    def test_seed_reproducibility(self) -> None:
        """Same seed should produce same private values."""
        config = AuctionConfig(seed=123)
        game1 = Auction(config)
        game1.reset()
        v1 = game1.private_values

        game2 = Auction(config)
        game2.reset()
        v2 = game2.private_values

        assert v1 == v2


class TestAuctionFirstPrice:
    """Tests for first-price sealed-bid auction."""

    def test_winner_selection(self) -> None:
        """Highest bidder wins."""
        config = AuctionConfig(seed=42)
        game = Auction(config)
        game.reset()
        result = game.step({"player_0": 60.0, "player_1": 40.0})
        # player_0 bid higher, wins
        assert result.info["winner"] == "player_0"

    def test_winner_pays_own_bid(self) -> None:
        """First-price: winner pays own bid."""
        config = AuctionConfig(seed=42)
        game = Auction(config)
        game.reset()
        values = game.private_values

        result = game.step({"player_0": 60.0, "player_1": 40.0})
        assert result.info["price"] == 60.0
        expected_payoff = values["player_0"] - 60.0
        assert result.payoffs["player_0"] == pytest.approx(expected_payoff)
        assert result.payoffs["player_1"] == 0.0

    def test_loser_gets_zero(self) -> None:
        """Loser payoff is always 0."""
        config = AuctionConfig(seed=42)
        game = Auction(config)
        game.reset()
        result = game.step({"player_0": 60.0, "player_1": 40.0})
        assert result.payoffs["player_1"] == 0.0

    def test_three_player_first_price(self) -> None:
        """Highest of 3 bidders wins."""
        config = AuctionConfig(num_players=3, seed=42)
        game = Auction(config)
        game.reset()
        result = game.step(
            {
                "player_0": 30.0,
                "player_1": 50.0,
                "player_2": 40.0,
            }
        )
        assert result.info["winner"] == "player_1"
        assert result.info["price"] == 50.0


class TestAuctionSecondPrice:
    """Tests for second-price (Vickrey) sealed-bid auction."""

    def test_winner_pays_second_highest_bid(self) -> None:
        """Second-price: winner pays second-highest bid."""
        config = AuctionConfig(auction_type=AuctionType.SECOND_PRICE, seed=42)
        game = Auction(config)
        game.reset()
        values = game.private_values

        result = game.step({"player_0": 70.0, "player_1": 30.0})
        assert result.info["winner"] == "player_0"
        assert result.info["price"] == 30.0
        expected_payoff = values["player_0"] - 30.0
        assert result.payoffs["player_0"] == pytest.approx(expected_payoff)
        assert result.payoffs["player_1"] == 0.0

    def test_three_player_second_price(self) -> None:
        """Winner pays second-highest of 3 bids."""
        config = AuctionConfig(
            auction_type=AuctionType.SECOND_PRICE,
            num_players=3,
            seed=42,
        )
        game = Auction(config)
        game.reset()
        result = game.step(
            {
                "player_0": 30.0,
                "player_1": 50.0,
                "player_2": 40.0,
            }
        )
        assert result.info["winner"] == "player_1"
        # Second-highest bid is 40.0
        assert result.info["price"] == 40.0

    def test_truthful_bidding_dominant(self) -> None:
        """Verify truthful bidding is dominant in Vickrey.

        A player who bids their true value can never do
        worse than bidding something else. We test that
        bidding the true value gives at least as good a
        payoff as any deviation.
        """
        config = AuctionConfig(
            auction_type=AuctionType.SECOND_PRICE,
            seed=42,
        )
        game = Auction(config)
        game.reset()
        values = game.private_values
        v0 = values["player_0"]

        # Opponent bids some fixed amount
        opponent_bid = 50.0

        # Truthful bid
        game_truth = Auction(config)
        game_truth.reset()
        r_truth = game_truth.step({"player_0": v0, "player_1": opponent_bid})

        # Overbid
        game_over = Auction(config)
        game_over.reset()
        r_over = game_over.step(
            {
                "player_0": min(v0 + 20.0, 100.0),
                "player_1": opponent_bid,
            }
        )

        # Underbid
        game_under = Auction(config)
        game_under.reset()
        r_under = game_under.step(
            {
                "player_0": max(v0 - 20.0, 0.0),
                "player_1": opponent_bid,
            }
        )

        truth_payoff = r_truth.payoffs["player_0"]
        over_payoff = r_over.payoffs["player_0"]
        under_payoff = r_under.payoffs["player_0"]

        assert truth_payoff >= over_payoff
        assert truth_payoff >= under_payoff

    def test_vickrey_many_scenarios(self) -> None:
        """Truthful bidding dominates across many seeds."""
        for seed in range(50):
            config = AuctionConfig(
                auction_type=AuctionType.SECOND_PRICE,
                seed=seed,
            )
            game = Auction(config)
            game.reset()
            v0 = game.private_values["player_0"]

            for opponent_bid in [10.0, 30.0, 50.0, 70.0, 90.0]:
                # Truthful
                g1 = Auction(config)
                g1.reset()
                r1 = g1.step(
                    {
                        "player_0": v0,
                        "player_1": opponent_bid,
                    }
                )

                # Shade down
                g2 = Auction(config)
                g2.reset()
                r2 = g2.step(
                    {
                        "player_0": max(v0 * 0.5, 0.0),
                        "player_1": opponent_bid,
                    }
                )

                assert r1.payoffs["player_0"] >= r2.payoffs["player_0"]


class TestAuctionFirstPriceOptimalShading:
    """Verify optimal shading in first-price auctions."""

    def test_optimal_shade_is_equilibrium(self) -> None:
        """With uniform values, optimal bid = (n-1)/n * v.

        In the Bayes-Nash equilibrium, all bidders shade
        by (n-1)/n. We verify that deviating from this
        strategy (while opponents play equilibrium) does
        not improve expected payoff.
        """
        n = 3
        shade = (n - 1) / n
        num_auctions = 2000

        payoff_equilibrium = 0.0
        payoff_overbid = 0.0
        payoff_underbid = 0.0

        for seed in range(num_auctions):
            config = AuctionConfig(num_players=n, seed=seed)

            # All at equilibrium
            g1 = Auction(config)
            g1.reset()
            vals = g1.private_values
            bids_eq = {pid: shade * vals[pid] for pid in g1.player_ids}
            r1 = g1.step(bids_eq)
            payoff_equilibrium += r1.payoffs["player_0"]

            # Player 0 deviates to truthful, others at eq
            g2 = Auction(config)
            g2.reset()
            bids_dev = {pid: shade * g2.private_values[pid] for pid in g2.player_ids}
            bids_dev["player_0"] = g2.private_values["player_0"]
            r2 = g2.step(bids_dev)
            payoff_overbid += r2.payoffs["player_0"]

            # Player 0 deviates to 50% shade, others at eq
            g3 = Auction(config)
            g3.reset()
            bids_under = {pid: shade * g3.private_values[pid] for pid in g3.player_ids}
            bids_under["player_0"] = 0.5 * g3.private_values["player_0"]
            r3 = g3.step(bids_under)
            payoff_underbid += r3.payoffs["player_0"]

        avg_eq = payoff_equilibrium / num_auctions
        avg_over = payoff_overbid / num_auctions
        avg_under = payoff_underbid / num_auctions

        # Equilibrium play should beat deviations
        assert avg_eq > avg_over
        assert avg_eq > avg_under


class TestAuctionReservePrice:
    """Tests for reserve price mechanics."""

    def test_no_winner_below_reserve(self) -> None:
        """No winner when all bids below reserve."""
        config = AuctionConfig(reserve_price=80.0, seed=42)
        game = Auction(config)
        game.reset()
        result = game.step({"player_0": 70.0, "player_1": 60.0})
        assert result.info["winner"] is None
        assert result.payoffs["player_0"] == 0.0
        assert result.payoffs["player_1"] == 0.0

    def test_winner_at_reserve(self) -> None:
        """Bid exactly at reserve wins."""
        config = AuctionConfig(reserve_price=50.0, seed=42)
        game = Auction(config)
        game.reset()
        result = game.step({"player_0": 50.0, "player_1": 30.0})
        assert result.info["winner"] == "player_0"

    def test_reserve_price_second_price_floor(self) -> None:
        """In second-price, price = max(second_bid, reserve)."""
        config = AuctionConfig(
            auction_type=AuctionType.SECOND_PRICE,
            reserve_price=40.0,
            seed=42,
        )
        game = Auction(config)
        game.reset()
        result = game.step({"player_0": 60.0, "player_1": 30.0})
        # Second bid is 30, but reserve is 40, so pay 40
        assert result.info["price"] == 40.0

    def test_reserve_second_price_second_bid_above(
        self,
    ) -> None:
        """When second bid > reserve, pay second bid."""
        config = AuctionConfig(
            auction_type=AuctionType.SECOND_PRICE,
            reserve_price=20.0,
            seed=42,
        )
        game = Auction(config)
        game.reset()
        result = game.step({"player_0": 60.0, "player_1": 45.0})
        assert result.info["price"] == 45.0


class TestAuctionTies:
    """Tests for tie-breaking in auctions."""

    def test_tie_lower_id_wins(self) -> None:
        """Deterministic tie-break: lower player_id wins."""
        config = AuctionConfig(seed=42)
        game = Auction(config)
        game.reset()
        result = game.step({"player_0": 50.0, "player_1": 50.0})
        assert result.info["winner"] == "player_0"

    def test_three_way_tie(self) -> None:
        """Three-way tie: lowest player_id wins."""
        config = AuctionConfig(num_players=3, seed=42)
        game = Auction(config)
        game.reset()
        result = game.step(
            {
                "player_0": 50.0,
                "player_1": 50.0,
                "player_2": 50.0,
            }
        )
        assert result.info["winner"] == "player_0"

    def test_tie_second_price_payment(self) -> None:
        """Tie in second-price: pay the tied bid."""
        config = AuctionConfig(
            auction_type=AuctionType.SECOND_PRICE,
            seed=42,
        )
        game = Auction(config)
        game.reset()
        result = game.step({"player_0": 50.0, "player_1": 50.0})
        assert result.info["winner"] == "player_0"
        assert result.info["price"] == 50.0


class TestAuctionObserve:
    """Tests for observation / partial observability."""

    def test_observe_shows_own_value(self) -> None:
        """Player sees their own private value."""
        config = AuctionConfig(seed=42)
        game = Auction(config)
        game.reset()
        values = game.private_values

        obs0 = game.observe("player_0")
        assert obs0.game_state["your_private_value"] == (values["player_0"])

        obs1 = game.observe("player_1")
        assert obs1.game_state["your_private_value"] == (values["player_1"])

    def test_observe_hides_others_values(self) -> None:
        """Player does NOT see others' private values."""
        config = AuctionConfig(seed=42)
        game = Auction(config)
        game.reset()

        obs0 = game.observe("player_0")
        gs = obs0.game_state
        # Should NOT contain other players' values
        for key in gs:
            assert "player_1" not in str(key) or key in ("num_bidders",)
        # The value shown is only "your_private_value"
        assert "your_private_value" in gs

    def test_observe_includes_auction_info(self) -> None:
        """Observation contains auction parameters."""
        config = AuctionConfig(
            auction_type=AuctionType.SECOND_PRICE,
            reserve_price=25.0,
            min_bid=5.0,
            max_bid=80.0,
            seed=42,
        )
        game = Auction(config)
        game.reset()
        obs = game.observe("player_0")
        gs = obs.game_state

        assert gs["auction_type"] == "second_price"
        assert gs["num_bidders"] == 2
        assert gs["reserve_price"] == 25.0
        assert gs["bid_range"] == [5.0, 80.0]
        assert "payment_rule" in gs

    def test_observe_payment_rule_first_price(self) -> None:
        config = AuctionConfig(auction_type=AuctionType.FIRST_PRICE, seed=42)
        game = Auction(config)
        game.reset()
        obs = game.observe("player_0")
        assert "own bid" in obs.game_state["payment_rule"]

    def test_observe_payment_rule_second_price(self) -> None:
        config = AuctionConfig(auction_type=AuctionType.SECOND_PRICE, seed=42)
        game = Auction(config)
        game.reset()
        obs = game.observe("player_0")
        assert "second-highest" in obs.game_state["payment_rule"]

    def test_to_prompt_contains_key_info(self) -> None:
        """to_prompt() output includes relevant info."""
        config = AuctionConfig(seed=42)
        game = Auction(config)
        game.reset()
        obs = game.observe("player_0")
        prompt = obs.to_prompt()

        assert "player_0" in prompt
        assert "Round" in prompt

    def test_observation_serialization(self) -> None:
        """Observation can round-trip through dict."""
        config = AuctionConfig(num_rounds=3, seed=42)
        game = Auction(config)
        game.reset()
        game.step({"player_0": 50.0, "player_1": 30.0})

        obs = game.observe("player_0")
        d = obs.to_dict()
        restored = Observation.from_dict(d)
        assert restored.player_id == obs.player_id
        assert restored.round_number == obs.round_number
        assert len(restored.history) == len(obs.history)


class TestAuctionToPrompt:
    """Tests for Auction.to_prompt() game description."""

    def test_first_price_prompt(self) -> None:
        game = Auction()
        prompt = game.to_prompt()
        assert "first-price" in prompt
        assert "sealed-bid" in prompt
        assert "2 bidders" in prompt
        assert "pays their own bid" in prompt.lower()

    def test_second_price_prompt(self) -> None:
        config = AuctionConfig(auction_type=AuctionType.SECOND_PRICE)
        game = Auction(config)
        prompt = game.to_prompt()
        assert "second-price" in prompt
        assert "Vickrey" in prompt
        assert "second-highest bid" in prompt.lower()
        assert "dominant strategy" in prompt.lower()

    def test_reserve_price_in_prompt(self) -> None:
        config = AuctionConfig(reserve_price=25.0)
        game = Auction(config)
        prompt = game.to_prompt()
        assert "25.0" in prompt
        assert "Reserve" in prompt or "reserve" in prompt

    def test_repeated_in_prompt(self) -> None:
        config = AuctionConfig(num_rounds=10)
        game = Auction(config)
        prompt = game.to_prompt()
        assert "10 rounds" in prompt
        assert "repeated" in prompt

    def test_value_distribution_in_prompt(self) -> None:
        config = AuctionConfig(
            value_distribution=ValueDistribution.NORMAL,
            value_min=10.0,
            value_max=90.0,
        )
        game = Auction(config)
        prompt = game.to_prompt()
        assert "normal" in prompt
        assert "10.0" in prompt
        assert "90.0" in prompt


class TestAuctionRepeated:
    """Tests for repeated auction games."""

    def test_repeated_not_terminal_early(self) -> None:
        config = AuctionConfig(num_rounds=3, seed=42)
        game = Auction(config)
        game.reset()
        r = game.step({"player_0": 50.0, "player_1": 30.0})
        assert not r.is_terminal

    def test_repeated_terminal_at_end(self) -> None:
        config = AuctionConfig(num_rounds=3, seed=42)
        game = Auction(config)
        game.reset()
        for i in range(3):
            r = game.step({"player_0": 50.0, "player_1": 30.0})
        assert r.is_terminal

    def test_new_values_each_round(self) -> None:
        """Private values are re-drawn each round."""
        config = AuctionConfig(num_rounds=3, seed=42)
        game = Auction(config)
        game.reset()
        values_round_0 = game.private_values.copy()

        game.step({"player_0": 50.0, "player_1": 30.0})
        values_round_1 = game.private_values.copy()

        # Values should differ (extremely unlikely same)
        assert values_round_0 != values_round_1

    def test_cumulative_payoffs(self) -> None:
        """get_payoffs() returns sum across rounds."""
        config = AuctionConfig(num_rounds=2, seed=42)
        game = Auction(config)
        game.reset()

        r1 = game.step({"player_0": 50.0, "player_1": 30.0})
        r2 = game.step({"player_0": 50.0, "player_1": 30.0})

        total = game.get_payoffs()
        # Cumulative should be sum of individual rounds
        for pid in game.player_ids:
            assert total[pid] == pytest.approx(r1.payoffs[pid] + r2.payoffs[pid])

    def test_discount_factor(self) -> None:
        config = AuctionConfig(
            num_rounds=2,
            discount_factor=0.5,
            seed=42,
        )
        game = Auction(config)
        game.reset()

        r1 = game.step({"player_0": 50.0, "player_1": 30.0})
        r2 = game.step({"player_0": 50.0, "player_1": 30.0})

        total = game.get_payoffs()
        # Round 0: discount = 0.5^0 = 1.0
        # Round 1: discount = 0.5^1 = 0.5
        for pid in game.player_ids:
            expected = r1.payoffs[pid] * 1.0 + r2.payoffs[pid] * 0.5
            assert total[pid] == pytest.approx(expected)

    def test_history_grows(self) -> None:
        config = AuctionConfig(num_rounds=3, seed=42)
        game = Auction(config)
        game.reset()
        for i in range(3):
            game.step({"player_0": 50.0, "player_1": 30.0})
            assert len(game.history) == i + 1

    def test_step_after_terminal_raises(self) -> None:
        game = Auction(AuctionConfig(seed=42))
        game.reset()
        game.step({"player_0": 50.0, "player_1": 30.0})
        with pytest.raises(RuntimeError, match="terminal"):
            game.step({"player_0": 50.0, "player_1": 30.0})


class TestAuctionRegistry:
    """Tests for game registry integration."""

    def test_registered(self) -> None:
        assert "auction" in GameRegistry.list_games()

    def test_create_from_registry(self) -> None:
        game = GameRegistry.create("auction")
        assert isinstance(game, Auction)

    def test_create_with_config(self) -> None:
        config = AuctionConfig(
            auction_type=AuctionType.SECOND_PRICE,
            num_players=4,
        )
        game = GameRegistry.create("auction", config=config)
        assert isinstance(game, Auction)
        assert len(game.player_ids) == 4
