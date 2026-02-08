"""Tests for Auction strategies."""

from __future__ import annotations

import pytest

from game_envs.core.state import Observation
from game_envs.games.auction import Auction, AuctionConfig, AuctionType
from game_envs.strategies.auction_strategies import (
    RandomBidder,
    ShadeBidder,
    TruthfulBidder,
)


def _make_obs(
    player_id: str = "player_0",
    private_value: float = 50.0,
    bid_range: list[float] | None = None,
) -> Observation:
    """Create a minimal auction observation."""
    if bid_range is None:
        bid_range = [0.0, 100.0]
    return Observation(
        player_id=player_id,
        game_state={
            "game": "Sealed-Bid Auction",
            "your_role": player_id,
            "auction_type": "second_price",
            "num_bidders": 2,
            "reserve_price": 0.0,
            "bid_range": bid_range,
            "your_private_value": private_value,
            "payment_rule": "Winner pays second-highest bid",
        },
        available_actions=["[0.0, 100.0]"],
        history=[],
        round_number=0,
        total_rounds=1,
    )


class TestTruthfulBidder:
    def test_bids_private_value(self) -> None:
        s = TruthfulBidder()
        obs = _make_obs(private_value=42.0)
        assert s.choose_action(obs) == 42.0

    def test_bids_zero_value(self) -> None:
        s = TruthfulBidder()
        obs = _make_obs(private_value=0.0)
        assert s.choose_action(obs) == 0.0

    def test_name(self) -> None:
        assert TruthfulBidder().name == "truthful_bidder"


class TestShadeBidder:
    def test_shades_by_factor(self) -> None:
        s = ShadeBidder(factor=0.5)
        obs = _make_obs(private_value=80.0)
        assert s.choose_action(obs) == 40.0

    def test_factor_one_is_truthful(self) -> None:
        s = ShadeBidder(factor=1.0)
        obs = _make_obs(private_value=60.0)
        assert s.choose_action(obs) == 60.0

    def test_factor_zero_bids_zero(self) -> None:
        s = ShadeBidder(factor=0.0)
        obs = _make_obs(private_value=60.0)
        assert s.choose_action(obs) == 0.0

    def test_clamps_to_bid_range(self) -> None:
        s = ShadeBidder(factor=0.8)
        obs = _make_obs(
            private_value=200.0,
            bid_range=[0.0, 100.0],
        )
        assert s.choose_action(obs) == 100.0

    def test_invalid_factor(self) -> None:
        with pytest.raises(ValueError, match="factor must be"):
            ShadeBidder(factor=1.5)

    def test_name_includes_factor(self) -> None:
        assert ShadeBidder(factor=0.5).name == "shade_bidder_0.5"


class TestRandomBidder:
    def test_bids_within_range(self) -> None:
        s = RandomBidder(seed=42)
        obs = _make_obs(bid_range=[10.0, 50.0])
        for _ in range(20):
            bid = s.choose_action(obs)
            assert 10.0 <= bid <= 50.0

    def test_deterministic_with_seed(self) -> None:
        s1 = RandomBidder(seed=123)
        s2 = RandomBidder(seed=123)
        obs = _make_obs()
        for _ in range(10):
            assert s1.choose_action(obs) == s2.choose_action(obs)

    def test_name(self) -> None:
        assert RandomBidder().name == "random_bidder"


class TestAuctionIntegration:
    """Integration tests with actual Auction game."""

    def test_truthful_wins_vickrey(self) -> None:
        """In Vickrey (second-price), truthful bidding wins
        and pays second-highest bid."""
        game = Auction(
            AuctionConfig(
                auction_type=AuctionType.SECOND_PRICE,
                num_players=2,
                num_rounds=1,
                seed=42,
            )
        )
        game.reset()
        values = game.private_values

        truthful = TruthfulBidder()
        shader = ShadeBidder(factor=0.5)

        obs_0 = game.observe("player_0")
        obs_1 = game.observe("player_1")

        bid_0 = truthful.choose_action(obs_0)
        bid_1 = shader.choose_action(obs_1)

        result = game.step(
            {
                "player_0": bid_0,
                "player_1": bid_1,
            }
        )

        # Truthful bids their value, shader bids half
        assert bid_0 == pytest.approx(values["player_0"])
        assert bid_1 == pytest.approx(values["player_1"] * 0.5)

        # If truthful's value > shader's bid, truthful wins
        if values["player_0"] > bid_1:
            assert result.payoffs["player_0"] > 0
        elif values["player_0"] < bid_1:
            assert result.payoffs["player_0"] == 0.0

    def test_truthful_nonnegative_in_vickrey(self) -> None:
        """Truthful bidding never yields negative payoff
        in second-price auctions."""
        game = Auction(
            AuctionConfig(
                auction_type=AuctionType.SECOND_PRICE,
                num_players=3,
                num_rounds=20,
                seed=42,
            )
        )
        truthful = TruthfulBidder()
        shader = ShadeBidder(factor=0.7)
        random_b = RandomBidder(seed=99)

        game.reset()
        while not game.is_terminal:
            obs_0 = game.observe("player_0")
            obs_1 = game.observe("player_1")
            obs_2 = game.observe("player_2")
            result = game.step(
                {
                    "player_0": truthful.choose_action(obs_0),
                    "player_1": shader.choose_action(obs_1),
                    "player_2": random_b.choose_action(obs_2),
                }
            )
            # Truthful bidder should never have negative
            # payoff in second-price auction
            assert result.payoffs["player_0"] >= 0.0
