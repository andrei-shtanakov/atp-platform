"""Auction baseline strategies."""

from __future__ import annotations

import random
from typing import Any

from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy


class TruthfulBidder(Strategy):
    """Bids exactly the private value.

    Dominant strategy in second-price (Vickrey) auctions.
    Suboptimal in first-price auctions where bid shading
    is typically better.
    """

    @property
    def name(self) -> str:
        return "truthful_bidder"

    def choose_action(self, observation: Observation) -> Any:
        value = observation.game_state.get("your_private_value", 0.0)
        return float(value)


class ShadeBidder(Strategy):
    """Bids a fraction of the private value.

    In first-price auctions with n uniform bidders, the
    optimal bid is (n-1)/n * value. The shade factor
    controls how aggressively to shade the bid.

    Args:
        factor: Fraction of value to bid (0.0 to 1.0).
            Default 0.5 (bid half of value).
    """

    def __init__(self, factor: float = 0.5) -> None:
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"factor must be in [0, 1], got {factor}")
        self._factor = factor

    @property
    def name(self) -> str:
        return f"shade_bidder_{self._factor}"

    def choose_action(self, observation: Observation) -> Any:
        value = observation.game_state.get("your_private_value", 0.0)
        bid_range = observation.game_state.get("bid_range", [0.0, 100.0])
        bid = float(value) * self._factor
        return max(bid_range[0], min(bid_range[1], bid))


class RandomBidder(Strategy):
    """Bids randomly within the allowed range.

    Useful as a baseline for measuring strategy quality.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random_bidder"

    def choose_action(self, observation: Observation) -> Any:
        bid_range = observation.game_state.get("bid_range", [0.0, 100.0])
        return self._rng.uniform(bid_range[0], bid_range[1])

    def reset(self) -> None:
        pass
