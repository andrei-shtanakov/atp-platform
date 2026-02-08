"""Sealed-bid auction game implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from game_envs.core.action import ContinuousActionSpace
from game_envs.core.communication import InformationSet
from game_envs.core.game import Game, GameConfig, GameType, MoveOrder
from game_envs.core.state import (
    GameState,
    Observation,
    RoundResult,
    StepResult,
)
from game_envs.games.registry import register_game


class AuctionType(StrEnum):
    """Type of sealed-bid auction."""

    FIRST_PRICE = "first_price"
    SECOND_PRICE = "second_price"


class ValueDistribution(StrEnum):
    """Distribution for private value draws."""

    UNIFORM = "uniform"
    NORMAL = "normal"


@dataclass(frozen=True)
class AuctionConfig(GameConfig):
    """Sealed-bid auction configuration.

    Attributes:
        auction_type: First-price or second-price (Vickrey).
        min_bid: Minimum allowed bid.
        max_bid: Maximum allowed bid.
        reserve_price: Minimum price for a sale to occur.
        value_distribution: How private values are drawn.
        value_min: Minimum private value (uniform) or
            mean minus 3*std (normal).
        value_max: Maximum private value (uniform) or
            mean plus 3*std (normal).
    """

    auction_type: str = AuctionType.FIRST_PRICE
    min_bid: float = 0.0
    max_bid: float = 100.0
    reserve_price: float = 0.0
    value_distribution: str = ValueDistribution.UNIFORM
    value_min: float = 0.0
    value_max: float = 100.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.auction_type not in (
            AuctionType.FIRST_PRICE,
            AuctionType.SECOND_PRICE,
        ):
            raise ValueError(
                f"auction_type must be 'first_price' or "
                f"'second_price', got '{self.auction_type}'"
            )
        if self.min_bid < 0:
            raise ValueError(f"min_bid must be >= 0, got {self.min_bid}")
        if self.max_bid <= self.min_bid:
            raise ValueError(
                f"max_bid ({self.max_bid}) must be > min_bid ({self.min_bid})"
            )
        if self.reserve_price < 0:
            raise ValueError(f"reserve_price must be >= 0, got {self.reserve_price}")
        if self.value_distribution not in (
            ValueDistribution.UNIFORM,
            ValueDistribution.NORMAL,
        ):
            raise ValueError(
                f"value_distribution must be 'uniform' or "
                f"'normal', got '{self.value_distribution}'"
            )
        if self.value_max <= self.value_min:
            raise ValueError(
                f"value_max ({self.value_max}) must be > value_min ({self.value_min})"
            )
        if self.num_players < 2:
            raise ValueError(
                f"num_players must be >= 2 for auction, got {self.num_players}"
            )


@register_game("auction", AuctionConfig)
class Auction(Game):
    """Sealed-bid auction game.

    Each player receives a private value drawn from a
    distribution and simultaneously submits a sealed bid.
    The highest bidder wins the item.

    Payment rules:
        - First-price: winner pays own bid.
        - Second-price (Vickrey): winner pays second-highest
          bid.

    Partial observability: each player sees only their own
    private value, not others'.

    Theoretical predictions:
        - Second-price: truthful bidding is dominant
          (bid = value).
        - First-price with uniform values: optimal bid
          = (n-1)/n * value.
    """

    def __init__(self, config: AuctionConfig | None = None) -> None:
        super().__init__(config or AuctionConfig())
        self._terminal = False
        self._cumulative: dict[str, float] = {pid: 0.0 for pid in self.player_ids}
        self._private_values: dict[str, float] = {}

    @property
    def _auction_config(self) -> AuctionConfig:
        """Typed access to auction-specific config."""
        return self.config  # type: ignore[return-value]

    @property
    def name(self) -> str:
        c = self._auction_config
        atype = (
            "First-Price"
            if c.auction_type == AuctionType.FIRST_PRICE
            else "Second-Price (Vickrey)"
        )
        base = f"Sealed-Bid Auction ({atype})"
        if c.num_rounds > 1:
            base += f" (repeated x{c.num_rounds})"
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

    def action_space(self, player_id: str) -> ContinuousActionSpace:
        """Bid action space."""
        c = self._auction_config
        return ContinuousActionSpace(
            low=c.min_bid,
            high=c.max_bid,
            description=(f"Submit a bid between {c.min_bid} and {c.max_bid}"),
        )

    def _draw_private_value(self) -> float:
        """Draw a private value from the configured distribution."""
        c = self._auction_config
        if c.value_distribution == ValueDistribution.UNIFORM:
            return self._rng.uniform(c.value_min, c.value_max)
        # Normal distribution: mean at midpoint, std so
        # ~99.7% falls in [value_min, value_max]
        mean = (c.value_min + c.value_max) / 2.0
        std = (c.value_max - c.value_min) / 6.0
        value = self._rng.gauss(mean, std)
        return max(c.value_min, min(c.value_max, value))

    def reset(self) -> StepResult:
        """Reset and draw new private values."""
        self._reset_base()
        self._terminal = False
        self._cumulative = {pid: 0.0 for pid in self.player_ids}
        # Draw private values for all players
        self._private_values = {
            pid: self._draw_private_value() for pid in self.player_ids
        }
        state = GameState(
            round_number=0,
            player_states={
                pid: {"value": self._private_values[pid]} for pid in self.player_ids
            },
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
        """Process bids and determine winner."""
        if self._terminal:
            raise RuntimeError("Game is already terminal")

        c = self._auction_config
        bids = {pid: float(actions[pid]) for pid in self.player_ids}

        # Sort by bid descending, then by player_id for
        # deterministic tie-breaking (lower id wins)
        ranked = sorted(
            bids.items(),
            key=lambda x: (-x[1], x[0]),
        )

        highest_bid = ranked[0][1]
        winner_id: str | None = None
        price = 0.0

        # Check reserve price
        if highest_bid >= c.reserve_price:
            winner_id = ranked[0][0]
            if c.auction_type == AuctionType.FIRST_PRICE:
                price = highest_bid
            else:
                # Second-price: pay second-highest bid
                second_bid = ranked[1][1] if len(ranked) > 1 else 0.0
                price = max(second_bid, c.reserve_price)

        # Compute payoffs
        payoffs: dict[str, float] = {}
        for pid in self.player_ids:
            if pid == winner_id:
                payoffs[pid] = self._private_values[pid] - price
            else:
                payoffs[pid] = 0.0

        # Apply discount factor
        discount = c.discount_factor**self._current_round
        discounted = {pid: p * discount for pid, p in payoffs.items()}
        for pid in self.player_ids:
            self._cumulative[pid] += discounted[pid]

        current_round_number = self._current_round
        self._current_round += 1

        # Record round result (bids visible after round)
        info: dict[str, Any] = {
            "winner": winner_id,
            "price": price,
        }
        rr = RoundResult(
            round_number=current_round_number,
            actions=bids,
            payoffs=payoffs,
        )
        self._history.add_round(rr)

        if self._current_round >= self.config.num_rounds:
            self._terminal = True
        else:
            # Draw new private values for next round
            self._private_values = {
                pid: self._draw_private_value() for pid in self.player_ids
            }

        state = GameState(
            round_number=self._current_round,
            player_states={
                pid: {"value": self._private_values[pid]} for pid in self.player_ids
            },
            public_state={
                "game": self.name,
                "last_winner": winner_id,
                "last_price": price,
            },
            is_terminal=self._terminal,
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs=payoffs,
            is_terminal=self._terminal,
            info=info,
        )

    def get_information_set(
        self,
        player_id: str,
    ) -> InformationSet:
        """Auction uses partial observability.

        Each player can only see their own actions and
        payoffs in history. Other players' bids are hidden.
        """
        return InformationSet(
            player_id=player_id,
            visible_players=[player_id],
        )

    def observe(self, player_id: str) -> Observation:
        """Partial observability: player sees own value only."""
        c = self._auction_config
        game_state: dict[str, Any] = {
            "game": self.name,
            "your_role": player_id,
            "auction_type": str(c.auction_type),
            "num_bidders": c.num_players,
            "reserve_price": c.reserve_price,
            "bid_range": [c.min_bid, c.max_bid],
            "your_private_value": self._private_values.get(player_id, 0.0),
        }
        if c.auction_type == AuctionType.FIRST_PRICE:
            game_state["payment_rule"] = "Winner pays their own bid"
        else:
            game_state["payment_rule"] = "Winner pays the second-highest bid"

        info_set = self.get_information_set(player_id)
        filtered_history = info_set.filter_history(self._history.for_player(player_id))

        return Observation(
            player_id=player_id,
            game_state=game_state,
            available_actions=self.action_space(player_id).to_list(),
            history=filtered_history,
            round_number=self._current_round,
            total_rounds=self.config.num_rounds,
            messages=self._get_pending_messages(player_id),
        )

    def get_payoffs(self) -> dict[str, float]:
        """Get cumulative (discounted) payoffs."""
        return dict(self._cumulative)

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    @property
    def private_values(self) -> dict[str, float]:
        """Current private values (for testing/analysis)."""
        return dict(self._private_values)

    def to_prompt(self) -> str:
        """Describe the auction scenario for LLM agents."""
        c = self._auction_config
        atype = (
            "first-price"
            if c.auction_type == AuctionType.FIRST_PRICE
            else "second-price (Vickrey)"
        )
        lines = [
            f"This is a {atype} sealed-bid auction.",
            f"There are {c.num_players} bidders.",
            "",
            "Rules:",
            "- Each bidder has a private value for the "
            "item, drawn from a "
            f"{c.value_distribution} distribution "
            f"over [{c.value_min}, {c.value_max}].",
            f"- Bids must be between {c.min_bid} and {c.max_bid}.",
        ]
        if c.reserve_price > 0:
            lines.append(
                f"- Reserve price: {c.reserve_price}. "
                "No sale if all bids are below this."
            )
        if c.auction_type == AuctionType.FIRST_PRICE:
            lines.extend(
                [
                    "- The highest bidder wins and pays their own bid.",
                    "- Payoff = private value - bid (if you win), 0 otherwise.",
                ]
            )
        else:
            lines.extend(
                [
                    "- The highest bidder wins and pays the second-highest bid.",
                    "- Payoff = private value - second-highest "
                    "bid (if you win), 0 otherwise.",
                    "",
                    "Strategy hint: In a second-price auction, "
                    "bidding your true value is a dominant "
                    "strategy.",
                ]
            )
        if c.num_rounds > 1:
            lines.append(
                f"\nThis auction is repeated for "
                f"{c.num_rounds} rounds with new value "
                "draws each round."
            )
        return "\n".join(lines)
