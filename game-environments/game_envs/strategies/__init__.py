"""Built-in baseline strategies for game environments."""

from game_envs.strategies.auction_strategies import (
    RandomBidder,
    ShadeBidder,
    TruthfulBidder,
)
from game_envs.strategies.blotto_strategies import (
    ConcentratedAllocation,
    NashMixed,
    UniformAllocation,
)
from game_envs.strategies.congestion_strategies import (
    EpsilonGreedy,
    SelfishRouter,
    SocialOptimum,
)
from game_envs.strategies.pd_strategies import (
    AlwaysCooperate,
    AlwaysDefect,
    GrimTrigger,
    Pavlov,
    RandomStrategy,
    TitForTat,
)
from game_envs.strategies.pg_strategies import (
    ConditionalCooperator,
    FreeRider,
    FullContributor,
    Punisher,
)
from game_envs.strategies.registry import StrategyRegistry

__all__ = [
    "AlwaysCooperate",
    "AlwaysDefect",
    "ConcentratedAllocation",
    "ConditionalCooperator",
    "EpsilonGreedy",
    "FreeRider",
    "FullContributor",
    "GrimTrigger",
    "NashMixed",
    "Pavlov",
    "Punisher",
    "RandomBidder",
    "RandomStrategy",
    "SelfishRouter",
    "ShadeBidder",
    "SocialOptimum",
    "StrategyRegistry",
    "TitForTat",
    "TruthfulBidder",
    "UniformAllocation",
]
