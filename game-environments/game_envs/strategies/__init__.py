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
from game_envs.strategies.bos_strategies import (
    Alternating,
    AlwaysA,
    AlwaysB,
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
from game_envs.strategies.stag_hunt_strategies import (
    AlwaysHare,
    AlwaysStag,
    StagTitForTat,
)

__all__ = [
    "Alternating",
    "AlwaysA",
    "AlwaysB",
    "AlwaysCooperate",
    "AlwaysDefect",
    "AlwaysHare",
    "AlwaysStag",
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
    "StagTitForTat",
    "StrategyRegistry",
    "TitForTat",
    "TruthfulBidder",
    "UniformAllocation",
]
