"""Game environments for agent evaluation."""

from game_envs.analysis.exploitability import (
    EmpiricalStrategy,
    ExploitabilityResult,
    compute_best_response,
    compute_exploitability,
    compute_exploitability_from_game,
)
from game_envs.analysis.models import NashEquilibrium
from game_envs.analysis.nash_solver import NashSolver
from game_envs.core.action import (
    ActionSpace,
    ContinuousActionSpace,
    DiscreteActionSpace,
    StructuredActionSpace,
)
from game_envs.core.communication import (
    CommunicationChannel,
    CommunicationMode,
    InformationSet,
)
from game_envs.core.game import Game, GameConfig, GameType, MoveOrder
from game_envs.core.history import GameHistory
from game_envs.core.state import (
    GameState,
    Message,
    Observation,
    RoundResult,
    StepResult,
)
from game_envs.core.strategy import Strategy
from game_envs.games.auction import Auction, AuctionConfig
from game_envs.games.colonel_blotto import BlottoConfig, ColonelBlotto
from game_envs.games.congestion import CongestionConfig, CongestionGame, RouteDefinition
from game_envs.games.prisoners_dilemma import PDConfig, PrisonersDilemma
from game_envs.games.public_goods import PGConfig, PublicGoodsGame
from game_envs.games.registry import GameRegistry, register_game
from game_envs.strategies import (
    AlwaysCooperate,
    AlwaysDefect,
    ConcentratedAllocation,
    ConditionalCooperator,
    EpsilonGreedy,
    FreeRider,
    FullContributor,
    GrimTrigger,
    NashMixed,
    Pavlov,
    Punisher,
    RandomBidder,
    RandomStrategy,
    SelfishRouter,
    ShadeBidder,
    SocialOptimum,
    StrategyRegistry,
    TitForTat,
    TruthfulBidder,
    UniformAllocation,
)

__all__ = [
    "ActionSpace",
    "CommunicationChannel",
    "CommunicationMode",
    "EmpiricalStrategy",
    "ExploitabilityResult",
    "AlwaysCooperate",
    "AlwaysDefect",
    "Auction",
    "AuctionConfig",
    "BlottoConfig",
    "ColonelBlotto",
    "CongestionConfig",
    "CongestionGame",
    "ConcentratedAllocation",
    "ConditionalCooperator",
    "ContinuousActionSpace",
    "DiscreteActionSpace",
    "EpsilonGreedy",
    "FreeRider",
    "FullContributor",
    "Game",
    "GameConfig",
    "GameHistory",
    "GameRegistry",
    "GameState",
    "GameType",
    "GrimTrigger",
    "InformationSet",
    "Message",
    "MoveOrder",
    "NashEquilibrium",
    "NashMixed",
    "NashSolver",
    "Observation",
    "PDConfig",
    "PGConfig",
    "Pavlov",
    "PrisonersDilemma",
    "PublicGoodsGame",
    "Punisher",
    "RandomBidder",
    "RandomStrategy",
    "RouteDefinition",
    "RoundResult",
    "SelfishRouter",
    "ShadeBidder",
    "SocialOptimum",
    "StepResult",
    "Strategy",
    "StrategyRegistry",
    "StructuredActionSpace",
    "TitForTat",
    "TruthfulBidder",
    "UniformAllocation",
    "compute_best_response",
    "compute_exploitability",
    "compute_exploitability_from_game",
    "register_game",
]

# Register built-in games
GameRegistry.register("auction", Auction, AuctionConfig)
GameRegistry.register("colonel_blotto", ColonelBlotto, BlottoConfig)
GameRegistry.register("congestion", CongestionGame, CongestionConfig)
GameRegistry.register("prisoners_dilemma", PrisonersDilemma, PDConfig)
GameRegistry.register("public_goods", PublicGoodsGame, PGConfig)

# Register built-in strategies
StrategyRegistry.register("always_cooperate", AlwaysCooperate)
StrategyRegistry.register("always_defect", AlwaysDefect)
StrategyRegistry.register("tit_for_tat", TitForTat)
StrategyRegistry.register("grim_trigger", GrimTrigger)
StrategyRegistry.register("pavlov", Pavlov)
StrategyRegistry.register("random", RandomStrategy)
StrategyRegistry.register("full_contributor", FullContributor)
StrategyRegistry.register("free_rider", FreeRider)
StrategyRegistry.register("conditional_cooperator", ConditionalCooperator)
StrategyRegistry.register("punisher", Punisher)
StrategyRegistry.register("truthful_bidder", TruthfulBidder)
StrategyRegistry.register("shade_bidder", ShadeBidder)
StrategyRegistry.register("random_bidder", RandomBidder)
StrategyRegistry.register("uniform_allocation", UniformAllocation)
StrategyRegistry.register("concentrated_allocation", ConcentratedAllocation)
StrategyRegistry.register("nash_mixed", NashMixed)
StrategyRegistry.register("selfish_router", SelfishRouter)
StrategyRegistry.register("social_optimum", SocialOptimum)
StrategyRegistry.register("epsilon_greedy", EpsilonGreedy)
