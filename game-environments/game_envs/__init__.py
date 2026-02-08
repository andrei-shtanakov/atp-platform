"""Game environments for agent evaluation."""

from game_envs.analysis.cooperation import (
    CooperationMetrics,
    conditional_cooperation,
    cooperation_rate,
    reciprocity_index,
)
from game_envs.analysis.exploitability import (
    EmpiricalStrategy,
    ExploitabilityResult,
    compute_best_response,
    compute_exploitability,
    compute_exploitability_from_game,
)
from game_envs.analysis.fairness import (
    FairnessMetrics,
    envy_freeness,
    gini_coefficient,
    proportionality,
    utilitarian_welfare,
)
from game_envs.analysis.models import NashEquilibrium
from game_envs.analysis.nash_solver import NashSolver
from game_envs.analysis.population import (
    MoranProcess,
    PopulationDynamics,
    PopulationResult,
    PopulationSimulator,
    PopulationSnapshot,
    ReplicatorDynamics,
    is_ess,
)
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
    "AlwaysCooperate",
    "AlwaysDefect",
    "Auction",
    "AuctionConfig",
    "BlottoConfig",
    "ColonelBlotto",
    "CommunicationChannel",
    "CommunicationMode",
    "ConcentratedAllocation",
    "ConditionalCooperator",
    "CongestionConfig",
    "CongestionGame",
    "ContinuousActionSpace",
    "CooperationMetrics",
    "DiscreteActionSpace",
    "EmpiricalStrategy",
    "EpsilonGreedy",
    "ExploitabilityResult",
    "FairnessMetrics",
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
    "MoranProcess",
    "MoveOrder",
    "NashEquilibrium",
    "NashMixed",
    "NashSolver",
    "Observation",
    "PDConfig",
    "PGConfig",
    "Pavlov",
    "PopulationDynamics",
    "PopulationResult",
    "PopulationSimulator",
    "PopulationSnapshot",
    "PrisonersDilemma",
    "PublicGoodsGame",
    "Punisher",
    "RandomBidder",
    "RandomStrategy",
    "ReplicatorDynamics",
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
    "conditional_cooperation",
    "cooperation_rate",
    "envy_freeness",
    "gini_coefficient",
    "is_ess",
    "proportionality",
    "reciprocity_index",
    "register_game",
    "utilitarian_welfare",
]

# Built-in games are auto-registered via @register_game decorators
# on their class definitions (see game_envs/games/*.py)

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
