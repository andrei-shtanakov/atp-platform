"""Game-theoretic analysis modules."""

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

__all__ = [
    "CooperationMetrics",
    "EmpiricalStrategy",
    "ExploitabilityResult",
    "FairnessMetrics",
    "MoranProcess",
    "NashEquilibrium",
    "NashSolver",
    "PopulationDynamics",
    "PopulationResult",
    "PopulationSimulator",
    "PopulationSnapshot",
    "ReplicatorDynamics",
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
    "utilitarian_welfare",
]
