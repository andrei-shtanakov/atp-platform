"""ATP Games plugin for game-theoretic agent evaluation."""

from atp_games.evaluators.exploitability_evaluator import (
    ExploitabilityEvaluator,
)
from atp_games.evaluators.payoff_evaluator import PayoffEvaluator
from atp_games.mapping.action_mapper import ActionMapper
from atp_games.mapping.observation_mapper import ObservationMapper
from atp_games.models import (
    AgentComparison,
    EpisodeResult,
    GameResult,
    GameRunConfig,
    PlayerStats,
    compare_agents,
)
from atp_games.runner.action_validator import ActionValidator
from atp_games.runner.builtin_adapter import BuiltinAdapter
from atp_games.runner.game_runner import GameRunner, ProgressReporter
from atp_games.suites.alympics import (
    AlympicsResult,
    CategoryScore,
    run_alympics,
    score_benchmark,
)
from atp_games.suites.cross_play import CrossPlayResult, run_cross_play
from atp_games.suites.game_suite_loader import GameSuiteLoader
from atp_games.suites.models import GameSuiteConfig
from atp_games.suites.stress_test import StressTestResult, run_stress_test
from atp_games.suites.tournament import (
    TournamentResult,
    run_double_elimination,
    run_round_robin,
    run_single_elimination,
)

__all__ = [
    "ActionMapper",
    "ActionValidator",
    "AgentComparison",
    "AlympicsResult",
    "BuiltinAdapter",
    "CategoryScore",
    "CrossPlayResult",
    "EpisodeResult",
    "ExploitabilityEvaluator",
    "GameResult",
    "GameRunConfig",
    "GameRunner",
    "GameSuiteConfig",
    "GameSuiteLoader",
    "ObservationMapper",
    "PayoffEvaluator",
    "PlayerStats",
    "ProgressReporter",
    "StressTestResult",
    "TournamentResult",
    "compare_agents",
    "run_alympics",
    "run_cross_play",
    "run_double_elimination",
    "run_round_robin",
    "run_single_elimination",
    "run_stress_test",
    "score_benchmark",
]
