"""Game suite YAML loading, validation, and advanced modes."""

from atp_games.suites.alympics import (
    AlympicsResult,
    CategoryScore,
    run_alympics,
    score_benchmark,
)
from atp_games.suites.cross_play import CrossPlayResult, run_cross_play
from atp_games.suites.game_suite_loader import GameSuiteLoader
from atp_games.suites.models import (
    GameAgentConfig,
    GameEvaluationConfig,
    GameMetricConfig,
    GameSuiteConfig,
)
from atp_games.suites.schema import GAME_SUITE_SCHEMA, validate_game_suite_schema
from atp_games.suites.stress_test import StressTestResult, run_stress_test
from atp_games.suites.tournament import (
    TournamentResult,
    run_double_elimination,
    run_round_robin,
    run_single_elimination,
)

__all__ = [
    "AlympicsResult",
    "CategoryScore",
    "CrossPlayResult",
    "GAME_SUITE_SCHEMA",
    "GameAgentConfig",
    "GameEvaluationConfig",
    "GameMetricConfig",
    "GameSuiteConfig",
    "GameSuiteLoader",
    "StressTestResult",
    "TournamentResult",
    "run_alympics",
    "run_cross_play",
    "run_double_elimination",
    "run_round_robin",
    "run_single_elimination",
    "run_stress_test",
    "score_benchmark",
    "validate_game_suite_schema",
]
