"""Score aggregation for test results."""

from .aggregator import ScoreAggregator
from .models import (
    ComponentScore,
    NormalizationConfig,
    ScoreBreakdown,
    ScoredTestResult,
)

__all__ = [
    "ScoreAggregator",
    "ComponentScore",
    "NormalizationConfig",
    "ScoreBreakdown",
    "ScoredTestResult",
]
