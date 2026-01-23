"""Data models for statistical analysis of test runs."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StabilityLevel(str, Enum):
    """Stability level based on coefficient of variation."""

    STABLE = "stable"
    MODERATE = "moderate"
    UNSTABLE = "unstable"
    CRITICAL = "critical"


class StabilityAssessment(BaseModel):
    """Assessment of result stability based on coefficient of variation.

    Thresholds (from DESIGN-006):
    - stable: CV < 0.05 - Consistent results
    - moderate: CV 0.05-0.15 - Acceptable variance
    - unstable: CV 0.15-0.30 - Results may be unreliable
    - critical: CV > 0.30 - Unpredictable behavior
    """

    level: StabilityLevel = Field(..., description="Stability level")
    cv: float = Field(..., description="Coefficient of variation", ge=0.0)
    message: str = Field(..., description="Human-readable assessment message")


class StatisticalResult(BaseModel):
    """Statistical summary of values from multiple runs.

    Implements DESIGN-006 metrics for statistical analysis.
    """

    mean: float = Field(..., description="Arithmetic mean")
    std: float = Field(..., description="Standard deviation", ge=0.0)
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    median: float = Field(..., description="Median value")
    confidence_interval: tuple[float, float] = Field(
        ..., description="95% confidence interval (lower, upper)"
    )
    n_runs: int = Field(..., description="Number of runs", ge=1)
    coefficient_of_variation: float = Field(
        ..., description="Coefficient of variation (std/mean)", ge=0.0
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "median": round(self.median, 4),
            "confidence_interval": (
                round(self.confidence_interval[0], 4),
                round(self.confidence_interval[1], 4),
            ),
            "n_runs": self.n_runs,
            "coefficient_of_variation": round(self.coefficient_of_variation, 4),
        }


class MetricStatistics(BaseModel):
    """Statistics for a specific metric across runs."""

    metric_name: str = Field(..., description="Name of the metric")
    stats: StatisticalResult = Field(..., description="Statistical results")
    stability: StabilityAssessment = Field(..., description="Stability assessment")
    unit: str | None = Field(None, description="Unit of measurement (e.g., 'seconds')")


class TestRunStatistics(BaseModel):
    """Complete statistical analysis for a test with multiple runs."""

    test_id: str = Field(..., description="Test identifier")
    n_runs: int = Field(..., description="Total number of runs", ge=1)
    successful_runs: int = Field(..., description="Number of successful runs", ge=0)
    success_rate: float = Field(
        ..., description="Proportion of successful runs", ge=0.0, le=1.0
    )

    score_stats: StatisticalResult | None = Field(
        None, description="Statistics for test scores"
    )
    score_stability: StabilityAssessment | None = Field(
        None, description="Score stability assessment"
    )

    duration_stats: StatisticalResult | None = Field(
        None, description="Statistics for execution duration"
    )
    steps_stats: StatisticalResult | None = Field(
        None, description="Statistics for steps taken"
    )
    tokens_stats: StatisticalResult | None = Field(
        None, description="Statistics for token usage"
    )
    cost_stats: StatisticalResult | None = Field(
        None, description="Statistics for cost"
    )

    overall_stability: StabilityAssessment = Field(
        ..., description="Overall stability assessment"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        result: dict[str, Any] = {
            "test_id": self.test_id,
            "n_runs": self.n_runs,
            "successful_runs": self.successful_runs,
            "success_rate": round(self.success_rate, 4),
            "overall_stability": {
                "level": self.overall_stability.level.value,
                "cv": round(self.overall_stability.cv, 4),
                "message": self.overall_stability.message,
            },
        }

        if self.score_stats:
            result["score"] = self.score_stats.to_dict()
            if self.score_stability:
                result["score"]["stability"] = {
                    "level": self.score_stability.level.value,
                    "cv": round(self.score_stability.cv, 4),
                    "message": self.score_stability.message,
                }

        if self.duration_stats:
            result["duration_seconds"] = self.duration_stats.to_dict()

        if self.steps_stats:
            result["steps"] = self.steps_stats.to_dict()

        if self.tokens_stats:
            result["tokens"] = self.tokens_stats.to_dict()

        if self.cost_stats:
            result["cost_usd"] = self.cost_stats.to_dict()

        return result
