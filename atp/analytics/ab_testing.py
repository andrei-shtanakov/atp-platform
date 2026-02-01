"""A/B Testing Framework for ATP.

This module provides a complete A/B testing framework for comparing agent versions
with statistical rigor. It includes experiment lifecycle management, traffic routing,
statistical significance calculation, and automatic rollback on degradation.

Features:
- Define A/B experiments with traffic split
- Automatic statistical significance calculation
- Experiment lifecycle: draft -> running -> concluded
- Winner determination with confidence intervals
- Automatic rollback on degradation
- Experiment history and reports
"""

import hashlib
import math
import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# t-distribution critical values for 95% confidence interval
# Indexed by degrees of freedom (n-1), capped at df=30+
_T_CRITICAL_VALUES: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _get_t_critical(df: int) -> float:
    """Get t-critical value for given degrees of freedom."""
    if df <= 0:
        return float("inf")
    if df > 30:
        return 1.96
    return _T_CRITICAL_VALUES.get(df, 1.96)


def _calculate_p_value(t_stat: float, df: float) -> float:
    """Calculate two-tailed p-value from t-statistic and degrees of freedom."""
    if t_stat <= 0:
        return 1.0
    if df < 1:
        return 1.0

    t_critical_95 = _get_t_critical(int(df))

    if t_stat < t_critical_95 * 0.5:
        return min(1.0, 1.0 - (t_stat / t_critical_95) * 0.5)
    elif t_stat < t_critical_95:
        ratio = t_stat / t_critical_95
        return max(0.05, 0.5 * (1.0 - ratio) + 0.05)
    elif t_stat < t_critical_95 * 1.5:
        ratio = (t_stat - t_critical_95) / (t_critical_95 * 0.5)
        return max(0.01, 0.05 * (1.0 - ratio))
    elif t_stat < t_critical_95 * 2.5:
        ratio = (t_stat - t_critical_95 * 1.5) / t_critical_95
        return max(0.001, 0.01 * (1.0 - ratio * 0.9))
    else:
        return 0.001


def _welchs_t_test(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
) -> tuple[float, float]:
    """Perform Welch's t-test for two samples with unequal variances.

    Args:
        mean1: Mean of first sample.
        std1: Standard deviation of first sample.
        n1: Size of first sample.
        mean2: Mean of second sample.
        std2: Standard deviation of second sample.
        n2: Size of second sample.

    Returns:
        Tuple of (t_statistic, p_value).
    """
    if n1 < 2 or n2 < 2:
        return (0.0, 1.0)

    var1 = std1**2
    var2 = std2**2

    se1 = var1 / n1
    se2 = var2 / n2
    se_sum = se1 + se2

    if se_sum < 1e-10:
        if abs(mean1 - mean2) < 1e-10:
            return (0.0, 1.0)
        return (float("inf"), 0.0)

    t_stat = (mean1 - mean2) / math.sqrt(se_sum)

    numerator = se_sum**2
    denominator = (se1**2 / (n1 - 1)) + (se2**2 / (n2 - 1))

    if denominator < 1e-10:
        df = min(n1, n2) - 1
    else:
        df = numerator / denominator

    p_value = _calculate_p_value(abs(t_stat), df)

    return (t_stat, p_value)


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


# ==================== Enums ====================


class ExperimentStatus(str, Enum):
    """Status of an A/B experiment."""

    DRAFT = "draft"  # Experiment is being configured
    RUNNING = "running"  # Experiment is active
    PAUSED = "paused"  # Temporarily paused
    CONCLUDED = "concluded"  # Experiment finished
    ROLLED_BACK = "rolled_back"  # Experiment rolled back due to degradation


class VariantType(str, Enum):
    """Type of experiment variant."""

    CONTROL = "control"  # Baseline variant (A)
    TREATMENT = "treatment"  # Test variant (B)


class WinnerDecision(str, Enum):
    """Winner determination result."""

    CONTROL = "control"  # Control variant wins
    TREATMENT = "treatment"  # Treatment variant wins
    INCONCLUSIVE = "inconclusive"  # Not enough data or no significant difference
    TIE = "tie"  # Both variants perform equally


class MetricType(str, Enum):
    """Type of metric to track in experiment."""

    SCORE = "score"  # Test score (higher is better)
    SUCCESS_RATE = "success_rate"  # Pass rate (higher is better)
    DURATION = "duration"  # Execution time (lower is better)
    COST = "cost"  # Cost (lower is better)
    TOKENS = "tokens"  # Token usage (lower is better)


# ==================== Pydantic Models ====================


class Variant(BaseModel):
    """Configuration for an experiment variant."""

    name: str = Field(..., description="Variant name (e.g., 'control', 'treatment')")
    variant_type: VariantType = Field(..., description="Type of variant")
    agent_name: str = Field(..., description="Agent to use for this variant")
    traffic_weight: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Traffic allocation percentage (0-100)",
    )
    description: str | None = Field(None, description="Variant description")


class MetricConfig(BaseModel):
    """Configuration for a metric to track."""

    metric_type: MetricType = Field(..., description="Type of metric")
    is_primary: bool = Field(
        default=False, description="Whether this is the primary metric"
    )
    minimize: bool = Field(
        default=False,
        description="Whether lower values are better (for duration, cost)",
    )
    min_effect_size: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum effect size to detect",
    )


class VariantMetrics(BaseModel):
    """Aggregated metrics for a variant."""

    variant_name: str = Field(..., description="Name of the variant")
    sample_size: int = Field(default=0, description="Number of observations")
    mean: float = Field(default=0.0, description="Mean value")
    std: float = Field(default=0.0, description="Standard deviation")
    min_value: float | None = Field(None, description="Minimum value")
    max_value: float | None = Field(None, description="Maximum value")
    ci_lower: float | None = Field(None, description="95% CI lower bound")
    ci_upper: float | None = Field(None, description="95% CI upper bound")


class StatisticalResult(BaseModel):
    """Result of statistical significance test."""

    metric_type: MetricType = Field(..., description="Metric being compared")
    control_metrics: VariantMetrics = Field(..., description="Control variant metrics")
    treatment_metrics: VariantMetrics = Field(
        ..., description="Treatment variant metrics"
    )
    t_statistic: float = Field(default=0.0, description="Welch's t-test statistic")
    p_value: float = Field(default=1.0, description="p-value from t-test")
    is_significant: bool = Field(
        default=False, description="Whether result is significant (p < 0.05)"
    )
    effect_size: float = Field(default=0.0, description="Cohen's d effect size")
    relative_change: float = Field(
        default=0.0, description="Relative change percentage"
    )
    confidence_level: float = Field(default=0.95, description="Confidence level used")
    winner: WinnerDecision = Field(
        default=WinnerDecision.INCONCLUSIVE,
        description="Winner determination",
    )


class RollbackConfig(BaseModel):
    """Configuration for automatic rollback."""

    enabled: bool = Field(default=True, description="Whether rollback is enabled")
    degradation_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Relative degradation threshold to trigger rollback",
    )
    min_samples_before_rollback: int = Field(
        default=30,
        ge=10,
        description="Minimum samples before considering rollback",
    )
    consecutive_checks: int = Field(
        default=3,
        ge=1,
        description="Number of consecutive degradation checks before rollback",
    )


class ExperimentConfig(BaseModel):
    """Configuration for an A/B experiment."""

    name: str = Field(..., min_length=1, max_length=100, description="Experiment name")
    description: str | None = Field(None, max_length=2000, description="Description")
    suite_name: str = Field(..., description="Test suite to run experiment on")
    test_ids: list[str] | None = Field(
        None, description="Specific test IDs to include (None = all tests)"
    )
    control_variant: Variant = Field(..., description="Control variant (A)")
    treatment_variant: Variant = Field(..., description="Treatment variant (B)")
    metrics: list[MetricConfig] = Field(
        default_factory=lambda: [
            MetricConfig(metric_type=MetricType.SCORE, is_primary=True)
        ],
        description="Metrics to track",
    )
    rollback: RollbackConfig = Field(
        default_factory=RollbackConfig,
        description="Rollback configuration",
    )
    min_sample_size: int = Field(
        default=30,
        ge=10,
        description="Minimum sample size per variant for significance",
    )
    max_sample_size: int | None = Field(
        None, description="Maximum sample size (auto-conclude when reached)"
    )
    max_duration_days: int | None = Field(
        None, description="Maximum duration in days (auto-conclude when reached)"
    )
    significance_level: float = Field(
        default=0.05,
        ge=0.01,
        le=0.10,
        description="Significance level (alpha)",
    )


class ExperimentObservation(BaseModel):
    """Single observation in an experiment."""

    experiment_id: int = Field(..., description="Experiment ID")
    variant_name: str = Field(..., description="Variant that was assigned")
    test_id: str = Field(..., description="Test ID")
    run_id: str = Field(..., description="Unique run identifier")
    timestamp: datetime = Field(default_factory=_utcnow, description="Observation time")
    score: float | None = Field(None, description="Test score")
    success: bool = Field(default=False, description="Whether test passed")
    duration_seconds: float | None = Field(None, description="Execution duration")
    cost_usd: float | None = Field(None, description="Cost in USD")
    tokens: int | None = Field(None, description="Token usage")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class Experiment(BaseModel):
    """A/B Testing experiment."""

    id: int | None = Field(None, description="Experiment ID")
    config: ExperimentConfig = Field(..., description="Experiment configuration")
    status: ExperimentStatus = Field(
        default=ExperimentStatus.DRAFT, description="Current status"
    )
    created_at: datetime = Field(default_factory=_utcnow, description="Creation time")
    started_at: datetime | None = Field(None, description="When experiment started")
    concluded_at: datetime | None = Field(None, description="When experiment concluded")
    paused_at: datetime | None = Field(None, description="When experiment was paused")
    conclusion_reason: str | None = Field(None, description="Reason for conclusion")

    # Observations stored per variant
    control_observations: list[ExperimentObservation] = Field(
        default_factory=list, description="Control variant observations"
    )
    treatment_observations: list[ExperimentObservation] = Field(
        default_factory=list, description="Treatment variant observations"
    )

    # Results
    results: list[StatisticalResult] = Field(
        default_factory=list, description="Statistical results per metric"
    )
    winner: WinnerDecision = Field(
        default=WinnerDecision.INCONCLUSIVE, description="Overall winner"
    )
    rollback_triggered: bool = Field(
        default=False, description="Whether rollback was triggered"
    )
    consecutive_degradation_checks: int = Field(
        default=0, description="Current consecutive degradation count"
    )

    @property
    def control_sample_size(self) -> int:
        """Get control variant sample size."""
        return len(self.control_observations)

    @property
    def treatment_sample_size(self) -> int:
        """Get treatment variant sample size."""
        return len(self.treatment_observations)

    @property
    def total_sample_size(self) -> int:
        """Get total sample size."""
        return self.control_sample_size + self.treatment_sample_size

    @property
    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        return self.status == ExperimentStatus.RUNNING

    @property
    def can_conclude(self) -> bool:
        """Check if experiment has enough data to conclude."""
        min_size = self.config.min_sample_size
        return (
            self.control_sample_size >= min_size
            and self.treatment_sample_size >= min_size
        )

    @property
    def should_auto_conclude(self) -> bool:
        """Check if experiment should auto-conclude."""
        # Check max sample size
        if self.config.max_sample_size:
            if self.total_sample_size >= self.config.max_sample_size:
                return True

        # Check max duration
        if self.config.max_duration_days and self.started_at:
            max_end = self.started_at + timedelta(days=self.config.max_duration_days)
            if _utcnow() >= max_end:
                return True

        return False


class ExperimentReport(BaseModel):
    """Comprehensive report for an experiment."""

    experiment: Experiment = Field(..., description="The experiment")
    statistical_results: list[StatisticalResult] = Field(
        default_factory=list, description="Statistical analysis results"
    )
    recommendation: str = Field(
        default="",
        description="Recommendation based on results",
    )
    summary: dict[str, Any] = Field(
        default_factory=dict, description="Summary statistics"
    )


# ==================== Traffic Router ====================


class TrafficRouter:
    """Routes traffic to experiment variants based on configured split.

    Uses consistent hashing to ensure the same user/run always gets
    the same variant for reproducibility.
    """

    def __init__(self, experiment: Experiment) -> None:
        """Initialize traffic router.

        Args:
            experiment: The experiment to route traffic for.
        """
        self.experiment = experiment
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize traffic weights to sum to 100."""
        control = self.experiment.config.control_variant
        treatment = self.experiment.config.treatment_variant
        total = control.traffic_weight + treatment.traffic_weight
        if total > 0:
            self._control_threshold = control.traffic_weight / total
        else:
            self._control_threshold = 0.5

    def get_variant(
        self,
        run_id: str | None = None,
        deterministic: bool = True,
    ) -> Variant:
        """Get the variant assignment for a run.

        Args:
            run_id: Unique identifier for the run. If provided and
                deterministic=True, the same run_id will always get
                the same variant.
            deterministic: Whether to use consistent hashing.

        Returns:
            The assigned variant.
        """
        if deterministic and run_id:
            # Use consistent hashing for deterministic assignment
            hash_input = f"{self.experiment.id}:{run_id}"
            hash_value = hashlib.md5(hash_input.encode()).hexdigest()
            # Convert first 8 hex chars to float between 0 and 1
            normalized = int(hash_value[:8], 16) / 0xFFFFFFFF
        else:
            # Random assignment
            normalized = random.random()

        if normalized < self._control_threshold:
            return self.experiment.config.control_variant
        return self.experiment.config.treatment_variant


# ==================== Statistical Analysis ====================


def _compute_statistics(values: Sequence[float]) -> dict[str, Any]:
    """Compute basic statistics for a sequence of values.

    Args:
        values: Sequence of numeric values.

    Returns:
        Dictionary with mean, std, min, max, n, and confidence interval.
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": None,
            "max": None,
            "n": 0,
            "ci": (0.0, 0.0),
        }

    n = len(values)
    mean = sum(values) / n

    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    # Calculate 95% confidence interval
    if n > 1:
        df = n - 1
        t_critical = _get_t_critical(df)
        margin = t_critical * std / math.sqrt(n)
        ci = (mean - margin, mean + margin)
    else:
        ci = (mean, mean)

    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "n": n,
        "ci": ci,
    }


class ExperimentAnalyzer:
    """Performs statistical analysis on experiment data."""

    def __init__(self) -> None:
        """Initialize the analyzer."""
        pass

    def compute_variant_metrics(
        self,
        observations: Sequence[ExperimentObservation],
        metric_type: MetricType,
    ) -> VariantMetrics:
        """Compute metrics for a variant.

        Args:
            observations: List of observations.
            metric_type: Type of metric to compute.

        Returns:
            Aggregated metrics for the variant.
        """
        if not observations:
            return VariantMetrics(
                variant_name="",
                sample_size=0,
                mean=0.0,
                std=0.0,
            )

        # Extract values based on metric type
        values: list[float] = []
        for obs in observations:
            value = self._get_metric_value(obs, metric_type)
            if value is not None:
                values.append(value)

        if not values:
            return VariantMetrics(
                variant_name=observations[0].variant_name,
                sample_size=0,
                mean=0.0,
                std=0.0,
            )

        # Compute statistics
        stats = _compute_statistics(values)

        return VariantMetrics(
            variant_name=observations[0].variant_name,
            sample_size=stats["n"],
            mean=stats["mean"],
            std=stats["std"],
            min_value=stats["min"],
            max_value=stats["max"],
            ci_lower=stats["ci"][0],
            ci_upper=stats["ci"][1],
        )

    def _get_metric_value(
        self,
        obs: ExperimentObservation,
        metric_type: MetricType,
    ) -> float | None:
        """Extract metric value from observation.

        Args:
            obs: Observation to extract from.
            metric_type: Type of metric to extract.

        Returns:
            The metric value or None if not available.
        """
        if metric_type == MetricType.SCORE:
            return obs.score
        elif metric_type == MetricType.SUCCESS_RATE:
            return 1.0 if obs.success else 0.0
        elif metric_type == MetricType.DURATION:
            return obs.duration_seconds
        elif metric_type == MetricType.COST:
            return obs.cost_usd
        elif metric_type == MetricType.TOKENS:
            return float(obs.tokens) if obs.tokens else None
        return None

    def compute_statistical_result(
        self,
        experiment: Experiment,
        metric_config: MetricConfig,
    ) -> StatisticalResult:
        """Compute statistical significance for a metric.

        Args:
            experiment: The experiment to analyze.
            metric_config: Configuration for the metric.

        Returns:
            Statistical result with significance test.
        """
        control_metrics = self.compute_variant_metrics(
            experiment.control_observations,
            metric_config.metric_type,
        )
        control_metrics.variant_name = experiment.config.control_variant.name

        treatment_metrics = self.compute_variant_metrics(
            experiment.treatment_observations,
            metric_config.metric_type,
        )
        treatment_metrics.variant_name = experiment.config.treatment_variant.name

        # Perform Welch's t-test
        t_stat, p_value = _welchs_t_test(
            mean1=treatment_metrics.mean,
            std1=treatment_metrics.std,
            n1=treatment_metrics.sample_size,
            mean2=control_metrics.mean,
            std2=control_metrics.std,
            n2=control_metrics.sample_size,
        )

        # Calculate effect size (Cohen's d)
        pooled_std = math.sqrt((control_metrics.std**2 + treatment_metrics.std**2) / 2)
        effect_size = 0.0
        if pooled_std > 1e-10:
            effect_size = (treatment_metrics.mean - control_metrics.mean) / pooled_std

        # Calculate relative change
        relative_change = 0.0
        if abs(control_metrics.mean) > 1e-10:
            relative_change = (
                (treatment_metrics.mean - control_metrics.mean)
                / abs(control_metrics.mean)
                * 100
            )

        # Determine significance
        is_significant = p_value < experiment.config.significance_level

        # Determine winner
        winner = self._determine_winner(
            is_significant=is_significant,
            effect_size=effect_size,
            relative_change=relative_change,
            minimize=metric_config.minimize,
            min_effect_size=metric_config.min_effect_size,
        )

        return StatisticalResult(
            metric_type=metric_config.metric_type,
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            effect_size=effect_size,
            relative_change=relative_change,
            confidence_level=1 - experiment.config.significance_level,
            winner=winner,
        )

    def _determine_winner(
        self,
        is_significant: bool,
        effect_size: float,
        relative_change: float,
        minimize: bool,
        min_effect_size: float,
    ) -> WinnerDecision:
        """Determine the winner based on statistical results.

        Args:
            is_significant: Whether the result is statistically significant.
            effect_size: Cohen's d effect size.
            relative_change: Relative change percentage.
            minimize: Whether lower values are better.
            min_effect_size: Minimum effect size to consider meaningful.

        Returns:
            Winner determination.
        """
        if not is_significant:
            return WinnerDecision.INCONCLUSIVE

        if abs(effect_size) < min_effect_size:
            return WinnerDecision.TIE

        # For minimize metrics (duration, cost), negative effect = treatment is better
        # For maximize metrics (score), positive effect = treatment is better
        if minimize:
            if effect_size < -min_effect_size:
                return WinnerDecision.TREATMENT
            elif effect_size > min_effect_size:
                return WinnerDecision.CONTROL
        else:
            if effect_size > min_effect_size:
                return WinnerDecision.TREATMENT
            elif effect_size < -min_effect_size:
                return WinnerDecision.CONTROL

        return WinnerDecision.TIE

    def analyze_experiment(self, experiment: Experiment) -> list[StatisticalResult]:
        """Analyze all metrics for an experiment.

        Args:
            experiment: The experiment to analyze.

        Returns:
            List of statistical results for each metric.
        """
        results: list[StatisticalResult] = []
        for metric_config in experiment.config.metrics:
            result = self.compute_statistical_result(experiment, metric_config)
            results.append(result)
        return results

    def check_for_degradation(
        self,
        experiment: Experiment,
    ) -> tuple[bool, str | None]:
        """Check if treatment variant shows degradation.

        Checks if the treatment variant is performing significantly worse
        than control on the primary metric.

        Args:
            experiment: The experiment to check.

        Returns:
            Tuple of (is_degraded, reason).
        """
        rollback = experiment.config.rollback

        # Check if we have enough samples
        min_samples = rollback.min_samples_before_rollback
        if (
            experiment.control_sample_size < min_samples
            or experiment.treatment_sample_size < min_samples
        ):
            return False, None

        # Find primary metric
        primary_config = None
        for mc in experiment.config.metrics:
            if mc.is_primary:
                primary_config = mc
                break

        if not primary_config:
            return False, None

        # Get statistical result for primary metric
        result = self.compute_statistical_result(experiment, primary_config)

        # Check for degradation
        threshold = rollback.degradation_threshold

        # For minimize metrics, positive relative change is bad
        # For maximize metrics, negative relative change is bad
        if primary_config.minimize:
            is_degraded = result.relative_change > (threshold * 100)
        else:
            is_degraded = result.relative_change < -(threshold * 100)

        if is_degraded:
            direction = "increase" if primary_config.minimize else "decrease"
            threshold_pct = threshold * 100
            reason = (
                f"Treatment shows {abs(result.relative_change):.1f}% "
                f"{direction} in {primary_config.metric_type.value} "
                f"(threshold: {threshold_pct:.0f}%)"
            )
            return True, reason

        return False, None


# ==================== Experiment Manager ====================


class ExperimentManager:
    """Manages A/B experiments lifecycle and operations."""

    def __init__(self, analyzer: ExperimentAnalyzer | None = None) -> None:
        """Initialize experiment manager.

        Args:
            analyzer: Experiment analyzer to use.
        """
        self.analyzer = analyzer or ExperimentAnalyzer()
        self._experiments: dict[int, Experiment] = {}
        self._next_id = 1

    def create_experiment(self, config: ExperimentConfig) -> Experiment:
        """Create a new experiment.

        Args:
            config: Experiment configuration.

        Returns:
            The created experiment.
        """
        experiment = Experiment(
            id=self._next_id,
            config=config,
            status=ExperimentStatus.DRAFT,
        )
        self._experiments[self._next_id] = experiment
        self._next_id += 1
        return experiment

    def get_experiment(self, experiment_id: int) -> Experiment | None:
        """Get an experiment by ID.

        Args:
            experiment_id: Experiment ID.

        Returns:
            The experiment or None if not found.
        """
        return self._experiments.get(experiment_id)

    def list_experiments(
        self,
        status: ExperimentStatus | None = None,
        suite_name: str | None = None,
    ) -> list[Experiment]:
        """List experiments with optional filters.

        Args:
            status: Filter by status.
            suite_name: Filter by suite name.

        Returns:
            List of matching experiments.
        """
        results = list(self._experiments.values())

        if status:
            results = [e for e in results if e.status == status]

        if suite_name:
            results = [e for e in results if e.config.suite_name == suite_name]

        return sorted(results, key=lambda e: e.created_at, reverse=True)

    def start_experiment(self, experiment_id: int) -> Experiment:
        """Start an experiment (transition from draft to running).

        Args:
            experiment_id: Experiment ID.

        Returns:
            The updated experiment.

        Raises:
            ValueError: If experiment not found or cannot be started.
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(
                f"Cannot start experiment in status {experiment.status.value}"
            )

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = _utcnow()
        return experiment

    def pause_experiment(self, experiment_id: int) -> Experiment:
        """Pause a running experiment.

        Args:
            experiment_id: Experiment ID.

        Returns:
            The updated experiment.

        Raises:
            ValueError: If experiment not found or cannot be paused.
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(
                f"Cannot pause experiment in status {experiment.status.value}"
            )

        experiment.status = ExperimentStatus.PAUSED
        experiment.paused_at = _utcnow()
        return experiment

    def resume_experiment(self, experiment_id: int) -> Experiment:
        """Resume a paused experiment.

        Args:
            experiment_id: Experiment ID.

        Returns:
            The updated experiment.

        Raises:
            ValueError: If experiment not found or cannot be resumed.
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        if experiment.status != ExperimentStatus.PAUSED:
            raise ValueError(
                f"Cannot resume experiment in status {experiment.status.value}"
            )

        experiment.status = ExperimentStatus.RUNNING
        experiment.paused_at = None
        return experiment

    def conclude_experiment(
        self,
        experiment_id: int,
        reason: str = "Manual conclusion",
    ) -> Experiment:
        """Conclude an experiment.

        Args:
            experiment_id: Experiment ID.
            reason: Reason for conclusion.

        Returns:
            The updated experiment.

        Raises:
            ValueError: If experiment not found or cannot be concluded.
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        if experiment.status not in (ExperimentStatus.RUNNING, ExperimentStatus.PAUSED):
            raise ValueError(
                f"Cannot conclude experiment in status {experiment.status.value}"
            )

        # Run final analysis
        results = self.analyzer.analyze_experiment(experiment)
        experiment.results = results

        # Determine overall winner from primary metric
        for result in results:
            for mc in experiment.config.metrics:
                if mc.is_primary and mc.metric_type == result.metric_type:
                    experiment.winner = result.winner
                    break

        experiment.status = ExperimentStatus.CONCLUDED
        experiment.concluded_at = _utcnow()
        experiment.conclusion_reason = reason
        return experiment

    def record_observation(
        self,
        experiment_id: int,
        observation: ExperimentObservation,
    ) -> Experiment:
        """Record an observation for an experiment.

        Args:
            experiment_id: Experiment ID.
            observation: The observation to record.

        Returns:
            The updated experiment.

        Raises:
            ValueError: If experiment not found or not running.
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(
                f"Cannot record observation for experiment in status "
                f"{experiment.status.value}"
            )

        # Add observation to appropriate variant
        if observation.variant_name == experiment.config.control_variant.name:
            experiment.control_observations.append(observation)
        elif observation.variant_name == experiment.config.treatment_variant.name:
            experiment.treatment_observations.append(observation)
        else:
            raise ValueError(f"Unknown variant name: {observation.variant_name}")

        # Check for degradation if rollback is enabled
        if experiment.config.rollback.enabled:
            is_degraded, reason = self.analyzer.check_for_degradation(experiment)
            if is_degraded:
                experiment.consecutive_degradation_checks += 1
                if (
                    experiment.consecutive_degradation_checks
                    >= experiment.config.rollback.consecutive_checks
                ):
                    self._trigger_rollback(experiment, reason or "Degradation detected")
            else:
                experiment.consecutive_degradation_checks = 0

        # Check for auto-conclusion
        if experiment.should_auto_conclude:
            self.conclude_experiment(experiment_id, "Auto-concluded: limits reached")

        return experiment

    def _trigger_rollback(self, experiment: Experiment, reason: str) -> None:
        """Trigger rollback for an experiment.

        Args:
            experiment: The experiment to roll back.
            reason: Reason for rollback.
        """
        experiment.status = ExperimentStatus.ROLLED_BACK
        experiment.rollback_triggered = True
        experiment.concluded_at = _utcnow()
        experiment.conclusion_reason = f"Rollback: {reason}"

        # Run final analysis
        results = self.analyzer.analyze_experiment(experiment)
        experiment.results = results
        experiment.winner = WinnerDecision.CONTROL

    def get_traffic_router(self, experiment_id: int) -> TrafficRouter:
        """Get a traffic router for an experiment.

        Args:
            experiment_id: Experiment ID.

        Returns:
            Traffic router for the experiment.

        Raises:
            ValueError: If experiment not found.
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        return TrafficRouter(experiment)

    def generate_report(self, experiment_id: int) -> ExperimentReport:
        """Generate a comprehensive report for an experiment.

        Args:
            experiment_id: Experiment ID.

        Returns:
            Experiment report.

        Raises:
            ValueError: If experiment not found.
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Run analysis
        results = self.analyzer.analyze_experiment(experiment)

        # Generate summary
        summary: dict[str, Any] = {
            "experiment_name": experiment.config.name,
            "suite_name": experiment.config.suite_name,
            "status": experiment.status.value,
            "control_variant": experiment.config.control_variant.name,
            "treatment_variant": experiment.config.treatment_variant.name,
            "control_samples": experiment.control_sample_size,
            "treatment_samples": experiment.treatment_sample_size,
            "total_samples": experiment.total_sample_size,
            "winner": experiment.winner.value,
            "rollback_triggered": experiment.rollback_triggered,
        }

        if experiment.started_at:
            summary["started_at"] = experiment.started_at.isoformat()
        if experiment.concluded_at:
            summary["concluded_at"] = experiment.concluded_at.isoformat()
            start = experiment.started_at or experiment.created_at
            duration = experiment.concluded_at - start
            summary["duration_days"] = duration.days

        # Generate recommendation
        recommendation = self._generate_recommendation(experiment, results)

        return ExperimentReport(
            experiment=experiment,
            statistical_results=results,
            recommendation=recommendation,
            summary=summary,
        )

    def _generate_recommendation(
        self,
        experiment: Experiment,
        results: list[StatisticalResult],
    ) -> str:
        """Generate a recommendation based on results.

        Args:
            experiment: The experiment.
            results: Statistical results.

        Returns:
            Recommendation text.
        """
        if experiment.rollback_triggered:
            return (
                f"The treatment variant ({experiment.config.treatment_variant.name}) "
                f"showed significant degradation and was rolled back. "
                f"Recommend keeping the control variant "
                f"({experiment.config.control_variant.name})."
            )

        if experiment.winner == WinnerDecision.TREATMENT:
            primary_result = None
            for r in results:
                for mc in experiment.config.metrics:
                    if mc.is_primary and mc.metric_type == r.metric_type:
                        primary_result = r
                        break

            change_desc = ""
            if primary_result:
                direction = (
                    "improvement" if primary_result.relative_change > 0 else "reduction"
                )
                change_desc = (
                    f" with {abs(primary_result.relative_change):.1f}% "
                    f"{direction} in {primary_result.metric_type.value}"
                )

            return (
                f"The treatment variant ({experiment.config.treatment_variant.name}) "
                f"is the clear winner{change_desc}. "
                f"Recommend deploying the treatment variant."
            )

        elif experiment.winner == WinnerDecision.CONTROL:
            return (
                f"The control variant ({experiment.config.control_variant.name}) "
                f"performed better than treatment. "
                f"Recommend keeping the control variant."
            )

        elif experiment.winner == WinnerDecision.TIE:
            return (
                "Both variants performed similarly with no practical difference. "
                "Consider other factors (cost, complexity) for decision."
            )

        else:  # INCONCLUSIVE
            if not experiment.can_conclude:
                return (
                    f"Not enough data to determine a winner. "
                    f"Need at least {experiment.config.min_sample_size} samples "
                    f"per variant (current: control={experiment.control_sample_size}, "
                    f"treatment={experiment.treatment_sample_size})."
                )
            return (
                "Results are not statistically significant. "
                "Consider running the experiment longer or with more traffic."
            )


# ==================== Module-level functions ====================


_default_manager: ExperimentManager | None = None


def get_experiment_manager() -> ExperimentManager:
    """Get the default experiment manager singleton.

    Returns:
        The default experiment manager.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = ExperimentManager()
    return _default_manager


def reset_experiment_manager() -> None:
    """Reset the default experiment manager (for testing)."""
    global _default_manager
    _default_manager = None
