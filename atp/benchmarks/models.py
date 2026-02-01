"""Data models for benchmark definitions and results."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BenchmarkCategory(str, Enum):
    """Categories of benchmarks."""

    CODING = "coding"
    RESEARCH = "research"
    REASONING = "reasoning"
    DATA_PROCESSING = "data_processing"


class BenchmarkDifficulty(str, Enum):
    """Difficulty levels for benchmarks."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BaselineScore(BaseModel):
    """Baseline score for benchmark comparison."""

    model_name: str = Field(..., description="Name of the model/agent")
    score: float = Field(..., ge=0.0, le=100.0, description="Normalized score (0-100)")
    date: str = Field(..., description="Date baseline was established (YYYY-MM-DD)")
    notes: str | None = Field(None, description="Optional notes about the baseline")


class BenchmarkMetadata(BaseModel):
    """Metadata for a benchmark test."""

    category: BenchmarkCategory = Field(..., description="Benchmark category")
    difficulty: BenchmarkDifficulty = Field(
        BenchmarkDifficulty.MEDIUM, description="Difficulty level"
    )
    estimated_time_seconds: int = Field(
        60, ge=1, description="Estimated time to complete"
    )
    skills_tested: list[str] = Field(
        default_factory=list, description="Skills being tested"
    )
    baseline_scores: list[BaselineScore] = Field(
        default_factory=list, description="Baseline scores for comparison"
    )


class BenchmarkTest(BaseModel):
    """A single benchmark test definition."""

    id: str = Field(..., description="Unique test identifier")
    name: str = Field(..., description="Human-readable test name")
    description: str = Field(..., description="Detailed test description")
    task_description: str = Field(..., description="Task description for the agent")
    input_data: dict[str, Any] | None = Field(None, description="Optional input data")
    expected_artifacts: list[str] = Field(
        default_factory=list, description="Expected output artifacts"
    )
    assertions: list[dict[str, Any]] = Field(
        default_factory=list, description="Test assertions"
    )
    metadata: BenchmarkMetadata = Field(..., description="Benchmark metadata")
    max_steps: int | None = Field(None, description="Maximum steps allowed")
    timeout_seconds: int = Field(300, ge=1, description="Timeout in seconds")
    tags: list[str] = Field(default_factory=list, description="Test tags")


class BenchmarkSuiteInfo(BaseModel):
    """Information about a benchmark suite."""

    name: str = Field(..., description="Suite name")
    category: BenchmarkCategory = Field(..., description="Benchmark category")
    description: str = Field(..., description="Suite description")
    version: str = Field("1.0.0", description="Suite version")
    test_count: int = Field(..., ge=1, description="Number of tests in the suite")
    difficulty_distribution: dict[str, int] = Field(
        default_factory=dict, description="Distribution of test difficulties"
    )
    average_baseline_score: float | None = Field(
        None, ge=0.0, le=100.0, description="Average baseline score across tests"
    )


class BenchmarkSuite(BaseModel):
    """Complete benchmark suite definition."""

    name: str = Field(..., description="Suite name")
    category: BenchmarkCategory = Field(..., description="Benchmark category")
    version: str = Field("1.0.0", description="Suite version")
    description: str = Field(..., description="Suite description")
    tests: list[BenchmarkTest] = Field(..., description="List of tests", min_length=1)
    default_timeout_seconds: int = Field(300, ge=1, description="Default timeout")
    default_max_steps: int | None = Field(None, description="Default max steps")

    def get_info(self) -> BenchmarkSuiteInfo:
        """Get summary information about this suite."""
        difficulty_dist: dict[str, int] = {}
        total_baseline = 0.0
        baseline_count = 0

        for test in self.tests:
            diff = test.metadata.difficulty.value
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1

            for baseline in test.metadata.baseline_scores:
                total_baseline += baseline.score
                baseline_count += 1

        avg_baseline = total_baseline / baseline_count if baseline_count > 0 else None

        return BenchmarkSuiteInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            version=self.version,
            test_count=len(self.tests),
            difficulty_distribution=difficulty_dist,
            average_baseline_score=avg_baseline,
        )


class BenchmarkResult(BaseModel):
    """Result of running a single benchmark test."""

    test_id: str = Field(..., description="Test ID that was run")
    raw_score: float = Field(..., ge=0.0, le=1.0, description="Raw score (0-1)")
    normalized_score: float = Field(
        ..., ge=0.0, le=100.0, description="Normalized score (0-100)"
    )
    passed: bool = Field(..., description="Whether the test passed")
    execution_time_seconds: float = Field(
        ..., ge=0.0, description="Time taken to execute"
    )
    steps_used: int | None = Field(None, description="Number of steps used")
    tokens_used: int | None = Field(None, description="Number of tokens used")
    cost_usd: float | None = Field(None, description="Cost in USD")
    error: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional result metadata"
    )


class BenchmarkSuiteResult(BaseModel):
    """Result of running a complete benchmark suite."""

    suite_name: str = Field(..., description="Suite name that was run")
    category: BenchmarkCategory = Field(..., description="Benchmark category")
    agent_name: str = Field(..., description="Name of the agent tested")
    total_tests: int = Field(..., ge=0, description="Total number of tests")
    passed_tests: int = Field(..., ge=0, description="Number of tests passed")
    failed_tests: int = Field(..., ge=0, description="Number of tests failed")
    average_normalized_score: float = Field(
        ..., ge=0.0, le=100.0, description="Average normalized score"
    )
    total_execution_time_seconds: float = Field(
        ..., ge=0.0, description="Total execution time"
    )
    results: list[BenchmarkResult] = Field(
        default_factory=list, description="Individual test results"
    )
    baseline_comparison: dict[str, float] | None = Field(
        None, description="Comparison to baseline scores (delta)"
    )

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate as a percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100.0


class NormalizationConfig(BaseModel):
    """Configuration for score normalization."""

    min_raw_score: float = Field(0.0, description="Minimum expected raw score")
    max_raw_score: float = Field(1.0, description="Maximum expected raw score")
    target_min: float = Field(0.0, description="Target minimum (usually 0)")
    target_max: float = Field(100.0, description="Target maximum (usually 100)")
    curve_type: str = Field(
        "linear", description="Normalization curve type (linear, logarithmic, sigmoid)"
    )

    def normalize(self, raw_score: float) -> float:
        """Normalize a raw score to the target range.

        Args:
            raw_score: The raw score to normalize.

        Returns:
            Normalized score in the target range.
        """
        raw_score = max(self.min_raw_score, min(self.max_raw_score, raw_score))

        raw_range = self.max_raw_score - self.min_raw_score
        if raw_range == 0:
            return self.target_min

        normalized = (raw_score - self.min_raw_score) / raw_range

        if self.curve_type == "logarithmic":
            import math

            normalized = math.log1p(normalized * 9) / math.log(10)
        elif self.curve_type == "sigmoid":
            import math

            x = (normalized - 0.5) * 10
            normalized = 1 / (1 + math.exp(-x))

        target_range = self.target_max - self.target_min
        return self.target_min + (normalized * target_range)
