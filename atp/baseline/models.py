"""Data models for baseline storage and comparison."""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


class ChangeType(str, Enum):
    """Type of change detected when comparing to baseline."""

    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    NO_CHANGE = "no_change"
    NEW_TEST = "new_test"
    REMOVED_TEST = "removed_test"


class TestBaseline(BaseModel):
    """Baseline data for a single test.

    Stores statistical summary from test runs to use as a reference
    for regression detection.
    """

    test_id: str = Field(..., description="Test identifier")
    test_name: str = Field(..., description="Human-readable test name")
    mean_score: float = Field(
        ..., description="Mean score across runs", ge=0.0, le=100.0
    )
    std: float = Field(..., description="Standard deviation of scores", ge=0.0)
    n_runs: int = Field(
        ..., description="Number of runs used to compute statistics", ge=1
    )
    ci_95: tuple[float, float] = Field(
        ..., description="95% confidence interval (lower, upper)"
    )
    success_rate: float = Field(
        ..., description="Success rate across runs", ge=0.0, le=1.0
    )
    mean_duration: float | None = Field(
        None, description="Mean duration in seconds", ge=0.0
    )
    mean_tokens: float | None = Field(None, description="Mean token usage", ge=0.0)
    mean_cost: float | None = Field(None, description="Mean cost in USD", ge=0.0)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "mean_score": round(self.mean_score, 4),
            "std": round(self.std, 4),
            "n_runs": self.n_runs,
            "ci_95": [round(self.ci_95[0], 4), round(self.ci_95[1], 4)],
            "success_rate": round(self.success_rate, 4),
        }
        if self.mean_duration is not None:
            result["mean_duration"] = round(self.mean_duration, 4)
        if self.mean_tokens is not None:
            result["mean_tokens"] = round(self.mean_tokens, 4)
        if self.mean_cost is not None:
            result["mean_cost"] = round(self.mean_cost, 6)
        return result


class Baseline(BaseModel):
    """Complete baseline data for a test suite.

    Stores baseline statistics for all tests in a suite, along with
    metadata about when and how the baseline was created.
    """

    version: str = Field(default="1.0", description="Baseline format version")
    created_at: datetime = Field(
        default_factory=_utcnow, description="When baseline was created"
    )
    suite_name: str = Field(..., description="Test suite name")
    agent_name: str = Field(..., description="Agent name used for baseline")
    runs_per_test: int = Field(..., description="Number of runs per test", ge=1)
    tests: dict[str, TestBaseline] = Field(
        default_factory=dict, description="Baseline data keyed by test_id"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "suite_name": self.suite_name,
            "agent_name": self.agent_name,
            "runs_per_test": self.runs_per_test,
            "tests": {
                test_id: baseline.to_dict() for test_id, baseline in self.tests.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Baseline":
        """Create Baseline from dictionary.

        Args:
            data: Dictionary with baseline data.

        Returns:
            Baseline instance.
        """
        tests = {}
        for test_id, test_data in data.get("tests", {}).items():
            # Handle ci_95 which might be a list
            ci_95 = test_data.get("ci_95", [0.0, 0.0])
            if isinstance(ci_95, list):
                ci_95 = tuple(ci_95)

            tests[test_id] = TestBaseline(
                test_id=test_data.get("test_id", test_id),
                test_name=test_data.get("test_name", test_id),
                mean_score=test_data["mean_score"],
                std=test_data["std"],
                n_runs=test_data["n_runs"],
                ci_95=ci_95,
                success_rate=test_data.get("success_rate", 1.0),
                mean_duration=test_data.get("mean_duration"),
                mean_tokens=test_data.get("mean_tokens"),
                mean_cost=test_data.get("mean_cost"),
            )

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        return cls(
            version=data.get("version", "1.0"),
            created_at=created_at or _utcnow(),
            suite_name=data["suite_name"],
            agent_name=data["agent_name"],
            runs_per_test=data.get("runs_per_test", 1),
            tests=tests,
        )
