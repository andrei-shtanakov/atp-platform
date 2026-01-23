"""Base evaluator interface and result models."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse


class EvalCheck(BaseModel):
    """Single evaluation check result."""

    name: str = Field(..., description="Check name", min_length=1)
    passed: bool = Field(..., description="Whether the check passed")
    score: float = Field(..., description="Score from 0.0 to 1.0", ge=0.0, le=1.0)
    message: str | None = Field(None, description="Human-readable message")
    details: dict[str, Any] | None = Field(None, description="Additional check details")


class EvalResult(BaseModel):
    """Result of an evaluator run containing multiple checks."""

    evaluator: str = Field(..., description="Evaluator name that produced this result")
    checks: list[EvalCheck] = Field(
        default_factory=list, description="List of check results"
    )

    @property
    def passed(self) -> bool:
        """Check if all checks passed."""
        return all(c.passed for c in self.checks)

    @property
    def score(self) -> float:
        """Calculate average score across all checks."""
        if not self.checks:
            return 1.0
        return sum(c.score for c in self.checks) / len(self.checks)

    @property
    def total_checks(self) -> int:
        """Return total number of checks."""
        return len(self.checks)

    @property
    def passed_checks(self) -> int:
        """Return number of passed checks."""
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_checks(self) -> int:
        """Return number of failed checks."""
        return sum(1 for c in self.checks if not c.passed)

    def add_check(self, check: EvalCheck) -> None:
        """Add a check to the result."""
        self.checks.append(check)

    def merge(self, other: "EvalResult") -> "EvalResult":
        """Merge another EvalResult into this one.

        Creates a new EvalResult containing checks from both.
        The evaluator name is preserved from the first result.

        Args:
            other: Another EvalResult to merge.

        Returns:
            New EvalResult with combined checks.
        """
        return EvalResult(
            evaluator=self.evaluator,
            checks=self.checks + other.checks,
        )

    @classmethod
    def aggregate(cls, results: list["EvalResult"]) -> "EvalResult":
        """Aggregate multiple EvalResults into one.

        Args:
            results: List of EvalResults to aggregate.

        Returns:
            Single EvalResult with all checks combined.
        """
        if not results:
            return cls(evaluator="aggregate", checks=[])

        all_checks: list[EvalCheck] = []
        for result in results:
            all_checks.extend(result.checks)

        return cls(evaluator="aggregate", checks=all_checks)


class Evaluator(ABC):
    """
    Base class for evaluators.

    Evaluators assess agent results against assertions defined in tests.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the evaluator name."""

    @abstractmethod
    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """
        Evaluate agent results against an assertion.

        Args:
            task: Test definition containing task details.
            response: ATP Response from the agent.
            trace: List of ATP Events from execution.
            assertion: Assertion to evaluate against.

        Returns:
            EvalResult containing check results.
        """

    def _create_check(
        self,
        name: str,
        passed: bool,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> EvalCheck:
        """Helper to create an EvalCheck with score based on passed status."""
        return EvalCheck(
            name=name,
            passed=passed,
            score=1.0 if passed else 0.0,
            message=message,
            details=details,
        )

    def _create_result(self, checks: list[EvalCheck]) -> EvalResult:
        """Helper to create an EvalResult with this evaluator's name."""
        return EvalResult(evaluator=self.name, checks=checks)
