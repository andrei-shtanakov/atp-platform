"""Pre-evaluation guardrails inspired by arbiter's invariant rules.

Run checks before each evaluator to skip wasteful evaluations early
(e.g. don't call LLM-judge on an empty response).
"""

import logging
from dataclasses import dataclass

from atp.loader.models import TestDefinition
from atp.protocol import ATPResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of a single guardrail check."""

    name: str
    passed: bool
    reason: str


def check_response_not_empty(response: ATPResponse) -> CheckResult:
    """Skip evaluation if the agent returned no output."""
    has_artifacts = bool(response.artifacts)
    has_output = bool(getattr(response, "output", None))

    if response.status.value == "failed" and not has_artifacts and not has_output:
        return CheckResult(
            name="response_not_empty",
            passed=False,
            reason="Agent response is empty/failed with no artifacts",
        )
    return CheckResult(name="response_not_empty", passed=True, reason="")


def check_timeout_not_exceeded(
    test: TestDefinition, response: ATPResponse
) -> CheckResult:
    """Skip evaluation if the agent timed out."""
    if response.status.value == "timeout":
        return CheckResult(
            name="timeout_not_exceeded",
            passed=False,
            reason=(f"Agent timed out (limit: {test.constraints.timeout_seconds}s)"),
        )
    return CheckResult(name="timeout_not_exceeded", passed=True, reason="")


def check_within_budget(test: TestDefinition, response: ATPResponse) -> CheckResult:
    """Skip evaluation if the agent exceeded its budget."""
    budget = test.constraints.budget_usd
    if budget is None:
        return CheckResult(name="within_budget", passed=True, reason="")

    cost = 0.0
    if response.metrics and response.metrics.cost_usd is not None:
        cost = response.metrics.cost_usd

    if cost > budget:
        return CheckResult(
            name="within_budget",
            passed=False,
            reason=f"Agent exceeded budget: ${cost:.4f} > ${budget:.4f}",
        )
    return CheckResult(name="within_budget", passed=True, reason="")


def run_guardrails(
    test: TestDefinition,
    response: ATPResponse,
) -> list[CheckResult]:
    """Run all pre-evaluation guardrail checks.

    Args:
        test: The test definition with constraints.
        response: The agent's response.

    Returns:
        List of check results. If any check fails, evaluators
        should be skipped for this test.
    """
    return [
        check_response_not_empty(response),
        check_timeout_not_exceeded(test, response),
        check_within_budget(test, response),
    ]


def should_skip_evaluation(checks: list[CheckResult]) -> str | None:
    """Return skip reason if any guardrail failed, else None."""
    for check in checks:
        if not check.passed:
            logger.info(
                "Guardrail '%s' failed: %s — skipping evaluation",
                check.name,
                check.reason,
            )
            return check.reason
    return None
