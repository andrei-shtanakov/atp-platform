"""Post-execution, pre-evaluation guardrails.

Runs after an agent produces a response but before ATP dispatches any
evaluator, so that empty, timed-out, or over-budget responses can
short-circuit the expensive evaluator pipeline (LLM-judge, container
execution, etc.).

Relationship to arbiter
-----------------------
The short-circuit *pattern* — list of independent predicates, each
returning a (name, passed, reason) tuple — is borrowed from
arbiter-core's invariant rules. The *rule set* is NOT borrowed:

- arbiter's invariants gate agent *assignment* (pre-dispatch, operating
  on TaskInput + AgentContext + SystemContext).
- These guardrails gate *evaluation* (post-dispatch, operating on an
  ATPResponse after the agent has already run).

Of the three rules here, two (``within_budget``, ``timeout_not_exceeded``)
share a concept — budget and time — with arbiter counterparts
(``budget_remaining``, ``sla_feasible``), but use inverted predicates
(measurement vs. estimate). ``not_silently_failed`` has no arbiter
analogue, and eight arbiter invariants have no analogue here.

Semantic mapping: ``arbiter/docs/guardrails-atp-mapping.md`` in the
sibling repo.
"""

import logging
from dataclasses import dataclass

from atp.loader.models import TestDefinition
from atp.protocol import ATPResponse, ResponseStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of a single guardrail check."""

    name: str
    passed: bool
    reason: str


def check_not_silently_failed(response: ATPResponse) -> CheckResult:
    """Skip evaluation only when the agent both failed AND produced no artifacts.

    A ``status=failed`` response with no artifacts carries nothing an
    evaluator can score — running LLM-judge on it is pure waste. A
    ``status=failed`` response *with* artifacts is still evaluated: the
    agent may have partially succeeded (e.g. produced a file before
    dying) and some evaluators grade exactly that.

    A ``status=completed`` response with empty artifacts is NOT caught
    here — that's a valid shape (some tests score events / metrics
    rather than outputs). Behavior and event evaluators pick those up
    downstream.
    """
    if response.status == ResponseStatus.FAILED and not response.artifacts:
        return CheckResult(
            name="not_silently_failed",
            passed=False,
            reason="Agent response is empty/failed with no artifacts",
        )
    return CheckResult(name="not_silently_failed", passed=True, reason="")


def check_timeout_not_exceeded(
    test: TestDefinition, response: ATPResponse
) -> CheckResult:
    """Skip evaluation if the agent timed out."""
    if response.status == ResponseStatus.TIMEOUT:
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
        check_not_silently_failed(response),
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
