"""Composite evaluator: boolean logic over other evaluators.

Two properties matter here beyond the boolean algebra, and they are the reason
this module was withheld from the benchmark plane until now (ADR-008 track A).

**Children are built through an injected resolver, never a global registry.**
An evaluator that calls `get_registry()` evaluates its children under no
policy at all, so a `composite` wrapping `pytest` walked straight past the
server allowlist. The resolver arrives from whoever resolved *this* evaluator
and is passed down to whoever this evaluator resolves, so a leaf three levels
deep is restricted exactly as the root was. With no resolver bound there is no
fallback: the leaf is unevaluated. Fail-closed is the point — a fallback is
precisely the hole being closed.

**A leaf that was not evaluated is not a leaf that failed.** A refused,
unknown or broken leaf yields `UNEVALUATED`, and the operators propagate it by
Kleene logic. Collapsing it to `False`/`0.0` would publish "assessed and bad"
for something nobody assessed, which is the defect the benchmark score
contract exists to prevent.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from atp.evaluation import AssertionUnevaluated, EvaluatorResolver, bind_resolver
from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse

from .base import EvalCheck, EvalResult, Evaluator

logger = logging.getLogger(__name__)


class Verdict(StrEnum):
    """Three-valued outcome of a condition."""

    PASS = "pass"
    FAIL = "fail"
    #: Not measured. Never a synonym for "failed".
    UNEVALUATED = "unevaluated"


#: A condition's score as an interval. Degenerate (`lo == hi`) when everything
#: underneath was evaluated; widened to the unknown's full range otherwise, so
#: a threshold can tell "provably above" from "might be either".
ScoreRange = tuple[float, float]

#: What an unevaluated leaf contributes: no information at all.
UNKNOWN_RANGE: ScoreRange = (0.0, 1.0)

#: Internal result of evaluating one condition.
Outcome = tuple[Verdict, ScoreRange, list[EvalCheck]]


class CompositeEvaluator(Evaluator):
    """
    Evaluator that combines sub-assertions with boolean logic.

    Supports AND, OR, NOT operators and threshold conditions
    for building complex pass/fail criteria from simpler evaluators.

    Config format:
        operator: and | or
        conditions:
          - type: artifact_exists
            config: { path: "output.txt" }
          - operator: not
            condition:
              type: security
              config: { checks: ["pii"] }
          - operator: threshold
            value: 0.8
            comparator: ">="
            condition:
              type: llm_eval
              config: { ... }
    """

    def __init__(self, resolver: EvaluatorResolver | None = None) -> None:
        self._resolver = resolver

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "composite"

    def bind_resolver(self, resolver: EvaluatorResolver) -> None:
        """Receive the resolver this evaluator must build its leaves through."""
        self._resolver = resolver

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """
        Evaluate composite assertions using boolean logic.

        Raises:
            AssertionUnevaluated: if the operator cannot be decided from the
                leaves this evaluator was permitted and able to evaluate.
        """
        config = assertion.config
        operator = config.get("operator", "and")
        conditions = config.get("conditions", [])

        if not conditions:
            return self._create_result(
                [
                    self._create_check(
                        name="composite",
                        passed=True,
                        message="No conditions specified (vacuous truth)",
                    )
                ]
            )

        verdict, score_range, sub_checks = await self._evaluate_operator(
            operator, conditions, task, response, trace
        )

        if verdict is Verdict.UNEVALUATED:
            raise AssertionUnevaluated(
                f"composite '{operator}' could not be decided: "
                f"{self._unevaluated_detail(sub_checks)}"
            )

        passed = verdict is Verdict.PASS
        score = _report_score(verdict, score_range)

        summary_check = self._create_check(
            name=f"composite_{operator}",
            passed=passed,
            message=(
                f"Composite {operator.upper()}: {'passed' if passed else 'failed'}"
            ),
            details={
                "operator": operator,
                "score": score,
                "num_conditions": len(conditions),
                # Present whenever the verdict held despite missing leaves, so
                # a reader can see the answer was decided, not complete.
                "score_bounds": list(score_range),
                "sub_checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "score": c.score,
                        "message": c.message,
                    }
                    for c in sub_checks
                ],
            },
        )
        summary_check.score = score

        return self._create_result([summary_check])

    @staticmethod
    def _unevaluated_detail(sub_checks: list[EvalCheck]) -> str:
        """Summarize why, from the leaf checks that recorded a reason."""
        reasons = [c.message for c in sub_checks if c.message and not c.passed]
        return "; ".join(reasons) if reasons else "no leaf produced a verdict"

    async def _evaluate_operator(
        self,
        operator: str,
        conditions: list[dict[str, Any]],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> Outcome:
        """Evaluate conditions with a boolean operator."""
        if operator == "and":
            return await self._evaluate_and(conditions, task, response, trace)
        if operator == "or":
            return await self._evaluate_or(conditions, task, response, trace)
        if operator == "not":
            return await self._evaluate_not(conditions, task, response, trace)
        if operator == "threshold":
            return await self._evaluate_threshold(conditions, task, response, trace)

        # An operator nobody implements is a malformed suite, not a failed
        # agent. Reporting it as a failure would score the author's typo
        # against the agent.
        return self._unevaluated(f"Unknown operator: {operator}")

    async def _evaluate_and(
        self,
        conditions: list[dict[str, Any]],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> Outcome:
        """AND: a real failure decides it; otherwise an unknown blocks it.

        The asymmetry is deliberate. A conjunction containing one genuine
        failure is false whatever the unknowns turn out to be, and refusing to
        say so would be its own kind of dishonesty.

        An empty conjunction is vacuously true. `evaluate` catches the
        top-level case before dispatching, but a *nested* `{operator: "and",
        conditions: []}` arrives here, and the score average has nothing to
        divide by.
        """
        if not conditions:
            return Verdict.PASS, (1.0, 1.0), []

        all_checks: list[EvalCheck] = []
        verdicts: list[Verdict] = []
        los: list[float] = []
        his: list[float] = []

        for condition in conditions:
            verdict, (lo, hi), checks = await self._evaluate_condition(
                condition, task, response, trace
            )
            all_checks.extend(checks)
            verdicts.append(verdict)
            los.append(lo)
            his.append(hi)

        count = len(conditions)
        score_range = (sum(los) / count, sum(his) / count)

        if Verdict.FAIL in verdicts:
            return Verdict.FAIL, score_range, all_checks
        if Verdict.UNEVALUATED in verdicts:
            return Verdict.UNEVALUATED, score_range, all_checks
        return Verdict.PASS, score_range, all_checks

    async def _evaluate_or(
        self,
        conditions: list[dict[str, Any]],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> Outcome:
        """OR: a real pass decides it; otherwise an unknown blocks it.

        An empty disjunction is vacuously false — the classical counterpart of
        the empty conjunction above, and what this returned before the
        three-valued rewrite. It does not divide, so it never crashed; it is
        spelled out here so the pair is decided rather than incidental.
        """
        if not conditions:
            return Verdict.FAIL, (0.0, 0.0), []

        all_checks: list[EvalCheck] = []
        verdicts: list[Verdict] = []
        lo_max = 0.0
        hi_max = 0.0

        for condition in conditions:
            verdict, (lo, hi), checks = await self._evaluate_condition(
                condition, task, response, trace
            )
            all_checks.extend(checks)
            verdicts.append(verdict)
            lo_max = max(lo_max, lo)
            hi_max = max(hi_max, hi)

        score_range = (lo_max, hi_max)

        if Verdict.PASS in verdicts:
            return Verdict.PASS, score_range, all_checks
        if Verdict.UNEVALUATED in verdicts:
            return Verdict.UNEVALUATED, score_range, all_checks
        return Verdict.FAIL, score_range, all_checks

    async def _evaluate_not(
        self,
        conditions: list[dict[str, Any]],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> Outcome:
        """NOT: inverts a verdict, but the negation of unknown is unknown."""
        if not conditions:
            return Verdict.PASS, (1.0, 1.0), []

        verdict, (lo, hi), checks = await self._evaluate_condition(
            conditions[0], task, response, trace
        )
        inverted_range = (1.0 - hi, 1.0 - lo)

        if verdict is Verdict.UNEVALUATED:
            return Verdict.UNEVALUATED, inverted_range, checks
        flipped = Verdict.FAIL if verdict is Verdict.PASS else Verdict.PASS
        return flipped, inverted_range, checks

    async def _evaluate_threshold(
        self,
        conditions: list[dict[str, Any]],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> Outcome:
        """Threshold: decided only when both ends of the interval agree.

        With every leaf evaluated the interval is a point and this is the
        comparison it always was. With an unknown in it, "score >= 0.8" has no
        answer unless the unknown cannot change it.
        """
        if not conditions:
            return Verdict.PASS, (1.0, 1.0), []

        threshold_config = conditions[0]
        value = threshold_config.get("value", 0.0)
        comparator = threshold_config.get("comparator", ">=")
        inner = threshold_config.get("condition", {})

        if not inner:
            return self._unevaluated("No inner condition for threshold")

        verdict, (lo, hi), checks = await self._evaluate_condition(
            inner, task, response, trace
        )
        if verdict is Verdict.UNEVALUATED and lo == hi:
            # Unknown with no usable bounds at all.
            return Verdict.UNEVALUATED, UNKNOWN_RANGE, checks

        at_lo = _compare(lo, comparator, value)
        at_hi = _compare(hi, comparator, value)
        if at_lo != at_hi:
            return Verdict.UNEVALUATED, (lo, hi), checks
        return (Verdict.PASS if at_lo else Verdict.FAIL), (lo, hi), checks

    async def _evaluate_condition(
        self,
        condition: dict[str, Any],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> Outcome:
        """Evaluate a single condition (may be nested or leaf).

        A condition is either:
        - A leaf assertion: { type: "...", config: {...} }
        - A nested operator: { operator: "and|or|not", conditions: [...] }
        - A threshold: { operator: "threshold", value: 0.8,
                         comparator: ">=", condition: {...} }
        """
        operator = condition.get("operator")

        if operator == "not":
            inner = condition.get("condition", {})
            return await self._evaluate_operator(
                "not", [inner] if inner else [], task, response, trace
            )

        if operator == "threshold":
            return await self._evaluate_operator(
                "threshold", [condition], task, response, trace
            )

        if operator in ("and", "or"):
            nested_conditions = condition.get("conditions", [])
            return await self._evaluate_operator(
                operator, nested_conditions, task, response, trace
            )

        return await self._evaluate_leaf(condition, task, response, trace)

    async def _evaluate_leaf(
        self,
        condition: dict[str, Any],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> Outcome:
        """Evaluate a leaf through the injected resolver, or report unknown.

        Every failure mode here is `UNEVALUATED`, never `FAIL`: a leaf the
        policy withheld, a type nobody implements, an evaluator that crashed
        and a malformed condition are all things nobody measured. Scoring them
        as failures would charge the agent for the suite author's typo or for
        the server's own restrictions.
        """
        assertion_type = condition.get("type", "")
        if not assertion_type:
            return self._unevaluated("Condition missing 'type' field")

        if self._resolver is None:
            # No global-registry fallback by design: reaching for one is
            # exactly how this evaluator used to escape its policy.
            return self._unevaluated(
                f"No resolver bound; cannot evaluate leaf '{assertion_type}'"
            )

        try:
            evaluator = self._resolver.create_for_assertion(assertion_type)
        except Exception as exc:
            # Covers both "policy withheld it" and "nobody implements it";
            # the resolver's message says which.
            return self._unevaluated(f"Leaf '{assertion_type}' unavailable: {exc}")

        # Pass the restriction down: a nested composite is bound as we were.
        bind_resolver(evaluator, self._resolver)

        sub_assertion = Assertion(
            type=assertion_type, config=condition.get("config", {})
        )
        try:
            result = await evaluator.evaluate(task, response, trace, sub_assertion)
        except AssertionUnevaluated as exc:
            return self._unevaluated(f"Leaf '{assertion_type}': {exc}")
        except Exception as exc:
            logger.warning(
                "composite leaf '%s' failed: %s", assertion_type, exc, exc_info=True
            )
            return self._unevaluated(f"Leaf '{assertion_type}' errored: {exc}")

        verdict = Verdict.PASS if result.passed else Verdict.FAIL
        return verdict, (result.score, result.score), list(result.checks)

    def _unevaluated(self, message: str) -> Outcome:
        """An unknown, carrying a check that records why."""
        check = self._create_check(
            name="composite_unevaluated", passed=False, message=message
        )
        return Verdict.UNEVALUATED, UNKNOWN_RANGE, [check]


def _report_score(verdict: Verdict, score_range: ScoreRange) -> float:
    """The score to publish for a decided composite.

    When every leaf was evaluated the interval is a point and that point is
    the score, exactly as before. When the verdict held *despite* an unknown,
    the interval is not a measurement — so the published score states the
    verdict (1.0 passed, 0.0 failed) rather than averaging a number that was
    never measured into one that was.
    """
    lo, hi = score_range
    if lo == hi:
        return lo
    return 1.0 if verdict is Verdict.PASS else 0.0


def _compare(score: float, comparator: str, value: float) -> bool:
    """Compare a score against a threshold value."""
    if comparator == ">":
        return score > value
    elif comparator == ">=":
        return score >= value
    elif comparator == "<":
        return score < value
    elif comparator == "<=":
        return score <= value
    elif comparator == "==":
        return abs(score - value) < 1e-9
    elif comparator == "!=":
        return abs(score - value) >= 1e-9
    return score >= value
