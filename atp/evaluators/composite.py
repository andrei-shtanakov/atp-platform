"""Composite evaluator for combining evaluators with boolean logic."""

from typing import Any

from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse

from .base import EvalCheck, EvalResult, Evaluator


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

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "composite"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """
        Evaluate composite assertions using boolean logic.

        Args:
            task: Test definition containing task details.
            response: ATP Response from the agent.
            trace: List of ATP Events from execution.
            assertion: Assertion with composite config.

        Returns:
            EvalResult with combined check outcomes.
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

        passed, score, sub_checks = await self._evaluate_operator(
            operator, conditions, task, response, trace
        )

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
        if passed:
            summary_check.score = score

        return self._create_result([summary_check])

    async def _evaluate_operator(
        self,
        operator: str,
        conditions: list[dict[str, Any]],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> tuple[bool, float, list[EvalCheck]]:
        """Evaluate conditions with a boolean operator.

        Returns:
            Tuple of (passed, score, sub_checks).
        """
        if operator == "and":
            return await self._evaluate_and(conditions, task, response, trace)
        elif operator == "or":
            return await self._evaluate_or(conditions, task, response, trace)
        elif operator == "not":
            return await self._evaluate_not(conditions, task, response, trace)
        elif operator == "threshold":
            return await self._evaluate_threshold(conditions, task, response, trace)
        else:
            check = self._create_check(
                name="composite_error",
                passed=False,
                message=f"Unknown operator: {operator}",
            )
            return False, 0.0, [check]

    async def _evaluate_and(
        self,
        conditions: list[dict[str, Any]],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> tuple[bool, float, list[EvalCheck]]:
        """Evaluate AND: all conditions must pass."""
        all_checks: list[EvalCheck] = []
        all_passed = True
        total_score = 0.0

        for condition in conditions:
            passed, score, checks = await self._evaluate_condition(
                condition, task, response, trace
            )
            all_checks.extend(checks)
            if not passed:
                all_passed = False
            total_score += score

        avg_score = total_score / len(conditions) if conditions else 1.0
        return all_passed, avg_score, all_checks

    async def _evaluate_or(
        self,
        conditions: list[dict[str, Any]],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> tuple[bool, float, list[EvalCheck]]:
        """Evaluate OR: at least one condition must pass."""
        all_checks: list[EvalCheck] = []
        any_passed = False
        max_score = 0.0

        for condition in conditions:
            passed, score, checks = await self._evaluate_condition(
                condition, task, response, trace
            )
            all_checks.extend(checks)
            if passed:
                any_passed = True
            max_score = max(max_score, score)

        return any_passed, max_score, all_checks

    async def _evaluate_not(
        self,
        conditions: list[dict[str, Any]],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> tuple[bool, float, list[EvalCheck]]:
        """Evaluate NOT: invert the first condition's result."""
        if not conditions:
            return True, 1.0, []

        condition = conditions[0]
        passed, score, checks = await self._evaluate_condition(
            condition, task, response, trace
        )
        inverted = not passed
        inverted_score = 1.0 - score
        return inverted, inverted_score, checks

    async def _evaluate_threshold(
        self,
        conditions: list[dict[str, Any]],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> tuple[bool, float, list[EvalCheck]]:
        """Evaluate threshold: check if score meets threshold.

        Expects conditions to be a single-element list with
        threshold config embedded in the first element.
        This method is called from _evaluate_condition when
        operator=threshold is used.
        """
        if not conditions:
            return True, 1.0, []

        # The threshold config is passed through conditions[0]
        threshold_config = conditions[0]
        value = threshold_config.get("value", 0.0)
        comparator = threshold_config.get("comparator", ">=")
        inner = threshold_config.get("condition", {})

        if not inner:
            check = self._create_check(
                name="threshold_error",
                passed=False,
                message="No inner condition for threshold",
            )
            return False, 0.0, [check]

        passed, score, checks = await self._evaluate_condition(
            inner, task, response, trace
        )

        threshold_passed = _compare(score, comparator, value)
        return threshold_passed, score, checks

    async def _evaluate_condition(
        self,
        condition: dict[str, Any],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> tuple[bool, float, list[EvalCheck]]:
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

        # Leaf assertion — delegate to the appropriate evaluator
        return await self._evaluate_leaf(condition, task, response, trace)

    async def _evaluate_leaf(
        self,
        condition: dict[str, Any],
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> tuple[bool, float, list[EvalCheck]]:
        """Evaluate a leaf assertion via the evaluator registry."""
        from .registry import get_registry

        assertion_type = condition.get("type", "")
        assertion_config = condition.get("config", {})

        if not assertion_type:
            check = self._create_check(
                name="composite_error",
                passed=False,
                message="Condition missing 'type' field",
            )
            return False, 0.0, [check]

        registry = get_registry()

        if not registry.supports_assertion(assertion_type):
            check = self._create_check(
                name="composite_error",
                passed=False,
                message=(f"Unknown assertion type: {assertion_type}"),
            )
            return False, 0.0, [check]

        evaluator = registry.create_for_assertion(assertion_type)
        sub_assertion = Assertion(type=assertion_type, config=assertion_config)
        result = await evaluator.evaluate(task, response, trace, sub_assertion)

        return result.passed, result.score, list(result.checks)


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
