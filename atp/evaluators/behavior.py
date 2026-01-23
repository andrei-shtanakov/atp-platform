"""Behavior evaluator for checking agent execution patterns."""

from typing import Any

from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse, EventType

from .base import EvalCheck, EvalResult, Evaluator


class BehaviorEvaluator(Evaluator):
    """
    Evaluator for behavior-related assertions.

    Analyzes agent execution trace to verify behavior constraints.

    Supports the following assertion types:
    - must_use_tools: Check that specific tools were called
    - max_tool_calls: Check that tool call count doesn't exceed limit
    - no_errors: Check that no error events occurred
    - min_tool_calls: Check that at least N tool calls were made
    - forbidden_tools: Check that specific tools were NOT called
    """

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "behavior"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """
        Evaluate behavior assertions.

        Args:
            task: Test definition (may be used for constraint context).
            response: ATP Response from the agent.
            trace: Event trace to analyze.
            assertion: Assertion to evaluate.

        Returns:
            EvalResult with check outcomes.
        """
        assertion_type = assertion.type
        config = assertion.config

        if assertion_type == "behavior":
            checks = self._evaluate_behavior_config(trace, response, config)
        elif assertion_type == "must_use_tools":
            checks = [self._check_must_use_tools(trace, config)]
        elif assertion_type == "max_tool_calls":
            checks = [self._check_max_tool_calls(trace, response, config)]
        elif assertion_type == "min_tool_calls":
            checks = [self._check_min_tool_calls(trace, response, config)]
        elif assertion_type == "no_errors":
            checks = [self._check_no_errors(trace, response)]
        elif assertion_type == "forbidden_tools":
            checks = [self._check_forbidden_tools(trace, config)]
        else:
            checks = [
                self._create_check(
                    name=f"unknown_{assertion_type}",
                    passed=False,
                    message=f"Unknown assertion type: {assertion_type}",
                )
            ]

        return self._create_result(checks)

    def _evaluate_behavior_config(
        self,
        trace: list[ATPEvent],
        response: ATPResponse,
        config: dict[str, Any],
    ) -> list[EvalCheck]:
        """Evaluate all behavior checks from a single config block."""
        checks: list[EvalCheck] = []

        if "must_use_tools" in config:
            checks.append(
                self._check_must_use_tools(trace, {"tools": config["must_use_tools"]})
            )

        if "max_tool_calls" in config:
            checks.append(
                self._check_max_tool_calls(
                    trace, response, {"limit": config["max_tool_calls"]}
                )
            )

        if "min_tool_calls" in config:
            checks.append(
                self._check_min_tool_calls(
                    trace, response, {"limit": config["min_tool_calls"]}
                )
            )

        if "forbidden_tools" in config:
            checks.append(
                self._check_forbidden_tools(trace, {"tools": config["forbidden_tools"]})
            )

        if config.get("no_errors", False):
            checks.append(self._check_no_errors(trace, response))

        if not checks:
            checks.append(
                self._create_check(
                    name="behavior",
                    passed=True,
                    message="No behavior constraints specified",
                )
            )

        return checks

    def _check_must_use_tools(
        self, trace: list[ATPEvent], config: dict[str, Any]
    ) -> EvalCheck:
        """Check that specific tools were called."""
        required_tools = config.get("tools", [])
        if not required_tools:
            return self._create_check(
                name="must_use_tools",
                passed=True,
                message="No required tools specified",
            )

        used_tools = self._extract_used_tools(trace)
        missing_tools = [t for t in required_tools if t not in used_tools]

        if not missing_tools:
            return self._create_check(
                name="must_use_tools",
                passed=True,
                message=f"All required tools were used: {', '.join(required_tools)}",
                details={
                    "required": required_tools,
                    "used": list(used_tools),
                    "missing": [],
                },
            )

        return self._create_check(
            name="must_use_tools",
            passed=False,
            message=f"Missing required tools: {', '.join(missing_tools)}",
            details={
                "required": required_tools,
                "used": list(used_tools),
                "missing": missing_tools,
            },
        )

    def _check_max_tool_calls(
        self,
        trace: list[ATPEvent],
        response: ATPResponse,
        config: dict[str, Any],
    ) -> EvalCheck:
        """Check that tool call count doesn't exceed limit."""
        limit = config.get("limit")
        if limit is None:
            return self._create_check(
                name="max_tool_calls",
                passed=True,
                message="No limit specified",
            )

        actual = self._count_tool_calls(trace, response)

        if actual <= limit:
            return self._create_check(
                name="max_tool_calls",
                passed=True,
                message=f"Tool calls within limit: {actual} <= {limit}",
                details={"actual": actual, "limit": limit},
            )

        return self._create_check(
            name="max_tool_calls",
            passed=False,
            message=f"Tool calls exceeded limit: {actual} > {limit}",
            details={"actual": actual, "limit": limit},
        )

    def _check_min_tool_calls(
        self,
        trace: list[ATPEvent],
        response: ATPResponse,
        config: dict[str, Any],
    ) -> EvalCheck:
        """Check that at least N tool calls were made."""
        limit = config.get("limit")
        if limit is None:
            return self._create_check(
                name="min_tool_calls",
                passed=True,
                message="No minimum specified",
            )

        actual = self._count_tool_calls(trace, response)

        if actual >= limit:
            return self._create_check(
                name="min_tool_calls",
                passed=True,
                message=f"Tool calls meet minimum: {actual} >= {limit}",
                details={"actual": actual, "limit": limit},
            )

        return self._create_check(
            name="min_tool_calls",
            passed=False,
            message=f"Tool calls below minimum: {actual} < {limit}",
            details={"actual": actual, "limit": limit},
        )

    def _check_no_errors(
        self, trace: list[ATPEvent], response: ATPResponse
    ) -> EvalCheck:
        """Check that no error events occurred."""
        error_events = [e for e in trace if e.event_type == EventType.ERROR]

        if response.error:
            return self._create_check(
                name="no_errors",
                passed=False,
                message=f"Response contains error: {response.error}",
                details={
                    "response_error": response.error,
                    "trace_errors": len(error_events),
                },
            )

        if error_events:
            error_messages = []
            for event in error_events[:5]:
                msg = event.payload.get("message", "Unknown error")
                error_messages.append(msg)

            return self._create_check(
                name="no_errors",
                passed=False,
                message=f"Found {len(error_events)} error event(s) in trace",
                details={
                    "error_count": len(error_events),
                    "errors": error_messages,
                },
            )

        return self._create_check(
            name="no_errors",
            passed=True,
            message="No errors found",
            details={"error_count": 0},
        )

    def _check_forbidden_tools(
        self, trace: list[ATPEvent], config: dict[str, Any]
    ) -> EvalCheck:
        """Check that specific tools were NOT called."""
        forbidden_tools = config.get("tools", [])
        if not forbidden_tools:
            return self._create_check(
                name="forbidden_tools",
                passed=True,
                message="No forbidden tools specified",
            )

        used_tools = self._extract_used_tools(trace)
        violations = [t for t in forbidden_tools if t in used_tools]

        if not violations:
            return self._create_check(
                name="forbidden_tools",
                passed=True,
                message="No forbidden tools were used",
                details={
                    "forbidden": forbidden_tools,
                    "used": list(used_tools),
                    "violations": [],
                },
            )

        return self._create_check(
            name="forbidden_tools",
            passed=False,
            message=f"Forbidden tools were used: {', '.join(violations)}",
            details={
                "forbidden": forbidden_tools,
                "used": list(used_tools),
                "violations": violations,
            },
        )

    def _extract_used_tools(self, trace: list[ATPEvent]) -> set[str]:
        """Extract set of tool names used from trace."""
        tools: set[str] = set()
        for event in trace:
            if event.event_type == EventType.TOOL_CALL:
                tool_name = event.payload.get("tool")
                if tool_name:
                    tools.add(tool_name)
        return tools

    def _count_tool_calls(self, trace: list[ATPEvent], response: ATPResponse) -> int:
        """Count total tool calls from trace or response metrics."""
        if response.metrics and response.metrics.tool_calls is not None:
            return response.metrics.tool_calls

        return sum(1 for e in trace if e.event_type == EventType.TOOL_CALL)
