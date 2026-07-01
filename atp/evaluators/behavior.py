"""Behavior evaluator for checking agent execution patterns."""

from typing import Any

from pydantic import BaseModel, Field

from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse, EventType

from .base import EvalCheck, EvalResult, Evaluator
from .json_path.resolver import InvalidPath, resolve


class PayloadMatch(BaseModel):
    """A single partial payload match rule."""

    path: str
    operator: str
    expected: Any = None


class ToolCallExpectation(BaseModel):
    """A tool-call pattern matched against trace events."""

    tool: str
    status: str | None = None
    input_matches: list[PayloadMatch] = Field(default_factory=list)
    output_matches: list[PayloadMatch] = Field(default_factory=list)


class ToolCallMatchResult(BaseModel):
    """Result for matching one tool-call expectation."""

    matched: bool
    reason: str | None = None
    event_sequence: int | None = None


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

        if "expected_tool_calls" in config or "tool_call_order" in config:
            checks.append(self._check_expected_tool_calls(trace, config))

        if "forbidden_tool_calls" in config:
            checks.append(self._check_forbidden_tool_calls(trace, config))

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

    def _check_expected_tool_calls(
        self, trace: list[ATPEvent], config: dict[str, Any]
    ) -> EvalCheck:
        """Check that configured tool-call patterns appear in the trace."""
        order = config.get("tool_call_order", "any")
        if order not in {"any", "expected"}:
            return self._create_check(
                name="expected_tool_calls",
                passed=False,
                message=(
                    "Configuration error: tool_call_order must be 'any' "
                    f"or 'expected', got {order!r}"
                ),
            )

        expectations, error = self._parse_tool_call_expectations(
            config.get("expected_tool_calls", []),
            "expected_tool_calls",
        )
        if error:
            return self._create_check(
                name="expected_tool_calls",
                passed=False,
                message=error,
            )

        if not expectations:
            return self._create_check(
                name="expected_tool_calls",
                passed=True,
                message="No expected tool calls specified",
            )

        events = self._iter_tool_call_events(trace)
        if order == "expected":
            return self._check_expected_tool_calls_ordered(events, expectations)

        return self._check_expected_tool_calls_unordered(events, expectations)

    def _check_expected_tool_calls_unordered(
        self, events: list[ATPEvent], expectations: list[ToolCallExpectation]
    ) -> EvalCheck:
        """Check expected tool calls without ordering constraints."""
        matches: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []

        for expectation in expectations:
            result = self._find_matching_tool_call(events, expectation)
            if result.matched:
                matches.append(
                    {
                        "tool": expectation.tool,
                        "event_sequence": result.event_sequence,
                    }
                )
            else:
                missing.append(
                    {
                        "tool": expectation.tool,
                        "reason": result.reason,
                    }
                )

        if not missing:
            return self._create_check(
                name="expected_tool_calls",
                passed=True,
                message="All expected tool calls matched",
                details={
                    "expected": len(expectations),
                    "matched": len(matches),
                    "matches": matches,
                },
            )

        first_missing = missing[0]
        reason = first_missing.get("reason")
        message = f"Missing expected tool call: {first_missing['tool']}"
        if reason:
            message = f"{message} ({reason})"
        return self._create_check(
            name="expected_tool_calls",
            passed=False,
            message=message,
            details={
                "missing": missing,
                "observed_tools": self._observed_tool_names(events),
            },
        )

    def _check_expected_tool_calls_ordered(
        self, events: list[ATPEvent], expectations: list[ToolCallExpectation]
    ) -> EvalCheck:
        """Check expected tool calls in listed order by event sequence."""
        matches: list[dict[str, Any]] = []
        last_sequence = -1
        start_index = 0

        for expectation in expectations:
            found = False
            last_reason: str | None = None

            for index, event in enumerate(events[start_index:], start=start_index):
                if event.sequence <= last_sequence:
                    continue

                result = self._match_tool_call_event(event, expectation)
                if result.matched:
                    matches.append(
                        {
                            "tool": expectation.tool,
                            "event_sequence": event.sequence,
                        }
                    )
                    last_sequence = event.sequence
                    start_index = index + 1
                    found = True
                    break

                last_reason = result.reason

            if not found:
                message = (
                    "Missing expected tool call in configured order: "
                    f"{expectation.tool}"
                )
                if last_reason:
                    message = f"{message} ({last_reason})"
                return self._create_check(
                    name="expected_tool_calls",
                    passed=False,
                    message=message,
                    details={
                        "missing": [
                            {
                                "tool": expectation.tool,
                                "reason": last_reason,
                            }
                        ],
                        "matches": matches,
                        "observed_tools": self._observed_tool_names(events),
                    },
                )

        return self._create_check(
            name="expected_tool_calls",
            passed=True,
            message="All expected tool calls matched in order",
            details={
                "expected": len(expectations),
                "matched": len(matches),
                "matches": matches,
            },
        )

    def _check_forbidden_tool_calls(
        self, trace: list[ATPEvent], config: dict[str, Any]
    ) -> EvalCheck:
        """Check that configured forbidden tool-call patterns do not appear."""
        expectations, error = self._parse_tool_call_expectations(
            config.get("forbidden_tool_calls", []),
            "forbidden_tool_calls",
        )
        if error:
            return self._create_check(
                name="forbidden_tool_calls",
                passed=False,
                message=error,
            )

        if not expectations:
            return self._create_check(
                name="forbidden_tool_calls",
                passed=True,
                message="No forbidden tool calls specified",
            )

        violations: list[dict[str, Any]] = []
        for event in self._iter_tool_call_events(trace):
            for expectation in expectations:
                result = self._match_tool_call_event(event, expectation)
                if result.matched:
                    violations.append(
                        {
                            "tool": expectation.tool,
                            "event_sequence": event.sequence,
                        }
                    )

        if not violations:
            return self._create_check(
                name="forbidden_tool_calls",
                passed=True,
                message="No forbidden tool call patterns matched",
                details={"violations": []},
            )

        first_violation = violations[0]
        return self._create_check(
            name="forbidden_tool_calls",
            passed=False,
            message=f"Forbidden tool call matched: {first_violation['tool']}",
            details={"violations": violations},
        )

    def _parse_tool_call_expectations(
        self, raw_expectations: Any, config_key: str
    ) -> tuple[list[ToolCallExpectation], str | None]:
        """Parse raw tool-call expectation config into internal matchers."""
        if not isinstance(raw_expectations, list):
            return [], f"Configuration error: {config_key} must be a list"

        expectations: list[ToolCallExpectation] = []
        for index, raw_expectation in enumerate(raw_expectations):
            if not isinstance(raw_expectation, dict):
                return (
                    [],
                    f"Configuration error: {config_key}[{index}] must be a mapping",
                )

            tool = raw_expectation.get("tool")
            if not isinstance(tool, str) or not tool:
                return (
                    [],
                    f"Configuration error: {config_key}[{index}].tool is required",
                )

            input_matches, error = self._parse_payload_matches(
                raw_expectation.get("input_matches", []),
                f"{config_key}[{index}].input_matches",
            )
            if error:
                return [], error

            output_matches, error = self._parse_payload_matches(
                raw_expectation.get("output_matches", []),
                f"{config_key}[{index}].output_matches",
            )
            if error:
                return [], error

            status = raw_expectation.get("status")
            if status is not None and not isinstance(status, str):
                return (
                    [],
                    "Configuration error: "
                    f"{config_key}[{index}].status must be a string",
                )

            expectations.append(
                ToolCallExpectation(
                    tool=tool,
                    status=status,
                    input_matches=input_matches,
                    output_matches=output_matches,
                )
            )

        return expectations, None

    def _parse_payload_matches(
        self, raw_matches: Any, config_key: str
    ) -> tuple[list[PayloadMatch], str | None]:
        """Parse payload match rules from raw config."""
        if not isinstance(raw_matches, list):
            return [], f"Configuration error: {config_key} must be a list"

        matches: list[PayloadMatch] = []
        for index, raw_match in enumerate(raw_matches):
            if not isinstance(raw_match, dict):
                return (
                    [],
                    f"Configuration error: {config_key}[{index}] must be a mapping",
                )

            path = raw_match.get("path")
            if not isinstance(path, str):
                return (
                    [],
                    f"Configuration error: {config_key}[{index}].path is required",
                )
            try:
                resolve({}, path)
            except InvalidPath:
                return (
                    [],
                    f"Configuration error: invalid JSONPath {path!r} "
                    f"in {config_key}[{index}]",
                )

            operators = [
                operator
                for operator in ("equals", "exists", "absent")
                if operator in raw_match
            ]
            if len(operators) != 1:
                return (
                    [],
                    "Configuration error: "
                    f"{config_key}[{index}] must specify exactly one of "
                    "equals, exists, or absent",
                )

            operator = operators[0]
            if operator in {"exists", "absent"} and raw_match[operator] is not True:
                return (
                    [],
                    "Configuration error: "
                    f"{config_key}[{index}].{operator} must be true",
                )

            matches.append(
                PayloadMatch(
                    path=path,
                    operator=operator,
                    expected=raw_match.get(operator),
                )
            )

        return matches, None

    def _find_matching_tool_call(
        self, events: list[ATPEvent], expectation: ToolCallExpectation
    ) -> ToolCallMatchResult:
        """Find the first event matching an expectation."""
        last_reason: str | None = None
        for event in events:
            result = self._match_tool_call_event(event, expectation)
            if result.matched:
                return result
            last_reason = result.reason

        return ToolCallMatchResult(matched=False, reason=last_reason)

    def _match_tool_call_event(
        self, event: ATPEvent, expectation: ToolCallExpectation
    ) -> ToolCallMatchResult:
        """Match a single tool-call event against one expectation."""
        payload = event.payload
        tool = payload.get("tool")
        if tool != expectation.tool:
            return ToolCallMatchResult(
                matched=False,
                reason=f"tool expected {expectation.tool!r}, got {tool!r}",
            )

        if (
            expectation.status is not None
            and payload.get("status") != expectation.status
        ):
            return ToolCallMatchResult(
                matched=False,
                reason=(
                    f"status expected {expectation.status!r}, "
                    f"got {payload.get('status')!r}"
                ),
            )

        input_result = self._match_payload(
            self._tool_call_input(payload), expectation.input_matches
        )
        if not input_result.matched:
            return input_result

        output_result = self._match_payload(
            self._tool_call_output(payload), expectation.output_matches
        )
        if not output_result.matched:
            return output_result

        return ToolCallMatchResult(matched=True, event_sequence=event.sequence)

    def _match_payload(
        self, data: Any, matches: list[PayloadMatch]
    ) -> ToolCallMatchResult:
        """Match all payload rules against input or output data."""
        for rule in matches:
            result = self._match_payload_rule(data, rule)
            if not result.matched:
                return result
        return ToolCallMatchResult(matched=True)

    def _match_payload_rule(self, data: Any, rule: PayloadMatch) -> ToolCallMatchResult:
        """Match one payload rule against input or output data."""
        try:
            found, value = resolve(data, rule.path)
        except InvalidPath:
            return ToolCallMatchResult(
                matched=False,
                reason=f"Configuration error: invalid JSONPath {rule.path!r}",
            )

        if rule.operator == "equals":
            if found and value == rule.expected:
                return ToolCallMatchResult(matched=True)
            return ToolCallMatchResult(
                matched=False,
                reason=f"{rule.path} expected {rule.expected!r}, got {value!r}",
            )

        if rule.operator == "exists":
            if found:
                return ToolCallMatchResult(matched=True)
            return ToolCallMatchResult(
                matched=False,
                reason=f"{rule.path} expected to exist",
            )

        if not found:
            return ToolCallMatchResult(matched=True)
        return ToolCallMatchResult(
            matched=False,
            reason=f"{rule.path} expected to be absent",
        )

    def _iter_tool_call_events(self, trace: list[ATPEvent]) -> list[ATPEvent]:
        """Extract tool-call events from a trace."""
        return [event for event in trace if event.event_type == EventType.TOOL_CALL]

    def _tool_call_input(self, payload: dict[str, Any]) -> Any:
        """Return canonical tool input, falling back to legacy args."""
        if "input" in payload:
            return payload["input"]
        return payload.get("args")

    def _tool_call_output(self, payload: dict[str, Any]) -> Any:
        """Return tool output payload."""
        return payload.get("output")

    def _observed_tool_names(self, events: list[ATPEvent]) -> list[str]:
        """Return unique observed tool names in trace order."""
        tools: list[str] = []
        for event in events:
            tool = event.payload.get("tool")
            if isinstance(tool, str) and tool not in tools:
                tools.append(tool)
        return tools

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
