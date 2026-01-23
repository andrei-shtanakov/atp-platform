"""Unit tests for BehaviorEvaluator."""

from datetime import datetime

import pytest

from atp.evaluators.behavior import BehaviorEvaluator
from atp.loader.models import Assertion, Constraints, TaskDefinition, TestDefinition
from atp.protocol import (
    ATPEvent,
    ATPResponse,
    EventType,
    Metrics,
    ResponseStatus,
)


@pytest.fixture
def evaluator() -> BehaviorEvaluator:
    """Create BehaviorEvaluator instance."""
    return BehaviorEvaluator()


@pytest.fixture
def sample_task() -> TestDefinition:
    """Create a sample test definition."""
    return TestDefinition(
        id="test-001",
        name="Sample Test",
        task=TaskDefinition(description="Test task"),
        constraints=Constraints(),
    )


@pytest.fixture
def successful_response() -> ATPResponse:
    """Create a successful response."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        metrics=Metrics(tool_calls=5, total_steps=10),
    )


@pytest.fixture
def failed_response() -> ATPResponse:
    """Create a failed response with error."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.FAILED,
        error="Task execution failed due to timeout",
    )


@pytest.fixture
def response_no_metrics() -> ATPResponse:
    """Create a response without metrics."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
    )


def create_tool_call_event(
    tool: str, task_id: str = "test-001", sequence: int = 0
) -> ATPEvent:
    """Helper to create tool call events."""
    return ATPEvent(
        task_id=task_id,
        timestamp=datetime.now(),
        sequence=sequence,
        event_type=EventType.TOOL_CALL,
        payload={"tool": tool, "status": "success"},
    )


def create_error_event(
    message: str, task_id: str = "test-001", sequence: int = 0
) -> ATPEvent:
    """Helper to create error events."""
    return ATPEvent(
        task_id=task_id,
        timestamp=datetime.now(),
        sequence=sequence,
        event_type=EventType.ERROR,
        payload={"error_type": "ExecutionError", "message": message},
    )


@pytest.fixture
def trace_with_tools() -> list[ATPEvent]:
    """Create trace with multiple tool calls."""
    return [
        create_tool_call_event("web_search", sequence=0),
        create_tool_call_event("file_write", sequence=1),
        create_tool_call_event("web_search", sequence=2),
        ATPEvent(
            task_id="test-001",
            timestamp=datetime.now(),
            sequence=3,
            event_type=EventType.LLM_REQUEST,
            payload={"model": "claude-3-sonnet", "input_tokens": 100},
        ),
        create_tool_call_event("file_read", sequence=4),
    ]


@pytest.fixture
def trace_with_errors() -> list[ATPEvent]:
    """Create trace with error events."""
    return [
        create_tool_call_event("web_search", sequence=0),
        create_error_event("API rate limit exceeded", sequence=1),
        create_error_event("Connection timeout", sequence=2),
    ]


@pytest.fixture
def empty_trace() -> list[ATPEvent]:
    """Create empty trace."""
    return []


class TestMustUseTools:
    """Tests for must_use_tools assertion type."""

    @pytest.mark.anyio
    async def test_must_use_tools_pass(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test must_use_tools passes when all required tools used."""
        assertion = Assertion(
            type="must_use_tools",
            config={"tools": ["web_search", "file_write"]},
        )
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is True
        assert result.checks[0].details["missing"] == []

    @pytest.mark.anyio
    async def test_must_use_tools_fail(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test must_use_tools fails when required tool not used."""
        assertion = Assertion(
            type="must_use_tools",
            config={"tools": ["database_query"]},
        )
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is False
        assert "database_query" in result.checks[0].details["missing"]

    @pytest.mark.anyio
    async def test_must_use_tools_partial_fail(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test must_use_tools fails when some required tools not used."""
        assertion = Assertion(
            type="must_use_tools",
            config={"tools": ["web_search", "missing_tool"]},
        )
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is False
        assert "missing_tool" in result.checks[0].details["missing"]
        assert "web_search" in result.checks[0].details["used"]

    @pytest.mark.anyio
    async def test_must_use_tools_empty_list(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test must_use_tools passes with empty tools list."""
        assertion = Assertion(type="must_use_tools", config={"tools": []})
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is True

    @pytest.mark.anyio
    async def test_must_use_tools_empty_trace(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        empty_trace: list[ATPEvent],
    ) -> None:
        """Test must_use_tools fails when trace is empty."""
        assertion = Assertion(
            type="must_use_tools",
            config={"tools": ["web_search"]},
        )
        result = await evaluator.evaluate(
            sample_task, successful_response, empty_trace, assertion
        )
        assert result.passed is False


class TestMaxToolCalls:
    """Tests for max_tool_calls assertion type."""

    @pytest.mark.anyio
    async def test_max_tool_calls_pass_from_metrics(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test max_tool_calls passes using metrics count."""
        assertion = Assertion(type="max_tool_calls", config={"limit": 10})
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is True
        assert result.checks[0].details["actual"] == 5
        assert result.checks[0].details["limit"] == 10

    @pytest.mark.anyio
    async def test_max_tool_calls_fail(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test max_tool_calls fails when limit exceeded."""
        assertion = Assertion(type="max_tool_calls", config={"limit": 3})
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is False
        assert "exceeded" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_max_tool_calls_from_trace(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        response_no_metrics: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test max_tool_calls counts from trace when no metrics."""
        assertion = Assertion(type="max_tool_calls", config={"limit": 5})
        result = await evaluator.evaluate(
            sample_task, response_no_metrics, trace_with_tools, assertion
        )
        assert result.passed is True
        assert result.checks[0].details["actual"] == 4

    @pytest.mark.anyio
    async def test_max_tool_calls_exact_limit(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test max_tool_calls passes at exact limit."""
        assertion = Assertion(type="max_tool_calls", config={"limit": 5})
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is True

    @pytest.mark.anyio
    async def test_max_tool_calls_no_limit(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test max_tool_calls passes when no limit specified."""
        assertion = Assertion(type="max_tool_calls", config={})
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is True


class TestMinToolCalls:
    """Tests for min_tool_calls assertion type."""

    @pytest.mark.anyio
    async def test_min_tool_calls_pass(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test min_tool_calls passes when minimum met."""
        assertion = Assertion(type="min_tool_calls", config={"limit": 3})
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is True

    @pytest.mark.anyio
    async def test_min_tool_calls_fail(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test min_tool_calls fails when below minimum."""
        assertion = Assertion(type="min_tool_calls", config={"limit": 10})
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is False
        assert "below" in result.checks[0].message.lower()


class TestNoErrors:
    """Tests for no_errors assertion type."""

    @pytest.mark.anyio
    async def test_no_errors_pass(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test no_errors passes when no errors present."""
        assertion = Assertion(type="no_errors", config={})
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is True
        assert result.checks[0].details["error_count"] == 0

    @pytest.mark.anyio
    async def test_no_errors_fail_response_error(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        failed_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test no_errors fails when response has error."""
        assertion = Assertion(type="no_errors", config={})
        result = await evaluator.evaluate(
            sample_task, failed_response, trace_with_tools, assertion
        )
        assert result.passed is False
        assert "response contains error" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_no_errors_fail_trace_errors(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_errors: list[ATPEvent],
    ) -> None:
        """Test no_errors fails when trace has error events."""
        assertion = Assertion(type="no_errors", config={})
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_errors, assertion
        )
        assert result.passed is False
        assert result.checks[0].details["error_count"] == 2


class TestForbiddenTools:
    """Tests for forbidden_tools assertion type."""

    @pytest.mark.anyio
    async def test_forbidden_tools_pass(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test forbidden_tools passes when none used."""
        assertion = Assertion(
            type="forbidden_tools",
            config={"tools": ["delete_file", "execute_code"]},
        )
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is True
        assert result.checks[0].details["violations"] == []

    @pytest.mark.anyio
    async def test_forbidden_tools_fail(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test forbidden_tools fails when forbidden tool used."""
        assertion = Assertion(
            type="forbidden_tools",
            config={"tools": ["web_search"]},
        )
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is False
        assert "web_search" in result.checks[0].details["violations"]

    @pytest.mark.anyio
    async def test_forbidden_tools_empty_list(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test forbidden_tools passes with empty list."""
        assertion = Assertion(type="forbidden_tools", config={"tools": []})
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is True


class TestBehaviorCombined:
    """Tests for combined behavior assertion type."""

    @pytest.mark.anyio
    async def test_behavior_combined_all_pass(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test behavior assertion with multiple passing checks."""
        assertion = Assertion(
            type="behavior",
            config={
                "must_use_tools": ["web_search"],
                "max_tool_calls": 10,
                "no_errors": True,
            },
        )
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is True
        assert result.total_checks == 3

    @pytest.mark.anyio
    async def test_behavior_combined_partial_fail(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test behavior assertion with mixed pass/fail checks."""
        assertion = Assertion(
            type="behavior",
            config={
                "must_use_tools": ["web_search"],
                "max_tool_calls": 3,
            },
        )
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is False
        assert result.passed_checks == 1
        assert result.failed_checks == 1

    @pytest.mark.anyio
    async def test_behavior_empty_config(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test behavior assertion with empty config passes."""
        assertion = Assertion(type="behavior", config={})
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is True
        assert result.total_checks == 1


class TestUnknownAssertionType:
    """Tests for unknown assertion types."""

    @pytest.mark.anyio
    async def test_unknown_type(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        trace_with_tools: list[ATPEvent],
    ) -> None:
        """Test unknown assertion type returns failure."""
        assertion = Assertion(type="unknown_behavior_type", config={})
        result = await evaluator.evaluate(
            sample_task, successful_response, trace_with_tools, assertion
        )
        assert result.passed is False
        assert "unknown" in result.checks[0].message.lower()


class TestEdgeCases:
    """Edge case tests for BehaviorEvaluator."""

    @pytest.mark.anyio
    async def test_empty_trace_no_errors(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
        empty_trace: list[ATPEvent],
    ) -> None:
        """Test no_errors passes with empty trace."""
        assertion = Assertion(type="no_errors", config={})
        result = await evaluator.evaluate(
            sample_task, successful_response, empty_trace, assertion
        )
        assert result.passed is True

    @pytest.mark.anyio
    async def test_tool_call_without_tool_name(
        self,
        evaluator: BehaviorEvaluator,
        sample_task: TestDefinition,
        successful_response: ATPResponse,
    ) -> None:
        """Test tool call events without tool name are ignored."""
        trace = [
            ATPEvent(
                task_id="test-001",
                timestamp=datetime.now(),
                sequence=0,
                event_type=EventType.TOOL_CALL,
                payload={"status": "success"},
            ),
        ]
        assertion = Assertion(
            type="must_use_tools",
            config={"tools": ["any_tool"]},
        )
        result = await evaluator.evaluate(
            sample_task, successful_response, trace, assertion
        )
        assert result.passed is False
        assert result.checks[0].details["used"] == []


class TestEvaluatorProperties:
    """Tests for evaluator properties."""

    def test_evaluator_name(self, evaluator: BehaviorEvaluator) -> None:
        """Test evaluator name property."""
        assert evaluator.name == "behavior"
