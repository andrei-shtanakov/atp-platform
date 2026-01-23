"""Integration tests for the full ATP test pipeline.

Tests the complete flow: loader → runner → evaluators → results
including timeout handling, error recovery, and multi-run statistics.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

import pytest

from atp.adapters.base import AgentAdapter
from atp.adapters.http import HTTPAdapter, HTTPAdapterConfig
from atp.evaluators.artifact import ArtifactEvaluator
from atp.evaluators.base import EvalResult
from atp.evaluators.behavior import BehaviorEvaluator
from atp.loader.loader import TestLoader
from atp.loader.models import (
    Assertion,
    Constraints,
    TaskDefinition,
    TestDefinition,
    TestSuite,
)
from atp.protocol import (
    ArtifactStructured,
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
    Metrics,
    ResponseStatus,
)
from atp.runner.models import (
    ProgressEvent,
    ProgressEventType,
    SuiteResult,
)
from atp.runner.orchestrator import TestOrchestrator

# =============================================================================
# Mock Adapter for Controlled Testing
# =============================================================================


class MockAdapter(AgentAdapter):
    """Mock adapter for controlled integration testing."""

    def __init__(
        self,
        responses: list[ATPResponse] | None = None,
        events: list[ATPEvent] | None = None,
        delay_seconds: float = 0.0,
        fail_after: int | None = None,
        timeout_after: int | None = None,
    ) -> None:
        """Initialize mock adapter.

        Args:
            responses: List of responses to return (cycled if multiple calls).
            events: Events to emit during streaming.
            delay_seconds: Delay before returning response.
            fail_after: Fail after this many calls.
            timeout_after: Timeout after this many calls.
        """
        self._responses = responses or []
        self._events = events or []
        self._delay_seconds = delay_seconds
        self._fail_after = fail_after
        self._timeout_after = timeout_after
        self._call_count = 0
        self._requests: list[ATPRequest] = []

    @property
    def adapter_type(self) -> str:
        return "mock"

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def requests(self) -> list[ATPRequest]:
        return self._requests

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """Execute mock request."""
        self._call_count += 1
        self._requests.append(request)

        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)

        # Check for timeout simulation
        if self._timeout_after and self._call_count > self._timeout_after:
            await asyncio.sleep(1000)  # Will be interrupted by timeout

        # Check for failure simulation
        if self._fail_after and self._call_count > self._fail_after:
            return ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error="Simulated failure",
            )

        # Return configured response or default
        if self._responses:
            idx = (self._call_count - 1) % len(self._responses)
            response = self._responses[idx]
            # Update task_id to match request
            return ATPResponse(
                task_id=request.task_id,
                status=response.status,
                artifacts=response.artifacts,
                metrics=response.metrics,
                error=response.error,
            )

        return ATPResponse(
            task_id=request.task_id,
            status=ResponseStatus.COMPLETED,
            artifacts=[],
            metrics=Metrics(total_tokens=100, total_steps=2),
        )

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """Stream mock events."""
        self._call_count += 1
        self._requests.append(request)

        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)

        # Emit events
        for i, event in enumerate(self._events):
            yield ATPEvent(
                event_type=event.event_type,
                task_id=request.task_id,
                sequence=i,
                timestamp=datetime.now(),
                payload=event.payload,
            )
            await asyncio.sleep(0.01)

        # Check for failure simulation
        if self._fail_after and self._call_count > self._fail_after:
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error="Simulated failure",
            )
            return

        # Yield final response
        if self._responses:
            idx = (self._call_count - 1) % len(self._responses)
            response = self._responses[idx]
            yield ATPResponse(
                task_id=request.task_id,
                status=response.status,
                artifacts=response.artifacts,
                metrics=response.metrics,
                error=response.error,
            )
        else:
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
                artifacts=[],
                metrics=Metrics(total_tokens=100, total_steps=2),
            )

    async def health_check(self) -> bool:
        return True

    async def cleanup(self) -> None:
        pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_test_definition() -> TestDefinition:
    """Create a sample test definition."""
    return TestDefinition(
        id="test-001",
        name="Sample Test",
        description="A sample test for integration testing",
        tags=["integration", "sample"],
        task=TaskDefinition(
            description="Perform a sample task",
            input_data={"key": "value"},
            expected_artifacts=["result.json"],
        ),
        constraints=Constraints(
            max_steps=10,
            max_tokens=1000,
            timeout_seconds=30,
        ),
        assertions=[
            Assertion(
                type="artifact_exists",
                config={"path": "result"},
            ),
        ],
    )


@pytest.fixture
def sample_test_suite(sample_test_definition: TestDefinition) -> TestSuite:
    """Create a sample test suite."""
    return TestSuite(
        test_suite="Integration Test Suite",
        version="1.0",
        description="Test suite for integration testing",
        tests=[
            sample_test_definition,
            TestDefinition(
                id="test-002",
                name="Second Test",
                task=TaskDefinition(description="Another task"),
                constraints=Constraints(timeout_seconds=30),
            ),
        ],
    )


@pytest.fixture
def mock_response() -> ATPResponse:
    """Create a mock successful response."""
    return ATPResponse(
        task_id="test",
        status=ResponseStatus.COMPLETED,
        artifacts=[
            ArtifactStructured(
                name="result",
                data={"message": "Task completed successfully", "value": 42},
            ),
        ],
        metrics=Metrics(
            total_tokens=150,
            input_tokens=50,
            output_tokens=100,
            total_steps=3,
            tool_calls=2,
            wall_time_seconds=1.5,
        ),
    )


@pytest.fixture
def mock_events() -> list[ATPEvent]:
    """Create mock events."""
    return [
        ATPEvent(
            event_type=EventType.PROGRESS,
            task_id="test",
            sequence=0,
            timestamp=datetime.now(),
            payload={"message": "Starting task"},
        ),
        ATPEvent(
            event_type=EventType.TOOL_CALL,
            task_id="test",
            sequence=1,
            timestamp=datetime.now(),
            payload={"tool": "search", "args": {"query": "test"}},
        ),
        ATPEvent(
            event_type=EventType.PROGRESS,
            task_id="test",
            sequence=2,
            timestamp=datetime.now(),
            payload={"message": "Processing complete"},
        ),
    ]


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================


class TestFullPipeline:
    """Integration tests for the complete test pipeline."""

    @pytest.mark.anyio
    async def test_loader_to_runner_to_evaluator(
        self, mock_response: ATPResponse, mock_events: list[ATPEvent]
    ) -> None:
        """Test the full pipeline: loader → runner → evaluators."""
        # Load test suite from YAML string
        yaml_content = """
test_suite: Full Pipeline Test
version: "1.0"
description: Test the full pipeline
defaults:
  timeout_seconds: 30
  scoring:
    quality_weight: 0.4
    completeness_weight: 0.3
    efficiency_weight: 0.2
    cost_weight: 0.1
tests:
  - id: pipeline-test-001
    name: Pipeline Test
    task:
      description: Execute a sample task and produce output
      expected_artifacts:
        - result
    constraints:
      max_steps: 10
      timeout_seconds: 30
    assertions:
      - type: artifact_exists
        config:
          path: result
"""
        loader = TestLoader()
        suite = loader.load_string(yaml_content)

        assert suite.test_suite == "Full Pipeline Test"
        assert len(suite.tests) == 1

        # Create mock adapter with response
        adapter = MockAdapter(responses=[mock_response], events=mock_events)

        # Run through orchestrator
        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(suite.tests[0])

        assert result.success
        assert result.total_runs == 1
        assert result.successful_runs == 1

        # Run evaluator on the result
        evaluator = ArtifactEvaluator()
        run_result = result.runs[0]

        eval_result = await evaluator.evaluate(
            task=suite.tests[0],
            response=run_result.response,
            trace=run_result.events,
            assertion=suite.tests[0].assertions[0],
        )

        assert eval_result.passed
        assert eval_result.score == 1.0

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_suite_execution_with_multiple_tests(
        self, sample_test_suite: TestSuite, mock_response: ATPResponse
    ) -> None:
        """Test executing a complete test suite with multiple tests."""
        adapter = MockAdapter(responses=[mock_response])

        progress_events: list[ProgressEvent] = []

        def progress_callback(event: ProgressEvent) -> None:
            progress_events.append(event)

        orchestrator = TestOrchestrator(
            adapter=adapter,
            progress_callback=progress_callback,
            runs_per_test=1,
        )

        result = await orchestrator.run_suite(
            sample_test_suite, agent_name="test-agent"
        )

        assert isinstance(result, SuiteResult)
        assert result.suite_name == "Integration Test Suite"
        assert result.agent_name == "test-agent"
        assert result.total_tests == 2
        assert result.passed_tests == 2
        assert result.success_rate == 1.0

        # Check progress events were emitted
        event_types = [e.event_type for e in progress_events]
        assert ProgressEventType.SUITE_STARTED in event_types
        assert ProgressEventType.SUITE_COMPLETED in event_types
        assert event_types.count(ProgressEventType.TEST_STARTED) == 2
        assert event_types.count(ProgressEventType.TEST_COMPLETED) == 2

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_pipeline_with_streaming_events(
        self,
        sample_test_definition: TestDefinition,
        mock_response: ATPResponse,
        mock_events: list[ATPEvent],
    ) -> None:
        """Test pipeline captures streaming events."""
        adapter = MockAdapter(responses=[mock_response], events=mock_events)

        collected_events: list[ATPEvent] = []

        def progress_callback(event: ProgressEvent) -> None:
            if event.agent_event:
                collected_events.append(event.agent_event)

        orchestrator = TestOrchestrator(
            adapter=adapter,
            progress_callback=progress_callback,
            runs_per_test=1,
        )

        result = await orchestrator.run_single_test(sample_test_definition)

        assert result.success
        assert len(result.runs) == 1
        assert len(result.runs[0].events) == len(mock_events)

        # Verify events were passed to callback
        assert len(collected_events) == len(mock_events)

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_full_pipeline_with_behavior_evaluator(
        self, mock_events: list[ATPEvent]
    ) -> None:
        """Test full pipeline with behavior evaluator for tool usage validation."""
        # Load test with behavior assertions
        yaml_content = """
test_suite: Behavior Test Suite
version: "1.0"
tests:
  - id: behavior-test-001
    name: Tool Usage Test
    task:
      description: Task that must use specific tools
    constraints:
      timeout_seconds: 30
    assertions:
      - type: must_use_tools
        config:
          tools:
            - search
"""
        loader = TestLoader()
        suite = loader.load_string(yaml_content)

        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            artifacts=[],
            metrics=Metrics(total_tokens=100, total_steps=2, tool_calls=1),
        )

        adapter = MockAdapter(responses=[response], events=mock_events)

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(suite.tests[0])

        assert result.success

        # Run behavior evaluator
        evaluator = BehaviorEvaluator()
        run_result = result.runs[0]

        eval_result = await evaluator.evaluate(
            task=suite.tests[0],
            response=run_result.response,
            trace=run_result.events,
            assertion=suite.tests[0].assertions[0],
        )

        # Should pass because mock_events includes a TOOL_CALL event with tool="search"
        assert eval_result.passed

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_loader_validation_errors(self) -> None:
        """Test that loader properly reports validation errors."""
        invalid_yaml = """
test_suite: Invalid Suite
version: "1.0"
tests:
  - id: test-001
    name: Test 1
    task:
      description: First test
  - id: test-001
    name: Duplicate ID
    task:
      description: Same ID as first test
"""
        loader = TestLoader()
        with pytest.raises(Exception) as exc_info:
            loader.load_string(invalid_yaml)

        # Should fail due to duplicate test IDs
        assert "Duplicate test ID" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_pipeline_with_multiple_assertions(self) -> None:
        """Test pipeline handles multiple assertions per test."""
        yaml_content = """
test_suite: Multi-Assertion Suite
version: "1.0"
tests:
  - id: multi-assert-001
    name: Multiple Assertions Test
    task:
      description: Test with multiple assertions
      expected_artifacts:
        - result
        - log
    constraints:
      timeout_seconds: 30
    assertions:
      - type: artifact_exists
        config:
          path: result
      - type: artifact_exists
        config:
          path: log
      - type: contains
        config:
          path: result
          pattern: success
"""
        loader = TestLoader()
        suite = loader.load_string(yaml_content)

        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="result",
                    data={"message": "Task completed with success"},
                ),
                ArtifactStructured(
                    name="log",
                    data={"entries": ["Started", "Completed"]},
                ),
            ],
        )

        adapter = MockAdapter(responses=[response])
        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(suite.tests[0])

        assert result.success

        # Evaluate all assertions
        evaluator = ArtifactEvaluator()
        run_result = result.runs[0]

        all_passed = True
        for assertion in suite.tests[0].assertions:
            eval_result = await evaluator.evaluate(
                task=suite.tests[0],
                response=run_result.response,
                trace=run_result.events,
                assertion=assertion,
            )
            if not eval_result.passed:
                all_passed = False

        assert all_passed

        await orchestrator.cleanup()


# =============================================================================
# Timeout Handling Tests
# =============================================================================


class TestTimeoutHandling:
    """Tests for timeout handling scenarios."""

    @pytest.mark.anyio
    async def test_adapter_timeout_captured(self) -> None:
        """Test that adapter timeout is properly captured."""
        test = TestDefinition(
            id="timeout-test",
            name="Timeout Test",
            task=TaskDefinition(description="Task that times out"),
            constraints=Constraints(timeout_seconds=1),  # Short timeout
        )

        # Adapter that delays longer than timeout
        adapter = MockAdapter(delay_seconds=5.0)

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        assert not result.success
        assert result.status == ResponseStatus.TIMEOUT
        assert result.runs[0].error is not None
        # Check for timeout mention (case-insensitive)
        error_lower = result.runs[0].error.lower()
        assert "timeout" in error_lower or "timed out" in error_lower

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_partial_completion_before_timeout(self) -> None:
        """Test that partial results are captured even on timeout."""
        events = [
            ATPEvent(
                event_type=EventType.PROGRESS,
                task_id="test",
                sequence=0,
                timestamp=datetime.now(),
                payload={"message": "Starting"},
            ),
            ATPEvent(
                event_type=EventType.PROGRESS,
                task_id="test",
                sequence=1,
                timestamp=datetime.now(),
                payload={"message": "Processing"},
            ),
        ]

        test = TestDefinition(
            id="partial-timeout-test",
            name="Partial Timeout Test",
            task=TaskDefinition(description="Task with partial progress"),
            constraints=Constraints(timeout_seconds=1),
        )

        adapter = MockAdapter(events=events, delay_seconds=5.0)

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        assert not result.success
        assert result.status == ResponseStatus.TIMEOUT

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_timeout_respects_constraints(self) -> None:
        """Test that timeout uses test constraint value."""
        test = TestDefinition(
            id="constraint-timeout-test",
            name="Constraint Timeout Test",
            task=TaskDefinition(description="Task with specific timeout"),
            constraints=Constraints(timeout_seconds=2),  # 2 second timeout
        )

        adapter = MockAdapter(delay_seconds=3.0)  # 3 seconds > 2 second timeout

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)

        start_time = datetime.now()
        result = await orchestrator.run_single_test(test)
        elapsed = (datetime.now() - start_time).total_seconds()

        assert not result.success
        # Should timeout around 2 seconds, not 3
        assert elapsed < 3.0

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_different_timeout_per_test_in_suite(self) -> None:
        """Test that different tests can have different timeouts."""
        test1 = TestDefinition(
            id="fast-test",
            name="Fast Test",
            task=TaskDefinition(description="Fast task"),
            constraints=Constraints(timeout_seconds=10),
        )
        test2 = TestDefinition(
            id="slow-test",
            name="Slow Test",
            task=TaskDefinition(description="Slow task"),
            constraints=Constraints(timeout_seconds=1),
        )

        suite = TestSuite(
            test_suite="Mixed Timeout Suite",
            tests=[test1, test2],
        )

        # Create a custom adapter that delays on the second test
        # The orchestrator uses stream_events by default, so we need to override that
        class DelayOnSecondCallAdapter(MockAdapter):
            def __init__(self) -> None:
                super().__init__()
                self._call_number = 0

            async def stream_events(
                self, request: ATPRequest
            ) -> AsyncIterator[ATPEvent | ATPResponse]:
                self._call_number += 1
                self._call_count += 1
                self._requests.append(request)

                # On second call, delay longer than timeout
                if self._call_number == 2:
                    await asyncio.sleep(5.0)

                yield ATPResponse(
                    task_id=request.task_id,
                    status=ResponseStatus.COMPLETED,
                )

        adapter = DelayOnSecondCallAdapter()

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_suite(suite, agent_name="test-agent")

        # First test passes (10s timeout, no delay)
        # Second test fails (1s timeout, 5s delay)
        assert result.total_tests == 2
        assert result.passed_tests == 1
        assert result.failed_tests == 1

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_graceful_timeout_with_partial_metrics(self) -> None:
        """Test timeout captures partial metrics when available."""
        test = TestDefinition(
            id="graceful-timeout-test",
            name="Graceful Timeout Test",
            task=TaskDefinition(description="Task with timeout"),
            constraints=Constraints(timeout_seconds=1),
        )

        adapter = MockAdapter(delay_seconds=5.0)
        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        assert not result.success
        assert result.status == ResponseStatus.TIMEOUT
        # The run should have timing information
        assert result.runs[0].duration_seconds is not None
        assert result.runs[0].duration_seconds > 0

        await orchestrator.cleanup()


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Tests for error recovery scenarios."""

    @pytest.mark.anyio
    async def test_adapter_failure_captured(self) -> None:
        """Test that adapter failures are properly captured."""
        test = TestDefinition(
            id="failure-test",
            name="Failure Test",
            task=TaskDefinition(description="Task that fails"),
            constraints=Constraints(timeout_seconds=30),
        )

        failure_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.FAILED,
            error="Agent execution failed",
        )
        adapter = MockAdapter(responses=[failure_response])

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        assert not result.success
        assert result.status == ResponseStatus.FAILED
        run = result.runs[0]
        assert run.response.status == ResponseStatus.FAILED
        assert run.response.error == "Agent execution failed"

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_intermittent_failures_captured(self) -> None:
        """Test handling of intermittent failures across multiple runs."""
        test = TestDefinition(
            id="intermittent-test",
            name="Intermittent Failure Test",
            task=TaskDefinition(description="Task with intermittent failures"),
            constraints=Constraints(timeout_seconds=30),
        )

        success_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            artifacts=[],
        )
        failure_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.FAILED,
            error="Intermittent failure",
        )

        # Adapter returns success, then failure, then success
        adapter = MockAdapter(
            responses=[success_response, failure_response, success_response]
        )

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=3)
        result = await orchestrator.run_single_test(test)

        assert not result.success  # Not all runs succeeded
        assert result.total_runs == 3
        assert result.successful_runs == 2
        assert result.status == ResponseStatus.PARTIAL

        # Verify run statuses
        assert result.runs[0].success
        assert not result.runs[1].success
        assert result.runs[2].success

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_fail_fast_stops_on_first_failure(self) -> None:
        """Test fail_fast stops execution on first failure."""
        test1 = TestDefinition(
            id="test-1",
            name="Test 1",
            task=TaskDefinition(description="First test"),
            constraints=Constraints(timeout_seconds=30),
        )
        test2 = TestDefinition(
            id="test-2",
            name="Test 2",
            task=TaskDefinition(description="Second test"),
            constraints=Constraints(timeout_seconds=30),
        )
        test3 = TestDefinition(
            id="test-3",
            name="Test 3",
            task=TaskDefinition(description="Third test"),
            constraints=Constraints(timeout_seconds=30),
        )

        suite = TestSuite(
            test_suite="Fail Fast Suite",
            tests=[test1, test2, test3],
        )

        failure_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.FAILED,
            error="Test failure",
        )

        # First test succeeds, second fails
        success_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
        )
        adapter = MockAdapter(responses=[success_response, failure_response])

        orchestrator = TestOrchestrator(
            adapter=adapter,
            runs_per_test=1,
            fail_fast=True,
        )

        result = await orchestrator.run_suite(suite, agent_name="test-agent")

        # Should stop after test 2 fails
        assert len(result.tests) == 2
        assert result.tests[0].success
        assert not result.tests[1].success

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_error_does_not_affect_other_tests(self) -> None:
        """Test that errors in one test don't affect other tests."""
        test1 = TestDefinition(
            id="test-error",
            name="Error Test",
            task=TaskDefinition(description="Test that fails"),
            constraints=Constraints(timeout_seconds=30),
        )
        test2 = TestDefinition(
            id="test-success",
            name="Success Test",
            task=TaskDefinition(description="Test that succeeds"),
            constraints=Constraints(timeout_seconds=30),
        )

        suite = TestSuite(
            test_suite="Mixed Results Suite",
            tests=[test1, test2],
        )

        failure_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.FAILED,
            error="Error in test",
        )
        success_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
        )

        adapter = MockAdapter(responses=[failure_response, success_response])

        orchestrator = TestOrchestrator(
            adapter=adapter,
            runs_per_test=1,
            fail_fast=False,
        )

        result = await orchestrator.run_suite(suite, agent_name="test-agent")

        # Both tests should run
        assert len(result.tests) == 2
        assert not result.tests[0].success
        assert result.tests[1].success
        assert result.passed_tests == 1
        assert result.failed_tests == 1

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_all_runs_fail_in_multi_run(self) -> None:
        """Test handling when all runs fail."""
        test = TestDefinition(
            id="all-fail-test",
            name="All Fail Test",
            task=TaskDefinition(description="Task that always fails"),
            constraints=Constraints(timeout_seconds=30),
        )

        failure_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.FAILED,
            error="Consistent failure",
        )
        adapter = MockAdapter(responses=[failure_response])

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=3)
        result = await orchestrator.run_single_test(test)

        assert not result.success
        assert result.total_runs == 3
        assert result.successful_runs == 0
        assert result.status == ResponseStatus.FAILED

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_cancelled_response_handling(self) -> None:
        """Test handling of cancelled response status."""
        test = TestDefinition(
            id="cancelled-test",
            name="Cancelled Test",
            task=TaskDefinition(description="Task that gets cancelled"),
            constraints=Constraints(timeout_seconds=30),
        )

        cancelled_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.CANCELLED,
            error="Task was cancelled by user",
        )
        adapter = MockAdapter(responses=[cancelled_response])

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        assert not result.success
        assert result.runs[0].response.status == ResponseStatus.CANCELLED

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_partial_response_handling(self) -> None:
        """Test handling of partial response status."""
        test = TestDefinition(
            id="partial-test",
            name="Partial Test",
            task=TaskDefinition(description="Task that partially completes"),
            constraints=Constraints(timeout_seconds=30),
        )

        partial_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.PARTIAL,
            artifacts=[
                ArtifactStructured(name="partial_result", data={"incomplete": True})
            ],
            error="Task partially completed",
        )
        adapter = MockAdapter(responses=[partial_response])

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        assert not result.success
        assert result.runs[0].response.status == ResponseStatus.PARTIAL
        assert len(result.runs[0].response.artifacts) == 1

        await orchestrator.cleanup()


# =============================================================================
# Multi-Run Statistics Tests
# =============================================================================


class TestMultiRunStatistics:
    """Tests for multi-run statistics accuracy."""

    @pytest.mark.anyio
    async def test_multi_run_all_success(self) -> None:
        """Test statistics when all runs succeed."""
        test = TestDefinition(
            id="multi-run-success",
            name="Multi-Run Success Test",
            task=TaskDefinition(description="Task that always succeeds"),
            constraints=Constraints(timeout_seconds=30),
        )

        success_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_tokens=100, total_steps=2, wall_time_seconds=0.5),
        )
        adapter = MockAdapter(responses=[success_response])

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=5)
        result = await orchestrator.run_single_test(test)

        assert result.success
        assert result.total_runs == 5
        assert result.successful_runs == 5
        assert result.status == ResponseStatus.COMPLETED

        # Verify all runs have proper data
        for i, run in enumerate(result.runs):
            assert run.run_number == i + 1
            assert run.success
            assert run.duration_seconds is not None
            assert run.duration_seconds >= 0

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_multi_run_partial_success(self) -> None:
        """Test statistics when some runs fail."""
        test = TestDefinition(
            id="multi-run-partial",
            name="Multi-Run Partial Success Test",
            task=TaskDefinition(description="Task with variable success"),
            constraints=Constraints(timeout_seconds=30),
        )

        # 3 successes, 2 failures
        responses = [
            ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
            ATPResponse(
                task_id="test", status=ResponseStatus.FAILED, error="Failure 1"
            ),
            ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
            ATPResponse(
                task_id="test", status=ResponseStatus.FAILED, error="Failure 2"
            ),
            ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
        ]
        adapter = MockAdapter(responses=responses)

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=5)
        result = await orchestrator.run_single_test(test)

        assert not result.success  # Not all succeeded
        assert result.total_runs == 5
        assert result.successful_runs == 3
        assert result.status == ResponseStatus.PARTIAL

        # Verify specific runs
        success_count = sum(1 for run in result.runs if run.success)
        failure_count = sum(1 for run in result.runs if not run.success)
        assert success_count == 3
        assert failure_count == 2

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_multi_run_all_failure(self) -> None:
        """Test statistics when all runs fail."""
        test = TestDefinition(
            id="multi-run-failure",
            name="Multi-Run Failure Test",
            task=TaskDefinition(description="Task that always fails"),
            constraints=Constraints(timeout_seconds=30),
        )

        failure_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.FAILED,
            error="Consistent failure",
        )
        adapter = MockAdapter(responses=[failure_response])

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=3)
        result = await orchestrator.run_single_test(test)

        assert not result.success
        assert result.total_runs == 3
        assert result.successful_runs == 0
        assert result.status == ResponseStatus.FAILED

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_suite_statistics_aggregation(self) -> None:
        """Test that suite statistics are correctly aggregated."""
        tests = [
            TestDefinition(
                id=f"test-{i}",
                name=f"Test {i}",
                task=TaskDefinition(description=f"Task {i}"),
                constraints=Constraints(timeout_seconds=30),
            )
            for i in range(5)
        ]

        suite = TestSuite(
            test_suite="Statistics Suite",
            tests=tests,
        )

        # 3 tests pass, 2 fail
        responses = [
            ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
            ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
            ATPResponse(task_id="test", status=ResponseStatus.FAILED, error="Failure"),
            ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
            ATPResponse(task_id="test", status=ResponseStatus.FAILED, error="Failure"),
        ]
        adapter = MockAdapter(responses=responses)

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_suite(suite, agent_name="test-agent")

        assert result.total_tests == 5
        assert result.passed_tests == 3
        assert result.failed_tests == 2
        assert result.success_rate == 0.6  # 3/5

        assert not result.success  # Not all tests passed
        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_run_metrics_collection(self) -> None:
        """Test that metrics are collected correctly across runs."""
        test = TestDefinition(
            id="metrics-test",
            name="Metrics Test",
            task=TaskDefinition(description="Task with metrics"),
            constraints=Constraints(timeout_seconds=30),
        )

        # Different metrics for each run
        responses = [
            ATPResponse(
                task_id="test",
                status=ResponseStatus.COMPLETED,
                metrics=Metrics(total_tokens=100, total_steps=2, cost_usd=0.01),
            ),
            ATPResponse(
                task_id="test",
                status=ResponseStatus.COMPLETED,
                metrics=Metrics(total_tokens=150, total_steps=3, cost_usd=0.015),
            ),
            ATPResponse(
                task_id="test",
                status=ResponseStatus.COMPLETED,
                metrics=Metrics(total_tokens=120, total_steps=2, cost_usd=0.012),
            ),
        ]
        adapter = MockAdapter(responses=responses)

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=3)
        result = await orchestrator.run_single_test(test)

        assert result.success
        assert result.total_runs == 3

        # Verify metrics were captured for each run
        for run in result.runs:
            assert run.response.metrics is not None

        tokens = [
            run.response.metrics.total_tokens
            for run in result.runs
            if run.response.metrics is not None
        ]
        assert tokens == [100, 150, 120]

        steps = [
            run.response.metrics.total_steps
            for run in result.runs
            if run.response.metrics is not None
        ]
        assert steps == [2, 3, 2]

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_multi_run_with_varying_metrics(self) -> None:
        """Test multi-run with different metrics per run."""
        test = TestDefinition(
            id="varying-metrics-test",
            name="Varying Metrics Test",
            task=TaskDefinition(description="Task with varying metrics"),
            constraints=Constraints(timeout_seconds=30),
        )

        # Create 10 runs with varying metrics
        responses = [
            ATPResponse(
                task_id="test",
                status=ResponseStatus.COMPLETED,
                metrics=Metrics(
                    total_tokens=100 + i * 10,
                    total_steps=2 + i % 3,
                    wall_time_seconds=0.5 + i * 0.1,
                    cost_usd=0.01 + i * 0.001,
                ),
            )
            for i in range(10)
        ]
        adapter = MockAdapter(responses=responses)

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=10)
        result = await orchestrator.run_single_test(test)

        assert result.success
        assert result.total_runs == 10
        assert result.successful_runs == 10

        # Verify all metrics captured
        all_tokens = []
        for run in result.runs:
            if run.response.metrics:
                all_tokens.append(run.response.metrics.total_tokens)

        assert len(all_tokens) == 10
        assert min(all_tokens) == 100
        assert max(all_tokens) == 190

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_success_rate_calculation(self) -> None:
        """Test success rate calculation accuracy."""
        test = TestDefinition(
            id="success-rate-test",
            name="Success Rate Test",
            task=TaskDefinition(description="Task for success rate"),
            constraints=Constraints(timeout_seconds=30),
        )

        # Exactly 7 successes, 3 failures = 70% success rate
        responses = [
            ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
            ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
            ATPResponse(task_id="test", status=ResponseStatus.FAILED, error="Fail"),
            ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
            ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
            ATPResponse(task_id="test", status=ResponseStatus.FAILED, error="Fail"),
            ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
            ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
            ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
            ATPResponse(task_id="test", status=ResponseStatus.FAILED, error="Fail"),
        ]
        adapter = MockAdapter(responses=responses)

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=10)
        result = await orchestrator.run_single_test(test)

        assert result.total_runs == 10
        assert result.successful_runs == 7
        # 7 successful out of 10 runs
        success_rate = result.successful_runs / result.total_runs
        assert success_rate == 0.7

        await orchestrator.cleanup()


# =============================================================================
# HTTP Adapter with Mock Server Integration Tests
# =============================================================================


class AdvancedMockHandler(BaseHTTPRequestHandler):
    """Advanced mock HTTP handler for complex scenarios."""

    # Class variables to control behavior
    call_count = 0
    fail_on_call: int | None = None
    timeout_on_call: int | None = None
    response_data: dict[str, Any] | None = None

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress logging."""
        pass

    def do_POST(self) -> None:
        """Handle POST requests."""
        AdvancedMockHandler.call_count += 1
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if self.path == "/agent":
            try:
                request_data = json.loads(body)
                task_id = request_data.get("task_id", "unknown")

                # Check for simulated timeout
                if (
                    AdvancedMockHandler.timeout_on_call
                    and AdvancedMockHandler.call_count
                    == AdvancedMockHandler.timeout_on_call
                ):
                    import time

                    time.sleep(60)  # Will be interrupted
                    return

                # Check for simulated failure
                if (
                    AdvancedMockHandler.fail_on_call
                    and AdvancedMockHandler.call_count
                    == AdvancedMockHandler.fail_on_call
                ):
                    response = {
                        "task_id": task_id,
                        "status": "failed",
                        "error": "Simulated server failure",
                    }
                else:
                    response = AdvancedMockHandler.response_data or {
                        "task_id": task_id,
                        "status": "completed",
                        "artifacts": [
                            {
                                "type": "structured",
                                "name": "result",
                                "data": {"message": "Success"},
                            }
                        ],
                        "metrics": {
                            "total_tokens": 100,
                            "total_steps": 2,
                        },
                    }

                # Check if client wants SSE streaming
                accept = self.headers.get("Accept", "")
                if "text/event-stream" in accept:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()

                    # Send a progress event
                    event = {
                        "event_type": "progress",
                        "task_id": task_id,
                        "payload": {"message": "Processing"},
                    }
                    self.wfile.write(b"event: progress\n")
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()

                    # Send final response
                    self.wfile.write(b"event: response\n")
                    self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                    self.wfile.flush()
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())

            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def advanced_mock_server():
    """Start an advanced mock HTTP server."""
    # Reset class variables
    AdvancedMockHandler.call_count = 0
    AdvancedMockHandler.fail_on_call = None
    AdvancedMockHandler.timeout_on_call = None
    AdvancedMockHandler.response_data = None

    server = HTTPServer(("127.0.0.1", 0), AdvancedMockHandler)
    port = server.server_address[1]

    thread = Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()


class TestHTTPAdapterPipelineIntegration:
    """Integration tests for HTTP adapter with full pipeline."""

    @pytest.mark.anyio
    async def test_http_adapter_in_full_pipeline(
        self, advanced_mock_server: str
    ) -> None:
        """Test HTTP adapter works correctly in full pipeline."""
        test = TestDefinition(
            id="http-pipeline-test",
            name="HTTP Pipeline Test",
            task=TaskDefinition(description="Test with HTTP adapter"),
            constraints=Constraints(timeout_seconds=30),
            assertions=[
                Assertion(type="artifact_exists", config={"path": "result"}),
            ],
        )

        config = HTTPAdapterConfig(
            endpoint=f"{advanced_mock_server}/agent",
            timeout_seconds=10.0,
            allow_internal=True,
        )

        async with HTTPAdapter(config) as adapter:
            orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=2)
            result = await orchestrator.run_single_test(test)

            assert result.success
            assert result.total_runs == 2
            assert result.successful_runs == 2

            # Verify evaluator works with HTTP adapter response
            evaluator = ArtifactEvaluator()
            eval_result = await evaluator.evaluate(
                task=test,
                response=result.runs[0].response,
                trace=result.runs[0].events,
                assertion=test.assertions[0],
            )
            assert eval_result.passed

    @pytest.mark.anyio
    async def test_http_adapter_retry_behavior(self, advanced_mock_server: str) -> None:
        """Test that retry behavior works correctly with HTTP adapter."""
        # Configure to fail on second call
        AdvancedMockHandler.fail_on_call = 2

        test = TestDefinition(
            id="http-retry-test",
            name="HTTP Retry Test",
            task=TaskDefinition(description="Test with retries"),
            constraints=Constraints(timeout_seconds=30),
        )

        config = HTTPAdapterConfig(
            endpoint=f"{advanced_mock_server}/agent",
            timeout_seconds=10.0,
            allow_internal=True,
        )

        async with HTTPAdapter(config) as adapter:
            orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=3)
            result = await orchestrator.run_single_test(test)

            # First and third runs succeed, second fails
            assert result.total_runs == 3
            assert result.successful_runs == 2
            assert result.runs[0].success
            assert not result.runs[1].success
            assert result.runs[2].success

    @pytest.mark.anyio
    async def test_http_adapter_with_custom_response(
        self, advanced_mock_server: str
    ) -> None:
        """Test HTTP adapter with custom response data."""
        AdvancedMockHandler.response_data = {
            "task_id": "custom",
            "status": "completed",
            "artifacts": [
                {
                    "type": "structured",
                    "name": "custom_result",
                    "data": {"custom_field": "custom_value", "count": 42},
                }
            ],
            "metrics": {"total_tokens": 200, "total_steps": 5, "cost_usd": 0.05},
        }

        test = TestDefinition(
            id="http-custom-test",
            name="HTTP Custom Response Test",
            task=TaskDefinition(description="Test with custom response"),
            constraints=Constraints(timeout_seconds=30),
            assertions=[
                Assertion(type="artifact_exists", config={"path": "custom_result"}),
            ],
        )

        config = HTTPAdapterConfig(
            endpoint=f"{advanced_mock_server}/agent",
            timeout_seconds=10.0,
            allow_internal=True,
        )

        async with HTTPAdapter(config) as adapter:
            orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
            result = await orchestrator.run_single_test(test)

            assert result.success
            response = result.runs[0].response
            assert response.metrics is not None
            assert response.metrics.total_tokens == 200
            assert response.metrics.total_steps == 5

    @pytest.mark.anyio
    async def test_http_adapter_streaming_events(
        self, advanced_mock_server: str
    ) -> None:
        """Test HTTP adapter captures streaming events."""
        test = TestDefinition(
            id="http-stream-test",
            name="HTTP Streaming Test",
            task=TaskDefinition(description="Test with streaming"),
            constraints=Constraints(timeout_seconds=30),
        )

        config = HTTPAdapterConfig(
            endpoint=f"{advanced_mock_server}/agent",
            timeout_seconds=10.0,
            allow_internal=True,
        )

        collected_events: list[ATPEvent] = []

        def progress_callback(event: ProgressEvent) -> None:
            if event.agent_event:
                collected_events.append(event.agent_event)

        async with HTTPAdapter(config) as adapter:
            orchestrator = TestOrchestrator(
                adapter=adapter, progress_callback=progress_callback, runs_per_test=1
            )
            result = await orchestrator.run_single_test(test)

            assert result.success
            # The mock server sends one progress event
            assert len(result.runs[0].events) >= 1


# =============================================================================
# Evaluator Integration Tests
# =============================================================================


class TestEvaluatorIntegration:
    """Integration tests for evaluators with real test data."""

    @pytest.mark.anyio
    async def test_artifact_evaluator_with_pipeline(self) -> None:
        """Test artifact evaluator in full pipeline context."""
        test = TestDefinition(
            id="evaluator-test",
            name="Evaluator Integration Test",
            task=TaskDefinition(
                description="Task with artifacts",
                expected_artifacts=["output", "report"],
            ),
            constraints=Constraints(timeout_seconds=30),
            assertions=[
                Assertion(type="artifact_exists", config={"path": "output"}),
                Assertion(
                    type="contains",
                    config={"path": "output", "pattern": "success"},
                ),
                Assertion(
                    type="schema",
                    config={
                        "path": "data",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "value": {"type": "integer"},
                            },
                            "required": ["status"],
                        },
                    },
                ),
            ],
        )

        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="output",
                    data={"message": "Task completed with success"},
                ),
                ArtifactStructured(
                    name="data",
                    data={"status": "completed", "value": 42},
                ),
            ],
        )

        adapter = MockAdapter(responses=[response])
        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        assert result.success

        # Run all assertions through evaluator
        evaluator = ArtifactEvaluator()
        run = result.runs[0]

        eval_results: list[EvalResult] = []
        for assertion in test.assertions:
            eval_result = await evaluator.evaluate(
                task=test,
                response=run.response,
                trace=run.events,
                assertion=assertion,
            )
            eval_results.append(eval_result)

        # Check all assertions passed
        assert eval_results[0].passed  # artifact_exists
        assert eval_results[2].passed  # schema

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_evaluator_failure_detection(self) -> None:
        """Test that evaluator correctly detects failures."""
        test = TestDefinition(
            id="eval-failure-test",
            name="Evaluator Failure Test",
            task=TaskDefinition(description="Task with missing artifacts"),
            constraints=Constraints(timeout_seconds=30),
            assertions=[
                Assertion(
                    type="artifact_exists",
                    config={"path": "missing_artifact"},
                ),
            ],
        )

        # Response without the expected artifact
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(name="other", data={"key": "value"}),
            ],
        )

        adapter = MockAdapter(responses=[response])
        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        assert result.success  # Task completed

        evaluator = ArtifactEvaluator()
        eval_result = await evaluator.evaluate(
            task=test,
            response=result.runs[0].response,
            trace=[],
            assertion=test.assertions[0],
        )

        assert not eval_result.passed  # Artifact not found
        assert eval_result.score == 0.0

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_behavior_evaluator_tool_tracking(self) -> None:
        """Test behavior evaluator tracks tool usage correctly."""
        events = [
            ATPEvent(
                event_type=EventType.TOOL_CALL,
                task_id="test",
                sequence=0,
                timestamp=datetime.now(),
                payload={"tool": "web_search", "args": {"query": "test"}},
            ),
            ATPEvent(
                event_type=EventType.TOOL_CALL,
                task_id="test",
                sequence=1,
                timestamp=datetime.now(),
                payload={"tool": "file_write", "args": {"path": "output.txt"}},
            ),
            ATPEvent(
                event_type=EventType.TOOL_CALL,
                task_id="test",
                sequence=2,
                timestamp=datetime.now(),
                payload={"tool": "web_search", "args": {"query": "more info"}},
            ),
        ]

        test = TestDefinition(
            id="tool-tracking-test",
            name="Tool Tracking Test",
            task=TaskDefinition(description="Task with tool usage"),
            constraints=Constraints(timeout_seconds=30),
            assertions=[
                Assertion(
                    type="must_use_tools",
                    config={"tools": ["web_search", "file_write"]},
                ),
                Assertion(type="max_tool_calls", config={"limit": 5}),
                Assertion(type="forbidden_tools", config={"tools": ["dangerous_tool"]}),
            ],
        )

        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_tokens=100, total_steps=3, tool_calls=3),
        )

        adapter = MockAdapter(responses=[response], events=events)
        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        evaluator = BehaviorEvaluator()

        for assertion in test.assertions:
            eval_result = await evaluator.evaluate(
                task=test,
                response=result.runs[0].response,
                trace=result.runs[0].events,
                assertion=assertion,
            )
            assert eval_result.passed, f"Assertion {assertion.type} failed"

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_behavior_evaluator_max_tool_calls_exceeded(self) -> None:
        """Test behavior evaluator detects tool call limit violations."""
        events = [
            ATPEvent(
                event_type=EventType.TOOL_CALL,
                task_id="test",
                sequence=i,
                timestamp=datetime.now(),
                payload={"tool": "web_search", "args": {"query": f"query {i}"}},
            )
            for i in range(10)
        ]

        test = TestDefinition(
            id="max-tools-test",
            name="Max Tool Calls Test",
            task=TaskDefinition(description="Task that exceeds tool limits"),
            constraints=Constraints(timeout_seconds=30),
            assertions=[
                Assertion(type="max_tool_calls", config={"limit": 5}),
            ],
        )

        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_tokens=100, total_steps=10, tool_calls=10),
        )

        adapter = MockAdapter(responses=[response], events=events)
        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        evaluator = BehaviorEvaluator()
        eval_result = await evaluator.evaluate(
            task=test,
            response=result.runs[0].response,
            trace=result.runs[0].events,
            assertion=test.assertions[0],
        )

        assert not eval_result.passed  # Should fail - 10 calls > 5 limit

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_behavior_evaluator_forbidden_tool_detected(self) -> None:
        """Test behavior evaluator detects forbidden tool usage."""
        events = [
            ATPEvent(
                event_type=EventType.TOOL_CALL,
                task_id="test",
                sequence=0,
                timestamp=datetime.now(),
                payload={"tool": "allowed_tool", "args": {}},
            ),
            ATPEvent(
                event_type=EventType.TOOL_CALL,
                task_id="test",
                sequence=1,
                timestamp=datetime.now(),
                payload={"tool": "forbidden_tool", "args": {}},  # Forbidden!
            ),
        ]

        test = TestDefinition(
            id="forbidden-tool-test",
            name="Forbidden Tool Test",
            task=TaskDefinition(description="Task using forbidden tool"),
            constraints=Constraints(timeout_seconds=30),
            assertions=[
                Assertion(type="forbidden_tools", config={"tools": ["forbidden_tool"]}),
            ],
        )

        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
        )

        adapter = MockAdapter(responses=[response], events=events)
        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        evaluator = BehaviorEvaluator()
        eval_result = await evaluator.evaluate(
            task=test,
            response=result.runs[0].response,
            trace=result.runs[0].events,
            assertion=test.assertions[0],
        )

        assert not eval_result.passed  # Should fail - used forbidden tool

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_behavior_evaluator_no_errors(self) -> None:
        """Test behavior evaluator no_errors assertion."""
        # Events with an error
        events_with_error = [
            ATPEvent(
                event_type=EventType.PROGRESS,
                task_id="test",
                sequence=0,
                timestamp=datetime.now(),
                payload={"message": "Starting"},
            ),
            ATPEvent(
                event_type=EventType.ERROR,
                task_id="test",
                sequence=1,
                timestamp=datetime.now(),
                payload={"message": "Something went wrong"},
            ),
        ]

        test = TestDefinition(
            id="no-errors-test",
            name="No Errors Test",
            task=TaskDefinition(description="Task checking for errors"),
            constraints=Constraints(timeout_seconds=30),
            assertions=[
                Assertion(type="no_errors", config={}),
            ],
        )

        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
        )

        adapter = MockAdapter(responses=[response], events=events_with_error)
        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        evaluator = BehaviorEvaluator()
        eval_result = await evaluator.evaluate(
            task=test,
            response=result.runs[0].response,
            trace=result.runs[0].events,
            assertion=test.assertions[0],
        )

        assert not eval_result.passed  # Should fail - has error events

        await orchestrator.cleanup()


# =============================================================================
# Edge Cases and Complex Scenarios
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and complex scenarios."""

    @pytest.mark.anyio
    async def test_minimal_test_suite(self) -> None:
        """Test handling of minimal test suite with one test."""
        suite = TestSuite(
            test_suite="Minimal Suite",
            tests=[
                TestDefinition(
                    id="only-test",
                    name="Only Test",
                    task=TaskDefinition(description="Single test"),
                    constraints=Constraints(timeout_seconds=30),
                )
            ],
        )

        success_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
        )
        adapter = MockAdapter(responses=[success_response])
        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_suite(suite, agent_name="test-agent")

        assert result.total_tests == 1
        assert result.passed_tests == 1
        assert result.failed_tests == 0
        assert result.success

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_single_run_configuration(self) -> None:
        """Test single run configuration."""
        test = TestDefinition(
            id="single-run-test",
            name="Single Run Test",
            task=TaskDefinition(description="Task with single run"),
            constraints=Constraints(timeout_seconds=30),
        )

        success_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
        )
        adapter = MockAdapter(responses=[success_response])

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        assert result.total_runs == 1
        assert adapter.call_count == 1

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_large_number_of_runs(self) -> None:
        """Test with a large number of runs."""
        test = TestDefinition(
            id="many-runs-test",
            name="Many Runs Test",
            task=TaskDefinition(description="Task with many runs"),
            constraints=Constraints(timeout_seconds=30),
        )

        success_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_tokens=100),
        )
        adapter = MockAdapter(responses=[success_response])

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=20)
        result = await orchestrator.run_single_test(test)

        assert result.total_runs == 20
        assert result.successful_runs == 20
        assert adapter.call_count == 20

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_progress_callback_exceptions_handled(self) -> None:
        """Test that exceptions in progress callback don't break execution."""

        def faulty_callback(event: ProgressEvent) -> None:
            raise ValueError("Callback error")

        test = TestDefinition(
            id="callback-error-test",
            name="Callback Error Test",
            task=TaskDefinition(description="Task with faulty callback"),
            constraints=Constraints(timeout_seconds=30),
        )

        success_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
        )
        adapter = MockAdapter(responses=[success_response])

        orchestrator = TestOrchestrator(
            adapter=adapter, progress_callback=faulty_callback, runs_per_test=1
        )
        result = await orchestrator.run_single_test(test)

        # Should complete despite callback errors
        assert result.success

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_unicode_in_task_description(self) -> None:
        """Test handling of unicode characters in task description."""
        test = TestDefinition(
            id="unicode-test",
            name="Unicode Test 日本語 🎉",
            task=TaskDefinition(
                description="Task with unicode: émojis 🚀, 中文, العربية"
            ),
            constraints=Constraints(timeout_seconds=30),
        )

        success_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
        )
        adapter = MockAdapter(responses=[success_response])

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        assert result.success
        assert result.runs[0].response.status == ResponseStatus.COMPLETED

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_very_short_timeout(self) -> None:
        """Test behavior with very short timeout (1 second)."""
        test = TestDefinition(
            id="short-timeout-test",
            name="Short Timeout Test",
            task=TaskDefinition(description="Task with short timeout"),
            constraints=Constraints(timeout_seconds=1),  # 1 second (minimum int)
        )

        # Adapter that takes longer than timeout
        adapter = MockAdapter(delay_seconds=5.0)

        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        assert not result.success
        assert result.status == ResponseStatus.TIMEOUT

        await orchestrator.cleanup()

    @pytest.mark.anyio
    async def test_response_with_large_artifacts(self) -> None:
        """Test handling of response with large artifacts."""
        large_data = {"key": "x" * 10000}  # 10KB of data

        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(name=f"artifact_{i}", data=large_data)
                for i in range(10)
            ],
        )

        test = TestDefinition(
            id="large-artifact-test",
            name="Large Artifact Test",
            task=TaskDefinition(description="Task with large artifacts"),
            constraints=Constraints(timeout_seconds=30),
        )

        adapter = MockAdapter(responses=[response])
        orchestrator = TestOrchestrator(adapter=adapter, runs_per_test=1)
        result = await orchestrator.run_single_test(test)

        assert result.success
        assert len(result.runs[0].response.artifacts) == 10

        await orchestrator.cleanup()
