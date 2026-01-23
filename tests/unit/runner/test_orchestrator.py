"""Tests for test orchestrator."""

from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from atp.adapters.base import AgentAdapter
from atp.adapters.exceptions import AdapterError, AdapterTimeoutError
from atp.loader.models import (
    Constraints,
    TaskDefinition,
    TestDefaults,
    TestDefinition,
    TestSuite,
)
from atp.protocol import ATPEvent, ATPRequest, ATPResponse, EventType, ResponseStatus
from atp.runner.models import (
    ProgressEvent,
    ProgressEventType,
    SandboxConfig,
)
from atp.runner.orchestrator import TestOrchestrator, run_suite, run_test


class MockAdapter(AgentAdapter):
    """Mock adapter for testing."""

    def __init__(
        self,
        response: ATPResponse | None = None,
        events: list[ATPEvent] | None = None,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        super().__init__()
        self._response = response or ATPResponse(
            task_id="test-task",
            status=ResponseStatus.COMPLETED,
        )
        self._events = events or []
        self._error = error
        self._delay = delay

    @property
    def adapter_type(self) -> str:
        return "mock"

    async def execute(self, request: ATPRequest) -> ATPResponse:
        import asyncio

        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        # Copy response with correct task_id
        return ATPResponse(
            task_id=request.task_id,
            status=self._response.status,
            artifacts=self._response.artifacts,
            metrics=self._response.metrics,
            error=self._response.error,
        )

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        import asyncio

        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        for event in self._events:
            yield event
        yield ATPResponse(
            task_id=request.task_id,
            status=self._response.status,
            artifacts=self._response.artifacts,
            metrics=self._response.metrics,
            error=self._response.error,
        )


@pytest.fixture
def test_definition() -> TestDefinition:
    """Create a test definition."""
    return TestDefinition(
        id="test-001",
        name="Sample Test",
        task=TaskDefinition(description="Test task"),
        constraints=Constraints(timeout_seconds=10),
    )


@pytest.fixture
def test_suite(test_definition: TestDefinition) -> TestSuite:
    """Create a test suite."""
    test2 = TestDefinition(
        id="test-002",
        name="Second Test",
        task=TaskDefinition(description="Second task"),
        constraints=Constraints(timeout_seconds=10),
    )
    return TestSuite(
        test_suite="test-suite",
        tests=[test_definition, test2],
        defaults=TestDefaults(runs_per_test=1),
    )


@pytest.fixture
def success_response() -> ATPResponse:
    """Create a successful response."""
    return ATPResponse(
        task_id="test-task",
        status=ResponseStatus.COMPLETED,
    )


@pytest.fixture
def failed_response() -> ATPResponse:
    """Create a failed response."""
    return ATPResponse(
        task_id="test-task",
        status=ResponseStatus.FAILED,
        error="Agent failed",
    )


class TestTestOrchestrator:
    """Tests for TestOrchestrator."""

    def test_init_with_defaults(self, success_response: ATPResponse) -> None:
        """Orchestrator initializes with defaults."""
        adapter = MockAdapter(success_response)
        orchestrator = TestOrchestrator(adapter=adapter)

        assert orchestrator.adapter is adapter
        assert orchestrator.runs_per_test == 1
        assert orchestrator.fail_fast is False
        assert orchestrator.progress_callback is None

    def test_init_with_options(self, success_response: ATPResponse) -> None:
        """Orchestrator initializes with options."""
        adapter = MockAdapter(success_response)
        callback = MagicMock()

        orchestrator = TestOrchestrator(
            adapter=adapter,
            progress_callback=callback,
            runs_per_test=3,
            fail_fast=True,
        )

        assert orchestrator.runs_per_test == 3
        assert orchestrator.fail_fast is True
        assert orchestrator.progress_callback is callback


class TestRunSingleTest:
    """Tests for running single tests."""

    @pytest.mark.anyio
    async def test_successful_execution(
        self,
        test_definition: TestDefinition,
        success_response: ATPResponse,
    ) -> None:
        """Test successful execution."""
        adapter = MockAdapter(success_response)

        async with TestOrchestrator(adapter=adapter) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.success is True
        assert result.status == ResponseStatus.COMPLETED
        assert result.total_runs == 1
        assert result.successful_runs == 1
        assert result.test.id == "test-001"

    @pytest.mark.anyio
    async def test_failed_execution(
        self,
        test_definition: TestDefinition,
        failed_response: ATPResponse,
    ) -> None:
        """Test failed execution."""
        adapter = MockAdapter(failed_response)

        async with TestOrchestrator(adapter=adapter) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.success is False
        assert result.status == ResponseStatus.FAILED

    @pytest.mark.anyio
    async def test_multiple_runs(
        self,
        test_definition: TestDefinition,
        success_response: ATPResponse,
    ) -> None:
        """Test with multiple runs."""
        adapter = MockAdapter(success_response)

        async with TestOrchestrator(adapter=adapter, runs_per_test=3) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.total_runs == 3
        assert result.successful_runs == 3

    @pytest.mark.anyio
    async def test_runs_override(
        self,
        test_definition: TestDefinition,
        success_response: ATPResponse,
    ) -> None:
        """Test runs parameter override."""
        adapter = MockAdapter(success_response)

        async with TestOrchestrator(adapter=adapter, runs_per_test=1) as orchestrator:
            result = await orchestrator.run_single_test(test_definition, runs=5)

        assert result.total_runs == 5

    @pytest.mark.anyio
    async def test_adapter_error(
        self,
        test_definition: TestDefinition,
    ) -> None:
        """Test adapter error handling."""
        adapter = MockAdapter(error=AdapterError("Connection failed"))

        async with TestOrchestrator(adapter=adapter) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.success is False
        assert result.status == ResponseStatus.FAILED
        assert len(result.runs) == 1
        assert result.runs[0].error is not None

    @pytest.mark.anyio
    async def test_events_collected(
        self,
        test_definition: TestDefinition,
        success_response: ATPResponse,
    ) -> None:
        """Test that events are collected."""
        events = [
            ATPEvent(
                task_id="test-task",
                sequence=0,
                event_type=EventType.PROGRESS,
                payload={"message": "Starting"},
            ),
            ATPEvent(
                task_id="test-task",
                sequence=1,
                event_type=EventType.TOOL_CALL,
                payload={"tool": "python"},
            ),
        ]
        adapter = MockAdapter(success_response, events=events)

        async with TestOrchestrator(adapter=adapter) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert len(result.runs[0].events) == 2


class TestProgressCallback:
    """Tests for progress callback."""

    @pytest.mark.anyio
    async def test_progress_events_emitted(
        self,
        test_definition: TestDefinition,
        success_response: ATPResponse,
    ) -> None:
        """Test that progress events are emitted."""
        adapter = MockAdapter(success_response)
        events: list[ProgressEvent] = []

        def callback(event: ProgressEvent) -> None:
            events.append(event)

        async with TestOrchestrator(
            adapter=adapter, progress_callback=callback
        ) as orchestrator:
            await orchestrator.run_single_test(test_definition)

        event_types = [e.event_type for e in events]
        assert ProgressEventType.TEST_STARTED in event_types
        assert ProgressEventType.RUN_STARTED in event_types
        assert ProgressEventType.RUN_COMPLETED in event_types
        assert ProgressEventType.TEST_COMPLETED in event_types

    @pytest.mark.anyio
    async def test_callback_exception_handled(
        self,
        test_definition: TestDefinition,
        success_response: ATPResponse,
    ) -> None:
        """Test that callback exceptions are handled."""
        adapter = MockAdapter(success_response)

        def bad_callback(event: ProgressEvent) -> None:
            raise RuntimeError("Callback error")

        async with TestOrchestrator(
            adapter=adapter, progress_callback=bad_callback
        ) as orchestrator:
            # Should not raise
            result = await orchestrator.run_single_test(test_definition)

        assert result.success is True


class TestTimeoutEnforcement:
    """Tests for timeout enforcement."""

    @pytest.mark.anyio
    async def test_timeout_response(
        self,
        test_definition: TestDefinition,
    ) -> None:
        """Test timeout creates timeout response."""
        # Create adapter that takes longer than timeout
        adapter = MockAdapter(
            response=ATPResponse(
                task_id="test",
                status=ResponseStatus.COMPLETED,
            ),
            delay=2.0,  # Will timeout
        )

        # Set very short timeout
        test_definition.constraints.timeout_seconds = 0.1

        async with TestOrchestrator(adapter=adapter) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.status == ResponseStatus.TIMEOUT
        assert result.runs[0].response.status == ResponseStatus.TIMEOUT

    @pytest.mark.anyio
    async def test_adapter_timeout_error(
        self,
        test_definition: TestDefinition,
    ) -> None:
        """Test adapter timeout error handling."""
        adapter = MockAdapter(
            error=AdapterTimeoutError("Adapter timeout", timeout_seconds=30.0)
        )

        async with TestOrchestrator(adapter=adapter) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.status == ResponseStatus.TIMEOUT


class TestSuiteExecution:
    """Tests for suite execution."""

    @pytest.mark.anyio
    async def test_run_suite(
        self,
        test_suite: TestSuite,
        success_response: ATPResponse,
    ) -> None:
        """Test running a complete suite."""
        adapter = MockAdapter(success_response)

        async with TestOrchestrator(adapter=adapter) as orchestrator:
            result = await orchestrator.run_suite(test_suite, "test-agent")

        assert result.suite_name == "test-suite"
        assert result.agent_name == "test-agent"
        assert result.total_tests == 2
        assert result.passed_tests == 2
        assert result.success is True

    @pytest.mark.anyio
    async def test_suite_with_failures(
        self,
        test_suite: TestSuite,
    ) -> None:
        """Test suite with some failures."""
        # Alternate between success and failure
        call_count = 0

        class AlternatingAdapter(MockAdapter):
            async def stream_events(
                self, request: ATPRequest
            ) -> AsyncIterator[ATPEvent | ATPResponse]:
                nonlocal call_count
                call_count += 1
                status = (
                    ResponseStatus.COMPLETED
                    if call_count % 2 == 1
                    else ResponseStatus.FAILED
                )
                yield ATPResponse(task_id=request.task_id, status=status)

        adapter = AlternatingAdapter()

        async with TestOrchestrator(adapter=adapter) as orchestrator:
            result = await orchestrator.run_suite(test_suite, "test-agent")

        assert result.total_tests == 2
        assert result.passed_tests == 1
        assert result.failed_tests == 1
        assert result.success is False
        assert result.success_rate == 0.5

    @pytest.mark.anyio
    async def test_suite_fail_fast(
        self,
        test_suite: TestSuite,
        failed_response: ATPResponse,
    ) -> None:
        """Test fail_fast stops suite on first failure."""
        adapter = MockAdapter(failed_response)

        async with TestOrchestrator(adapter=adapter, fail_fast=True) as orchestrator:
            result = await orchestrator.run_suite(test_suite, "test-agent")

        # Only first test should run
        assert len(result.tests) == 1

    @pytest.mark.anyio
    async def test_suite_progress_events(
        self,
        test_suite: TestSuite,
        success_response: ATPResponse,
    ) -> None:
        """Test suite progress events."""
        adapter = MockAdapter(success_response)
        events: list[ProgressEvent] = []

        async with TestOrchestrator(
            adapter=adapter, progress_callback=events.append
        ) as orchestrator:
            await orchestrator.run_suite(test_suite, "test-agent")

        event_types = [e.event_type for e in events]
        assert ProgressEventType.SUITE_STARTED in event_types
        assert ProgressEventType.SUITE_COMPLETED in event_types
        assert event_types.count(ProgressEventType.TEST_STARTED) == 2
        assert event_types.count(ProgressEventType.TEST_COMPLETED) == 2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.anyio
    async def test_run_test_function(
        self,
        test_definition: TestDefinition,
        success_response: ATPResponse,
    ) -> None:
        """Test run_test convenience function."""
        adapter = MockAdapter(success_response)
        result = await run_test(adapter, test_definition)
        assert result.success is True

    @pytest.mark.anyio
    async def test_run_test_with_callback(
        self,
        test_definition: TestDefinition,
        success_response: ATPResponse,
    ) -> None:
        """Test run_test with callback."""
        adapter = MockAdapter(success_response)
        events: list[ProgressEvent] = []

        result = await run_test(
            adapter, test_definition, progress_callback=events.append, runs=2
        )

        assert result.total_runs == 2
        assert len(events) > 0

    @pytest.mark.anyio
    async def test_run_suite_function(
        self,
        test_suite: TestSuite,
        success_response: ATPResponse,
    ) -> None:
        """Test run_suite convenience function."""
        adapter = MockAdapter(success_response)
        result = await run_suite(adapter, test_suite, "test-agent")
        assert result.success is True
        assert result.total_tests == 2


class TestSandboxIntegration:
    """Tests for sandbox integration."""

    @pytest.mark.anyio
    async def test_sandbox_disabled_by_default(
        self,
        test_definition: TestDefinition,
        success_response: ATPResponse,
    ) -> None:
        """Sandbox is disabled by default."""
        adapter = MockAdapter(success_response)

        async with TestOrchestrator(adapter=adapter) as orchestrator:
            assert orchestrator.sandbox_config.enabled is False
            await orchestrator.run_single_test(test_definition)
            # No sandbox manager created
            assert orchestrator._sandbox_manager is None

    @pytest.mark.anyio
    async def test_sandbox_enabled(
        self,
        test_definition: TestDefinition,
        success_response: ATPResponse,
        tmp_path,
    ) -> None:
        """Test with sandbox enabled."""
        adapter = MockAdapter(success_response)
        sandbox_config = SandboxConfig(enabled=True)

        with patch("atp.runner.orchestrator.SandboxManager") as MockSandboxManager:
            mock_manager = MagicMock()
            mock_manager.create.return_value = "sandbox-123"
            mock_manager.get_workspace.return_value = tmp_path / "workspace"
            MockSandboxManager.return_value = mock_manager

            async with TestOrchestrator(
                adapter=adapter, sandbox_config=sandbox_config
            ) as orchestrator:
                await orchestrator.run_single_test(test_definition)

            mock_manager.create.assert_called_once()
            mock_manager.cleanup.assert_called_once_with("sandbox-123")


class TestRequestCreation:
    """Tests for ATP request creation."""

    @pytest.mark.anyio
    async def test_request_has_correct_fields(
        self,
        test_definition: TestDefinition,
        success_response: ATPResponse,
    ) -> None:
        """Test that request is created correctly."""
        captured_request: ATPRequest | None = None

        class CapturingAdapter(MockAdapter):
            async def stream_events(
                self, request: ATPRequest
            ) -> AsyncIterator[ATPEvent | ATPResponse]:
                nonlocal captured_request
                captured_request = request
                yield ATPResponse(
                    task_id=request.task_id, status=ResponseStatus.COMPLETED
                )

        adapter = CapturingAdapter()

        async with TestOrchestrator(adapter=adapter) as orchestrator:
            await orchestrator.run_single_test(test_definition)

        assert captured_request is not None
        assert captured_request.task.description == "Test task"
        assert captured_request.constraints.get("timeout_seconds") == 10
        assert captured_request.metadata is not None
        assert captured_request.metadata.get("test_id") == "test-001"
        assert captured_request.metadata.get("test_name") == "Sample Test"


class TestParallelTestExecution:
    """Tests for parallel test execution."""

    @pytest.fixture
    def large_test_suite(self) -> TestSuite:
        """Create a test suite with multiple tests."""
        tests = [
            TestDefinition(
                id=f"test-{i:03d}",
                name=f"Test {i}",
                task=TaskDefinition(description=f"Task {i}"),
                constraints=Constraints(timeout_seconds=10),
            )
            for i in range(1, 6)
        ]
        return TestSuite(
            test_suite="parallel-test-suite",
            tests=tests,
            defaults=TestDefaults(runs_per_test=1),
        )

    @pytest.mark.anyio
    async def test_parallel_tests_initialization(
        self,
        success_response: ATPResponse,
    ) -> None:
        """Orchestrator initializes with parallel tests options."""
        adapter = MockAdapter(success_response)
        orchestrator = TestOrchestrator(
            adapter=adapter,
            parallel_tests=True,
            max_parallel_tests=3,
        )

        assert orchestrator.parallel_tests is True
        assert orchestrator.max_parallel_tests == 3
        assert orchestrator._tests_semaphore is None  # Created on first use

    @pytest.mark.anyio
    async def test_parallel_suite_execution(
        self,
        large_test_suite: TestSuite,
        success_response: ATPResponse,
    ) -> None:
        """Test running suite with parallel tests."""
        adapter = MockAdapter(success_response)

        async with TestOrchestrator(
            adapter=adapter,
            parallel_tests=True,
            max_parallel_tests=3,
        ) as orchestrator:
            result = await orchestrator.run_suite(large_test_suite, "test-agent")

        assert result.suite_name == "parallel-test-suite"
        assert result.agent_name == "test-agent"
        assert result.total_tests == 5
        assert result.passed_tests == 5
        assert result.success is True

    @pytest.mark.anyio
    async def test_parallel_tests_with_delay_verifies_concurrency(
        self,
        large_test_suite: TestSuite,
    ) -> None:
        """Test that parallel execution actually runs concurrently."""
        import time

        execution_times: list[tuple[str, float, float]] = []

        class TimingAdapter(MockAdapter):
            async def stream_events(
                self, request: ATPRequest
            ) -> AsyncIterator[ATPEvent | ATPResponse]:
                import asyncio

                start = time.monotonic()
                await asyncio.sleep(0.1)  # Simulate work
                end = time.monotonic()
                test_id = str(
                    request.metadata.get("test_id", "?") if request.metadata else "?"
                )
                execution_times.append((test_id, start, end))
                yield ATPResponse(
                    task_id=request.task_id, status=ResponseStatus.COMPLETED
                )

        adapter = TimingAdapter()

        start_time = time.monotonic()
        async with TestOrchestrator(
            adapter=adapter,
            parallel_tests=True,
            max_parallel_tests=5,  # All 5 tests can run together
        ) as orchestrator:
            result = await orchestrator.run_suite(large_test_suite, "test-agent")
        total_time = time.monotonic() - start_time

        assert result.success is True
        assert len(execution_times) == 5

        # If sequential, this would take ~0.5s (5 * 0.1s)
        # If parallel, it should be close to 0.1s (all run together)
        # Allow some margin for overhead
        assert total_time < 0.4, f"Expected parallel execution, took {total_time}s"

    @pytest.mark.anyio
    async def test_parallel_tests_respects_semaphore_limit(
        self,
        large_test_suite: TestSuite,
    ) -> None:
        """Test that semaphore limits concurrent tests."""

        max_concurrent = 0
        current_concurrent = 0

        class ConcurrencyTrackingAdapter(MockAdapter):
            async def stream_events(
                self, request: ATPRequest
            ) -> AsyncIterator[ATPEvent | ATPResponse]:
                import asyncio

                nonlocal max_concurrent, current_concurrent
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
                await asyncio.sleep(0.05)  # Simulate work
                current_concurrent -= 1
                yield ATPResponse(
                    task_id=request.task_id, status=ResponseStatus.COMPLETED
                )

        adapter = ConcurrencyTrackingAdapter()

        async with TestOrchestrator(
            adapter=adapter,
            parallel_tests=True,
            max_parallel_tests=2,  # Limit to 2 concurrent
        ) as orchestrator:
            result = await orchestrator.run_suite(large_test_suite, "test-agent")

        assert result.success is True
        assert max_concurrent <= 2, f"Expected max 2 concurrent, got {max_concurrent}"

    @pytest.mark.anyio
    async def test_parallel_tests_result_order_preserved(
        self,
        large_test_suite: TestSuite,
    ) -> None:
        """Test that results are returned in original test order."""
        import random

        class RandomDelayAdapter(MockAdapter):
            async def stream_events(
                self, request: ATPRequest
            ) -> AsyncIterator[ATPEvent | ATPResponse]:
                import asyncio

                # Random delay to ensure out-of-order completion
                await asyncio.sleep(random.uniform(0.01, 0.05))
                yield ATPResponse(
                    task_id=request.task_id, status=ResponseStatus.COMPLETED
                )

        adapter = RandomDelayAdapter()

        async with TestOrchestrator(
            adapter=adapter,
            parallel_tests=True,
            max_parallel_tests=5,
        ) as orchestrator:
            result = await orchestrator.run_suite(large_test_suite, "test-agent")

        # Verify order is preserved
        expected_ids = ["test-001", "test-002", "test-003", "test-004", "test-005"]
        actual_ids = [t.test.id for t in result.tests]
        assert actual_ids == expected_ids

    @pytest.mark.anyio
    async def test_parallel_tests_with_some_failures(
        self,
        large_test_suite: TestSuite,
    ) -> None:
        """Test parallel execution with some test failures."""
        call_count = 0

        class AlternatingAdapter(MockAdapter):
            async def stream_events(
                self, request: ATPRequest
            ) -> AsyncIterator[ATPEvent | ATPResponse]:
                nonlocal call_count
                call_count += 1
                # Fail every second test
                status = (
                    ResponseStatus.COMPLETED
                    if call_count % 2 == 1
                    else ResponseStatus.FAILED
                )
                yield ATPResponse(task_id=request.task_id, status=status)

        adapter = AlternatingAdapter()

        async with TestOrchestrator(
            adapter=adapter,
            parallel_tests=True,
            max_parallel_tests=3,
        ) as orchestrator:
            result = await orchestrator.run_suite(large_test_suite, "test-agent")

        assert result.total_tests == 5
        assert result.passed_tests == 3  # Tests 1, 3, 5 pass
        assert result.failed_tests == 2  # Tests 2, 4 fail
        assert result.success is False

    @pytest.mark.anyio
    async def test_parallel_tests_disabled_with_fail_fast(
        self,
        large_test_suite: TestSuite,
        failed_response: ATPResponse,
    ) -> None:
        """Test that fail_fast disables parallel execution."""
        adapter = MockAdapter(failed_response)

        async with TestOrchestrator(
            adapter=adapter,
            parallel_tests=True,  # Enabled
            max_parallel_tests=3,
            fail_fast=True,  # But fail_fast takes precedence
        ) as orchestrator:
            result = await orchestrator.run_suite(large_test_suite, "test-agent")

        # With fail_fast, should stop after first failure (sequential)
        assert len(result.tests) == 1

    @pytest.mark.anyio
    async def test_parallel_tests_progress_events(
        self,
        large_test_suite: TestSuite,
        success_response: ATPResponse,
    ) -> None:
        """Test progress events are emitted correctly for parallel tests."""
        adapter = MockAdapter(success_response)
        events: list[ProgressEvent] = []

        async with TestOrchestrator(
            adapter=adapter,
            parallel_tests=True,
            max_parallel_tests=3,
            progress_callback=events.append,
        ) as orchestrator:
            await orchestrator.run_suite(large_test_suite, "test-agent")

        event_types = [e.event_type for e in events]

        # Suite events
        assert ProgressEventType.SUITE_STARTED in event_types
        assert ProgressEventType.SUITE_COMPLETED in event_types

        # Each test should have started and completed events
        assert event_types.count(ProgressEventType.TEST_STARTED) == 5
        assert event_types.count(ProgressEventType.TEST_COMPLETED) == 5

        # Each test should have run started and completed events
        assert event_types.count(ProgressEventType.RUN_STARTED) == 5
        assert event_types.count(ProgressEventType.RUN_COMPLETED) == 5

    @pytest.mark.anyio
    async def test_parallel_tests_with_exception(
        self,
    ) -> None:
        """Test parallel execution handles exceptions gracefully."""
        tests = [
            TestDefinition(
                id="test-001",
                name="Test 1",
                task=TaskDefinition(description="Task 1"),
                constraints=Constraints(timeout_seconds=10),
            ),
            TestDefinition(
                id="test-002",
                name="Test 2",
                task=TaskDefinition(description="Task 2"),
                constraints=Constraints(timeout_seconds=10),
            ),
        ]
        suite = TestSuite(
            test_suite="exception-suite",
            tests=tests,
            defaults=TestDefaults(runs_per_test=1),
        )

        call_count = 0

        class ExceptionAdapter(MockAdapter):
            async def stream_events(
                self, request: ATPRequest
            ) -> AsyncIterator[ATPEvent | ATPResponse]:
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("Simulated error")
                yield ATPResponse(
                    task_id=request.task_id, status=ResponseStatus.COMPLETED
                )

        adapter = ExceptionAdapter()

        async with TestOrchestrator(
            adapter=adapter,
            parallel_tests=True,
            max_parallel_tests=2,
        ) as orchestrator:
            result = await orchestrator.run_suite(suite, "test-agent")

        assert result.total_tests == 2
        # First test should succeed, second should have error
        assert result.tests[0].success is True
        assert result.tests[1].success is False

    @pytest.mark.anyio
    async def test_single_test_no_parallel(
        self,
        test_definition: TestDefinition,
        success_response: ATPResponse,
    ) -> None:
        """Test that single test doesn't use parallel execution."""
        suite = TestSuite(
            test_suite="single-test-suite",
            tests=[test_definition],
            defaults=TestDefaults(runs_per_test=1),
        )
        adapter = MockAdapter(success_response)

        async with TestOrchestrator(
            adapter=adapter,
            parallel_tests=True,
            max_parallel_tests=3,
        ) as orchestrator:
            result = await orchestrator.run_suite(suite, "test-agent")

        assert result.success is True
        assert result.total_tests == 1


class TestRunSuiteConvenience:
    """Tests for run_suite convenience function with parallel options."""

    @pytest.mark.anyio
    async def test_run_suite_with_parallel_option(
        self,
        test_suite: TestSuite,
        success_response: ATPResponse,
    ) -> None:
        """Test run_suite convenience function with parallel tests."""
        adapter = MockAdapter(success_response)
        result = await run_suite(
            adapter,
            test_suite,
            "test-agent",
            parallel_tests=True,
            max_parallel_tests=2,
        )
        assert result.success is True
        assert result.total_tests == 2
