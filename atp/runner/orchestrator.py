"""Test orchestrator for running tests."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from atp.adapters.base import AgentAdapter
from atp.adapters.exceptions import AdapterError, AdapterTimeoutError
from atp.loader.models import TestDefinition, TestSuite
from atp.protocol import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    Context,
    Metrics,
    ResponseStatus,
    Task,
)
from atp.runner.exceptions import RunnerTimeoutError, TestExecutionError
from atp.runner.models import (
    ProgressCallback,
    ProgressEvent,
    ProgressEventType,
    RunResult,
    SandboxConfig,
    SuiteResult,
    TestResult,
)
from atp.runner.sandbox import SandboxManager

logger = logging.getLogger(__name__)


class TestOrchestrator:
    """
    Orchestrates test execution.

    Coordinates adapters, sandboxes, and result collection for running
    individual tests and complete test suites.
    """

    def __init__(
        self,
        adapter: AgentAdapter,
        sandbox_config: SandboxConfig | None = None,
        progress_callback: ProgressCallback | None = None,
        runs_per_test: int = 1,
        fail_fast: bool = False,
        parallel_runs: bool = False,
        max_parallel: int = 5,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            adapter: Agent adapter for executing tests.
            sandbox_config: Sandbox configuration for isolation.
            progress_callback: Optional callback for progress reporting.
            runs_per_test: Number of times to run each test (default 1).
            fail_fast: Stop suite execution on first failure if True.
            parallel_runs: If True, run multiple runs in parallel.
            max_parallel: Maximum number of parallel runs (default 5).
        """
        self.adapter = adapter
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.progress_callback = progress_callback
        self.runs_per_test = runs_per_test
        self.fail_fast = fail_fast
        self.parallel_runs = parallel_runs
        self.max_parallel = max_parallel
        self._sandbox_manager: SandboxManager | None = None
        self._semaphore: asyncio.Semaphore | None = None

    def _emit_progress(self, event: ProgressEvent) -> None:
        """Emit a progress event if callback is registered."""
        if self.progress_callback:
            try:
                self.progress_callback(event)
            except Exception as e:
                logger.warning("Progress callback failed: %s", e)

    def _create_request(
        self,
        test: TestDefinition,
        workspace_path: str | None = None,
    ) -> ATPRequest:
        """
        Create an ATP Request from a test definition.

        Args:
            test: Test definition.
            workspace_path: Optional workspace path for the test.

        Returns:
            ATPRequest for the agent.
        """
        task = Task(
            description=test.task.description,
            input_data=test.task.input_data,
            expected_artifacts=test.task.expected_artifacts,
        )

        constraints: dict[str, Any] = {}
        if test.constraints.max_steps is not None:
            constraints["max_steps"] = test.constraints.max_steps
        if test.constraints.max_tokens is not None:
            constraints["max_tokens"] = test.constraints.max_tokens
        if test.constraints.timeout_seconds is not None:
            constraints["timeout_seconds"] = test.constraints.timeout_seconds
        if test.constraints.allowed_tools is not None:
            constraints["allowed_tools"] = test.constraints.allowed_tools
        if test.constraints.budget_usd is not None:
            constraints["budget_usd"] = test.constraints.budget_usd

        context = None
        if workspace_path:
            context = Context(workspace_path=workspace_path)

        return ATPRequest(
            task_id=str(uuid.uuid4()),
            task=task,
            constraints=constraints,
            context=context,
            metadata={"test_id": test.id, "test_name": test.name},
        )

    async def _execute_with_timeout(
        self,
        request: ATPRequest,
        timeout_seconds: float,
        collect_events: bool = True,
    ) -> tuple[ATPResponse, list[ATPEvent]]:
        """
        Execute a request with timeout enforcement.

        Args:
            request: ATP Request to execute.
            timeout_seconds: Timeout in seconds.
            collect_events: Whether to collect streaming events.

        Returns:
            Tuple of (response, events).

        Raises:
            RunnerTimeoutError: If execution times out.
            TestExecutionError: If execution fails.
        """
        events: list[ATPEvent] = []

        try:
            if collect_events:
                response = await asyncio.wait_for(
                    self._execute_with_events(request, events),
                    timeout=timeout_seconds,
                )
            else:
                response = await asyncio.wait_for(
                    self.adapter.execute(request),
                    timeout=timeout_seconds,
                )
            return response, events

        except TimeoutError:
            # Create timeout response with collected data
            response = ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.TIMEOUT,
                error=f"Execution timed out after {timeout_seconds} seconds",
                metrics=Metrics(wall_time_seconds=timeout_seconds),
            )
            raise RunnerTimeoutError(
                f"Test execution timed out after {timeout_seconds}s",
                test_id=request.metadata.get("test_id") if request.metadata else None,
                timeout_seconds=timeout_seconds,
            ) from None

        except AdapterTimeoutError as e:
            response = ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.TIMEOUT,
                error=str(e),
            )
            raise RunnerTimeoutError(
                str(e),
                test_id=request.metadata.get("test_id") if request.metadata else None,
                timeout_seconds=e.timeout_seconds,
            ) from e

        except AdapterError as e:
            raise TestExecutionError(
                f"Adapter error: {e}",
                test_id=request.metadata.get("test_id") if request.metadata else None,
                cause=e,
            ) from e

    async def _execute_with_events(
        self,
        request: ATPRequest,
        events: list[ATPEvent],
    ) -> ATPResponse:
        """
        Execute a request and collect streaming events.

        Args:
            request: ATP Request to execute.
            events: List to populate with collected events.

        Returns:
            ATPResponse from the agent.
        """
        response: ATPResponse | None = None

        async def collect_events(
            stream: AsyncIterator[ATPEvent | ATPResponse],
        ) -> ATPResponse:
            nonlocal response
            async for item in stream:
                if isinstance(item, ATPEvent):
                    events.append(item)
                    # Emit progress event for agent event
                    self._emit_progress(
                        ProgressEvent(
                            event_type=ProgressEventType.AGENT_EVENT,
                            test_id=(
                                request.metadata.get("test_id")
                                if request.metadata
                                else None
                            ),
                            agent_event=item,
                        )
                    )
                else:
                    response = item
            if response is None:
                raise TestExecutionError(
                    "No response received from agent",
                    test_id=(
                        request.metadata.get("test_id") if request.metadata else None
                    ),
                )
            return response

        stream = self.adapter.stream_events(request)
        return await collect_events(stream)

    async def run_single_test(
        self,
        test: TestDefinition,
        runs: int | None = None,
        parallel: bool | None = None,
    ) -> TestResult:
        """
        Execute a single test.

        Args:
            test: Test definition to execute.
            runs: Number of runs (overrides instance default).
            parallel: Run tests in parallel (overrides instance default).

        Returns:
            TestResult with all run results.
        """
        num_runs = runs if runs is not None else self.runs_per_test
        use_parallel = parallel if parallel is not None else self.parallel_runs
        result = TestResult(test=test, start_time=datetime.now())

        # Emit test started event
        self._emit_progress(
            ProgressEvent(
                event_type=ProgressEventType.TEST_STARTED,
                test_id=test.id,
                test_name=test.name,
                total_runs=num_runs,
            )
        )

        # Create sandbox if enabled
        sandbox_id: str | None = None
        workspace_path: str | None = None

        if self.sandbox_config.enabled:
            if self._sandbox_manager is None:
                self._sandbox_manager = SandboxManager(self.sandbox_config)
            sandbox_id = self._sandbox_manager.create(test.id)
            workspace_path = str(self._sandbox_manager.get_workspace(sandbox_id))

        try:
            if use_parallel and num_runs > 1:
                # Execute runs in parallel with semaphore limiting
                result.runs = await self._execute_runs_parallel(
                    test=test,
                    num_runs=num_runs,
                    workspace_path=workspace_path,
                )
            else:
                # Execute runs sequentially
                for run_number in range(1, num_runs + 1):
                    run_result = await self._execute_run(
                        test=test,
                        run_number=run_number,
                        total_runs=num_runs,
                        workspace_path=workspace_path,
                    )
                    result.runs.append(run_result)

                    # Check for fail-fast on this test
                    if not run_result.success and self.fail_fast:
                        logger.info(
                            "Test %s failed on run %d, stopping due to fail_fast",
                            test.id,
                            run_number,
                        )
                        break

        except Exception as e:
            result.error = str(e)
            logger.error("Test %s failed with error: %s", test.id, e)

        finally:
            # Cleanup sandbox
            if sandbox_id and self._sandbox_manager:
                try:
                    self._sandbox_manager.cleanup(sandbox_id)
                except Exception as e:
                    logger.warning("Failed to cleanup sandbox %s: %s", sandbox_id, e)

        result.end_time = datetime.now()

        # Emit completion event
        event_type = (
            ProgressEventType.TEST_COMPLETED
            if result.success
            else ProgressEventType.TEST_FAILED
        )
        if result.status == ResponseStatus.TIMEOUT:
            event_type = ProgressEventType.TEST_TIMEOUT

        self._emit_progress(
            ProgressEvent(
                event_type=event_type,
                test_id=test.id,
                test_name=test.name,
                success=result.success,
                error=result.error,
                details={
                    "status": result.status.value,
                    "total_runs": result.total_runs,
                    "successful_runs": result.successful_runs,
                    "duration_seconds": result.duration_seconds,
                },
            )
        )

        return result

    async def _execute_runs_parallel(
        self,
        test: TestDefinition,
        num_runs: int,
        workspace_path: str | None = None,
    ) -> list[RunResult]:
        """
        Execute multiple runs in parallel with semaphore limiting.

        Args:
            test: Test definition.
            num_runs: Number of runs to execute.
            workspace_path: Optional workspace path.

        Returns:
            List of RunResult objects, sorted by run_number.
        """
        # Initialize semaphore if not already done
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_parallel)

        async def run_with_semaphore(run_number: int) -> RunResult:
            async with self._semaphore:  # type: ignore[union-attr]
                return await self._execute_run(
                    test=test,
                    run_number=run_number,
                    total_runs=num_runs,
                    workspace_path=workspace_path,
                )

        # Create tasks for all runs
        tasks = [run_with_semaphore(i) for i in range(1, num_runs + 1)]

        # Execute all tasks concurrently and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, converting exceptions to failed runs
        run_results: list[RunResult] = []
        for i, result in enumerate(results, start=1):
            if isinstance(result, BaseException):
                logger.error("Run %d failed with exception: %s", i, result)
                # Create a failed run result for the exception
                run_results.append(
                    RunResult(
                        test_id=test.id,
                        run_number=i,
                        response=ATPResponse(
                            task_id=f"{test.id}-run-{i}",
                            status=ResponseStatus.FAILED,
                            error=str(result),
                        ),
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error=str(result),
                    )
                )
            elif isinstance(result, RunResult):
                run_results.append(result)

        # Sort by run_number to maintain consistent ordering
        run_results.sort(key=lambda r: r.run_number)

        return run_results

    async def _execute_run(
        self,
        test: TestDefinition,
        run_number: int,
        total_runs: int,
        workspace_path: str | None = None,
    ) -> RunResult:
        """
        Execute a single run of a test.

        Args:
            test: Test definition.
            run_number: Current run number (1-indexed).
            total_runs: Total number of runs.
            workspace_path: Optional workspace path.

        Returns:
            RunResult for this execution.
        """
        start_time = datetime.now()

        # Emit run started event
        self._emit_progress(
            ProgressEvent(
                event_type=ProgressEventType.RUN_STARTED,
                test_id=test.id,
                test_name=test.name,
                run_number=run_number,
                total_runs=total_runs,
            )
        )

        request = self._create_request(test, workspace_path)
        timeout = float(test.constraints.timeout_seconds)

        events: list[ATPEvent] = []
        error: str | None = None

        try:
            response, events = await self._execute_with_timeout(
                request=request,
                timeout_seconds=timeout,
                collect_events=True,
            )
        except RunnerTimeoutError as e:
            response = ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.TIMEOUT,
                error=str(e),
                metrics=Metrics(wall_time_seconds=timeout),
            )
            error = str(e)
        except TestExecutionError as e:
            response = ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error=str(e),
            )
            error = str(e)

        end_time = datetime.now()

        run_result = RunResult(
            test_id=test.id,
            run_number=run_number,
            response=response,
            events=events,
            start_time=start_time,
            end_time=end_time,
            error=error,
        )

        # Emit run completed event
        self._emit_progress(
            ProgressEvent(
                event_type=ProgressEventType.RUN_COMPLETED,
                test_id=test.id,
                test_name=test.name,
                run_number=run_number,
                total_runs=total_runs,
                success=run_result.success,
                error=error,
                details={
                    "status": response.status.value,
                    "duration_seconds": run_result.duration_seconds,
                },
            )
        )

        return run_result

    async def run_suite(
        self,
        suite: TestSuite,
        agent_name: str,
        runs_per_test: int | None = None,
    ) -> SuiteResult:
        """
        Execute a complete test suite.

        Args:
            suite: Test suite to execute.
            agent_name: Name of the agent being tested.
            runs_per_test: Override number of runs per test.

        Returns:
            SuiteResult with all test results.
        """
        num_runs = runs_per_test if runs_per_test is not None else self.runs_per_test
        total_tests = len(suite.tests)

        result = SuiteResult(
            suite_name=suite.test_suite,
            agent_name=agent_name,
            start_time=datetime.now(),
        )

        # Emit suite started event
        self._emit_progress(
            ProgressEvent(
                event_type=ProgressEventType.SUITE_STARTED,
                suite_name=suite.test_suite,
                total_tests=total_tests,
                details={"agent_name": agent_name, "runs_per_test": num_runs},
            )
        )

        # Apply defaults to tests
        suite.apply_defaults()

        try:
            for idx, test in enumerate(suite.tests):
                logger.info(
                    "Running test %d/%d: %s (%s)",
                    idx + 1,
                    total_tests,
                    test.name,
                    test.id,
                )

                test_result = await self.run_single_test(test, runs=num_runs)
                result.tests.append(test_result)

                # Check for fail-fast
                if not test_result.success and self.fail_fast:
                    logger.info(
                        "Test %s failed, stopping suite due to fail_fast",
                        test.id,
                    )
                    break

        except Exception as e:
            result.error = str(e)
            logger.error("Suite execution failed: %s", e)

        result.end_time = datetime.now()

        # Emit suite completed event
        self._emit_progress(
            ProgressEvent(
                event_type=ProgressEventType.SUITE_COMPLETED,
                suite_name=suite.test_suite,
                total_tests=result.total_tests,
                completed_tests=len(result.tests),
                success=result.success,
                error=result.error,
                details={
                    "passed_tests": result.passed_tests,
                    "failed_tests": result.failed_tests,
                    "success_rate": result.success_rate,
                    "duration_seconds": result.duration_seconds,
                },
            )
        )

        return result

    async def cleanup(self) -> None:
        """Clean up any resources."""
        if self._sandbox_manager:
            self._sandbox_manager.cleanup_all()
            self._sandbox_manager = None

        await self.adapter.cleanup()

    async def __aenter__(self) -> "TestOrchestrator":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.cleanup()


async def run_test(
    adapter: AgentAdapter,
    test: TestDefinition,
    progress_callback: ProgressCallback | None = None,
    runs: int = 1,
) -> TestResult:
    """
    Convenience function to run a single test.

    Args:
        adapter: Agent adapter.
        test: Test definition.
        progress_callback: Optional progress callback.
        runs: Number of runs.

    Returns:
        TestResult.
    """
    async with TestOrchestrator(
        adapter=adapter,
        progress_callback=progress_callback,
        runs_per_test=runs,
    ) as orchestrator:
        return await orchestrator.run_single_test(test)


async def run_suite(
    adapter: AgentAdapter,
    suite: TestSuite,
    agent_name: str,
    progress_callback: ProgressCallback | None = None,
    runs_per_test: int = 1,
    fail_fast: bool = False,
) -> SuiteResult:
    """
    Convenience function to run a test suite.

    Args:
        adapter: Agent adapter.
        suite: Test suite.
        agent_name: Name of the agent.
        progress_callback: Optional progress callback.
        runs_per_test: Number of runs per test.
        fail_fast: Stop on first failure.

    Returns:
        SuiteResult.
    """
    async with TestOrchestrator(
        adapter=adapter,
        progress_callback=progress_callback,
        runs_per_test=runs_per_test,
        fail_fast=fail_fast,
    ) as orchestrator:
        return await orchestrator.run_suite(suite, agent_name, runs_per_test)
