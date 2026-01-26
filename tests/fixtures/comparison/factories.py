"""Factory functions for creating test database objects.

This module provides factory classes for creating TestExecution and related
model instances with realistic data for testing the comparison dashboard.
"""

from datetime import datetime, timedelta
from typing import Any

from atp.dashboard.models import (
    Agent,
    Artifact,
    EvaluationResult,
    RunResult,
    ScoreComponent,
    SuiteExecution,
    TestExecution,
)
from tests.fixtures.comparison.events import generate_realistic_event_sequence


class AgentFactory:
    """Factory for creating Agent instances."""

    _counter = 0

    @classmethod
    def create(
        cls,
        name: str | None = None,
        agent_type: str = "http",
        config: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> Agent:
        """Create an Agent instance.

        Args:
            name: Agent name (auto-generated if not provided).
            agent_type: Type of agent ("http", "cli", "docker", etc.).
            config: Agent configuration dictionary.
            description: Optional description.

        Returns:
            Agent model instance.
        """
        cls._counter += 1
        if name is None:
            name = f"test-agent-{cls._counter}"
        if config is None:
            config = {"endpoint": f"http://localhost:800{cls._counter}"}

        return Agent(
            name=name,
            agent_type=agent_type,
            config=config,
            description=description,
        )

    @classmethod
    def create_batch(
        cls,
        count: int,
        prefix: str = "agent",
        agent_type: str = "http",
    ) -> list[Agent]:
        """Create multiple Agent instances.

        Args:
            count: Number of agents to create.
            prefix: Name prefix for agents.
            agent_type: Type for all agents.

        Returns:
            List of Agent instances.
        """
        return [
            cls.create(
                name=f"{prefix}-{i + 1}",
                agent_type=agent_type,
            )
            for i in range(count)
        ]

    @classmethod
    def reset(cls) -> None:
        """Reset the counter for testing."""
        cls._counter = 0


class SuiteExecutionFactory:
    """Factory for creating SuiteExecution instances."""

    _counter = 0

    @classmethod
    def create(
        cls,
        suite_name: str | None = None,
        agent_id: int = 1,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        duration_seconds: float | None = None,
        runs_per_test: int = 1,
        total_tests: int = 10,
        passed_tests: int = 8,
        failed_tests: int = 2,
        status: str = "completed",
        error: str | None = None,
    ) -> SuiteExecution:
        """Create a SuiteExecution instance.

        Args:
            suite_name: Name of the test suite.
            agent_id: Foreign key to agent.
            started_at: Start timestamp.
            completed_at: Completion timestamp.
            duration_seconds: Total duration.
            runs_per_test: Number of runs per test.
            total_tests: Total number of tests.
            passed_tests: Number of passed tests.
            failed_tests: Number of failed tests.
            status: Execution status.
            error: Error message if failed.

        Returns:
            SuiteExecution model instance.
        """
        cls._counter += 1
        if suite_name is None:
            suite_name = f"test-suite-{cls._counter}"
        if started_at is None:
            started_at = datetime.now() - timedelta(hours=1)
        if completed_at is None and status == "completed":
            completed_at = datetime.now()
        if duration_seconds is None and completed_at:
            duration_seconds = (completed_at - started_at).total_seconds()

        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0

        return SuiteExecution(
            suite_name=suite_name,
            agent_id=agent_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            runs_per_test=runs_per_test,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            status=status,
            error=error,
        )

    @classmethod
    def reset(cls) -> None:
        """Reset the counter for testing."""
        cls._counter = 0


class TestExecutionFactory:
    """Factory for creating TestExecution instances."""

    _counter = 0

    @classmethod
    def create(
        cls,
        suite_execution_id: int = 1,
        test_id: str | None = None,
        test_name: str | None = None,
        tags: list[str] | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        duration_seconds: float | None = None,
        total_runs: int = 1,
        successful_runs: int = 1,
        success: bool = True,
        score: float | None = 85.0,
        status: str = "completed",
        error: str | None = None,
        statistics: dict[str, Any] | None = None,
    ) -> TestExecution:
        """Create a TestExecution instance.

        Args:
            suite_execution_id: Foreign key to suite execution.
            test_id: Unique test identifier.
            test_name: Human-readable test name.
            tags: List of tags for filtering.
            started_at: Start timestamp.
            completed_at: Completion timestamp.
            duration_seconds: Total duration.
            total_runs: Total number of runs.
            successful_runs: Number of successful runs.
            success: Whether test passed overall.
            score: Test score (0-100).
            status: Execution status.
            error: Error message if failed.
            statistics: Additional statistics JSON.

        Returns:
            TestExecution model instance.
        """
        cls._counter += 1
        if test_id is None:
            test_id = f"test-{cls._counter:03d}"
        if test_name is None:
            test_name = f"Test Case {cls._counter}"
        if tags is None:
            tags = []
        if started_at is None:
            started_at = datetime.now() - timedelta(minutes=30)
        if completed_at is None and status == "completed":
            completed_at = datetime.now()
        if duration_seconds is None and completed_at:
            duration_seconds = (completed_at - started_at).total_seconds()

        return TestExecution(
            suite_execution_id=suite_execution_id,
            test_id=test_id,
            test_name=test_name,
            tags=tags,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            total_runs=total_runs,
            successful_runs=successful_runs,
            success=success,
            score=score,
            status=status,
            error=error,
            statistics=statistics,
        )

    @classmethod
    def create_with_runs(
        cls,
        suite_execution_id: int = 1,
        test_id: str | None = None,
        test_name: str | None = None,
        num_runs: int = 3,
        include_events: bool = True,
        event_steps: int = 5,
        tags: list[str] | None = None,
        score: float | None = 85.0,
    ) -> tuple[TestExecution, list[RunResult]]:
        """Create a TestExecution with associated RunResults.

        Args:
            suite_execution_id: Foreign key to suite execution.
            test_id: Unique test identifier.
            test_name: Human-readable test name.
            num_runs: Number of run results to create.
            include_events: Whether to include events_json.
            event_steps: Number of steps in event sequence.
            tags: List of tags for filtering.
            score: Test score (0-100).

        Returns:
            Tuple of (TestExecution, list of RunResults).
        """
        cls._counter += 1
        if test_id is None:
            test_id = f"test-{cls._counter:03d}"
        if test_name is None:
            test_name = f"Test Case {cls._counter}"
        if tags is None:
            tags = []

        started_at = datetime.now() - timedelta(minutes=30)
        successful_runs = 0
        runs: list[RunResult] = []

        for run_num in range(1, num_runs + 1):
            run_start = started_at + timedelta(minutes=run_num * 5)
            run_success = run_num % 3 != 0  # Every 3rd run fails

            if run_success:
                successful_runs += 1

            events_json = None
            if include_events:
                events_json = generate_realistic_event_sequence(
                    task_id=test_id,
                    num_steps=event_steps,
                    include_error=not run_success,
                    start_time=run_start,
                )

            run = RunResultFactory.create(
                test_execution_id=cls._counter,  # Will be updated after insert
                run_number=run_num,
                started_at=run_start,
                success=run_success,
                events_json=events_json,
            )
            runs.append(run)

        completed_at = datetime.now()
        success = successful_runs > num_runs / 2

        test_exec = TestExecution(
            suite_execution_id=suite_execution_id,
            test_id=test_id,
            test_name=test_name,
            tags=tags,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            total_runs=num_runs,
            successful_runs=successful_runs,
            success=success,
            score=score if success else score * 0.5 if score else None,
            status="completed",
        )

        return test_exec, runs

    @classmethod
    def reset(cls) -> None:
        """Reset the counter for testing."""
        cls._counter = 0


class RunResultFactory:
    """Factory for creating RunResult instances."""

    _counter = 0

    @classmethod
    def create(
        cls,
        test_execution_id: int = 1,
        run_number: int = 1,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        duration_seconds: float | None = None,
        response_status: str = "completed",
        success: bool = True,
        error: str | None = None,
        total_tokens: int | None = 1500,
        input_tokens: int | None = 1000,
        output_tokens: int | None = 500,
        total_steps: int | None = 5,
        tool_calls: int | None = 3,
        llm_calls: int | None = 5,
        cost_usd: float | None = 0.015,
        response_json: dict[str, Any] | None = None,
        events_json: list[dict[str, Any]] | None = None,
    ) -> RunResult:
        """Create a RunResult instance.

        Args:
            test_execution_id: Foreign key to test execution.
            run_number: Run number within the test.
            started_at: Start timestamp.
            completed_at: Completion timestamp.
            duration_seconds: Total duration.
            response_status: Status of the response.
            success: Whether the run succeeded.
            error: Error message if failed.
            total_tokens: Total tokens used.
            input_tokens: Input tokens used.
            output_tokens: Output tokens generated.
            total_steps: Total steps executed.
            tool_calls: Number of tool calls.
            llm_calls: Number of LLM calls.
            cost_usd: Cost in USD.
            response_json: Full response JSON.
            events_json: List of event dictionaries.

        Returns:
            RunResult model instance.
        """
        cls._counter += 1
        if started_at is None:
            started_at = datetime.now() - timedelta(minutes=10)
        if completed_at is None and response_status == "completed":
            completed_at = datetime.now()
        if duration_seconds is None and completed_at:
            duration_seconds = (completed_at - started_at).total_seconds()

        if not success:
            response_status = "failed"
            if error is None:
                error = "Test execution failed"

        return RunResult(
            test_execution_id=test_execution_id,
            run_number=run_number,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            response_status=response_status,
            success=success,
            error=error,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_steps=total_steps,
            tool_calls=tool_calls,
            llm_calls=llm_calls,
            cost_usd=cost_usd,
            response_json=response_json,
            events_json=events_json,
        )

    @classmethod
    def create_with_events(
        cls,
        test_execution_id: int = 1,
        run_number: int = 1,
        num_steps: int = 5,
        include_error: bool = False,
        **kwargs: Any,
    ) -> RunResult:
        """Create a RunResult with generated events.

        Args:
            test_execution_id: Foreign key to test execution.
            run_number: Run number within the test.
            num_steps: Number of steps in the event sequence.
            include_error: Whether to include an error event.
            **kwargs: Additional arguments for create().

        Returns:
            RunResult model instance with events_json populated.
        """
        started_at = kwargs.pop("started_at", None)
        if started_at is None:
            started_at = datetime.now() - timedelta(minutes=10)

        events = generate_realistic_event_sequence(
            task_id=f"run-{run_number}",
            num_steps=num_steps,
            include_error=include_error,
            start_time=started_at,
        )

        # Calculate metrics from events
        tool_calls = sum(1 for e in events if e["event_type"] == "tool_call")
        llm_calls = sum(1 for e in events if e["event_type"] == "llm_request")
        total_tokens = sum(
            e["payload"].get("input_tokens", 0) + e["payload"].get("output_tokens", 0)
            for e in events
            if e["event_type"] == "llm_request"
        )

        return cls.create(
            test_execution_id=test_execution_id,
            run_number=run_number,
            started_at=started_at,
            success=not include_error,
            events_json=events,
            total_steps=num_steps,
            tool_calls=tool_calls,
            llm_calls=llm_calls,
            total_tokens=total_tokens if total_tokens > 0 else None,
            **kwargs,
        )

    @classmethod
    def reset(cls) -> None:
        """Reset the counter for testing."""
        cls._counter = 0


class ArtifactFactory:
    """Factory for creating Artifact instances."""

    _counter = 0

    @classmethod
    def create(
        cls,
        run_result_id: int = 1,
        artifact_type: str = "file",
        path: str | None = None,
        name: str | None = None,
        content_type: str | None = "text/plain",
        size_bytes: int | None = 1024,
        content: str | None = None,
        data_json: dict[str, Any] | None = None,
    ) -> Artifact:
        """Create an Artifact instance.

        Args:
            run_result_id: Foreign key to run result.
            artifact_type: Type of artifact ("file", "structured", "reference").
            path: File path for file artifacts.
            name: Name for structured artifacts.
            content_type: MIME type.
            size_bytes: File size.
            content: Inline content.
            data_json: Structured data.

        Returns:
            Artifact model instance.
        """
        cls._counter += 1
        if path is None and artifact_type == "file":
            path = f"output/result_{cls._counter}.txt"
        if name is None and artifact_type == "structured":
            name = f"result_{cls._counter}"

        return Artifact(
            run_result_id=run_result_id,
            artifact_type=artifact_type,
            path=path,
            name=name,
            content_type=content_type,
            size_bytes=size_bytes,
            content=content,
            data_json=data_json,
        )

    @classmethod
    def reset(cls) -> None:
        """Reset the counter for testing."""
        cls._counter = 0


class EvaluationResultFactory:
    """Factory for creating EvaluationResult instances."""

    _counter = 0

    @classmethod
    def create(
        cls,
        test_execution_id: int = 1,
        evaluator_name: str = "artifact",
        passed: bool = True,
        score: float | None = 0.95,
        total_checks: int = 5,
        passed_checks: int = 5,
        failed_checks: int = 0,
        checks_json: list[dict[str, Any]] | None = None,
    ) -> EvaluationResult:
        """Create an EvaluationResult instance.

        Args:
            test_execution_id: Foreign key to test execution.
            evaluator_name: Name of the evaluator.
            passed: Whether evaluation passed.
            score: Evaluation score (0-1).
            total_checks: Total number of checks.
            passed_checks: Number of passed checks.
            failed_checks: Number of failed checks.
            checks_json: Detailed check results.

        Returns:
            EvaluationResult model instance.
        """
        cls._counter += 1
        if checks_json is None:
            checks_json = [
                {"name": f"check_{i + 1}", "passed": i < passed_checks, "score": 1.0}
                for i in range(total_checks)
            ]

        return EvaluationResult(
            test_execution_id=test_execution_id,
            evaluator_name=evaluator_name,
            passed=passed,
            score=score,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            checks_json=checks_json,
        )

    @classmethod
    def reset(cls) -> None:
        """Reset the counter for testing."""
        cls._counter = 0


class ScoreComponentFactory:
    """Factory for creating ScoreComponent instances."""

    _counter = 0

    @classmethod
    def create(
        cls,
        test_execution_id: int = 1,
        component_name: str = "quality",
        raw_value: float | None = 0.85,
        normalized_value: float = 0.85,
        weight: float = 0.4,
        weighted_value: float | None = None,
        details_json: dict[str, Any] | None = None,
    ) -> ScoreComponent:
        """Create a ScoreComponent instance.

        Args:
            test_execution_id: Foreign key to test execution.
            component_name: Name of the component.
            raw_value: Raw score value.
            normalized_value: Normalized score (0-1).
            weight: Weight for this component.
            weighted_value: Weighted value (calculated if not provided).
            details_json: Additional details.

        Returns:
            ScoreComponent model instance.
        """
        cls._counter += 1
        if weighted_value is None:
            weighted_value = normalized_value * weight

        return ScoreComponent(
            test_execution_id=test_execution_id,
            component_name=component_name,
            raw_value=raw_value,
            normalized_value=normalized_value,
            weight=weight,
            weighted_value=weighted_value,
            details_json=details_json,
        )

    @classmethod
    def create_full_breakdown(
        cls,
        test_execution_id: int = 1,
        quality: float = 0.85,
        completeness: float = 0.90,
        efficiency: float = 0.75,
        cost: float = 0.80,
    ) -> list[ScoreComponent]:
        """Create a full set of score components.

        Args:
            test_execution_id: Foreign key to test execution.
            quality: Quality score (0-1).
            completeness: Completeness score (0-1).
            efficiency: Efficiency score (0-1).
            cost: Cost efficiency score (0-1).

        Returns:
            List of ScoreComponent instances.
        """
        weights = {"quality": 0.4, "completeness": 0.3, "efficiency": 0.2, "cost": 0.1}
        scores = {
            "quality": quality,
            "completeness": completeness,
            "efficiency": efficiency,
            "cost": cost,
        }

        return [
            cls.create(
                test_execution_id=test_execution_id,
                component_name=name,
                raw_value=score,
                normalized_value=score,
                weight=weights[name],
            )
            for name, score in scores.items()
        ]

    @classmethod
    def reset(cls) -> None:
        """Reset the counter for testing."""
        cls._counter = 0


def reset_all_factories() -> None:
    """Reset all factory counters for clean test isolation."""
    AgentFactory.reset()
    SuiteExecutionFactory.reset()
    TestExecutionFactory.reset()
    RunResultFactory.reset()
    ArtifactFactory.reset()
    EvaluationResultFactory.reset()
    ScoreComponentFactory.reset()
