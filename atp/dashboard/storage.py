"""Storage layer for persisting ATP test results to the database."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atp.dashboard.models import (
    Agent,
    Artifact,
    EvaluationResult,
    RunResult,
    ScoreComponent,
    SuiteExecution,
    TestExecution,
)
from atp.evaluators.base import EvalResult
from atp.protocol import ATPEvent, ATPResponse
from atp.reporters.base import SuiteReport, TestReport
from atp.runner.models import RunResult as ATPRunResult
from atp.runner.models import SuiteResult, TestResult
from atp.scoring.models import ScoredTestResult
from atp.statistics.models import TestRunStatistics


class ResultStorage:
    """Storage service for persisting ATP test results."""

    def __init__(self, session: AsyncSession):
        """Initialize storage with database session.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session

    # ==================== Agent Operations ====================

    async def get_or_create_agent(
        self, name: str, agent_type: str, config: dict[str, Any] | None = None
    ) -> Agent:
        """Get existing agent or create new one.

        Args:
            name: Agent name.
            agent_type: Agent type (http, container, langgraph, etc.).
            config: Agent configuration.

        Returns:
            Agent instance.
        """
        stmt = select(Agent).where(Agent.name == name)
        result = await self._session.execute(stmt)
        agent = result.scalar_one_or_none()

        if agent is None:
            agent = Agent(
                name=name,
                agent_type=agent_type,
                config=config or {},
            )
            self._session.add(agent)
            await self._session.flush()

        return agent

    async def get_agent_by_name(self, name: str) -> Agent | None:
        """Get agent by name.

        Args:
            name: Agent name.

        Returns:
            Agent if found, None otherwise.
        """
        stmt = select(Agent).where(Agent.name == name)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_agents(self) -> list[Agent]:
        """List all agents.

        Returns:
            List of agents.
        """
        stmt = select(Agent).order_by(Agent.name)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ==================== Suite Execution Operations ====================

    async def create_suite_execution(
        self,
        suite_name: str,
        agent: Agent,
        runs_per_test: int = 1,
        started_at: datetime | None = None,
    ) -> SuiteExecution:
        """Create a new suite execution record.

        Args:
            suite_name: Name of the test suite.
            agent: Agent running the tests.
            runs_per_test: Number of runs per test.
            started_at: Execution start time.

        Returns:
            SuiteExecution instance.
        """
        execution = SuiteExecution(
            suite_name=suite_name,
            agent_id=agent.id,
            runs_per_test=runs_per_test,
            started_at=started_at or datetime.now(tz=UTC),
            status="running",
        )
        self._session.add(execution)
        await self._session.flush()
        return execution

    async def update_suite_execution(
        self,
        execution: SuiteExecution,
        *,
        completed_at: datetime | None = None,
        total_tests: int | None = None,
        passed_tests: int | None = None,
        failed_tests: int | None = None,
        success_rate: float | None = None,
        status: str | None = None,
        error: str | None = None,
    ) -> SuiteExecution:
        """Update suite execution record.

        Args:
            execution: Suite execution to update.
            completed_at: Completion time.
            total_tests: Total number of tests.
            passed_tests: Number of passed tests.
            failed_tests: Number of failed tests.
            success_rate: Success rate (0.0-1.0).
            status: Execution status.
            error: Error message if failed.

        Returns:
            Updated SuiteExecution instance.
        """
        if completed_at is not None:
            execution.completed_at = completed_at
            if execution.started_at:
                execution.duration_seconds = (
                    completed_at - execution.started_at
                ).total_seconds()
        if total_tests is not None:
            execution.total_tests = total_tests
        if passed_tests is not None:
            execution.passed_tests = passed_tests
        if failed_tests is not None:
            execution.failed_tests = failed_tests
        if success_rate is not None:
            execution.success_rate = success_rate
        if status is not None:
            execution.status = status
        if error is not None:
            execution.error = error

        await self._session.flush()
        return execution

    async def get_suite_execution(
        self, execution_id: int, include_tests: bool = False
    ) -> SuiteExecution | None:
        """Get suite execution by ID.

        Args:
            execution_id: Suite execution ID.
            include_tests: Whether to eagerly load test executions.

        Returns:
            SuiteExecution if found, None otherwise.
        """
        stmt = select(SuiteExecution).where(SuiteExecution.id == execution_id)
        if include_tests:
            stmt = stmt.options(selectinload(SuiteExecution.test_executions))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_suite_executions(
        self,
        suite_name: str | None = None,
        agent_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SuiteExecution]:
        """List suite executions with optional filtering.

        Args:
            suite_name: Filter by suite name.
            agent_id: Filter by agent ID.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of suite executions.
        """
        stmt = select(SuiteExecution).order_by(SuiteExecution.started_at.desc())
        if suite_name is not None:
            stmt = stmt.where(SuiteExecution.suite_name == suite_name)
        if agent_id is not None:
            stmt = stmt.where(SuiteExecution.agent_id == agent_id)
        stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ==================== Test Execution Operations ====================

    async def create_test_execution(
        self,
        suite_execution: SuiteExecution,
        test_id: str,
        test_name: str,
        tags: list[str] | None = None,
        started_at: datetime | None = None,
        total_runs: int = 1,
    ) -> TestExecution:
        """Create a new test execution record.

        Args:
            suite_execution: Parent suite execution.
            test_id: Test identifier.
            test_name: Human-readable test name.
            tags: Test tags.
            started_at: Execution start time.
            total_runs: Expected number of runs.

        Returns:
            TestExecution instance.
        """
        execution = TestExecution(
            suite_execution_id=suite_execution.id,
            test_id=test_id,
            test_name=test_name,
            tags=tags or [],
            started_at=started_at or datetime.now(tz=UTC),
            total_runs=total_runs,
            status="running",
        )
        self._session.add(execution)
        await self._session.flush()
        return execution

    async def update_test_execution(
        self,
        execution: TestExecution,
        *,
        completed_at: datetime | None = None,
        successful_runs: int | None = None,
        success: bool | None = None,
        score: float | None = None,
        status: str | None = None,
        error: str | None = None,
        statistics: dict[str, Any] | None = None,
    ) -> TestExecution:
        """Update test execution record.

        Args:
            execution: Test execution to update.
            completed_at: Completion time.
            successful_runs: Number of successful runs.
            success: Whether test passed.
            score: Test score (0-100).
            status: Execution status.
            error: Error message if failed.
            statistics: Statistical analysis results.

        Returns:
            Updated TestExecution instance.
        """
        if completed_at is not None:
            execution.completed_at = completed_at
            if execution.started_at:
                execution.duration_seconds = (
                    completed_at - execution.started_at
                ).total_seconds()
        if successful_runs is not None:
            execution.successful_runs = successful_runs
        if success is not None:
            execution.success = success
        if score is not None:
            execution.score = score
        if status is not None:
            execution.status = status
        if error is not None:
            execution.error = error
        if statistics is not None:
            execution.statistics = statistics

        await self._session.flush()
        return execution

    async def get_test_execution(
        self, execution_id: int, include_runs: bool = False
    ) -> TestExecution | None:
        """Get test execution by ID.

        Args:
            execution_id: Test execution ID.
            include_runs: Whether to eagerly load run results.

        Returns:
            TestExecution if found, None otherwise.
        """
        stmt = select(TestExecution).where(TestExecution.id == execution_id)
        if include_runs:
            stmt = stmt.options(
                selectinload(TestExecution.run_results),
                selectinload(TestExecution.evaluation_results),
                selectinload(TestExecution.score_components),
            )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    # ==================== Run Result Operations ====================

    async def create_run_result(
        self,
        test_execution: TestExecution,
        run_number: int,
        response: ATPResponse,
        events: list[ATPEvent] | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> RunResult:
        """Create a run result record from ATP response.

        Args:
            test_execution: Parent test execution.
            run_number: Run number (1-indexed).
            response: ATP response from agent.
            events: ATP events from agent.
            started_at: Run start time.
            completed_at: Run completion time.

        Returns:
            RunResult instance.
        """
        now = datetime.now(tz=UTC)
        started = started_at or now
        completed = completed_at or now

        # Calculate duration
        duration = (completed - started).total_seconds() if completed else None

        # Extract metrics
        metrics = response.metrics
        run = RunResult(
            test_execution_id=test_execution.id,
            run_number=run_number,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            response_status=response.status.value,
            success=response.status.value == "completed",
            error=response.error,
            total_tokens=metrics.total_tokens if metrics else None,
            input_tokens=metrics.input_tokens if metrics else None,
            output_tokens=metrics.output_tokens if metrics else None,
            total_steps=metrics.total_steps if metrics else None,
            tool_calls=metrics.tool_calls if metrics else None,
            llm_calls=metrics.llm_calls if metrics else None,
            cost_usd=metrics.cost_usd if metrics else None,
            response_json=response.model_dump(mode="json"),
            events_json=[e.model_dump(mode="json") for e in events] if events else None,
        )
        self._session.add(run)
        await self._session.flush()

        # Store artifacts
        for artifact in response.artifacts:
            await self._create_artifact(run, artifact)

        return run

    async def _create_artifact(self, run_result: RunResult, artifact: Any) -> Artifact:
        """Create artifact record.

        Args:
            run_result: Parent run result.
            artifact: ATP artifact object.

        Returns:
            Artifact instance.
        """
        artifact_record = Artifact(
            run_result_id=run_result.id,
            artifact_type=artifact.type,
            path=getattr(artifact, "path", None),
            name=getattr(artifact, "name", None),
            content_type=getattr(artifact, "content_type", None),
            size_bytes=getattr(artifact, "size_bytes", None),
            content_hash=getattr(artifact, "content_hash", None),
            content=getattr(artifact, "content", None),
            data_json=getattr(artifact, "data", None),
        )
        self._session.add(artifact_record)
        await self._session.flush()
        return artifact_record

    # ==================== Evaluation Result Operations ====================

    async def store_evaluation_results(
        self, test_execution: TestExecution, eval_results: list[EvalResult]
    ) -> list[EvaluationResult]:
        """Store evaluation results for a test execution.

        Args:
            test_execution: Parent test execution.
            eval_results: List of evaluation results.

        Returns:
            List of stored EvaluationResult instances.
        """
        records = []
        for eval_result in eval_results:
            record = EvaluationResult(
                test_execution_id=test_execution.id,
                evaluator_name=eval_result.evaluator,
                passed=eval_result.passed,
                score=eval_result.score,
                total_checks=eval_result.total_checks,
                passed_checks=eval_result.passed_checks,
                failed_checks=eval_result.failed_checks,
                checks_json=[c.model_dump(mode="json") for c in eval_result.checks],
            )
            self._session.add(record)
            records.append(record)

        await self._session.flush()
        return records

    # ==================== Score Component Operations ====================

    async def store_scored_result(
        self, test_execution: TestExecution, scored: ScoredTestResult
    ) -> list[ScoreComponent]:
        """Store score breakdown for a test execution.

        Args:
            test_execution: Parent test execution.
            scored: Scored test result with breakdown.

        Returns:
            List of stored ScoreComponent instances.
        """
        # Update test execution score
        test_execution.score = scored.score
        await self._session.flush()

        # Store individual components
        components = []
        for component in scored.breakdown.components:
            record = ScoreComponent(
                test_execution_id=test_execution.id,
                component_name=component.name,
                raw_value=component.raw_value,
                normalized_value=component.normalized_value,
                weight=component.weight,
                weighted_value=component.weighted_value,
                details_json=component.details,
            )
            self._session.add(record)
            components.append(record)

        await self._session.flush()
        return components

    # ==================== High-Level Persistence Operations ====================

    async def persist_suite_result(  # pragma: no cover
        self,
        result: SuiteResult,
        agent_type: str = "unknown",
        eval_results: dict[str, list[EvalResult]] | None = None,
        scored_results: dict[str, ScoredTestResult] | None = None,
        statistics: dict[str, TestRunStatistics] | None = None,
    ) -> SuiteExecution:
        """Persist a complete suite result to the database.

        This is the main entry point for storing test results.

        Args:
            result: Suite result from test execution.
            agent_type: Type of agent.
            eval_results: Mapping of test_id to evaluation results.
            scored_results: Mapping of test_id to scored results.
            statistics: Mapping of test_id to statistics.

        Returns:
            Stored SuiteExecution instance.
        """
        eval_results = eval_results or {}
        scored_results = scored_results or {}
        statistics = statistics or {}

        # Get or create agent
        agent = await self.get_or_create_agent(
            name=result.agent_name,
            agent_type=agent_type,
        )

        # Create suite execution
        suite_exec = await self.create_suite_execution(
            suite_name=result.suite_name,
            agent=agent,
            runs_per_test=result.runs_per_test,
            started_at=result.start_time,
        )

        # Update with results
        await self.update_suite_execution(
            suite_exec,
            completed_at=result.end_time,
            total_tests=result.total_tests,
            passed_tests=result.passed_tests,
            failed_tests=result.failed_tests,
            success_rate=result.success_rate,
            status="completed" if result.error is None else "failed",
            error=result.error,
        )

        # Store test results
        for test_result in result.tests:
            await self._persist_test_result(
                suite_exec,
                test_result,
                eval_results.get(test_result.test.id, []),
                scored_results.get(test_result.test.id),
                statistics.get(test_result.test.id),
            )

        return suite_exec

    async def _persist_test_result(
        self,
        suite_exec: SuiteExecution,
        test_result: TestResult,
        eval_results: list[EvalResult],
        scored_result: ScoredTestResult | None,
        statistics: TestRunStatistics | None,
    ) -> TestExecution:
        """Persist a single test result.

        Args:
            suite_exec: Parent suite execution.
            test_result: Test result to persist.
            eval_results: Evaluation results for this test.
            scored_result: Scored result for this test.
            statistics: Statistics for this test.

        Returns:
            Stored TestExecution instance.
        """
        # Create test execution
        test_exec = await self.create_test_execution(
            suite_execution=suite_exec,
            test_id=test_result.test.id,
            test_name=test_result.test.name,
            tags=test_result.test.tags or [],
            started_at=test_result.start_time,
            total_runs=test_result.total_runs,
        )

        # Update with results
        await self.update_test_execution(
            test_exec,
            completed_at=test_result.end_time,
            successful_runs=test_result.successful_runs,
            success=test_result.success,
            score=scored_result.score if scored_result else None,
            status="completed" if test_result.error is None else "failed",
            error=test_result.error,
            statistics=statistics.to_dict() if statistics else None,
        )

        # Store run results
        for run_result in test_result.runs:
            await self._persist_run_result(test_exec, run_result)

        # Store evaluation results
        if eval_results:
            await self.store_evaluation_results(test_exec, eval_results)

        # Store score breakdown
        if scored_result:
            await self.store_scored_result(test_exec, scored_result)

        return test_exec

    async def _persist_run_result(
        self, test_exec: TestExecution, run_result: ATPRunResult
    ) -> RunResult:
        """Persist a single run result.

        Args:
            test_exec: Parent test execution.
            run_result: Run result to persist.

        Returns:
            Stored RunResult instance.
        """
        return await self.create_run_result(
            test_execution=test_exec,
            run_number=run_result.run_number,
            response=run_result.response,
            events=run_result.events,
            started_at=run_result.start_time,
            completed_at=run_result.end_time,
        )

    async def persist_suite_report(
        self,
        report: SuiteReport,
        agent_type: str = "unknown",
    ) -> SuiteExecution:
        """Persist a suite report to the database.

        Alternative entry point using SuiteReport instead of SuiteResult.

        Args:
            report: Suite report to persist.
            agent_type: Type of agent.

        Returns:
            Stored SuiteExecution instance.
        """
        # Get or create agent
        agent = await self.get_or_create_agent(
            name=report.agent_name,
            agent_type=agent_type,
        )

        # Create suite execution
        suite_exec = await self.create_suite_execution(
            suite_name=report.suite_name,
            agent=agent,
            runs_per_test=report.runs_per_test,
            started_at=datetime.now(tz=UTC),  # Report doesn't have start_time
        )

        # Update with results
        await self.update_suite_execution(
            suite_exec,
            completed_at=datetime.now(tz=UTC),
            total_tests=report.total_tests,
            passed_tests=report.passed_tests,
            failed_tests=report.failed_tests,
            success_rate=report.success_rate,
            status="completed" if report.error is None else "failed",
            error=report.error,
        )

        # Store test reports
        for test_report in report.tests:
            await self._persist_test_report(suite_exec, test_report)

        return suite_exec

    async def _persist_test_report(
        self, suite_exec: SuiteExecution, test_report: TestReport
    ) -> TestExecution:
        """Persist a single test report.

        Args:
            suite_exec: Parent suite execution.
            test_report: Test report to persist.

        Returns:
            Stored TestExecution instance.
        """
        # Create test execution
        test_exec = await self.create_test_execution(
            suite_execution=suite_exec,
            test_id=test_report.test_id,
            test_name=test_report.test_name,
            tags=[],  # Report doesn't include tags
            started_at=datetime.now(tz=UTC),
            total_runs=test_report.total_runs,
        )

        # Update with results
        stats_dict = (
            test_report.statistics.to_dict() if test_report.statistics else None
        )
        await self.update_test_execution(
            test_exec,
            completed_at=datetime.now(tz=UTC),
            successful_runs=test_report.successful_runs,
            success=test_report.success,
            score=test_report.score,
            status="completed" if test_report.error is None else "failed",
            error=test_report.error,
            statistics=stats_dict,
        )

        # Store evaluation results
        if test_report.eval_results:
            await self.store_evaluation_results(test_exec, test_report.eval_results)

        # Store score breakdown
        if test_report.scored_result:
            await self.store_scored_result(test_exec, test_report.scored_result)

        return test_exec

    # ==================== Query Operations ====================

    async def get_historical_executions(
        self,
        suite_name: str,
        agent_name: str | None = None,
        limit: int = 50,
    ) -> list[SuiteExecution]:
        """Get historical executions for a suite.

        Args:
            suite_name: Name of the test suite.
            agent_name: Optional agent name filter.
            limit: Maximum number of results.

        Returns:
            List of suite executions.
        """
        stmt = (
            select(SuiteExecution)
            .where(SuiteExecution.suite_name == suite_name)
            .order_by(SuiteExecution.started_at.desc())
            .limit(limit)
        )
        if agent_name:
            stmt = stmt.join(Agent).where(Agent.name == agent_name)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_test_history(
        self,
        suite_name: str,
        test_id: str,
        agent_name: str | None = None,
        limit: int = 50,
    ) -> list[TestExecution]:
        """Get historical executions for a specific test.

        Args:
            suite_name: Name of the test suite.
            test_id: Test identifier.
            agent_name: Optional agent name filter.
            limit: Maximum number of results.

        Returns:
            List of test executions.
        """
        stmt = (
            select(TestExecution)
            .join(SuiteExecution)
            .where(
                SuiteExecution.suite_name == suite_name,
                TestExecution.test_id == test_id,
            )
            .order_by(TestExecution.started_at.desc())
            .limit(limit)
        )
        if agent_name:
            stmt = stmt.join(Agent).where(Agent.name == agent_name)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def compare_agents(
        self,
        suite_name: str,
        agent_names: list[str],
        limit_per_agent: int = 10,
    ) -> dict[str, list[SuiteExecution]]:
        """Get recent executions for multiple agents for comparison.

        Args:
            suite_name: Name of the test suite.
            agent_names: List of agent names to compare.
            limit_per_agent: Maximum executions per agent.

        Returns:
            Mapping of agent name to list of suite executions.
        """
        result_map: dict[str, list[SuiteExecution]] = {}

        for agent_name in agent_names:
            stmt = (
                select(SuiteExecution)
                .join(Agent)
                .where(
                    SuiteExecution.suite_name == suite_name,
                    Agent.name == agent_name,
                )
                .order_by(SuiteExecution.started_at.desc())
                .limit(limit_per_agent)
            )
            result = await self._session.execute(stmt)
            result_map[agent_name] = list(result.scalars().all())

        return result_map
