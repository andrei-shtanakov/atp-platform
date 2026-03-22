"""Test service for managing test execution data.

This module provides the TestService class that encapsulates all
business logic related to test executions, including querying,
filtering, and aggregating test results.
"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atp.dashboard.models import (
    Agent,
    SuiteExecution,
    TestExecution,
)
from atp.dashboard.schemas import (
    DashboardSummary,
    EvaluationResultResponse,
    RunResultSummary,
    ScoreComponentResponse,
    SuiteExecutionDetail,
    SuiteExecutionList,
    SuiteExecutionSummary,
    SuiteTrend,
    TestExecutionDetail,
    TestExecutionList,
    TestExecutionSummary,
    TestTrend,
    TrendDataPoint,
    TrendResponse,
)


class TestService:
    """Service for test execution operations.

    This service encapsulates all business logic related to test
    executions, suite executions, and dashboard summary data.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the test service.

        Args:
            session: Database session for queries.
        """
        self._session = session

    async def get_dashboard_summary(
        self,
        recent_limit: int = 10,
    ) -> DashboardSummary:
        """Get dashboard summary statistics.

        Args:
            recent_limit: Number of recent executions to include.

        Returns:
            Dashboard summary with counts and recent executions.
        """
        # Count agents
        agent_count = (
            await self._session.execute(select(func.count(Agent.id)))
        ).scalar() or 0

        # Count unique suites
        suite_count = (
            await self._session.execute(
                select(func.count(func.distinct(SuiteExecution.suite_name)))
            )
        ).scalar() or 0

        # Count total executions
        execution_count = (
            await self._session.execute(select(func.count(SuiteExecution.id)))
        ).scalar() or 0

        # Get recent executions
        stmt = (
            select(SuiteExecution)
            .options(selectinload(SuiteExecution.agent))
            .order_by(SuiteExecution.started_at.desc())
            .limit(recent_limit)
        )
        result = await self._session.execute(stmt)
        recent_execs = list(result.scalars().all())

        # Calculate recent success rate and score
        recent_success_rate = 0.0
        recent_avg_score: float | None = None
        if recent_execs:
            recent_success_rate = sum(e.success_rate for e in recent_execs) / len(
                recent_execs
            )

            # Get scores from test executions
            scores: list[float] = []
            for exec in recent_execs:
                stmt = select(TestExecution.score).where(
                    TestExecution.suite_execution_id == exec.id,
                    TestExecution.score.isnot(None),
                )
                score_result = await self._session.execute(stmt)
                scores.extend(
                    [s for s in score_result.scalars().all() if s is not None]
                )
            if scores:
                recent_avg_score = sum(scores) / len(scores)

        # Build recent execution summaries
        recent_summaries = []
        for exec in recent_execs:
            summary = SuiteExecutionSummary.model_validate(exec)
            summary.agent_name = exec.agent.name if exec.agent else None
            recent_summaries.append(summary)

        return DashboardSummary(
            total_agents=agent_count,
            total_suites=suite_count,
            total_executions=execution_count,
            recent_success_rate=recent_success_rate,
            recent_avg_score=recent_avg_score,
            recent_executions=recent_summaries,
        )

    async def list_suite_executions(
        self,
        suite_name: str | None = None,
        agent_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SuiteExecutionList:
        """List suite executions with optional filtering.

        Args:
            suite_name: Filter by suite name.
            agent_id: Filter by agent ID.
            limit: Maximum number of results.
            offset: Offset for pagination.

        Returns:
            Paginated list of suite executions.
        """
        stmt = select(SuiteExecution).options(selectinload(SuiteExecution.agent))
        if suite_name:
            stmt = stmt.where(SuiteExecution.suite_name == suite_name)
        if agent_id:
            stmt = stmt.where(SuiteExecution.agent_id == agent_id)

        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self._session.execute(count_stmt)).scalar() or 0

        # Get paginated results
        stmt = (
            stmt.order_by(SuiteExecution.started_at.desc()).limit(limit).offset(offset)
        )
        result = await self._session.execute(stmt)
        executions = result.scalars().all()

        items = []
        for exec in executions:
            summary = SuiteExecutionSummary.model_validate(exec)
            summary.agent_name = exec.agent.name if exec.agent else None
            items.append(summary)

        return SuiteExecutionList(
            total=total,
            items=items,
            limit=limit,
            offset=offset,
        )

    async def list_suite_names(self) -> list[str]:
        """List unique suite names.

        Returns:
            List of unique suite names ordered alphabetically.
        """
        stmt = (
            select(SuiteExecution.suite_name)
            .distinct()
            .order_by(SuiteExecution.suite_name)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_suite_execution(
        self,
        execution_id: int,
    ) -> SuiteExecutionDetail | None:
        """Get suite execution details.

        Args:
            execution_id: Suite execution ID.

        Returns:
            Suite execution detail or None if not found.
        """
        stmt = (
            select(SuiteExecution)
            .where(SuiteExecution.id == execution_id)
            .options(
                selectinload(SuiteExecution.agent),
                selectinload(SuiteExecution.test_executions),
            )
        )
        result = await self._session.execute(stmt)
        execution = result.scalar_one_or_none()

        if execution is None:
            return None

        detail = SuiteExecutionDetail.model_validate(execution)
        detail.agent_name = execution.agent.name if execution.agent else None
        detail.tests = [
            TestExecutionSummary.model_validate(t) for t in execution.test_executions
        ]
        return detail

    async def list_test_executions(
        self,
        suite_execution_id: int | None = None,
        test_id: str | None = None,
        success: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> TestExecutionList:
        """List test executions with optional filtering.

        Args:
            suite_execution_id: Filter by suite execution ID.
            test_id: Filter by test ID.
            success: Filter by success status.
            limit: Maximum number of results.
            offset: Offset for pagination.

        Returns:
            Paginated list of test executions.
        """
        stmt = select(TestExecution)
        if suite_execution_id:
            stmt = stmt.where(TestExecution.suite_execution_id == suite_execution_id)
        if test_id:
            stmt = stmt.where(TestExecution.test_id == test_id)
        if success is not None:
            stmt = stmt.where(TestExecution.success == success)

        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self._session.execute(count_stmt)).scalar() or 0

        # Get paginated results
        stmt = (
            stmt.order_by(TestExecution.started_at.desc()).limit(limit).offset(offset)
        )
        result = await self._session.execute(stmt)
        executions = result.scalars().all()

        return TestExecutionList(
            total=total,
            items=[TestExecutionSummary.model_validate(e) for e in executions],
            limit=limit,
            offset=offset,
        )

    async def get_test_execution(
        self,
        execution_id: int,
    ) -> TestExecutionDetail | None:
        """Get test execution details.

        Args:
            execution_id: Test execution ID.

        Returns:
            Test execution detail or None if not found.
        """
        stmt = (
            select(TestExecution)
            .where(TestExecution.id == execution_id)
            .options(
                selectinload(TestExecution.run_results),
                selectinload(TestExecution.evaluation_results),
                selectinload(TestExecution.score_components),
            )
        )
        result = await self._session.execute(stmt)
        execution = result.scalar_one_or_none()

        if execution is None:
            return None

        detail = TestExecutionDetail.model_validate(execution)
        detail.runs = [
            RunResultSummary.model_validate(r) for r in execution.run_results
        ]
        detail.evaluations = [
            EvaluationResultResponse.model_validate(e)
            for e in execution.evaluation_results
        ]
        detail.score_components = [
            ScoreComponentResponse.model_validate(s) for s in execution.score_components
        ]
        return detail

    async def get_suite_trends(
        self,
        suite_name: str,
        agent_name: str | None = None,
        metric: str = "success_rate",
        limit: int = 20,
    ) -> TrendResponse:
        """Get suite execution trends over time.

        Args:
            suite_name: Name of the suite to get trends for.
            agent_name: Optional agent name filter.
            metric: Metric to track (success_rate, score, duration).
            limit: Maximum number of data points.

        Returns:
            Trend response with data points per agent.
        """
        # Build query
        stmt = (
            select(SuiteExecution)
            .join(Agent)
            .where(SuiteExecution.suite_name == suite_name)
            .options(selectinload(SuiteExecution.agent))
            .order_by(SuiteExecution.started_at.desc())
            .limit(limit)
        )
        if agent_name:
            stmt = stmt.where(Agent.name == agent_name)

        result = await self._session.execute(stmt)
        executions = list(result.scalars().all())

        # Group by agent
        agent_data: dict[str, list[SuiteExecution]] = {}
        for exec in executions:
            if exec.agent:
                name = exec.agent.name
                if name not in agent_data:
                    agent_data[name] = []
                agent_data[name].append(exec)

        # Build trends
        suite_trends: list[SuiteTrend] = []
        for name, execs in agent_data.items():
            data_points: list[TrendDataPoint] = []
            # Reverse to chronological order
            for exec in reversed(execs):
                value = self._get_metric_value(exec, metric)
                if value is not None:
                    data_points.append(
                        TrendDataPoint(
                            timestamp=exec.started_at,
                            value=value,
                            execution_id=exec.id,
                        )
                    )
            suite_trends.append(
                SuiteTrend(
                    suite_name=suite_name,
                    agent_name=name,
                    data_points=data_points,
                    metric=metric,
                )
            )

        return TrendResponse(suite_trends=suite_trends, test_trends=[])

    async def get_test_trends(
        self,
        suite_name: str,
        test_id: str,
        agent_name: str | None = None,
        metric: str = "score",
        limit: int = 20,
    ) -> TrendResponse:
        """Get test execution trends over time.

        Args:
            suite_name: Name of the suite.
            test_id: ID of the test.
            agent_name: Optional agent name filter.
            metric: Metric to track (score, duration, success).
            limit: Maximum number of data points per agent.

        Returns:
            Trend response with data points per agent.
        """
        # Build query
        stmt = (
            select(TestExecution)
            .join(SuiteExecution)
            .join(Agent)
            .where(
                SuiteExecution.suite_name == suite_name,
                TestExecution.test_id == test_id,
            )
            .options(
                selectinload(TestExecution.suite_execution).selectinload(
                    SuiteExecution.agent
                )
            )
            .order_by(TestExecution.started_at.desc())
            .limit(limit * 10)  # Get more to account for multiple agents
        )
        if agent_name:
            stmt = stmt.where(Agent.name == agent_name)

        result = await self._session.execute(stmt)
        executions = list(result.scalars().all())

        # Group by agent
        agent_data: dict[str, list[TestExecution]] = {}
        for exec in executions:
            if exec.suite_execution and exec.suite_execution.agent:
                name = exec.suite_execution.agent.name
                if name not in agent_data:
                    agent_data[name] = []
                if len(agent_data[name]) < limit:
                    agent_data[name].append(exec)

        # Build trends
        test_trends: list[TestTrend] = []
        test_name = executions[0].test_name if executions else test_id
        for name, execs in agent_data.items():
            data_points: list[TrendDataPoint] = []
            # Reverse to chronological order
            for exec in reversed(execs):
                value = self._get_test_metric_value(exec, metric)
                if value is not None:
                    data_points.append(
                        TrendDataPoint(
                            timestamp=exec.started_at,
                            value=value,
                            execution_id=exec.id,
                        )
                    )
            # Include agent name in test_name for multi-agent trends
            trend_name = f"{test_name} ({name})" if not agent_name else test_name
            test_trends.append(
                TestTrend(
                    test_id=test_id,
                    test_name=trend_name,
                    data_points=data_points,
                    metric=metric,
                )
            )

        return TrendResponse(suite_trends=[], test_trends=test_trends)

    def _get_metric_value(self, execution: SuiteExecution, metric: str) -> float | None:
        """Get a metric value from a suite execution.

        Args:
            execution: Suite execution to get metric from.
            metric: Metric name.

        Returns:
            Metric value or None.
        """
        if metric == "success_rate":
            return execution.success_rate
        elif metric == "duration":
            return execution.duration_seconds
        return None

    def _get_test_metric_value(
        self, execution: TestExecution, metric: str
    ) -> float | None:
        """Get a metric value from a test execution.

        Args:
            execution: Test execution to get metric from.
            metric: Metric name.

        Returns:
            Metric value or None.
        """
        if metric == "score":
            return execution.score
        elif metric == "duration":
            return execution.duration_seconds
        elif metric == "success":
            return 1.0 if execution.success else 0.0
        return None
