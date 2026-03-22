"""Comparison service for multi-agent comparison operations.

This module provides the ComparisonService class that encapsulates all
business logic related to comparing multiple agents on tests and suites.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atp.dashboard.models import (
    Agent,
    SuiteExecution,
    TestExecution,
)
from atp.dashboard.schemas import (
    AgentComparisonMetrics,
    AgentComparisonResponse,
    AgentExecutionDetail,
    AgentTimeline,
    EventSummary,
    MultiTimelineResponse,
    SideBySideComparisonResponse,
    TestComparisonMetrics,
    TimelineEvent,
    TimelineEventsResponse,
)


class ComparisonService:
    """Service for comparison operations.

    This service encapsulates all business logic related to
    comparing agents, generating timelines, and building leaderboards.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the comparison service.

        Args:
            session: Database session for queries.
        """
        self._session = session

    async def compare_agents(
        self,
        suite_name: str,
        agents: list[str],
        limit_per_agent: int = 10,
    ) -> AgentComparisonResponse:
        """Compare multiple agents on a suite.

        Args:
            suite_name: Name of the test suite to compare.
            agents: List of agent names to compare.
            limit_per_agent: Max executions per agent to consider.

        Returns:
            AgentComparisonResponse with aggregate and per-test metrics.
        """
        agent_metrics: list[AgentComparisonMetrics] = []
        test_metrics_map: dict[str, dict[str, Any]] = {}

        for agent_name in agents:
            # Get agent's executions
            stmt = (
                select(SuiteExecution)
                .join(Agent)
                .where(
                    SuiteExecution.suite_name == suite_name,
                    Agent.name == agent_name,
                )
                .options(selectinload(SuiteExecution.test_executions))
                .order_by(SuiteExecution.started_at.desc())
                .limit(limit_per_agent)
            )
            result = await self._session.execute(stmt)
            executions = list(result.scalars().all())

            if not executions:
                continue

            # Calculate aggregate metrics
            total_executions = len(executions)
            avg_success_rate = (
                sum(e.success_rate for e in executions) / total_executions
            )

            # Calculate average score
            all_scores: list[float] = []
            for exec in executions:
                for test in exec.test_executions:
                    if test.score is not None:
                        all_scores.append(test.score)
            avg_score = sum(all_scores) / len(all_scores) if all_scores else None

            # Calculate average duration
            durations = [e.duration_seconds for e in executions if e.duration_seconds]
            avg_duration = sum(durations) / len(durations) if durations else None

            # Latest execution metrics
            latest = executions[0]
            latest_scores = [
                t.score for t in latest.test_executions if t.score is not None
            ]
            latest_score = (
                sum(latest_scores) / len(latest_scores) if latest_scores else None
            )

            agent_metrics.append(
                AgentComparisonMetrics(
                    agent_name=agent_name,
                    total_executions=total_executions,
                    avg_success_rate=avg_success_rate,
                    avg_score=avg_score,
                    avg_duration_seconds=avg_duration,
                    latest_success_rate=latest.success_rate,
                    latest_score=latest_score,
                )
            )

            # Collect per-test metrics
            for exec in executions:
                for test in exec.test_executions:
                    if test.test_id not in test_metrics_map:
                        test_metrics_map[test.test_id] = {"_name": test.test_name}
                    if agent_name not in test_metrics_map[test.test_id]:
                        test_metrics_map[test.test_id][agent_name] = {
                            "scores": [],
                            "durations": [],
                            "successes": [],
                        }
                    data = test_metrics_map[test.test_id][agent_name]
                    if isinstance(data, dict) and "scores" in data:
                        if test.score is not None:
                            data["scores"].append(test.score)
                        if test.duration_seconds is not None:
                            data["durations"].append(test.duration_seconds)
                        data["successes"].append(1 if test.success else 0)

        # Build test comparison metrics
        test_comparisons: list[TestComparisonMetrics] = []
        for test_id, agent_data in test_metrics_map.items():
            test_name = agent_data.pop("_name", test_id)
            metrics_by_agent: dict[str, AgentComparisonMetrics] = {}

            for agent_name_key, data in agent_data.items():
                if isinstance(data, dict) and "scores" in data:
                    scores = data["scores"]
                    durations = data["durations"]
                    successes = data["successes"]

                    metrics_by_agent[agent_name_key] = AgentComparisonMetrics(
                        agent_name=agent_name_key,
                        total_executions=len(successes),
                        avg_success_rate=(
                            sum(successes) / len(successes) if successes else 0
                        ),
                        avg_score=sum(scores) / len(scores) if scores else None,
                        avg_duration_seconds=(
                            sum(durations) / len(durations) if durations else None
                        ),
                        latest_success_rate=successes[-1] if successes else None,
                        latest_score=scores[-1] if scores else None,
                    )

            test_comparisons.append(
                TestComparisonMetrics(
                    test_id=test_id,
                    test_name=test_name if isinstance(test_name, str) else test_id,
                    metrics_by_agent=metrics_by_agent,
                )
            )

        return AgentComparisonResponse(
            suite_name=suite_name,
            agents=agent_metrics,
            tests=test_comparisons,
        )

    async def get_side_by_side_comparison(
        self,
        suite_name: str,
        test_id: str,
        agents: list[str],
    ) -> SideBySideComparisonResponse | None:
        """Get detailed side-by-side comparison of agents on a specific test.

        Args:
            suite_name: Name of the test suite.
            test_id: ID of the specific test.
            agents: List of agent names to compare (2-3 agents).

        Returns:
            SideBySideComparisonResponse with execution details, or None
            if no executions found.
        """
        agent_details: list[AgentExecutionDetail] = []
        test_name: str | None = None

        for agent_name in agents:
            # Query latest test execution for this agent on this test
            stmt = (
                select(TestExecution)
                .join(SuiteExecution)
                .join(Agent)
                .where(
                    SuiteExecution.suite_name == suite_name,
                    TestExecution.test_id == test_id,
                    Agent.name == agent_name,
                )
                .options(selectinload(TestExecution.run_results))
                .order_by(TestExecution.started_at.desc())
                .limit(1)
            )
            result = await self._session.execute(stmt)
            execution = result.scalar_one_or_none()

            if execution is None:
                continue

            # Get test name from first found execution
            if test_name is None:
                test_name = execution.test_name

            # Get the latest run result
            run_results = sorted(execution.run_results, key=lambda r: r.run_number)
            latest_run = run_results[-1] if run_results else None

            # Extract and format events
            events: list[EventSummary] = []
            if latest_run and latest_run.events_json:
                for event in latest_run.events_json:
                    events.append(self._format_event_summary(event))
                events.sort(key=lambda e: e.sequence)

            # Extract metrics
            total_tokens: int | None = None
            total_steps: int | None = None
            tool_calls: int | None = None
            llm_calls: int | None = None
            cost_usd: float | None = None

            if latest_run:
                total_tokens = latest_run.total_tokens
                total_steps = latest_run.total_steps
                tool_calls = latest_run.tool_calls
                llm_calls = latest_run.llm_calls
                cost_usd = latest_run.cost_usd

            agent_details.append(
                AgentExecutionDetail(
                    agent_name=agent_name,
                    test_execution_id=execution.id,
                    score=execution.score,
                    success=execution.success,
                    duration_seconds=execution.duration_seconds,
                    total_tokens=total_tokens,
                    total_steps=total_steps,
                    tool_calls=tool_calls,
                    llm_calls=llm_calls,
                    cost_usd=cost_usd,
                    events=events,
                )
            )

        if not agent_details:
            return None

        return SideBySideComparisonResponse(
            suite_name=suite_name,
            test_id=test_id,
            test_name=test_name or test_id,
            agents=agent_details,
        )

    async def get_timeline_events(
        self,
        suite_name: str,
        test_id: str,
        agent_name: str,
        event_types: list[str] | None = None,
        limit: int = 1000,
    ) -> TimelineEventsResponse | None:
        """Get event timeline for a single agent execution.

        Args:
            suite_name: Name of the test suite.
            test_id: ID of the specific test.
            agent_name: Name of the agent.
            event_types: Optional filter for event types.
            limit: Maximum number of events to return.

        Returns:
            TimelineEventsResponse with events, or None if not found.
        """
        # Query the latest test execution
        stmt = (
            select(TestExecution)
            .join(SuiteExecution)
            .join(Agent)
            .where(
                SuiteExecution.suite_name == suite_name,
                TestExecution.test_id == test_id,
                Agent.name == agent_name,
            )
            .options(selectinload(TestExecution.run_results))
            .order_by(TestExecution.started_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        execution = result.scalar_one_or_none()

        if execution is None:
            return None

        # Get run results with events
        run_results = sorted(execution.run_results, key=lambda r: r.run_number)
        latest_run = run_results[-1] if run_results else None

        events: list[TimelineEvent] = []
        first_timestamp: datetime | None = None
        last_timestamp: datetime | None = None

        if latest_run and latest_run.events_json:
            raw_events = latest_run.events_json
            if event_types:
                raw_events = [
                    e for e in raw_events if e.get("event_type") in event_types
                ]

            for event in raw_events[:limit]:
                timeline_event = self._format_timeline_event(event, first_timestamp)
                if first_timestamp is None:
                    first_timestamp = timeline_event.timestamp
                last_timestamp = timeline_event.timestamp
                events.append(timeline_event)

            events.sort(key=lambda e: e.sequence)

        total_duration_ms: float | None = None
        if first_timestamp and last_timestamp:
            total_duration_ms = (
                last_timestamp - first_timestamp
            ).total_seconds() * 1000

        # Calculate total events count
        total_events = 0
        if latest_run and latest_run.events_json:
            total_events = len(latest_run.events_json)

        return TimelineEventsResponse(
            suite_name=suite_name,
            test_id=test_id,
            test_name=execution.test_name,
            agent_name=agent_name,
            total_events=total_events,
            events=events,
            total_duration_ms=total_duration_ms,
            execution_id=execution.id,
        )

    async def get_multi_agent_timeline(
        self,
        suite_name: str,
        test_id: str,
        agents: list[str],
        limit: int = 500,
    ) -> MultiTimelineResponse | None:
        """Get aligned timelines for multiple agents.

        Args:
            suite_name: Name of the test suite.
            test_id: ID of the specific test.
            agents: List of agent names (2-3).
            limit: Maximum events per agent.

        Returns:
            MultiTimelineResponse with aligned timelines, or None if not found.
        """
        timelines: list[AgentTimeline] = []
        test_name: str | None = None

        for agent_name in agents:
            # Query the latest test execution
            stmt = (
                select(TestExecution)
                .join(SuiteExecution)
                .join(Agent)
                .where(
                    SuiteExecution.suite_name == suite_name,
                    TestExecution.test_id == test_id,
                    Agent.name == agent_name,
                )
                .options(selectinload(TestExecution.run_results))
                .order_by(TestExecution.started_at.desc())
                .limit(1)
            )
            result = await self._session.execute(stmt)
            execution = result.scalar_one_or_none()

            if execution is None:
                continue

            if test_name is None:
                test_name = execution.test_name

            # Get run results with events
            run_results = sorted(execution.run_results, key=lambda r: r.run_number)
            latest_run = run_results[-1] if run_results else None

            events: list[TimelineEvent] = []
            first_timestamp: datetime | None = None
            last_timestamp: datetime | None = None

            if latest_run and latest_run.events_json:
                for event in latest_run.events_json[:limit]:
                    timeline_event = self._format_timeline_event(event, first_timestamp)
                    if first_timestamp is None:
                        first_timestamp = timeline_event.timestamp
                    last_timestamp = timeline_event.timestamp
                    events.append(timeline_event)

                events.sort(key=lambda e: e.sequence)

            total_duration_ms = 0.0
            if first_timestamp and last_timestamp:
                total_duration_ms = (
                    last_timestamp - first_timestamp
                ).total_seconds() * 1000

            timelines.append(
                AgentTimeline(
                    agent_name=agent_name,
                    test_execution_id=execution.id,
                    start_time=first_timestamp or execution.started_at,
                    total_duration_ms=total_duration_ms,
                    events=events,
                )
            )

        if not timelines:
            return None

        return MultiTimelineResponse(
            suite_name=suite_name,
            test_id=test_id,
            test_name=test_name or test_id,
            timelines=timelines,
        )

    def _format_event_summary(self, event: dict) -> EventSummary:
        """Format a raw event dict into an EventSummary.

        Args:
            event: Raw event dictionary from events_json.

        Returns:
            EventSummary with formatted data.
        """
        event_type = event.get("event_type", "unknown")
        payload = event.get("payload", {})

        # Generate summary based on event type
        if event_type == "tool_call":
            tool = payload.get("tool", "unknown")
            event_status = payload.get("status", "")
            summary = f"Tool call: {tool} ({event_status})"
        elif event_type == "llm_request":
            model = payload.get("model", "unknown")
            tokens = payload.get("input_tokens", 0) + payload.get("output_tokens", 0)
            summary = f"LLM request: {model} ({tokens} tokens)"
        elif event_type == "reasoning":
            thought = payload.get("thought", "")
            step = payload.get("step", "")
            summary = thought[:50] + "..." if len(thought) > 50 else thought
            if not summary and step:
                summary = step
            if not summary:
                summary = "Reasoning step"
        elif event_type == "error":
            error_type = payload.get("error_type", "Error")
            message = payload.get("message", "")[:50]
            summary = f"{error_type}: {message}"
        elif event_type == "progress":
            percentage = payload.get("percentage", 0)
            message = payload.get("message", "")
            if message:
                summary = f"Progress: {percentage}% - {message}"
            else:
                summary = f"Progress: {percentage}%"
        else:
            summary = f"Event: {event_type}"

        # Parse timestamp
        timestamp = self._parse_timestamp(event.get("timestamp", ""))

        return EventSummary(
            sequence=event.get("sequence", 0),
            timestamp=timestamp,
            event_type=event_type,
            summary=summary,
            data=payload,
        )

    def _format_timeline_event(
        self, event: dict, first_timestamp: datetime | None
    ) -> TimelineEvent:
        """Format a raw event dict into a TimelineEvent.

        Args:
            event: Raw event dictionary from events_json.
            first_timestamp: Timestamp of the first event for relative timing.

        Returns:
            TimelineEvent with formatted data and relative timing.
        """
        event_summary = self._format_event_summary(event)
        timestamp = event_summary.timestamp

        # Calculate relative time
        relative_time_ms = 0.0
        if first_timestamp:
            relative_time_ms = (timestamp - first_timestamp).total_seconds() * 1000

        # Get duration from payload if available
        payload = event.get("payload", {})
        duration_ms: float | None = payload.get("duration_ms")

        return TimelineEvent(
            sequence=event_summary.sequence,
            timestamp=timestamp,
            event_type=event_summary.event_type,
            summary=event_summary.summary,
            data=event_summary.data,
            relative_time_ms=relative_time_ms,
            duration_ms=duration_ms,
        )

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse a timestamp string.

        Args:
            timestamp_str: ISO format timestamp string.

        Returns:
            Parsed datetime, or current time if parsing fails.
        """
        if timestamp_str:
            try:
                return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now()
