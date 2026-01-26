"""Optimized database queries for ATP Dashboard.

This module provides optimized query functions that replace N+1 query patterns
with efficient bulk queries using proper JOINs and aggregations.
"""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atp.dashboard.models import (
    Agent,
    RunResult,
    SuiteExecution,
    TestExecution,
)


async def get_agents_by_names(
    session: AsyncSession,
    agent_names: list[str] | None = None,
) -> list[Agent]:
    """Get agents by names or all agents if not specified.

    Args:
        session: Database session.
        agent_names: Optional list of agent names to filter by.

    Returns:
        List of Agent objects.
    """
    if agent_names:
        stmt = select(Agent).where(Agent.name.in_(agent_names)).order_by(Agent.name)
    else:
        stmt = select(Agent).order_by(Agent.name)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_suite_executions_for_agents(
    session: AsyncSession,
    suite_name: str,
    agent_ids: list[int],
    limit_per_agent: int = 5,
) -> list[SuiteExecution]:
    """Get recent suite executions for multiple agents in a single query.

    Uses a subquery with ROW_NUMBER to get the most recent N executions
    per agent efficiently.

    Args:
        session: Database session.
        suite_name: Name of the test suite.
        agent_ids: List of agent IDs to query.
        limit_per_agent: Maximum executions per agent.

    Returns:
        List of SuiteExecution objects with test_executions loaded.
    """
    if not agent_ids:
        return []

    # For better compatibility, we'll use a simpler approach:
    # Get all relevant executions and filter in Python
    # This is still better than N+1 queries
    stmt = (
        select(SuiteExecution)
        .where(
            SuiteExecution.suite_name == suite_name,
            SuiteExecution.agent_id.in_(agent_ids),
        )
        .options(selectinload(SuiteExecution.test_executions))
        .order_by(SuiteExecution.agent_id, SuiteExecution.started_at.desc())
    )

    result = await session.execute(stmt)
    all_executions = list(result.scalars().all())

    # Filter to keep only the most recent N per agent
    agent_counts: dict[int, int] = {}
    filtered: list[SuiteExecution] = []
    for exec in all_executions:
        count = agent_counts.get(exec.agent_id, 0)
        if count < limit_per_agent:
            filtered.append(exec)
            agent_counts[exec.agent_id] = count + 1

    return filtered


async def get_run_results_for_test_executions(
    session: AsyncSession,
    test_execution_ids: list[int],
) -> dict[int, list[RunResult]]:
    """Get run results for multiple test executions in a single query.

    Args:
        session: Database session.
        test_execution_ids: List of test execution IDs.

    Returns:
        Dictionary mapping test_execution_id to list of RunResults.
    """
    if not test_execution_ids:
        return {}

    stmt = (
        select(RunResult)
        .where(RunResult.test_execution_id.in_(test_execution_ids))
        .order_by(RunResult.test_execution_id, RunResult.run_number)
    )

    result = await session.execute(stmt)
    all_runs = result.scalars().all()

    # Group by test_execution_id
    grouped: dict[int, list[RunResult]] = {}
    for run in all_runs:
        if run.test_execution_id not in grouped:
            grouped[run.test_execution_id] = []
        grouped[run.test_execution_id].append(run)

    return grouped


async def get_aggregated_metrics_for_suite(
    session: AsyncSession,
    suite_name: str,
    agent_ids: list[int],
    limit_per_agent: int = 5,
) -> dict[int, dict[str, Any]]:
    """Get aggregated metrics (tokens, cost) for agents in a single query.

    Uses SQL aggregation to compute totals instead of loading all records.

    Args:
        session: Database session.
        suite_name: Name of the test suite.
        agent_ids: List of agent IDs.
        limit_per_agent: Maximum executions per agent to consider.

    Returns:
        Dictionary mapping agent_id to metrics dict with tokens and cost.
    """
    if not agent_ids:
        return {}

    # First get the relevant suite execution IDs (most recent per agent)
    # We need to get IDs first, then aggregate
    executions = await get_suite_executions_for_agents(
        session, suite_name, agent_ids, limit_per_agent
    )

    suite_exec_ids = [e.id for e in executions]
    if not suite_exec_ids:
        return {aid: {"total_tokens": 0, "total_cost": 0.0} for aid in agent_ids}

    # Get test execution IDs
    test_exec_ids = []
    agent_to_suite_execs: dict[int, list[int]] = {}
    for exec in executions:
        if exec.agent_id not in agent_to_suite_execs:
            agent_to_suite_execs[exec.agent_id] = []
        agent_to_suite_execs[exec.agent_id].append(exec.id)
        test_exec_ids.extend([te.id for te in exec.test_executions])

    if not test_exec_ids:
        return {aid: {"total_tokens": 0, "total_cost": 0.0} for aid in agent_ids}

    # Aggregate metrics in a single query
    stmt = (
        select(
            SuiteExecution.agent_id,
            func.coalesce(func.sum(RunResult.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(RunResult.cost_usd), 0.0).label("total_cost"),
        )
        .select_from(RunResult)
        .join(TestExecution)
        .join(SuiteExecution)
        .where(
            SuiteExecution.id.in_(suite_exec_ids),
        )
        .group_by(SuiteExecution.agent_id)
    )

    result = await session.execute(stmt)
    rows = result.all()

    # Build result dict
    metrics: dict[int, dict[str, Any]] = {
        aid: {"total_tokens": 0, "total_cost": 0.0} for aid in agent_ids
    }
    for row in rows:
        agent_id = row.agent_id
        if agent_id in metrics:
            metrics[agent_id] = {
                "total_tokens": int(row.total_tokens or 0),
                "total_cost": float(row.total_cost or 0.0),
            }

    return metrics


async def build_leaderboard_data(
    session: AsyncSession,
    suite_name: str,
    agent_names: list[str],
    limit_executions: int = 5,
) -> tuple[
    dict[str, dict[str, dict[str, list[float]]]],  # test_data
    dict[str, str],  # test_names
    dict[str, list[str]],  # test_tags
    dict[str, dict[str, Any]],  # agent_metrics
]:
    """Build leaderboard data efficiently using bulk queries.

    This replaces the N+1 query pattern with a more efficient approach:
    1. Single query to get all agents
    2. Single query to get all recent suite executions with test_executions
    3. Aggregation queries for metrics

    Args:
        session: Database session.
        suite_name: Name of the test suite.
        agent_names: List of agent names.
        limit_executions: Max executions per agent.

    Returns:
        Tuple of (test_data, test_names, test_tags, agent_metrics).
    """
    # Initialize data structures
    test_data: dict[str, dict[str, dict[str, list[float]]]] = {}
    test_names: dict[str, str] = {}
    test_tags: dict[str, list[str]] = {}
    agent_metrics: dict[str, dict[str, Any]] = {
        name: {"scores": [], "successes": [], "tokens": 0, "cost": 0.0}
        for name in agent_names
    }

    # Early return if no agents specified
    if not agent_names:
        return test_data, test_names, test_tags, agent_metrics

    # Get agents
    agents = await get_agents_by_names(session, agent_names)
    if not agents:
        return test_data, test_names, test_tags, agent_metrics

    agent_id_to_name = {a.id: a.name for a in agents}
    agent_ids = [a.id for a in agents]

    # Get all suite executions in one query
    executions = await get_suite_executions_for_agents(
        session, suite_name, agent_ids, limit_executions
    )

    # Collect all test execution IDs for metrics query
    all_test_exec_ids: list[int] = []
    for exec in executions:
        all_test_exec_ids.extend([te.id for te in exec.test_executions])

    # Get run results for metrics (tokens, cost)
    run_results_by_test = await get_run_results_for_test_executions(
        session, all_test_exec_ids
    )

    # Process executions and build data structures
    for exec in executions:
        agent_name = agent_id_to_name.get(exec.agent_id)
        if not agent_name:
            continue

        for test in exec.test_executions:
            test_id = test.test_id

            # Initialize test data structure
            if test_id not in test_data:
                test_data[test_id] = {}
                test_names[test_id] = test.test_name
                test_tags[test_id] = test.tags or []

            if agent_name not in test_data[test_id]:
                test_data[test_id][agent_name] = {
                    "scores": [],
                    "successes": [],
                }

            # Collect score and success data
            if test.score is not None:
                test_data[test_id][agent_name]["scores"].append(test.score)
                agent_metrics[agent_name]["scores"].append(test.score)
            test_data[test_id][agent_name]["successes"].append(
                1.0 if test.success else 0.0
            )
            agent_metrics[agent_name]["successes"].append(1.0 if test.success else 0.0)

            # Collect token and cost data from run results
            runs = run_results_by_test.get(test.id, [])
            for run in runs:
                if run.total_tokens:
                    agent_metrics[agent_name]["tokens"] += run.total_tokens
                if run.cost_usd:
                    agent_metrics[agent_name]["cost"] += run.cost_usd

    return test_data, test_names, test_tags, agent_metrics
