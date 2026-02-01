"""Leaderboard routes.

This module provides endpoints for the performance leaderboard matrix,
showing test scores across agents with rankings and difficulty analysis.

Permissions:
    - GET /leaderboard/matrix: RESULTS_READ
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from atp.dashboard.models import Agent
from atp.dashboard.optimized_queries import build_leaderboard_data
from atp.dashboard.query_cache import get_leaderboard_cache
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.schemas import (
    AgentColumn,
    LeaderboardMatrixResponse,
    TestRow,
    TestScore,
)
from atp.dashboard.v2.dependencies import DBSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


def _calculate_difficulty(avg_score: float | None) -> str:
    """Calculate difficulty rating based on average score.

    Args:
        avg_score: Average score across all agents (0-100 scale).

    Returns:
        Difficulty string: easy, medium, hard, very_hard, or unknown.
    """
    if avg_score is None:
        return "unknown"
    if avg_score >= 80:
        return "easy"
    if avg_score >= 60:
        return "medium"
    if avg_score >= 40:
        return "hard"
    return "very_hard"


def _detect_pattern(scores: list[float | None], pass_rates: list[float]) -> str | None:
    """Detect patterns in scores across agents.

    Args:
        scores: List of scores for each agent.
        pass_rates: List of pass rates for each agent.

    Returns:
        Pattern string or None if no pattern detected.
    """
    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return None

    avg = sum(valid_scores) / len(valid_scores)
    all_pass = all(r >= 0.8 for r in pass_rates)
    all_fail = all(r <= 0.2 for r in pass_rates)

    if all_fail or avg < 40:
        return "hard_for_all"
    if all_pass and avg >= 80:
        return "easy"
    if len(valid_scores) >= 2:
        score_range = max(valid_scores) - min(valid_scores)
        if score_range >= 40:
            return "high_variance"
    return None


@router.get(
    "/matrix",
    response_model=LeaderboardMatrixResponse,
)
async def get_leaderboard_matrix(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_READ))],
    suite_name: str,
    agents: list[str] | None = Query(None),
    limit_executions: int = Query(default=5, le=20),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> LeaderboardMatrixResponse:
    """Get leaderboard matrix for a test suite.

    Requires RESULTS_READ permission.

    Returns a matrix of tests (rows) vs agents (columns) with scores,
    difficulty ratings, and agent rankings.

    Uses optimized bulk queries and caching for performance.

    Args:
        session: Database session.
        suite_name: Name of the test suite.
        agents: Optional list of agent names to filter by.
        limit_executions: Max number of recent executions per agent to consider.
        limit: Max number of tests to return (pagination).
        offset: Offset for pagination.

    Returns:
        LeaderboardMatrixResponse with test rows and agent columns.
    """
    # Get list of agent names if not specified
    if agents is None:
        stmt = select(Agent.name).order_by(Agent.name)
        result = await session.execute(stmt)
        agent_names = list(result.scalars().all())
    else:
        agent_names = agents

    if not agent_names:
        return LeaderboardMatrixResponse(
            suite_name=suite_name,
            tests=[],
            agents=[],
            total_tests=0,
            total_agents=0,
            limit=limit,
            offset=offset,
        )

    # Build cache key for the query
    cache = get_leaderboard_cache()
    sorted_agents_key = ",".join(sorted(agent_names))
    cache_key = f"leaderboard:{suite_name}:{sorted_agents_key}:{limit_executions}"

    # Check cache for raw data (before pagination)
    cached_data = cache.get(cache_key)

    if cached_data is not None:
        logger.debug("Leaderboard cache hit for %s", suite_name)
        test_data = cached_data["test_data"]
        test_names = cached_data["test_names"]
        test_tags = cached_data["test_tags"]
        agent_metrics = cached_data["agent_metrics"]
    else:
        logger.debug("Leaderboard cache miss for %s", suite_name)
        # Use optimized bulk query instead of N+1 queries
        test_data, test_names, test_tags, agent_metrics = await build_leaderboard_data(
            session, suite_name, agent_names, limit_executions
        )
        # Cache the raw data
        cache.put(
            cache_key,
            {
                "test_data": test_data,
                "test_names": test_names,
                "test_tags": test_tags,
                "agent_metrics": agent_metrics,
            },
        )

    # Get total count of tests before pagination
    total_tests = len(test_data)

    # Apply pagination to test IDs
    sorted_test_ids = sorted(test_data.keys())
    paginated_test_ids = sorted_test_ids[offset : offset + limit]

    # Build test rows with pagination
    test_rows: list[TestRow] = []
    for test_id in paginated_test_ids:
        agent_data = test_data[test_id]
        scores_by_agent: dict[str, TestScore] = {}
        all_scores: list[float | None] = []
        all_pass_rates: list[float] = []

        for agent_name in agent_names:
            if agent_name in agent_data:
                data = agent_data[agent_name]
                scores = data["scores"]
                successes = data["successes"]

                avg_score = sum(scores) / len(scores) if scores else None
                pass_rate = sum(successes) / len(successes) if successes else 0.0

                scores_by_agent[agent_name] = TestScore(
                    score=avg_score,
                    success=pass_rate >= 0.5,
                    execution_count=len(successes),
                )
                all_scores.append(avg_score)
                all_pass_rates.append(pass_rate)
            else:
                scores_by_agent[agent_name] = TestScore(
                    score=None,
                    success=False,
                    execution_count=0,
                )
                all_scores.append(None)
                all_pass_rates.append(0.0)

        # Calculate overall average score for this test
        valid_scores = [s for s in all_scores if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        test_rows.append(
            TestRow(
                test_id=test_id,
                test_name=test_names.get(test_id, test_id),
                tags=test_tags.get(test_id, []),
                scores_by_agent=scores_by_agent,
                avg_score=avg_score,
                difficulty=_calculate_difficulty(avg_score),
                pattern=_detect_pattern(all_scores, all_pass_rates),
            )
        )

    # Build agent columns with rankings
    agent_columns: list[AgentColumn] = []
    agent_scores_for_ranking: list[tuple[str, float | None]] = []

    for agent_name in agent_names:
        metrics = agent_metrics[agent_name]
        scores = metrics["scores"]
        successes = metrics["successes"]

        avg_score = sum(scores) / len(scores) if scores else None
        pass_rate = sum(successes) / len(successes) if successes else 0.0
        total_cost = metrics["cost"] if metrics["cost"] > 0 else None

        agent_scores_for_ranking.append((agent_name, avg_score))
        agent_columns.append(
            AgentColumn(
                agent_name=agent_name,
                avg_score=avg_score,
                pass_rate=pass_rate,
                total_tokens=metrics["tokens"],
                total_cost=total_cost,
                rank=0,  # Will be set below
            )
        )

    # Calculate rankings (higher score = better rank)
    sorted_agents = sorted(
        agent_scores_for_ranking,
        key=lambda x: (x[1] is not None, x[1] or 0),
        reverse=True,
    )
    rank_map = {name: rank + 1 for rank, (name, _) in enumerate(sorted_agents)}

    for col in agent_columns:
        col.rank = rank_map[col.agent_name]

    # Sort agent columns by rank
    agent_columns.sort(key=lambda x: x.rank)

    return LeaderboardMatrixResponse(
        suite_name=suite_name,
        tests=test_rows,
        agents=agent_columns,
        total_tests=total_tests,
        total_agents=len(agent_names),
        limit=limit,
        offset=offset,
    )
