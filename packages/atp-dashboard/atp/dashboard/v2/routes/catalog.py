"""Catalog API routes.

This module provides endpoints for the ATP Test Catalog, exposing
categories, suites, tests, and submission leaderboards.
"""

import logging

from atp.catalog.repository import CatalogRepository
from atp.catalog.sync import sync_builtin_catalog
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from atp.dashboard.v2.dependencies import DBSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/catalog", tags=["catalog"])


# ==================== Response Models ====================


class CategoryResponse(BaseModel):
    """Response model for a catalog category."""

    slug: str
    name: str
    description: str | None
    suite_count: int
    test_count: int


class SuiteResponse(BaseModel):
    """Response model for a catalog suite summary."""

    slug: str
    name: str
    description: str | None
    difficulty: str | None
    author: str
    source: str
    test_count: int


class TestResponse(BaseModel):
    """Response model for a catalog test."""

    slug: str
    name: str
    task_description: str
    difficulty: str | None
    total_submissions: int
    avg_score: float | None
    best_score: float | None


class SuiteDetailResponse(BaseModel):
    """Response model for a catalog suite with its tests."""

    slug: str
    name: str
    description: str | None
    difficulty: str | None
    author: str
    source: str
    category: str
    tests: list[TestResponse]


class SubmissionResponse(BaseModel):
    """Response model for a single leaderboard submission."""

    rank: int
    agent_name: str
    score: float
    quality_score: float
    completeness_score: float
    efficiency_score: float
    cost_score: float


class GlobalLeaderboardEntry(BaseModel):
    """Response model for a global leaderboard entry."""

    rank: int
    agent_name: str
    tests_completed: int
    avg_score: float
    total_score: float


# ==================== Endpoints ====================


@router.get("/categories", response_model=list[CategoryResponse])
async def list_categories(session: DBSession) -> list[CategoryResponse]:
    """List all catalog categories, auto-syncing builtin data if empty.

    Returns:
        List of CategoryResponse objects with suite and test counts.
    """
    repo = CatalogRepository(session)
    categories = await repo.list_categories()

    if not categories:
        logger.info("Catalog is empty — syncing builtin catalog")
        await sync_builtin_catalog(session)
        categories = await repo.list_categories()

    return [
        CategoryResponse(
            slug=cat.slug,
            name=cat.name,
            description=cat.description,
            suite_count=len(cat.suites),
            test_count=sum(len(s.tests) for s in cat.suites),
        )
        for cat in categories
    ]


@router.get(
    "/categories/{category_slug}/suites",
    response_model=list[SuiteResponse],
)
async def list_suites(
    category_slug: str,
    session: DBSession,
) -> list[SuiteResponse]:
    """List suites within a category.

    Args:
        category_slug: Category slug to filter by.
        session: Database session.

    Returns:
        List of SuiteResponse objects.
    """
    repo = CatalogRepository(session)
    suites = await repo.list_suites(category_slug=category_slug)
    return [
        SuiteResponse(
            slug=s.slug,
            name=s.name,
            description=s.description,
            difficulty=s.difficulty,
            author=s.author,
            source=s.source,
            test_count=len(s.tests),
        )
        for s in suites
    ]


@router.get(
    "/categories/{category_slug}/suites/{suite_slug}",
    response_model=SuiteDetailResponse,
)
async def get_suite(
    category_slug: str,
    suite_slug: str,
    session: DBSession,
) -> SuiteDetailResponse:
    """Get a suite with its tests.

    Args:
        category_slug: Category slug.
        suite_slug: Suite slug within the category.
        session: Database session.

    Returns:
        SuiteDetailResponse with tests.

    Raises:
        HTTPException: 404 if suite not found.
    """
    repo = CatalogRepository(session)
    suite = await repo.get_suite_by_path(
        category_slug=category_slug, suite_slug=suite_slug
    )
    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{category_slug}/{suite_slug}' not found",
        )

    return SuiteDetailResponse(
        slug=suite.slug,
        name=suite.name,
        description=suite.description,
        difficulty=suite.difficulty,
        author=suite.author,
        source=suite.source,
        category=category_slug,
        tests=[
            TestResponse(
                slug=t.slug,
                name=t.name,
                task_description=t.task_description,
                difficulty=t.difficulty,
                total_submissions=t.total_submissions or 0,
                avg_score=t.avg_score,
                best_score=t.best_score,
            )
            for t in suite.tests
        ],
    )


@router.get(
    "/categories/{category_slug}/suites/{suite_slug}/leaderboard",
    response_model=dict[str, list[SubmissionResponse]],
)
async def get_suite_leaderboard(
    category_slug: str,
    suite_slug: str,
    session: DBSession,
    limit: int = Query(default=10, ge=1, le=1000),
) -> dict[str, list[SubmissionResponse]]:
    """Get per-test leaderboards for a suite.

    Args:
        category_slug: Category slug.
        suite_slug: Suite slug within the category.
        session: Database session.
        limit: Max submissions per test (default 10, max 1000).

    Returns:
        Dict mapping test slug to ranked submission list.

    Raises:
        HTTPException: 404 if suite not found.
    """
    repo = CatalogRepository(session)
    suite = await repo.get_suite_by_path(
        category_slug=category_slug, suite_slug=suite_slug
    )
    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{category_slug}/{suite_slug}' not found",
        )

    result: dict[str, list[SubmissionResponse]] = {}
    for test in suite.tests:
        submissions = await repo.get_top_submissions(test_id=test.id, limit=limit)
        result[test.slug] = [
            SubmissionResponse(
                rank=idx + 1,
                agent_name=sub.agent_name,
                score=sub.score,
                quality_score=sub.quality_score,
                completeness_score=sub.completeness_score,
                efficiency_score=sub.efficiency_score,
                cost_score=sub.cost_score,
            )
            for idx, sub in enumerate(submissions)
        ]

    return result


@router.get("/leaderboard", response_model=list[GlobalLeaderboardEntry])
async def get_global_leaderboard(
    session: DBSession,
    limit: int = Query(default=50, ge=1, le=1000),
) -> list[GlobalLeaderboardEntry]:
    """Get global agent rankings across all catalog tests.

    Args:
        session: Database session.
        limit: Max number of entries (default 50, max 1000).

    Returns:
        List of GlobalLeaderboardEntry ranked by avg score descending.
    """
    repo = CatalogRepository(session)
    rows = await repo.get_global_leaderboard(limit=limit)
    return [
        GlobalLeaderboardEntry(
            rank=idx + 1,
            agent_name=row["agent_name"],
            tests_completed=row["tests_completed"],
            avg_score=row["avg_score"],
            total_score=row["total_score"],
        )
        for idx, row in enumerate(rows)
    ]
