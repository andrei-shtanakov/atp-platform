"""Repository layer for CRUD operations on the ATP Test Catalog."""

import statistics
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atp.catalog.models import (
    CatalogCategory,
    CatalogSubmission,
    CatalogSuite,
    CatalogTest,
)


class CatalogRepository:
    """Repository for managing catalog entities with CRUD and aggregate queries."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session

    # ==================== Category Operations ====================

    async def upsert_category(
        self,
        slug: str,
        name: str,
        description: str | None = None,
        icon: str | None = None,
    ) -> CatalogCategory:
        """Create or update a category by slug.

        Args:
            slug: Unique identifier slug.
            name: Display name.
            description: Optional description.
            icon: Optional icon identifier.

        Returns:
            CatalogCategory instance.
        """
        stmt = select(CatalogCategory).where(CatalogCategory.slug == slug)
        result = await self._session.execute(stmt)
        category = result.scalar_one_or_none()

        if category is None:
            category = CatalogCategory(
                slug=slug,
                name=name,
                description=description,
                icon=icon,
            )
            self._session.add(category)
        else:
            category.name = name
            category.description = description
            category.icon = icon

        await self._session.flush()
        return category

    async def list_categories(self) -> list[CatalogCategory]:
        """List all categories with suites eagerly loaded, sorted by slug.

        Returns:
            List of CatalogCategory instances.
        """
        stmt = (
            select(CatalogCategory)
            .options(selectinload(CatalogCategory.suites))
            .order_by(CatalogCategory.slug)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ==================== Suite Operations ====================

    async def upsert_suite(
        self,
        category_id: int,
        slug: str,
        name: str,
        author: str,
        source: str,
        suite_yaml: str,
        description: str | None = None,
        difficulty: str | None = None,
        estimated_minutes: int | None = None,
        tags: dict[str, Any] | None = None,
        version: str = "1.0",
    ) -> CatalogSuite:
        """Create or update a suite by (category_id, slug).

        Args:
            category_id: Parent category ID.
            slug: Unique slug within the category.
            name: Display name.
            author: Suite author.
            source: Suite source (e.g. builtin, community).
            suite_yaml: Raw YAML content.
            description: Optional description.
            difficulty: Optional difficulty level.
            estimated_minutes: Optional estimated completion time.
            tags: Optional list of tags.
            version: Suite version string.

        Returns:
            CatalogSuite instance.
        """
        stmt = select(CatalogSuite).where(
            CatalogSuite.category_id == category_id,
            CatalogSuite.slug == slug,
        )
        result = await self._session.execute(stmt)
        suite = result.scalar_one_or_none()

        if suite is None:
            suite = CatalogSuite(
                category_id=category_id,
                slug=slug,
                name=name,
                author=author,
                source=source,
                suite_yaml=suite_yaml,
                description=description,
                difficulty=difficulty,
                estimated_minutes=estimated_minutes,
                tags=tags,
                version=version,
            )
            self._session.add(suite)
        else:
            suite.name = name
            suite.author = author
            suite.source = source
            suite.suite_yaml = suite_yaml
            suite.description = description
            suite.difficulty = difficulty
            suite.estimated_minutes = estimated_minutes
            suite.tags = tags
            suite.version = version

        await self._session.flush()
        return suite

    async def list_suites(self, category_slug: str | None = None) -> list[CatalogSuite]:
        """List suites with tests and category eagerly loaded.

        Args:
            category_slug: Optional category slug to filter by.

        Returns:
            List of CatalogSuite instances.
        """
        stmt = select(CatalogSuite).options(
            selectinload(CatalogSuite.tests),
            selectinload(CatalogSuite.category),
        )
        if category_slug is not None:
            stmt = stmt.join(CatalogCategory).where(
                CatalogCategory.slug == category_slug
            )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_suite_by_path(
        self, category_slug: str, suite_slug: str
    ) -> CatalogSuite | None:
        """Get a suite by category and suite slug with tests eagerly loaded.

        Args:
            category_slug: Category slug.
            suite_slug: Suite slug within the category.

        Returns:
            CatalogSuite if found, None otherwise.
        """
        stmt = (
            select(CatalogSuite)
            .join(CatalogCategory)
            .where(
                CatalogCategory.slug == category_slug,
                CatalogSuite.slug == suite_slug,
            )
            .options(selectinload(CatalogSuite.tests))
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    # ==================== Test Operations ====================

    async def upsert_test(
        self,
        suite_id: int,
        slug: str,
        name: str,
        task_description: str,
        description: str | None = None,
        difficulty: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> CatalogTest:
        """Create or update a test by (suite_id, slug).

        Args:
            suite_id: Parent suite ID.
            slug: Unique slug within the suite.
            name: Display name.
            task_description: Task description for the agent.
            description: Optional human-readable description.
            difficulty: Optional difficulty level.
            tags: Optional list of tags.

        Returns:
            CatalogTest instance.
        """
        stmt = select(CatalogTest).where(
            CatalogTest.suite_id == suite_id,
            CatalogTest.slug == slug,
        )
        result = await self._session.execute(stmt)
        test = result.scalar_one_or_none()

        if test is None:
            test = CatalogTest(
                suite_id=suite_id,
                slug=slug,
                name=name,
                task_description=task_description,
                description=description,
                difficulty=difficulty,
                tags=tags,
            )
            self._session.add(test)
        else:
            test.name = name
            test.task_description = task_description
            test.description = description
            test.difficulty = difficulty
            test.tags = tags

        await self._session.flush()
        return test

    # ==================== Submission Operations ====================

    async def create_submission(
        self,
        test_id: int,
        agent_name: str,
        agent_type: str,
        score: float,
        quality_score: float = 0.0,
        completeness_score: float = 0.0,
        efficiency_score: float = 0.0,
        cost_score: float = 0.0,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        duration_seconds: float | None = None,
        suite_execution_id: int | None = None,
    ) -> CatalogSubmission:
        """Create a submission record for a test.

        Args:
            test_id: Parent test ID.
            agent_name: Name of the agent.
            agent_type: Type of the agent.
            score: Overall score (0-100).
            quality_score: Quality sub-score.
            completeness_score: Completeness sub-score.
            efficiency_score: Efficiency sub-score.
            cost_score: Cost sub-score.
            total_tokens: Optional total tokens used.
            cost_usd: Optional cost in USD.
            duration_seconds: Optional duration in seconds.
            suite_execution_id: Optional linked suite execution ID.

        Returns:
            CatalogSubmission instance.
        """
        submission = CatalogSubmission(
            test_id=test_id,
            agent_name=agent_name,
            agent_type=agent_type,
            score=score,
            quality_score=quality_score,
            completeness_score=completeness_score,
            efficiency_score=efficiency_score,
            cost_score=cost_score,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            duration_seconds=duration_seconds,
            suite_execution_id=suite_execution_id,
            submitted_at=datetime.now(tz=UTC),
        )
        self._session.add(submission)
        await self._session.flush()
        return submission

    async def update_test_stats(self, test_id: int) -> None:
        """Recalculate and update aggregate stats for a test from its submissions.

        Computes total_submissions, avg_score, best_score, and median_score.

        Args:
            test_id: Test ID to update stats for.
        """
        # Fetch aggregate stats via SQL
        agg_stmt = select(
            func.count(CatalogSubmission.id),
            func.avg(CatalogSubmission.score),
            func.max(CatalogSubmission.score),
        ).where(CatalogSubmission.test_id == test_id)
        agg_result = await self._session.execute(agg_stmt)
        count, avg, best = agg_result.one()

        # Fetch all scores for median (SQLite has no MEDIAN aggregate)
        scores_stmt = (
            select(CatalogSubmission.score)
            .where(CatalogSubmission.test_id == test_id)
            .order_by(CatalogSubmission.score)
        )
        scores_result = await self._session.execute(scores_stmt)
        scores = [row[0] for row in scores_result.all()]
        median = statistics.median(scores) if scores else None

        # Update the test record
        test_stmt = select(CatalogTest).where(CatalogTest.id == test_id)
        test_result = await self._session.execute(test_stmt)
        test = test_result.scalar_one()

        test.total_submissions = count or 0
        test.avg_score = avg
        test.best_score = best
        test.median_score = median

        await self._session.flush()

    async def get_top_submissions(
        self, test_id: int, limit: int = 10
    ) -> list[CatalogSubmission]:
        """Get top submissions for a test sorted by score descending.

        Args:
            test_id: Test ID.
            limit: Maximum number of results.

        Returns:
            List of CatalogSubmission instances.
        """
        stmt = (
            select(CatalogSubmission)
            .where(CatalogSubmission.test_id == test_id)
            .order_by(CatalogSubmission.score.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_global_leaderboard(self, limit: int = 50) -> list[dict]:
        """Get global agent rankings across all catalog tests.

        Groups submissions by agent_name and aggregates count, average
        score, and total score, ordered by average score descending.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of dicts with keys: agent_name, tests_completed,
            avg_score, total_score.
        """
        stmt = (
            select(
                CatalogSubmission.agent_name,
                func.count(CatalogSubmission.id).label("tests_completed"),
                func.avg(CatalogSubmission.score).label("avg_score"),
                func.sum(CatalogSubmission.score).label("total_score"),
            )
            .group_by(CatalogSubmission.agent_name)
            .order_by(func.avg(CatalogSubmission.score).desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [
            {
                "agent_name": row.agent_name,
                "tests_completed": row.tests_completed,
                "avg_score": float(row.avg_score or 0.0),
                "total_score": float(row.total_score or 0.0),
            }
            for row in result.all()
        ]

    async def get_avg_scores_for_test(self, test_id: int) -> dict[str, float]:
        """Get average score breakdown for a test.

        Args:
            test_id: Test ID.

        Returns:
            Dict mapping score component name to average value.
        """
        stmt = select(
            func.avg(CatalogSubmission.score),
            func.avg(CatalogSubmission.quality_score),
            func.avg(CatalogSubmission.completeness_score),
            func.avg(CatalogSubmission.efficiency_score),
            func.avg(CatalogSubmission.cost_score),
        ).where(CatalogSubmission.test_id == test_id)
        result = await self._session.execute(stmt)
        row = result.one()
        return {
            "score": row[0] or 0.0,
            "quality_score": row[1] or 0.0,
            "completeness_score": row[2] or 0.0,
            "efficiency_score": row[3] or 0.0,
            "cost_score": row[4] or 0.0,
        }
