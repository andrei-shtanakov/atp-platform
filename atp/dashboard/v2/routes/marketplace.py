"""Test Suite Marketplace routes (TASK-803).

This module provides endpoints for the test suite marketplace, enabling:
- Publishing and managing test suites
- Version management
- Search and discovery
- Ratings and reviews
- GitHub import

Public endpoints (no auth required):
    - GET /marketplace: List/search published suites
    - GET /marketplace/{slug}: Get suite details
    - GET /marketplace/{slug}/versions: List versions
    - GET /marketplace/{slug}/reviews: List reviews
    - GET /marketplace/stats: Get marketplace statistics
    - GET /marketplace/categories: List categories

Authenticated endpoints:
    - POST /marketplace: Publish a new suite
    - PATCH /marketplace/{slug}: Update suite
    - DELETE /marketplace/{slug}: Unpublish suite
    - POST /marketplace/{slug}/versions: Create new version
    - POST /marketplace/{slug}/reviews: Add review
    - PATCH /marketplace/reviews/{id}: Update own review
    - DELETE /marketplace/reviews/{id}: Delete own review
    - POST /marketplace/{slug}/install: Install suite
    - DELETE /marketplace/{slug}/install: Uninstall suite
    - POST /marketplace/import/github: Import from GitHub

Admin endpoints:
    - POST /marketplace/{slug}/feature: Feature/unfeature a suite
    - POST /marketplace/{slug}/verify: Verify a suite
    - DELETE /marketplace/reviews/{id}/flag: Moderate review
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, func, or_, select
from sqlalchemy.orm import selectinload

from atp.dashboard.models import (
    MarketplaceSuite,
    MarketplaceSuiteInstall,
    MarketplaceSuiteReview,
    MarketplaceSuiteVersion,
)
from atp.dashboard.query_cache import QueryCache
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.schemas import (
    GitHubImportRequest,
    GitHubImportResponse,
    MarketplaceCategoryStats,
    MarketplaceStats,
    MarketplaceSuiteCreate,
    MarketplaceSuiteDetail,
    MarketplaceSuiteInstallList,
    MarketplaceSuiteInstallResponse,
    MarketplaceSuiteList,
    MarketplaceSuiteResponse,
    MarketplaceSuiteReviewCreate,
    MarketplaceSuiteReviewList,
    MarketplaceSuiteReviewResponse,
    MarketplaceSuiteReviewUpdate,
    MarketplaceSuiteSummary,
    MarketplaceSuiteUpdate,
    MarketplaceSuiteVersionCreate,
    MarketplaceSuiteVersionList,
    MarketplaceSuiteVersionResponse,
    MarketplaceSuiteVersionUpdate,
)
from atp.dashboard.v2.dependencies import (
    AdminUser,
    DBSession,
    RequiredUser,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/marketplace", tags=["marketplace"])

# Cache for marketplace queries
_marketplace_cache: QueryCache[dict] | None = None


def get_marketplace_cache() -> QueryCache[dict]:
    """Get the marketplace cache (5 minute TTL)."""
    global _marketplace_cache
    if _marketplace_cache is None:
        _marketplace_cache = QueryCache(max_size=200, ttl_seconds=300)
    return _marketplace_cache


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse semantic version string into tuple."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return 1, 0, 0


# ==================== Public Endpoints (No Auth) ====================


@router.get("", response_model=MarketplaceSuiteList)
async def list_marketplace_suites(
    session: DBSession,
    query: str | None = Query(None, max_length=200),
    category: str | None = Query(None, max_length=100),
    tags: list[str] | None = Query(None),
    license_type: str | None = Query(None, max_length=50),
    verified_only: bool = Query(default=False),
    featured_only: bool = Query(default=False),
    min_rating: float | None = Query(None, ge=0, le=5),
    sort_by: str = Query(default="downloads"),
    sort_order: str = Query(default="desc"),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
) -> MarketplaceSuiteList:
    """List and search marketplace test suites.

    This is a public endpoint that does not require authentication.

    Args:
        session: Database session.
        query: Search query (searches name and description).
        category: Filter by category.
        tags: Filter by tags (any match).
        license_type: Filter by license type.
        verified_only: Only show verified suites.
        featured_only: Only show featured suites.
        min_rating: Minimum average rating.
        sort_by: Sort field (downloads, rating, updated, name, created).
        sort_order: Sort order (asc, desc).
        limit: Maximum results.
        offset: Pagination offset.

    Returns:
        Paginated list of marketplace suites.
    """
    stmt = select(MarketplaceSuite).where(MarketplaceSuite.is_published.is_(True))

    # Apply filters
    if query:
        search_pattern = f"%{query}%"
        stmt = stmt.where(
            or_(
                MarketplaceSuite.name.ilike(search_pattern),
                MarketplaceSuite.description.ilike(search_pattern),
                MarketplaceSuite.short_description.ilike(search_pattern),
            )
        )

    if category:
        stmt = stmt.where(MarketplaceSuite.category == category)

    if tags:
        # Filter suites that have any of the specified tags
        for tag in tags:
            stmt = stmt.where(MarketplaceSuite.tags.contains([tag]))

    if license_type:
        stmt = stmt.where(MarketplaceSuite.license_type == license_type)

    if verified_only:
        stmt = stmt.where(MarketplaceSuite.is_verified.is_(True))

    if featured_only:
        stmt = stmt.where(MarketplaceSuite.is_featured.is_(True))

    if min_rating is not None:
        stmt = stmt.where(
            or_(
                MarketplaceSuite.average_rating >= min_rating,
                MarketplaceSuite.average_rating.is_(None),
            )
        )

    # Count total
    count_stmt = select(func.count()).select_from(stmt.subquery())
    count_result = await session.execute(count_stmt)
    total = count_result.scalar() or 0

    # Apply sorting
    sort_columns = {
        "downloads": MarketplaceSuite.total_downloads,
        "rating": MarketplaceSuite.average_rating,
        "updated": MarketplaceSuite.updated_at,
        "name": MarketplaceSuite.name,
        "created": MarketplaceSuite.created_at,
    }
    sort_col = sort_columns.get(sort_by, MarketplaceSuite.total_downloads)
    if sort_order == "desc":
        stmt = stmt.order_by(desc(sort_col))
    else:
        stmt = stmt.order_by(sort_col)

    # Apply pagination
    stmt = stmt.offset(offset).limit(limit)

    result = await session.execute(stmt)
    suites = list(result.scalars().all())

    return MarketplaceSuiteList(
        items=[MarketplaceSuiteSummary.model_validate(s) for s in suites],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/stats", response_model=MarketplaceStats)
async def get_marketplace_stats(
    session: DBSession,
) -> MarketplaceStats:
    """Get marketplace statistics.

    This is a public endpoint that does not require authentication.

    Returns:
        Overall marketplace statistics.
    """
    cache = get_marketplace_cache()
    cache_key = "stats"
    cached = cache.get(cache_key)
    if cached:
        return MarketplaceStats(**cached)

    # Total counts
    total_stmt = (
        select(func.count())
        .select_from(MarketplaceSuite)
        .where(MarketplaceSuite.is_published.is_(True))
    )
    total_result = await session.execute(total_stmt)
    total_suites = total_result.scalar() or 0

    downloads_stmt = select(func.sum(MarketplaceSuite.total_downloads)).where(
        MarketplaceSuite.is_published.is_(True)
    )
    downloads_result = await session.execute(downloads_stmt)
    total_downloads = downloads_result.scalar() or 0

    installs_stmt = (
        select(func.count())
        .select_from(MarketplaceSuiteInstall)
        .where(MarketplaceSuiteInstall.is_active.is_(True))
    )
    installs_result = await session.execute(installs_stmt)
    total_installs = installs_result.scalar() or 0

    reviews_stmt = (
        select(func.count())
        .select_from(MarketplaceSuiteReview)
        .where(MarketplaceSuiteReview.is_approved.is_(True))
    )
    reviews_result = await session.execute(reviews_stmt)
    total_reviews = reviews_result.scalar() or 0

    # Category stats
    cat_stmt = (
        select(
            MarketplaceSuite.category,
            func.count().label("count"),
            func.sum(MarketplaceSuite.total_downloads).label("downloads"),
            func.avg(MarketplaceSuite.average_rating).label("avg_rating"),
        )
        .where(MarketplaceSuite.is_published.is_(True))
        .group_by(MarketplaceSuite.category)
        .order_by(desc("count"))
    )
    cat_result = await session.execute(cat_stmt)
    categories = [
        MarketplaceCategoryStats(
            category=row.category,
            count=row.count,
            total_downloads=row.downloads or 0,
            average_rating=row.avg_rating,
        )
        for row in cat_result
    ]

    # Top tags (simplified - just count tag occurrences)
    # In a real implementation, you'd want a more sophisticated tag analysis
    top_tags: list[tuple[str, int]] = []

    stats = MarketplaceStats(
        total_suites=total_suites,
        total_downloads=total_downloads,
        total_installs=total_installs,
        total_reviews=total_reviews,
        categories=categories,
        top_tags=top_tags,
    )

    cache.put(cache_key, stats.model_dump())
    return stats


@router.get("/categories", response_model=list[str])
async def list_categories(
    session: DBSession,
) -> list[str]:
    """List all unique categories in the marketplace.

    This is a public endpoint that does not require authentication.

    Returns:
        List of category names.
    """
    stmt = (
        select(MarketplaceSuite.category)
        .where(MarketplaceSuite.is_published.is_(True))
        .distinct()
        .order_by(MarketplaceSuite.category)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


@router.get("/{slug}", response_model=MarketplaceSuiteDetail)
async def get_marketplace_suite(
    session: DBSession,
    slug: str,
) -> MarketplaceSuiteDetail:
    """Get a marketplace suite by slug.

    This is a public endpoint that does not require authentication.

    Args:
        session: Database session.
        slug: Suite slug.

    Returns:
        Detailed suite information.

    Raises:
        HTTPException: If suite not found.
    """
    stmt = (
        select(MarketplaceSuite)
        .where(
            MarketplaceSuite.slug == slug,
            MarketplaceSuite.is_published.is_(True),
        )
        .options(
            selectinload(MarketplaceSuite.versions),
            selectinload(MarketplaceSuite.reviews).selectinload(
                MarketplaceSuiteReview.user
            ),
        )
    )
    result = await session.execute(stmt)
    suite = result.scalar_one_or_none()

    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{slug}' not found",
        )

    # Get test count from latest version
    test_count = 0
    if suite.versions:
        latest = next((v for v in suite.versions if v.is_latest), None)
        if latest:
            test_count = latest.test_count

    # Build response
    versions = [
        MarketplaceSuiteVersionResponse.model_validate(v)
        for v in suite.versions[:10]  # Limit to 10 most recent
    ]

    recent_reviews = [
        MarketplaceSuiteReviewResponse(
            **MarketplaceSuiteReviewResponse.model_validate(r).model_dump(),
            username=r.user.username if r.user else "Unknown",
        )
        for r in suite.reviews[:5]  # Limit to 5 most recent
        if r.is_approved
    ]

    return MarketplaceSuiteDetail(
        **MarketplaceSuiteResponse.model_validate(suite).model_dump(),
        versions=versions,
        recent_reviews=recent_reviews,
        test_count=test_count,
    )


@router.get("/{slug}/versions", response_model=MarketplaceSuiteVersionList)
async def list_suite_versions(
    session: DBSession,
    slug: str,
) -> MarketplaceSuiteVersionList:
    """List all versions of a marketplace suite.

    This is a public endpoint that does not require authentication.

    Args:
        session: Database session.
        slug: Suite slug.

    Returns:
        List of versions.

    Raises:
        HTTPException: If suite not found.
    """
    # Get suite
    suite_stmt = select(MarketplaceSuite).where(
        MarketplaceSuite.slug == slug,
        MarketplaceSuite.is_published.is_(True),
    )
    suite_result = await session.execute(suite_stmt)
    suite = suite_result.scalar_one_or_none()

    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{slug}' not found",
        )

    # Get versions
    stmt = (
        select(MarketplaceSuiteVersion)
        .where(MarketplaceSuiteVersion.marketplace_suite_id == suite.id)
        .order_by(
            desc(MarketplaceSuiteVersion.version_major),
            desc(MarketplaceSuiteVersion.version_minor),
            desc(MarketplaceSuiteVersion.version_patch),
        )
    )
    result = await session.execute(stmt)
    versions = list(result.scalars().all())

    return MarketplaceSuiteVersionList(
        items=[MarketplaceSuiteVersionResponse.model_validate(v) for v in versions],
        total=len(versions),
    )


@router.get("/{slug}/reviews", response_model=MarketplaceSuiteReviewList)
async def list_suite_reviews(
    session: DBSession,
    slug: str,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
) -> MarketplaceSuiteReviewList:
    """List reviews for a marketplace suite.

    This is a public endpoint that does not require authentication.

    Args:
        session: Database session.
        slug: Suite slug.
        limit: Maximum results.
        offset: Pagination offset.

    Returns:
        Paginated list of reviews.

    Raises:
        HTTPException: If suite not found.
    """
    # Get suite
    suite_stmt = select(MarketplaceSuite).where(
        MarketplaceSuite.slug == slug,
        MarketplaceSuite.is_published.is_(True),
    )
    suite_result = await session.execute(suite_stmt)
    suite = suite_result.scalar_one_or_none()

    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{slug}' not found",
        )

    # Count total approved reviews
    count_stmt = (
        select(func.count())
        .select_from(MarketplaceSuiteReview)
        .where(
            MarketplaceSuiteReview.marketplace_suite_id == suite.id,
            MarketplaceSuiteReview.is_approved.is_(True),
        )
    )
    count_result = await session.execute(count_stmt)
    total = count_result.scalar() or 0

    # Get rating distribution
    dist_stmt = (
        select(
            MarketplaceSuiteReview.rating,
            func.count().label("count"),
        )
        .where(
            MarketplaceSuiteReview.marketplace_suite_id == suite.id,
            MarketplaceSuiteReview.is_approved.is_(True),
        )
        .group_by(MarketplaceSuiteReview.rating)
    )
    dist_result = await session.execute(dist_stmt)
    rating_distribution = {row.rating: row.count for row in dist_result}

    # Get reviews
    stmt = (
        select(MarketplaceSuiteReview)
        .where(
            MarketplaceSuiteReview.marketplace_suite_id == suite.id,
            MarketplaceSuiteReview.is_approved.is_(True),
        )
        .options(selectinload(MarketplaceSuiteReview.user))
        .order_by(desc(MarketplaceSuiteReview.created_at))
        .offset(offset)
        .limit(limit)
    )
    result = await session.execute(stmt)
    reviews = list(result.scalars().all())

    return MarketplaceSuiteReviewList(
        items=[
            MarketplaceSuiteReviewResponse(
                **MarketplaceSuiteReviewResponse.model_validate(r).model_dump(),
                username=r.user.username if r.user else "Unknown",
            )
            for r in reviews
        ],
        total=total,
        limit=limit,
        offset=offset,
        average_rating=suite.average_rating,
        rating_distribution=rating_distribution,
    )


@router.get("/{slug}/download")
async def download_suite(
    session: DBSession,
    slug: str,
    version: str | None = Query(None),
) -> dict[str, Any]:
    """Download a marketplace suite.

    This is a public endpoint that does not require authentication.
    Increments download count.

    Args:
        session: Database session.
        slug: Suite slug.
        version: Specific version to download (default: latest).

    Returns:
        Suite content as JSON.

    Raises:
        HTTPException: If suite or version not found.
    """
    # Get suite
    suite_stmt = select(MarketplaceSuite).where(
        MarketplaceSuite.slug == slug,
        MarketplaceSuite.is_published.is_(True),
    )
    suite_result = await session.execute(suite_stmt)
    suite = suite_result.scalar_one_or_none()

    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{slug}' not found",
        )

    # Get version
    version_stmt = select(MarketplaceSuiteVersion).where(
        MarketplaceSuiteVersion.marketplace_suite_id == suite.id
    )
    if version:
        version_stmt = version_stmt.where(MarketplaceSuiteVersion.version == version)
    else:
        version_stmt = version_stmt.where(MarketplaceSuiteVersion.is_latest.is_(True))

    version_result = await session.execute(version_stmt)
    suite_version = version_result.scalar_one_or_none()

    if suite_version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version '{version or 'latest'}' not found",
        )

    # Increment download counts
    suite.total_downloads += 1
    suite_version.downloads += 1
    await session.commit()

    # Invalidate cache
    cache = get_marketplace_cache()
    cache.invalidate_prefix("stats")

    return {
        "name": suite.name,
        "slug": suite.slug,
        "version": suite_version.version,
        "license": suite.license_type,
        "content": suite_version.suite_content,
    }


# ==================== Authenticated Endpoints ====================


@router.post(
    "",
    response_model=MarketplaceSuiteResponse,
    status_code=status.HTTP_201_CREATED,
)
async def publish_suite(
    session: DBSession,
    data: MarketplaceSuiteCreate,
    user: RequiredUser,
    _: Annotated[None, Depends(require_permission(Permission.MARKETPLACE_PUBLISH))],
) -> MarketplaceSuiteResponse:
    """Publish a test suite to the marketplace.

    Requires MARKETPLACE_PUBLISH permission.

    Args:
        session: Database session.
        data: Suite creation data.
        user: Current user.

    Returns:
        The published suite.

    Raises:
        HTTPException: If slug already exists.
    """
    # Check slug uniqueness
    existing_stmt = select(MarketplaceSuite).where(MarketplaceSuite.slug == data.slug)
    existing_result = await session.execute(existing_stmt)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Suite with slug '{data.slug}' already exists",
        )

    # Parse version
    major, minor, patch = parse_semver(data.version)

    # Calculate content hash
    content_json = json.dumps(data.suite_content, sort_keys=True)
    content_hash = hashlib.sha256(content_json.encode()).hexdigest()

    # Create suite
    suite = MarketplaceSuite(
        tenant_id=user.tenant_id,
        name=data.name,
        slug=data.slug,
        description=data.description,
        short_description=data.short_description,
        publisher_id=user.id,
        publisher_name=user.username,
        category=data.category,
        tags=data.tags,
        license_type=data.license_type,
        license_url=data.license_url,
        source_type=data.source_type,
        github_url=data.github_url,
        github_branch=data.github_branch,
        github_path=data.github_path,
        latest_version=data.version,
        published_at=datetime.now(),
    )
    session.add(suite)
    await session.flush()

    # Create initial version
    test_count = len(data.suite_content.get("tests", []))
    version = MarketplaceSuiteVersion(
        marketplace_suite_id=suite.id,
        version=data.version,
        version_major=major,
        version_minor=minor,
        version_patch=patch,
        changelog=data.changelog,
        suite_content=data.suite_content,
        test_count=test_count,
        file_size_bytes=len(content_json),
        content_hash=content_hash,
        is_latest=True,
        released_at=datetime.now(),
    )
    session.add(version)
    await session.flush()

    suite.latest_version_id = version.id
    await session.commit()

    # Invalidate cache
    cache = get_marketplace_cache()
    cache.invalidate_prefix("stats")

    return MarketplaceSuiteResponse.model_validate(suite)


@router.patch("/{slug}", response_model=MarketplaceSuiteResponse)
async def update_suite(
    session: DBSession,
    slug: str,
    data: MarketplaceSuiteUpdate,
    user: RequiredUser,
    _: Annotated[None, Depends(require_permission(Permission.MARKETPLACE_WRITE))],
) -> MarketplaceSuiteResponse:
    """Update a marketplace suite.

    Requires MARKETPLACE_WRITE permission. Only the publisher can update.

    Args:
        session: Database session.
        slug: Suite slug.
        data: Update data.
        user: Current user.

    Returns:
        The updated suite.

    Raises:
        HTTPException: If suite not found or user is not the publisher.
    """
    stmt = select(MarketplaceSuite).where(MarketplaceSuite.slug == slug)
    result = await session.execute(stmt)
    suite = result.scalar_one_or_none()

    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{slug}' not found",
        )

    # Check ownership
    if suite.publisher_id != user.id and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own suites",
        )

    # Apply updates
    if data.name is not None:
        suite.name = data.name
    if data.description is not None:
        suite.description = data.description
    if data.short_description is not None:
        suite.short_description = data.short_description
    if data.category is not None:
        suite.category = data.category
    if data.tags is not None:
        suite.tags = data.tags
    if data.license_type is not None:
        suite.license_type = data.license_type
    if data.license_url is not None:
        suite.license_url = data.license_url
    if data.github_url is not None:
        suite.github_url = data.github_url
    if data.github_branch is not None:
        suite.github_branch = data.github_branch
    if data.github_path is not None:
        suite.github_path = data.github_path
    if data.is_published is not None:
        suite.is_published = data.is_published
        if data.is_published and suite.published_at is None:
            suite.published_at = datetime.now()

    await session.commit()

    # Invalidate cache
    cache = get_marketplace_cache()
    cache.invalidate_prefix("stats")

    return MarketplaceSuiteResponse.model_validate(suite)


@router.delete("/{slug}", status_code=status.HTTP_204_NO_CONTENT)
async def unpublish_suite(
    session: DBSession,
    slug: str,
    user: RequiredUser,
    _: Annotated[None, Depends(require_permission(Permission.MARKETPLACE_DELETE))],
) -> None:
    """Unpublish (soft delete) a marketplace suite.

    Requires MARKETPLACE_DELETE permission. Only the publisher can unpublish.

    Args:
        session: Database session.
        slug: Suite slug.
        user: Current user.

    Raises:
        HTTPException: If suite not found or user is not the publisher.
    """
    stmt = select(MarketplaceSuite).where(MarketplaceSuite.slug == slug)
    result = await session.execute(stmt)
    suite = result.scalar_one_or_none()

    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{slug}' not found",
        )

    # Check ownership
    if suite.publisher_id != user.id and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only unpublish your own suites",
        )

    suite.is_published = False
    await session.commit()

    # Invalidate cache
    cache = get_marketplace_cache()
    cache.invalidate_prefix("stats")


@router.post(
    "/{slug}/versions",
    response_model=MarketplaceSuiteVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_version(
    session: DBSession,
    slug: str,
    data: MarketplaceSuiteVersionCreate,
    user: RequiredUser,
    _: Annotated[None, Depends(require_permission(Permission.MARKETPLACE_WRITE))],
) -> MarketplaceSuiteVersionResponse:
    """Create a new version of a marketplace suite.

    Requires MARKETPLACE_WRITE permission. Only the publisher can add versions.

    Args:
        session: Database session.
        slug: Suite slug.
        data: Version data.
        user: Current user.

    Returns:
        The created version.

    Raises:
        HTTPException: If suite not found, user not publisher, or version exists.
    """
    # Get suite
    suite_stmt = select(MarketplaceSuite).where(MarketplaceSuite.slug == slug)
    suite_result = await session.execute(suite_stmt)
    suite = suite_result.scalar_one_or_none()

    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{slug}' not found",
        )

    # Check ownership
    if suite.publisher_id != user.id and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only add versions to your own suites",
        )

    # Check version doesn't exist
    existing_stmt = select(MarketplaceSuiteVersion).where(
        MarketplaceSuiteVersion.marketplace_suite_id == suite.id,
        MarketplaceSuiteVersion.version == data.version,
    )
    existing_result = await session.execute(existing_stmt)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Version '{data.version}' already exists",
        )

    # Parse version
    major, minor, patch = parse_semver(data.version)

    # Calculate content hash
    content_json = json.dumps(data.suite_content, sort_keys=True)
    content_hash = hashlib.sha256(content_json.encode()).hexdigest()

    # Unmark current latest
    await session.execute(
        MarketplaceSuiteVersion.__table__.update()
        .where(MarketplaceSuiteVersion.marketplace_suite_id == suite.id)
        .values(is_latest=False)
    )

    # Create version
    test_count = len(data.suite_content.get("tests", []))
    version = MarketplaceSuiteVersion(
        marketplace_suite_id=suite.id,
        version=data.version,
        version_major=major,
        version_minor=minor,
        version_patch=patch,
        changelog=data.changelog,
        breaking_changes=data.breaking_changes,
        suite_content=data.suite_content,
        test_count=test_count,
        file_size_bytes=len(content_json),
        content_hash=content_hash,
        is_latest=True,
        released_at=datetime.now(),
    )
    session.add(version)
    await session.flush()

    # Update suite
    suite.latest_version = data.version
    suite.latest_version_id = version.id
    await session.commit()

    return MarketplaceSuiteVersionResponse.model_validate(version)


@router.patch(
    "/{slug}/versions/{version_str}",
    response_model=MarketplaceSuiteVersionResponse,
)
async def update_version(
    session: DBSession,
    slug: str,
    version_str: str,
    data: MarketplaceSuiteVersionUpdate,
    user: RequiredUser,
    _: Annotated[None, Depends(require_permission(Permission.MARKETPLACE_WRITE))],
) -> MarketplaceSuiteVersionResponse:
    """Update a version (changelog, deprecation).

    Requires MARKETPLACE_WRITE permission. Only the publisher can update.

    Args:
        session: Database session.
        slug: Suite slug.
        version_str: Version string.
        data: Update data.
        user: Current user.

    Returns:
        The updated version.
    """
    # Get suite
    suite_stmt = select(MarketplaceSuite).where(MarketplaceSuite.slug == slug)
    suite_result = await session.execute(suite_stmt)
    suite = suite_result.scalar_one_or_none()

    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{slug}' not found",
        )

    # Check ownership
    if suite.publisher_id != user.id and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own suites",
        )

    # Get version
    version_stmt = select(MarketplaceSuiteVersion).where(
        MarketplaceSuiteVersion.marketplace_suite_id == suite.id,
        MarketplaceSuiteVersion.version == version_str,
    )
    version_result = await session.execute(version_stmt)
    version = version_result.scalar_one_or_none()

    if version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version '{version_str}' not found",
        )

    # Apply updates
    if data.changelog is not None:
        version.changelog = data.changelog
    if data.is_deprecated is not None:
        version.is_deprecated = data.is_deprecated
    if data.deprecation_message is not None:
        version.deprecation_message = data.deprecation_message

    await session.commit()

    return MarketplaceSuiteVersionResponse.model_validate(version)


@router.post(
    "/{slug}/reviews",
    response_model=MarketplaceSuiteReviewResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_review(
    session: DBSession,
    slug: str,
    data: MarketplaceSuiteReviewCreate,
    user: RequiredUser,
    _: Annotated[None, Depends(require_permission(Permission.MARKETPLACE_REVIEW))],
) -> MarketplaceSuiteReviewResponse:
    """Add a review for a marketplace suite.

    Requires MARKETPLACE_REVIEW permission. Users can only review each suite once.

    Args:
        session: Database session.
        slug: Suite slug.
        data: Review data.
        user: Current user.

    Returns:
        The created review.

    Raises:
        HTTPException: If suite not found or user already reviewed.
    """
    # Get suite
    suite_stmt = select(MarketplaceSuite).where(
        MarketplaceSuite.slug == slug,
        MarketplaceSuite.is_published.is_(True),
    )
    suite_result = await session.execute(suite_stmt)
    suite = suite_result.scalar_one_or_none()

    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{slug}' not found",
        )

    # Check for existing review
    existing_stmt = select(MarketplaceSuiteReview).where(
        MarketplaceSuiteReview.marketplace_suite_id == suite.id,
        MarketplaceSuiteReview.user_id == user.id,
    )
    existing_result = await session.execute(existing_stmt)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You have already reviewed this suite",
        )

    # Create review
    review = MarketplaceSuiteReview(
        marketplace_suite_id=suite.id,
        user_id=user.id,
        rating=data.rating,
        title=data.title,
        content=data.content,
        version_reviewed=data.version_reviewed,
    )
    session.add(review)

    # Update suite rating
    suite.total_ratings += 1
    if suite.average_rating is None:
        suite.average_rating = float(data.rating)
    else:
        # Recalculate average
        total = suite.average_rating * (suite.total_ratings - 1) + data.rating
        suite.average_rating = total / suite.total_ratings

    await session.commit()

    # Invalidate cache
    cache = get_marketplace_cache()
    cache.invalidate_prefix("stats")

    return MarketplaceSuiteReviewResponse(
        **MarketplaceSuiteReviewResponse.model_validate(review).model_dump(),
        username=user.username,
    )


@router.patch("/reviews/{review_id}", response_model=MarketplaceSuiteReviewResponse)
async def update_review(
    session: DBSession,
    review_id: int,
    data: MarketplaceSuiteReviewUpdate,
    user: RequiredUser,
) -> MarketplaceSuiteReviewResponse:
    """Update a review.

    Only the review author can update.

    Args:
        session: Database session.
        review_id: Review ID.
        data: Update data.
        user: Current user.

    Returns:
        The updated review.

    Raises:
        HTTPException: If review not found or user is not the author.
    """
    stmt = select(MarketplaceSuiteReview).where(MarketplaceSuiteReview.id == review_id)
    result = await session.execute(stmt)
    review = result.scalar_one_or_none()

    if review is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review {review_id} not found",
        )

    # Check ownership
    if review.user_id != user.id and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own reviews",
        )

    old_rating = review.rating

    # Apply updates
    if data.rating is not None:
        review.rating = data.rating
    if data.title is not None:
        review.title = data.title
    if data.content is not None:
        review.content = data.content

    # Update suite rating if rating changed
    if data.rating is not None and data.rating != old_rating:
        suite = await session.get(MarketplaceSuite, review.marketplace_suite_id)
        if suite and suite.average_rating is not None:
            total = (
                suite.average_rating * suite.total_ratings - old_rating + data.rating
            )
            suite.average_rating = total / suite.total_ratings

    await session.commit()

    return MarketplaceSuiteReviewResponse(
        **MarketplaceSuiteReviewResponse.model_validate(review).model_dump(),
        username=user.username,
    )


@router.delete("/reviews/{review_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_review(
    session: DBSession,
    review_id: int,
    user: RequiredUser,
) -> None:
    """Delete a review.

    Only the review author can delete.

    Args:
        session: Database session.
        review_id: Review ID.
        user: Current user.

    Raises:
        HTTPException: If review not found or user is not the author.
    """
    stmt = select(MarketplaceSuiteReview).where(MarketplaceSuiteReview.id == review_id)
    result = await session.execute(stmt)
    review = result.scalar_one_or_none()

    if review is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review {review_id} not found",
        )

    # Check ownership
    if review.user_id != user.id and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own reviews",
        )

    # Update suite rating
    suite = await session.get(MarketplaceSuite, review.marketplace_suite_id)
    if suite and suite.total_ratings > 0:
        if suite.total_ratings == 1:
            suite.average_rating = None
        elif suite.average_rating is not None:
            total = suite.average_rating * suite.total_ratings - review.rating
            suite.average_rating = total / (suite.total_ratings - 1)
        suite.total_ratings -= 1

    await session.delete(review)
    await session.commit()


@router.post(
    "/{slug}/install",
    response_model=MarketplaceSuiteInstallResponse,
    status_code=status.HTTP_201_CREATED,
)
async def install_suite(
    session: DBSession,
    slug: str,
    user: RequiredUser,
    version: str | None = Query(None),
) -> MarketplaceSuiteInstallResponse:
    """Install a marketplace suite.

    Requires authentication. Creates an install record.

    Args:
        session: Database session.
        slug: Suite slug.
        user: Current user.
        version: Specific version to install (default: latest).

    Returns:
        The install record.

    Raises:
        HTTPException: If suite or version not found.
    """
    # Get suite
    suite_stmt = select(MarketplaceSuite).where(
        MarketplaceSuite.slug == slug,
        MarketplaceSuite.is_published.is_(True),
    )
    suite_result = await session.execute(suite_stmt)
    suite = suite_result.scalar_one_or_none()

    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{slug}' not found",
        )

    # Get version
    version_stmt = select(MarketplaceSuiteVersion).where(
        MarketplaceSuiteVersion.marketplace_suite_id == suite.id
    )
    if version:
        version_stmt = version_stmt.where(MarketplaceSuiteVersion.version == version)
    else:
        version_stmt = version_stmt.where(MarketplaceSuiteVersion.is_latest.is_(True))

    version_result = await session.execute(version_stmt)
    suite_version = version_result.scalar_one_or_none()

    if suite_version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version '{version or 'latest'}' not found",
        )

    # Check for existing active install
    existing_stmt = select(MarketplaceSuiteInstall).where(
        MarketplaceSuiteInstall.marketplace_suite_id == suite.id,
        MarketplaceSuiteInstall.user_id == user.id,
        MarketplaceSuiteInstall.is_active.is_(True),
    )
    existing_result = await session.execute(existing_stmt)
    existing = existing_result.scalar_one_or_none()

    if existing:
        # Update existing install
        existing.version_installed = suite_version.version
        existing.version_id = suite_version.id
        existing.updated_at = datetime.now()
        await session.commit()
        return MarketplaceSuiteInstallResponse.model_validate(existing)

    # Create install record
    install = MarketplaceSuiteInstall(
        tenant_id=user.tenant_id,
        marketplace_suite_id=suite.id,
        user_id=user.id,
        version_installed=suite_version.version,
        version_id=suite_version.id,
    )
    session.add(install)

    # Update install count
    suite.total_installs += 1

    await session.commit()

    # Invalidate cache
    cache = get_marketplace_cache()
    cache.invalidate_prefix("stats")

    return MarketplaceSuiteInstallResponse.model_validate(install)


@router.delete("/{slug}/install", status_code=status.HTTP_204_NO_CONTENT)
async def uninstall_suite(
    session: DBSession,
    slug: str,
    user: RequiredUser,
) -> None:
    """Uninstall a marketplace suite.

    Requires authentication. Marks the install as inactive.

    Args:
        session: Database session.
        slug: Suite slug.
        user: Current user.

    Raises:
        HTTPException: If suite not found or not installed.
    """
    # Get suite
    suite_stmt = select(MarketplaceSuite).where(MarketplaceSuite.slug == slug)
    suite_result = await session.execute(suite_stmt)
    suite = suite_result.scalar_one_or_none()

    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{slug}' not found",
        )

    # Get install
    install_stmt = select(MarketplaceSuiteInstall).where(
        MarketplaceSuiteInstall.marketplace_suite_id == suite.id,
        MarketplaceSuiteInstall.user_id == user.id,
        MarketplaceSuiteInstall.is_active.is_(True),
    )
    install_result = await session.execute(install_stmt)
    install = install_result.scalar_one_or_none()

    if install is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Suite is not installed",
        )

    install.is_active = False
    install.uninstalled_at = datetime.now()
    suite.total_installs = max(0, suite.total_installs - 1)

    await session.commit()

    # Invalidate cache
    cache = get_marketplace_cache()
    cache.invalidate_prefix("stats")


@router.get("/installed", response_model=MarketplaceSuiteInstallList)
async def list_installed_suites(
    session: DBSession,
    user: RequiredUser,
) -> MarketplaceSuiteInstallList:
    """List suites installed by the current user.

    Requires authentication.

    Args:
        session: Database session.
        user: Current user.

    Returns:
        List of installed suites.
    """
    stmt = (
        select(MarketplaceSuiteInstall)
        .where(
            MarketplaceSuiteInstall.user_id == user.id,
            MarketplaceSuiteInstall.is_active.is_(True),
        )
        .order_by(desc(MarketplaceSuiteInstall.installed_at))
    )
    result = await session.execute(stmt)
    installs = list(result.scalars().all())

    return MarketplaceSuiteInstallList(
        items=[MarketplaceSuiteInstallResponse.model_validate(i) for i in installs],
        total=len(installs),
    )


@router.post("/import/github", response_model=GitHubImportResponse)
async def import_from_github(
    session: DBSession,
    data: GitHubImportRequest,
    user: RequiredUser,
    _: Annotated[None, Depends(require_permission(Permission.MARKETPLACE_PUBLISH))],
) -> GitHubImportResponse:
    """Import a test suite from GitHub.

    Requires MARKETPLACE_PUBLISH permission.

    Args:
        session: Database session.
        data: GitHub import request.
        user: Current user.

    Returns:
        Import result with the created suite.
    """
    # Note: In a real implementation, this would:
    # 1. Parse the GitHub URL to get owner/repo
    # 2. Use GitHub API to fetch the repository contents
    # 3. Download and parse the test suite YAML file(s)
    # 4. Create the marketplace suite and version

    # For now, return a placeholder response
    # The actual implementation would require:
    # - GitHub API integration (httpx for async HTTP)
    # - YAML parsing
    # - Validation of test suite format

    return GitHubImportResponse(
        success=False,
        error="GitHub import is not yet implemented. "
        "Please publish your test suite manually.",
        files_imported=[],
    )


# ==================== Admin Endpoints ====================


@router.post("/{slug}/feature", status_code=status.HTTP_200_OK)
async def toggle_featured(
    session: DBSession,
    slug: str,
    admin: AdminUser,
    featured: bool = Query(default=True),
) -> MarketplaceSuiteResponse:
    """Feature or unfeature a marketplace suite.

    Requires admin privileges.

    Args:
        session: Database session.
        slug: Suite slug.
        admin: Admin user.
        featured: Whether to feature the suite.

    Returns:
        The updated suite.
    """
    stmt = select(MarketplaceSuite).where(MarketplaceSuite.slug == slug)
    result = await session.execute(stmt)
    suite = result.scalar_one_or_none()

    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{slug}' not found",
        )

    suite.is_featured = featured
    await session.commit()

    # Invalidate cache
    cache = get_marketplace_cache()
    cache.invalidate_prefix("stats")

    return MarketplaceSuiteResponse.model_validate(suite)


@router.post("/{slug}/verify", status_code=status.HTTP_200_OK)
async def verify_suite(
    session: DBSession,
    slug: str,
    admin: AdminUser,
    verified: bool = Query(default=True),
) -> MarketplaceSuiteResponse:
    """Verify or unverify a marketplace suite.

    Requires admin privileges.

    Args:
        session: Database session.
        slug: Suite slug.
        admin: Admin user.
        verified: Whether to verify the suite.

    Returns:
        The updated suite.
    """
    stmt = select(MarketplaceSuite).where(MarketplaceSuite.slug == slug)
    result = await session.execute(stmt)
    suite = result.scalar_one_or_none()

    if suite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite '{slug}' not found",
        )

    suite.is_verified = verified
    await session.commit()

    # Invalidate cache
    cache = get_marketplace_cache()
    cache.invalidate_prefix("stats")

    return MarketplaceSuiteResponse.model_validate(suite)


@router.post("/reviews/{review_id}/flag", status_code=status.HTTP_200_OK)
async def flag_review(
    session: DBSession,
    review_id: int,
    admin: AdminUser,
    flagged: bool = Query(default=True),
) -> MarketplaceSuiteReviewResponse:
    """Flag or unflag a review for moderation.

    Requires admin privileges.

    Args:
        session: Database session.
        review_id: Review ID.
        admin: Admin user.
        flagged: Whether to flag the review.

    Returns:
        The updated review.
    """
    stmt = (
        select(MarketplaceSuiteReview)
        .where(MarketplaceSuiteReview.id == review_id)
        .options(selectinload(MarketplaceSuiteReview.user))
    )
    result = await session.execute(stmt)
    review = result.scalar_one_or_none()

    if review is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review {review_id} not found",
        )

    review.is_flagged = flagged
    review.is_approved = not flagged
    await session.commit()

    return MarketplaceSuiteReviewResponse(
        **MarketplaceSuiteReviewResponse.model_validate(review).model_dump(),
        username=review.user.username if review.user else "Unknown",
    )
