"""Public leaderboard routes.

This module provides public endpoints for viewing benchmark leaderboards,
agent profiles, and publishing results. Most read endpoints are public,
while write operations require authentication.

Public endpoints (no auth required):
    - GET /public/leaderboard: Get public leaderboard
    - GET /public/leaderboard/categories: List benchmark categories
    - GET /public/leaderboard/categories/{slug}: Get category details
    - GET /public/leaderboard/agents/{id}: Get public agent profile
    - GET /public/leaderboard/history/{category}: Get historical trends

Authenticated endpoints:
    - POST /public/leaderboard/publish: Publish a result (RESULTS_WRITE)
    - POST /public/leaderboard/agents: Create agent profile (AGENTS_WRITE)
    - PATCH /public/leaderboard/agents/{id}: Update profile (owner only)
    - POST /public/leaderboard/verify: Add verification badge (admin)
"""

import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, func, select
from sqlalchemy.orm import selectinload

from atp.dashboard.models import (
    AgentProfile,
    BenchmarkCategory,
    PublishedResult,
    SuiteExecution,
)
from atp.dashboard.query_cache import QueryCache
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.schemas import (
    AgentLeaderboardHistory,
    AgentProfileCreate,
    AgentProfileList,
    AgentProfileResponse,
    AgentProfileSummary,
    AgentProfileUpdate,
    BenchmarkCategoryCreate,
    BenchmarkCategoryList,
    BenchmarkCategoryResponse,
    BenchmarkCategoryUpdate,
    LeaderboardEntry,
    LeaderboardTrendPoint,
    PublicLeaderboardResponse,
    PublishedResultList,
    PublishedResultResponse,
    PublishedResultWithProfile,
    PublishResultRequest,
    VerificationBadgeRequest,
)
from atp.dashboard.v2.dependencies import (
    AdminUser,
    DBSession,
    RequiredUser,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/public/leaderboard", tags=["public-leaderboard"])

# Cache for public leaderboard (longer TTL since it's public data)
_public_leaderboard_cache: QueryCache[dict] | None = None


def get_public_leaderboard_cache() -> QueryCache[dict]:
    """Get the public leaderboard cache (5 minute TTL)."""
    global _public_leaderboard_cache
    if _public_leaderboard_cache is None:
        _public_leaderboard_cache = QueryCache(max_size=100, ttl_seconds=300)
    return _public_leaderboard_cache


# ==================== Public Endpoints (No Auth) ====================


@router.get("/categories", response_model=BenchmarkCategoryList)
async def list_categories(
    session: DBSession,
    active_only: bool = Query(default=True),
) -> BenchmarkCategoryList:
    """List all benchmark categories.

    This is a public endpoint that does not require authentication.

    Args:
        session: Database session.
        active_only: Only return active categories (default True).

    Returns:
        List of benchmark categories.
    """
    stmt = select(BenchmarkCategory).order_by(
        BenchmarkCategory.display_order, BenchmarkCategory.name
    )
    if active_only:
        stmt = stmt.where(BenchmarkCategory.is_active.is_(True))

    result = await session.execute(stmt)
    categories = list(result.scalars().all())

    return BenchmarkCategoryList(
        items=[BenchmarkCategoryResponse.model_validate(c) for c in categories],
        total=len(categories),
    )


@router.get("/categories/{slug}", response_model=BenchmarkCategoryResponse)
async def get_category(
    session: DBSession,
    slug: str,
) -> BenchmarkCategoryResponse:
    """Get a benchmark category by slug.

    This is a public endpoint that does not require authentication.

    Args:
        session: Database session.
        slug: Category slug.

    Returns:
        Benchmark category details.

    Raises:
        HTTPException: If category not found.
    """
    stmt = select(BenchmarkCategory).where(BenchmarkCategory.slug == slug)
    result = await session.execute(stmt)
    category = result.scalar_one_or_none()

    if category is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category '{slug}' not found",
        )

    return BenchmarkCategoryResponse.model_validate(category)


@router.get("", response_model=PublicLeaderboardResponse)
async def get_public_leaderboard(
    session: DBSession,
    category: str = Query(..., description="Category slug"),
    verified_only: bool = Query(default=False),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> PublicLeaderboardResponse:
    """Get the public leaderboard for a category.

    This is a public endpoint that does not require authentication.
    Results are cached for 5 minutes for performance.

    Args:
        session: Database session.
        category: Benchmark category slug.
        verified_only: Only show verified results.
        limit: Maximum results to return.
        offset: Pagination offset.

    Returns:
        Public leaderboard with rankings.

    Raises:
        HTTPException: If category not found.
    """
    # Check cache
    cache = get_public_leaderboard_cache()
    cache_key = f"leaderboard:{category}:{verified_only}:{limit}:{offset}"
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return PublicLeaderboardResponse(**cached_result)

    # Get category
    cat_stmt = select(BenchmarkCategory).where(
        BenchmarkCategory.slug == category,
        BenchmarkCategory.is_active.is_(True),
    )
    cat_result = await session.execute(cat_stmt)
    cat_obj = cat_result.scalar_one_or_none()

    if cat_obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category '{category}' not found",
        )

    # Build query for leaderboard
    stmt = (
        select(PublishedResult)
        .where(PublishedResult.category == category)
        .options(selectinload(PublishedResult.agent_profile))
        .order_by(desc(PublishedResult.score))
    )

    if verified_only:
        stmt = stmt.where(PublishedResult.is_verified.is_(True))

    # Only show public profiles
    stmt = stmt.join(AgentProfile).where(AgentProfile.is_public.is_(True))

    # Count total
    count_stmt = (
        select(func.count())
        .select_from(PublishedResult)
        .join(AgentProfile)
        .where(
            PublishedResult.category == category,
            AgentProfile.is_public.is_(True),
        )
    )
    if verified_only:
        count_stmt = count_stmt.where(PublishedResult.is_verified.is_(True))

    count_result = await session.execute(count_stmt)
    total = count_result.scalar() or 0

    # Apply pagination
    stmt = stmt.offset(offset).limit(limit)

    result = await session.execute(stmt)
    results = list(result.scalars().all())

    # Build leaderboard entries
    entries: list[LeaderboardEntry] = []
    for rank, pub_result in enumerate(results, start=offset + 1):
        profile = pub_result.agent_profile
        profile_summary = AgentProfileSummary(
            id=profile.id,
            display_name=profile.display_name,
            avatar_url=profile.avatar_url,
            is_verified=profile.is_verified,
            verification_badges=profile.verification_badges,
            total_submissions=profile.total_submissions,
            best_overall_score=profile.best_overall_score,
            best_overall_rank=profile.best_overall_rank,
        )

        # Determine trend based on history
        trend = _calculate_trend(pub_result.score, profile.best_overall_score)

        entries.append(
            LeaderboardEntry(
                rank=rank,
                agent_profile=profile_summary,
                score=pub_result.score,
                success_rate=pub_result.success_rate,
                total_tests=pub_result.total_tests,
                passed_tests=pub_result.passed_tests,
                total_tokens=pub_result.total_tokens,
                total_cost_usd=pub_result.total_cost_usd,
                duration_seconds=pub_result.duration_seconds,
                is_verified=pub_result.is_verified,
                published_at=pub_result.published_at,
                trend=trend,
            )
        )

    response = PublicLeaderboardResponse(
        category=BenchmarkCategoryResponse.model_validate(cat_obj),
        entries=entries,
        total_entries=total,
        last_updated=datetime.now(),
        limit=limit,
        offset=offset,
    )

    # Cache the result
    cache.put(cache_key, response.model_dump())

    return response


def _calculate_trend(current_score: float, best_score: float | None) -> str | None:
    """Calculate trend indicator."""
    if best_score is None:
        return None
    diff = current_score - best_score
    if abs(diff) < 1.0:
        return "stable"
    return "up" if diff > 0 else "down"


@router.get("/agents/{profile_id}", response_model=AgentProfileResponse)
async def get_agent_profile(
    session: DBSession,
    profile_id: int,
) -> AgentProfileResponse:
    """Get a public agent profile.

    This is a public endpoint that does not require authentication.
    Only returns profiles that are marked as public.

    Args:
        session: Database session.
        profile_id: Agent profile ID.

    Returns:
        Agent profile details.

    Raises:
        HTTPException: If profile not found or not public.
    """
    stmt = select(AgentProfile).where(
        AgentProfile.id == profile_id,
        AgentProfile.is_public.is_(True),
    )
    result = await session.execute(stmt)
    profile = result.scalar_one_or_none()

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent profile {profile_id} not found",
        )

    return AgentProfileResponse.model_validate(profile)


@router.get("/agents", response_model=AgentProfileList)
async def list_agent_profiles(
    session: DBSession,
    verified_only: bool = Query(default=False),
    search: str | None = Query(None, max_length=100),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> AgentProfileList:
    """List public agent profiles.

    This is a public endpoint that does not require authentication.

    Args:
        session: Database session.
        verified_only: Only show verified agents.
        search: Search by display name.
        limit: Maximum results.
        offset: Pagination offset.

    Returns:
        Paginated list of agent profiles.
    """
    stmt = select(AgentProfile).where(AgentProfile.is_public.is_(True))

    if verified_only:
        stmt = stmt.where(AgentProfile.is_verified.is_(True))

    if search:
        stmt = stmt.where(AgentProfile.display_name.ilike(f"%{search}%"))

    # Count total
    count_stmt = (
        select(func.count())
        .select_from(AgentProfile)
        .where(AgentProfile.is_public.is_(True))
    )
    if verified_only:
        count_stmt = count_stmt.where(AgentProfile.is_verified.is_(True))
    if search:
        count_stmt = count_stmt.where(AgentProfile.display_name.ilike(f"%{search}%"))

    count_result = await session.execute(count_stmt)
    total = count_result.scalar() or 0

    # Apply pagination and ordering
    stmt = stmt.order_by(
        desc(AgentProfile.best_overall_score.is_not(None)),
        desc(AgentProfile.best_overall_score),
    )
    stmt = stmt.offset(offset).limit(limit)

    result = await session.execute(stmt)
    profiles = list(result.scalars().all())

    return AgentProfileList(
        items=[AgentProfileResponse.model_validate(p) for p in profiles],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/agents/{profile_id}/results", response_model=PublishedResultList)
async def get_agent_published_results(
    session: DBSession,
    profile_id: int,
    category: str | None = Query(None),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
) -> PublishedResultList:
    """Get published results for an agent profile.

    This is a public endpoint that does not require authentication.

    Args:
        session: Database session.
        profile_id: Agent profile ID.
        category: Filter by category.
        limit: Maximum results.
        offset: Pagination offset.

    Returns:
        Paginated list of published results.
    """
    # Verify profile exists and is public
    profile_stmt = select(AgentProfile).where(
        AgentProfile.id == profile_id,
        AgentProfile.is_public.is_(True),
    )
    profile_result = await session.execute(profile_stmt)
    profile = profile_result.scalar_one_or_none()

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent profile {profile_id} not found",
        )

    # Build query
    stmt = (
        select(PublishedResult)
        .where(PublishedResult.agent_profile_id == profile_id)
        .options(selectinload(PublishedResult.agent_profile))
        .order_by(desc(PublishedResult.published_at))
    )

    if category:
        stmt = stmt.where(PublishedResult.category == category)

    # Count total
    count_stmt = (
        select(func.count())
        .select_from(PublishedResult)
        .where(PublishedResult.agent_profile_id == profile_id)
    )
    if category:
        count_stmt = count_stmt.where(PublishedResult.category == category)

    count_result = await session.execute(count_stmt)
    total = count_result.scalar() or 0

    # Apply pagination
    stmt = stmt.offset(offset).limit(limit)

    result = await session.execute(stmt)
    results = list(result.scalars().all())

    items = []
    for pub_result in results:
        profile_summary = AgentProfileSummary(
            id=profile.id,
            display_name=profile.display_name,
            avatar_url=profile.avatar_url,
            is_verified=profile.is_verified,
            verification_badges=profile.verification_badges,
            total_submissions=profile.total_submissions,
            best_overall_score=profile.best_overall_score,
            best_overall_rank=profile.best_overall_rank,
        )
        items.append(
            PublishedResultWithProfile(
                **PublishedResultResponse.model_validate(pub_result).model_dump(),
                agent_profile=profile_summary,
            )
        )

    return PublishedResultList(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/history/{category}", response_model=list[AgentLeaderboardHistory])
async def get_leaderboard_history(
    session: DBSession,
    category: str,
    profile_ids: list[int] | None = Query(None),
    limit_points: int = Query(default=30, le=100),
) -> list[AgentLeaderboardHistory]:
    """Get historical leaderboard data for agents in a category.

    This is a public endpoint that does not require authentication.

    Args:
        session: Database session.
        category: Benchmark category slug.
        profile_ids: Optional list of profile IDs to include.
        limit_points: Maximum data points per agent.

    Returns:
        List of historical leaderboard data for each agent.
    """
    # Build query for published results
    stmt = (
        select(PublishedResult)
        .join(AgentProfile)
        .where(
            PublishedResult.category == category,
            AgentProfile.is_public.is_(True),
        )
        .options(selectinload(PublishedResult.agent_profile))
        .order_by(PublishedResult.agent_profile_id, desc(PublishedResult.published_at))
    )

    if profile_ids:
        stmt = stmt.where(PublishedResult.agent_profile_id.in_(profile_ids))

    result = await session.execute(stmt)
    results = list(result.scalars().all())

    # Group by agent profile
    profiles_data: dict[int, list[PublishedResult]] = {}
    for pub_result in results:
        pid = pub_result.agent_profile_id
        if pid not in profiles_data:
            profiles_data[pid] = []
        if len(profiles_data[pid]) < limit_points:
            profiles_data[pid].append(pub_result)

    # Build response
    histories: list[AgentLeaderboardHistory] = []
    for profile_id, pub_results in profiles_data.items():
        if not pub_results:
            continue

        profile = pub_results[0].agent_profile
        profile_summary = AgentProfileSummary(
            id=profile.id,
            display_name=profile.display_name,
            avatar_url=profile.avatar_url,
            is_verified=profile.is_verified,
            verification_badges=profile.verification_badges,
            total_submissions=profile.total_submissions,
            best_overall_score=profile.best_overall_score,
            best_overall_rank=profile.best_overall_rank,
        )

        # Build data points (calculate rank based on score at that time)
        data_points = []
        for i, pub_result in enumerate(reversed(pub_results)):
            data_points.append(
                LeaderboardTrendPoint(
                    timestamp=pub_result.published_at,
                    rank=i + 1,  # Simplified: actual rank would need historical context
                    score=pub_result.score,
                )
            )

        histories.append(
            AgentLeaderboardHistory(
                agent_profile=profile_summary,
                category=category,
                data_points=data_points,
            )
        )

    return histories


# ==================== Authenticated Endpoints ====================


@router.post(
    "/publish",
    response_model=PublishedResultResponse,
    status_code=status.HTTP_201_CREATED,
)
async def publish_result(
    session: DBSession,
    request: PublishResultRequest,
    user: RequiredUser,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_WRITE))],
) -> PublishedResultResponse:
    """Publish a benchmark result to the public leaderboard.

    Requires RESULTS_WRITE permission.

    Args:
        session: Database session.
        request: Publish request with suite execution ID and category.
        user: Current user.

    Returns:
        The published result.

    Raises:
        HTTPException: If suite execution not found, already published,
            or agent has no profile.
    """
    # Get the suite execution
    exec_stmt = select(SuiteExecution).where(
        SuiteExecution.id == request.suite_execution_id
    )
    exec_result = await session.execute(exec_stmt)
    suite_exec = exec_result.scalar_one_or_none()

    if suite_exec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite execution {request.suite_execution_id} not found",
        )

    # Check if already published
    existing_stmt = select(PublishedResult).where(
        PublishedResult.suite_execution_id == request.suite_execution_id
    )
    existing_result = await session.execute(existing_stmt)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This suite execution has already been published",
        )

    # Verify category exists
    cat_stmt = select(BenchmarkCategory).where(
        BenchmarkCategory.slug == request.category,
        BenchmarkCategory.is_active.is_(True),
    )
    cat_result = await session.execute(cat_stmt)
    if cat_result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Category '{request.category}' not found or inactive",
        )

    # Get or create agent profile
    profile_stmt = select(AgentProfile).where(
        AgentProfile.agent_id == suite_exec.agent_id
    )
    profile_result = await session.execute(profile_stmt)
    profile = profile_result.scalar_one_or_none()

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent must have a public profile before publishing results. "
            "Create a profile first using POST /public/leaderboard/agents",
        )

    # Calculate aggregated score from suite execution
    avg_score = suite_exec.success_rate * 100  # Default to success rate as score

    # Create published result
    published = PublishedResult(
        tenant_id=profile.tenant_id,
        suite_execution_id=suite_exec.id,
        agent_profile_id=profile.id,
        category=request.category,
        score=avg_score,
        success_rate=suite_exec.success_rate,
        total_tests=suite_exec.total_tests,
        passed_tests=suite_exec.passed_tests,
        duration_seconds=suite_exec.duration_seconds,
    )

    session.add(published)

    # Update profile statistics
    profile.total_submissions += 1
    if profile.best_overall_score is None or avg_score > profile.best_overall_score:
        profile.best_overall_score = avg_score

    await session.commit()

    # Invalidate cache
    cache = get_public_leaderboard_cache()
    cache.invalidate_prefix(f"leaderboard:{request.category}:")

    return PublishedResultResponse.model_validate(published)


@router.post(
    "/agents",
    response_model=AgentProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_agent_profile(
    session: DBSession,
    profile_data: AgentProfileCreate,
    user: RequiredUser,
    _: Annotated[None, Depends(require_permission(Permission.AGENTS_WRITE))],
) -> AgentProfileResponse:
    """Create a public agent profile.

    Requires AGENTS_WRITE permission.

    Args:
        session: Database session.
        profile_data: Profile creation data.
        user: Current user.

    Returns:
        The created agent profile.

    Raises:
        HTTPException: If agent not found or profile already exists.
    """
    from atp.dashboard.models import Agent

    # Verify agent exists
    agent_stmt = select(Agent).where(Agent.id == profile_data.agent_id)
    agent_result = await session.execute(agent_stmt)
    agent = agent_result.scalar_one_or_none()

    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {profile_data.agent_id} not found",
        )

    # Check for existing profile
    existing_stmt = select(AgentProfile).where(
        AgentProfile.agent_id == profile_data.agent_id
    )
    existing_result = await session.execute(existing_stmt)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Profile already exists for agent {profile_data.agent_id}",
        )

    profile = AgentProfile(
        tenant_id=agent.tenant_id,
        agent_id=profile_data.agent_id,
        display_name=profile_data.display_name,
        description=profile_data.description,
        website_url=profile_data.website_url,
        repository_url=profile_data.repository_url,
        avatar_url=profile_data.avatar_url,
        tags=profile_data.tags,
        is_public=profile_data.is_public,
        owner_id=user.id,
    )

    session.add(profile)
    await session.commit()

    return AgentProfileResponse.model_validate(profile)


@router.patch("/agents/{profile_id}", response_model=AgentProfileResponse)
async def update_agent_profile(
    session: DBSession,
    profile_id: int,
    profile_data: AgentProfileUpdate,
    user: RequiredUser,
) -> AgentProfileResponse:
    """Update an agent profile.

    Only the profile owner can update the profile.

    Args:
        session: Database session.
        profile_id: Profile ID.
        profile_data: Update data.
        user: Current user.

    Returns:
        The updated profile.

    Raises:
        HTTPException: If profile not found or user is not the owner.
    """
    stmt = select(AgentProfile).where(AgentProfile.id == profile_id)
    result = await session.execute(stmt)
    profile = result.scalar_one_or_none()

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found",
        )

    # Check ownership
    if profile.owner_id != user.id and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own profiles",
        )

    # Apply updates
    if profile_data.display_name is not None:
        profile.display_name = profile_data.display_name
    if profile_data.description is not None:
        profile.description = profile_data.description
    if profile_data.website_url is not None:
        profile.website_url = profile_data.website_url
    if profile_data.repository_url is not None:
        profile.repository_url = profile_data.repository_url
    if profile_data.avatar_url is not None:
        profile.avatar_url = profile_data.avatar_url
    if profile_data.tags is not None:
        profile.tags = profile_data.tags
    if profile_data.is_public is not None:
        profile.is_public = profile_data.is_public

    await session.commit()

    return AgentProfileResponse.model_validate(profile)


# ==================== Admin Endpoints ====================


@router.post(
    "/categories",
    response_model=BenchmarkCategoryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_category(
    session: DBSession,
    category_data: BenchmarkCategoryCreate,
    admin: AdminUser,
) -> BenchmarkCategoryResponse:
    """Create a benchmark category.

    Requires admin privileges.

    Args:
        session: Database session.
        category_data: Category creation data.
        admin: Admin user.

    Returns:
        The created category.

    Raises:
        HTTPException: If category with same name or slug exists.
    """
    # Check for existing category
    existing_stmt = select(BenchmarkCategory).where(
        (BenchmarkCategory.name == category_data.name)
        | (BenchmarkCategory.slug == category_data.slug)
    )
    existing_result = await session.execute(existing_stmt)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Category with this name or slug already exists",
        )

    category = BenchmarkCategory(
        name=category_data.name,
        slug=category_data.slug,
        description=category_data.description,
        icon=category_data.icon,
        display_order=category_data.display_order,
        parent_id=category_data.parent_id,
        min_submissions_for_ranking=category_data.min_submissions_for_ranking,
    )

    session.add(category)
    await session.commit()

    return BenchmarkCategoryResponse.model_validate(category)


@router.patch("/categories/{category_id}", response_model=BenchmarkCategoryResponse)
async def update_category(
    session: DBSession,
    category_id: int,
    category_data: BenchmarkCategoryUpdate,
    admin: AdminUser,
) -> BenchmarkCategoryResponse:
    """Update a benchmark category.

    Requires admin privileges.

    Args:
        session: Database session.
        category_id: Category ID.
        category_data: Update data.
        admin: Admin user.

    Returns:
        The updated category.

    Raises:
        HTTPException: If category not found.
    """
    category = await session.get(BenchmarkCategory, category_id)

    if category is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category {category_id} not found",
        )

    if category_data.name is not None:
        category.name = category_data.name
    if category_data.description is not None:
        category.description = category_data.description
    if category_data.icon is not None:
        category.icon = category_data.icon
    if category_data.display_order is not None:
        category.display_order = category_data.display_order
    if category_data.is_active is not None:
        category.is_active = category_data.is_active
    if category_data.min_submissions_for_ranking is not None:
        category.min_submissions_for_ranking = category_data.min_submissions_for_ranking

    await session.commit()

    return BenchmarkCategoryResponse.model_validate(category)


@router.post("/verify", status_code=status.HTTP_200_OK)
async def add_verification(
    session: DBSession,
    request: VerificationBadgeRequest,
    admin: AdminUser,
) -> AgentProfileResponse:
    """Add a verification badge to an agent profile.

    Requires admin privileges.

    Args:
        session: Database session.
        request: Verification badge request.
        admin: Admin user.

    Returns:
        The updated agent profile.

    Raises:
        HTTPException: If profile not found.
    """
    stmt = select(AgentProfile).where(AgentProfile.id == request.agent_profile_id)
    result = await session.execute(stmt)
    profile = result.scalar_one_or_none()

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {request.agent_profile_id} not found",
        )

    # Add badge if not already present
    if request.badge not in profile.verification_badges:
        profile.verification_badges = [*profile.verification_badges, request.badge]

    # Mark as verified if any badge is added
    profile.is_verified = len(profile.verification_badges) > 0

    await session.commit()

    # Invalidate cache
    cache = get_public_leaderboard_cache()
    cache.invalidate_prefix("leaderboard:")

    return AgentProfileResponse.model_validate(profile)


@router.post("/verify/result", status_code=status.HTTP_200_OK)
async def verify_published_result(
    session: DBSession,
    published_result_id: int,
    admin: AdminUser,
) -> PublishedResultResponse:
    """Verify a published result.

    Requires admin privileges.

    Args:
        session: Database session.
        published_result_id: Published result ID.
        admin: Admin user.

    Returns:
        The verified result.

    Raises:
        HTTPException: If result not found.
    """
    stmt = select(PublishedResult).where(PublishedResult.id == published_result_id)
    result = await session.execute(stmt)
    pub_result = result.scalar_one_or_none()

    if pub_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Published result {published_result_id} not found",
        )

    pub_result.is_verified = True
    pub_result.verified_at = datetime.now()
    pub_result.verified_by_id = admin.id

    await session.commit()

    # Invalidate cache
    cache = get_public_leaderboard_cache()
    cache.invalidate_prefix(f"leaderboard:{pub_result.category}:")

    return PublishedResultResponse.model_validate(pub_result)
