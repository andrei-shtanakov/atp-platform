"""Unit tests for catalog API route functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.catalog.models import CatalogCategory, CatalogSuite, CatalogTest
from atp.dashboard.v2.routes.catalog import (
    CategoryResponse,
    GlobalLeaderboardEntry,
    get_global_leaderboard,
    get_suite,
    list_categories,
)


@pytest.fixture
def mock_session() -> AsyncMock:
    """Return a mock database session."""
    return AsyncMock()


@pytest.fixture
def mock_test() -> MagicMock:
    """Build a mock CatalogTest."""
    t = MagicMock(spec=CatalogTest)
    t.id = 1
    t.slug = "test-one"
    t.name = "Test One"
    t.task_description = "Do something"
    t.difficulty = "easy"
    t.total_submissions = 5
    t.avg_score = 72.0
    t.best_score = 95.0
    return t


@pytest.fixture
def mock_suite(mock_test: MagicMock) -> MagicMock:
    """Build a mock CatalogSuite with one test."""
    s = MagicMock(spec=CatalogSuite)
    s.id = 1
    s.slug = "suite-a"
    s.name = "Suite A"
    s.description = "A sample suite"
    s.difficulty = "easy"
    s.author = "curated"
    s.source = "builtin"
    s.tests = [mock_test]
    return s


@pytest.fixture
def mock_category(mock_suite: MagicMock) -> MagicMock:
    """Build a mock CatalogCategory with one suite."""
    cat = MagicMock(spec=CatalogCategory)
    cat.id = 1
    cat.slug = "coding"
    cat.name = "Coding Tasks"
    cat.description = "Coding challenges"
    cat.suites = [mock_suite]
    return cat


@pytest.mark.anyio
async def test_list_categories_returns_formatted_data(
    mock_session: AsyncMock,
    mock_category: MagicMock,
) -> None:
    """Test that list_categories returns properly formatted CategoryResponse."""
    with patch("atp.dashboard.v2.routes.catalog.CatalogRepository") as MockRepo:
        repo_instance = AsyncMock()
        repo_instance.list_categories.return_value = [mock_category]
        MockRepo.return_value = repo_instance

        result = await list_categories(session=mock_session)

    assert len(result) == 1
    entry = result[0]
    assert isinstance(entry, CategoryResponse)
    assert entry.slug == "coding"
    assert entry.name == "Coding Tasks"
    assert entry.suite_count == 1
    assert entry.test_count == 1


@pytest.mark.anyio
async def test_list_categories_auto_syncs_when_empty(
    mock_session: AsyncMock,
) -> None:
    """Test that list_categories triggers sync when catalog is empty."""
    with (
        patch("atp.dashboard.v2.routes.catalog.CatalogRepository") as MockRepo,
        patch(
            "atp.dashboard.v2.routes.catalog.sync_builtin_catalog",
            new_callable=AsyncMock,
        ) as mock_sync,
    ):
        repo_instance = AsyncMock()
        repo_instance.list_categories.side_effect = [[], []]
        MockRepo.return_value = repo_instance

        result = await list_categories(session=mock_session)

    mock_sync.assert_called_once_with(mock_session)
    assert result == []


@pytest.mark.anyio
async def test_global_leaderboard_returns_empty_when_no_submissions(
    mock_session: AsyncMock,
) -> None:
    """Test that global_leaderboard returns empty list when no submissions."""
    with patch("atp.dashboard.v2.routes.catalog.CatalogRepository") as MockRepo:
        repo_instance = AsyncMock()
        repo_instance.get_global_leaderboard.return_value = []
        MockRepo.return_value = repo_instance

        result = await get_global_leaderboard(session=mock_session)

    assert result == []


@pytest.mark.anyio
async def test_global_leaderboard_returns_ranked_entries(
    mock_session: AsyncMock,
) -> None:
    """Test that global_leaderboard returns entries with correct ranks."""
    rows = [
        {
            "agent_name": "gpt-4o",
            "tests_completed": 5,
            "avg_score": 85.0,
            "total_score": 425.0,
        },
        {
            "agent_name": "claude-3",
            "tests_completed": 3,
            "avg_score": 72.0,
            "total_score": 216.0,
        },
    ]

    with patch("atp.dashboard.v2.routes.catalog.CatalogRepository") as MockRepo:
        repo_instance = AsyncMock()
        repo_instance.get_global_leaderboard.return_value = rows
        MockRepo.return_value = repo_instance

        result = await get_global_leaderboard(session=mock_session)

    assert len(result) == 2
    assert isinstance(result[0], GlobalLeaderboardEntry)
    assert result[0].rank == 1
    assert result[0].agent_name == "gpt-4o"
    assert result[1].rank == 2
    assert result[1].agent_name == "claude-3"


@pytest.mark.anyio
async def test_get_suite_not_found_raises_404(
    mock_session: AsyncMock,
) -> None:
    """Test that get_suite raises HTTPException 404 for missing suite."""
    from fastapi import HTTPException

    with patch("atp.dashboard.v2.routes.catalog.CatalogRepository") as MockRepo:
        repo_instance = AsyncMock()
        repo_instance.get_suite_by_path.return_value = None
        MockRepo.return_value = repo_instance

        with pytest.raises(HTTPException) as exc_info:
            await get_suite(
                category_slug="coding",
                suite_slug="no-such-suite",
                session=mock_session,
            )

    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_get_suite_returns_detail_response(
    mock_session: AsyncMock,
    mock_suite: MagicMock,
    mock_test: MagicMock,
) -> None:
    """Test that get_suite returns a SuiteDetailResponse with tests."""
    from atp.dashboard.v2.routes.catalog import SuiteDetailResponse

    with patch("atp.dashboard.v2.routes.catalog.CatalogRepository") as MockRepo:
        repo_instance = AsyncMock()
        repo_instance.get_suite_by_path.return_value = mock_suite
        MockRepo.return_value = repo_instance

        result = await get_suite(
            category_slug="coding",
            suite_slug="suite-a",
            session=mock_session,
        )

    assert isinstance(result, SuiteDetailResponse)
    assert result.slug == "suite-a"
    assert result.category == "coding"
    assert len(result.tests) == 1
    assert result.tests[0].slug == "test-one"
    assert result.tests[0].total_submissions == 5
