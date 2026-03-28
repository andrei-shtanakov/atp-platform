"""Unit tests for CatalogRepository."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from atp.catalog.repository import CatalogRepository
from atp.dashboard.models import Base


@pytest.fixture
async def session():
    """Create an in-memory SQLite async session for testing."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as s:
        yield s

    await engine.dispose()


@pytest.fixture
async def repo(session: AsyncSession) -> CatalogRepository:
    """Create a CatalogRepository backed by the test session."""
    return CatalogRepository(session)


@pytest.mark.anyio
async def test_upsert_category(repo: CatalogRepository) -> None:
    """Test creating then updating a category."""
    cat = await repo.upsert_category(
        slug="coding",
        name="Coding Tasks",
        description="Initial description",
        icon="code",
    )
    assert cat.id is not None
    assert cat.slug == "coding"
    assert cat.name == "Coding Tasks"
    assert cat.description == "Initial description"

    # Update the same category
    updated = await repo.upsert_category(
        slug="coding",
        name="Coding Tasks Updated",
        description="New description",
        icon="terminal",
    )
    assert updated.id == cat.id
    assert updated.name == "Coding Tasks Updated"
    assert updated.description == "New description"
    assert updated.icon == "terminal"


@pytest.mark.anyio
async def test_upsert_suite(repo: CatalogRepository) -> None:
    """Test creating then updating a suite."""
    cat = await repo.upsert_category(slug="coding", name="Coding")
    suite = await repo.upsert_suite(
        category_id=cat.id,
        slug="basic-python",
        name="Basic Python",
        author="curated",
        source="builtin",
        suite_yaml="name: basic-python",
        difficulty="easy",
    )
    assert suite.id is not None
    assert suite.slug == "basic-python"
    assert suite.difficulty == "easy"

    # Update the same suite
    updated = await repo.upsert_suite(
        category_id=cat.id,
        slug="basic-python",
        name="Basic Python v2",
        author="curated",
        source="builtin",
        suite_yaml="name: basic-python-v2",
        difficulty="medium",
        version="2.0",
    )
    assert updated.id == suite.id
    assert updated.name == "Basic Python v2"
    assert updated.difficulty == "medium"
    assert updated.version == "2.0"


@pytest.mark.anyio
async def test_upsert_test(repo: CatalogRepository) -> None:
    """Test creating then updating a test."""
    cat = await repo.upsert_category(slug="coding", name="Coding")
    suite = await repo.upsert_suite(
        category_id=cat.id,
        slug="basic-python",
        name="Basic Python",
        author="curated",
        source="builtin",
        suite_yaml="name: basic-python",
    )
    test = await repo.upsert_test(
        suite_id=suite.id,
        slug="hello-world",
        name="Hello World",
        task_description="Print hello world",
        difficulty="easy",
    )
    assert test.id is not None
    assert test.slug == "hello-world"
    assert test.task_description == "Print hello world"

    # Update
    updated = await repo.upsert_test(
        suite_id=suite.id,
        slug="hello-world",
        name="Hello World v2",
        task_description="Print hello world with formatting",
        difficulty="medium",
    )
    assert updated.id == test.id
    assert updated.name == "Hello World v2"
    assert updated.difficulty == "medium"


@pytest.mark.anyio
async def test_create_submission_and_update_stats(repo: CatalogRepository) -> None:
    """Test creating submissions and updating aggregate test stats."""
    cat = await repo.upsert_category(slug="coding", name="Coding")
    suite = await repo.upsert_suite(
        category_id=cat.id,
        slug="basic-python",
        name="Basic Python",
        author="curated",
        source="builtin",
        suite_yaml="name: basic-python",
    )
    test = await repo.upsert_test(
        suite_id=suite.id,
        slug="hello-world",
        name="Hello World",
        task_description="Print hello world",
    )

    scores = [60.0, 80.0, 100.0]
    for score in scores:
        await repo.create_submission(
            test_id=test.id,
            agent_name="agent-alpha",
            agent_type="http",
            score=score,
            quality_score=score * 0.9,
            completeness_score=score * 0.8,
            efficiency_score=score * 0.7,
            cost_score=score * 0.6,
        )

    await repo.update_test_stats(test.id)

    # Use the raw session query to check updated fields
    from sqlalchemy import select as sa_select

    from atp.catalog.models import CatalogTest as CT

    result = await repo._session.execute(sa_select(CT).where(CT.id == test.id))
    t = result.scalar_one()

    assert t.total_submissions == 3
    assert t.avg_score == pytest.approx(80.0)
    assert t.best_score == pytest.approx(100.0)
    assert t.median_score == pytest.approx(80.0)


@pytest.mark.anyio
async def test_list_categories(repo: CatalogRepository) -> None:
    """Test listing categories with suites loaded."""
    cat_a = await repo.upsert_category(slug="zzz-last", name="Last")
    await repo.upsert_category(slug="aaa-first", name="First")

    await repo.upsert_suite(
        category_id=cat_a.id,
        slug="suite-a",
        name="Suite A",
        author="curated",
        source="builtin",
        suite_yaml="name: suite-a",
    )

    categories = await repo.list_categories()
    assert len(categories) == 2
    # sorted by slug: aaa-first before zzz-last
    assert categories[0].slug == "aaa-first"
    assert categories[1].slug == "zzz-last"
    # suites eagerly loaded
    assert isinstance(categories[1].suites, list)
    assert len(categories[1].suites) == 1


@pytest.mark.anyio
async def test_get_suite_by_path(repo: CatalogRepository) -> None:
    """Test retrieving a suite by category and suite slug."""
    cat = await repo.upsert_category(slug="coding", name="Coding")
    await repo.upsert_suite(
        category_id=cat.id,
        slug="basic-python",
        name="Basic Python",
        author="curated",
        source="builtin",
        suite_yaml="name: basic-python",
    )
    await repo.upsert_test(
        suite_id=(await repo.get_suite_by_path("coding", "basic-python")).id,  # type: ignore[union-attr]
        slug="t1",
        name="Test 1",
        task_description="Do something",
    )

    suite = await repo.get_suite_by_path("coding", "basic-python")
    assert suite is not None
    assert suite.slug == "basic-python"
    assert len(suite.tests) == 1

    missing = await repo.get_suite_by_path("coding", "nonexistent")
    assert missing is None


@pytest.mark.anyio
async def test_get_top_submissions(repo: CatalogRepository) -> None:
    """Test top submissions are returned sorted by score descending."""
    cat = await repo.upsert_category(slug="coding", name="Coding")
    suite = await repo.upsert_suite(
        category_id=cat.id,
        slug="basic-python",
        name="Basic Python",
        author="curated",
        source="builtin",
        suite_yaml="name: basic-python",
    )
    test = await repo.upsert_test(
        suite_id=suite.id,
        slug="hello-world",
        name="Hello World",
        task_description="Print hello world",
    )

    for score in [30.0, 90.0, 50.0, 70.0, 100.0]:
        await repo.create_submission(
            test_id=test.id,
            agent_name="agent",
            agent_type="http",
            score=score,
        )

    top3 = await repo.get_top_submissions(test.id, limit=3)
    assert len(top3) == 3
    assert top3[0].score == pytest.approx(100.0)
    assert top3[1].score == pytest.approx(90.0)
    assert top3[2].score == pytest.approx(70.0)

    # Test get_avg_scores_for_test
    avgs = await repo.get_avg_scores_for_test(test.id)
    assert "score" in avgs
    assert avgs["score"] == pytest.approx((30 + 90 + 50 + 70 + 100) / 5)
    assert "quality_score" in avgs
    assert "completeness_score" in avgs
    assert "efficiency_score" in avgs
    assert "cost_score" in avgs
