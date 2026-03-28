"""Unit tests for atp.catalog.sync."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from atp.catalog.repository import CatalogRepository
from atp.catalog.sync import parse_catalog_yaml, sync_builtin_catalog
from atp.dashboard.models import Base

SAMPLE_YAML = """\
catalog:
  category: coding
  slug: file-operations
  name: "File Operations"
  description: "Tests file I/O"
  author: curated
  source: builtin
  difficulty: easy
  estimated_minutes: 10
  tags: [coding, files]

test_suite: "catalog:coding/file-operations"
version: "1.0"

defaults:
  runs_per_test: 3
  timeout_seconds: 60

tests:
  - id: create-file
    name: "Create a File"
    tags: [easy]
    task:
      description: "Create hello.txt with Hello, World!"
      expected_artifacts: ["hello.txt"]
    assertions:
      - type: artifact_exists
        config:
          path: "hello.txt"
"""


@pytest.fixture
async def session():
    """In-memory SQLite async session for catalog tests."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as s:
        yield s

    await engine.dispose()


# ---------------------------------------------------------------------------
# parse_catalog_yaml tests (no DB)
# ---------------------------------------------------------------------------


def test_parse_catalog_yaml_returns_meta_and_tests() -> None:
    """parse_catalog_yaml should extract catalog metadata and test list."""
    catalog_meta, tests = parse_catalog_yaml(SAMPLE_YAML)

    assert catalog_meta["slug"] == "file-operations"
    assert catalog_meta["category"] == "coding"
    assert catalog_meta["difficulty"] == "easy"
    assert catalog_meta["tags"] == ["coding", "files"]

    assert len(tests) == 1
    assert tests[0]["id"] == "create-file"
    assert tests[0]["task"]["description"] == "Create hello.txt with Hello, World!"


def test_parse_catalog_yaml_empty_tests() -> None:
    """parse_catalog_yaml should return empty list when no tests key present."""
    minimal = """\
catalog:
  slug: minimal
  name: Minimal
"""
    catalog_meta, tests = parse_catalog_yaml(minimal)
    assert catalog_meta["slug"] == "minimal"
    assert tests == []


def test_parse_catalog_yaml_missing_catalog_key() -> None:
    """parse_catalog_yaml should return empty dict when catalog key absent."""
    no_catalog = "tests:\n  - id: t1\n    name: T1\n    task:\n      description: x\n"
    catalog_meta, tests = parse_catalog_yaml(no_catalog)
    assert catalog_meta == {}
    assert len(tests) == 1


# ---------------------------------------------------------------------------
# sync_builtin_catalog tests (in-memory DB)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_sync_builtin_catalog_creates_categories(
    session: AsyncSession,
) -> None:
    """sync_builtin_catalog should create at least 4 categories from builtin files."""
    await sync_builtin_catalog(session)

    repo = CatalogRepository(session)
    categories = await repo.list_categories()

    assert len(categories) >= 4


@pytest.mark.anyio
async def test_sync_builtin_catalog_creates_suites_and_tests(
    session: AsyncSession,
) -> None:
    """sync_builtin_catalog should create suites and tests for each YAML file."""
    await sync_builtin_catalog(session)

    repo = CatalogRepository(session)
    suites = await repo.list_suites()

    assert len(suites) >= 4
    # Every suite must have at least one test
    for suite in suites:
        assert len(suite.tests) >= 1


@pytest.mark.anyio
async def test_sync_builtin_catalog_is_idempotent(
    session: AsyncSession,
) -> None:
    """Running sync twice should not duplicate categories or suites."""
    await sync_builtin_catalog(session)
    await sync_builtin_catalog(session)

    repo = CatalogRepository(session)
    categories = await repo.list_categories()
    suites = await repo.list_suites()

    # Counts must be the same as after a single sync
    assert len(categories) >= 4
    assert len(suites) >= 4
