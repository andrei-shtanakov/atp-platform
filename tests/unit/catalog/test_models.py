"""Unit tests for ATP Test Catalog SQLAlchemy models."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from atp.catalog.models import (
    CatalogCategory,
    CatalogSubmission,
    CatalogSuite,
    CatalogTest,
)
from atp.dashboard.models import Base


@pytest.fixture(scope="module")
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    """Provide a transactional test session."""
    with Session(engine) as sess:
        yield sess
        sess.rollback()


def test_category_creation(session: Session) -> None:
    """CatalogCategory can be created and retrieved with correct fields."""
    category = CatalogCategory(
        slug="reasoning",
        name="Reasoning",
        description="Tests for agent reasoning ability",
        icon="brain",
    )
    session.add(category)
    session.flush()

    assert category.id is not None
    assert category.slug == "reasoning"
    assert category.name == "Reasoning"
    assert category.description == "Tests for agent reasoning ability"
    assert category.icon == "brain"
    assert isinstance(category.created_at, datetime)
    assert category.suites == []


def test_suite_creation(session: Session) -> None:
    """CatalogSuite can be created with a parent category."""
    category = CatalogCategory(slug="coding", name="Coding")
    session.add(category)
    session.flush()

    suite = CatalogSuite(
        category_id=category.id,
        slug="python-basics",
        name="Python Basics",
        description="Basic Python coding tasks",
        suite_yaml="name: python-basics\ntests: []",
        difficulty="easy",
        estimated_minutes=30,
        tags=["python", "basics"],
        version="1.0",
    )
    session.add(suite)
    session.flush()

    assert suite.id is not None
    assert suite.category_id == category.id
    assert suite.slug == "python-basics"
    assert suite.author == "curated"
    assert suite.source == "builtin"
    assert suite.version == "1.0"
    assert suite.tests == []


def test_test_creation(session: Session) -> None:
    """CatalogTest can be created with a parent suite."""
    category = CatalogCategory(slug="math", name="Math")
    session.add(category)
    session.flush()

    suite = CatalogSuite(
        category_id=category.id,
        slug="arithmetic",
        name="Arithmetic",
        suite_yaml="name: arithmetic\ntests: []",
    )
    session.add(suite)
    session.flush()

    test = CatalogTest(
        suite_id=suite.id,
        slug="add-two-numbers",
        name="Add Two Numbers",
        description="Write a function to add two numbers",
        task_description="Implement a function add(a, b) that returns a + b.",
        difficulty="easy",
        tags=["arithmetic", "functions"],
    )
    session.add(test)
    session.flush()

    assert test.id is not None
    assert test.suite_id == suite.id
    assert test.slug == "add-two-numbers"
    assert test.total_submissions == 0
    assert test.avg_score is None
    assert test.best_score is None
    assert test.median_score is None
    assert test.submissions == []


def test_submission_creation(session: Session) -> None:
    """CatalogSubmission can be created linked to a test."""
    category = CatalogCategory(slug="nlp", name="NLP")
    session.add(category)
    session.flush()

    suite = CatalogSuite(
        category_id=category.id,
        slug="text-tasks",
        name="Text Tasks",
        suite_yaml="name: text-tasks\ntests: []",
    )
    session.add(suite)
    session.flush()

    test = CatalogTest(
        suite_id=suite.id,
        slug="summarize",
        name="Summarize Text",
        task_description="Summarize the given text in one sentence.",
    )
    session.add(test)
    session.flush()

    submission = CatalogSubmission(
        test_id=test.id,
        agent_name="gpt-4o-mini",
        agent_type="openai",
        score=0.85,
        quality_score=0.9,
        completeness_score=0.8,
        efficiency_score=0.85,
        cost_score=0.75,
        total_tokens=512,
        cost_usd=0.001,
        duration_seconds=2.5,
        submitted_at=datetime.now(),
    )
    session.add(submission)
    session.flush()

    assert submission.id is not None
    assert submission.test_id == test.id
    assert submission.agent_name == "gpt-4o-mini"
    assert submission.score == pytest.approx(0.85)
    assert submission.quality_score == pytest.approx(0.9)
    assert submission.total_tokens == 512
    assert submission.suite_execution_id is None
