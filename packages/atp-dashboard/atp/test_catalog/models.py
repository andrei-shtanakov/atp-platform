"""SQLAlchemy ORM models for the ATP Test Catalog."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from atp.dashboard.models import Base


class CatalogCategory(Base):
    """Top-level category grouping related test suites."""

    __tablename__ = "catalog_categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    icon: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    # Relationships
    suites: Mapped[list["CatalogSuite"]] = relationship(
        back_populates="category", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"CatalogCategory(id={self.id}, slug={self.slug!r})"


class CatalogSuite(Base):
    """A named collection of tests within a category."""

    __tablename__ = "catalog_suites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("catalog_categories.id"), nullable=False
    )
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    author: Mapped[str] = mapped_column(String(100), nullable=False, default="curated")
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="builtin")
    difficulty: Mapped[str | None] = mapped_column(String(50), nullable=True)
    estimated_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tags: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    version: Mapped[str] = mapped_column(String(20), nullable=False, default="1.0")
    suite_yaml: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    # Relationships
    category: Mapped["CatalogCategory"] = relationship(back_populates="suites")
    tests: Mapped[list["CatalogTest"]] = relationship(
        back_populates="suite", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"CatalogSuite(id={self.id}, slug={self.slug!r})"


class CatalogTest(Base):
    """An individual test within a catalog suite."""

    __tablename__ = "catalog_tests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    suite_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("catalog_suites.id"), nullable=False
    )
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_description: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[str | None] = mapped_column(String(50), nullable=True)
    tags: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    total_submissions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    median_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    # Relationships
    suite: Mapped["CatalogSuite"] = relationship(back_populates="tests")
    submissions: Mapped[list["CatalogSubmission"]] = relationship(
        back_populates="test", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"CatalogTest(id={self.id}, slug={self.slug!r})"


class CatalogSubmission(Base):
    """A recorded agent submission against a catalog test."""

    __tablename__ = "catalog_submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    test_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("catalog_tests.id"), nullable=False
    )
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    completeness_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    efficiency_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    cost_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    suite_execution_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("suite_executions.id"), nullable=True
    )
    submitted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Relationships
    test: Mapped["CatalogTest"] = relationship(back_populates="submissions")

    def __repr__(self) -> str:
        return (
            f"CatalogSubmission(id={self.id}, test_id={self.test_id}, "
            f"agent={self.agent_name!r}, score={self.score})"
        )
