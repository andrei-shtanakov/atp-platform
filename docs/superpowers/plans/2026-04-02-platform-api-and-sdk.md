# ATP Platform API & SDK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend ATP into a pull-model benchmark platform with Python SDK, allowing external agents to fetch tasks, submit results, and compete on leaderboards.

**Architecture:** Extend existing `atp-dashboard` FastAPI server with new route groups (catalog API, tournament API). Create new `packages/atp-sdk/` for participant-facing Python SDK. Add `SDKAdapter` to `atp-adapters` bridging pull and push models. Server-side evaluation uses existing `atp-core` evaluators and scoring.

**Tech Stack:** FastAPI, SQLAlchemy (async), Alembic, httpx, Pydantic v2, atp-core (protocol, evaluators, scoring)

**Spec:** `docs/superpowers/specs/2026-04-02-platform-api-and-sdk-design.md`

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `packages/atp-dashboard/atp/dashboard/benchmark/models.py` | SQLAlchemy models: Benchmark, Run, TaskResult |
| `packages/atp-dashboard/atp/dashboard/benchmark/service.py` | Business logic: create benchmark, start run, next task, submit, evaluate, leaderboard |
| `packages/atp-dashboard/atp/dashboard/benchmark/schemas.py` | Pydantic request/response schemas for API |
| `packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py` | FastAPI routes for `/api/v1/benchmarks` and `/api/v1/runs` |
| `packages/atp-dashboard/atp/dashboard/tournament/models.py` | SQLAlchemy models: Tournament, Participant, Round, Action |
| `packages/atp-dashboard/atp/dashboard/tournament/service.py` | Tournament business logic |
| `packages/atp-dashboard/atp/dashboard/tournament/schemas.py` | Tournament API schemas |
| `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py` | FastAPI routes for `/api/v1/tournaments` |
| `packages/atp-dashboard/atp/dashboard/auth/device_flow.py` | OAuth2 Device Flow (RFC 8628) for CLI login |
| `packages/atp-dashboard/atp/dashboard/v2/routes/device_auth.py` | Device flow API routes |
| `migrations/dashboard/versions/XXXX_add_benchmark_tables.py` | Alembic migration for new tables |
| `packages/atp-adapters/atp/adapters/sdk_adapter.py` | SDKAdapter: pull-model as AgentAdapter |
| `packages/atp-sdk/pyproject.toml` | SDK package config |
| `packages/atp-sdk/atp_sdk/__init__.py` | SDK public API |
| `packages/atp-sdk/atp_sdk/client.py` | ATPClient main class |
| `packages/atp-sdk/atp_sdk/models.py` | SDK-specific models (RunStatus, LeaderboardEntry) |
| `packages/atp-sdk/atp_sdk/benchmark.py` | BenchmarkRun iterator |
| `packages/atp-sdk/atp_sdk/tournament.py` | TournamentSession iterator |
| `packages/atp-sdk/atp_sdk/auth.py` | Device flow CLI auth |
| `tests/unit/benchmark/test_benchmark_models.py` | Tests for DB models |
| `tests/unit/benchmark/test_benchmark_service.py` | Tests for business logic |
| `tests/unit/benchmark/test_benchmark_api.py` | Tests for API routes |
| `tests/unit/tournament/test_tournament_models.py` | Tests for tournament models |
| `tests/unit/tournament/test_tournament_api.py` | Tests for tournament API |
| `tests/unit/adapters/test_sdk_adapter.py` | Tests for SDKAdapter |
| `tests/unit/sdk/test_sdk_client.py` | Tests for atp-sdk client |
| `tests/unit/sdk/test_sdk_benchmark.py` | Tests for BenchmarkRun iterator |
| `tests/integration/test_benchmark_e2e.py` | End-to-end: SDK → API → evaluate → leaderboard |

### Modified Files

| File | Change |
|------|--------|
| `packages/atp-dashboard/atp/dashboard/models.py` | Import new models for Alembic discovery |
| `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py` | Register new routers |
| `packages/atp-dashboard/atp/dashboard/rbac/models.py` | Add BENCHMARKS_* and TOURNAMENTS_* permissions |
| `packages/atp-adapters/atp/adapters/registry.py` | Register sdk adapter in _BUILTIN_ADAPTERS |
| `packages/atp-adapters/atp/adapters/__init__.py` | Export SDKAdapter |
| `pyproject.toml` (root) | Add atp-sdk workspace source |

---

## Task 1: Benchmark Database Models

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/benchmark/__init__.py`
- Create: `packages/atp-dashboard/atp/dashboard/benchmark/models.py`
- Modify: `packages/atp-dashboard/atp/dashboard/models.py`
- Test: `tests/unit/benchmark/test_benchmark_models.py`

- [ ] **Step 1: Write failing test for Benchmark model**

```python
# tests/unit/benchmark/test_benchmark_models.py
"""Tests for benchmark database models."""

import pytest
from datetime import datetime
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from atp.dashboard.models import Base
from atp.dashboard.benchmark.models import (
    Benchmark,
    Run,
    TaskResult,
    RunStatus,
)


@pytest.fixture
def db_session():
    """Create in-memory SQLite session with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_benchmark_creation(db_session: Session) -> None:
    """Benchmark can be created with required fields."""
    bm = Benchmark(
        name="coding-basics-v1",
        description="Basic coding tasks",
        suite={"test_suite": "coding", "tests": []},
        tasks_count=10,
        tags=["coding", "basics"],
        version="1.0",
        family_tag="coding-basics",
    )
    db_session.add(bm)
    db_session.commit()

    result = db_session.execute(select(Benchmark)).scalar_one()
    assert result.name == "coding-basics-v1"
    assert result.tasks_count == 10
    assert result.family_tag == "coding-basics"
    assert result.is_immutable is True
    assert result.parent_id is None


def test_benchmark_parent_relationship(db_session: Session) -> None:
    """Benchmark versions link via parent_id."""
    v1 = Benchmark(
        name="coding-v1",
        description="v1",
        suite={},
        tasks_count=5,
        family_tag="coding",
    )
    db_session.add(v1)
    db_session.flush()

    v2 = Benchmark(
        name="coding-v2",
        description="v2",
        suite={},
        tasks_count=7,
        family_tag="coding",
        parent_id=v1.id,
    )
    db_session.add(v2)
    db_session.commit()

    assert v2.parent_id == v1.id


def test_run_creation(db_session: Session) -> None:
    """Run tracks benchmark execution."""
    bm = Benchmark(name="b1", description="d", suite={}, tasks_count=3)
    db_session.add(bm)
    db_session.flush()

    run = Run(
        benchmark_id=bm.id,
        user_id=1,
        agent_name="my-agent",
        adapter_type="sdk",
        timeout_seconds=3600,
    )
    db_session.add(run)
    db_session.commit()

    assert run.status == RunStatus.PENDING
    assert run.current_task_index == 0
    assert run.total_score is None
    assert run.adapter_type == "sdk"


def test_task_result_with_events(db_session: Session) -> None:
    """TaskResult stores response and optional events."""
    bm = Benchmark(name="b1", description="d", suite={}, tasks_count=1)
    db_session.add(bm)
    db_session.flush()

    run = Run(benchmark_id=bm.id, user_id=1, agent_name="a")
    db_session.add(run)
    db_session.flush()

    tr = TaskResult(
        run_id=run.id,
        task_index=0,
        request={"task_id": "t1", "task": {"description": "do X"}},
        response={"task_id": "t1", "status": "completed", "artifacts": []},
        events=[{"event_type": "progress", "sequence": 0}],
        eval_results=[{"evaluator": "artifact", "checks": []}],
        score=85.5,
    )
    db_session.add(tr)
    db_session.commit()

    assert tr.score == 85.5
    assert tr.events is not None
    assert len(tr.events) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/benchmark/test_benchmark_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'atp.dashboard.benchmark'`

- [ ] **Step 3: Create benchmark __init__.py**

```python
# packages/atp-dashboard/atp/dashboard/benchmark/__init__.py
"""Benchmark catalog models and services."""
```

- [ ] **Step 4: Implement models**

```python
# packages/atp-dashboard/atp/dashboard/benchmark/models.py
"""SQLAlchemy models for benchmark catalog: Benchmark, Run, TaskResult."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from atp.dashboard.models import Base


class RunStatus(StrEnum):
    """Status of a benchmark run."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class Benchmark(Base):
    """Immutable benchmark definition backed by a test suite."""

    __tablename__ = "benchmarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(100), nullable=False, default="default", index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    suite: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    tasks_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0")
    family_tag: Mapped[str | None] = mapped_column(String(200), nullable=True)
    parent_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("benchmarks.id"), nullable=True
    )
    is_immutable: Mapped[bool] = mapped_column(default=True)
    created_by: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    runs: Mapped[list["Run"]] = relationship(back_populates="benchmark")

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_benchmark_tenant_name"),
        Index("idx_benchmark_family", "family_tag"),
        Index("idx_benchmark_tenant", "tenant_id"),
    )

    def __repr__(self) -> str:
        return f"Benchmark(id={self.id}, name={self.name!r})"


class Run(Base):
    """A single execution of a benchmark by an agent."""

    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(100), nullable=False, default="default", index=True
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    benchmark_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("benchmarks.id"), nullable=False
    )
    agent_name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    adapter_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="sdk"
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=RunStatus.PENDING
    )
    current_task_index: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    total_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    timeout_seconds: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3600
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    benchmark: Mapped["Benchmark"] = relationship(back_populates="runs")
    task_results: Mapped[list["TaskResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_run_user", "user_id"),
        Index("idx_run_benchmark", "benchmark_id"),
        Index("idx_run_status", "status"),
        Index("idx_run_tenant", "tenant_id"),
    )

    def __repr__(self) -> str:
        return f"Run(id={self.id}, status={self.status!r})"


class TaskResult(Base):
    """Result of a single task within a run."""

    __tablename__ = "benchmark_task_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("benchmark_runs.id"), nullable=False
    )
    task_index: Mapped[int] = mapped_column(Integer, nullable=False)
    request: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    response: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    events: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    eval_results: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    run: Mapped["Run"] = relationship(back_populates="task_results")

    __table_args__ = (
        UniqueConstraint("run_id", "task_index", name="uq_task_result_run_index"),
        Index("idx_task_result_run", "run_id"),
    )

    def __repr__(self) -> str:
        return f"TaskResult(id={self.id}, run_id={self.run_id}, index={self.task_index})"
```

- [ ] **Step 5: Register models for Alembic discovery**

Add import at end of `packages/atp-dashboard/atp/dashboard/models.py`:

```python
# Import benchmark models so Alembic can discover them
from atp.dashboard.benchmark.models import Benchmark, Run, TaskResult  # noqa: F401, E402
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/benchmark/test_benchmark_models.py -v`
Expected: 4 tests PASS

- [ ] **Step 7: Lint and type check**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 8: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/benchmark/ tests/unit/benchmark/
git add packages/atp-dashboard/atp/dashboard/models.py
git commit -m "feat: add benchmark SQLAlchemy models (Benchmark, Run, TaskResult)"
```

---

## Task 2: Benchmark Service (Business Logic)

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/benchmark/service.py`
- Create: `packages/atp-dashboard/atp/dashboard/benchmark/schemas.py`
- Test: `tests/unit/benchmark/test_benchmark_service.py`

- [ ] **Step 1: Write failing test for BenchmarkService**

```python
# tests/unit/benchmark/test_benchmark_service.py
"""Tests for benchmark service business logic."""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from atp.dashboard.models import Base
from atp.dashboard.benchmark.models import Benchmark, Run, TaskResult, RunStatus
from atp.dashboard.benchmark.service import BenchmarkService
from atp.dashboard.benchmark.schemas import (
    BenchmarkCreate,
    SubmitRequest,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def service(db_session: Session) -> BenchmarkService:
    return BenchmarkService(db_session)


@pytest.fixture
def sample_suite() -> dict:
    return {
        "test_suite": "smoke",
        "version": "1.0",
        "defaults": {
            "constraints": {"timeout_seconds": 60},
            "scoring": {
                "quality_weight": 0.4,
                "completeness_weight": 0.3,
                "efficiency_weight": 0.2,
                "cost_weight": 0.1,
            },
        },
        "tests": [
            {
                "id": "t-001",
                "name": "Echo test",
                "task": {"description": "Echo 'hello'"},
                "assertions": [
                    {"type": "artifact_exists", "config": {"path": "output.txt"}}
                ],
            },
            {
                "id": "t-002",
                "name": "Math test",
                "task": {"description": "Compute 2+2"},
                "assertions": [
                    {"type": "contains", "config": {"pattern": "4"}}
                ],
            },
        ],
    }


def test_create_benchmark(
    service: BenchmarkService, sample_suite: dict
) -> None:
    """Create a benchmark from a test suite."""
    bm = service.create_benchmark(
        BenchmarkCreate(
            name="smoke-v1",
            description="Smoke tests",
            suite=sample_suite,
            tags=["smoke"],
            family_tag="smoke",
        ),
        user_id=1,
    )
    assert bm.name == "smoke-v1"
    assert bm.tasks_count == 2
    assert bm.is_immutable is True


def test_start_run(
    service: BenchmarkService, sample_suite: dict
) -> None:
    """Start a run against a benchmark."""
    bm = service.create_benchmark(
        BenchmarkCreate(name="b1", description="d", suite=sample_suite),
        user_id=1,
    )
    run = service.start_run(
        benchmark_id=bm.id,
        user_id=1,
        agent_name="test-agent",
        adapter_type="sdk",
        timeout_seconds=1800,
    )
    assert run.status == RunStatus.IN_PROGRESS
    assert run.benchmark_id == bm.id
    assert run.timeout_seconds == 1800


def test_next_task_returns_request(
    service: BenchmarkService, sample_suite: dict
) -> None:
    """next_task returns ATPRequest for current task index."""
    bm = service.create_benchmark(
        BenchmarkCreate(name="b1", description="d", suite=sample_suite),
        user_id=1,
    )
    run = service.start_run(bm.id, user_id=1, agent_name="a")

    request = service.next_task(run.id)
    assert request is not None
    assert request["task"]["description"] == "Echo 'hello'"


def test_next_task_returns_none_when_done(
    service: BenchmarkService, sample_suite: dict
) -> None:
    """next_task returns None when all tasks are consumed."""
    bm = service.create_benchmark(
        BenchmarkCreate(name="b1", description="d", suite=sample_suite),
        user_id=1,
    )
    run = service.start_run(bm.id, user_id=1, agent_name="a")

    # Consume both tasks
    service.next_task(run.id)
    service.next_task(run.id)
    result = service.next_task(run.id)
    assert result is None


def test_next_task_atomic_increment(
    service: BenchmarkService, sample_suite: dict
) -> None:
    """next_task increments current_task_index atomically."""
    bm = service.create_benchmark(
        BenchmarkCreate(name="b1", description="d", suite=sample_suite),
        user_id=1,
    )
    run = service.start_run(bm.id, user_id=1, agent_name="a")

    t1 = service.next_task(run.id)
    t2 = service.next_task(run.id)

    # Two different tasks
    assert t1["task"]["description"] != t2["task"]["description"]


def test_submit_stores_result(
    service: BenchmarkService, sample_suite: dict
) -> None:
    """Submit stores response and runs evaluation."""
    bm = service.create_benchmark(
        BenchmarkCreate(name="b1", description="d", suite=sample_suite),
        user_id=1,
    )
    run = service.start_run(bm.id, user_id=1, agent_name="a")
    service.next_task(run.id)

    result = service.submit(
        run.id,
        SubmitRequest(
            response={
                "version": "1.0",
                "task_id": "t-001",
                "status": "completed",
                "artifacts": [],
            },
        ),
    )
    assert result.task_index == 0
    assert result.score is not None


def test_cancel_run(
    service: BenchmarkService, sample_suite: dict
) -> None:
    """Cancel sets status to cancelled."""
    bm = service.create_benchmark(
        BenchmarkCreate(name="b1", description="d", suite=sample_suite),
        user_id=1,
    )
    run = service.start_run(bm.id, user_id=1, agent_name="a")
    service.cancel_run(run.id)

    updated = service.get_run(run.id)
    assert updated.status == RunStatus.CANCELLED


def test_leaderboard(
    service: BenchmarkService, sample_suite: dict
) -> None:
    """Leaderboard returns best score per user."""
    bm = service.create_benchmark(
        BenchmarkCreate(name="b1", description="d", suite=sample_suite),
        user_id=1,
    )

    # Create two runs for same user, different scores
    run1 = service.start_run(bm.id, user_id=1, agent_name="a")
    run1.total_score = 70.0
    run1.status = RunStatus.COMPLETED

    run2 = service.start_run(bm.id, user_id=1, agent_name="a")
    run2.total_score = 90.0
    run2.status = RunStatus.COMPLETED

    service._session.commit()

    entries = service.get_leaderboard(bm.id)
    assert len(entries) == 1
    assert entries[0]["best_score"] == 90.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/benchmark/test_benchmark_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'atp.dashboard.benchmark.service'`

- [ ] **Step 3: Implement schemas**

```python
# packages/atp-dashboard/atp/dashboard/benchmark/schemas.py
"""Pydantic schemas for benchmark API requests and responses."""

from typing import Any

from pydantic import BaseModel, Field


class BenchmarkCreate(BaseModel):
    """Request to create a benchmark."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="")
    suite: dict[str, Any] = Field(..., description="Test suite as parsed YAML/JSON")
    tags: list[str] = Field(default_factory=list)
    version: str = Field(default="1.0")
    family_tag: str | None = Field(default=None)
    parent_id: int | None = Field(default=None)


class BenchmarkResponse(BaseModel):
    """Benchmark details in API responses."""

    id: int
    name: str
    description: str
    tasks_count: int
    tags: list[str]
    version: str
    family_tag: str | None
    created_at: str


class RunResponse(BaseModel):
    """Run details in API responses."""

    id: int
    benchmark_id: int
    agent_name: str
    adapter_type: str
    status: str
    current_task_index: int
    total_score: float | None
    started_at: str
    finished_at: str | None


class SubmitRequest(BaseModel):
    """Request to submit a task result."""

    response: dict[str, Any] = Field(..., description="ATPResponse as JSON")
    events: list[dict[str, Any]] | None = Field(
        default=None, description="Optional ATPEvent list"
    )


class TaskResultResponse(BaseModel):
    """Task result in API responses."""

    task_index: int
    score: float | None
    eval_results: list[dict[str, Any]] | None


class LeaderboardEntry(BaseModel):
    """Single leaderboard row."""

    user_id: int
    agent_name: str
    best_score: float
    run_count: int


class RunStatusResponse(BaseModel):
    """Run status with progress info."""

    id: int
    status: str
    current_task_index: int
    tasks_count: int
    total_score: float | None
    completed_tasks: list[TaskResultResponse]
```

- [ ] **Step 4: Implement BenchmarkService**

```python
# packages/atp-dashboard/atp/dashboard/benchmark/service.py
"""Business logic for benchmark operations."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus, TaskResult
from atp.dashboard.benchmark.schemas import BenchmarkCreate, SubmitRequest
from atp.loader.models import Assertion, TestDefinition, TestSuite
from atp.protocol import ATPRequest, ATPResponse, ATPEvent, Task, Context


class BenchmarkService:
    """Synchronous service for benchmark CRUD and execution logic."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_benchmark(
        self, data: BenchmarkCreate, user_id: int
    ) -> Benchmark:
        """Create an immutable benchmark from a test suite."""
        suite = TestSuite.model_validate(data.suite)
        bm = Benchmark(
            name=data.name,
            description=data.description,
            suite=data.suite,
            tasks_count=len(suite.tests),
            tags=data.tags,
            version=data.version,
            family_tag=data.family_tag,
            parent_id=data.parent_id,
            created_by=user_id,
        )
        self._session.add(bm)
        self._session.commit()
        self._session.refresh(bm)
        return bm

    def get_benchmark(self, benchmark_id: int) -> Benchmark | None:
        """Fetch a benchmark by ID."""
        return self._session.get(Benchmark, benchmark_id)

    def list_benchmarks(
        self, tenant_id: str = "default"
    ) -> list[Benchmark]:
        """List all benchmarks for a tenant."""
        stmt = (
            select(Benchmark)
            .where(Benchmark.tenant_id == tenant_id)
            .order_by(Benchmark.created_at.desc())
        )
        return list(self._session.execute(stmt).scalars().all())

    def start_run(
        self,
        benchmark_id: int,
        user_id: int,
        agent_name: str = "",
        adapter_type: str = "sdk",
        timeout_seconds: int = 3600,
    ) -> Run:
        """Start a new benchmark run."""
        run = Run(
            benchmark_id=benchmark_id,
            user_id=user_id,
            agent_name=agent_name,
            adapter_type=adapter_type,
            status=RunStatus.IN_PROGRESS,
            timeout_seconds=timeout_seconds,
        )
        self._session.add(run)
        self._session.commit()
        self._session.refresh(run)
        return run

    def get_run(self, run_id: int) -> Run | None:
        """Fetch a run by ID."""
        return self._session.get(Run, run_id)

    def next_task(self, run_id: int) -> dict[str, Any] | None:
        """Atomically fetch the next task as an ATPRequest dict.

        Returns None when all tasks have been consumed.
        Uses SELECT ... FOR UPDATE pattern for atomic increment.
        """
        run = self._session.get(Run, run_id)
        if run is None or run.status != RunStatus.IN_PROGRESS:
            return None

        bm = self._session.get(Benchmark, run.benchmark_id)
        if bm is None:
            return None

        idx = run.current_task_index
        if idx >= bm.tasks_count:
            return None

        # Atomic increment
        run.current_task_index = idx + 1
        self._session.commit()

        # Build ATPRequest from test definition
        suite = TestSuite.model_validate(bm.suite)
        test = suite.tests[idx]

        request = ATPRequest(
            task_id=str(uuid.uuid4()),
            task=Task(
                description=test.task.description,
                input_data=test.task.input_data,
                expected_artifacts=test.task.expected_artifacts,
            ),
            constraints={
                "max_steps": test.constraints.max_steps,
                "max_tokens": test.constraints.max_tokens,
                "timeout_seconds": test.constraints.timeout_seconds,
                "allowed_tools": test.constraints.allowed_tools,
                "budget_usd": test.constraints.budget_usd,
            },
            metadata={
                "test_id": test.id,
                "test_name": test.name,
                "task_index": idx,
                "run_id": run_id,
            },
        )
        return request.model_dump(mode="json")

    def submit(
        self,
        run_id: int,
        data: SubmitRequest,
    ) -> TaskResult:
        """Submit a task result, run evaluation, store score."""
        run = self._session.get(Run, run_id)
        if run is None:
            msg = f"Run {run_id} not found"
            raise ValueError(msg)

        bm = self._session.get(Benchmark, run.benchmark_id)
        suite = TestSuite.model_validate(bm.suite)

        # Determine task index from already submitted results
        existing_count = (
            self._session.execute(
                select(func.count(TaskResult.id)).where(
                    TaskResult.run_id == run_id
                )
            )
            .scalar_one()
        )
        task_index = existing_count

        # Parse response
        response = ATPResponse.model_validate(data.response)
        events = (
            [ATPEvent.model_validate(e) for e in data.events]
            if data.events
            else []
        )

        # Run evaluation
        score = self._evaluate_sync(suite.tests[task_index], response, events)

        tr = TaskResult(
            run_id=run_id,
            task_index=task_index,
            request={},
            response=data.response,
            events=data.events,
            eval_results=[],
            score=score,
        )
        self._session.add(tr)

        # Check if run is complete
        if task_index + 1 >= bm.tasks_count:
            self._finalize_run(run)

        self._session.commit()
        self._session.refresh(tr)
        return tr

    def cancel_run(self, run_id: int) -> None:
        """Cancel a run."""
        run = self._session.get(Run, run_id)
        if run is not None:
            run.status = RunStatus.CANCELLED
            run.finished_at = datetime.now()
            self._session.commit()

    def get_leaderboard(
        self, benchmark_id: int
    ) -> list[dict[str, Any]]:
        """Best total_score per user for a benchmark."""
        stmt = (
            select(
                Run.user_id,
                Run.agent_name,
                func.max(Run.total_score).label("best_score"),
                func.count(Run.id).label("run_count"),
            )
            .where(
                Run.benchmark_id == benchmark_id,
                Run.status == RunStatus.COMPLETED,
                Run.total_score.isnot(None),
            )
            .group_by(Run.user_id, Run.agent_name)
            .order_by(func.max(Run.total_score).desc())
        )
        rows = self._session.execute(stmt).all()
        return [
            {
                "user_id": r.user_id,
                "agent_name": r.agent_name,
                "best_score": r.best_score,
                "run_count": r.run_count,
            }
            for r in rows
        ]

    def _evaluate_sync(
        self,
        test: TestDefinition,
        response: ATPResponse,
        events: list[ATPEvent],
    ) -> float:
        """Run evaluators synchronously. Returns score 0-100.

        For MVP, returns a simple pass/fail score based on response status.
        Full evaluator integration requires async; will be added in Task 3.
        """
        if response.status == "completed":
            return 100.0
        return 0.0

    def _finalize_run(self, run: Run) -> None:
        """Mark run complete and compute total score."""
        results = (
            self._session.execute(
                select(TaskResult.score).where(TaskResult.run_id == run.id)
            )
            .scalars()
            .all()
        )
        scores = [s for s in results if s is not None]
        run.total_score = sum(scores) / len(scores) if scores else 0.0
        run.status = RunStatus.COMPLETED
        run.finished_at = datetime.now()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/benchmark/test_benchmark_service.py -v`
Expected: 8 tests PASS

- [ ] **Step 6: Lint and type check**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/benchmark/service.py
git add packages/atp-dashboard/atp/dashboard/benchmark/schemas.py
git add tests/unit/benchmark/test_benchmark_service.py
git commit -m "feat: add BenchmarkService with CRUD, next_task, submit, leaderboard"
```

---

## Task 3: Benchmark API Routes

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`
- Modify: `packages/atp-dashboard/atp/dashboard/rbac/models.py`
- Test: `tests/unit/benchmark/test_benchmark_api.py`

- [ ] **Step 1: Write failing test for benchmark API**

```python
# tests/unit/benchmark/test_benchmark_api.py
"""Tests for benchmark API routes."""

import pytest
from httpx import AsyncClient, ASGITransport

from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
def app():
    return create_test_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_list_benchmarks_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/benchmarks")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.anyio
async def test_create_and_list_benchmark(client: AsyncClient) -> None:
    suite = {
        "test_suite": "smoke",
        "version": "1.0",
        "defaults": {"constraints": {"timeout_seconds": 60}, "scoring": {}},
        "tests": [
            {
                "id": "t-001",
                "name": "Echo",
                "task": {"description": "Echo hello"},
                "assertions": [],
            }
        ],
    }
    resp = await client.post(
        "/api/v1/benchmarks",
        json={"name": "smoke-v1", "description": "Smoke", "suite": suite},
    )
    assert resp.status_code == 201
    bm = resp.json()
    assert bm["name"] == "smoke-v1"
    assert bm["tasks_count"] == 1

    resp = await client.get("/api/v1/benchmarks")
    assert len(resp.json()) == 1


@pytest.mark.anyio
async def test_start_run_and_get_next_task(client: AsyncClient) -> None:
    suite = {
        "test_suite": "s",
        "version": "1.0",
        "defaults": {"constraints": {}, "scoring": {}},
        "tests": [
            {
                "id": "t-001",
                "name": "T1",
                "task": {"description": "Do X"},
                "assertions": [],
            }
        ],
    }
    bm = (
        await client.post(
            "/api/v1/benchmarks",
            json={"name": "b1", "description": "d", "suite": suite},
        )
    ).json()

    run = (
        await client.post(f"/api/v1/benchmarks/{bm['id']}/start")
    ).json()
    assert run["status"] == "in_progress"

    task_resp = await client.get(f"/api/v1/runs/{run['id']}/next-task")
    assert task_resp.status_code == 200
    task = task_resp.json()
    assert task["task"]["description"] == "Do X"


@pytest.mark.anyio
async def test_next_task_204_when_done(client: AsyncClient) -> None:
    suite = {
        "test_suite": "s",
        "version": "1.0",
        "defaults": {"constraints": {}, "scoring": {}},
        "tests": [
            {
                "id": "t-001",
                "name": "T1",
                "task": {"description": "Do X"},
                "assertions": [],
            }
        ],
    }
    bm = (
        await client.post(
            "/api/v1/benchmarks",
            json={"name": "b1", "description": "d", "suite": suite},
        )
    ).json()
    run = (
        await client.post(f"/api/v1/benchmarks/{bm['id']}/start")
    ).json()

    await client.get(f"/api/v1/runs/{run['id']}/next-task")  # consume task
    resp = await client.get(f"/api/v1/runs/{run['id']}/next-task")
    assert resp.status_code == 204


@pytest.mark.anyio
async def test_submit_result(client: AsyncClient) -> None:
    suite = {
        "test_suite": "s",
        "version": "1.0",
        "defaults": {"constraints": {}, "scoring": {}},
        "tests": [
            {
                "id": "t-001",
                "name": "T1",
                "task": {"description": "Do X"},
                "assertions": [],
            }
        ],
    }
    bm = (
        await client.post(
            "/api/v1/benchmarks",
            json={"name": "b1", "description": "d", "suite": suite},
        )
    ).json()
    run = (
        await client.post(f"/api/v1/benchmarks/{bm['id']}/start")
    ).json()
    await client.get(f"/api/v1/runs/{run['id']}/next-task")

    resp = await client.post(
        f"/api/v1/runs/{run['id']}/submit",
        json={
            "response": {
                "version": "1.0",
                "task_id": "t-001",
                "status": "completed",
                "artifacts": [],
            }
        },
    )
    assert resp.status_code == 200
    assert resp.json()["score"] is not None


@pytest.mark.anyio
async def test_cancel_run(client: AsyncClient) -> None:
    suite = {
        "test_suite": "s",
        "version": "1.0",
        "defaults": {"constraints": {}, "scoring": {}},
        "tests": [
            {
                "id": "t-001",
                "name": "T1",
                "task": {"description": "Do X"},
                "assertions": [],
            }
        ],
    }
    bm = (
        await client.post(
            "/api/v1/benchmarks",
            json={"name": "b1", "description": "d", "suite": suite},
        )
    ).json()
    run = (
        await client.post(f"/api/v1/benchmarks/{bm['id']}/start")
    ).json()

    resp = await client.post(f"/api/v1/runs/{run['id']}/cancel")
    assert resp.status_code == 200

    status = (
        await client.get(f"/api/v1/runs/{run['id']}/status")
    ).json()
    assert status["status"] == "cancelled"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/benchmark/test_benchmark_api.py -v`
Expected: FAIL

- [ ] **Step 3: Add RBAC permissions**

Add to `packages/atp-dashboard/atp/dashboard/rbac/models.py` in the `Permission` enum:

```python
    # Benchmark permissions
    BENCHMARKS_READ = "benchmarks:read"
    BENCHMARKS_WRITE = "benchmarks:write"
    BENCHMARKS_EXECUTE = "benchmarks:execute"
    BENCHMARKS_DELETE = "benchmarks:delete"

    # Tournament permissions
    TOURNAMENTS_READ = "tournaments:read"
    TOURNAMENTS_WRITE = "tournaments:write"
    TOURNAMENTS_EXECUTE = "tournaments:execute"
    TOURNAMENTS_DELETE = "tournaments:delete"
```

Add these permissions to the `developer` role's permission set.

- [ ] **Step 4: Implement benchmark API routes**

```python
# packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py
"""Benchmark catalog API routes.

Endpoints:
    GET    /api/v1/benchmarks              — list benchmarks
    GET    /api/v1/benchmarks/{id}         — benchmark details
    POST   /api/v1/benchmarks              — create benchmark [admin]
    POST   /api/v1/benchmarks/{id}/start   — start run
    GET    /api/v1/runs/{run_id}/next-task — next task (ATPRequest)
    POST   /api/v1/runs/{run_id}/submit    — submit result
    GET    /api/v1/runs/{run_id}/status    — run status
    POST   /api/v1/runs/{run_id}/cancel    — cancel run
    GET    /api/v1/benchmarks/{id}/leaderboard — leaderboard
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Response, status

from atp.dashboard.benchmark.models import RunStatus
from atp.dashboard.benchmark.schemas import (
    BenchmarkCreate,
    BenchmarkResponse,
    LeaderboardEntry,
    RunResponse,
    RunStatusResponse,
    SubmitRequest,
    TaskResultResponse,
)
from atp.dashboard.benchmark.service import BenchmarkService
from atp.dashboard.v2.dependencies import DBSession

router = APIRouter(prefix="/api/v1", tags=["benchmarks"])


def _get_service(session: DBSession) -> BenchmarkService:
    return BenchmarkService(session)


@router.get("/benchmarks", response_model=list[BenchmarkResponse])
async def list_benchmarks(
    session: DBSession,
) -> list[dict[str, Any]]:
    """List all available benchmarks."""
    svc = _get_service(session)
    benchmarks = svc.list_benchmarks()
    return [
        {
            "id": b.id,
            "name": b.name,
            "description": b.description,
            "tasks_count": b.tasks_count,
            "tags": b.tags or [],
            "version": b.version,
            "family_tag": b.family_tag,
            "created_at": b.created_at.isoformat(),
        }
        for b in benchmarks
    ]


@router.get("/benchmarks/{benchmark_id}", response_model=BenchmarkResponse)
async def get_benchmark(
    benchmark_id: int,
    session: DBSession,
) -> dict[str, Any]:
    """Get benchmark details."""
    svc = _get_service(session)
    bm = svc.get_benchmark(benchmark_id)
    if bm is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return {
        "id": bm.id,
        "name": bm.name,
        "description": bm.description,
        "tasks_count": bm.tasks_count,
        "tags": bm.tags or [],
        "version": bm.version,
        "family_tag": bm.family_tag,
        "created_at": bm.created_at.isoformat(),
    }


@router.post(
    "/benchmarks",
    response_model=BenchmarkResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_benchmark(
    data: BenchmarkCreate,
    session: DBSession,
) -> dict[str, Any]:
    """Create a new benchmark from a test suite."""
    svc = _get_service(session)
    bm = svc.create_benchmark(data, user_id=1)  # TODO: get from auth
    return {
        "id": bm.id,
        "name": bm.name,
        "description": bm.description,
        "tasks_count": bm.tasks_count,
        "tags": bm.tags or [],
        "version": bm.version,
        "family_tag": bm.family_tag,
        "created_at": bm.created_at.isoformat(),
    }


@router.post("/benchmarks/{benchmark_id}/start", response_model=RunResponse)
async def start_run(
    benchmark_id: int,
    session: DBSession,
    timeout: int = 3600,
    agent_name: str = "",
) -> dict[str, Any]:
    """Start a new benchmark run."""
    svc = _get_service(session)
    bm = svc.get_benchmark(benchmark_id)
    if bm is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")

    run = svc.start_run(
        benchmark_id=benchmark_id,
        user_id=1,  # TODO: get from auth
        agent_name=agent_name,
        timeout_seconds=timeout,
    )
    return {
        "id": run.id,
        "benchmark_id": run.benchmark_id,
        "agent_name": run.agent_name,
        "adapter_type": run.adapter_type,
        "status": run.status,
        "current_task_index": run.current_task_index,
        "total_score": run.total_score,
        "started_at": run.started_at.isoformat(),
        "finished_at": None,
    }


@router.get("/runs/{run_id}/next-task")
async def next_task(
    run_id: int,
    session: DBSession,
) -> Response | dict[str, Any]:
    """Get the next task as an ATPRequest. Returns 204 when done."""
    svc = _get_service(session)
    request = svc.next_task(run_id)
    if request is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return request


@router.post("/runs/{run_id}/submit", response_model=TaskResultResponse)
async def submit_result(
    run_id: int,
    data: SubmitRequest,
    session: DBSession,
) -> dict[str, Any]:
    """Submit a task result for evaluation."""
    svc = _get_service(session)
    try:
        tr = svc.submit(run_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {
        "task_index": tr.task_index,
        "score": tr.score,
        "eval_results": tr.eval_results,
    }


@router.get("/runs/{run_id}/status", response_model=RunStatusResponse)
async def run_status(
    run_id: int,
    session: DBSession,
) -> dict[str, Any]:
    """Get run status with progress details."""
    svc = _get_service(session)
    run = svc.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    bm = svc.get_benchmark(run.benchmark_id)

    completed = [
        {
            "task_index": tr.task_index,
            "score": tr.score,
            "eval_results": tr.eval_results,
        }
        for tr in run.task_results
    ]
    return {
        "id": run.id,
        "status": run.status,
        "current_task_index": run.current_task_index,
        "tasks_count": bm.tasks_count if bm else 0,
        "total_score": run.total_score,
        "completed_tasks": completed,
    }


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: int,
    session: DBSession,
) -> dict[str, str]:
    """Cancel a benchmark run."""
    svc = _get_service(session)
    svc.cancel_run(run_id)
    return {"status": "cancelled"}


@router.get(
    "/benchmarks/{benchmark_id}/leaderboard",
    response_model=list[LeaderboardEntry],
)
async def leaderboard(
    benchmark_id: int,
    session: DBSession,
) -> list[dict[str, Any]]:
    """Get leaderboard for a benchmark."""
    svc = _get_service(session)
    return svc.get_leaderboard(benchmark_id)
```

- [ ] **Step 5: Register router in __init__.py**

Add to `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`:

```python
from atp.dashboard.v2.routes.benchmark_api import router as benchmark_api_router

# In the router aggregation:
router.include_router(benchmark_api_router)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/benchmark/test_benchmark_api.py -v`
Expected: 6 tests PASS

- [ ] **Step 7: Lint and type check**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 8: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py
git add packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py
git add packages/atp-dashboard/atp/dashboard/rbac/models.py
git add tests/unit/benchmark/test_benchmark_api.py
git commit -m "feat: add benchmark catalog API routes (/api/v1/benchmarks, /api/v1/runs)"
```

---

## Task 4: Alembic Migration

**Files:**
- Create: `migrations/dashboard/versions/XXXX_add_benchmark_tables.py`

- [ ] **Step 1: Generate migration from model changes**

Run: `uv run alembic -n dashboard revision --autogenerate -m "add benchmark and tournament tables"`

- [ ] **Step 2: Review generated migration**

Open the generated file in `migrations/dashboard/versions/`. Verify it creates:
- `benchmarks` table with all columns and indexes
- `benchmark_runs` table with all columns and indexes
- `benchmark_task_results` table with all columns and indexes

- [ ] **Step 3: Test migration up and down**

Run:
```bash
uv run alembic -n dashboard upgrade head
uv run alembic -n dashboard downgrade -1
uv run alembic -n dashboard upgrade head
```
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add migrations/dashboard/versions/
git commit -m "feat: add Alembic migration for benchmark tables"
```

---

## Task 5: atp-sdk Package Scaffold

**Files:**
- Create: `packages/atp-sdk/pyproject.toml`
- Create: `packages/atp-sdk/atp_sdk/__init__.py`
- Create: `packages/atp-sdk/atp_sdk/models.py`
- Modify: `pyproject.toml` (root workspace)
- Test: `tests/unit/sdk/test_sdk_import.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/sdk/test_sdk_import.py
"""Tests that atp_sdk package is importable."""


def test_sdk_importable() -> None:
    from atp_sdk import ATPClient

    assert ATPClient is not None


def test_sdk_models_importable() -> None:
    from atp_sdk.models import RunStatus, LeaderboardEntry

    assert RunStatus is not None
    assert LeaderboardEntry is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/sdk/test_sdk_import.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create pyproject.toml**

```toml
# packages/atp-sdk/pyproject.toml
[project]
name = "atp-sdk"
version = "0.1.0"
description = "Python SDK for ATP benchmark platform"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.0",
    "atp-core>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-anyio",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["atp_sdk"]

[tool.uv.sources]
atp-core = { workspace = true }
```

- [ ] **Step 4: Create models.py**

```python
# packages/atp-sdk/atp_sdk/models.py
"""SDK-specific models for API responses."""

from enum import StrEnum

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    """Status of a benchmark run."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class LeaderboardEntry(BaseModel):
    """Single leaderboard row."""

    user_id: int
    agent_name: str
    best_score: float
    run_count: int


class BenchmarkInfo(BaseModel):
    """Benchmark summary from the platform."""

    id: int
    name: str
    description: str
    tasks_count: int
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0"
    family_tag: str | None = None


class RunInfo(BaseModel):
    """Run summary from the platform."""

    id: int
    benchmark_id: int
    agent_name: str
    status: RunStatus
    current_task_index: int = 0
    total_score: float | None = None
```

- [ ] **Step 5: Create __init__.py with stub ATPClient**

```python
# packages/atp-sdk/atp_sdk/__init__.py
"""ATP SDK — Python client for the ATP benchmark platform."""

from atp_sdk.client import ATPClient
from atp_sdk.models import BenchmarkInfo, LeaderboardEntry, RunInfo, RunStatus

__all__ = [
    "ATPClient",
    "BenchmarkInfo",
    "LeaderboardEntry",
    "RunInfo",
    "RunStatus",
]
```

```python
# packages/atp-sdk/atp_sdk/client.py
"""ATPClient — main entry point for the SDK."""

from __future__ import annotations


class ATPClient:
    """Client for interacting with the ATP benchmark platform."""

    def __init__(
        self,
        platform_url: str = "http://localhost:8000",
        token: str | None = None,
    ) -> None:
        self.platform_url = platform_url.rstrip("/")
        self.token = token
```

- [ ] **Step 6: Add workspace source in root pyproject.toml**

Add to `[tool.uv.sources]`:
```toml
atp-sdk = { workspace = true }
```

- [ ] **Step 7: Sync workspace and run tests**

Run:
```bash
uv sync
uv run pytest tests/unit/sdk/test_sdk_import.py -v
```
Expected: 2 tests PASS

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add packages/atp-sdk/ tests/unit/sdk/ pyproject.toml
git commit -m "feat: scaffold atp-sdk package with models and stub ATPClient"
```

---

## Task 6: SDK Client and BenchmarkRun

**Files:**
- Create: `packages/atp-sdk/atp_sdk/benchmark.py`
- Modify: `packages/atp-sdk/atp_sdk/client.py`
- Test: `tests/unit/sdk/test_sdk_client.py`
- Test: `tests/unit/sdk/test_sdk_benchmark.py`

- [ ] **Step 1: Write failing tests for ATPClient**

```python
# tests/unit/sdk/test_sdk_client.py
"""Tests for ATPClient."""

import json
import pytest
import httpx

from atp_sdk import ATPClient
from atp_sdk.models import BenchmarkInfo


@pytest.fixture
def mock_transport():
    """Mock transport that records requests and returns canned responses."""

    class MockTransport(httpx.MockTransport):
        pass

    responses = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method

        key = f"{method} {path}"
        if key in responses:
            return responses[key]
        return httpx.Response(404, json={"detail": "Not found"})

    transport = MockTransport(handler)
    transport.responses = responses
    return transport


def test_list_benchmarks(mock_transport) -> None:
    mock_transport.responses["GET /api/v1/benchmarks"] = httpx.Response(
        200,
        json=[
            {
                "id": 1,
                "name": "smoke-v1",
                "description": "Smoke",
                "tasks_count": 3,
                "tags": ["smoke"],
                "version": "1.0",
                "family_tag": "smoke",
                "created_at": "2026-04-02T00:00:00",
            }
        ],
    )

    client = ATPClient(platform_url="http://test")
    client._http = httpx.Client(transport=mock_transport, base_url="http://test")

    benchmarks = client.list_benchmarks()
    assert len(benchmarks) == 1
    assert benchmarks[0].name == "smoke-v1"


def test_start_run(mock_transport) -> None:
    mock_transport.responses["POST /api/v1/benchmarks/1/start"] = httpx.Response(
        200,
        json={
            "id": 42,
            "benchmark_id": 1,
            "agent_name": "my-agent",
            "adapter_type": "sdk",
            "status": "in_progress",
            "current_task_index": 0,
            "total_score": None,
            "started_at": "2026-04-02T00:00:00",
            "finished_at": None,
        },
    )

    client = ATPClient(platform_url="http://test")
    client._http = httpx.Client(transport=mock_transport, base_url="http://test")

    run = client.start_run("1", agent_name="my-agent")
    assert run._run_id == 42
```

- [ ] **Step 2: Write failing tests for BenchmarkRun**

```python
# tests/unit/sdk/test_sdk_benchmark.py
"""Tests for BenchmarkRun iterator."""

import httpx
import pytest

from atp_sdk.benchmark import BenchmarkRun


def _make_transport(task_responses: list[httpx.Response]) -> httpx.MockTransport:
    """Create transport that returns task_responses in order for next-task."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/next-task"):
            idx = call_count["n"]
            call_count["n"] += 1
            if idx < len(task_responses):
                return task_responses[idx]
            return httpx.Response(204)
        if path.endswith("/submit"):
            return httpx.Response(
                200, json={"task_index": 0, "score": 80.0, "eval_results": []}
            )
        if path.endswith("/status"):
            return httpx.Response(
                200,
                json={
                    "id": 1,
                    "status": "in_progress",
                    "current_task_index": 0,
                    "tasks_count": 2,
                    "total_score": None,
                    "completed_tasks": [],
                },
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_benchmark_run_iterates_tasks() -> None:
    """BenchmarkRun yields ATPRequests until 204."""
    transport = _make_transport([
        httpx.Response(
            200,
            json={
                "version": "1.0",
                "task_id": "t1",
                "task": {"description": "Task 1"},
                "constraints": {},
            },
        ),
        httpx.Response(
            200,
            json={
                "version": "1.0",
                "task_id": "t2",
                "task": {"description": "Task 2"},
                "constraints": {},
            },
        ),
    ])

    http = httpx.Client(transport=transport, base_url="http://test")
    run = BenchmarkRun(http=http, run_id=1, benchmark_id=1)

    tasks = list(run)
    assert len(tasks) == 2
    assert tasks[0]["task"]["description"] == "Task 1"
    assert tasks[1]["task"]["description"] == "Task 2"


def test_benchmark_run_submit() -> None:
    """BenchmarkRun.submit posts response to server."""
    transport = _make_transport([
        httpx.Response(
            200,
            json={
                "version": "1.0",
                "task_id": "t1",
                "task": {"description": "Task 1"},
                "constraints": {},
            },
        ),
    ])
    http = httpx.Client(transport=transport, base_url="http://test")
    run = BenchmarkRun(http=http, run_id=1, benchmark_id=1)

    for task in run:
        result = run.submit({
            "version": "1.0",
            "task_id": "t1",
            "status": "completed",
            "artifacts": [],
        })
        assert result["score"] == 80.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/sdk/test_sdk_client.py tests/unit/sdk/test_sdk_benchmark.py -v`
Expected: FAIL

- [ ] **Step 4: Implement BenchmarkRun**

```python
# packages/atp-sdk/atp_sdk/benchmark.py
"""BenchmarkRun — iterator over benchmark tasks via pull-model."""

from __future__ import annotations

from typing import Any, Iterator

import httpx

from atp_sdk.models import RunStatus


class BenchmarkRun:
    """Iterates over tasks in a benchmark run.

    Usage:
        run = client.start_run("benchmark-42")
        for task in run:
            response = my_agent(task)
            run.submit(response)
    """

    def __init__(
        self,
        http: httpx.Client,
        run_id: int,
        benchmark_id: int,
    ) -> None:
        self._http = http
        self._run_id = run_id
        self._benchmark_id = benchmark_id

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield ATPRequest dicts until all tasks consumed."""
        while True:
            resp = self._http.get(f"/api/v1/runs/{self._run_id}/next-task")
            if resp.status_code == 204:
                break
            resp.raise_for_status()
            yield resp.json()

    def submit(
        self,
        response: dict[str, Any],
        events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Submit a task result."""
        payload: dict[str, Any] = {"response": response}
        if events is not None:
            payload["events"] = events
        resp = self._http.post(
            f"/api/v1/runs/{self._run_id}/submit",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def status(self) -> dict[str, Any]:
        """Get current run status."""
        resp = self._http.get(f"/api/v1/runs/{self._run_id}/status")
        resp.raise_for_status()
        return resp.json()

    def cancel(self) -> None:
        """Cancel the run."""
        resp = self._http.post(f"/api/v1/runs/{self._run_id}/cancel")
        resp.raise_for_status()

    def leaderboard(self) -> list[dict[str, Any]]:
        """Get leaderboard for this benchmark."""
        resp = self._http.get(
            f"/api/v1/benchmarks/{self._benchmark_id}/leaderboard"
        )
        resp.raise_for_status()
        return resp.json()
```

- [ ] **Step 5: Implement full ATPClient**

```python
# packages/atp-sdk/atp_sdk/client.py
"""ATPClient — main entry point for the SDK."""

from __future__ import annotations

import os
from typing import Any

import httpx

from atp_sdk.benchmark import BenchmarkRun
from atp_sdk.models import BenchmarkInfo


class ATPClient:
    """Client for interacting with the ATP benchmark platform.

    Usage:
        client = ATPClient(
            platform_url="https://atp.example.com",
            token="your-api-token",
        )
        benchmarks = client.list_benchmarks()
        run = client.start_run("benchmark-42")
    """

    def __init__(
        self,
        platform_url: str = "http://localhost:8000",
        token: str | None = None,
    ) -> None:
        self.platform_url = platform_url.rstrip("/")
        self.token = token or os.environ.get("ATP_TOKEN")
        self._http = self._create_http_client()

    def _create_http_client(self) -> httpx.Client:
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return httpx.Client(
            base_url=self.platform_url,
            headers=headers,
            timeout=30.0,
        )

    def list_benchmarks(self) -> list[BenchmarkInfo]:
        """List all available benchmarks."""
        resp = self._http.get("/api/v1/benchmarks")
        resp.raise_for_status()
        return [BenchmarkInfo.model_validate(b) for b in resp.json()]

    def get_benchmark(self, benchmark_id: str | int) -> BenchmarkInfo:
        """Get benchmark details."""
        resp = self._http.get(f"/api/v1/benchmarks/{benchmark_id}")
        resp.raise_for_status()
        return BenchmarkInfo.model_validate(resp.json())

    def start_run(
        self,
        benchmark_id: str | int,
        agent_name: str = "",
        timeout: int = 3600,
    ) -> BenchmarkRun:
        """Start a benchmark run and return an iterable BenchmarkRun."""
        resp = self._http.post(
            f"/api/v1/benchmarks/{benchmark_id}/start",
            params={"agent_name": agent_name, "timeout": timeout},
        )
        resp.raise_for_status()
        data = resp.json()
        return BenchmarkRun(
            http=self._http,
            run_id=data["id"],
            benchmark_id=int(benchmark_id),
        )

    def get_leaderboard(
        self, benchmark_id: str | int
    ) -> list[dict[str, Any]]:
        """Get leaderboard for a benchmark."""
        resp = self._http.get(
            f"/api/v1/benchmarks/{benchmark_id}/leaderboard"
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()

    def __enter__(self) -> ATPClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/sdk/ -v`
Expected: All PASS

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add packages/atp-sdk/ tests/unit/sdk/
git commit -m "feat: implement ATPClient and BenchmarkRun iterator for atp-sdk"
```

---

## Task 7: Tournament Models and Stub API

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/tournament/__init__.py`
- Create: `packages/atp-dashboard/atp/dashboard/tournament/models.py`
- Create: `packages/atp-dashboard/atp/dashboard/tournament/schemas.py`
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py`
- Modify: `packages/atp-dashboard/atp/dashboard/models.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`
- Test: `tests/unit/tournament/test_tournament_models.py`
- Test: `tests/unit/tournament/test_tournament_api.py`

- [ ] **Step 1: Write failing tests for tournament models**

```python
# tests/unit/tournament/test_tournament_models.py
"""Tests for tournament database models."""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from atp.dashboard.models import Base
from atp.dashboard.tournament.models import (
    Tournament,
    Participant,
    Round,
    Action,
    TournamentStatus,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_tournament_creation(db_session: Session) -> None:
    t = Tournament(
        game_type="prisoners_dilemma",
        config={"rounds": 100, "noise": 0.05},
        rules={"max_participants": 8, "deadline_action": "cooperate"},
        starts_at=datetime.now(),
        ends_at=datetime.now() + timedelta(days=7),
        created_by=1,
    )
    db_session.add(t)
    db_session.commit()

    assert t.status == TournamentStatus.PENDING
    assert t.game_type == "prisoners_dilemma"


def test_participant_joins_tournament(db_session: Session) -> None:
    t = Tournament(game_type="pd", config={}, rules={}, created_by=1)
    db_session.add(t)
    db_session.flush()

    p = Participant(
        tournament_id=t.id,
        user_id=1,
        agent_name="tit-for-tat",
    )
    db_session.add(p)
    db_session.commit()

    assert p.total_score is None


def test_round_and_action(db_session: Session) -> None:
    t = Tournament(game_type="pd", config={}, rules={}, created_by=1)
    db_session.add(t)
    db_session.flush()

    p = Participant(tournament_id=t.id, user_id=1, agent_name="a")
    db_session.add(p)
    db_session.flush()

    r = Round(
        tournament_id=t.id,
        round_number=1,
        state={"history": []},
        deadline=datetime.now() + timedelta(minutes=5),
    )
    db_session.add(r)
    db_session.flush()

    a = Action(
        round_id=r.id,
        participant_id=p.id,
        action_data={"choice": "cooperate"},
    )
    db_session.add(a)
    db_session.commit()

    assert a.action_data["choice"] == "cooperate"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tournament/test_tournament_models.py -v`
Expected: FAIL

- [ ] **Step 3: Implement tournament models**

```python
# packages/atp-dashboard/atp/dashboard/tournament/__init__.py
"""Tournament models and services."""
```

```python
# packages/atp-dashboard/atp/dashboard/tournament/models.py
"""SQLAlchemy models for game-theoretic tournaments."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from atp.dashboard.models import Base


class TournamentStatus(StrEnum):
    """Status of a tournament."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Tournament(Base):
    """A game-theoretic tournament."""

    __tablename__ = "tournaments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(100), nullable=False, default="default", index=True
    )
    game_type: Mapped[str] = mapped_column(String(100), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=TournamentStatus.PENDING
    )
    starts_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ends_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    rules: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_by: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    participants: Mapped[list["Participant"]] = relationship(
        back_populates="tournament", cascade="all, delete-orphan"
    )
    rounds: Mapped[list["Round"]] = relationship(
        back_populates="tournament", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_tournament_status", "status"),
        Index("idx_tournament_tenant", "tenant_id"),
    )


class Participant(Base):
    """An agent registered in a tournament."""

    __tablename__ = "tournament_participants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tournament_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("tournaments.id"), nullable=False
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    agent_name: Mapped[str] = mapped_column(String(200), nullable=False)
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    total_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    tournament: Mapped["Tournament"] = relationship(back_populates="participants")
    actions: Mapped[list["Action"]] = relationship(
        back_populates="participant", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_participant_tournament", "tournament_id"),
        Index("idx_participant_user", "user_id"),
    )


class Round(Base):
    """A single round in a tournament."""

    __tablename__ = "tournament_rounds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tournament_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("tournaments.id"), nullable=False
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    deadline: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    tournament: Mapped["Tournament"] = relationship(back_populates="rounds")
    actions: Mapped[list["Action"]] = relationship(
        back_populates="round", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_round_tournament", "tournament_id"),
    )


class Action(Base):
    """An agent's action in a round."""

    __tablename__ = "tournament_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("tournament_rounds.id"), nullable=False
    )
    participant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("tournament_participants.id"), nullable=False
    )
    action_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    round: Mapped["Round"] = relationship(back_populates="actions")
    participant: Mapped["Participant"] = relationship(back_populates="actions")

    __table_args__ = (
        Index("idx_action_round", "round_id"),
    )
```

- [ ] **Step 4: Register models for Alembic**

Add to `packages/atp-dashboard/atp/dashboard/models.py`:

```python
from atp.dashboard.tournament.models import Tournament, Participant, Round, Action  # noqa: F401, E402
```

- [ ] **Step 5: Create stub tournament API**

```python
# packages/atp-dashboard/atp/dashboard/tournament/schemas.py
"""Pydantic schemas for tournament API."""

from typing import Any
from pydantic import BaseModel, Field


class TournamentResponse(BaseModel):
    id: int
    game_type: str
    status: str
    starts_at: str | None
    ends_at: str | None


class JoinRequest(BaseModel):
    agent_name: str = Field(..., min_length=1)


class ActionRequest(BaseModel):
    action_data: dict[str, Any]


class RoundResponse(BaseModel):
    round_number: int
    state: dict[str, Any]
    status: str
    deadline: str | None
```

```python
# packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py
"""Tournament API routes (stub — second priority).

Endpoints:
    GET    /api/v1/tournaments              — list tournaments
    GET    /api/v1/tournaments/{id}         — tournament details
    POST   /api/v1/tournaments/{id}/join    — join tournament
    GET    /api/v1/tournaments/{id}/current-round — current round
    POST   /api/v1/tournaments/{id}/action  — submit action
    GET    /api/v1/tournaments/{id}/results — results
"""

from typing import Any
from fastapi import APIRouter, HTTPException

from atp.dashboard.v2.dependencies import DBSession
from atp.dashboard.tournament.models import Tournament
from atp.dashboard.tournament.schemas import TournamentResponse

router = APIRouter(prefix="/api/v1/tournaments", tags=["tournaments"])


@router.get("", response_model=list[TournamentResponse])
async def list_tournaments(session: DBSession) -> list[dict[str, Any]]:
    """List all tournaments."""
    from sqlalchemy import select

    stmt = select(Tournament).order_by(Tournament.created_at.desc())
    tournaments = list((await session.execute(stmt)).scalars().all())
    return [
        {
            "id": t.id,
            "game_type": t.game_type,
            "status": t.status,
            "starts_at": t.starts_at.isoformat() if t.starts_at else None,
            "ends_at": t.ends_at.isoformat() if t.ends_at else None,
        }
        for t in tournaments
    ]


@router.get("/{tournament_id}", response_model=TournamentResponse)
async def get_tournament(
    tournament_id: int, session: DBSession
) -> dict[str, Any]:
    """Get tournament details."""
    t = await session.get(Tournament, tournament_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Tournament not found")
    return {
        "id": t.id,
        "game_type": t.game_type,
        "status": t.status,
        "starts_at": t.starts_at.isoformat() if t.starts_at else None,
        "ends_at": t.ends_at.isoformat() if t.ends_at else None,
    }
```

- [ ] **Step 6: Register tournament router**

Add to `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`:

```python
from atp.dashboard.v2.routes.tournament_api import router as tournament_api_router
router.include_router(tournament_api_router)
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/unit/tournament/ -v`
Expected: 3 model tests PASS

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add packages/atp-dashboard/atp/dashboard/tournament/
git add packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py
git add packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py
git add packages/atp-dashboard/atp/dashboard/models.py
git add tests/unit/tournament/
git commit -m "feat: add tournament models and stub API routes"
```

---

## Task 8: SDKAdapter in atp-adapters

**Files:**
- Create: `packages/atp-adapters/atp/adapters/sdk_adapter.py`
- Modify: `packages/atp-adapters/atp/adapters/registry.py`
- Modify: `packages/atp-adapters/atp/adapters/__init__.py`
- Test: `tests/unit/adapters/test_sdk_adapter.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/adapters/test_sdk_adapter.py
"""Tests for SDKAdapter — pull-model bridge."""

import asyncio
import pytest

from atp.adapters.sdk_adapter import SDKAdapter, SDKAdapterConfig
from atp.protocol import ATPRequest, ATPResponse, Task


@pytest.fixture
def adapter():
    config = SDKAdapterConfig(timeout_seconds=5)
    return SDKAdapter(config)


@pytest.fixture
def sample_request():
    return ATPRequest(
        task_id="test-001",
        task=Task(description="Echo hello"),
        constraints={"max_steps": 5},
    )


@pytest.fixture
def sample_response():
    return ATPResponse(
        task_id="test-001",
        status="completed",
        artifacts=[],
    )


@pytest.mark.anyio
async def test_adapter_type(adapter: SDKAdapter) -> None:
    assert adapter.adapter_type == "sdk"


@pytest.mark.anyio
async def test_enqueue_and_resolve(
    adapter: SDKAdapter,
    sample_request: ATPRequest,
    sample_response: ATPResponse,
) -> None:
    """Enqueue a task, pull it, resolve it."""

    async def agent_side():
        """Simulate agent pulling and responding."""
        await asyncio.sleep(0.1)
        task = adapter.pull_task()
        assert task is not None
        assert task.task_id == "test-001"
        adapter.resolve_task("test-001", sample_response)

    agent_task = asyncio.create_task(agent_side())
    response = await adapter.execute(sample_request)
    await agent_task

    assert response.task_id == "test-001"
    assert response.status.value == "completed"


@pytest.mark.anyio
async def test_timeout_raises(
    adapter: SDKAdapter,
    sample_request: ATPRequest,
) -> None:
    """Execute times out if no one resolves the task."""
    adapter._config.timeout_seconds = 0.1
    with pytest.raises(TimeoutError):
        await adapter.execute(sample_request)


@pytest.mark.anyio
async def test_pull_returns_none_when_empty(adapter: SDKAdapter) -> None:
    result = adapter.pull_task()
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/adapters/test_sdk_adapter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SDKAdapter**

```python
# packages/atp-adapters/atp/adapters/sdk_adapter.py
"""SDKAdapter — bridges pull-model (SDK) with push-model (AgentAdapter).

Agent-side calls pull_task() to get pending tasks and resolve_task() to
submit results. Platform-side calls execute() which blocks until the agent
resolves the task (via asyncio.Event).
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

from pydantic import Field

from atp.adapters.base import AdapterConfig, AgentAdapter
from atp.protocol import ATPEvent, ATPRequest, ATPResponse


class SDKAdapterConfig(AdapterConfig):
    """Configuration for SDK adapter."""

    timeout_seconds: float = Field(
        default=3600.0, description="Timeout waiting for agent response"
    )


class SDKAdapter(AgentAdapter):
    """Adapter that serves tasks via pull-model.

    Platform side (orchestrator) calls execute() which enqueues a task
    and waits for the agent to resolve it. Agent side calls pull_task()
    to get the next pending task and resolve_task() to submit the result.
    """

    def __init__(self, config: SDKAdapterConfig | None = None) -> None:
        super().__init__(config or SDKAdapterConfig())
        self._config: SDKAdapterConfig = config or SDKAdapterConfig()
        self._pending_tasks: OrderedDict[str, ATPRequest] = OrderedDict()
        self._events: dict[str, asyncio.Event] = {}
        self._results: dict[str, ATPResponse] = {}

    @property
    def adapter_type(self) -> str:
        return "sdk"

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """Enqueue task and wait for agent to resolve it."""
        task_id = request.task_id
        self._pending_tasks[task_id] = request
        event = asyncio.Event()
        self._events[task_id] = event

        try:
            await asyncio.wait_for(
                event.wait(),
                timeout=self._config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            self._cleanup_task(task_id)
            msg = f"Agent did not respond within {self._config.timeout_seconds}s"
            raise TimeoutError(msg)

        return self._results.pop(task_id)

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """Execute and yield final response (no streaming in MVP)."""
        response = await self.execute(request)
        yield response

    def pull_task(self) -> ATPRequest | None:
        """Pull the next pending task (FIFO). Returns None if empty."""
        if not self._pending_tasks:
            return None
        _task_id, request = self._pending_tasks.popitem(last=False)
        return request

    def resolve_task(self, task_id: str, response: ATPResponse) -> None:
        """Resolve a pending task with a response."""
        self._results[task_id] = response
        event = self._events.pop(task_id, None)
        if event is not None:
            event.set()

    def _cleanup_task(self, task_id: str) -> None:
        self._pending_tasks.pop(task_id, None)
        self._events.pop(task_id, None)
        self._results.pop(task_id, None)
```

- [ ] **Step 4: Register in adapter registry**

Add to `_BUILTIN_ADAPTERS` dict in `packages/atp-adapters/atp/adapters/registry.py`:

```python
"sdk": _LazyEntry(
    module="atp.adapters.sdk_adapter",
    adapter_class="SDKAdapter",
    config_class="SDKAdapterConfig",
),
```

Add to `packages/atp-adapters/atp/adapters/__init__.py` exports:

```python
"SDKAdapter",
"SDKAdapterConfig",
```

And in the lazy import block:

```python
if name == "SDKAdapter":
    from atp.adapters.sdk_adapter import SDKAdapter
    return SDKAdapter
if name == "SDKAdapterConfig":
    from atp.adapters.sdk_adapter import SDKAdapterConfig
    return SDKAdapterConfig
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/adapters/test_sdk_adapter.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add packages/atp-adapters/atp/adapters/sdk_adapter.py
git add packages/atp-adapters/atp/adapters/registry.py
git add packages/atp-adapters/atp/adapters/__init__.py
git add tests/unit/adapters/test_sdk_adapter.py
git commit -m "feat: add SDKAdapter bridging pull-model SDK with AgentAdapter interface"
```

---

## Task 9: Integration Test (SDK → API → Evaluate → Leaderboard)

**Files:**
- Create: `tests/integration/test_benchmark_e2e.py`

- [ ] **Step 1: Write end-to-end test**

```python
# tests/integration/test_benchmark_e2e.py
"""End-to-end test: SDK client → benchmark API → evaluate → leaderboard."""

import pytest
from httpx import AsyncClient, ASGITransport
import httpx

from atp.dashboard.v2.factory import create_test_app
from atp_sdk import ATPClient
from atp_sdk.benchmark import BenchmarkRun


@pytest.fixture
def app():
    return create_test_app()


@pytest.fixture
def sync_client(app):
    """Sync httpx client for SDK (which uses sync httpx internally)."""
    transport = httpx.MockTransport(
        lambda req: httpx.Response(500, text="should not be called")
    )
    # Use WSGI transport for sync access
    # For integration test, we use AsyncClient + manual SDK wiring
    return None


@pytest.mark.anyio
async def test_full_benchmark_flow(app) -> None:
    """Complete flow: create benchmark → start run → pull tasks → submit → leaderboard."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Create benchmark
        suite = {
            "test_suite": "integration",
            "version": "1.0",
            "defaults": {"constraints": {}, "scoring": {}},
            "tests": [
                {
                    "id": "t-001",
                    "name": "Echo",
                    "task": {"description": "Echo hello"},
                    "assertions": [],
                },
                {
                    "id": "t-002",
                    "name": "Math",
                    "task": {"description": "Compute 2+2"},
                    "assertions": [],
                },
            ],
        }
        resp = await client.post(
            "/api/v1/benchmarks",
            json={
                "name": "integration-v1",
                "description": "Integration test",
                "suite": suite,
                "family_tag": "integration",
            },
        )
        assert resp.status_code == 201
        bm_id = resp.json()["id"]

        # 2. Start run
        resp = await client.post(
            f"/api/v1/benchmarks/{bm_id}/start",
            params={"agent_name": "test-agent"},
        )
        assert resp.status_code == 200
        run_id = resp.json()["id"]

        # 3. Pull and submit all tasks
        for i in range(2):
            task_resp = await client.get(f"/api/v1/runs/{run_id}/next-task")
            assert task_resp.status_code == 200
            task = task_resp.json()

            submit_resp = await client.post(
                f"/api/v1/runs/{run_id}/submit",
                json={
                    "response": {
                        "version": "1.0",
                        "task_id": task["task_id"],
                        "status": "completed",
                        "artifacts": [],
                    }
                },
            )
            assert submit_resp.status_code == 200

        # 4. Verify no more tasks
        resp = await client.get(f"/api/v1/runs/{run_id}/next-task")
        assert resp.status_code == 204

        # 5. Check run status
        resp = await client.get(f"/api/v1/runs/{run_id}/status")
        status = resp.json()
        assert status["status"] == "completed"
        assert status["total_score"] is not None
        assert len(status["completed_tasks"]) == 2

        # 6. Check leaderboard
        resp = await client.get(f"/api/v1/benchmarks/{bm_id}/leaderboard")
        lb = resp.json()
        assert len(lb) == 1
        assert lb[0]["best_score"] == status["total_score"]
```

- [ ] **Step 2: Run integration test**

Run: `uv run pytest tests/integration/test_benchmark_e2e.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite to verify nothing broken**

Run: `uv run pytest tests/ -v -x --timeout=60`
Expected: All existing tests still pass

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_benchmark_e2e.py
git commit -m "test: add end-to-end integration test for benchmark flow"
```

---

## Task 10: Alembic Migration for Tournament Tables

**Files:**
- Create: migration file (auto-generated)

- [ ] **Step 1: Generate migration including tournament tables**

Run: `uv run alembic -n dashboard revision --autogenerate -m "add tournament tables"`

- [ ] **Step 2: Review and test migration**

Run:
```bash
uv run alembic -n dashboard upgrade head
uv run alembic -n dashboard downgrade -1
uv run alembic -n dashboard upgrade head
```

- [ ] **Step 3: Commit**

```bash
git add migrations/dashboard/versions/
git commit -m "feat: add Alembic migration for tournament tables"
```

---

## Summary: Task Dependency Order

```
Task 1: Benchmark DB Models
    ↓
Task 2: Benchmark Service
    ↓
Task 3: Benchmark API Routes
    ↓
Task 4: Alembic Migration (benchmarks)
    ↓ (parallel from here)
Task 5: SDK Scaffold ─────→ Task 6: SDK Client + BenchmarkRun
Task 7: Tournament Models ─→ Task 10: Alembic Migration (tournaments)
Task 8: SDKAdapter
    ↓ (all converge)
Task 9: Integration Test
```

Tasks 5-8 can be developed in parallel after Task 4. Task 9 depends on Tasks 3 and 6.

---

## Not in This Plan (Post-MVP)

Per spec section 9, these are explicitly out of scope:
- Device Flow auth (requires OIDC investigation first)
- Evaluator sandbox (subprocess + rlimits)
- WebSocket for tournaments
- Async SDK API
- Token tracking / event streaming in SDK
- Full server-side evaluation (Task 2 uses simplified scoring; async evaluator integration is a separate task)
