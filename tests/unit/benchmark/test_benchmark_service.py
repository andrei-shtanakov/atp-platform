"""Tests for BenchmarkService business logic."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from atp.dashboard.benchmark.models import (
    RunStatus,
)
from atp.dashboard.benchmark.schemas import BenchmarkCreate, SubmitRequest
from atp.dashboard.benchmark.service import BenchmarkService
from atp.dashboard.models import Base, User

SAMPLE_SUITE: dict = {
    "test_suite": "sample-benchmark",
    "version": "1.0",
    "tests": [
        {
            "id": "test-1",
            "name": "Test One",
            "task": {"description": "Do something"},
        },
        {
            "id": "test-2",
            "name": "Test Two",
            "task": {"description": "Do another thing"},
        },
    ],
}


@pytest.fixture()
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session(engine):
    """Create a new database session for testing."""
    with Session(engine) as sess:
        yield sess


@pytest.fixture()
def user(session: Session) -> User:
    """Create a minimal user for FK references."""
    u = User(
        username="tester",
        email="tester@example.com",
        hashed_password="fakehash",
    )
    session.add(u)
    session.commit()
    session.refresh(u)
    return u


@pytest.fixture()
def svc(session: Session) -> BenchmarkService:
    """Create a BenchmarkService bound to the test session."""
    return BenchmarkService(session)


class TestCreateBenchmark:
    """Tests for create_benchmark."""

    def test_creates_benchmark_with_correct_tasks_count(
        self, svc: BenchmarkService, user: User
    ) -> None:
        data = BenchmarkCreate(name="My Bench", suite=SAMPLE_SUITE)
        bm = svc.create_benchmark(data, user_id=user.id)

        assert bm.id is not None
        assert bm.name == "My Bench"
        assert bm.tasks_count == 2
        assert bm.version == "1.0"

    def test_stores_tags_and_family_tag(
        self, svc: BenchmarkService, user: User
    ) -> None:
        data = BenchmarkCreate(
            name="Tagged",
            suite=SAMPLE_SUITE,
            tags=["a", "b"],
            family_tag="family-1",
        )
        bm = svc.create_benchmark(data, user_id=user.id)

        assert bm.tags == ["a", "b"]
        assert bm.family_tag == "family-1"

    def test_invalid_suite_raises(self, svc: BenchmarkService, user: User) -> None:
        data = BenchmarkCreate(name="Bad", suite={"invalid": True})
        with pytest.raises(ValueError):
            svc.create_benchmark(data, user_id=user.id)


class TestGetAndListBenchmarks:
    """Tests for get_benchmark and list_benchmarks."""

    def test_get_benchmark_returns_none_for_missing(
        self, svc: BenchmarkService
    ) -> None:
        assert svc.get_benchmark(999) is None

    def test_get_benchmark_returns_existing(
        self, svc: BenchmarkService, user: User
    ) -> None:
        data = BenchmarkCreate(name="Find Me", suite=SAMPLE_SUITE)
        bm = svc.create_benchmark(data, user_id=user.id)
        found = svc.get_benchmark(bm.id)

        assert found is not None
        assert found.id == bm.id

    def test_list_benchmarks(self, svc: BenchmarkService, user: User) -> None:
        svc.create_benchmark(
            BenchmarkCreate(name="B1", suite=SAMPLE_SUITE),
            user_id=user.id,
        )
        svc.create_benchmark(
            BenchmarkCreate(name="B2", suite=SAMPLE_SUITE),
            user_id=user.id,
        )
        benchmarks = svc.list_benchmarks()

        assert len(benchmarks) == 2


class TestStartRun:
    """Tests for start_run."""

    def test_creates_run_in_progress(self, svc: BenchmarkService, user: User) -> None:
        bm = svc.create_benchmark(
            BenchmarkCreate(name="RunBench", suite=SAMPLE_SUITE),
            user_id=user.id,
        )
        run = svc.start_run(
            benchmark_id=bm.id,
            user_id=user.id,
            agent_name="my-agent",
        )

        assert run.id is not None
        assert run.status == RunStatus.IN_PROGRESS
        assert run.agent_name == "my-agent"
        assert run.benchmark_id == bm.id
        assert run.current_task_index == 0
        assert run.started_at is not None

    def test_start_run_defaults(self, svc: BenchmarkService, user: User) -> None:
        bm = svc.create_benchmark(
            BenchmarkCreate(name="Defaults", suite=SAMPLE_SUITE),
            user_id=user.id,
        )
        run = svc.start_run(benchmark_id=bm.id, user_id=user.id)

        assert run.adapter_type == "sdk"
        assert run.agent_name == ""


class TestNextTask:
    """Tests for next_task."""

    def test_returns_atp_request_dict(self, svc: BenchmarkService, user: User) -> None:
        bm = svc.create_benchmark(
            BenchmarkCreate(name="NTBench", suite=SAMPLE_SUITE),
            user_id=user.id,
        )
        run = svc.start_run(benchmark_id=bm.id, user_id=user.id)
        task = svc.next_task(run.id)

        assert task is not None
        assert "task_id" in task
        assert "task" in task
        assert task["task"]["description"] == "Do something"
        assert task["metadata"]["task_index"] == 0
        assert task["metadata"]["run_id"] == run.id

    def test_increments_task_index(self, svc: BenchmarkService, user: User) -> None:
        bm = svc.create_benchmark(
            BenchmarkCreate(name="IncBench", suite=SAMPLE_SUITE),
            user_id=user.id,
        )
        run = svc.start_run(benchmark_id=bm.id, user_id=user.id)

        t1 = svc.next_task(run.id)
        assert t1 is not None
        assert t1["metadata"]["task_index"] == 0

        t2 = svc.next_task(run.id)
        assert t2 is not None
        assert t2["metadata"]["task_index"] == 1

    def test_returns_none_when_all_consumed(
        self, svc: BenchmarkService, user: User
    ) -> None:
        bm = svc.create_benchmark(
            BenchmarkCreate(name="AllDone", suite=SAMPLE_SUITE),
            user_id=user.id,
        )
        run = svc.start_run(benchmark_id=bm.id, user_id=user.id)

        svc.next_task(run.id)  # task 0
        svc.next_task(run.id)  # task 1
        result = svc.next_task(run.id)  # no more

        assert result is None


class TestSubmit:
    """Tests for submit."""

    def test_stores_task_result(self, svc: BenchmarkService, user: User) -> None:
        bm = svc.create_benchmark(
            BenchmarkCreate(name="SubBench", suite=SAMPLE_SUITE),
            user_id=user.id,
        )
        run = svc.start_run(benchmark_id=bm.id, user_id=user.id)
        svc.next_task(run.id)

        data = SubmitRequest(
            response={
                "task_id": "test-1",
                "status": "completed",
            }
        )
        tr = svc.submit(run.id, data)

        assert tr.task_index == 0
        assert tr.score == 100.0

    def test_scores_zero_for_failed(self, svc: BenchmarkService, user: User) -> None:
        bm = svc.create_benchmark(
            BenchmarkCreate(name="FailBench", suite=SAMPLE_SUITE),
            user_id=user.id,
        )
        run = svc.start_run(benchmark_id=bm.id, user_id=user.id)
        svc.next_task(run.id)

        data = SubmitRequest(
            response={
                "task_id": "test-1",
                "status": "failed",
                "error": "oops",
            }
        )
        tr = svc.submit(run.id, data)

        assert tr.score == 0.0

    def test_finalizes_run_when_all_submitted(
        self, svc: BenchmarkService, user: User
    ) -> None:
        bm = svc.create_benchmark(
            BenchmarkCreate(name="FinBench", suite=SAMPLE_SUITE),
            user_id=user.id,
        )
        run = svc.start_run(benchmark_id=bm.id, user_id=user.id)

        # Submit for both tasks
        svc.next_task(run.id)
        svc.submit(
            run.id,
            SubmitRequest(response={"task_id": "t1", "status": "completed"}),
        )

        svc.next_task(run.id)
        svc.submit(
            run.id,
            SubmitRequest(response={"task_id": "t2", "status": "failed", "error": "e"}),
        )

        svc.session.refresh(run)
        assert run.status == RunStatus.COMPLETED
        assert run.total_score == 50.0  # average of 100 and 0
        assert run.finished_at is not None


class TestCancelRun:
    """Tests for cancel_run."""

    def test_sets_cancelled_status(self, svc: BenchmarkService, user: User) -> None:
        bm = svc.create_benchmark(
            BenchmarkCreate(name="CancelBench", suite=SAMPLE_SUITE),
            user_id=user.id,
        )
        run = svc.start_run(benchmark_id=bm.id, user_id=user.id)
        svc.cancel_run(run.id)

        svc.session.refresh(run)
        assert run.status == RunStatus.CANCELLED
        assert run.finished_at is not None


class TestLeaderboard:
    """Tests for get_leaderboard."""

    def test_returns_best_score_per_user(
        self, svc: BenchmarkService, user: User, session: Session
    ) -> None:
        bm = svc.create_benchmark(
            BenchmarkCreate(name="LBBench", suite=SAMPLE_SUITE),
            user_id=user.id,
        )

        # Create two completed runs with different scores
        run1 = svc.start_run(
            benchmark_id=bm.id,
            user_id=user.id,
            agent_name="agent-a",
        )
        svc.next_task(run1.id)
        svc.submit(
            run1.id,
            SubmitRequest(response={"task_id": "t1", "status": "completed"}),
        )
        svc.next_task(run1.id)
        svc.submit(
            run1.id,
            SubmitRequest(response={"task_id": "t2", "status": "completed"}),
        )

        run2 = svc.start_run(
            benchmark_id=bm.id,
            user_id=user.id,
            agent_name="agent-a",
        )
        svc.next_task(run2.id)
        svc.submit(
            run2.id,
            SubmitRequest(response={"task_id": "t1", "status": "failed", "error": "e"}),
        )
        svc.next_task(run2.id)
        svc.submit(
            run2.id,
            SubmitRequest(response={"task_id": "t2", "status": "failed", "error": "e"}),
        )

        lb = svc.get_leaderboard(bm.id)

        assert len(lb) == 1
        assert lb[0]["best_score"] == 100.0
        assert lb[0]["run_count"] == 2
