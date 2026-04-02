"""Tests for benchmark SQLAlchemy models."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from atp.dashboard.benchmark.models import (
    Benchmark,
    Run,
    RunStatus,
    TaskResult,
)
from atp.dashboard.models import Base


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


class TestRunStatus:
    """Tests for RunStatus enum."""

    def test_all_statuses_exist(self) -> None:
        assert RunStatus.PENDING == "PENDING"
        assert RunStatus.IN_PROGRESS == "IN_PROGRESS"
        assert RunStatus.COMPLETED == "COMPLETED"
        assert RunStatus.FAILED == "FAILED"
        assert RunStatus.CANCELLED == "CANCELLED"
        assert RunStatus.PARTIAL == "PARTIAL"

    def test_status_is_str(self) -> None:
        assert isinstance(RunStatus.PENDING, str)


class TestBenchmark:
    """Tests for the Benchmark model."""

    def test_create_benchmark_all_fields(self, session: Session) -> None:
        bm = Benchmark(
            tenant_id="acme",
            name="math-eval",
            description="Math evaluation suite",
            suite={"tasks": [{"prompt": "2+2"}]},
            tasks_count=1,
            tags=["math", "basic"],
            version="1.0.0",
            family_tag="math-family",
            is_immutable=True,
            created_by=None,
        )
        session.add(bm)
        session.commit()
        session.refresh(bm)

        assert bm.id is not None
        assert bm.tenant_id == "acme"
        assert bm.name == "math-eval"
        assert bm.description == "Math evaluation suite"
        assert bm.suite == {"tasks": [{"prompt": "2+2"}]}
        assert bm.tasks_count == 1
        assert bm.tags == ["math", "basic"]
        assert bm.version == "1.0.0"
        assert bm.family_tag == "math-family"
        assert bm.is_immutable is True
        assert isinstance(bm.created_at, datetime)

    def test_benchmark_defaults(self, session: Session) -> None:
        bm = Benchmark(name="minimal", tasks_count=0)
        session.add(bm)
        session.commit()
        session.refresh(bm)

        assert bm.tenant_id == "default"
        assert bm.is_immutable is True
        assert bm.suite == {}
        assert bm.tags == []

    def test_benchmark_parent_relationship(self, session: Session) -> None:
        v1 = Benchmark(
            name="bench-v1",
            tasks_count=5,
            version="1.0",
            family_tag="bench-family",
        )
        session.add(v1)
        session.commit()
        session.refresh(v1)

        v2 = Benchmark(
            name="bench-v2",
            tasks_count=6,
            version="2.0",
            family_tag="bench-family",
            parent_id=v1.id,
        )
        session.add(v2)
        session.commit()
        session.refresh(v2)

        assert v2.parent_id == v1.id
        assert v2.parent.id == v1.id
        assert v1.children[0].id == v2.id

    def test_benchmark_unique_constraint(self, session: Session) -> None:
        bm1 = Benchmark(tenant_id="t1", name="same-name", tasks_count=1)
        session.add(bm1)
        session.commit()

        bm2 = Benchmark(tenant_id="t1", name="same-name", tasks_count=2)
        session.add(bm2)
        with pytest.raises(Exception):  # IntegrityError
            session.commit()

    def test_benchmark_table_name(self) -> None:
        assert Benchmark.__tablename__ == "benchmarks"


class TestRun:
    """Tests for the Run model."""

    def test_create_run_defaults(self, session: Session) -> None:
        bm = Benchmark(name="run-bench", tasks_count=3)
        session.add(bm)
        session.commit()
        session.refresh(bm)

        run = Run(
            benchmark_id=bm.id,
            agent_name="gpt-4o",
            adapter_type="http",
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        assert run.id is not None
        assert run.status == RunStatus.PENDING
        assert run.current_task_index == 0
        assert run.total_score is None
        assert run.timeout_seconds == 3600
        assert run.tenant_id == "default"
        assert run.started_at is None
        assert run.finished_at is None

    def test_run_with_all_fields(self, session: Session) -> None:
        bm = Benchmark(name="full-bench", tasks_count=2)
        session.add(bm)
        session.commit()
        session.refresh(bm)

        now = datetime.now()
        run = Run(
            tenant_id="corp",
            benchmark_id=bm.id,
            agent_name="claude-3",
            adapter_type="cli",
            status=RunStatus.IN_PROGRESS,
            current_task_index=1,
            total_score=0.85,
            timeout_seconds=7200,
            started_at=now,
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        assert run.status == RunStatus.IN_PROGRESS
        assert run.current_task_index == 1
        assert run.total_score == pytest.approx(0.85)
        assert run.timeout_seconds == 7200

    def test_run_table_name(self) -> None:
        assert Run.__tablename__ == "benchmark_runs"


class TestTaskResult:
    """Tests for the TaskResult model."""

    def test_create_task_result(self, session: Session) -> None:
        bm = Benchmark(name="tr-bench", tasks_count=1)
        session.add(bm)
        session.commit()
        session.refresh(bm)

        run = Run(
            benchmark_id=bm.id,
            agent_name="test-agent",
            adapter_type="http",
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        tr = TaskResult(
            run_id=run.id,
            task_index=0,
            request={"task": "solve 2+2"},
            response={"answer": "4"},
            score=1.0,
            submitted_at=datetime.now(),
        )
        session.add(tr)
        session.commit()
        session.refresh(tr)

        assert tr.id is not None
        assert tr.run_id == run.id
        assert tr.task_index == 0
        assert tr.request == {"task": "solve 2+2"}
        assert tr.response == {"answer": "4"}
        assert tr.score == pytest.approx(1.0)
        assert tr.events is None
        assert tr.eval_results is None

    def test_task_result_with_events(self, session: Session) -> None:
        bm = Benchmark(name="ev-bench", tasks_count=1)
        session.add(bm)
        session.commit()
        session.refresh(bm)

        run = Run(
            benchmark_id=bm.id,
            agent_name="ev-agent",
            adapter_type="cli",
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        events = [
            {"type": "tool_call", "name": "calculator"},
            {"type": "llm_request", "tokens": 100},
        ]
        tr = TaskResult(
            run_id=run.id,
            task_index=0,
            request={"task": "test"},
            response={"result": "ok"},
            events=events,
            eval_results={"accuracy": 0.95},
            score=0.95,
            submitted_at=datetime.now(),
        )
        session.add(tr)
        session.commit()
        session.refresh(tr)

        assert tr.events == events
        assert tr.eval_results == {"accuracy": 0.95}

    def test_task_result_unique_constraint(self, session: Session) -> None:
        bm = Benchmark(name="uc-bench", tasks_count=2)
        session.add(bm)
        session.commit()
        session.refresh(bm)

        run = Run(
            benchmark_id=bm.id,
            agent_name="uc-agent",
            adapter_type="http",
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        tr1 = TaskResult(
            run_id=run.id,
            task_index=0,
            request={},
            response={},
            submitted_at=datetime.now(),
        )
        session.add(tr1)
        session.commit()

        tr2 = TaskResult(
            run_id=run.id,
            task_index=0,
            request={},
            response={},
            submitted_at=datetime.now(),
        )
        session.add(tr2)
        with pytest.raises(Exception):  # IntegrityError
            session.commit()

    def test_task_result_table_name(self) -> None:
        assert TaskResult.__tablename__ == "benchmark_task_results"
