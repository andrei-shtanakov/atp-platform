"""Business logic for the Benchmark platform."""

import uuid
from datetime import datetime
from typing import Any

from atp.loader.models import TestSuite
from atp.protocol import ATPRequest, ATPResponse, Task
from sqlalchemy import func
from sqlalchemy.orm import Session

from atp.dashboard.benchmark.models import (
    Benchmark,
    Run,
    RunStatus,
    TaskResult,
)
from atp.dashboard.benchmark.schemas import BenchmarkCreate, SubmitRequest


class BenchmarkService:
    """Synchronous service for benchmark CRUD and run management."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Benchmark CRUD
    # ------------------------------------------------------------------

    def create_benchmark(self, data: BenchmarkCreate, user_id: int) -> Benchmark:
        """Create a benchmark, validating the suite and counting tests."""
        suite = TestSuite.model_validate(data.suite)
        tasks_count = len(suite.tests)

        bm = Benchmark(
            name=data.name,
            description=data.description,
            suite=data.suite,
            tasks_count=tasks_count,
            tags=data.tags,
            version=data.version,
            family_tag=data.family_tag,
            parent_id=data.parent_id,
            created_by=user_id,
        )
        self.session.add(bm)
        self.session.commit()
        self.session.refresh(bm)
        return bm

    def get_benchmark(self, benchmark_id: int) -> Benchmark | None:
        """Return a benchmark by id, or None."""
        return self.session.get(Benchmark, benchmark_id)

    def list_benchmarks(self, tenant_id: str = "default") -> list[Benchmark]:
        """List all benchmarks for a tenant."""
        return list(
            self.session.query(Benchmark).filter(Benchmark.tenant_id == tenant_id).all()
        )

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

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
            current_task_index=0,
            timeout_seconds=timeout_seconds,
            started_at=datetime.now(),
        )
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def get_run(self, run_id: int) -> Run | None:
        """Return a run by id, or None."""
        return self.session.get(Run, run_id)

    def next_task(self, run_id: int) -> dict[str, Any] | None:
        """Get the next task for a run, or None if all consumed.

        Atomically increments current_task_index and builds an
        ATPRequest dict from the corresponding TestDefinition.
        """
        run = self.session.get(Run, run_id)
        if run is None:
            return None

        bm = self.session.get(Benchmark, run.benchmark_id)
        if bm is None:
            return None

        suite = TestSuite.model_validate(bm.suite)
        idx = run.current_task_index

        if idx >= len(suite.tests):
            return None

        test_def = suite.tests[idx]

        # Increment atomically
        run.current_task_index = idx + 1
        self.session.commit()

        # Build ATPRequest
        constraints: dict[str, Any] = {}
        if test_def.constraints.max_steps is not None:
            constraints["max_steps"] = test_def.constraints.max_steps
        if test_def.constraints.max_tokens is not None:
            constraints["max_tokens"] = test_def.constraints.max_tokens
        if test_def.constraints.timeout_seconds is not None:
            constraints["timeout_seconds"] = test_def.constraints.timeout_seconds
        if test_def.constraints.allowed_tools is not None:
            constraints["allowed_tools"] = test_def.constraints.allowed_tools
        if test_def.constraints.budget_usd is not None:
            constraints["budget_usd"] = test_def.constraints.budget_usd

        request = ATPRequest(
            task_id=str(uuid.uuid4()),
            task=Task(
                description=test_def.task.description,
                input_data=test_def.task.input_data,
                expected_artifacts=test_def.task.expected_artifacts,
            ),
            constraints=constraints,
            metadata={
                "test_id": test_def.id,
                "test_name": test_def.name,
                "task_index": idx,
                "run_id": run.id,
            },
        )
        return request.model_dump()

    # ------------------------------------------------------------------
    # Submission and scoring
    # ------------------------------------------------------------------

    def submit(self, run_id: int, data: SubmitRequest) -> TaskResult:
        """Submit a task result, score it, and finalize if done."""
        run = self.session.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        task_index = data.task_index

        response = ATPResponse.model_validate(data.response)
        score = self._evaluate_sync(response)

        tr = TaskResult(
            run_id=run_id,
            task_index=task_index,
            request={},
            response=data.response,
            events=data.events,
            score=score,
            submitted_at=datetime.now(),
        )
        self.session.add(tr)
        self.session.commit()
        self.session.refresh(tr)

        # Check if all tasks are done
        bm = self.session.get(Benchmark, run.benchmark_id)
        if bm is not None and task_index + 1 >= bm.tasks_count:
            self._finalize_run(run)

        return tr

    def cancel_run(self, run_id: int) -> None:
        """Cancel a run."""
        run = self.session.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        run.status = RunStatus.CANCELLED
        run.finished_at = datetime.now()
        self.session.commit()

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def get_leaderboard(self, benchmark_id: int) -> list[dict[str, Any]]:
        """Best total_score per user per benchmark."""
        rows = (
            self.session.query(
                Run.user_id,
                Run.agent_name,
                func.max(Run.total_score).label("best_score"),
                func.count(Run.id).label("run_count"),
            )
            .filter(
                Run.benchmark_id == benchmark_id,
                Run.status == RunStatus.COMPLETED,
            )
            .group_by(Run.user_id, Run.agent_name)
            .order_by(func.max(Run.total_score).desc())
            .all()
        )
        return [
            {
                "user_id": row.user_id,
                "agent_name": row.agent_name,
                "best_score": row.best_score,
                "run_count": row.run_count,
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_sync(self, response: ATPResponse) -> float:
        """MVP scoring: 100 if completed, 0 otherwise."""
        return 100.0 if response.status == "completed" else 0.0

    def _finalize_run(self, run: Run) -> None:
        """Compute average score and mark run as completed."""
        results = (
            self.session.query(TaskResult).filter(TaskResult.run_id == run.id).all()
        )
        scores = [r.score for r in results if r.score is not None]
        run.total_score = sum(scores) / len(scores) if scores else 0.0
        run.status = RunStatus.COMPLETED
        run.finished_at = datetime.now()
        self.session.commit()
