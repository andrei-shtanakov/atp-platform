"""Benchmark API routes.

Provides endpoints for benchmark CRUD, run lifecycle (start, next-task,
submit, cancel), and leaderboards.  All database access goes through the
async session provided by the ``DBSession`` dependency.
"""

import uuid
from datetime import datetime
from typing import Any

from atp.loader.models import TestSuite
from atp.protocol import ATPRequest, ATPResponse, Task
from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from atp.dashboard.benchmark.models import (
    Benchmark,
    Run,
    RunStatus,
    TaskResult,
)
from atp.dashboard.benchmark.schemas import (
    BenchmarkCreate,
    BenchmarkResponse,
    LeaderboardEntry,
    RunResponse,
    RunStatusResponse,
    SubmitRequest,
    TaskResultResponse,
)
from atp.dashboard.v2.dependencies import DBSession
from atp.dashboard.v2.rate_limit import limiter
from atp.dashboard.webhook import (
    build_webhook_payload,
    schedule_webhook,
    validate_webhook_url,
)

router = APIRouter(prefix="/v1", tags=["benchmarks"])

MAX_RUN_EVENTS = 1000


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _benchmark_to_response(bm: Benchmark) -> BenchmarkResponse:
    return BenchmarkResponse(
        id=bm.id,
        name=bm.name,
        description=bm.description or "",
        tasks_count=bm.tasks_count,
        tags=bm.tags or [],
        version=bm.version or "",
        family_tag=bm.family_tag,
        created_at=bm.created_at.isoformat() if bm.created_at else "",
    )


def _run_to_response(run: Run) -> RunResponse:
    return RunResponse(
        id=run.id,
        benchmark_id=run.benchmark_id,
        agent_name=run.agent_name,
        adapter_type=run.adapter_type,
        status=run.status,
        current_task_index=run.current_task_index,
        total_score=run.total_score,
        started_at=(run.started_at.isoformat() if run.started_at else ""),
        finished_at=(run.finished_at.isoformat() if run.finished_at else None),
    )


# ------------------------------------------------------------------
# Benchmark CRUD
# ------------------------------------------------------------------


@router.get("/benchmarks", response_model=list[BenchmarkResponse])
@limiter.limit("120/minute")
async def list_benchmarks(
    request: Request,
    session: DBSession,
) -> list[BenchmarkResponse]:
    """List all benchmarks."""
    result = await session.execute(select(Benchmark).order_by(Benchmark.id))
    benchmarks = result.scalars().all()
    return [_benchmark_to_response(bm) for bm in benchmarks]


@router.get(
    "/benchmarks/{benchmark_id}",
    response_model=BenchmarkResponse,
)
@limiter.limit("120/minute")
async def get_benchmark(
    request: Request,
    benchmark_id: int,
    session: DBSession,
) -> BenchmarkResponse:
    """Get benchmark details by id."""
    bm = await session.get(Benchmark, benchmark_id)
    if bm is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark {benchmark_id} not found",
        )
    return _benchmark_to_response(bm)


@router.post(
    "/benchmarks",
    response_model=BenchmarkResponse,
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit("120/minute")
async def create_benchmark(
    request: Request,
    data: BenchmarkCreate,
    session: DBSession,
) -> BenchmarkResponse:
    """Create a new benchmark from a test suite definition."""
    suite = TestSuite.model_validate(data.suite)
    tasks_count = len(suite.tests)

    if data.webhook_url:
        try:
            validate_webhook_url(data.webhook_url)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid webhook URL: {e}",
            )

    bm = Benchmark(
        name=data.name,
        description=data.description,
        suite=data.suite,
        tasks_count=tasks_count,
        tags=data.tags,
        version=data.version,
        family_tag=data.family_tag,
        parent_id=data.parent_id,
        webhook_url=data.webhook_url,
    )
    session.add(bm)
    await session.flush()
    await session.refresh(bm)
    return _benchmark_to_response(bm)


# ------------------------------------------------------------------
# Run lifecycle
# ------------------------------------------------------------------


@router.post(
    "/benchmarks/{benchmark_id}/start",
    response_model=RunResponse,
)
@limiter.limit("120/minute")
async def start_run(
    request: Request,
    benchmark_id: int,
    session: DBSession,
    timeout: int = Query(default=3600),
    agent_name: str = Query(default=""),
) -> RunResponse:
    """Start a new benchmark run."""
    bm = await session.get(Benchmark, benchmark_id)
    if bm is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark {benchmark_id} not found",
        )

    run = Run(
        benchmark_id=benchmark_id,
        agent_name=agent_name,
        adapter_type="sdk",
        status=RunStatus.IN_PROGRESS,
        current_task_index=0,
        timeout_seconds=timeout,
        started_at=datetime.now(),
    )
    session.add(run)
    await session.flush()
    await session.refresh(run)
    return _run_to_response(run)


@router.get("/runs/{run_id}/next-task")
@limiter.limit("120/minute")
async def next_task(
    request: Request,
    run_id: int,
    session: DBSession,
    batch: int = Query(default=1, ge=1),
) -> Any:
    """Get the next task(s) for a run as ATPRequest dict(s).

    Use ``?batch=N`` to fetch up to N tasks atomically.  When ``batch=1``
    (default) a single dict is returned for backward compatibility.  When
    ``batch>1`` a list is returned.  Returns 204 No Content when all tasks
    have been consumed.
    """
    from atp.dashboard.v2.config import get_config

    config = get_config()
    batch = min(batch, config.batch_max_size)

    # Atomic increment by batch: avoids read-then-write race condition
    stmt = (
        update(Run)
        .where(Run.id == run_id, Run.status == RunStatus.IN_PROGRESS)
        .values(current_task_index=Run.current_task_index + batch)
        .returning(Run.current_task_index, Run.benchmark_id)
    )
    result = await session.execute(stmt)
    row = result.one_or_none()
    if row is None:
        # Either run doesn't exist or is not in progress
        run = await session.get(Run, run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run {run_id} not found",
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    new_index, benchmark_id = row
    start_idx = new_index - batch

    bm = await session.get(Benchmark, benchmark_id)
    if bm is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmark not found",
        )

    suite = TestSuite.model_validate(bm.suite)

    tasks: list[dict[str, Any]] = []
    for idx in range(start_idx, new_index):
        if idx >= len(suite.tests):
            break
        test_def = suite.tests[idx]

        # Build constraints
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

        atp_request = ATPRequest(
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
                "run_id": run_id,
            },
        )
        tasks.append(atp_request.model_dump())

    await session.flush()

    if not tasks:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # Backward compat: batch=1 returns a single dict, not a list
    if batch == 1:
        return tasks[0]
    return tasks


@router.post("/runs/{run_id}/submit")
@limiter.limit("120/minute")
async def submit_result(
    request: Request,
    run_id: int,
    data: SubmitRequest,
    session: DBSession,
) -> dict[str, Any]:
    """Submit a task result for a run."""
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    task_index = data.task_index

    response = ATPResponse.model_validate(data.response)
    score = 100.0 if response.status == "completed" else 0.0

    tr = TaskResult(
        run_id=run_id,
        task_index=task_index,
        request={},
        response=data.response,
        events=data.events,
        score=score,
        submitted_at=datetime.now(),
    )
    session.add(tr)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(f"Duplicate submission for run {run_id} task_index {task_index}"),
        )
    await session.refresh(tr)

    # Check if all tasks done
    bm = await session.get(Benchmark, run.benchmark_id)
    if bm is not None and task_index + 1 >= bm.tasks_count:
        # Finalize run
        results = await session.execute(
            select(TaskResult).where(TaskResult.run_id == run.id)
        )
        all_results = results.scalars().all()
        scores = [r.score for r in all_results if r.score is not None]
        run.total_score = sum(scores) / len(scores) if scores else 0.0
        run.status = RunStatus.COMPLETED
        run.finished_at = datetime.now()
        await session.flush()

        # Trigger webhook if configured
        if bm.webhook_url:
            payload = build_webhook_payload(
                "run.completed",
                bm,
                run,
                tasks_total=bm.tasks_count,
            )
            schedule_webhook(bm.webhook_url, payload)

    return {
        "task_index": tr.task_index,
        "score": tr.score,
    }


@router.get(
    "/runs/{run_id}/status",
    response_model=RunStatusResponse,
)
@limiter.limit("120/minute")
async def get_run_status(
    request: Request,
    run_id: int,
    session: DBSession,
) -> RunStatusResponse:
    """Get run status with completed tasks."""
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    bm = await session.get(Benchmark, run.benchmark_id)
    tasks_count = bm.tasks_count if bm else 0

    results = await session.execute(
        select(TaskResult)
        .where(TaskResult.run_id == run_id)
        .order_by(TaskResult.task_index)
    )
    completed = results.scalars().all()

    return RunStatusResponse(
        id=run.id,
        status=run.status,
        current_task_index=run.current_task_index,
        tasks_count=tasks_count,
        total_score=run.total_score,
        completed_tasks=[
            TaskResultResponse(
                task_index=tr.task_index,
                score=tr.score,
                eval_results=None,
            )
            for tr in completed
        ],
    )


@router.post("/runs/{run_id}/cancel")
@limiter.limit("120/minute")
async def cancel_run(
    request: Request,
    run_id: int,
    session: DBSession,
) -> dict[str, str]:
    """Cancel a benchmark run."""
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    run.status = RunStatus.CANCELLED
    run.finished_at = datetime.now()
    await session.flush()
    return {"status": "cancelled"}


# ------------------------------------------------------------------
# Events
# ------------------------------------------------------------------


@router.post("/runs/{run_id}/events")
@limiter.limit("120/minute")
async def emit_events(
    request: Request,
    run_id: int,
    data: dict[str, Any],
    session: DBSession,
) -> dict[str, Any]:
    """Append events to a running benchmark run.

    Events are stored in Run.events JSON column.
    Maximum 1000 events per run.
    """
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )
    if run.status != RunStatus.IN_PROGRESS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(f"Run {run_id} is not in progress (status: {run.status})"),
        )

    new_events = data.get("events", [])
    if not isinstance(new_events, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="events must be a list",
        )

    current_events = run.events or []
    if len(current_events) + len(new_events) > MAX_RUN_EVENTS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Event limit exceeded: run has {len(current_events)}"
                f" events, trying to add {len(new_events)},"
                f" max is {MAX_RUN_EVENTS}"
            ),
        )

    # Append events (must reassign for SQLAlchemy to detect change)
    updated = current_events + new_events
    run.events = updated
    await session.flush()

    return {"accepted": len(new_events), "total": len(updated)}


# ------------------------------------------------------------------
# Leaderboard
# ------------------------------------------------------------------


@router.get(
    "/benchmarks/{benchmark_id}/leaderboard",
    response_model=list[LeaderboardEntry],
)
@limiter.limit("120/minute")
async def get_leaderboard(
    request: Request,
    benchmark_id: int,
    session: DBSession,
) -> list[LeaderboardEntry]:
    """Get best scores per agent for a benchmark."""
    result = await session.execute(
        select(
            Run.user_id,
            Run.agent_name,
            func.max(Run.total_score).label("best_score"),
            func.count(Run.id).label("run_count"),
        )
        .where(
            Run.benchmark_id == benchmark_id,
            Run.status == RunStatus.COMPLETED,
        )
        .group_by(Run.user_id, Run.agent_name)
        .order_by(func.max(Run.total_score).desc())
    )
    rows = result.all()
    return [
        LeaderboardEntry(
            user_id=row.user_id or 0,
            agent_name=row.agent_name,
            best_score=row.best_score or 0.0,
            run_count=row.run_count,
        )
        for row in rows
    ]
