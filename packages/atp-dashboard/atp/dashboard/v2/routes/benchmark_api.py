"""Benchmark API routes.

Provides endpoints for benchmark CRUD, run lifecycle (start, next-task,
submit, cancel), and leaderboards.  All database access goes through the
async session provided by the ``DBSession`` dependency.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from atp.loader.models import TestDefinition, TestSuite
from atp.protocol import ATPEvent, ATPRequest, ATPResponse, Task
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

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
from atp.dashboard.benchmark.score_contract import (
    empty_score_components,
    run_score_semantics,
)
from atp.dashboard.benchmark.scoring import derive_run_score_view, score_submission
from atp.dashboard.models import User
from atp.dashboard.v2.dependencies import (
    AdminUser,
    BenchmarkCaller,
    DBSession,
    reject_tournament_token,
)
from atp.dashboard.v2.rate_limit import limiter
from atp.dashboard.webhook import (
    build_webhook_payload,
    schedule_webhook,
    validate_webhook_url,
)

# LABS-TSA PR-3: router-level guard rejects tournament-agent tokens across
# the whole benchmark API, including endpoints that don't otherwise require
# auth (e.g. listing benchmarks). Endpoints that additionally require a
# user use the BenchmarkCaller dep, which stacks the same check but also
# enforces authentication.
router = APIRouter(
    prefix="/v1",
    tags=["benchmarks"],
    dependencies=[Depends(reject_tournament_token)],
)

logger = logging.getLogger(__name__)

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
    """A freshly started run: nothing submitted, so nothing evaluated.

    The completion label here is the honest one even on a server wired with
    evaluators — no evaluator has run on this run yet, and a server's capability
    is not evidence about a particular run. `GET /runs/{id}/status` derives the
    real label from the submitted tasks.
    """
    return RunResponse(
        id=run.id,
        benchmark_id=run.benchmark_id,
        agent_name=run.agent_name,
        adapter_type=run.adapter_type,
        status=run.status,
        current_task_index=run.current_task_index,
        total_score=run.total_score,
        score_semantics=run_score_semantics(),
        score_components=empty_score_components(),
        started_at=(run.started_at.isoformat() if run.started_at else ""),
        finished_at=(run.finished_at.isoformat() if run.finished_at else None),
    )


def _parse_events(raw: list[dict[str, Any]] | None) -> list[ATPEvent]:
    """Parse submitted events for evaluation, dropping the ones that won't.

    Behaviour assertions read the trace, so events have to become models. They
    arrive from a self-service client, though, and rejecting the whole
    submission over one malformed event would turn a scoring question into a
    400 — the raw list is stored either way, so nothing is lost.
    """
    parsed: list[ATPEvent] = []
    for item in raw or []:
        try:
            parsed.append(ATPEvent.model_validate(item))
        except Exception:
            logger.debug("dropping unparseable event from submission", exc_info=True)
    return parsed


def _test_definition_at(bm: Benchmark, task_index: int) -> TestDefinition | None:
    """The suite's test at this index, or None if there isn't one.

    A submission for an index the suite does not have is malformed rather than
    unscoreable, but it must not take the API down: an unparseable stored suite
    is a data problem, and the completion score remains available.
    """
    try:
        suite = TestSuite.model_validate(bm.suite)
    except Exception:
        logger.warning("benchmark %s has an unparseable suite", bm.id, exc_info=True)
        return None
    if not 0 <= task_index < len(suite.tests):
        return None
    return suite.tests[task_index]


async def _load_run_for_user(
    session: AsyncSession,
    run_id: int,
    user: User,
) -> Run:
    """Load a run and verify the current user owns it.

    Raises 404 if the run does not exist OR belongs to another user.
    We deliberately return 404 rather than 403 to avoid leaking the
    existence of run_ids that the caller does not own (run_id is a
    trivially-enumerable autoincrement integer).
    """
    run = await session.get(Run, run_id)
    if run is None or run.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )
    return run


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
    current_user: AdminUser,
) -> BenchmarkResponse:
    """Create a new benchmark from a test suite definition.

    Admin-only: benchmark definitions shape the public leaderboard and
    must not be created by arbitrary authenticated users.
    """
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
    current_user: BenchmarkCaller,
    timeout: int = Query(default=3600),
    agent_name: str = Query(default=""),
) -> RunResponse:
    """Start a new benchmark run owned by the current user."""
    bm = await session.get(Benchmark, benchmark_id)
    if bm is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark {benchmark_id} not found",
        )

    run = Run(
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
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
    current_user: BenchmarkCaller,
    batch: int = Query(default=1, ge=1),
) -> Any:
    """Get the next task(s) for a run as ATPRequest dict(s).

    Use ``?batch=N`` to fetch up to N tasks atomically.  When ``batch=1``
    (default) a single dict is returned for backward compatibility.  When
    ``batch>1`` a list is returned.  Returns 204 No Content when all tasks
    have been consumed.

    Requires the caller to own the run.  Non-owners receive 404.
    """
    from atp.dashboard.v2.config import get_config

    config = get_config()
    batch = min(batch, config.batch_max_size)

    # Ownership pre-check — raises 404 if missing or owned by another user.
    await _load_run_for_user(session, run_id, current_user)

    # Atomic increment by batch: avoids read-then-write race condition.
    # The user_id filter guards against a race between the pre-check and
    # the update (e.g. admin cancellation in between).
    stmt = (
        update(Run)
        .where(
            Run.id == run_id,
            Run.user_id == current_user.id,
            Run.status == RunStatus.IN_PROGRESS,
        )
        .values(current_task_index=Run.current_task_index + batch)
        .returning(Run.current_task_index, Run.benchmark_id)
    )
    result = await session.execute(stmt)
    row = result.one_or_none()
    if row is None:
        # Run exists and is owned by us (pre-check passed) but is not
        # IN_PROGRESS — either already COMPLETED or CANCELLED.
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
    current_user: BenchmarkCaller,
) -> dict[str, Any]:
    """Submit a task result for a run owned by the current user.

    The submission is evaluated here, under the server's policy, when the app
    was composed with evaluators and the suite asks for something this plane
    may run. Otherwise the score is the completion score it has always been —
    `GET /runs/{id}/status` reports which of the two happened.
    """
    run = await _load_run_for_user(session, run_id, current_user)

    task_index = data.task_index

    try:
        response = ATPResponse.model_validate(data.response)
    except ValidationError as exc:
        # `SubmitRequest.response` is a bare dict, so this is the first place
        # the protocol is actually checked — and a raised ValidationError here
        # is an unhandled exception, i.e. a 500. A self-service caller that
        # sent a malformed response should be told what is wrong.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"response is not a valid ATPResponse: {exc.errors()}",
        ) from exc

    bm = await session.get(Benchmark, run.benchmark_id)
    scored = await score_submission(
        request.app.state.evaluation.resolver,
        _test_definition_at(bm, task_index) if bm is not None else None,
        response,
        _parse_events(data.events),
    )

    tr = TaskResult(
        run_id=run_id,
        task_index=task_index,
        request={},
        response=data.response,
        events=data.events,
        eval_results=scored.records,
        score=scored.score,
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
    current_user: BenchmarkCaller,
) -> RunStatusResponse:
    """Get run status with completed tasks (owner-only)."""
    run = await _load_run_for_user(session, run_id, current_user)

    bm = await session.get(Benchmark, run.benchmark_id)
    tasks_count = bm.tasks_count if bm else 0

    results = await session.execute(
        select(TaskResult)
        .where(TaskResult.run_id == run_id)
        .order_by(TaskResult.task_index)
    )
    completed = results.scalars().all()

    # Derived from the stored evidence on every read, rather than frozen when
    # the run finished: what the number *means* is then always recomputed from
    # what actually ran, and a change here needs no backfill.
    semantics, components = derive_run_score_view(
        [tr.eval_results for tr in completed], tasks_count
    )

    return RunStatusResponse(
        id=run.id,
        status=run.status,
        current_task_index=run.current_task_index,
        tasks_count=tasks_count,
        total_score=run.total_score,
        score_semantics=semantics,
        score_components=components,
        completed_tasks=[
            TaskResultResponse(
                task_index=tr.task_index,
                score=tr.score,
                eval_results=tr.eval_results,
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
    current_user: BenchmarkCaller,
) -> dict[str, str]:
    """Cancel a benchmark run (owner-only)."""
    run = await _load_run_for_user(session, run_id, current_user)

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
    current_user: BenchmarkCaller,
) -> dict[str, Any]:
    """Append events to a running benchmark run (owner-only).

    Events are stored in Run.events JSON column.
    Maximum 1000 events per run.
    """
    run = await _load_run_for_user(session, run_id, current_user)
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
