"""FastAPI routes for ATP Dashboard API."""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atp.dashboard.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
    create_user,
    get_current_active_user,
    get_current_admin_user,
    get_current_user,
)
from atp.dashboard.database import get_database
from atp.dashboard.models import (
    Agent,
    RunResult,
    SuiteExecution,
    TestExecution,
    User,
)
from atp.dashboard.schemas import (
    AgentComparisonMetrics,
    AgentComparisonResponse,
    AgentCreate,
    AgentExecutionDetail,
    AgentResponse,
    AgentUpdate,
    DashboardSummary,
    EvaluationResultResponse,
    EventSummary,
    RunResultSummary,
    ScoreComponentResponse,
    SideBySideComparisonResponse,
    SuiteExecutionDetail,
    SuiteExecutionList,
    SuiteExecutionSummary,
    SuiteTrend,
    TestComparisonMetrics,
    TestExecutionDetail,
    TestExecutionList,
    TestExecutionSummary,
    TestTrend,
    Token,
    TrendDataPoint,
    TrendResponse,
    UserCreate,
    UserResponse,
)

# Create API router
router = APIRouter()


# ==================== Session Dependency ====================


async def get_session() -> AsyncSession:  # pragma: no cover
    """Get database session as a FastAPI dependency."""
    db = get_database()
    async with db.session() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
CurrentUser = Annotated[User | None, Depends(get_current_user)]
RequiredUser = Annotated[User, Depends(get_current_active_user)]
AdminUser = Annotated[User, Depends(get_current_admin_user)]


# ==================== Authentication Routes ====================


@router.post("/auth/token", response_model=Token, tags=["auth"])
async def login(
    session: SessionDep,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:  # pragma: no cover
    """Authenticate and get access token."""
    user = await authenticate_user(session, form_data.username, form_data.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires,
    )
    return Token(access_token=access_token)


@router.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["auth"],
)
async def register(  # pragma: no cover
    session: SessionDep, user_data: UserCreate
) -> UserResponse:
    """Register a new user."""
    try:
        user = await create_user(
            session,
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
        )
        await session.commit()
        return UserResponse.model_validate(user)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/auth/me", response_model=UserResponse, tags=["auth"])
async def get_me(current_user: RequiredUser) -> UserResponse:  # pragma: no cover
    """Get current user information."""
    return UserResponse.model_validate(current_user)


# ==================== Agent Routes ====================


@router.get("/agents", response_model=list[AgentResponse], tags=["agents"])
async def list_agents(  # pragma: no cover
    session: SessionDep, user: CurrentUser
) -> list[AgentResponse]:
    """List all agents."""
    stmt = select(Agent).order_by(Agent.name)
    result = await session.execute(stmt)
    agents = result.scalars().all()
    return [AgentResponse.model_validate(a) for a in agents]


@router.post(
    "/agents",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["agents"],
)
async def create_agent(
    session: SessionDep, agent_data: AgentCreate, user: RequiredUser
) -> AgentResponse:  # pragma: no cover
    """Create a new agent."""
    # Check for existing agent
    stmt = select(Agent).where(Agent.name == agent_data.name)
    result = await session.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent '{agent_data.name}' already exists",
        )

    agent = Agent(
        name=agent_data.name,
        agent_type=agent_data.agent_type,
        config=agent_data.config,
        description=agent_data.description,
    )
    session.add(agent)
    await session.commit()
    return AgentResponse.model_validate(agent)


@router.get("/agents/{agent_id}", response_model=AgentResponse, tags=["agents"])
async def get_agent(
    session: SessionDep, agent_id: int, user: CurrentUser
) -> AgentResponse:  # pragma: no cover
    """Get agent by ID."""
    agent = await session.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )
    return AgentResponse.model_validate(agent)


@router.patch("/agents/{agent_id}", response_model=AgentResponse, tags=["agents"])
async def update_agent(
    session: SessionDep,
    agent_id: int,
    agent_data: AgentUpdate,
    user: RequiredUser,
) -> AgentResponse:  # pragma: no cover
    """Update an agent."""
    agent = await session.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    if agent_data.agent_type is not None:
        agent.agent_type = agent_data.agent_type
    if agent_data.config is not None:
        agent.config = agent_data.config
    if agent_data.description is not None:
        agent.description = agent_data.description

    await session.commit()
    return AgentResponse.model_validate(agent)


@router.delete(
    "/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["agents"]
)
async def delete_agent(  # pragma: no cover
    session: SessionDep, agent_id: int, user: AdminUser
) -> None:
    """Delete an agent (admin only)."""
    agent = await session.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )
    await session.delete(agent)
    await session.commit()


# ==================== Suite Execution Routes ====================


@router.get("/suites", response_model=SuiteExecutionList, tags=["suites"])
async def list_suite_executions(
    session: SessionDep,
    user: CurrentUser,
    suite_name: str | None = None,
    agent_id: int | None = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> SuiteExecutionList:
    """List suite executions with optional filtering."""
    # Build query
    stmt = select(SuiteExecution).options(selectinload(SuiteExecution.agent))
    if suite_name:
        stmt = stmt.where(SuiteExecution.suite_name == suite_name)
    if agent_id:
        stmt = stmt.where(SuiteExecution.agent_id == agent_id)

    # Get total count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await session.execute(count_stmt)).scalar() or 0

    # Get paginated results
    stmt = stmt.order_by(SuiteExecution.started_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    executions = result.scalars().all()

    items = []
    for exec in executions:
        summary = SuiteExecutionSummary.model_validate(exec)
        summary.agent_name = exec.agent.name if exec.agent else None
        items.append(summary)

    return SuiteExecutionList(
        total=total,
        items=items,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/suites/{execution_id}", response_model=SuiteExecutionDetail, tags=["suites"]
)
async def get_suite_execution(
    session: SessionDep, execution_id: int, user: CurrentUser
) -> SuiteExecutionDetail:  # pragma: no cover
    """Get suite execution details."""
    stmt = (
        select(SuiteExecution)
        .where(SuiteExecution.id == execution_id)
        .options(
            selectinload(SuiteExecution.agent),
            selectinload(SuiteExecution.test_executions),
        )
    )
    result = await session.execute(stmt)
    execution = result.scalar_one_or_none()

    if execution is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite execution {execution_id} not found",
        )

    detail = SuiteExecutionDetail.model_validate(execution)
    detail.agent_name = execution.agent.name if execution.agent else None
    detail.tests = [
        TestExecutionSummary.model_validate(t) for t in execution.test_executions
    ]
    return detail


@router.get("/suites/names/list", response_model=list[str], tags=["suites"])
async def list_suite_names(  # pragma: no cover
    session: SessionDep, user: CurrentUser
) -> list[str]:
    """List unique suite names."""
    stmt = (
        select(SuiteExecution.suite_name).distinct().order_by(SuiteExecution.suite_name)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


# ==================== Test Execution Routes ====================


@router.get("/tests", response_model=TestExecutionList, tags=["tests"])
async def list_test_executions(
    session: SessionDep,
    user: CurrentUser,
    suite_execution_id: int | None = None,
    test_id: str | None = None,
    success: bool | None = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> TestExecutionList:
    """List test executions with optional filtering."""
    stmt = select(TestExecution)
    if suite_execution_id:
        stmt = stmt.where(TestExecution.suite_execution_id == suite_execution_id)
    if test_id:
        stmt = stmt.where(TestExecution.test_id == test_id)
    if success is not None:
        stmt = stmt.where(TestExecution.success == success)

    # Get total count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await session.execute(count_stmt)).scalar() or 0

    # Get paginated results
    stmt = stmt.order_by(TestExecution.started_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    executions = result.scalars().all()

    return TestExecutionList(
        total=total,
        items=[TestExecutionSummary.model_validate(e) for e in executions],
        limit=limit,
        offset=offset,
    )


@router.get("/tests/{execution_id}", response_model=TestExecutionDetail, tags=["tests"])
async def get_test_execution(
    session: SessionDep, execution_id: int, user: CurrentUser
) -> TestExecutionDetail:  # pragma: no cover
    """Get test execution details."""
    stmt = (
        select(TestExecution)
        .where(TestExecution.id == execution_id)
        .options(
            selectinload(TestExecution.run_results),
            selectinload(TestExecution.evaluation_results),
            selectinload(TestExecution.score_components),
        )
    )
    result = await session.execute(stmt)
    execution = result.scalar_one_or_none()

    if execution is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test execution {execution_id} not found",
        )

    detail = TestExecutionDetail.model_validate(execution)
    detail.runs = [RunResultSummary.model_validate(r) for r in execution.run_results]
    detail.evaluations = [
        EvaluationResultResponse.model_validate(e) for e in execution.evaluation_results
    ]
    detail.score_components = [
        ScoreComponentResponse.model_validate(s) for s in execution.score_components
    ]
    return detail


# ==================== Trend Routes ====================


@router.get("/trends/suite", response_model=TrendResponse, tags=["trends"])
async def get_suite_trends(
    session: SessionDep,
    user: CurrentUser,
    suite_name: str,
    agent_name: str | None = None,
    metric: str = Query(
        default="success_rate", pattern="^(success_rate|score|duration)$"
    ),
    limit: int = Query(default=30, le=100),
) -> TrendResponse:
    """Get historical trends for a suite."""
    # Build query
    stmt = (
        select(SuiteExecution)
        .where(SuiteExecution.suite_name == suite_name)
        .options(selectinload(SuiteExecution.agent))
        .order_by(SuiteExecution.started_at.desc())
        .limit(limit)
    )
    if agent_name:
        stmt = stmt.join(Agent).where(Agent.name == agent_name)

    result = await session.execute(stmt)
    executions = result.scalars().all()

    # Build trend data
    data_points: list[TrendDataPoint] = []
    for exec in reversed(executions):
        if metric == "success_rate":
            value = exec.success_rate
        elif metric == "score":
            # Calculate average score from tests
            if exec.test_executions:
                scores = [t.score for t in exec.test_executions if t.score is not None]
                value = sum(scores) / len(scores) if scores else 0.0
            else:
                value = 0.0
        else:  # duration
            value = exec.duration_seconds or 0.0

        data_points.append(
            TrendDataPoint(
                timestamp=exec.started_at,
                value=value,
                execution_id=exec.id,
            )
        )

    agent_name_display = agent_name or "all"
    return TrendResponse(
        suite_trends=[
            SuiteTrend(
                suite_name=suite_name,
                agent_name=agent_name_display,
                data_points=data_points,
                metric=metric,
            )
        ]
    )


@router.get("/trends/test", response_model=TrendResponse, tags=["trends"])
async def get_test_trends(
    session: SessionDep,
    user: CurrentUser,
    suite_name: str,
    test_id: str,
    agent_name: str | None = None,
    metric: str = Query(default="score", pattern="^(score|duration|success_rate)$"),
    limit: int = Query(default=30, le=100),
) -> TrendResponse:
    """Get historical trends for a specific test."""
    # Build query
    stmt = (
        select(TestExecution)
        .join(SuiteExecution)
        .where(
            SuiteExecution.suite_name == suite_name,
            TestExecution.test_id == test_id,
        )
        .order_by(TestExecution.started_at.desc())
        .limit(limit)
    )
    if agent_name:
        stmt = stmt.join(Agent).where(Agent.name == agent_name)

    result = await session.execute(stmt)
    executions = result.scalars().all()

    # Get test name from first execution
    test_name = executions[0].test_name if executions else test_id

    # Build trend data
    data_points: list[TrendDataPoint] = []
    for exec in reversed(executions):
        if metric == "score":
            value = exec.score or 0.0
        elif metric == "duration":
            value = exec.duration_seconds or 0.0
        else:  # success_rate
            value = 1.0 if exec.success else 0.0

        data_points.append(
            TrendDataPoint(
                timestamp=exec.started_at,
                value=value,
                execution_id=exec.id,
            )
        )

    return TrendResponse(
        test_trends=[
            TestTrend(
                test_id=test_id,
                test_name=test_name,
                data_points=data_points,
                metric=metric,
            )
        ]
    )


# ==================== Comparison Routes ====================


@router.get("/compare/agents", response_model=AgentComparisonResponse, tags=["compare"])
async def compare_agents(
    session: SessionDep,
    user: CurrentUser,
    suite_name: str,
    agents: list[str] = Query(...),
    limit_per_agent: int = Query(default=10, le=50),
) -> AgentComparisonResponse:
    """Compare multiple agents on a suite."""
    agent_metrics: list[AgentComparisonMetrics] = []
    test_metrics_map: dict[str, dict[str, AgentComparisonMetrics]] = {}

    for agent_name in agents:
        # Get agent's executions
        stmt = (
            select(SuiteExecution)
            .join(Agent)
            .where(
                SuiteExecution.suite_name == suite_name,
                Agent.name == agent_name,
            )
            .options(selectinload(SuiteExecution.test_executions))
            .order_by(SuiteExecution.started_at.desc())
            .limit(limit_per_agent)
        )
        result = await session.execute(stmt)
        executions = list(result.scalars().all())

        if not executions:
            continue

        # Calculate aggregate metrics
        total_executions = len(executions)
        avg_success_rate = sum(e.success_rate for e in executions) / total_executions

        # Calculate average score
        all_scores: list[float] = []
        for exec in executions:
            for test in exec.test_executions:
                if test.score is not None:
                    all_scores.append(test.score)
        avg_score = sum(all_scores) / len(all_scores) if all_scores else None

        # Calculate average duration
        durations = [e.duration_seconds for e in executions if e.duration_seconds]
        avg_duration = sum(durations) / len(durations) if durations else None

        # Latest execution metrics
        latest = executions[0]
        latest_scores = [t.score for t in latest.test_executions if t.score is not None]
        latest_score = (
            sum(latest_scores) / len(latest_scores) if latest_scores else None
        )

        agent_metrics.append(
            AgentComparisonMetrics(
                agent_name=agent_name,
                total_executions=total_executions,
                avg_success_rate=avg_success_rate,
                avg_score=avg_score,
                avg_duration_seconds=avg_duration,
                latest_success_rate=latest.success_rate,
                latest_score=latest_score,
            )
        )

        # Collect per-test metrics
        for exec in executions:
            for test in exec.test_executions:
                if test.test_id not in test_metrics_map:
                    test_metrics_map[test.test_id] = {"_name": test.test_name}
                if agent_name not in test_metrics_map[test.test_id]:
                    test_metrics_map[test.test_id][agent_name] = {
                        "scores": [],
                        "durations": [],
                        "successes": [],
                    }
                data = test_metrics_map[test.test_id][agent_name]
                if test.score is not None:
                    data["scores"].append(test.score)
                if test.duration_seconds is not None:
                    data["durations"].append(test.duration_seconds)
                data["successes"].append(1 if test.success else 0)

    # Build test comparison metrics
    test_comparisons: list[TestComparisonMetrics] = []
    for test_id, agent_data in test_metrics_map.items():
        test_name = agent_data.pop("_name", test_id)
        metrics_by_agent: dict[str, AgentComparisonMetrics] = {}

        for agent_name, data in agent_data.items():
            if isinstance(data, dict) and "scores" in data:
                scores = data["scores"]
                durations = data["durations"]
                successes = data["successes"]

                metrics_by_agent[agent_name] = AgentComparisonMetrics(
                    agent_name=agent_name,
                    total_executions=len(successes),
                    avg_success_rate=sum(successes) / len(successes)
                    if successes
                    else 0,
                    avg_score=sum(scores) / len(scores) if scores else None,
                    avg_duration_seconds=sum(durations) / len(durations)
                    if durations
                    else None,
                    latest_success_rate=successes[-1] if successes else None,
                    latest_score=scores[-1] if scores else None,
                )

        test_comparisons.append(
            TestComparisonMetrics(
                test_id=test_id,
                test_name=test_name,
                metrics_by_agent=metrics_by_agent,
            )
        )

    return AgentComparisonResponse(
        suite_name=suite_name,
        agents=agent_metrics,
        tests=test_comparisons,
    )


def _format_event_summary(event: dict) -> EventSummary:
    """Format a raw event dict into an EventSummary.

    Args:
        event: Raw event dictionary from events_json.

    Returns:
        EventSummary with formatted data.
    """
    from datetime import datetime as dt

    event_type = event.get("event_type", "unknown")
    payload = event.get("payload", {})

    # Generate summary based on event type
    if event_type == "tool_call":
        tool = payload.get("tool", "unknown")
        status = payload.get("status", "")
        summary = f"Tool call: {tool} ({status})"
    elif event_type == "llm_request":
        model = payload.get("model", "unknown")
        tokens = payload.get("input_tokens", 0) + payload.get("output_tokens", 0)
        summary = f"LLM request: {model} ({tokens} tokens)"
    elif event_type == "reasoning":
        thought = payload.get("thought", "")
        step = payload.get("step", "")
        summary = thought[:50] + "..." if len(thought) > 50 else thought
        if not summary and step:
            summary = step
        if not summary:
            summary = "Reasoning step"
    elif event_type == "error":
        error_type = payload.get("error_type", "Error")
        message = payload.get("message", "")[:50]
        summary = f"{error_type}: {message}"
    elif event_type == "progress":
        percentage = payload.get("percentage", 0)
        message = payload.get("message", "")
        if message:
            summary = f"Progress: {percentage}% - {message}"
        else:
            summary = f"Progress: {percentage}%"
    else:
        summary = f"Event: {event_type}"

    # Parse timestamp
    timestamp_str = event.get("timestamp", "")
    if timestamp_str:
        try:
            timestamp = dt.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            timestamp = dt.now()
    else:
        timestamp = dt.now()

    return EventSummary(
        sequence=event.get("sequence", 0),
        timestamp=timestamp,
        event_type=event_type,
        summary=summary,
        data=payload,
    )


def _calculate_metrics_from_run(
    run: RunResult,
) -> tuple[int | None, int | None, int | None, int | None, float | None]:
    """Calculate metrics from a run result.

    Args:
        run: RunResult instance.

    Returns:
        Tuple of (total_tokens, total_steps, tool_calls, llm_calls, cost_usd).
    """
    return (
        run.total_tokens,
        run.total_steps,
        run.tool_calls,
        run.llm_calls,
        run.cost_usd,
    )


@router.get(
    "/compare/side-by-side",
    response_model=SideBySideComparisonResponse,
    tags=["compare"],
)
async def get_side_by_side_comparison(
    session: SessionDep,
    user: CurrentUser,
    suite_name: str,
    test_id: str,
    agents: list[str] = Query(..., min_length=2, max_length=3),
) -> SideBySideComparisonResponse:
    """Get detailed side-by-side comparison of agents on a specific test.

    This endpoint returns the latest test execution for each agent on the
    specified test, including formatted events for timeline visualization
    and metrics for comparison.

    Args:
        session: Database session.
        user: Current user (optional auth).
        suite_name: Name of the test suite.
        test_id: ID of the specific test.
        agents: List of agent names to compare (2-3 agents).

    Returns:
        SideBySideComparisonResponse with execution details for each agent.

    Raises:
        HTTPException: If no executions found for any agent.
    """
    agent_details: list[AgentExecutionDetail] = []
    test_name: str | None = None

    for agent_name in agents:
        # Query latest test execution for this agent on this test
        stmt = (
            select(TestExecution)
            .join(SuiteExecution)
            .join(Agent)
            .where(
                SuiteExecution.suite_name == suite_name,
                TestExecution.test_id == test_id,
                Agent.name == agent_name,
            )
            .options(selectinload(TestExecution.run_results))
            .order_by(TestExecution.started_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        execution = result.scalar_one_or_none()

        if execution is None:
            # Agent has no execution for this test - skip but don't fail
            continue

        # Get test name from first found execution
        if test_name is None:
            test_name = execution.test_name

        # Get the latest run result (or first if only one)
        run_results = sorted(execution.run_results, key=lambda r: r.run_number)
        latest_run = run_results[-1] if run_results else None

        # Extract and format events
        events: list[EventSummary] = []
        if latest_run and latest_run.events_json:
            for event in latest_run.events_json:
                events.append(_format_event_summary(event))
            # Sort by sequence
            events.sort(key=lambda e: e.sequence)

        # Calculate metrics
        total_tokens: int | None = None
        total_steps: int | None = None
        tool_calls: int | None = None
        llm_calls: int | None = None
        cost_usd: float | None = None

        if latest_run:
            total_tokens, total_steps, tool_calls, llm_calls, cost_usd = (
                _calculate_metrics_from_run(latest_run)
            )

        agent_details.append(
            AgentExecutionDetail(
                agent_name=agent_name,
                test_execution_id=execution.id,
                score=execution.score,
                success=execution.success,
                duration_seconds=execution.duration_seconds,
                total_tokens=total_tokens,
                total_steps=total_steps,
                tool_calls=tool_calls,
                llm_calls=llm_calls,
                cost_usd=cost_usd,
                events=events,
            )
        )

    if not agent_details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No executions found for test '{test_id}' in suite '{suite_name}' "
            f"for any of the specified agents",
        )

    return SideBySideComparisonResponse(
        suite_name=suite_name,
        test_id=test_id,
        test_name=test_name or test_id,
        agents=agent_details,
    )


# ==================== Dashboard Routes ====================


@router.get("/dashboard/summary", response_model=DashboardSummary, tags=["dashboard"])
async def get_dashboard_summary(
    session: SessionDep, user: CurrentUser
) -> DashboardSummary:  # pragma: no cover
    """Get dashboard summary statistics."""
    # Count agents
    agent_count = (await session.execute(select(func.count(Agent.id)))).scalar() or 0

    # Count unique suites
    suite_count = (
        await session.execute(
            select(func.count(func.distinct(SuiteExecution.suite_name)))
        )
    ).scalar() or 0

    # Count total executions
    exec_count = (
        await session.execute(select(func.count(SuiteExecution.id)))
    ).scalar() or 0

    # Get recent executions (last 10)
    stmt = (
        select(SuiteExecution)
        .options(selectinload(SuiteExecution.agent))
        .order_by(SuiteExecution.started_at.desc())
        .limit(10)
    )
    result = await session.execute(stmt)
    recent_execs = result.scalars().all()

    # Calculate recent success rate and score
    if recent_execs:
        recent_success_rate = sum(e.success_rate for e in recent_execs) / len(
            recent_execs
        )

        # Get scores from test executions
        all_scores: list[float] = []
        for exec in recent_execs:
            if hasattr(exec, "test_executions"):
                for test in exec.test_executions:
                    if test.score is not None:
                        all_scores.append(test.score)
        recent_avg_score = sum(all_scores) / len(all_scores) if all_scores else None
    else:
        recent_success_rate = 0.0
        recent_avg_score = None

    recent_summaries = []
    for exec in recent_execs:
        summary = SuiteExecutionSummary.model_validate(exec)
        summary.agent_name = exec.agent.name if exec.agent else None
        recent_summaries.append(summary)

    return DashboardSummary(
        total_agents=agent_count,
        total_suites=suite_count,
        total_executions=exec_count,
        recent_success_rate=recent_success_rate,
        recent_avg_score=recent_avg_score,
        recent_executions=recent_summaries,
    )
