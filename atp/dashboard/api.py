"""FastAPI routes for ATP Dashboard API."""

from datetime import datetime, timedelta
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
    AgentColumn,
    AgentComparisonMetrics,
    AgentComparisonResponse,
    AgentCreate,
    AgentExecutionDetail,
    AgentResponse,
    AgentTimeline,
    AgentUpdate,
    DashboardSummary,
    EvaluationResultResponse,
    EventSummary,
    LeaderboardMatrixResponse,
    MultiTimelineResponse,
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
    TestRow,
    TestScore,
    TestTrend,
    TimelineEvent,
    TimelineEventsResponse,
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


# ==================== Leaderboard Routes ====================


def _calculate_difficulty(avg_score: float | None) -> str:
    """Calculate difficulty rating based on average score.

    Args:
        avg_score: Average score across all agents (0-100 scale).

    Returns:
        Difficulty string: easy, medium, hard, very_hard, or unknown.
    """
    if avg_score is None:
        return "unknown"
    if avg_score >= 80:
        return "easy"
    if avg_score >= 60:
        return "medium"
    if avg_score >= 40:
        return "hard"
    return "very_hard"


def _detect_pattern(scores: list[float | None], pass_rates: list[float]) -> str | None:
    """Detect patterns in scores across agents.

    Args:
        scores: List of scores for each agent.
        pass_rates: List of pass rates for each agent.

    Returns:
        Pattern string or None if no pattern detected.
    """
    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return None

    avg = sum(valid_scores) / len(valid_scores)
    all_pass = all(r >= 0.8 for r in pass_rates)
    all_fail = all(r <= 0.2 for r in pass_rates)

    if all_fail or avg < 40:
        return "hard_for_all"
    if all_pass and avg >= 80:
        return "easy"
    if len(valid_scores) >= 2:
        score_range = max(valid_scores) - min(valid_scores)
        if score_range >= 40:
            return "high_variance"
    return None


@router.get(
    "/leaderboard/matrix",
    response_model=LeaderboardMatrixResponse,
    tags=["leaderboard"],
)
async def get_leaderboard_matrix(
    session: SessionDep,
    user: CurrentUser,
    suite_name: str,
    agents: list[str] | None = Query(None),
    limit_executions: int = Query(default=5, le=20),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> LeaderboardMatrixResponse:
    """Get leaderboard matrix for a test suite.

    Returns a matrix of tests (rows) vs agents (columns) with scores,
    difficulty ratings, and agent rankings.

    Args:
        session: Database session.
        user: Current user (optional auth).
        suite_name: Name of the test suite.
        agents: Optional list of agent names to filter by.
        limit_executions: Max number of recent executions per agent to consider.
        limit: Max number of tests to return (pagination).
        offset: Offset for pagination.

    Returns:
        LeaderboardMatrixResponse with test rows and agent columns.
    """
    # Get list of agent names if not specified
    if agents is None:
        stmt = select(Agent.name).order_by(Agent.name)
        result = await session.execute(stmt)
        agent_names = list(result.scalars().all())
    else:
        agent_names = agents

    if not agent_names:
        return LeaderboardMatrixResponse(
            suite_name=suite_name,
            tests=[],
            agents=[],
            total_tests=0,
            total_agents=0,
            limit=limit,
            offset=offset,
        )

    # Collect data per agent
    # Structure: {test_id: {agent_name: {scores: [], successes: [], ...}}}
    test_data: dict[str, dict[str, dict]] = {}
    test_names: dict[str, str] = {}
    test_tags: dict[str, list[str]] = {}
    agent_metrics: dict[str, dict] = {
        name: {"scores": [], "successes": [], "tokens": 0, "cost": 0.0}
        for name in agent_names
    }

    for agent_name in agent_names:
        # Query recent suite executions for this agent
        stmt = (
            select(SuiteExecution)
            .join(Agent)
            .where(
                SuiteExecution.suite_name == suite_name,
                Agent.name == agent_name,
            )
            .options(
                selectinload(SuiteExecution.test_executions).selectinload(
                    TestExecution.run_results
                )
            )
            .order_by(SuiteExecution.started_at.desc())
            .limit(limit_executions)
        )
        result = await session.execute(stmt)
        executions = list(result.scalars().all())

        for exec in executions:
            for test in exec.test_executions:
                test_id = test.test_id

                # Initialize test data structure
                if test_id not in test_data:
                    test_data[test_id] = {}
                    test_names[test_id] = test.test_name
                    test_tags[test_id] = test.tags or []

                if agent_name not in test_data[test_id]:
                    test_data[test_id][agent_name] = {
                        "scores": [],
                        "successes": [],
                    }

                # Collect score and success data
                if test.score is not None:
                    test_data[test_id][agent_name]["scores"].append(test.score)
                    agent_metrics[agent_name]["scores"].append(test.score)
                test_data[test_id][agent_name]["successes"].append(
                    1.0 if test.success else 0.0
                )
                agent_metrics[agent_name]["successes"].append(
                    1.0 if test.success else 0.0
                )

                # Collect token and cost data from run results
                for run in test.run_results:
                    if run.total_tokens:
                        agent_metrics[agent_name]["tokens"] += run.total_tokens
                    if run.cost_usd:
                        agent_metrics[agent_name]["cost"] += run.cost_usd

    # Get total count of tests before pagination
    total_tests = len(test_data)

    # Apply pagination to test IDs
    sorted_test_ids = sorted(test_data.keys())
    paginated_test_ids = sorted_test_ids[offset : offset + limit]

    # Build test rows with pagination
    test_rows: list[TestRow] = []
    for test_id in paginated_test_ids:
        agent_data = test_data[test_id]
        scores_by_agent: dict[str, TestScore] = {}
        all_scores: list[float | None] = []
        all_pass_rates: list[float] = []

        for agent_name in agent_names:
            if agent_name in agent_data:
                data = agent_data[agent_name]
                scores = data["scores"]
                successes = data["successes"]

                avg_score = sum(scores) / len(scores) if scores else None
                pass_rate = sum(successes) / len(successes) if successes else 0.0

                scores_by_agent[agent_name] = TestScore(
                    score=avg_score,
                    success=pass_rate >= 0.5,
                    execution_count=len(successes),
                )
                all_scores.append(avg_score)
                all_pass_rates.append(pass_rate)
            else:
                scores_by_agent[agent_name] = TestScore(
                    score=None,
                    success=False,
                    execution_count=0,
                )
                all_scores.append(None)
                all_pass_rates.append(0.0)

        # Calculate overall average score for this test
        valid_scores = [s for s in all_scores if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        test_rows.append(
            TestRow(
                test_id=test_id,
                test_name=test_names.get(test_id, test_id),
                tags=test_tags.get(test_id, []),
                scores_by_agent=scores_by_agent,
                avg_score=avg_score,
                difficulty=_calculate_difficulty(avg_score),
                pattern=_detect_pattern(all_scores, all_pass_rates),
            )
        )

    # Build agent columns with rankings
    agent_columns: list[AgentColumn] = []
    agent_scores_for_ranking: list[tuple[str, float | None]] = []

    for agent_name in agent_names:
        metrics = agent_metrics[agent_name]
        scores = metrics["scores"]
        successes = metrics["successes"]

        avg_score = sum(scores) / len(scores) if scores else None
        pass_rate = sum(successes) / len(successes) if successes else 0.0
        total_cost = metrics["cost"] if metrics["cost"] > 0 else None

        agent_scores_for_ranking.append((agent_name, avg_score))
        agent_columns.append(
            AgentColumn(
                agent_name=agent_name,
                avg_score=avg_score,
                pass_rate=pass_rate,
                total_tokens=metrics["tokens"],
                total_cost=total_cost,
                rank=0,  # Will be set below
            )
        )

    # Calculate rankings (higher score = better rank)
    sorted_agents = sorted(
        agent_scores_for_ranking,
        key=lambda x: (x[1] is not None, x[1] or 0),
        reverse=True,
    )
    rank_map = {name: rank + 1 for rank, (name, _) in enumerate(sorted_agents)}

    for col in agent_columns:
        col.rank = rank_map[col.agent_name]

    # Sort agent columns by rank
    agent_columns.sort(key=lambda x: x.rank)

    return LeaderboardMatrixResponse(
        suite_name=suite_name,
        tests=test_rows,
        agents=agent_columns,
        total_tests=total_tests,
        total_agents=len(agent_names),
        limit=limit,
        offset=offset,
    )


# ==================== Timeline Routes ====================


def _format_timeline_event(
    event: dict, first_timestamp: datetime | None
) -> TimelineEvent:
    """Format a raw event dict into a TimelineEvent with relative timing.

    Args:
        event: Raw event dictionary from events_json.
        first_timestamp: Timestamp of the first event for relative time calculation.

    Returns:
        TimelineEvent with formatted data and relative timing.
    """
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
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            timestamp = datetime.now()
    else:
        timestamp = datetime.now()

    # Calculate relative time from first event
    relative_time_ms = 0.0
    if first_timestamp is not None:
        delta = timestamp - first_timestamp
        relative_time_ms = delta.total_seconds() * 1000

    # Extract duration from payload if available
    duration_ms: float | None = None
    if "duration_ms" in payload:
        duration_ms = float(payload["duration_ms"])

    return TimelineEvent(
        sequence=event.get("sequence", 0),
        timestamp=timestamp,
        event_type=event_type,
        summary=summary,
        data=payload,
        relative_time_ms=relative_time_ms,
        duration_ms=duration_ms,
    )


@router.get(
    "/timeline/events",
    response_model=TimelineEventsResponse,
    tags=["timeline"],
)
async def get_timeline_events(
    session: SessionDep,
    user: CurrentUser,
    suite_name: str,
    test_id: str,
    agent_name: str,
    event_types: list[str] | None = Query(None),
    limit: int = Query(default=1000, le=1000),
    offset: int = Query(default=0, ge=0),
) -> TimelineEventsResponse:
    """Get timeline events for a specific test execution.

    Returns events from the latest test execution for the specified agent,
    with relative timing information for timeline visualization.

    Args:
        session: Database session.
        user: Current user (optional auth).
        suite_name: Name of the test suite.
        test_id: ID of the specific test.
        agent_name: Name of the agent.
        event_types: Optional list of event types to filter by
            (tool_call, llm_request, reasoning, error, progress).
        limit: Maximum number of events to return (default 1000, max 1000).
        offset: Offset for pagination.

    Returns:
        TimelineEventsResponse with events and timing information.

    Raises:
        HTTPException: If no execution found for the specified parameters.
    """
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No execution found for test '{test_id}' in suite '{suite_name}' "
            f"by agent '{agent_name}'",
        )

    # Get the latest run result
    run_results = sorted(execution.run_results, key=lambda r: r.run_number)
    latest_run = run_results[-1] if run_results else None

    if latest_run is None or not latest_run.events_json:
        # Return empty response if no events
        return TimelineEventsResponse(
            suite_name=suite_name,
            test_id=test_id,
            test_name=execution.test_name,
            agent_name=agent_name,
            total_events=0,
            events=[],
            total_duration_ms=None,
            execution_id=execution.id,
        )

    # Get all events from the run
    raw_events = latest_run.events_json

    # Sort by sequence to ensure correct ordering
    raw_events = sorted(raw_events, key=lambda e: e.get("sequence", 0))

    # Apply event type filtering if specified
    if event_types:
        valid_types = set(event_types)
        raw_events = [e for e in raw_events if e.get("event_type") in valid_types]

    # Get total count before pagination
    total_events = len(raw_events)

    # Apply pagination
    paginated_events = raw_events[offset : offset + limit]

    # Parse first timestamp for relative time calculation
    first_timestamp: datetime | None = None
    if raw_events:
        first_ts_str = raw_events[0].get("timestamp", "")
        if first_ts_str:
            try:
                first_timestamp = datetime.fromisoformat(
                    first_ts_str.replace("Z", "+00:00")
                )
            except ValueError:
                pass

    # Format events with relative timing
    timeline_events: list[TimelineEvent] = []
    for event in paginated_events:
        timeline_events.append(_format_timeline_event(event, first_timestamp))

    # Calculate total duration from first to last event
    total_duration_ms: float | None = None
    if raw_events and len(raw_events) > 1:
        last_ts_str = raw_events[-1].get("timestamp", "")
        if first_timestamp and last_ts_str:
            try:
                last_timestamp = datetime.fromisoformat(
                    last_ts_str.replace("Z", "+00:00")
                )
                delta = last_timestamp - first_timestamp
                total_duration_ms = delta.total_seconds() * 1000
            except ValueError:
                pass

    return TimelineEventsResponse(
        suite_name=suite_name,
        test_id=test_id,
        test_name=execution.test_name,
        agent_name=agent_name,
        total_events=total_events,
        events=timeline_events,
        total_duration_ms=total_duration_ms,
        execution_id=execution.id,
    )


@router.get(
    "/timeline/compare",
    response_model=MultiTimelineResponse,
    tags=["timeline"],
)
async def get_multi_timeline(
    session: SessionDep,
    user: CurrentUser,
    suite_name: str,
    test_id: str,
    agents: list[str] = Query(..., min_length=2, max_length=3),
    event_types: list[str] | None = Query(None),
) -> MultiTimelineResponse:
    """Get aligned timelines for multiple agents on the same test.

    Returns timelines for 2-3 agents aligned by start time, enabling
    visual comparison of execution strategies and timing.

    Args:
        session: Database session.
        user: Current user (optional auth).
        suite_name: Name of the test suite.
        test_id: ID of the specific test.
        agents: List of agent names to compare (2-3 agents).
        event_types: Optional list of event types to filter by
            (tool_call, llm_request, reasoning, error, progress).

    Returns:
        MultiTimelineResponse with aligned timelines for each agent.

    Raises:
        HTTPException: If no executions found for any agent.
    """
    timelines: list[AgentTimeline] = []
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

        # Get the latest run result
        run_results = sorted(execution.run_results, key=lambda r: r.run_number)
        latest_run = run_results[-1] if run_results else None

        if latest_run is None or not latest_run.events_json:
            # No events - create an empty timeline
            timelines.append(
                AgentTimeline(
                    agent_name=agent_name,
                    test_execution_id=execution.id,
                    start_time=execution.started_at,
                    total_duration_ms=0.0,
                    events=[],
                )
            )
            continue

        # Get all events from the run
        raw_events = latest_run.events_json

        # Sort by sequence to ensure correct ordering
        raw_events = sorted(raw_events, key=lambda e: e.get("sequence", 0))

        # Apply event type filtering if specified
        if event_types:
            valid_types = set(event_types)
            raw_events = [e for e in raw_events if e.get("event_type") in valid_types]

        # Parse first timestamp for relative time calculation
        first_timestamp: datetime | None = None
        if raw_events:
            first_ts_str = raw_events[0].get("timestamp", "")
            if first_ts_str:
                try:
                    first_timestamp = datetime.fromisoformat(
                        first_ts_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

        # Format events with relative timing
        timeline_events: list[TimelineEvent] = []
        for event in raw_events:
            timeline_events.append(_format_timeline_event(event, first_timestamp))

        # Calculate total duration from first to last event
        total_duration_ms = 0.0
        if raw_events and len(raw_events) > 1:
            last_ts_str = raw_events[-1].get("timestamp", "")
            if first_timestamp and last_ts_str:
                try:
                    last_timestamp = datetime.fromisoformat(
                        last_ts_str.replace("Z", "+00:00")
                    )
                    delta = last_timestamp - first_timestamp
                    total_duration_ms = delta.total_seconds() * 1000
                except ValueError:
                    pass

        # Use first timestamp or execution start time
        start_time = first_timestamp or execution.started_at

        timelines.append(
            AgentTimeline(
                agent_name=agent_name,
                test_execution_id=execution.id,
                start_time=start_time,
                total_duration_ms=total_duration_ms,
                events=timeline_events,
            )
        )

    if not timelines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No executions found for test '{test_id}' in suite '{suite_name}' "
            f"for any of the specified agents",
        )

    return MultiTimelineResponse(
        suite_name=suite_name,
        test_id=test_id,
        test_name=test_name or test_id,
        timelines=timelines,
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
