"""Agent ownership management API endpoints."""

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import require_user_level_token
from atp.dashboard.models import Agent, User
from atp.dashboard.schemas import AgentOwnerCreate, AgentOwnerResponse, AgentOwnerUpdate
from atp.dashboard.tokens import APIToken
from atp.dashboard.tournament.models import Participant, Tournament, TournamentStatus
from atp.dashboard.v2.config import get_config
from atp.dashboard.v2.dependencies import DBSession, RequiredUser
from atp.dashboard.v2.rate_limit import limiter

router = APIRouter(
    prefix="/v1/agents",
    tags=["agent-management"],
    dependencies=[Depends(require_user_level_token)],
)


async def create_agent_for_user(
    *,
    session: AsyncSession,
    user: User,
    name: str,
    version: str,
    agent_type: str,
    description: str | None = None,
    config: dict[str, Any] | None = None,
    purpose: Literal["benchmark", "tournament"] = "benchmark",
) -> Agent:
    """Create an agent owned by `user`, enforcing per-purpose quota.

    Raises HTTPException(429) on quota exceeded (per-purpose; see
    LABS-TSA PR-2) and HTTPException(409) on duplicate name/version.
    Shared by the JSON API and the cookie-authenticated UI handler.

    The quota check is best-effort (COUNT then INSERT): concurrent
    requests can race past the cap by a small margin at high request
    rates. At current scale this is not observable; see LABS-18 for
    the upgrade path if hard caps become required. The name/version
    uniqueness check is backed by the DB unique index, so duplicates
    cannot race past the guard.
    """
    cfg = get_config()
    if purpose == "tournament":
        cap = cfg.max_tournament_agents_per_user
    else:
        cap = cfg.max_benchmark_agents_per_user

    count_result = await session.execute(
        select(func.count(Agent.id)).where(
            Agent.owner_id == user.id,
            Agent.purpose == purpose,
            Agent.deleted_at.is_(None),
        )
    )
    existing_count = count_result.scalar_one()
    if existing_count >= cap:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"{purpose} agent quota exceeded ({existing_count}/{cap})",
        )

    existing = await session.execute(
        select(Agent.id).where(
            Agent.owner_id == user.id,
            Agent.name == name,
            Agent.version == version,
            Agent.deleted_at.is_(None),
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent '{name}' version '{version}' already exists",
        )

    agent = Agent(
        name=name,
        version=version,
        agent_type=agent_type,
        config=config or {},
        description=description,
        owner_id=user.id,
        purpose=purpose,
    )
    # The COUNT-based quota check and the owner-scoped duplicate check
    # above filter on ``deleted_at IS NULL``, but the DB's unique
    # constraints don't. A soft-deleted row with the same name+version
    # (or a legacy pre-ownership schema that still carries a
    # ``(tenant_id, name)`` unique) manifests here as IntegrityError.
    # Wrap session.add + flush in a SAVEPOINT so failure rolls back only
    # the failed INSERT — the caller's outer transaction (and any ORM
    # instances loaded through it, e.g. the ``user`` row referenced by
    # the UI error-template render) stays live.
    try:
        async with session.begin_nested():
            session.add(agent)
            await session.flush()
    except IntegrityError as exc:
        # The detail deliberately does not single out ``version``: on the
        # current schema (tenant_id, owner_id, name, version) the
        # conflict may be on that 4-tuple, but on the legacy prod schema
        # (tenant_id, name) any new version of an existing name also
        # trips this path. Keep the copy accurate under both by
        # referring to the name only.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Agent name '{name}' is already in use "
                "(possibly by a soft-deleted row; pick a different name)"
            ),
        ) from exc
    await session.refresh(agent)
    return agent


@router.post("", response_model=AgentOwnerResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_agent(
    request: Request,
    session: DBSession,
    user: RequiredUser,
    body: AgentOwnerCreate,
) -> AgentOwnerResponse:
    """Create a new agent owned by the current user."""
    agent = await create_agent_for_user(
        session=session,
        user=user,
        name=body.name,
        version=body.version,
        agent_type=body.agent_type,
        description=body.description,
        config=body.config,
        purpose=body.purpose,
    )
    return AgentOwnerResponse.model_validate(agent)


@router.get("", response_model=list[AgentOwnerResponse])
async def list_my_agents(
    session: DBSession,
    user: RequiredUser,
    purpose: Literal["benchmark", "tournament"] | None = None,
) -> list[AgentOwnerResponse]:
    """List agents owned by the current user.

    If ``purpose`` is provided, only agents with that purpose are
    returned (LABS-TSA PR-2).
    """
    stmt = (
        select(Agent)
        .where(Agent.owner_id == user.id, Agent.deleted_at.is_(None))
        .order_by(Agent.created_at.desc())
    )
    if purpose is not None:
        stmt = stmt.where(Agent.purpose == purpose)
    result = await session.execute(stmt)
    return [AgentOwnerResponse.model_validate(a) for a in result.scalars().all()]


@router.get("/{agent_id}", response_model=AgentOwnerResponse)
async def get_agent(
    session: DBSession,
    user: RequiredUser,
    agent_id: int,
) -> AgentOwnerResponse:
    """Get agent details (owner or admin only)."""
    agent = await session.get(Agent, agent_id)
    if agent is None or agent.deleted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )
    if agent.owner_id != user.id and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="You don't own this agent"
        )
    return AgentOwnerResponse.model_validate(agent)


@router.patch("/{agent_id}", response_model=AgentOwnerResponse)
async def update_agent(
    session: DBSession,
    user: RequiredUser,
    agent_id: int,
    body: AgentOwnerUpdate,
) -> AgentOwnerResponse:
    """Update an agent (owner or admin only)."""
    agent = await session.get(Agent, agent_id)
    if agent is None or agent.deleted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )
    if agent.owner_id != user.id and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="You don't own this agent"
        )

    if body.version is not None:
        existing = await session.execute(
            select(Agent.id).where(
                Agent.owner_id == agent.owner_id,
                Agent.name == agent.name,
                Agent.version == body.version,
                Agent.id != agent.id,
                Agent.deleted_at.is_(None),
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent '{agent.name}' version '{body.version}' already exists",
            )
        agent.version = body.version

    if body.config is not None:
        agent.config = body.config
    if body.description is not None:
        agent.description = body.description

    await session.flush()
    await session.refresh(agent)
    return AgentOwnerResponse.model_validate(agent)


@router.delete("/{agent_id}", response_model=AgentOwnerResponse)
async def delete_agent(
    session: DBSession,
    user: RequiredUser,
    agent_id: int,
) -> AgentOwnerResponse:
    """Soft-delete an agent and revoke all its tokens."""
    agent = await session.get(Agent, agent_id)
    if agent is None or agent.deleted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )
    if agent.owner_id != user.id and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="You don't own this agent"
        )

    # Block if agent is in an active tournament
    active_participation = await session.execute(
        select(Participant.id)
        .join(Tournament, Participant.tournament_id == Tournament.id)
        .where(
            Participant.agent_id == agent_id,
            Participant.released_at.is_(None),
            Tournament.status.in_([TournamentStatus.PENDING, TournamentStatus.ACTIVE]),
        )
    )
    if active_participation.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Agent is in an active tournament",
        )

    agent.deleted_at = datetime.now()

    # Revoke all agent tokens
    await session.execute(
        update(APIToken)
        .where(APIToken.agent_id == agent_id, APIToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now())
    )

    await session.flush()
    await session.refresh(agent)
    return AgentOwnerResponse.model_validate(agent)
