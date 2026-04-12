"""Agent ownership management API endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func, select, update

from atp.dashboard.auth import require_user_level_token
from atp.dashboard.models import Agent
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


@router.post("", response_model=AgentOwnerResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_agent(
    request: Request,
    session: DBSession,
    user: RequiredUser,
    body: AgentOwnerCreate,
) -> AgentOwnerResponse:
    """Create a new agent owned by the current user."""
    config = get_config()

    # Check agent limit
    count_result = await session.execute(
        select(func.count(Agent.id)).where(
            Agent.owner_id == user.id,
            Agent.deleted_at.is_(None),
        )
    )
    if count_result.scalar_one() >= config.max_agents_per_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent limit reached (max {config.max_agents_per_user})",
        )

    # Check uniqueness
    existing = await session.execute(
        select(Agent.id).where(
            Agent.owner_id == user.id,
            Agent.name == body.name,
            Agent.version == body.version,
            Agent.deleted_at.is_(None),
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent '{body.name}' version '{body.version}' already exists",
        )

    agent = Agent(
        name=body.name,
        version=body.version,
        agent_type=body.agent_type,
        config=body.config,
        description=body.description,
        owner_id=user.id,
    )
    session.add(agent)
    await session.flush()
    await session.refresh(agent)
    return AgentOwnerResponse.model_validate(agent)


@router.get("", response_model=list[AgentOwnerResponse])
async def list_my_agents(
    session: DBSession,
    user: RequiredUser,
) -> list[AgentOwnerResponse]:
    """List agents owned by the current user."""
    result = await session.execute(
        select(Agent)
        .where(Agent.owner_id == user.id, Agent.deleted_at.is_(None))
        .order_by(Agent.created_at.desc())
    )
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
