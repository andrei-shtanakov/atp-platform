"""Agent management routes.

This module provides CRUD operations for managing agents
in the ATP Dashboard.
"""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from atp.dashboard.models import Agent
from atp.dashboard.schemas import AgentCreate, AgentResponse, AgentUpdate
from atp.dashboard.v2.dependencies import (
    AdminUser,
    CurrentUser,
    DBSession,
    RequiredUser,
)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=list[AgentResponse])
async def list_agents(session: DBSession, user: CurrentUser) -> list[AgentResponse]:
    """List all agents.

    Args:
        session: Database session.
        user: Current user (optional auth).

    Returns:
        List of all agents ordered by name.
    """
    stmt = select(Agent).order_by(Agent.name)
    result = await session.execute(stmt)
    agents = result.scalars().all()
    return [AgentResponse.model_validate(a) for a in agents]


@router.post(
    "",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_agent(
    session: DBSession, agent_data: AgentCreate, user: RequiredUser
) -> AgentResponse:
    """Create a new agent.

    Requires authentication.

    Args:
        session: Database session.
        agent_data: Agent creation data.
        user: Authenticated user.

    Returns:
        The created agent.

    Raises:
        HTTPException: If an agent with the same name already exists.
    """
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


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    session: DBSession, agent_id: int, user: CurrentUser
) -> AgentResponse:
    """Get agent by ID.

    Args:
        session: Database session.
        agent_id: Agent ID.
        user: Current user (optional auth).

    Returns:
        The requested agent.

    Raises:
        HTTPException: If agent not found.
    """
    agent = await session.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )
    return AgentResponse.model_validate(agent)


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    session: DBSession,
    agent_id: int,
    agent_data: AgentUpdate,
    user: RequiredUser,
) -> AgentResponse:
    """Update an agent.

    Requires authentication.

    Args:
        session: Database session.
        agent_id: Agent ID.
        agent_data: Agent update data.
        user: Authenticated user.

    Returns:
        The updated agent.

    Raises:
        HTTPException: If agent not found.
    """
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


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(session: DBSession, agent_id: int, user: AdminUser) -> None:
    """Delete an agent (admin only).

    Requires admin authentication.

    Args:
        session: Database session.
        agent_id: Agent ID.
        user: Admin user.

    Raises:
        HTTPException: If agent not found.
    """
    agent = await session.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )
    await session.delete(agent)
    await session.commit()
