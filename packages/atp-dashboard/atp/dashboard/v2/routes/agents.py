"""Agent management routes.

This module provides CRUD operations for managing agents
in the ATP Dashboard.

Permissions:
    - GET /agents: AGENTS_READ
    - POST /agents: deprecated, returns 410 — use POST /api/v1/agents
    - GET /agents/{id}: AGENTS_READ
    - PATCH /agents/{id}: AGENTS_WRITE
    - DELETE /agents/{id}: AGENTS_DELETE
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from atp.dashboard.models import Agent
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.schemas import AgentResponse, AgentUpdate
from atp.dashboard.v2.dependencies import (
    DBSession,
)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=list[AgentResponse])
async def list_agents(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.AGENTS_READ))],
) -> list[AgentResponse]:
    """List all agents.

    Requires AGENTS_READ permission.

    Args:
        session: Database session.

    Returns:
        List of all agents ordered by name.
    """
    stmt = select(Agent).order_by(Agent.name)
    result = await session.execute(stmt)
    agents = result.scalars().all()
    return [AgentResponse.model_validate(a) for a in agents]


@router.post(
    "",
    status_code=status.HTTP_410_GONE,
    deprecated=True,
    responses={
        status.HTTP_410_GONE: {
            "description": ("Deprecated endpoint. Use POST /api/v1/agents instead."),
            "headers": {
                "Deprecation": {
                    "description": "Indicates that this endpoint is deprecated.",
                    "schema": {"type": "string", "example": "true"},
                },
                "Sunset": {
                    "description": (
                        "HTTP-date after which clients should stop using this endpoint."
                    ),
                    "schema": {
                        "type": "string",
                        "example": "Fri, 17 Apr 2026 12:00:00 GMT",
                    },
                },
                "Link": {
                    "description": "Link to the successor endpoint (RFC 8288).",
                    "schema": {
                        "type": "string",
                        "example": '</api/v1/agents>; rel="successor-version"',
                    },
                },
            },
        }
    },
)
async def create_agent() -> None:
    """Deprecated. Use POST /api/v1/agents instead.

    Removed in LABS-54 Phase 2 because this endpoint created ownerless
    Agent rows. The replacement at /api/v1/agents resolves the owner
    from the authenticated user's JWT and enforces per-user quotas.
    """
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail=(
            "POST /api/agents is deprecated. Use POST /api/v1/agents "
            "(resolves owner from your JWT)."
        ),
        headers={
            "Deprecation": "true",
            "Sunset": "Fri, 17 Apr 2026 12:00:00 GMT",
            "Link": '</api/v1/agents>; rel="successor-version"',
        },
    )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    session: DBSession,
    agent_id: int,
    _: Annotated[None, Depends(require_permission(Permission.AGENTS_READ))],
) -> AgentResponse:
    """Get agent by ID.

    Requires AGENTS_READ permission.

    Args:
        session: Database session.
        agent_id: Agent ID.

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
    _: Annotated[None, Depends(require_permission(Permission.AGENTS_WRITE))],
) -> AgentResponse:
    """Update an agent.

    Requires AGENTS_WRITE permission.

    Args:
        session: Database session.
        agent_id: Agent ID.
        agent_data: Agent update data.

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
async def delete_agent(
    session: DBSession,
    agent_id: int,
    _: Annotated[None, Depends(require_permission(Permission.AGENTS_DELETE))],
) -> None:
    """Delete an agent.

    Requires AGENTS_DELETE permission.

    Args:
        session: Database session.
        agent_id: Agent ID.

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
