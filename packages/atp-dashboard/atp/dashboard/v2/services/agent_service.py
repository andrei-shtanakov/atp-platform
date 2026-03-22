"""Agent service for managing agent data.

This module provides the AgentService class that encapsulates all
business logic related to agent CRUD operations.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import Agent
from atp.dashboard.schemas import AgentCreate, AgentResponse, AgentUpdate


class AgentService:
    """Service for agent management operations.

    This service encapsulates all business logic related to
    creating, reading, updating, and deleting agents.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the agent service.

        Args:
            session: Database session for queries.
        """
        self._session = session

    async def list_agents(self) -> list[AgentResponse]:
        """List all agents.

        Returns:
            List of all agents ordered by name.
        """
        stmt = select(Agent).order_by(Agent.name)
        result = await self._session.execute(stmt)
        agents = result.scalars().all()
        return [AgentResponse.model_validate(a) for a in agents]

    async def get_agent(self, agent_id: int) -> AgentResponse | None:
        """Get agent by ID.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent response or None if not found.
        """
        agent = await self._session.get(Agent, agent_id)
        if agent is None:
            return None
        return AgentResponse.model_validate(agent)

    async def get_agent_by_name(self, name: str) -> AgentResponse | None:
        """Get agent by name.

        Args:
            name: Agent name.

        Returns:
            Agent response or None if not found.
        """
        stmt = select(Agent).where(Agent.name == name)
        result = await self._session.execute(stmt)
        agent = result.scalar_one_or_none()
        if agent is None:
            return None
        return AgentResponse.model_validate(agent)

    async def create_agent(self, agent_data: AgentCreate) -> AgentResponse | None:
        """Create a new agent.

        Args:
            agent_data: Agent creation data.

        Returns:
            Created agent response, or None if an agent with the
            same name already exists.
        """
        # Check for existing agent
        stmt = select(Agent).where(Agent.name == agent_data.name)
        result = await self._session.execute(stmt)
        if result.scalar_one_or_none():
            return None

        agent = Agent(
            name=agent_data.name,
            agent_type=agent_data.agent_type,
            config=agent_data.config,
            description=agent_data.description,
        )
        self._session.add(agent)
        await self._session.flush()
        await self._session.refresh(agent)
        return AgentResponse.model_validate(agent)

    async def update_agent(
        self, agent_id: int, agent_data: AgentUpdate
    ) -> AgentResponse | None:
        """Update an agent.

        Args:
            agent_id: Agent ID.
            agent_data: Agent update data.

        Returns:
            Updated agent response, or None if not found.
        """
        agent = await self._session.get(Agent, agent_id)
        if agent is None:
            return None

        if agent_data.agent_type is not None:
            agent.agent_type = agent_data.agent_type
        if agent_data.config is not None:
            agent.config = agent_data.config
        if agent_data.description is not None:
            agent.description = agent_data.description

        await self._session.flush()
        await self._session.refresh(agent)
        return AgentResponse.model_validate(agent)

    async def delete_agent(self, agent_id: int) -> bool:
        """Delete an agent.

        Args:
            agent_id: Agent ID.

        Returns:
            True if deleted, False if not found.
        """
        agent = await self._session.get(Agent, agent_id)
        if agent is None:
            return False
        await self._session.delete(agent)
        await self._session.flush()
        return True

    async def agent_exists(self, name: str) -> bool:
        """Check if an agent with the given name exists.

        Args:
            name: Agent name to check.

        Returns:
            True if agent exists, False otherwise.
        """
        stmt = select(Agent.id).where(Agent.name == name)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None
