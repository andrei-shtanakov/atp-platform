"""Adapter wrapping built-in strategies as ATP agents."""

from __future__ import annotations

from collections.abc import AsyncIterator

from atp.adapters.base import AdapterConfig, AgentAdapter
from atp.protocol.models import (
    ArtifactStructured,
    ATPEvent,
    ATPRequest,
    ATPResponse,
    ResponseStatus,
)
from game_envs.core.state import Observation, RoundResult
from game_envs.core.strategy import Strategy


class BuiltinAdapter(AgentAdapter):
    """Wraps a game-environments Strategy as an ATP adapter.

    Allows built-in strategies (TitForTat, AlwaysDefect, etc.)
    to participate in ATP game loops alongside LLM agents.
    """

    def __init__(
        self,
        strategy: Strategy,
        config: AdapterConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._strategy = strategy

    @property
    def adapter_type(self) -> str:
        return "builtin"

    @property
    def strategy(self) -> Strategy:
        """Access the underlying strategy."""
        return self._strategy

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """Execute the strategy for a game request.

        Reconstructs the Observation from request metadata,
        calls the strategy, and returns the action as an
        ATPResponse with a structured artifact.
        """
        observation = self._extract_observation(request)
        action = self._strategy.choose_action(observation)

        return ATPResponse(
            task_id=request.task_id,
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={
                        "action": action,
                        "reasoning": (f"Strategy: {self._strategy.name}"),
                    },
                ),
            ],
        )

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """Stream is not needed for builtin strategies."""
        response = await self.execute(request)
        yield response

    def _extract_observation(self, request: ATPRequest) -> Observation:
        """Reconstruct Observation from ATPRequest metadata.

        The ObservationMapper stores game state and history
        in the request metadata. This method reverses that
        mapping.
        """
        metadata = request.metadata or {}
        env = {}
        if request.context and request.context.environment:
            env = request.context.environment

        history_data = metadata.get("history", [])
        history = [RoundResult.from_dict(r) for r in history_data]

        return Observation(
            player_id=env.get("player_id", "unknown"),
            game_state=metadata.get("game_state", {}),
            available_actions=metadata.get("available_actions", []),
            history=history,
            round_number=int(env.get("round", 0)),
            total_rounds=int(env.get("total_rounds", 1)),
        )

    def reset(self) -> None:
        """Reset the strategy for a new episode."""
        self._strategy.reset()
