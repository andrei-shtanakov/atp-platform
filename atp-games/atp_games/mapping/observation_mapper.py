"""Maps GameObservation to ATPRequest."""

from __future__ import annotations

import uuid

from atp.protocol.models import ATPRequest, Context, Task
from game_envs.core.state import Observation


class ObservationMapper:
    """Maps game Observation to ATP protocol ATPRequest."""

    def to_atp_request(
        self,
        observation: Observation,
        game_name: str,
        episode: int,
    ) -> ATPRequest:
        """Convert a game observation to an ATPRequest.

        Args:
            observation: The player's observation.
            game_name: Name of the game being played.
            episode: Current episode number.

        Returns:
            ATPRequest ready to send to an agent adapter.
        """
        task_prompt = self._build_task_prompt(observation, game_name)
        task_id = (
            f"game-{game_name.replace(' ', '_')}"
            f"-ep{episode}"
            f"-r{observation.round_number}"
            f"-{observation.player_id}"
            f"-{uuid.uuid4().hex[:8]}"
        )
        # Sanitize task_id for ATP validation
        task_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in task_id)

        return ATPRequest(
            task_id=task_id,
            task=Task(description=task_prompt),
            context=Context(
                environment={
                    "game": game_name,
                    "round": str(observation.round_number),
                    "total_rounds": str(observation.total_rounds),
                    "player_id": observation.player_id,
                },
            ),
            constraints={
                "response_format": {
                    "action": "required",
                    "message": "optional",
                    "reasoning": "optional",
                },
            },
            metadata={
                "game_type": "game_theoretic",
                "episode": episode,
                "game_state": observation.game_state,
                "available_actions": (observation.available_actions),
                "history": [r.to_dict() for r in observation.history],
                "messages": [m.to_dict() for m in observation.messages],
            },
        )

    def _build_task_prompt(
        self,
        obs: Observation,
        game_name: str,
    ) -> str:
        """Build LLM-friendly task description."""
        return (
            f"You are playing {game_name}. "
            f"Round {obs.round_number}/{obs.total_rounds}."
            f"\n\n{obs.to_prompt()}\n\n"
            f"Respond with a JSON object: "
            f'{{"action": <your choice>, '
            f'"message": "<optional>", '
            f'"reasoning": "<optional>"}}'
        )
