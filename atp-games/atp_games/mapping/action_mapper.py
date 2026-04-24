"""Maps ATPResponse to game actions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from atp.protocol.models import (
    ArtifactStructured,
    ATPResponse,
    ResponseStatus,
)

logger = logging.getLogger(__name__)


_INTENT_MAX_CHARS = 500


@dataclass
class GameAction:
    """Action extracted from an ATP response.

    Attributes:
        action: The game action value.
        message: Optional message from the agent.
        reasoning: Optional reasoning explanation.
        intent: Optional free-text Tier-1 self-report from the agent
            (max 500 chars after stripping; None when absent or invalid).
        raw_response: The original response data.
    """

    action: Any
    message: str | None = None
    reasoning: str | None = None
    intent: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


def _sanitize_intent(value: Any) -> str | None:
    """Return a cleaned intent string or None.

    Rules (applied in order):
      * non-string types return None (with a debug log — never a raise)
      * strip surrounding whitespace
      * empty-after-strip returns None
      * longer than 500 chars is truncated to 500 (not rejected)
    """
    if value is None:
        return None
    if not isinstance(value, str):
        logger.warning(
            "intent field has unexpected type %s; ignoring",
            type(value).__name__,
        )
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if len(stripped) > _INTENT_MAX_CHARS:
        return stripped[:_INTENT_MAX_CHARS]
    return stripped


class ActionMapper:
    """Maps ATPResponse to GameAction."""

    def from_atp_response(
        self,
        response: ATPResponse,
    ) -> GameAction:
        """Extract a game action from an ATP response.

        Parses the response artifacts and/or task description
        to find the action, message, and reasoning.

        Args:
            response: ATPResponse from an agent adapter.

        Returns:
            GameAction with the extracted action.

        Raises:
            ValueError: If no valid action found.
        """
        if response.status == ResponseStatus.FAILED:
            raise ValueError(f"Agent returned error: {response.error}")

        # Try structured artifacts first
        for artifact in response.artifacts:
            if isinstance(artifact, ArtifactStructured):
                data = artifact.data
                if "action" in data:
                    return GameAction(
                        action=data["action"],
                        message=data.get("message"),
                        reasoning=data.get("reasoning"),
                        intent=_sanitize_intent(data.get("intent")),
                        raw_response=data,
                    )

        # Try parsing artifact content as JSON
        for artifact in response.artifacts:
            if hasattr(artifact, "content") and artifact.content:
                parsed = self._try_parse_json(artifact.content)
                if parsed and "action" in parsed:
                    return GameAction(
                        action=parsed["action"],
                        message=parsed.get("message"),
                        reasoning=parsed.get("reasoning"),
                        intent=_sanitize_intent(parsed.get("intent")),
                        raw_response=parsed,
                    )

        raise ValueError(
            "No valid action found in ATP response. "
            "Expected structured artifact with 'action' key."
        )

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        """Try to parse text as JSON."""
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON in text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                result = json.loads(text[start : end + 1])
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, TypeError):
                pass

        return None
