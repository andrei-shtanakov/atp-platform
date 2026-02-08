"""Action validation with retry logic for LLM agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from game_envs.core.action import ActionSpace

from atp_games.mapping.action_mapper import GameAction

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of action validation.

    Attributes:
        action: The final (possibly corrected) action.
        valid: Whether the original action was valid.
        attempts: Number of attempts before success.
        used_default: Whether the default action was used.
        errors: Error messages from failed attempts.
    """

    action: GameAction
    valid: bool
    attempts: int
    used_default: bool = False
    errors: list[str] = field(default_factory=list)


class ActionValidator:
    """Validates game actions and manages retry logic.

    For LLM agents, invalid actions trigger a retry with
    an error message. After max_retries failures, a random
    default action is used.
    """

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    def validate(
        self,
        action: GameAction,
        action_space: ActionSpace,
    ) -> ValidationResult:
        """Check whether an action is valid.

        Args:
            action: The game action to validate.
            action_space: The valid action space.

        Returns:
            ValidationResult indicating validity.
        """
        if action_space.contains(action.action):
            return ValidationResult(
                action=action,
                valid=True,
                attempts=1,
            )

        error = (
            f"Invalid action '{action.action}'. Valid: {action_space.to_description()}"
        )
        return ValidationResult(
            action=action,
            valid=False,
            attempts=1,
            errors=[error],
        )

    def get_default_action(
        self,
        action_space: ActionSpace,
        rng: Any = None,
    ) -> GameAction:
        """Get a random default action from the space.

        Args:
            action_space: The action space to sample from.
            rng: Optional random.Random instance.

        Returns:
            GameAction with a random valid action.
        """
        default = action_space.sample(rng)
        logger.warning("Using default action: %s", default)
        return GameAction(
            action=default,
            reasoning="Default action (validation failed)",
        )

    def build_retry_prompt(
        self,
        error: str,
        action_space: ActionSpace,
    ) -> str:
        """Build a retry prompt with error context.

        Args:
            error: The validation error message.
            action_space: The valid action space.

        Returns:
            Error context string for retry.
        """
        return (
            f"Your previous action was invalid: {error}\n"
            f"Valid actions: "
            f"{action_space.to_description()}\n"
            f"Please try again with a valid action."
        )
