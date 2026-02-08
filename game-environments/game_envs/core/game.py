"""Game abstract base class and configuration."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from game_envs.core.action import ActionSpace
from game_envs.core.communication import (
    CommunicationChannel,
    CommunicationMode,
)
from game_envs.core.history import GameHistory
from game_envs.core.state import Message, Observation, StepResult


class GameType(StrEnum):
    """Classification of game type."""

    NORMAL_FORM = "normal_form"
    EXTENSIVE_FORM = "extensive_form"
    REPEATED = "repeated"
    STOCHASTIC = "stochastic"


class MoveOrder(StrEnum):
    """How players submit actions."""

    SIMULTANEOUS = "simultaneous"
    SEQUENTIAL = "sequential"


@dataclass(frozen=True)
class GameConfig:
    """Immutable game configuration.

    Attributes:
        num_players: Number of players in the game.
        num_rounds: Number of rounds (1 = one-shot).
        discount_factor: Discount for future payoffs in
            repeated games.
        noise: Probability of action flip (trembling hand).
        communication: Whether pre-action messaging is
            enabled. Shorthand for setting communication_mode
            to PRE_ACTION.
        communication_mode: Detailed communication mode.
            Overrides 'communication' if set explicitly.
        seed: Random seed for reproducibility.
    """

    num_players: int = 2
    num_rounds: int = 1
    discount_factor: float = 1.0
    noise: float = 0.0
    communication: bool = False
    communication_mode: str = CommunicationMode.NO_COMMUNICATION
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_players < 1:
            raise ValueError(f"num_players must be >= 1, got {self.num_players}")
        if self.num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {self.num_rounds}")
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError(
                f"discount_factor must be in [0, 1], got {self.discount_factor}"
            )
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError(f"noise must be in [0, 1], got {self.noise}")
        # Sync communication flag with mode
        if (
            self.communication
            and self.communication_mode == CommunicationMode.NO_COMMUNICATION
        ):
            object.__setattr__(
                self,
                "communication_mode",
                CommunicationMode.PRE_ACTION,
            )


class Game(ABC):
    """Abstract base for all games.

    A Game defines the rules, action spaces, state transitions,
    and payoff computation. It does NOT define evaluation
    criteria — that's the evaluator's job.

    Subclasses must implement:
        - name: Human-readable game name
        - game_type: Classification of the game
        - move_order: Simultaneous or sequential
        - player_ids: List of player identifiers
        - action_space: Per-player action space
        - reset: Initialize/reset the game
        - step: Process a round of actions
        - get_payoffs: Return cumulative payoffs
        - is_terminal: Whether the game has ended
    """

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self._rng = random.Random(self.config.seed)
        self._history = GameHistory()
        self._current_round = 0
        self._channel: CommunicationChannel | None = None
        # Initialize communication channel if configured
        mode = CommunicationMode(self.config.communication_mode)
        if mode != CommunicationMode.NO_COMMUNICATION:
            self._channel = CommunicationChannel(
                mode=mode,
            )

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable game name."""
        ...

    @property
    @abstractmethod
    def game_type(self) -> GameType:
        """Classification of the game type."""
        ...

    @property
    @abstractmethod
    def move_order(self) -> MoveOrder:
        """How players submit actions."""
        ...

    @property
    @abstractmethod
    def player_ids(self) -> list[str]:
        """Ordered list of player identifiers."""
        ...

    @abstractmethod
    def action_space(self, player_id: str) -> ActionSpace:
        """Get the action space for a player."""
        ...

    @abstractmethod
    def reset(self) -> StepResult:
        """Initialize or reset the game state.

        Returns:
            Initial StepResult with starting observations.
        """
        ...

    @abstractmethod
    def step(self, actions: dict[str, Any]) -> StepResult:
        """Process one round of actions.

        Args:
            actions: Mapping of player_id to action.

        Returns:
            StepResult with new state, observations,
            payoffs, and terminal flag.
        """
        ...

    @abstractmethod
    def get_payoffs(self) -> dict[str, float]:
        """Get cumulative payoffs for all players."""
        ...

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """Whether the game has ended."""
        ...

    def observe(self, player_id: str) -> Observation:
        """Get the observation for a player.

        Default implementation provides full observability.
        Override for games with partial observability.
        """
        messages = self._get_pending_messages(player_id)
        return Observation(
            player_id=player_id,
            game_state={},
            available_actions=self.action_space(player_id).to_list(),
            history=self._history.for_player(player_id),
            round_number=self._current_round,
            total_rounds=self.config.num_rounds,
            messages=messages,
        )

    @property
    def channel(self) -> CommunicationChannel | None:
        """Access the communication channel, if any."""
        return self._channel

    def send_message(
        self,
        sender: str,
        content: str,
    ) -> Message:
        """Send a message through the communication channel.

        Args:
            sender: Player ID of the sender.
            content: Message text.

        Returns:
            The created Message.

        Raises:
            RuntimeError: If no communication channel
                is configured or communication is not
                allowed in the current phase.
        """
        if self._channel is None:
            raise RuntimeError(
                "No communication channel configured. "
                "Set communication_mode in GameConfig."
            )
        return self._channel.send_message(
            sender=sender,
            content=content,
            round_number=self._current_round,
        )

    def _get_pending_messages(
        self,
        player_id: str,
    ) -> list[Message]:
        """Get pending messages for a player.

        Returns messages from the current round, excluding
        messages the player sent themselves.
        """
        if self._channel is None:
            return []
        return self._channel.get_messages(
            round_number=self._current_round,
            receiver=player_id,
        )

    @property
    def history(self) -> GameHistory:
        """Access the game history."""
        return self._history

    @property
    def current_round(self) -> int:
        """Current round number."""
        return self._current_round
