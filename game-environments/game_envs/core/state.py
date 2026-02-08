"""Game state and observation models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A message exchanged between players.

    Attributes:
        sender: Player ID of the message sender.
        content: Message text content.
        round_number: Round when the message was sent.
        timestamp: Unix timestamp of message creation.
    """

    sender: str
    content: str
    round_number: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "content": self.content,
            "round_number": self.round_number,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            sender=data["sender"],
            content=data["content"],
            round_number=data["round_number"],
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class RoundResult:
    """Result of a single round."""

    round_number: int
    actions: dict[str, Any]
    payoffs: dict[str, float]
    messages: list[Message] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "actions": dict(self.actions),
            "payoffs": dict(self.payoffs),
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoundResult:
        return cls(
            round_number=data["round_number"],
            actions=data["actions"],
            payoffs=data["payoffs"],
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
        )


@dataclass
class GameState:
    """Full game state (god's eye view)."""

    round_number: int
    player_states: dict[str, dict[str, Any]]
    public_state: dict[str, Any]
    is_terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "player_states": {k: dict(v) for k, v in self.player_states.items()},
            "public_state": dict(self.public_state),
            "is_terminal": self.is_terminal,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameState:
        return cls(
            round_number=data["round_number"],
            player_states=data["player_states"],
            public_state=data["public_state"],
            is_terminal=data.get("is_terminal", False),
        )


@dataclass
class Observation:
    """What a single player observes."""

    player_id: str
    game_state: dict[str, Any]
    available_actions: list[Any]
    history: list[RoundResult]
    round_number: int
    total_rounds: int
    messages: list[Message] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Human-readable description for LLM agents."""
        lines = [
            f"You are player '{self.player_id}'.",
            f"Round {self.round_number} of {self.total_rounds}.",
            "",
            "Current state:",
        ]
        for key, value in self.game_state.items():
            lines.append(f"  {key}: {value}")

        lines.append("")
        lines.append("Available actions:")
        for action in self.available_actions:
            lines.append(f"  - {action}")

        if self.history:
            lines.append("")
            lines.append("History:")
            for rr in self.history:
                actions_str = ", ".join(f"{k}={v}" for k, v in rr.actions.items())
                payoffs_str = ", ".join(f"{k}={v}" for k, v in rr.payoffs.items())
                lines.append(
                    f"  Round {rr.round_number}: "
                    f"actions=[{actions_str}] "
                    f"payoffs=[{payoffs_str}]"
                )

        if self.messages:
            lines.append("")
            lines.append("Messages:")
            for msg in self.messages:
                lines.append(f"  [{msg.sender}]: {msg.content}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dictionary for ATP protocol."""
        return {
            "player_id": self.player_id,
            "game_state": self.game_state,
            "available_actions": self.available_actions,
            "history": [r.to_dict() for r in self.history],
            "round_number": self.round_number,
            "total_rounds": self.total_rounds,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(
            player_id=data["player_id"],
            game_state=data["game_state"],
            available_actions=data["available_actions"],
            history=[RoundResult.from_dict(r) for r in data.get("history", [])],
            round_number=data["round_number"],
            total_rounds=data["total_rounds"],
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
        )


@dataclass
class StepResult:
    """What step() returns."""

    state: GameState
    observations: dict[str, Observation]
    payoffs: dict[str, float]
    is_terminal: bool
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "observations": {k: v.to_dict() for k, v in self.observations.items()},
            "payoffs": dict(self.payoffs),
            "is_terminal": self.is_terminal,
            "info": dict(self.info),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepResult:
        return cls(
            state=GameState.from_dict(data["state"]),
            observations={
                k: Observation.from_dict(v) for k, v in data["observations"].items()
            },
            payoffs=data["payoffs"],
            is_terminal=data["is_terminal"],
            info=data.get("info", {}),
        )
