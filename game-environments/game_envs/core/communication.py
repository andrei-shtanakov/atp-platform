"""Communication channel and information set models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from game_envs.core.state import Message, RoundResult


class CommunicationMode(StrEnum):
    """When players may exchange messages."""

    NO_COMMUNICATION = "no_communication"
    PRE_ACTION = "pre_action"
    POST_ACTION = "post_action"
    FREE = "free"


class CommunicationChannel:
    """Manages message exchange between players.

    Enforces the communication mode rules:
        - NO_COMMUNICATION: no messages allowed.
        - PRE_ACTION: messages allowed before actions each round.
        - POST_ACTION: messages allowed after actions each round.
        - FREE: messages allowed at any time.

    Attributes:
        mode: The communication mode for this channel.
    """

    def __init__(
        self,
        mode: CommunicationMode = CommunicationMode.NO_COMMUNICATION,
        player_ids: list[str] | None = None,
    ) -> None:
        self.mode = mode
        self._player_ids = list(player_ids) if player_ids else []
        self._round_messages: dict[int, list[Message]] = {}
        self._phase: str = "idle"

    @property
    def phase(self) -> str:
        """Current phase: 'idle', 'pre_action', 'post_action'."""
        return self._phase

    def begin_round(self, round_number: int) -> None:
        """Signal the start of a new round.

        Transitions to pre_action phase if mode allows
        pre-action communication.
        """
        if round_number not in self._round_messages:
            self._round_messages[round_number] = []
        if self.mode == CommunicationMode.PRE_ACTION:
            self._phase = "pre_action"
        elif self.mode == CommunicationMode.FREE:
            self._phase = "pre_action"
        else:
            self._phase = "idle"

    def end_actions(self, round_number: int) -> None:
        """Signal that actions have been submitted.

        Transitions to post_action phase if mode allows
        post-action communication.
        """
        if round_number not in self._round_messages:
            self._round_messages[round_number] = []
        if self.mode == CommunicationMode.POST_ACTION:
            self._phase = "post_action"
        elif self.mode == CommunicationMode.FREE:
            self._phase = "post_action"
        else:
            self._phase = "idle"

    def end_round(self) -> None:
        """Signal the end of a round. Resets phase."""
        self._phase = "idle"

    def can_send(self) -> bool:
        """Whether messages can currently be sent."""
        if self.mode == CommunicationMode.NO_COMMUNICATION:
            return False
        if self.mode == CommunicationMode.FREE:
            return True
        if self.mode == CommunicationMode.PRE_ACTION:
            return self._phase == "pre_action"
        if self.mode == CommunicationMode.POST_ACTION:
            return self._phase == "post_action"
        return False

    def send_message(
        self,
        sender: str,
        content: str,
        round_number: int,
    ) -> Message:
        """Send a message through the channel.

        Args:
            sender: Player ID of the sender.
            content: Message text.
            round_number: Current round number.

        Returns:
            The created Message.

        Raises:
            RuntimeError: If communication is not allowed
                in the current phase.
            ValueError: If sender is not a known player.
        """
        if not self.can_send():
            raise RuntimeError(
                f"Cannot send messages in mode "
                f"'{self.mode}' during phase '{self._phase}'"
            )
        if self._player_ids and sender not in self._player_ids:
            raise ValueError(
                f"Unknown sender '{sender}'. Valid players: {self._player_ids}"
            )
        msg = Message(
            sender=sender,
            content=content,
            round_number=round_number,
        )
        if round_number not in self._round_messages:
            self._round_messages[round_number] = []
        self._round_messages[round_number].append(msg)
        return msg

    def get_messages(
        self,
        round_number: int,
        receiver: str | None = None,
    ) -> list[Message]:
        """Get messages for a round.

        Args:
            round_number: The round to get messages for.
            receiver: If set, exclude messages sent by
                this player (they already know what they sent).

        Returns:
            List of messages for the round.
        """
        messages = self._round_messages.get(round_number, [])
        if receiver is not None:
            return [m for m in messages if m.sender != receiver]
        return list(messages)

    def get_all_messages(self) -> list[Message]:
        """Get all messages across all rounds."""
        all_msgs: list[Message] = []
        for round_num in sorted(self._round_messages):
            all_msgs.extend(self._round_messages[round_num])
        return all_msgs

    def clear(self) -> None:
        """Clear all messages and reset phase."""
        self._round_messages.clear()
        self._phase = "idle"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mode": str(self.mode),
            "player_ids": list(self._player_ids),
            "phase": self._phase,
            "round_messages": {
                str(k): [m.to_dict() for m in v]
                for k, v in self._round_messages.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommunicationChannel:
        """Deserialize from dictionary."""
        channel = cls(
            mode=CommunicationMode(data["mode"]),
            player_ids=data.get("player_ids", []),
        )
        channel._phase = data.get("phase", "idle")
        for round_str, msgs in data.get("round_messages", {}).items():
            round_num = int(round_str)
            channel._round_messages[round_num] = [Message.from_dict(m) for m in msgs]
        return channel


@dataclass(frozen=True)
class InformationSet:
    """Defines what a player can observe.

    An information set represents the subset of game state
    visible to a specific player. Used for games with partial
    observability (e.g., sealed-bid auctions where bids are
    hidden, poker where cards are private).

    Attributes:
        player_id: The player this information set is for.
        visible_players: Which players' actions/payoffs
            are visible. None means full observability.
        visible_state_keys: Which keys in game_state are
            visible. None means all keys visible.
        hidden_state_keys: Keys explicitly hidden from
            game_state. Applied after visible_state_keys.
    """

    player_id: str
    visible_players: list[str] | None = None
    visible_state_keys: list[str] | None = None
    hidden_state_keys: list[str] = field(default_factory=list)

    def filter_state(
        self,
        game_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Filter game state based on visibility rules.

        Args:
            game_state: The full game state dict.

        Returns:
            Filtered state with only visible information.
        """
        if self.visible_state_keys is not None:
            filtered = {
                k: v for k, v in game_state.items() if k in self.visible_state_keys
            }
        else:
            filtered = dict(game_state)

        for key in self.hidden_state_keys:
            filtered.pop(key, None)

        return filtered

    def filter_history(
        self,
        history: list[RoundResult],
    ) -> list[RoundResult]:
        """Filter history based on visible players.

        If visible_players is set, only actions and payoffs
        for those players are included. Messages are filtered
        to only show messages from visible players or the
        player themselves.

        Args:
            history: Full round history list.

        Returns:
            Filtered history with only visible information.
        """
        if self.visible_players is None:
            return list(history)

        filtered: list[RoundResult] = []
        for rr in history:
            actions = {k: v for k, v in rr.actions.items() if k in self.visible_players}
            payoffs = {k: v for k, v in rr.payoffs.items() if k in self.visible_players}
            messages = [
                m
                for m in rr.messages
                if m.sender in self.visible_players or m.sender == self.player_id
            ]
            filtered.append(
                RoundResult(
                    round_number=rr.round_number,
                    actions=actions,
                    payoffs=payoffs,
                    messages=messages,
                )
            )
        return filtered

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "player_id": self.player_id,
            "visible_players": self.visible_players,
            "visible_state_keys": self.visible_state_keys,
            "hidden_state_keys": list(self.hidden_state_keys),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InformationSet:
        """Deserialize from dictionary."""
        return cls(
            player_id=data["player_id"],
            visible_players=data.get("visible_players"),
            visible_state_keys=data.get("visible_state_keys"),
            hidden_state_keys=data.get("hidden_state_keys", []),
        )
