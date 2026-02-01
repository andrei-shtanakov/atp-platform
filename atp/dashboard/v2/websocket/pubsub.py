"""Pub/Sub system for WebSocket real-time updates.

Implements a topic-based pub/sub system for broadcasting updates to
subscribed WebSocket clients.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Topic(str, Enum):
    """Available subscription topics."""

    # Suite-level topics
    SUITE_ALL = "suite:*"
    SUITE_PROGRESS = "suite:progress"
    SUITE_COMPLETED = "suite:completed"

    # Test-level topics
    TEST_ALL = "test:*"
    TEST_PROGRESS = "test:progress"
    TEST_COMPLETED = "test:completed"

    # Event streaming
    EVENTS_ALL = "events:*"
    EVENTS_TOOL_CALL = "events:tool_call"
    EVENTS_LLM_REQUEST = "events:llm_request"
    EVENTS_REASONING = "events:reasoning"
    EVENTS_ERROR = "events:error"
    EVENTS_PROGRESS = "events:progress"

    # Log streaming
    LOGS = "logs"

    # Delta updates
    DELTA = "delta"


@dataclass
class Subscription:
    """A client's subscription to a topic."""

    topic: str
    client_id: str
    filter: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def matches_filter(self, data: dict[str, Any]) -> bool:
        """Check if data matches the subscription filter.

        Args:
            data: The data to check against the filter.

        Returns:
            True if data matches all filter criteria.
        """
        if not self.filter:
            return True

        for key, value in self.filter.items():
            if key not in data:
                return False
            if isinstance(value, list):
                if data[key] not in value:
                    return False
            elif data[key] != value:
                return False

        return True


class PubSubManager:
    """Manages pub/sub subscriptions for WebSocket clients.

    This class handles:
    - Client subscriptions to topics
    - Message publishing to topics
    - Topic filtering and wildcards
    - Message queuing per client
    """

    def __init__(self, max_queue_size: int = 1000) -> None:
        """Initialize the PubSubManager.

        Args:
            max_queue_size: Maximum messages to queue per client.
        """
        # topic -> set of client_ids
        self._subscriptions: dict[str, dict[str, Subscription]] = defaultdict(dict)
        # client_id -> asyncio.Queue for messages
        self._client_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        # client_id -> list of topics
        self._client_topics: dict[str, set[str]] = defaultdict(set)
        self._max_queue_size = max_queue_size
        self._lock = asyncio.Lock()
        # Sequence counter for delta updates
        self._sequence_counter = 0

    async def subscribe(
        self,
        client_id: str,
        topic: str,
        filter_dict: dict[str, Any] | None = None,
    ) -> bool:
        """Subscribe a client to a topic.

        Args:
            client_id: Unique client identifier.
            topic: Topic to subscribe to.
            filter_dict: Optional filter for the subscription.

        Returns:
            True if subscription was successful.
        """
        async with self._lock:
            # Normalize topic
            topic = self._normalize_topic(topic)

            # Create subscription
            subscription = Subscription(
                topic=topic,
                client_id=client_id,
                filter=filter_dict or {},
            )

            # Add to subscriptions
            self._subscriptions[topic][client_id] = subscription
            self._client_topics[client_id].add(topic)

            # Ensure client has a queue
            if client_id not in self._client_queues:
                self._client_queues[client_id] = asyncio.Queue(
                    maxsize=self._max_queue_size
                )

            logger.debug(f"Client {client_id} subscribed to {topic}")
            return True

    async def unsubscribe(self, client_id: str, topic: str) -> bool:
        """Unsubscribe a client from a topic.

        Args:
            client_id: Unique client identifier.
            topic: Topic to unsubscribe from.

        Returns:
            True if unsubscription was successful.
        """
        async with self._lock:
            topic = self._normalize_topic(topic)

            if topic in self._subscriptions:
                self._subscriptions[topic].pop(client_id, None)
                if not self._subscriptions[topic]:
                    del self._subscriptions[topic]

            self._client_topics[client_id].discard(topic)

            logger.debug(f"Client {client_id} unsubscribed from {topic}")
            return True

    async def unsubscribe_all(self, client_id: str) -> None:
        """Remove all subscriptions for a client.

        Args:
            client_id: Unique client identifier.
        """
        async with self._lock:
            topics = list(self._client_topics.get(client_id, set()))
            for topic in topics:
                if topic in self._subscriptions:
                    self._subscriptions[topic].pop(client_id, None)
                    if not self._subscriptions[topic]:
                        del self._subscriptions[topic]

            self._client_topics.pop(client_id, None)
            self._client_queues.pop(client_id, None)

            logger.debug(f"Client {client_id} unsubscribed from all topics")

    async def publish(
        self,
        topic: str,
        data: dict[str, Any],
        include_sequence: bool = False,
    ) -> int:
        """Publish a message to a topic.

        Args:
            topic: Topic to publish to.
            data: Message data.
            include_sequence: Whether to include sequence number.

        Returns:
            Number of clients that received the message.
        """
        topic = self._normalize_topic(topic)
        matching_clients: set[str] = set()

        async with self._lock:
            # Find all matching subscriptions
            for sub_topic, subscribers in self._subscriptions.items():
                if self._topic_matches(sub_topic, topic):
                    for client_id, subscription in subscribers.items():
                        if subscription.matches_filter(data):
                            matching_clients.add(client_id)

            # Add sequence if requested
            if include_sequence:
                self._sequence_counter += 1
                data = {**data, "sequence": self._sequence_counter}

            # Queue message for each matching client
            message = {
                "topic": topic,
                "data": data,
                "timestamp": datetime.now().isoformat(),
            }

            for client_id in matching_clients:
                queue = self._client_queues.get(client_id)
                if queue:
                    try:
                        queue.put_nowait(message)
                    except asyncio.QueueFull:
                        # Drop oldest message if queue is full
                        try:
                            queue.get_nowait()
                            queue.put_nowait(message)
                        except asyncio.QueueEmpty:
                            pass

        return len(matching_clients)

    async def get_messages(
        self,
        client_id: str,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Get the next message for a client.

        Args:
            client_id: Unique client identifier.
            timeout: Timeout in seconds (None for blocking).

        Returns:
            Next message or None if timeout.
        """
        queue = self._client_queues.get(client_id)
        if not queue:
            return None

        try:
            if timeout is None:
                return await queue.get()
            else:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
        except TimeoutError:
            return None

    def get_client_subscriptions(self, client_id: str) -> list[str]:
        """Get all topics a client is subscribed to.

        Args:
            client_id: Unique client identifier.

        Returns:
            List of subscribed topics.
        """
        return list(self._client_topics.get(client_id, set()))

    def get_topic_subscribers(self, topic: str) -> list[str]:
        """Get all clients subscribed to a topic.

        Args:
            topic: Topic to check.

        Returns:
            List of client IDs.
        """
        topic = self._normalize_topic(topic)
        return list(self._subscriptions.get(topic, {}).keys())

    def _normalize_topic(self, topic: str) -> str:
        """Normalize a topic string.

        Args:
            topic: Topic to normalize.

        Returns:
            Normalized topic.
        """
        return topic.lower().strip()

    def _topic_matches(self, subscription_topic: str, published_topic: str) -> bool:
        """Check if a published topic matches a subscription.

        Supports wildcard subscriptions with '*'.

        Args:
            subscription_topic: The topic from the subscription.
            published_topic: The topic being published to.

        Returns:
            True if topics match.
        """
        if subscription_topic == published_topic:
            return True

        # Handle wildcards
        if subscription_topic.endswith(":*"):
            prefix = subscription_topic[:-1]  # Remove '*'
            return published_topic.startswith(prefix)

        if subscription_topic == "*":
            return True

        return False

    @property
    def active_subscriptions_count(self) -> int:
        """Return total number of active subscriptions."""
        return sum(len(subs) for subs in self._subscriptions.values())

    @property
    def connected_clients_count(self) -> int:
        """Return number of connected clients."""
        return len(self._client_queues)

    def get_current_sequence(self) -> int:
        """Return current sequence number for delta updates."""
        return self._sequence_counter
