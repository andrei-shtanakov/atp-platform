"""WebSocket connection management for ATP Dashboard.

Handles WebSocket connections, client lifecycle, and message routing.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import WebSocket

from atp.dashboard.v2.websocket.models import (
    WSClientInfo,
    WSMessage,
    WSMessageType,
)
from atp.dashboard.v2.websocket.pubsub import PubSubManager, Topic

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates.

    This class handles:
    - WebSocket connection lifecycle
    - Client registration/deregistration
    - Message sending to individual/all clients
    - Integration with PubSubManager for topic subscriptions
    - Heartbeat/ping-pong for connection health
    """

    def __init__(
        self,
        pubsub: PubSubManager | None = None,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0,
    ) -> None:
        """Initialize the ConnectionManager.

        Args:
            pubsub: PubSubManager instance for pub/sub functionality.
            ping_interval: Interval between ping messages in seconds.
            ping_timeout: Timeout for pong response in seconds.
        """
        self._connections: dict[str, WebSocket] = {}
        self._client_info: dict[str, WSClientInfo] = {}
        self._pubsub = pubsub or PubSubManager()
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._background_tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    @property
    def pubsub(self) -> PubSubManager:
        """Return the PubSubManager instance."""
        return self._pubsub

    async def connect(
        self,
        websocket: WebSocket,
        client_id: str | None = None,
        user_agent: str | None = None,
    ) -> str:
        """Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection.
            client_id: Optional client ID (generated if not provided).
            user_agent: Optional user agent string.

        Returns:
            The client ID.
        """
        await websocket.accept()

        if not client_id:
            client_id = str(uuid.uuid4())

        async with self._lock:
            self._connections[client_id] = websocket
            self._client_info[client_id] = WSClientInfo(
                client_id=client_id,
                user_agent=user_agent,
            )

        # Start message sender task for this client
        task = asyncio.create_task(self._message_sender(client_id))
        self._background_tasks[client_id] = task

        # Send connected message
        await self._send_message(
            client_id,
            WSMessage(
                type=WSMessageType.CONNECTED,
                payload={"client_id": client_id},
            ),
        )

        logger.info(f"WebSocket client connected: {client_id}")
        return client_id

    async def disconnect(self, client_id: str) -> None:
        """Disconnect a WebSocket client.

        Args:
            client_id: The client ID to disconnect.
        """
        async with self._lock:
            # Cancel background task
            task = self._background_tasks.pop(client_id, None)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Remove subscriptions
            await self._pubsub.unsubscribe_all(client_id)

            # Remove connection
            self._connections.pop(client_id, None)
            self._client_info.pop(client_id, None)

        logger.info(f"WebSocket client disconnected: {client_id}")

    async def handle_message(
        self,
        client_id: str,
        message: dict[str, Any],
    ) -> None:
        """Handle an incoming WebSocket message.

        Args:
            client_id: The client ID.
            message: The received message.
        """
        # Update last activity
        if client_id in self._client_info:
            self._client_info[client_id].last_activity = datetime.now()

        msg_type = message.get("type", "").lower()
        payload = message.get("payload", {})

        if msg_type == WSMessageType.PING.value:
            await self._handle_ping(client_id)

        elif msg_type == WSMessageType.SUBSCRIBE.value:
            await self._handle_subscribe(client_id, payload)

        elif msg_type == WSMessageType.UNSUBSCRIBE.value:
            await self._handle_unsubscribe(client_id, payload)

        else:
            await self._send_error(client_id, f"Unknown message type: {msg_type}")

    async def _handle_ping(self, client_id: str) -> None:
        """Handle ping message.

        Args:
            client_id: The client ID.
        """
        await self._send_message(
            client_id,
            WSMessage(type=WSMessageType.PONG, payload={}),
        )

    async def _handle_subscribe(
        self,
        client_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Handle subscribe request.

        Args:
            client_id: The client ID.
            payload: The subscription payload.
        """
        topic = payload.get("topic")
        if not topic:
            await self._send_error(client_id, "Missing topic in subscribe request")
            return

        filter_dict = payload.get("filter", {})

        await self._pubsub.subscribe(client_id, topic, filter_dict)

        # Update client info
        if client_id in self._client_info:
            if topic not in self._client_info[client_id].subscriptions:
                self._client_info[client_id].subscriptions.append(topic)

        await self._send_message(
            client_id,
            WSMessage(
                type=WSMessageType.SUBSCRIBED,
                payload={"topic": topic},
            ),
        )

    async def _handle_unsubscribe(
        self,
        client_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Handle unsubscribe request.

        Args:
            client_id: The client ID.
            payload: The unsubscribe payload.
        """
        topic = payload.get("topic")
        if not topic:
            await self._send_error(client_id, "Missing topic in unsubscribe request")
            return

        await self._pubsub.unsubscribe(client_id, topic)

        # Update client info
        if client_id in self._client_info:
            if topic in self._client_info[client_id].subscriptions:
                self._client_info[client_id].subscriptions.remove(topic)

        await self._send_message(
            client_id,
            WSMessage(
                type=WSMessageType.UNSUBSCRIBED,
                payload={"topic": topic},
            ),
        )

    async def _send_message(self, client_id: str, message: WSMessage) -> bool:
        """Send a message to a specific client.

        Args:
            client_id: The client ID.
            message: The message to send.

        Returns:
            True if message was sent successfully.
        """
        websocket = self._connections.get(client_id)
        if not websocket:
            return False

        try:
            await websocket.send_json(message.model_dump(mode="json"))
            return True
        except Exception as e:
            logger.warning(f"Failed to send message to {client_id}: {e}")
            return False

    async def _send_error(self, client_id: str, error: str) -> None:
        """Send an error message to a client.

        Args:
            client_id: The client ID.
            error: The error message.
        """
        await self._send_message(
            client_id,
            WSMessage(
                type=WSMessageType.ERROR,
                payload={"error": error},
            ),
        )

    async def _message_sender(self, client_id: str) -> None:
        """Background task to send queued messages to a client.

        Args:
            client_id: The client ID.
        """
        try:
            while True:
                # Get next message from pubsub queue
                message = await self._pubsub.get_messages(
                    client_id,
                    timeout=self._ping_interval,
                )

                if message:
                    # Determine message type based on topic
                    msg_type = self._get_message_type_for_topic(
                        message.get("topic", "")
                    )
                    ws_message = WSMessage(
                        type=msg_type,
                        payload=message.get("data", {}),
                        sequence=message.get("data", {}).get("sequence"),
                    )
                    await self._send_message(client_id, ws_message)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Message sender error for {client_id}: {e}")

    def _get_message_type_for_topic(self, topic: str) -> WSMessageType:
        """Map topic to message type.

        Args:
            topic: The topic string.

        Returns:
            Corresponding WSMessageType.
        """
        if topic.startswith("test:progress") or topic == Topic.TEST_PROGRESS.value:
            return WSMessageType.TEST_PROGRESS
        elif topic.startswith("test:completed") or topic == Topic.TEST_COMPLETED.value:
            return WSMessageType.TEST_COMPLETED
        elif topic.startswith("suite:progress") or topic == Topic.SUITE_PROGRESS.value:
            return WSMessageType.SUITE_PROGRESS
        elif (
            topic.startswith("suite:completed") or topic == Topic.SUITE_COMPLETED.value
        ):
            return WSMessageType.SUITE_COMPLETED
        elif topic.startswith("logs") or topic == Topic.LOGS.value:
            return WSMessageType.LOG_ENTRY
        elif topic.startswith("events") or topic.startswith(
            Topic.EVENTS_ALL.value[:-1]
        ):
            return WSMessageType.EVENT
        elif topic.startswith("delta") or topic == Topic.DELTA.value:
            return WSMessageType.DELTA_UPDATE
        else:
            return WSMessageType.EVENT

    async def broadcast(
        self,
        message: WSMessage,
        exclude: list[str] | None = None,
    ) -> int:
        """Broadcast a message to all connected clients.

        Args:
            message: The message to broadcast.
            exclude: List of client IDs to exclude.

        Returns:
            Number of clients that received the message.
        """
        exclude = exclude or []
        sent_count = 0

        for client_id in list(self._connections.keys()):
            if client_id not in exclude:
                if await self._send_message(client_id, message):
                    sent_count += 1

        return sent_count

    async def send_to_client(self, client_id: str, message: WSMessage) -> bool:
        """Send a message to a specific client.

        Args:
            client_id: The client ID.
            message: The message to send.

        Returns:
            True if message was sent successfully.
        """
        return await self._send_message(client_id, message)

    def get_client_info(self, client_id: str) -> WSClientInfo | None:
        """Get information about a connected client.

        Args:
            client_id: The client ID.

        Returns:
            Client info or None if not connected.
        """
        return self._client_info.get(client_id)

    def get_all_clients(self) -> list[WSClientInfo]:
        """Get information about all connected clients.

        Returns:
            List of client info objects.
        """
        return list(self._client_info.values())

    @property
    def connection_count(self) -> int:
        """Return number of active connections."""
        return len(self._connections)

    async def publish_test_progress(
        self,
        suite_execution_id: int,
        test_execution_id: int,
        test_id: str,
        test_name: str,
        status: str,
        progress_percent: float = 0.0,
        current_run: int = 1,
        total_runs: int = 1,
        message: str | None = None,
    ) -> int:
        """Publish a test progress update.

        Args:
            suite_execution_id: Suite execution ID.
            test_execution_id: Test execution ID.
            test_id: Test identifier.
            test_name: Test display name.
            status: Current test status.
            progress_percent: Progress percentage.
            current_run: Current run number.
            total_runs: Total runs.
            message: Optional status message.

        Returns:
            Number of clients that received the update.
        """
        data = {
            "suite_execution_id": suite_execution_id,
            "test_execution_id": test_execution_id,
            "test_id": test_id,
            "test_name": test_name,
            "status": status,
            "progress_percent": progress_percent,
            "current_run": current_run,
            "total_runs": total_runs,
            "message": message,
        }
        return await self._pubsub.publish(Topic.TEST_PROGRESS.value, data)

    async def publish_test_completed(
        self,
        suite_execution_id: int,
        test_execution_id: int,
        test_id: str,
        test_name: str,
        success: bool,
        score: float | None = None,
        duration_seconds: float | None = None,
        error: str | None = None,
    ) -> int:
        """Publish a test completion update.

        Args:
            suite_execution_id: Suite execution ID.
            test_execution_id: Test execution ID.
            test_id: Test identifier.
            test_name: Test display name.
            success: Whether the test passed.
            score: Optional test score.
            duration_seconds: Optional duration.
            error: Optional error message.

        Returns:
            Number of clients that received the update.
        """
        data = {
            "suite_execution_id": suite_execution_id,
            "test_execution_id": test_execution_id,
            "test_id": test_id,
            "test_name": test_name,
            "success": success,
            "score": score,
            "duration_seconds": duration_seconds,
            "error": error,
        }
        return await self._pubsub.publish(Topic.TEST_COMPLETED.value, data)

    async def publish_suite_progress(
        self,
        suite_execution_id: int,
        suite_name: str,
        agent_name: str,
        status: str,
        progress_percent: float = 0.0,
        completed_tests: int = 0,
        total_tests: int = 0,
        passed_tests: int = 0,
        failed_tests: int = 0,
    ) -> int:
        """Publish a suite progress update.

        Args:
            suite_execution_id: Suite execution ID.
            suite_name: Suite name.
            agent_name: Agent name.
            status: Current suite status.
            progress_percent: Progress percentage.
            completed_tests: Completed tests count.
            total_tests: Total tests count.
            passed_tests: Passed tests count.
            failed_tests: Failed tests count.

        Returns:
            Number of clients that received the update.
        """
        data = {
            "suite_execution_id": suite_execution_id,
            "suite_name": suite_name,
            "agent_name": agent_name,
            "status": status,
            "progress_percent": progress_percent,
            "completed_tests": completed_tests,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
        }
        return await self._pubsub.publish(Topic.SUITE_PROGRESS.value, data)

    async def publish_suite_completed(
        self,
        suite_execution_id: int,
        suite_name: str,
        agent_name: str,
        success_rate: float,
        total_tests: int,
        passed_tests: int,
        failed_tests: int,
        duration_seconds: float | None = None,
        error: str | None = None,
    ) -> int:
        """Publish a suite completion update.

        Args:
            suite_execution_id: Suite execution ID.
            suite_name: Suite name.
            agent_name: Agent name.
            success_rate: Success rate (0-1).
            total_tests: Total tests count.
            passed_tests: Passed tests count.
            failed_tests: Failed tests count.
            duration_seconds: Optional duration.
            error: Optional error message.

        Returns:
            Number of clients that received the update.
        """
        data = {
            "suite_execution_id": suite_execution_id,
            "suite_name": suite_name,
            "agent_name": agent_name,
            "success_rate": success_rate,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "duration_seconds": duration_seconds,
            "error": error,
        }
        return await self._pubsub.publish(Topic.SUITE_COMPLETED.value, data)

    async def publish_event(
        self,
        suite_execution_id: int,
        test_execution_id: int,
        sequence: int,
        event_type: str,
        timestamp: datetime,
        payload: dict[str, Any],
        run_id: int | None = None,
        duration_ms: float | None = None,
    ) -> int:
        """Publish an ATP event.

        Args:
            suite_execution_id: Suite execution ID.
            test_execution_id: Test execution ID.
            sequence: Event sequence number.
            event_type: Event type.
            timestamp: Event timestamp.
            payload: Event payload.
            run_id: Optional run result ID.
            duration_ms: Optional event duration.

        Returns:
            Number of clients that received the event.
        """
        # Publish to specific event type topic
        topic = f"events:{event_type.lower()}"
        data = {
            "suite_execution_id": suite_execution_id,
            "test_execution_id": test_execution_id,
            "run_id": run_id,
            "sequence": sequence,
            "event_type": event_type,
            "timestamp": timestamp.isoformat(),
            "payload": payload,
            "duration_ms": duration_ms,
        }
        return await self._pubsub.publish(topic, data)

    async def publish_log(
        self,
        suite_execution_id: int,
        message: str,
        level: str = "info",
        test_execution_id: int | None = None,
        run_id: int | None = None,
        source: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> int:
        """Publish a log entry.

        Args:
            suite_execution_id: Suite execution ID.
            message: Log message.
            level: Log level.
            test_execution_id: Optional test execution ID.
            run_id: Optional run result ID.
            source: Optional log source.
            details: Optional extra details.

        Returns:
            Number of clients that received the log.
        """
        data = {
            "suite_execution_id": suite_execution_id,
            "test_execution_id": test_execution_id,
            "run_id": run_id,
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "details": details or {},
        }
        return await self._pubsub.publish(Topic.LOGS.value, data)

    async def publish_delta(
        self,
        resource_type: str,
        resource_id: int,
        changes: dict[str, Any],
    ) -> int:
        """Publish a delta update.

        Delta updates send only changed fields for efficient updates.

        Args:
            resource_type: Type of resource being updated.
            resource_id: ID of the resource.
            changes: Dictionary of changed fields.

        Returns:
            Number of clients that received the update.
        """
        data = {
            "resource_type": resource_type,
            "resource_id": resource_id,
            "changes": changes,
        }
        return await self._pubsub.publish(
            Topic.DELTA.value, data, include_sequence=True
        )


# Global connection manager instance (singleton)
_connection_manager: ConnectionManager | None = None


def get_connection_manager() -> ConnectionManager:
    """Get the global ConnectionManager instance.

    Returns:
        The global ConnectionManager instance.
    """
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


def set_connection_manager(manager: ConnectionManager) -> None:
    """Set the global ConnectionManager instance.

    Args:
        manager: The ConnectionManager instance to set.
    """
    global _connection_manager
    _connection_manager = manager
