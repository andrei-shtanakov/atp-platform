"""Tests for the ConnectionManager module."""

from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest

from atp.dashboard.v2.websocket.manager import (
    ConnectionManager,
    get_connection_manager,
    set_connection_manager,
)
from atp.dashboard.v2.websocket.models import WSMessage, WSMessageType
from atp.dashboard.v2.websocket.pubsub import PubSubManager


class MockWebSocket:
    """Mock WebSocket for testing.

    This mock implements the minimal WebSocket interface needed for testing.
    Type checking is suppressed when using this mock with ConnectionManager.
    """

    def __init__(self) -> None:
        self.accepted = False
        self.sent_messages: list[dict[str, Any]] = []
        self.closed = False
        self.headers: dict[str, str] = {}

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, data: dict[str, Any]) -> None:
        self.sent_messages.append(data)

    async def receive_json(self) -> dict[str, Any]:
        return {"type": "ping"}

    def close(self) -> None:
        self.closed = True


class TestConnectionManagerUnit:
    """Unit tests for ConnectionManager without background tasks."""

    def test_init(self) -> None:
        """Test ConnectionManager initialization."""
        manager = ConnectionManager()
        assert manager.connection_count == 0
        assert manager.pubsub is not None

    def test_init_with_custom_pubsub(self) -> None:
        """Test initialization with custom PubSubManager."""
        pubsub = PubSubManager()
        manager = ConnectionManager(pubsub=pubsub)
        assert manager.pubsub is pubsub

    def test_get_client_info_nonexistent(self) -> None:
        """Test getting info for nonexistent client."""
        manager = ConnectionManager()
        assert manager.get_client_info("nonexistent") is None

    def test_get_all_clients_empty(self) -> None:
        """Test getting all clients when none connected."""
        manager = ConnectionManager()
        assert manager.get_all_clients() == []


class TestConnectionManagerAsync:
    """Async tests for ConnectionManager."""

    @pytest.fixture
    def manager(self) -> ConnectionManager:
        """Create a ConnectionManager instance."""
        return ConnectionManager(ping_interval=0.1)

    @pytest.fixture
    def websocket(self) -> MockWebSocket:
        """Create a mock WebSocket."""
        return MockWebSocket()

    @pytest.mark.anyio
    async def test_connect_basic(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test basic WebSocket connection."""
        # Patch the background task to not start
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket)

            assert client_id is not None
            assert websocket.accepted is True
            assert manager.connection_count == 1

            # Cleanup
            await manager.disconnect(client_id)

    @pytest.mark.anyio
    async def test_connect_with_client_id(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test connecting with a specific client ID."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket, client_id="my-client-id")

            assert client_id == "my-client-id"
            assert manager.connection_count == 1

            await manager.disconnect(client_id)

    @pytest.mark.anyio
    async def test_connect_sends_connected_message(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test that connect sends a connected message."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket)

            assert len(websocket.sent_messages) == 1
            message = websocket.sent_messages[0]
            assert message["type"] == "connected"
            assert message["payload"]["client_id"] == client_id

            await manager.disconnect(client_id)

    @pytest.mark.anyio
    async def test_disconnect(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test disconnecting a WebSocket client."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket)
            await manager.disconnect(client_id)

            assert manager.connection_count == 0
            assert manager.get_client_info(client_id) is None

    @pytest.mark.anyio
    async def test_handle_ping(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test handling ping message."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket)
            websocket.sent_messages.clear()

            await manager.handle_message(client_id, {"type": "ping"})

            assert len(websocket.sent_messages) == 1
            assert websocket.sent_messages[0]["type"] == "pong"

            await manager.disconnect(client_id)

    @pytest.mark.anyio
    async def test_handle_subscribe(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test handling subscribe message."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket)
            websocket.sent_messages.clear()

            await manager.handle_message(
                client_id,
                {"type": "subscribe", "payload": {"topic": "test:progress"}},
            )

            assert len(websocket.sent_messages) == 1
            assert websocket.sent_messages[0]["type"] == "subscribed"
            assert websocket.sent_messages[0]["payload"]["topic"] == "test:progress"

            await manager.disconnect(client_id)

    @pytest.mark.anyio
    async def test_handle_unsubscribe(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test handling unsubscribe message."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket)
            await manager.handle_message(
                client_id,
                {"type": "subscribe", "payload": {"topic": "test:progress"}},
            )
            websocket.sent_messages.clear()

            await manager.handle_message(
                client_id,
                {"type": "unsubscribe", "payload": {"topic": "test:progress"}},
            )

            assert len(websocket.sent_messages) == 1
            assert websocket.sent_messages[0]["type"] == "unsubscribed"

            await manager.disconnect(client_id)

    @pytest.mark.anyio
    async def test_handle_unknown_message(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test handling unknown message type."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket)
            websocket.sent_messages.clear()

            await manager.handle_message(client_id, {"type": "unknown"})

            assert len(websocket.sent_messages) == 1
            assert websocket.sent_messages[0]["type"] == "error"
            assert (
                "Unknown message type" in websocket.sent_messages[0]["payload"]["error"]
            )

            await manager.disconnect(client_id)

    @pytest.mark.anyio
    async def test_get_client_info(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test getting client info."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket, user_agent="Test Agent")
            info = manager.get_client_info(client_id)

            assert info is not None
            assert info.client_id == client_id
            assert info.user_agent == "Test Agent"

            await manager.disconnect(client_id)

    @pytest.mark.anyio
    async def test_get_all_clients(self, manager: ConnectionManager) -> None:
        """Test getting all clients."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        with patch.object(manager, "_message_sender", return_value=None):
            await manager.connect(ws1, client_id="client1")
            await manager.connect(ws2, client_id="client2")

            clients = manager.get_all_clients()
            assert len(clients) == 2
            client_ids = [c.client_id for c in clients]
            assert "client1" in client_ids
            assert "client2" in client_ids

            await manager.disconnect("client1")
            await manager.disconnect("client2")

    @pytest.mark.anyio
    async def test_broadcast(self, manager: ConnectionManager) -> None:
        """Test broadcasting to all clients."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        with patch.object(manager, "_message_sender", return_value=None):
            await manager.connect(ws1, client_id="client1")
            await manager.connect(ws2, client_id="client2")

            ws1.sent_messages.clear()
            ws2.sent_messages.clear()

            message = WSMessage(type=WSMessageType.EVENT, payload={"test": "data"})
            count = await manager.broadcast(message)

            assert count == 2
            assert len(ws1.sent_messages) == 1
            assert len(ws2.sent_messages) == 1

            await manager.disconnect("client1")
            await manager.disconnect("client2")

    @pytest.mark.anyio
    async def test_broadcast_with_exclude(self, manager: ConnectionManager) -> None:
        """Test broadcasting with exclusion."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        with patch.object(manager, "_message_sender", return_value=None):
            await manager.connect(ws1, client_id="client1")
            await manager.connect(ws2, client_id="client2")

            ws1.sent_messages.clear()
            ws2.sent_messages.clear()

            message = WSMessage(type=WSMessageType.EVENT, payload={"test": "data"})
            count = await manager.broadcast(message, exclude=["client1"])

            assert count == 1
            assert len(ws1.sent_messages) == 0
            assert len(ws2.sent_messages) == 1

            await manager.disconnect("client1")
            await manager.disconnect("client2")

    @pytest.mark.anyio
    async def test_send_to_client(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test sending to specific client."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket)
            websocket.sent_messages.clear()

            message = WSMessage(type=WSMessageType.EVENT, payload={"test": "data"})
            success = await manager.send_to_client(client_id, message)

            assert success is True
            assert len(websocket.sent_messages) == 1

            await manager.disconnect(client_id)

    @pytest.mark.anyio
    async def test_publish_test_progress(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test publishing test progress."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket)
            await manager.handle_message(
                client_id,
                {"type": "subscribe", "payload": {"topic": "test:progress"}},
            )

            count = await manager.publish_test_progress(
                suite_execution_id=1,
                test_execution_id=5,
                test_id="test-001",
                test_name="Test One",
                status="running",
                progress_percent=50.0,
            )

            assert count == 1
            await manager.disconnect(client_id)

    @pytest.mark.anyio
    async def test_publish_suite_progress(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test publishing suite progress."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket)
            await manager.handle_message(
                client_id,
                {"type": "subscribe", "payload": {"topic": "suite:progress"}},
            )

            count = await manager.publish_suite_progress(
                suite_execution_id=1,
                suite_name="Test Suite",
                agent_name="Test Agent",
                status="running",
                progress_percent=25.0,
            )

            assert count == 1
            await manager.disconnect(client_id)

    @pytest.mark.anyio
    async def test_publish_event(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test publishing an event."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket)
            await manager.handle_message(
                client_id,
                {"type": "subscribe", "payload": {"topic": "events:tool_call"}},
            )

            count = await manager.publish_event(
                suite_execution_id=1,
                test_execution_id=5,
                sequence=10,
                event_type="tool_call",
                timestamp=datetime.now(),
                payload={"tool": "search"},
            )

            assert count == 1
            await manager.disconnect(client_id)

    @pytest.mark.anyio
    async def test_publish_log(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test publishing a log entry."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket)
            await manager.handle_message(
                client_id,
                {"type": "subscribe", "payload": {"topic": "logs"}},
            )

            count = await manager.publish_log(
                suite_execution_id=1,
                message="Test log message",
                level="info",
            )

            assert count == 1
            await manager.disconnect(client_id)

    @pytest.mark.anyio
    async def test_publish_delta(
        self, manager: ConnectionManager, websocket: MockWebSocket
    ) -> None:
        """Test publishing a delta update."""
        with patch.object(manager, "_message_sender", return_value=None):
            client_id = await manager.connect(websocket)
            await manager.handle_message(
                client_id,
                {"type": "subscribe", "payload": {"topic": "delta"}},
            )

            count = await manager.publish_delta(
                resource_type="test_execution",
                resource_id=5,
                changes={"score": 0.85},
            )

            assert count == 1
            await manager.disconnect(client_id)


class TestGlobalConnectionManager:
    """Tests for global connection manager functions."""

    def test_get_connection_manager(self) -> None:
        """Test getting the global connection manager."""
        # Reset the global instance
        set_connection_manager(None)  # type: ignore

        manager1 = get_connection_manager()
        manager2 = get_connection_manager()

        assert manager1 is manager2
        assert isinstance(manager1, ConnectionManager)

    def test_set_connection_manager(self) -> None:
        """Test setting a custom connection manager."""
        custom_manager = ConnectionManager()
        set_connection_manager(custom_manager)

        retrieved = get_connection_manager()
        assert retrieved is custom_manager

        # Reset
        set_connection_manager(None)  # type: ignore
