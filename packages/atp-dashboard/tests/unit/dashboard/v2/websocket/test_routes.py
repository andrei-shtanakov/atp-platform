"""Tests for WebSocket routes."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from atp.dashboard.v2.factory import create_test_app
from atp.dashboard.v2.websocket.manager import ConnectionManager


class TestWebSocketInfoEndpoint:
    """Tests for /ws/info endpoint."""

    @pytest.fixture
    def app(self):
        """Create test app with v2 routes."""
        return create_test_app(use_v2_routes=True)

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_ws_info_returns_status(self, client: TestClient) -> None:
        """Test /ws/info returns status information."""
        response = client.get("/api/ws/info")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert "connected_clients" in data
        assert "active_subscriptions" in data

    def test_ws_info_returns_available_topics(self, client: TestClient) -> None:
        """Test /ws/info returns available topics."""
        response = client.get("/api/ws/info")

        assert response.status_code == 200
        data = response.json()
        assert "available_topics" in data
        topics = data["available_topics"]

        # Check that expected topics are present
        assert "suite:*" in topics
        assert "test:progress" in topics
        assert "events:*" in topics
        assert "logs" in topics
        assert "delta" in topics


class TestWebSocketClientsEndpoint:
    """Tests for /ws/clients endpoint."""

    @pytest.fixture
    def app(self):
        """Create test app with v2 routes."""
        return create_test_app(use_v2_routes=True)

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_ws_clients_returns_list(self, client: TestClient) -> None:
        """Test /ws/clients returns a list."""
        response = client.get("/api/ws/clients")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_ws_clients_empty_initially(self, client: TestClient) -> None:
        """Test /ws/clients returns empty list initially."""
        response = client.get("/api/ws/clients")

        assert response.status_code == 200
        data = response.json()
        # No WebSocket clients connected via HTTP test client
        assert len(data) == 0


class TestWebSocketConnection:
    """Tests for WebSocket connection functionality.

    These tests use a mock ConnectionManager to avoid the background
    _message_sender task which can cause tests to hang.
    """

    @pytest.fixture
    def mock_manager(self):
        """Create a ConnectionManager with _message_sender mocked."""
        manager = ConnectionManager(ping_interval=0.1)
        # Mock the background task to prevent hanging
        manager._message_sender = AsyncMock(return_value=None)
        return manager

    @pytest.fixture
    def app(self, mock_manager):
        """Create test app with mocked connection manager."""
        app = create_test_app(use_v2_routes=True)
        return app

    @pytest.fixture
    def client(self, app, mock_manager):
        """Create test client with mocked manager."""
        with patch(
            "atp.dashboard.v2.routes.websocket.get_connection_manager",
            return_value=mock_manager,
        ):
            yield TestClient(app)

    def test_websocket_connect(
        self, client: TestClient, mock_manager: ConnectionManager
    ) -> None:
        """Test WebSocket connection."""
        with patch(
            "atp.dashboard.v2.routes.websocket.get_connection_manager",
            return_value=mock_manager,
        ):
            with client.websocket_connect("/api/ws/updates") as websocket:
                # Should receive connected message
                data = websocket.receive_json()
                assert data["type"] == "connected"
                assert "client_id" in data["payload"]

    def test_websocket_ping_pong(
        self, client: TestClient, mock_manager: ConnectionManager
    ) -> None:
        """Test ping/pong messages."""
        with patch(
            "atp.dashboard.v2.routes.websocket.get_connection_manager",
            return_value=mock_manager,
        ):
            with client.websocket_connect("/api/ws/updates") as websocket:
                # Receive connected message
                websocket.receive_json()

                # Send ping
                websocket.send_json({"type": "ping"})

                # Should receive pong
                data = websocket.receive_json()
                assert data["type"] == "pong"

    def test_websocket_subscribe(
        self, client: TestClient, mock_manager: ConnectionManager
    ) -> None:
        """Test subscription to a topic."""
        with patch(
            "atp.dashboard.v2.routes.websocket.get_connection_manager",
            return_value=mock_manager,
        ):
            with client.websocket_connect("/api/ws/updates") as websocket:
                # Receive connected message
                websocket.receive_json()

                # Subscribe to topic
                websocket.send_json(
                    {"type": "subscribe", "payload": {"topic": "test:progress"}}
                )

                # Should receive subscribed confirmation
                data = websocket.receive_json()
                assert data["type"] == "subscribed"
                assert data["payload"]["topic"] == "test:progress"

    def test_websocket_unsubscribe(
        self, client: TestClient, mock_manager: ConnectionManager
    ) -> None:
        """Test unsubscription from a topic."""
        with patch(
            "atp.dashboard.v2.routes.websocket.get_connection_manager",
            return_value=mock_manager,
        ):
            with client.websocket_connect("/api/ws/updates") as websocket:
                # Receive connected message
                websocket.receive_json()

                # Subscribe first
                websocket.send_json(
                    {"type": "subscribe", "payload": {"topic": "test:progress"}}
                )
                websocket.receive_json()  # subscribed

                # Unsubscribe
                websocket.send_json(
                    {"type": "unsubscribe", "payload": {"topic": "test:progress"}}
                )

                # Should receive unsubscribed confirmation
                data = websocket.receive_json()
                assert data["type"] == "unsubscribed"
                assert data["payload"]["topic"] == "test:progress"

    def test_websocket_unknown_message_type(
        self, client: TestClient, mock_manager: ConnectionManager
    ) -> None:
        """Test handling of unknown message type."""
        with patch(
            "atp.dashboard.v2.routes.websocket.get_connection_manager",
            return_value=mock_manager,
        ):
            with client.websocket_connect("/api/ws/updates") as websocket:
                # Receive connected message
                websocket.receive_json()

                # Send unknown message type
                websocket.send_json({"type": "unknown_type"})

                # Should receive error
                data = websocket.receive_json()
                assert data["type"] == "error"
                assert "Unknown message type" in data["payload"]["error"]

    def test_websocket_subscribe_with_filter(
        self, client: TestClient, mock_manager: ConnectionManager
    ) -> None:
        """Test subscription with filter."""
        with patch(
            "atp.dashboard.v2.routes.websocket.get_connection_manager",
            return_value=mock_manager,
        ):
            with client.websocket_connect("/api/ws/updates") as websocket:
                # Receive connected message
                websocket.receive_json()

                # Subscribe with filter
                websocket.send_json(
                    {
                        "type": "subscribe",
                        "payload": {
                            "topic": "test:progress",
                            "filter": {"suite_execution_id": 123},
                        },
                    }
                )

                # Should receive subscribed confirmation
                data = websocket.receive_json()
                assert data["type"] == "subscribed"

    def test_websocket_with_client_id(
        self, client: TestClient, mock_manager: ConnectionManager
    ) -> None:
        """Test WebSocket connection with client ID."""
        with patch(
            "atp.dashboard.v2.routes.websocket.get_connection_manager",
            return_value=mock_manager,
        ):
            with client.websocket_connect(
                "/api/ws/updates?client_id=my-custom-id"
            ) as websocket:
                data = websocket.receive_json()
                assert data["type"] == "connected"
                assert data["payload"]["client_id"] == "my-custom-id"

    def test_websocket_subscribe_missing_topic(
        self, client: TestClient, mock_manager: ConnectionManager
    ) -> None:
        """Test subscribe without topic returns error."""
        with patch(
            "atp.dashboard.v2.routes.websocket.get_connection_manager",
            return_value=mock_manager,
        ):
            with client.websocket_connect("/api/ws/updates") as websocket:
                # Receive connected message
                websocket.receive_json()

                # Subscribe without topic
                websocket.send_json({"type": "subscribe", "payload": {}})

                # Should receive error
                data = websocket.receive_json()
                assert data["type"] == "error"
                assert "Missing topic" in data["payload"]["error"]
