"""WebSocket route for real-time dashboard updates.

Provides the /ws/updates WebSocket endpoint for real-time updates.
"""

import logging
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from atp.dashboard.v2.websocket.manager import (
    get_connection_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/updates")
async def websocket_updates(
    websocket: WebSocket,
    client_id: str | None = Query(
        None, description="Optional client ID for reconnection"
    ),
) -> None:
    """WebSocket endpoint for real-time dashboard updates.

    Connect to receive real-time updates about test executions, events, and logs.

    **Connection:**
    - URL: `ws://{host}/api/ws/updates`
    - Optional query parameter: `client_id` for reconnection with same ID

    **Client -> Server Messages:**

    Subscribe to topics:
    ```json
    {
        "type": "subscribe",
        "payload": {
            "topic": "suite:progress",
            "filter": {"suite_execution_id": 123}
        }
    }
    ```

    Unsubscribe from topics:
    ```json
    {
        "type": "unsubscribe",
        "payload": {"topic": "suite:progress"}
    }
    ```

    Ping (keepalive):
    ```json
    {"type": "ping"}
    ```

    **Server -> Client Messages:**

    Connection confirmation:
    ```json
    {
        "type": "connected",
        "payload": {"client_id": "uuid"},
        "timestamp": "2026-01-31T12:00:00"
    }
    ```

    Test progress:
    ```json
    {
        "type": "test_progress",
        "payload": {
            "suite_execution_id": 1,
            "test_execution_id": 5,
            "test_id": "test-001",
            "test_name": "Test One",
            "status": "running",
            "progress_percent": 50.0,
            "current_run": 2,
            "total_runs": 3
        },
        "timestamp": "2026-01-31T12:00:00"
    }
    ```

    Event streaming:
    ```json
    {
        "type": "event",
        "payload": {
            "suite_execution_id": 1,
            "test_execution_id": 5,
            "sequence": 10,
            "event_type": "tool_call",
            "timestamp": "2026-01-31T12:00:00",
            "payload": {"tool": "search", "input": {...}},
            "duration_ms": 150.5
        },
        "timestamp": "2026-01-31T12:00:00"
    }
    ```

    **Available Topics:**
    - `suite:*` - All suite updates
    - `suite:progress` - Suite progress updates
    - `suite:completed` - Suite completion notifications
    - `test:*` - All test updates
    - `test:progress` - Test progress updates
    - `test:completed` - Test completion notifications
    - `events:*` - All events
    - `events:tool_call` - Tool call events
    - `events:llm_request` - LLM request events
    - `events:reasoning` - Reasoning events
    - `events:error` - Error events
    - `events:progress` - Progress events
    - `logs` - Log entries
    - `delta` - Delta updates (efficient partial updates)

    **Filtering:**
    Subscriptions can include filters to receive only relevant updates:
    ```json
    {
        "type": "subscribe",
        "payload": {
            "topic": "test:progress",
            "filter": {
                "suite_execution_id": 123,
                "test_id": "test-001"
            }
        }
    }
    ```
    """
    manager = get_connection_manager()

    # Get user agent from headers if available
    user_agent = None
    if websocket.headers:
        user_agent = websocket.headers.get("user-agent")

    # Connect client
    assigned_client_id = await manager.connect(
        websocket,
        client_id=client_id,
        user_agent=user_agent,
    )

    try:
        while True:
            # Receive and handle messages
            data = await websocket.receive_json()
            await manager.handle_message(assigned_client_id, data)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {assigned_client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {assigned_client_id}: {e}")
    finally:
        await manager.disconnect(assigned_client_id)


@router.get("/ws/info", response_model=dict[str, Any])
async def websocket_info() -> dict[str, Any]:
    """Get WebSocket server information.

    Returns information about the WebSocket server and available topics.

    Returns:
        WebSocket server info including:
        - Number of connected clients
        - Available topics
        - Server status
    """
    manager = get_connection_manager()

    return {
        "status": "active",
        "connected_clients": manager.connection_count,
        "active_subscriptions": manager.pubsub.active_subscriptions_count,
        "current_sequence": manager.pubsub.get_current_sequence(),
        "available_topics": [
            "suite:*",
            "suite:progress",
            "suite:completed",
            "test:*",
            "test:progress",
            "test:completed",
            "events:*",
            "events:tool_call",
            "events:llm_request",
            "events:reasoning",
            "events:error",
            "events:progress",
            "logs",
            "delta",
        ],
    }


@router.get("/ws/clients", response_model=list[dict[str, Any]])
async def websocket_clients() -> list[dict[str, Any]]:
    """Get information about connected WebSocket clients.

    Returns a list of all connected clients with their subscription info.

    Returns:
        List of connected clients with:
        - client_id
        - connected_at
        - subscriptions
        - last_activity
    """
    manager = get_connection_manager()
    clients = manager.get_all_clients()

    return [
        {
            "client_id": client.client_id,
            "connected_at": client.connected_at.isoformat(),
            "subscriptions": client.subscriptions,
            "last_activity": client.last_activity.isoformat(),
            "user_agent": client.user_agent,
        }
        for client in clients
    ]
