"""WebSocket module for real-time dashboard updates.

This module provides WebSocket functionality for the ATP Dashboard:
- Connection management for multiple clients
- Pub/sub system for test updates
- Real-time test progress and log streaming
- Efficient delta updates
"""

from atp.dashboard.v2.websocket.manager import ConnectionManager
from atp.dashboard.v2.websocket.models import (
    WSClientInfo,
    WSMessage,
    WSMessageType,
    WSSubscription,
)
from atp.dashboard.v2.websocket.pubsub import PubSubManager, Topic

__all__ = [
    "ConnectionManager",
    "PubSubManager",
    "Topic",
    "WSClientInfo",
    "WSMessage",
    "WSMessageType",
    "WSSubscription",
]
