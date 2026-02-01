"""WebSocket data models for ATP Dashboard.

Defines Pydantic models for WebSocket messages, subscriptions, and client info.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WSMessageType(str, Enum):
    """WebSocket message types."""

    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"

    # Server -> Client
    CONNECTED = "connected"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"
    PONG = "pong"
    ERROR = "error"

    # Data updates
    TEST_PROGRESS = "test_progress"
    TEST_COMPLETED = "test_completed"
    SUITE_PROGRESS = "suite_progress"
    SUITE_COMPLETED = "suite_completed"
    LOG_ENTRY = "log_entry"
    EVENT = "event"
    DELTA_UPDATE = "delta_update"


class WSSubscription(BaseModel):
    """WebSocket subscription request."""

    topic: str = Field(..., description="Topic to subscribe to")
    filter: dict[str, Any] = Field(
        default_factory=dict, description="Optional filters for the subscription"
    )


class WSMessage(BaseModel):
    """WebSocket message model."""

    type: WSMessageType = Field(..., description="Message type")
    payload: dict[str, Any] = Field(default_factory=dict, description="Message payload")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Message timestamp"
    )
    sequence: int | None = Field(None, description="Sequence number for delta updates")


class WSClientInfo(BaseModel):
    """Information about a connected WebSocket client."""

    client_id: str = Field(..., description="Unique client identifier")
    connected_at: datetime = Field(
        default_factory=datetime.now, description="Connection timestamp"
    )
    subscriptions: list[str] = Field(
        default_factory=list, description="Active subscriptions"
    )
    last_activity: datetime = Field(
        default_factory=datetime.now, description="Last activity timestamp"
    )
    user_agent: str | None = Field(None, description="Client user agent")


class TestProgressData(BaseModel):
    """Data payload for test progress updates."""

    suite_execution_id: int = Field(..., description="Suite execution ID")
    test_execution_id: int = Field(..., description="Test execution ID")
    test_id: str = Field(..., description="Test identifier")
    test_name: str = Field(..., description="Test display name")
    status: str = Field(..., description="Current test status")
    progress_percent: float = Field(
        0.0, ge=0, le=100, description="Progress percentage"
    )
    current_run: int = Field(1, ge=1, description="Current run number")
    total_runs: int = Field(1, ge=1, description="Total runs")
    message: str | None = Field(None, description="Status message")


class SuiteProgressData(BaseModel):
    """Data payload for suite progress updates."""

    suite_execution_id: int = Field(..., description="Suite execution ID")
    suite_name: str = Field(..., description="Suite name")
    agent_name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Current suite status")
    progress_percent: float = Field(
        0.0, ge=0, le=100, description="Progress percentage"
    )
    completed_tests: int = Field(0, ge=0, description="Completed tests count")
    total_tests: int = Field(0, ge=0, description="Total tests count")
    passed_tests: int = Field(0, ge=0, description="Passed tests count")
    failed_tests: int = Field(0, ge=0, description="Failed tests count")


class LogEntryData(BaseModel):
    """Data payload for log streaming."""

    suite_execution_id: int = Field(..., description="Suite execution ID")
    test_execution_id: int | None = Field(None, description="Test execution ID")
    run_id: int | None = Field(None, description="Run result ID")
    level: str = Field("info", description="Log level")
    message: str = Field(..., description="Log message")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Log timestamp"
    )
    source: str | None = Field(None, description="Log source")
    details: dict[str, Any] = Field(default_factory=dict, description="Extra details")


class EventData(BaseModel):
    """Data payload for ATP events."""

    suite_execution_id: int = Field(..., description="Suite execution ID")
    test_execution_id: int = Field(..., description="Test execution ID")
    run_id: int | None = Field(None, description="Run result ID")
    sequence: int = Field(..., description="Event sequence number")
    event_type: str = Field(..., description="Event type")
    timestamp: datetime = Field(..., description="Event timestamp")
    payload: dict[str, Any] = Field(default_factory=dict, description="Event payload")
    duration_ms: float | None = Field(None, description="Event duration")


class DeltaUpdateData(BaseModel):
    """Data payload for delta updates.

    Delta updates send only changed fields since the last update,
    reducing bandwidth and improving efficiency.
    """

    resource_type: str = Field(..., description="Type of resource being updated")
    resource_id: int = Field(..., description="ID of the resource")
    sequence: int = Field(..., description="Sequence number for ordering")
    changes: dict[str, Any] = Field(..., description="Changed fields")
    previous_sequence: int | None = Field(
        None, description="Previous sequence for continuity check"
    )
