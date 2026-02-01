"""Tests for WebSocket data models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from atp.dashboard.v2.websocket.models import (
    DeltaUpdateData,
    EventData,
    LogEntryData,
    SuiteProgressData,
    TestProgressData,
    WSClientInfo,
    WSMessage,
    WSMessageType,
    WSSubscription,
)


class TestWSMessageType:
    """Tests for WSMessageType enum."""

    def test_client_message_types(self) -> None:
        """Test client -> server message types."""
        assert WSMessageType.SUBSCRIBE.value == "subscribe"
        assert WSMessageType.UNSUBSCRIBE.value == "unsubscribe"
        assert WSMessageType.PING.value == "ping"

    def test_server_message_types(self) -> None:
        """Test server -> client message types."""
        assert WSMessageType.CONNECTED.value == "connected"
        assert WSMessageType.SUBSCRIBED.value == "subscribed"
        assert WSMessageType.UNSUBSCRIBED.value == "unsubscribed"
        assert WSMessageType.PONG.value == "pong"
        assert WSMessageType.ERROR.value == "error"

    def test_data_message_types(self) -> None:
        """Test data update message types."""
        assert WSMessageType.TEST_PROGRESS.value == "test_progress"
        assert WSMessageType.TEST_COMPLETED.value == "test_completed"
        assert WSMessageType.SUITE_PROGRESS.value == "suite_progress"
        assert WSMessageType.SUITE_COMPLETED.value == "suite_completed"
        assert WSMessageType.LOG_ENTRY.value == "log_entry"
        assert WSMessageType.EVENT.value == "event"
        assert WSMessageType.DELTA_UPDATE.value == "delta_update"


class TestWSSubscription:
    """Tests for WSSubscription model."""

    def test_basic_subscription(self) -> None:
        """Test creating a basic subscription."""
        sub = WSSubscription(topic="test:progress")
        assert sub.topic == "test:progress"
        assert sub.filter == {}

    def test_subscription_with_filter(self) -> None:
        """Test creating subscription with filter."""
        sub = WSSubscription(
            topic="test:progress",
            filter={"suite_execution_id": 123},
        )
        assert sub.topic == "test:progress"
        assert sub.filter["suite_execution_id"] == 123

    def test_subscription_requires_topic(self) -> None:
        """Test that subscription requires topic."""
        with pytest.raises(ValidationError):
            WSSubscription()  # type: ignore


class TestWSMessage:
    """Tests for WSMessage model."""

    def test_basic_message(self) -> None:
        """Test creating a basic message."""
        msg = WSMessage(
            type=WSMessageType.PING,
            payload={},
        )
        assert msg.type == WSMessageType.PING
        assert msg.payload == {}
        assert msg.timestamp is not None

    def test_message_with_payload(self) -> None:
        """Test creating message with payload."""
        msg = WSMessage(
            type=WSMessageType.TEST_PROGRESS,
            payload={"test_id": "test-001", "progress": 50},
        )
        assert msg.type == WSMessageType.TEST_PROGRESS
        assert msg.payload["test_id"] == "test-001"

    def test_message_with_sequence(self) -> None:
        """Test creating message with sequence."""
        msg = WSMessage(
            type=WSMessageType.DELTA_UPDATE,
            payload={"changes": {}},
            sequence=42,
        )
        assert msg.sequence == 42

    def test_message_serialization(self) -> None:
        """Test message serialization."""
        msg = WSMessage(
            type=WSMessageType.EVENT,
            payload={"data": "test"},
        )
        data = msg.model_dump(mode="json")
        assert data["type"] == "event"
        assert data["payload"]["data"] == "test"


class TestWSClientInfo:
    """Tests for WSClientInfo model."""

    def test_basic_client_info(self) -> None:
        """Test creating basic client info."""
        info = WSClientInfo(client_id="client-123")
        assert info.client_id == "client-123"
        assert info.subscriptions == []
        assert info.connected_at is not None

    def test_client_info_with_subscriptions(self) -> None:
        """Test client info with subscriptions."""
        info = WSClientInfo(
            client_id="client-123",
            subscriptions=["test:progress", "suite:progress"],
        )
        assert len(info.subscriptions) == 2
        assert "test:progress" in info.subscriptions

    def test_client_info_with_user_agent(self) -> None:
        """Test client info with user agent."""
        info = WSClientInfo(
            client_id="client-123",
            user_agent="Mozilla/5.0",
        )
        assert info.user_agent == "Mozilla/5.0"


class TestTestProgressData:
    """Tests for TestProgressData model."""

    def test_basic_progress(self) -> None:
        """Test creating basic progress data."""
        data = TestProgressData(
            suite_execution_id=1,
            test_execution_id=5,
            test_id="test-001",
            test_name="Test One",
            status="running",
        )
        assert data.suite_execution_id == 1
        assert data.test_id == "test-001"
        assert data.progress_percent == 0.0

    def test_progress_with_all_fields(self) -> None:
        """Test progress data with all fields."""
        data = TestProgressData(
            suite_execution_id=1,
            test_execution_id=5,
            test_id="test-001",
            test_name="Test One",
            status="running",
            progress_percent=75.5,
            current_run=2,
            total_runs=3,
            message="Processing...",
        )
        assert data.progress_percent == 75.5
        assert data.current_run == 2
        assert data.message == "Processing..."

    def test_progress_percent_validation(self) -> None:
        """Test progress percentage validation."""
        with pytest.raises(ValidationError):
            TestProgressData(
                suite_execution_id=1,
                test_execution_id=5,
                test_id="test-001",
                test_name="Test One",
                status="running",
                progress_percent=150.0,  # Invalid: > 100
            )


class TestSuiteProgressData:
    """Tests for SuiteProgressData model."""

    def test_basic_suite_progress(self) -> None:
        """Test creating basic suite progress."""
        data = SuiteProgressData(
            suite_execution_id=1,
            suite_name="Test Suite",
            agent_name="Test Agent",
            status="running",
        )
        assert data.suite_name == "Test Suite"
        assert data.completed_tests == 0

    def test_suite_progress_with_counts(self) -> None:
        """Test suite progress with test counts."""
        data = SuiteProgressData(
            suite_execution_id=1,
            suite_name="Test Suite",
            agent_name="Test Agent",
            status="running",
            progress_percent=50.0,
            completed_tests=5,
            total_tests=10,
            passed_tests=4,
            failed_tests=1,
        )
        assert data.completed_tests == 5
        assert data.total_tests == 10
        assert data.passed_tests == 4


class TestLogEntryData:
    """Tests for LogEntryData model."""

    def test_basic_log_entry(self) -> None:
        """Test creating basic log entry."""
        data = LogEntryData(
            suite_execution_id=1,
            message="Test log message",
        )
        assert data.message == "Test log message"
        assert data.level == "info"

    def test_log_entry_with_details(self) -> None:
        """Test log entry with all details."""
        data = LogEntryData(
            suite_execution_id=1,
            test_execution_id=5,
            run_id=10,
            level="error",
            message="An error occurred",
            source="test_runner",
            details={"error_code": 500},
        )
        assert data.level == "error"
        assert data.source == "test_runner"
        assert data.details["error_code"] == 500


class TestEventData:
    """Tests for EventData model."""

    def test_basic_event(self) -> None:
        """Test creating basic event data."""
        data = EventData(
            suite_execution_id=1,
            test_execution_id=5,
            sequence=10,
            event_type="tool_call",
            timestamp=datetime.now(),
        )
        assert data.sequence == 10
        assert data.event_type == "tool_call"

    def test_event_with_payload(self) -> None:
        """Test event with payload and duration."""
        data = EventData(
            suite_execution_id=1,
            test_execution_id=5,
            run_id=10,
            sequence=10,
            event_type="tool_call",
            timestamp=datetime.now(),
            payload={"tool": "search", "input": {"query": "test"}},
            duration_ms=150.5,
        )
        assert data.payload["tool"] == "search"
        assert data.duration_ms == 150.5


class TestDeltaUpdateData:
    """Tests for DeltaUpdateData model."""

    def test_basic_delta(self) -> None:
        """Test creating basic delta update."""
        data = DeltaUpdateData(
            resource_type="test_execution",
            resource_id=5,
            sequence=42,
            changes={"score": 0.85},
        )
        assert data.resource_type == "test_execution"
        assert data.changes["score"] == 0.85

    def test_delta_with_previous_sequence(self) -> None:
        """Test delta with previous sequence for continuity."""
        data = DeltaUpdateData(
            resource_type="test_execution",
            resource_id=5,
            sequence=43,
            changes={"status": "completed"},
            previous_sequence=42,
        )
        assert data.previous_sequence == 42
