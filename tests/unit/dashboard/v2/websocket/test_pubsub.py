"""Tests for the PubSubManager module."""

import pytest

from atp.dashboard.v2.websocket.pubsub import PubSubManager, Subscription, Topic


class TestSubscription:
    """Tests for Subscription class."""

    def test_matches_filter_empty(self) -> None:
        """Test matches_filter with empty filter."""
        sub = Subscription(topic="test", client_id="client1")
        assert sub.matches_filter({"any": "data"}) is True

    def test_matches_filter_exact_match(self) -> None:
        """Test matches_filter with exact match."""
        sub = Subscription(topic="test", client_id="client1", filter={"suite_id": 123})
        assert sub.matches_filter({"suite_id": 123, "other": "data"}) is True

    def test_matches_filter_no_match(self) -> None:
        """Test matches_filter with no match."""
        sub = Subscription(topic="test", client_id="client1", filter={"suite_id": 123})
        assert sub.matches_filter({"suite_id": 456}) is False

    def test_matches_filter_list_value(self) -> None:
        """Test matches_filter with list value in filter."""
        sub = Subscription(
            topic="test", client_id="client1", filter={"status": ["running", "pending"]}
        )
        assert sub.matches_filter({"status": "running"}) is True
        assert sub.matches_filter({"status": "completed"}) is False

    def test_matches_filter_missing_key(self) -> None:
        """Test matches_filter when key is missing from data."""
        sub = Subscription(
            topic="test", client_id="client1", filter={"required_key": "value"}
        )
        assert sub.matches_filter({"other_key": "value"}) is False


class TestPubSubManager:
    """Tests for PubSubManager class."""

    @pytest.fixture
    def pubsub(self) -> PubSubManager:
        """Create a PubSubManager instance."""
        return PubSubManager()

    @pytest.mark.anyio
    async def test_subscribe(self, pubsub: PubSubManager) -> None:
        """Test subscribing to a topic."""
        result = await pubsub.subscribe("client1", "test:progress")
        assert result is True
        assert "test:progress" in pubsub.get_client_subscriptions("client1")

    @pytest.mark.anyio
    async def test_subscribe_with_filter(self, pubsub: PubSubManager) -> None:
        """Test subscribing with a filter."""
        result = await pubsub.subscribe("client1", "test:progress", {"suite_id": 123})
        assert result is True
        assert "test:progress" in pubsub.get_client_subscriptions("client1")

    @pytest.mark.anyio
    async def test_unsubscribe(self, pubsub: PubSubManager) -> None:
        """Test unsubscribing from a topic."""
        await pubsub.subscribe("client1", "test:progress")
        result = await pubsub.unsubscribe("client1", "test:progress")
        assert result is True
        assert "test:progress" not in pubsub.get_client_subscriptions("client1")

    @pytest.mark.anyio
    async def test_unsubscribe_all(self, pubsub: PubSubManager) -> None:
        """Test unsubscribing from all topics."""
        await pubsub.subscribe("client1", "test:progress")
        await pubsub.subscribe("client1", "suite:progress")
        await pubsub.unsubscribe_all("client1")
        assert pubsub.get_client_subscriptions("client1") == []

    @pytest.mark.anyio
    async def test_publish_no_subscribers(self, pubsub: PubSubManager) -> None:
        """Test publishing with no subscribers."""
        count = await pubsub.publish("test:progress", {"data": "test"})
        assert count == 0

    @pytest.mark.anyio
    async def test_publish_with_subscriber(self, pubsub: PubSubManager) -> None:
        """Test publishing with a subscriber."""
        await pubsub.subscribe("client1", "test:progress")
        count = await pubsub.publish("test:progress", {"data": "test"})
        assert count == 1

    @pytest.mark.anyio
    async def test_publish_with_filter_match(self, pubsub: PubSubManager) -> None:
        """Test publishing with filter match."""
        await pubsub.subscribe("client1", "test:progress", {"suite_id": 123})
        count = await pubsub.publish("test:progress", {"suite_id": 123, "data": "test"})
        assert count == 1

    @pytest.mark.anyio
    async def test_publish_with_filter_no_match(self, pubsub: PubSubManager) -> None:
        """Test publishing with filter no match."""
        await pubsub.subscribe("client1", "test:progress", {"suite_id": 123})
        count = await pubsub.publish("test:progress", {"suite_id": 456, "data": "test"})
        assert count == 0

    @pytest.mark.anyio
    async def test_wildcard_subscription(self, pubsub: PubSubManager) -> None:
        """Test wildcard subscription."""
        await pubsub.subscribe("client1", "test:*")
        count = await pubsub.publish("test:progress", {"data": "test"})
        assert count == 1

    @pytest.mark.anyio
    async def test_get_messages(self, pubsub: PubSubManager) -> None:
        """Test getting messages from queue."""
        await pubsub.subscribe("client1", "test:progress")
        await pubsub.publish("test:progress", {"data": "test"})

        message = await pubsub.get_messages("client1", timeout=1.0)
        assert message is not None
        assert message["data"]["data"] == "test"

    @pytest.mark.anyio
    async def test_get_messages_timeout(self, pubsub: PubSubManager) -> None:
        """Test get_messages with timeout."""
        await pubsub.subscribe("client1", "test:progress")
        message = await pubsub.get_messages("client1", timeout=0.1)
        assert message is None

    @pytest.mark.anyio
    async def test_multiple_subscribers(self, pubsub: PubSubManager) -> None:
        """Test publishing to multiple subscribers."""
        await pubsub.subscribe("client1", "test:progress")
        await pubsub.subscribe("client2", "test:progress")
        count = await pubsub.publish("test:progress", {"data": "test"})
        assert count == 2

    @pytest.mark.anyio
    async def test_sequence_counter(self, pubsub: PubSubManager) -> None:
        """Test sequence counter for delta updates."""
        await pubsub.subscribe("client1", "delta")

        await pubsub.publish("delta", {"change": 1}, include_sequence=True)
        message1 = await pubsub.get_messages("client1", timeout=1.0)

        await pubsub.publish("delta", {"change": 2}, include_sequence=True)
        message2 = await pubsub.get_messages("client1", timeout=1.0)

        assert message1 is not None
        assert message2 is not None
        assert message1["data"]["sequence"] == 1
        assert message2["data"]["sequence"] == 2

    @pytest.mark.anyio
    async def test_get_topic_subscribers(self, pubsub: PubSubManager) -> None:
        """Test getting subscribers for a topic."""
        await pubsub.subscribe("client1", "test:progress")
        await pubsub.subscribe("client2", "test:progress")
        subscribers = pubsub.get_topic_subscribers("test:progress")
        assert "client1" in subscribers
        assert "client2" in subscribers

    @pytest.mark.anyio
    async def test_active_subscriptions_count(self, pubsub: PubSubManager) -> None:
        """Test active subscriptions count."""
        await pubsub.subscribe("client1", "test:progress")
        await pubsub.subscribe("client1", "suite:progress")
        await pubsub.subscribe("client2", "test:progress")
        assert pubsub.active_subscriptions_count == 3

    @pytest.mark.anyio
    async def test_connected_clients_count(self, pubsub: PubSubManager) -> None:
        """Test connected clients count."""
        await pubsub.subscribe("client1", "test:progress")
        await pubsub.subscribe("client2", "test:progress")
        assert pubsub.connected_clients_count == 2


class TestTopic:
    """Tests for Topic enum."""

    def test_topic_values(self) -> None:
        """Test that Topic enum has expected values."""
        assert Topic.SUITE_ALL.value == "suite:*"
        assert Topic.SUITE_PROGRESS.value == "suite:progress"
        assert Topic.TEST_ALL.value == "test:*"
        assert Topic.EVENTS_ALL.value == "events:*"
        assert Topic.LOGS.value == "logs"
        assert Topic.DELTA.value == "delta"

    def test_topic_event_types(self) -> None:
        """Test event type topics."""
        assert Topic.EVENTS_TOOL_CALL.value == "events:tool_call"
        assert Topic.EVENTS_LLM_REQUEST.value == "events:llm_request"
        assert Topic.EVENTS_REASONING.value == "events:reasoning"
        assert Topic.EVENTS_ERROR.value == "events:error"
        assert Topic.EVENTS_PROGRESS.value == "events:progress"
