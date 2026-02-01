"""Tests for ATP Analytics CostTracker service."""

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.analytics.cost import (
    CostEvent,
    CostTracker,
    ModelPricing,
    PricingConfig,
    get_cost_tracker,
    set_cost_tracker,
    shutdown_cost_tracker,
)


class TestModelPricing:
    """Tests for ModelPricing class."""

    def test_calculate_cost(self) -> None:
        """Test calculating cost from token counts."""
        pricing = ModelPricing(
            input_per_1k=Decimal("0.003"),
            output_per_1k=Decimal("0.015"),
            name="Test Model",
        )

        cost = pricing.calculate_cost(input_tokens=1000, output_tokens=500)

        # 1000 * 0.003/1000 + 500 * 0.015/1000 = 0.003 + 0.0075 = 0.0105
        assert cost == Decimal("0.0105")

    def test_calculate_cost_large_tokens(self) -> None:
        """Test calculating cost with large token counts."""
        pricing = ModelPricing(
            input_per_1k=Decimal("0.003"),
            output_per_1k=Decimal("0.015"),
        )

        cost = pricing.calculate_cost(input_tokens=100000, output_tokens=50000)

        # 100000 * 0.003/1000 + 50000 * 0.015/1000 = 0.3 + 0.75 = 1.05
        assert cost == Decimal("1.05")

    def test_calculate_cost_zero_tokens(self) -> None:
        """Test calculating cost with zero tokens."""
        pricing = ModelPricing(
            input_per_1k=Decimal("0.003"),
            output_per_1k=Decimal("0.015"),
        )

        cost = pricing.calculate_cost(input_tokens=0, output_tokens=0)

        assert cost == Decimal("0")


class TestPricingConfig:
    """Tests for PricingConfig class."""

    def test_default_config(self) -> None:
        """Test default pricing configuration has major models."""
        config = PricingConfig.default()

        # Check Anthropic models
        assert "claude-sonnet-4-20250514" in config.models
        assert "claude-3-5-sonnet-20241022" in config.models
        assert "claude-3-opus-20240229" in config.models

        # Check OpenAI models
        assert "gpt-4o" in config.models
        assert "gpt-4o-mini" in config.models
        assert "gpt-4" in config.models

        # Check Google models
        assert "gemini-1.5-pro" in config.models
        assert "gemini-2.0-flash" in config.models

        # Check provider defaults
        assert "anthropic" in config.provider_defaults
        assert "openai" in config.provider_defaults
        assert "google" in config.provider_defaults
        assert "azure" in config.provider_defaults
        assert "bedrock" in config.provider_defaults

    def test_calculate_known_model(self) -> None:
        """Test calculating cost for a known model."""
        config = PricingConfig.default()

        cost = config.calculate(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
        )

        # claude-sonnet-4: $0.003/1k input, $0.015/1k output
        # 1000 * 0.003/1000 + 500 * 0.015/1000 = 0.003 + 0.0075 = 0.0105
        assert cost == Decimal("0.0105")

    def test_calculate_unknown_model_uses_provider_default(self) -> None:
        """Test calculating cost for unknown model uses provider default."""
        config = PricingConfig.default()

        cost = config.calculate(
            provider="anthropic",
            model="claude-unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )

        # Should use anthropic default: $0.003/1k input, $0.015/1k output
        expected = Decimal("1000") / Decimal("1000") * Decimal("0.003")
        expected += Decimal("500") / Decimal("1000") * Decimal("0.015")
        assert cost == expected

    def test_calculate_unknown_provider_returns_zero(self) -> None:
        """Test calculating cost for unknown provider returns zero."""
        config = PricingConfig.default()

        cost = config.calculate(
            provider="unknown-provider",
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )

        assert cost == Decimal("0")

    def test_add_custom_pricing(self) -> None:
        """Test adding custom pricing for a model."""
        config = PricingConfig.default()

        config.add_custom_pricing(
            model="custom-model",
            input_per_1k=0.001,
            output_per_1k=0.002,
            name="Custom Model",
        )

        assert "custom-model" in config.models
        assert config.models["custom-model"].input_per_1k == Decimal("0.001")
        assert config.models["custom-model"].output_per_1k == Decimal("0.002")
        assert config.models["custom-model"].name == "Custom Model"

    def test_get_model_pricing(self) -> None:
        """Test getting pricing for a specific model."""
        config = PricingConfig.default()

        pricing = config.get_model_pricing("gpt-4o")

        assert pricing is not None
        assert pricing.input_per_1k == Decimal("0.0025")
        assert pricing.output_per_1k == Decimal("0.01")

    def test_get_model_pricing_not_found(self) -> None:
        """Test getting pricing for unknown model returns None."""
        config = PricingConfig.default()

        pricing = config.get_model_pricing("unknown-model")

        assert pricing is None

    def test_from_yaml(self, tmp_path: Path) -> None:
        """Test loading pricing from YAML file."""
        yaml_content = """
models:
  custom-model-1:
    input_per_1k: 0.001
    output_per_1k: 0.002
    name: Custom Model 1
  custom-model-2:
    input_per_1k: 0.003
    output_per_1k: 0.006

provider_defaults:
  custom-provider:
    input_per_1k: 0.005
    output_per_1k: 0.01
"""
        yaml_path = tmp_path / "pricing.yaml"
        yaml_path.write_text(yaml_content)

        config = PricingConfig.from_yaml(yaml_path)

        # Check custom models were loaded
        assert "custom-model-1" in config.models
        assert config.models["custom-model-1"].input_per_1k == Decimal("0.001")

        # Check default models still exist
        assert "gpt-4o" in config.models

        # Check custom provider default
        assert "custom-provider" in config.provider_defaults
        assert config.provider_defaults["custom-provider"].input_per_1k == Decimal(
            "0.005"
        )


class TestCostEvent:
    """Tests for CostEvent dataclass."""

    def test_create_event(self) -> None:
        """Test creating a cost event."""
        now = datetime.now()
        event = CostEvent(
            timestamp=now,
            provider="anthropic",
            model="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=500,
        )

        assert event.timestamp == now
        assert event.provider == "anthropic"
        assert event.model == "claude-3-sonnet"
        assert event.input_tokens == 1000
        assert event.output_tokens == 500
        assert event.test_id is None
        assert event.suite_id is None
        assert event.agent_name is None
        assert event.metadata is None

    def test_create_event_with_associations(self) -> None:
        """Test creating a cost event with associations."""
        now = datetime.now()
        event = CostEvent(
            timestamp=now,
            provider="openai",
            model="gpt-4",
            input_tokens=500,
            output_tokens=200,
            test_id="test-001",
            suite_id="suite-001",
            agent_name="my-agent",
            metadata={"key": "value"},
        )

        assert event.test_id == "test-001"
        assert event.suite_id == "suite-001"
        assert event.agent_name == "my-agent"
        assert event.metadata == {"key": "value"}


class TestCostTracker:
    """Tests for CostTracker class."""

    @pytest.mark.anyio
    async def test_tracker_start_stop(self) -> None:
        """Test starting and stopping the tracker."""
        tracker = CostTracker()

        assert not tracker.is_running

        await tracker.start()
        assert tracker.is_running

        await tracker.stop()
        assert not tracker.is_running

    @pytest.mark.anyio
    async def test_tracker_double_start(self) -> None:
        """Test starting tracker twice is safe."""
        tracker = CostTracker()

        await tracker.start()
        await tracker.start()  # Should not raise

        assert tracker.is_running

        await tracker.stop()

    @pytest.mark.anyio
    async def test_tracker_stats(self) -> None:
        """Test getting tracker statistics."""
        tracker = CostTracker()

        stats = tracker.stats

        assert stats["events_processed"] == 0
        assert stats["events_failed"] == 0
        assert stats["batches_processed"] == 0
        assert stats["queue_size"] == 0

    @pytest.mark.anyio
    async def test_track_event(self) -> None:
        """Test tracking a cost event."""
        # Mock the database to avoid actual DB operations
        with patch("atp.analytics.cost.get_analytics_database") as mock_get_db:
            mock_db = MagicMock()
            mock_session = AsyncMock()
            mock_db.session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_db.session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.cost.CostRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo_cls.return_value = mock_repo

                tracker = CostTracker(batch_timeout=0.1)
                await tracker.start()

                event = CostEvent(
                    timestamp=datetime.now(),
                    provider="anthropic",
                    model="claude-3-sonnet",
                    input_tokens=1000,
                    output_tokens=500,
                )

                await tracker.track(event)

                # Wait for processing
                await tracker.flush()
                await tracker.stop()

                # Verify event was processed
                assert tracker.stats["events_processed"] >= 1

    @pytest.mark.anyio
    async def test_track_llm_call(self) -> None:
        """Test convenience method for tracking LLM calls."""
        # Mock the database to avoid actual DB operations
        with patch("atp.analytics.cost.get_analytics_database") as mock_get_db:
            mock_db = MagicMock()
            mock_session = AsyncMock()
            mock_db.session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_db.session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.cost.CostRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo_cls.return_value = mock_repo

                tracker = CostTracker(batch_timeout=0.1)
                await tracker.start()

                cost = await tracker.track_llm_call(
                    provider="anthropic",
                    model="claude-sonnet-4-20250514",
                    input_tokens=1000,
                    output_tokens=500,
                )

                # Should return calculated cost
                assert cost == Decimal("0.0105")

                await tracker.flush()
                await tracker.stop()

    def test_calculate_cost_without_tracking(self) -> None:
        """Test calculating cost without tracking."""
        tracker = CostTracker()

        cost = tracker.calculate_cost(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
        )

        assert cost == Decimal("0.0105")

    @pytest.mark.anyio
    async def test_batch_processing(self, tmp_path: Path) -> None:
        """Test batch processing of events."""
        # Create a tracker with small batch size for testing
        tracker = CostTracker(batch_size=5, batch_timeout=0.1)

        # Mock the database
        with patch("atp.analytics.cost.get_analytics_database") as mock_get_db:
            mock_db = MagicMock()
            mock_session = AsyncMock()
            mock_db.session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_db.session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_get_db.return_value = mock_db

            # Mock the repository
            with patch("atp.analytics.cost.CostRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo_cls.return_value = mock_repo

                await tracker.start()

                # Track multiple events
                for i in range(10):
                    await tracker.track(
                        CostEvent(
                            timestamp=datetime.now(),
                            provider="anthropic",
                            model="claude-3-sonnet",
                            input_tokens=100 * (i + 1),
                            output_tokens=50 * (i + 1),
                        )
                    )

                # Wait for processing
                await tracker.flush()
                await tracker.stop()

                # Verify batch processing occurred
                assert tracker.stats["events_processed"] == 10

    @pytest.mark.anyio
    async def test_custom_pricing(self) -> None:
        """Test using custom pricing configuration."""
        pricing = PricingConfig.default()
        pricing.add_custom_pricing(
            model="custom-model",
            input_per_1k=0.01,
            output_per_1k=0.02,
        )

        tracker = CostTracker(pricing=pricing)

        cost = tracker.calculate_cost(
            provider="custom",
            model="custom-model",
            input_tokens=1000,
            output_tokens=1000,
        )

        # 1000 * 0.01/1000 + 1000 * 0.02/1000 = 0.01 + 0.02 = 0.03
        assert cost == Decimal("0.03")

    @pytest.mark.anyio
    async def test_queue_overflow_handling(self) -> None:
        """Test handling queue overflow gracefully."""
        # Create tracker with very small queue
        tracker = CostTracker(max_queue_size=2)

        # Don't start the processor so queue fills up
        for _ in range(5):
            await tracker.track(
                CostEvent(
                    timestamp=datetime.now(),
                    provider="anthropic",
                    model="claude-3-sonnet",
                    input_tokens=100,
                    output_tokens=50,
                )
            )

        # Should have dropped some events
        assert tracker.stats["events_failed"] >= 2


class TestGlobalCostTracker:
    """Tests for global cost tracker functions."""

    @pytest.mark.anyio
    async def test_get_cost_tracker(self) -> None:
        """Test getting global cost tracker."""
        # Reset global tracker
        await shutdown_cost_tracker()

        tracker = await get_cost_tracker()

        assert tracker is not None
        assert tracker.is_running

        await shutdown_cost_tracker()

    @pytest.mark.anyio
    async def test_set_cost_tracker(self) -> None:
        """Test setting custom cost tracker."""
        await shutdown_cost_tracker()

        custom_tracker = CostTracker()
        set_cost_tracker(custom_tracker)

        tracker = await get_cost_tracker()

        assert tracker is custom_tracker

        await shutdown_cost_tracker()

    @pytest.mark.anyio
    async def test_shutdown_cost_tracker(self) -> None:
        """Test shutting down global cost tracker."""
        # Get and start tracker
        tracker = await get_cost_tracker()
        assert tracker.is_running

        # Shutdown
        await shutdown_cost_tracker()

        # Getting again should create new tracker
        new_tracker = await get_cost_tracker()
        assert new_tracker is not tracker

        await shutdown_cost_tracker()


class TestPricingConfigFromYaml:
    """Tests for loading pricing from YAML."""

    def test_from_yaml_invalid_format(self, tmp_path: Path) -> None:
        """Test loading from invalid YAML raises error."""
        yaml_path = tmp_path / "invalid.yaml"
        yaml_path.write_text("not a dict")

        with pytest.raises(ValueError, match="must be a dictionary"):
            PricingConfig.from_yaml(yaml_path)

    def test_from_yaml_invalid_model_pricing(self, tmp_path: Path) -> None:
        """Test loading invalid model pricing raises error."""
        yaml_content = """
models:
  bad-model: "not a dict"
"""
        yaml_path = tmp_path / "pricing.yaml"
        yaml_path.write_text(yaml_content)

        with pytest.raises(ValueError, match="Invalid pricing for model"):
            PricingConfig.from_yaml(yaml_path)

    def test_from_yaml_file_not_found(self, tmp_path: Path) -> None:
        """Test loading from non-existent file raises error."""
        yaml_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            PricingConfig.from_yaml(yaml_path)
