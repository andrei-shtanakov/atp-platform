"""Tests for ATP Analytics ORM models."""

from datetime import datetime
from decimal import Decimal

from atp.analytics.models import CostBudget, CostRecord


class TestCostRecord:
    """Tests for CostRecord model."""

    def test_cost_record_creation(self) -> None:
        """Test creating a CostRecord instance."""
        now = datetime.now()
        record = CostRecord(
            id=1,
            timestamp=now,
            provider="anthropic",
            model="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=Decimal("0.015"),
        )

        assert record.id == 1
        assert record.timestamp == now
        assert record.provider == "anthropic"
        assert record.model == "claude-3-sonnet"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.cost_usd == Decimal("0.015")

    def test_cost_record_with_optional_fields(self) -> None:
        """Test CostRecord with optional test/suite/agent associations."""
        record = CostRecord(
            id=1,
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4",
            input_tokens=500,
            output_tokens=200,
            cost_usd=Decimal("0.02"),
            test_id="test-001",
            suite_id="suite-001",
            agent_name="my-agent",
            metadata_json={"extra": "data"},
        )

        assert record.test_id == "test-001"
        assert record.suite_id == "suite-001"
        assert record.agent_name == "my-agent"
        assert record.metadata_json == {"extra": "data"}

    def test_cost_record_total_tokens(self) -> None:
        """Test total_tokens property."""
        record = CostRecord(
            id=1,
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=Decimal("0.015"),
        )

        assert record.total_tokens == 1500

    def test_cost_record_repr(self) -> None:
        """Test CostRecord string representation."""
        record = CostRecord(
            id=1,
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=Decimal("0.015"),
        )

        repr_str = repr(record)
        assert "CostRecord" in repr_str
        assert "id=1" in repr_str
        assert "anthropic" in repr_str
        assert "claude-3-sonnet" in repr_str

    def test_cost_record_providers(self) -> None:
        """Test CostRecord with various providers."""
        providers = ["anthropic", "openai", "google", "azure", "bedrock"]

        for provider in providers:
            record = CostRecord(
                id=1,
                timestamp=datetime.now(),
                provider=provider,
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=Decimal("0.001"),
            )
            assert record.provider == provider


class TestCostBudget:
    """Tests for CostBudget model."""

    def test_cost_budget_creation(self) -> None:
        """Test creating a CostBudget instance."""
        budget = CostBudget(
            id=1,
            name="daily-limit",
            period="daily",
            limit_usd=Decimal("100.00"),
            alert_threshold=0.8,
            is_active=True,
        )

        assert budget.id == 1
        assert budget.name == "daily-limit"
        assert budget.period == "daily"
        assert budget.limit_usd == Decimal("100.00")
        assert budget.alert_threshold == 0.8
        assert budget.is_active is True

    def test_cost_budget_with_scope(self) -> None:
        """Test CostBudget with scope filters."""
        budget = CostBudget(
            id=1,
            name="anthropic-budget",
            period="monthly",
            limit_usd=Decimal("2000.00"),
            scope_json={"provider": "anthropic", "model": "claude-3-opus"},
        )

        assert budget.scope == {"provider": "anthropic", "model": "claude-3-opus"}

    def test_cost_budget_with_alert_channels(self) -> None:
        """Test CostBudget with alert channels."""
        budget = CostBudget(
            id=1,
            name="critical-budget",
            period="daily",
            limit_usd=Decimal("500.00"),
            alert_channels_json=["slack", "email", "pagerduty"],
        )

        assert budget.alert_channels == ["slack", "email", "pagerduty"]

    def test_cost_budget_scope_property_default(self) -> None:
        """Test scope property returns empty dict when None."""
        budget = CostBudget(
            id=1,
            name="test-budget",
            period="daily",
            limit_usd=Decimal("100.00"),
            scope_json=None,
        )

        assert budget.scope == {}

    def test_cost_budget_alert_channels_property_default(self) -> None:
        """Test alert_channels property returns empty list when None."""
        budget = CostBudget(
            id=1,
            name="test-budget",
            period="daily",
            limit_usd=Decimal("100.00"),
            alert_channels_json=None,
        )

        assert budget.alert_channels == []

    def test_cost_budget_repr(self) -> None:
        """Test CostBudget string representation."""
        budget = CostBudget(
            id=1,
            name="daily-limit",
            period="daily",
            limit_usd=Decimal("100.00"),
        )

        repr_str = repr(budget)
        assert "CostBudget" in repr_str
        assert "id=1" in repr_str
        assert "daily-limit" in repr_str
        assert "daily" in repr_str

    def test_cost_budget_periods(self) -> None:
        """Test CostBudget with various periods."""
        periods = ["daily", "weekly", "monthly"]

        for period in periods:
            budget = CostBudget(
                id=1,
                name=f"{period}-budget",
                period=period,
                limit_usd=Decimal("100.00"),
            )
            assert budget.period == period

    def test_cost_budget_with_description(self) -> None:
        """Test CostBudget with description."""
        budget = CostBudget(
            id=1,
            name="test-budget",
            period="monthly",
            limit_usd=Decimal("1000.00"),
            description="Budget for testing LLM costs",
        )

        assert budget.description == "Budget for testing LLM costs"

    def test_cost_budget_inactive(self) -> None:
        """Test inactive CostBudget."""
        budget = CostBudget(
            id=1,
            name="archived-budget",
            period="monthly",
            limit_usd=Decimal("500.00"),
            is_active=False,
        )

        assert budget.is_active is False
