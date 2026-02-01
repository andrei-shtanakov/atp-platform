"""Tests for ATP Analytics CostRepository."""

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.analytics.database import AnalyticsDatabase
from atp.analytics.models import CostBudget, CostRecord
from atp.analytics.repository import CostRepository


class TestCostRecordCRUD:
    """Tests for CostRecord CRUD operations."""

    @pytest.mark.anyio
    async def test_create_cost_record(self) -> None:
        """Test creating a cost record."""
        mock_session = AsyncMock()

        repo = CostRepository(mock_session)
        now = datetime.now()

        record = await repo.create_cost_record(
            timestamp=now,
            provider="anthropic",
            model="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=Decimal("0.015"),
        )

        assert record.timestamp == now
        assert record.provider == "anthropic"
        assert record.model == "claude-3-sonnet"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.cost_usd == Decimal("0.015")
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_create_cost_record_with_associations(self) -> None:
        """Test creating a cost record with test/suite/agent associations."""
        mock_session = AsyncMock()

        repo = CostRepository(mock_session)

        record = await repo.create_cost_record(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4",
            input_tokens=500,
            output_tokens=200,
            cost_usd=Decimal("0.02"),
            test_id="test-001",
            suite_id="suite-001",
            agent_name="my-agent",
            metadata={"run_number": 1},
        )

        assert record.test_id == "test-001"
        assert record.suite_id == "suite-001"
        assert record.agent_name == "my-agent"
        assert record.metadata_json == {"run_number": 1}

    @pytest.mark.anyio
    async def test_create_cost_records_batch(self) -> None:
        """Test creating multiple cost records in batch."""
        mock_session = AsyncMock()
        repo = CostRepository(mock_session)

        now = datetime.now()
        records = [
            CostRecord(
                timestamp=now,
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=100,
                output_tokens=50,
                cost_usd=Decimal("0.001"),
            )
            for _ in range(5)
        ]

        result = await repo.create_cost_records_batch(records)

        assert len(result) == 5
        assert mock_session.add.call_count == 5
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_get_cost_record(self) -> None:
        """Test getting a cost record by ID."""
        existing_record = CostRecord(
            id=1,
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=Decimal("0.015"),
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_record
        mock_session.execute.return_value = mock_result

        repo = CostRepository(mock_session)
        record = await repo.get_cost_record(1)

        assert record is existing_record

    @pytest.mark.anyio
    async def test_get_cost_record_not_found(self) -> None:
        """Test getting a non-existent cost record."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = CostRepository(mock_session)
        record = await repo.get_cost_record(999)

        assert record is None

    @pytest.mark.anyio
    async def test_list_cost_records(self) -> None:
        """Test listing cost records."""
        records = [
            CostRecord(
                id=i,
                timestamp=datetime.now(),
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=100,
                output_tokens=50,
                cost_usd=Decimal("0.001"),
            )
            for i in range(1, 4)
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = records
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        repo = CostRepository(mock_session)
        result = await repo.list_cost_records()

        assert len(result) == 3

    @pytest.mark.anyio
    async def test_delete_cost_record(self) -> None:
        """Test deleting a cost record."""
        existing_record = CostRecord(
            id=1,
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=Decimal("0.015"),
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_record
        mock_session.execute.return_value = mock_result

        repo = CostRepository(mock_session)
        deleted = await repo.delete_cost_record(1)

        assert deleted is True
        mock_session.delete.assert_called_once_with(existing_record)

    @pytest.mark.anyio
    async def test_delete_cost_record_not_found(self) -> None:
        """Test deleting a non-existent cost record."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = CostRepository(mock_session)
        deleted = await repo.delete_cost_record(999)

        assert deleted is False
        mock_session.delete.assert_not_called()


class TestCostBudgetCRUD:
    """Tests for CostBudget CRUD operations."""

    @pytest.mark.anyio
    async def test_create_budget(self) -> None:
        """Test creating a budget."""
        mock_session = AsyncMock()

        repo = CostRepository(mock_session)

        budget = await repo.create_budget(
            name="daily-limit",
            period="daily",
            limit_usd=Decimal("100.00"),
            alert_threshold=0.8,
        )

        assert budget.name == "daily-limit"
        assert budget.period == "daily"
        assert budget.limit_usd == Decimal("100.00")
        assert budget.alert_threshold == 0.8
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_create_budget_with_scope(self) -> None:
        """Test creating a budget with scope filters."""
        mock_session = AsyncMock()

        repo = CostRepository(mock_session)

        budget = await repo.create_budget(
            name="anthropic-budget",
            period="monthly",
            limit_usd=Decimal("2000.00"),
            scope={"provider": "anthropic"},
            alert_channels=["slack", "email"],
            description="Monthly budget for Anthropic API",
        )

        assert budget.scope_json == {"provider": "anthropic"}
        assert budget.alert_channels_json == ["slack", "email"]
        assert budget.description == "Monthly budget for Anthropic API"

    @pytest.mark.anyio
    async def test_get_budget(self) -> None:
        """Test getting a budget by ID."""
        existing_budget = CostBudget(
            id=1,
            name="test-budget",
            period="daily",
            limit_usd=Decimal("100.00"),
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_budget
        mock_session.execute.return_value = mock_result

        repo = CostRepository(mock_session)
        budget = await repo.get_budget(1)

        assert budget is existing_budget

    @pytest.mark.anyio
    async def test_get_budget_by_name(self) -> None:
        """Test getting a budget by name."""
        existing_budget = CostBudget(
            id=1,
            name="test-budget",
            period="daily",
            limit_usd=Decimal("100.00"),
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_budget
        mock_session.execute.return_value = mock_result

        repo = CostRepository(mock_session)
        budget = await repo.get_budget_by_name("test-budget")

        assert budget is existing_budget
        assert budget.name == "test-budget"

    @pytest.mark.anyio
    async def test_list_budgets(self) -> None:
        """Test listing budgets."""
        budgets = [
            CostBudget(
                id=i,
                name=f"budget-{i}",
                period="daily",
                limit_usd=Decimal("100.00"),
            )
            for i in range(1, 4)
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = budgets
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        repo = CostRepository(mock_session)
        result = await repo.list_budgets()

        assert len(result) == 3

    @pytest.mark.anyio
    async def test_update_budget(self) -> None:
        """Test updating a budget."""
        budget = CostBudget(
            id=1,
            name="test-budget",
            period="daily",
            limit_usd=Decimal("100.00"),
            alert_threshold=0.8,
        )

        mock_session = AsyncMock()

        repo = CostRepository(mock_session)
        updated = await repo.update_budget(
            budget,
            name="updated-budget",
            limit_usd=Decimal("200.00"),
            alert_threshold=0.9,
        )

        assert updated.name == "updated-budget"
        assert updated.limit_usd == Decimal("200.00")
        assert updated.alert_threshold == 0.9
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_delete_budget(self) -> None:
        """Test deleting a budget."""
        existing_budget = CostBudget(
            id=1,
            name="test-budget",
            period="daily",
            limit_usd=Decimal("100.00"),
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_budget
        mock_session.execute.return_value = mock_result

        repo = CostRepository(mock_session)
        deleted = await repo.delete_budget(1)

        assert deleted is True
        mock_session.delete.assert_called_once_with(existing_budget)


class TestAggregationQueries:
    """Tests for aggregation query operations."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create temporary database path."""
        return f"sqlite+aiosqlite:///{tmp_path / 'test_analytics.db'}"

    @pytest.mark.anyio
    async def test_get_total_cost(self, temp_db_path: str) -> None:
        """Test getting total cost."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            repo = CostRepository(session)

            # Insert some records
            now = datetime.now()
            for i in range(3):
                await repo.create_cost_record(
                    timestamp=now,
                    provider="anthropic",
                    model="claude-3-sonnet",
                    input_tokens=100,
                    output_tokens=50,
                    cost_usd=Decimal("0.01"),
                )

        async with db.session() as session:
            repo = CostRepository(session)
            total = await repo.get_total_cost()

            assert total == Decimal("0.03")

        await db.close()

    @pytest.mark.anyio
    async def test_get_total_cost_with_filters(self, temp_db_path: str) -> None:
        """Test getting total cost with filters."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            repo = CostRepository(session)

            now = datetime.now()
            # Add anthropic records
            for _ in range(2):
                await repo.create_cost_record(
                    timestamp=now,
                    provider="anthropic",
                    model="claude-3-sonnet",
                    input_tokens=100,
                    output_tokens=50,
                    cost_usd=Decimal("0.01"),
                )
            # Add openai record
            await repo.create_cost_record(
                timestamp=now,
                provider="openai",
                model="gpt-4",
                input_tokens=100,
                output_tokens=50,
                cost_usd=Decimal("0.02"),
            )

        async with db.session() as session:
            repo = CostRepository(session)
            total = await repo.get_total_cost(provider="anthropic")

            assert total == Decimal("0.02")

        await db.close()

    @pytest.mark.anyio
    async def test_get_costs_by_provider(self, temp_db_path: str) -> None:
        """Test getting costs aggregated by provider."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            repo = CostRepository(session)

            now = datetime.now()
            await repo.create_cost_record(
                timestamp=now,
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=Decimal("0.015"),
            )
            await repo.create_cost_record(
                timestamp=now,
                provider="openai",
                model="gpt-4",
                input_tokens=500,
                output_tokens=200,
                cost_usd=Decimal("0.02"),
            )

        async with db.session() as session:
            repo = CostRepository(session)
            costs = await repo.get_costs_by_provider()

            assert len(costs) == 2
            # Results ordered by total_cost desc
            assert costs[0]["provider"] == "openai"
            assert costs[0]["total_cost"] == Decimal("0.02")
            assert costs[1]["provider"] == "anthropic"
            assert costs[1]["total_cost"] == Decimal("0.015")

        await db.close()

    @pytest.mark.anyio
    async def test_get_costs_by_model(self, temp_db_path: str) -> None:
        """Test getting costs aggregated by model."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            repo = CostRepository(session)

            now = datetime.now()
            await repo.create_cost_record(
                timestamp=now,
                provider="anthropic",
                model="claude-3-opus",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=Decimal("0.05"),
            )
            await repo.create_cost_record(
                timestamp=now,
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=Decimal("0.015"),
            )

        async with db.session() as session:
            repo = CostRepository(session)
            costs = await repo.get_costs_by_model()

            assert len(costs) == 2
            # Results ordered by total_cost desc
            assert costs[0]["model"] == "claude-3-opus"
            assert costs[1]["model"] == "claude-3-sonnet"

        await db.close()

    @pytest.mark.anyio
    async def test_get_costs_by_agent(self, temp_db_path: str) -> None:
        """Test getting costs aggregated by agent."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            repo = CostRepository(session)

            now = datetime.now()
            await repo.create_cost_record(
                timestamp=now,
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=Decimal("0.015"),
                agent_name="agent-1",
            )
            await repo.create_cost_record(
                timestamp=now,
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=500,
                output_tokens=200,
                cost_usd=Decimal("0.01"),
                agent_name="agent-2",
            )

        async with db.session() as session:
            repo = CostRepository(session)
            costs = await repo.get_costs_by_agent()

            assert len(costs) == 2
            # Results ordered by total_cost desc
            assert costs[0]["agent_name"] == "agent-1"
            assert costs[1]["agent_name"] == "agent-2"

        await db.close()

    @pytest.mark.anyio
    async def test_get_costs_by_suite(self, temp_db_path: str) -> None:
        """Test getting costs aggregated by suite."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            repo = CostRepository(session)

            now = datetime.now()
            await repo.create_cost_record(
                timestamp=now,
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=Decimal("0.015"),
                suite_id="suite-1",
            )
            await repo.create_cost_record(
                timestamp=now,
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=500,
                output_tokens=200,
                cost_usd=Decimal("0.01"),
                suite_id="suite-2",
            )

        async with db.session() as session:
            repo = CostRepository(session)
            costs = await repo.get_costs_by_suite()

            assert len(costs) == 2
            # Results ordered by total_cost desc
            assert costs[0]["suite_id"] == "suite-1"
            assert costs[1]["suite_id"] == "suite-2"

        await db.close()

    @pytest.mark.anyio
    async def test_get_costs_by_day(self, temp_db_path: str) -> None:
        """Test getting costs aggregated by day."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            repo = CostRepository(session)

            now = datetime.now()
            yesterday = now - timedelta(days=1)

            await repo.create_cost_record(
                timestamp=now,
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=Decimal("0.015"),
            )
            await repo.create_cost_record(
                timestamp=yesterday,
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=500,
                output_tokens=200,
                cost_usd=Decimal("0.01"),
            )

        async with db.session() as session:
            repo = CostRepository(session)
            costs = await repo.get_costs_by_day()

            assert len(costs) == 2
            # Results ordered by date
            assert costs[0]["total_cost"] == Decimal("0.01")
            assert costs[1]["total_cost"] == Decimal("0.015")

        await db.close()


class TestBudgetUsage:
    """Tests for budget usage operations."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create temporary database path."""
        return f"sqlite+aiosqlite:///{tmp_path / 'test_analytics.db'}"

    @pytest.mark.anyio
    async def test_get_budget_usage_daily(self, temp_db_path: str) -> None:
        """Test getting budget usage for daily budget."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            repo = CostRepository(session)

            # Create budget
            budget = await repo.create_budget(
                name="daily-limit",
                period="daily",
                limit_usd=Decimal("100.00"),
                alert_threshold=0.8,
            )

            # Add some costs for today
            now = datetime.now()
            await repo.create_cost_record(
                timestamp=now,
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=Decimal("50.00"),
            )

        async with db.session() as session:
            repo = CostRepository(session)
            budget = await repo.get_budget_by_name("daily-limit")
            assert budget is not None
            usage = await repo.get_budget_usage(budget)

            assert usage["spent"] == Decimal("50.00")
            assert usage["limit"] == Decimal("100.00")
            assert usage["remaining"] == Decimal("50.00")
            assert usage["percentage"] == 0.5
            assert usage["is_over_threshold"] is False
            assert usage["is_over_limit"] is False

        await db.close()

    @pytest.mark.anyio
    async def test_get_budget_usage_over_threshold(self, temp_db_path: str) -> None:
        """Test budget usage when over alert threshold."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            repo = CostRepository(session)

            budget = await repo.create_budget(
                name="daily-limit",
                period="daily",
                limit_usd=Decimal("100.00"),
                alert_threshold=0.8,
            )

            now = datetime.now()
            await repo.create_cost_record(
                timestamp=now,
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=Decimal("85.00"),  # 85% of budget
            )

        async with db.session() as session:
            repo = CostRepository(session)
            budget = await repo.get_budget_by_name("daily-limit")
            assert budget is not None
            usage = await repo.get_budget_usage(budget)

            assert usage["percentage"] == 0.85
            assert usage["is_over_threshold"] is True
            assert usage["is_over_limit"] is False

        await db.close()

    @pytest.mark.anyio
    async def test_get_budget_usage_over_limit(self, temp_db_path: str) -> None:
        """Test budget usage when over limit."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            repo = CostRepository(session)

            budget = await repo.create_budget(
                name="daily-limit",
                period="daily",
                limit_usd=Decimal("100.00"),
            )

            now = datetime.now()
            await repo.create_cost_record(
                timestamp=now,
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=Decimal("120.00"),  # Over limit
            )

        async with db.session() as session:
            repo = CostRepository(session)
            budget = await repo.get_budget_by_name("daily-limit")
            assert budget is not None
            usage = await repo.get_budget_usage(budget)

            assert usage["spent"] == Decimal("120.00")
            assert usage["remaining"] == Decimal("0")
            assert usage["is_over_limit"] is True

        await db.close()

    @pytest.mark.anyio
    async def test_get_budget_usage_with_scope(self, temp_db_path: str) -> None:
        """Test budget usage with scope filter."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            repo = CostRepository(session)

            budget = await repo.create_budget(
                name="anthropic-budget",
                period="daily",
                limit_usd=Decimal("100.00"),
                scope={"provider": "anthropic"},
            )

            now = datetime.now()
            # Anthropic cost
            await repo.create_cost_record(
                timestamp=now,
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=Decimal("30.00"),
            )
            # OpenAI cost (should not count)
            await repo.create_cost_record(
                timestamp=now,
                provider="openai",
                model="gpt-4",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=Decimal("50.00"),
            )

        async with db.session() as session:
            repo = CostRepository(session)
            budget = await repo.get_budget_by_name("anthropic-budget")
            assert budget is not None
            usage = await repo.get_budget_usage(budget)

            # Only anthropic costs should be counted
            assert usage["spent"] == Decimal("30.00")

        await db.close()

    @pytest.mark.anyio
    async def test_check_all_budgets(self, temp_db_path: str) -> None:
        """Test checking all active budgets."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            repo = CostRepository(session)

            # Create multiple budgets
            await repo.create_budget(
                name="daily-limit",
                period="daily",
                limit_usd=Decimal("100.00"),
            )
            await repo.create_budget(
                name="weekly-limit",
                period="weekly",
                limit_usd=Decimal("500.00"),
            )
            # Create inactive budget
            inactive = await repo.create_budget(
                name="inactive-budget",
                period="monthly",
                limit_usd=Decimal("1000.00"),
            )
            await repo.update_budget(inactive, is_active=False)

        async with db.session() as session:
            repo = CostRepository(session)
            results = await repo.check_all_budgets()

            # Should only have 2 active budgets
            assert len(results) == 2
            budget_names = {r["budget_name"] for r in results}
            assert "daily-limit" in budget_names
            assert "weekly-limit" in budget_names
            assert "inactive-budget" not in budget_names

        await db.close()
