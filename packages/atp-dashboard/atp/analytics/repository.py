"""Repository for cost tracking CRUD operations and aggregations."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.analytics.models import CostBudget, CostRecord


class CostRepository:
    """Repository for cost tracking operations.

    Provides CRUD operations and aggregation queries for cost records
    and budget management.
    """

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session

    # ==================== CostRecord CRUD Operations ====================

    async def create_cost_record(
        self,
        *,
        timestamp: datetime,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Decimal,
        test_id: str | None = None,
        suite_id: str | None = None,
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CostRecord:
        """Create a new cost record.

        Args:
            timestamp: When the operation occurred.
            provider: LLM provider (anthropic, openai, google, azure, bedrock).
            model: Model name (claude-3-sonnet, gpt-4, etc.).
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cost_usd: Cost in USD.
            test_id: Optional test ID for association.
            suite_id: Optional suite ID for association.
            agent_name: Optional agent name for association.
            metadata: Optional additional metadata.

        Returns:
            Created CostRecord instance.
        """
        record = CostRecord(
            timestamp=timestamp,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            test_id=test_id,
            suite_id=suite_id,
            agent_name=agent_name,
            metadata_json=metadata,
        )
        self._session.add(record)
        await self._session.flush()
        return record

    async def create_cost_records_batch(
        self, records: list[CostRecord]
    ) -> list[CostRecord]:
        """Create multiple cost records in a batch.

        Args:
            records: List of CostRecord instances to create.

        Returns:
            List of created CostRecord instances.
        """
        for record in records:
            self._session.add(record)
        await self._session.flush()
        return records

    async def get_cost_record(self, record_id: int) -> CostRecord | None:
        """Get a cost record by ID.

        Args:
            record_id: Cost record ID.

        Returns:
            CostRecord if found, None otherwise.
        """
        stmt = select(CostRecord).where(CostRecord.id == record_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_cost_records(
        self,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        provider: str | None = None,
        model: str | None = None,
        agent_name: str | None = None,
        suite_id: str | None = None,
        test_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CostRecord]:
        """List cost records with optional filtering.

        Args:
            start_date: Filter by start date.
            end_date: Filter by end date.
            provider: Filter by provider.
            model: Filter by model.
            agent_name: Filter by agent name.
            suite_id: Filter by suite ID.
            test_id: Filter by test ID.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of cost records.
        """
        stmt = select(CostRecord).order_by(CostRecord.timestamp.desc())

        if start_date is not None:
            stmt = stmt.where(CostRecord.timestamp >= start_date)
        if end_date is not None:
            stmt = stmt.where(CostRecord.timestamp <= end_date)
        if provider is not None:
            stmt = stmt.where(CostRecord.provider == provider)
        if model is not None:
            stmt = stmt.where(CostRecord.model == model)
        if agent_name is not None:
            stmt = stmt.where(CostRecord.agent_name == agent_name)
        if suite_id is not None:
            stmt = stmt.where(CostRecord.suite_id == suite_id)
        if test_id is not None:
            stmt = stmt.where(CostRecord.test_id == test_id)

        stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def delete_cost_record(self, record_id: int) -> bool:
        """Delete a cost record by ID.

        Args:
            record_id: Cost record ID.

        Returns:
            True if deleted, False if not found.
        """
        record = await self.get_cost_record(record_id)
        if record is None:
            return False

        await self._session.delete(record)
        await self._session.flush()
        return True

    # ==================== CostBudget CRUD Operations ====================

    async def create_budget(
        self,
        *,
        name: str,
        period: str,
        limit_usd: Decimal,
        alert_threshold: float = 0.8,
        scope: dict[str, Any] | None = None,
        alert_channels: list[str] | None = None,
        description: str | None = None,
    ) -> CostBudget:
        """Create a new cost budget.

        Args:
            name: Budget name for identification.
            period: Budget period (daily, weekly, monthly).
            limit_usd: Budget limit in USD.
            alert_threshold: Alert threshold (0.0-1.0).
            scope: Optional scope filters (provider, model, agent, etc.).
            alert_channels: Optional alert channels (slack, email, pagerduty).
            description: Optional description.

        Returns:
            Created CostBudget instance.
        """
        budget = CostBudget(
            name=name,
            period=period,
            limit_usd=limit_usd,
            alert_threshold=alert_threshold,
            scope_json=scope,
            alert_channels_json=alert_channels,
            description=description,
        )
        self._session.add(budget)
        await self._session.flush()
        return budget

    async def get_budget(self, budget_id: int) -> CostBudget | None:
        """Get a budget by ID.

        Args:
            budget_id: Budget ID.

        Returns:
            CostBudget if found, None otherwise.
        """
        stmt = select(CostBudget).where(CostBudget.id == budget_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_budget_by_name(self, name: str) -> CostBudget | None:
        """Get a budget by name.

        Args:
            name: Budget name.

        Returns:
            CostBudget if found, None otherwise.
        """
        stmt = select(CostBudget).where(CostBudget.name == name)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_budgets(
        self,
        *,
        period: str | None = None,
        is_active: bool | None = None,
    ) -> list[CostBudget]:
        """List budgets with optional filtering.

        Args:
            period: Filter by period.
            is_active: Filter by active status.

        Returns:
            List of budgets.
        """
        stmt = select(CostBudget).order_by(CostBudget.name)

        if period is not None:
            stmt = stmt.where(CostBudget.period == period)
        if is_active is not None:
            stmt = stmt.where(CostBudget.is_active == is_active)

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_budget(
        self,
        budget: CostBudget,
        *,
        name: str | None = None,
        period: str | None = None,
        limit_usd: Decimal | None = None,
        alert_threshold: float | None = None,
        scope: dict[str, Any] | None = None,
        alert_channels: list[str] | None = None,
        description: str | None = None,
        is_active: bool | None = None,
    ) -> CostBudget:
        """Update a budget.

        Args:
            budget: Budget to update.
            name: New name.
            period: New period.
            limit_usd: New limit.
            alert_threshold: New alert threshold.
            scope: New scope filters.
            alert_channels: New alert channels.
            description: New description.
            is_active: New active status.

        Returns:
            Updated CostBudget instance.
        """
        if name is not None:
            budget.name = name
        if period is not None:
            budget.period = period
        if limit_usd is not None:
            budget.limit_usd = limit_usd
        if alert_threshold is not None:
            budget.alert_threshold = alert_threshold
        if scope is not None:
            budget.scope_json = scope
        if alert_channels is not None:
            budget.alert_channels_json = alert_channels
        if description is not None:
            budget.description = description
        if is_active is not None:
            budget.is_active = is_active

        await self._session.flush()
        return budget

    async def delete_budget(self, budget_id: int) -> bool:
        """Delete a budget by ID.

        Args:
            budget_id: Budget ID.

        Returns:
            True if deleted, False if not found.
        """
        budget = await self.get_budget(budget_id)
        if budget is None:
            return False

        await self._session.delete(budget)
        await self._session.flush()
        return True

    # ==================== Aggregation Queries ====================

    async def get_total_cost(
        self,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        provider: str | None = None,
        model: str | None = None,
        agent_name: str | None = None,
        suite_id: str | None = None,
    ) -> Decimal:
        """Get total cost for the specified filters.

        Args:
            start_date: Filter by start date.
            end_date: Filter by end date.
            provider: Filter by provider.
            model: Filter by model.
            agent_name: Filter by agent name.
            suite_id: Filter by suite ID.

        Returns:
            Total cost in USD.
        """
        stmt = select(func.coalesce(func.sum(CostRecord.cost_usd), Decimal("0")))

        if start_date is not None:
            stmt = stmt.where(CostRecord.timestamp >= start_date)
        if end_date is not None:
            stmt = stmt.where(CostRecord.timestamp <= end_date)
        if provider is not None:
            stmt = stmt.where(CostRecord.provider == provider)
        if model is not None:
            stmt = stmt.where(CostRecord.model == model)
        if agent_name is not None:
            stmt = stmt.where(CostRecord.agent_name == agent_name)
        if suite_id is not None:
            stmt = stmt.where(CostRecord.suite_id == suite_id)

        result = await self._session.execute(stmt)
        return result.scalar() or Decimal("0")

    async def get_costs_by_day(
        self,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        provider: str | None = None,
        model: str | None = None,
        agent_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated costs by day.

        Args:
            start_date: Filter by start date.
            end_date: Filter by end date.
            provider: Filter by provider.
            model: Filter by model.
            agent_name: Filter by agent name.

        Returns:
            List of dicts with date, total_cost, total_tokens, record_count.
        """
        # Use date truncation for grouping
        date_trunc = func.date(CostRecord.timestamp)

        stmt = (
            select(
                date_trunc.label("date"),
                func.sum(CostRecord.cost_usd).label("total_cost"),
                func.sum(CostRecord.input_tokens + CostRecord.output_tokens).label(
                    "total_tokens"
                ),
                func.count(CostRecord.id).label("record_count"),
            )
            .group_by(date_trunc)
            .order_by(date_trunc)
        )

        if start_date is not None:
            stmt = stmt.where(CostRecord.timestamp >= start_date)
        if end_date is not None:
            stmt = stmt.where(CostRecord.timestamp <= end_date)
        if provider is not None:
            stmt = stmt.where(CostRecord.provider == provider)
        if model is not None:
            stmt = stmt.where(CostRecord.model == model)
        if agent_name is not None:
            stmt = stmt.where(CostRecord.agent_name == agent_name)

        result = await self._session.execute(stmt)
        return [
            {
                "date": row.date,
                "total_cost": row.total_cost,
                "total_tokens": row.total_tokens,
                "record_count": row.record_count,
            }
            for row in result.all()
        ]

    async def get_costs_by_provider(
        self,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated costs by provider.

        Args:
            start_date: Filter by start date.
            end_date: Filter by end date.

        Returns:
            List of dicts with provider, total_cost, total_tokens, record_count.
        """
        stmt = (
            select(
                CostRecord.provider,
                func.sum(CostRecord.cost_usd).label("total_cost"),
                func.sum(CostRecord.input_tokens).label("total_input_tokens"),
                func.sum(CostRecord.output_tokens).label("total_output_tokens"),
                func.count(CostRecord.id).label("record_count"),
            )
            .group_by(CostRecord.provider)
            .order_by(func.sum(CostRecord.cost_usd).desc())
        )

        if start_date is not None:
            stmt = stmt.where(CostRecord.timestamp >= start_date)
        if end_date is not None:
            stmt = stmt.where(CostRecord.timestamp <= end_date)

        result = await self._session.execute(stmt)
        return [
            {
                "provider": row.provider,
                "total_cost": row.total_cost,
                "total_input_tokens": row.total_input_tokens,
                "total_output_tokens": row.total_output_tokens,
                "record_count": row.record_count,
            }
            for row in result.all()
        ]

    async def get_costs_by_model(
        self,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        provider: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated costs by model.

        Args:
            start_date: Filter by start date.
            end_date: Filter by end date.
            provider: Filter by provider.

        Returns:
            List of dicts with provider, model, total_cost, total_tokens.
        """
        stmt = (
            select(
                CostRecord.provider,
                CostRecord.model,
                func.sum(CostRecord.cost_usd).label("total_cost"),
                func.sum(CostRecord.input_tokens).label("total_input_tokens"),
                func.sum(CostRecord.output_tokens).label("total_output_tokens"),
                func.count(CostRecord.id).label("record_count"),
            )
            .group_by(CostRecord.provider, CostRecord.model)
            .order_by(func.sum(CostRecord.cost_usd).desc())
        )

        if start_date is not None:
            stmt = stmt.where(CostRecord.timestamp >= start_date)
        if end_date is not None:
            stmt = stmt.where(CostRecord.timestamp <= end_date)
        if provider is not None:
            stmt = stmt.where(CostRecord.provider == provider)

        result = await self._session.execute(stmt)
        return [
            {
                "provider": row.provider,
                "model": row.model,
                "total_cost": row.total_cost,
                "total_input_tokens": row.total_input_tokens,
                "total_output_tokens": row.total_output_tokens,
                "record_count": row.record_count,
            }
            for row in result.all()
        ]

    async def get_costs_by_agent(
        self,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated costs by agent.

        Args:
            start_date: Filter by start date.
            end_date: Filter by end date.

        Returns:
            List of dicts with agent_name, total_cost, total_tokens.
        """
        stmt = (
            select(
                CostRecord.agent_name,
                func.sum(CostRecord.cost_usd).label("total_cost"),
                func.sum(CostRecord.input_tokens).label("total_input_tokens"),
                func.sum(CostRecord.output_tokens).label("total_output_tokens"),
                func.count(CostRecord.id).label("record_count"),
            )
            .where(CostRecord.agent_name.isnot(None))
            .group_by(CostRecord.agent_name)
            .order_by(func.sum(CostRecord.cost_usd).desc())
        )

        if start_date is not None:
            stmt = stmt.where(CostRecord.timestamp >= start_date)
        if end_date is not None:
            stmt = stmt.where(CostRecord.timestamp <= end_date)

        result = await self._session.execute(stmt)
        return [
            {
                "agent_name": row.agent_name,
                "total_cost": row.total_cost,
                "total_input_tokens": row.total_input_tokens,
                "total_output_tokens": row.total_output_tokens,
                "record_count": row.record_count,
            }
            for row in result.all()
        ]

    async def get_costs_by_suite(
        self,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated costs by suite.

        Args:
            start_date: Filter by start date.
            end_date: Filter by end date.

        Returns:
            List of dicts with suite_id, total_cost, total_tokens.
        """
        stmt = (
            select(
                CostRecord.suite_id,
                func.sum(CostRecord.cost_usd).label("total_cost"),
                func.sum(CostRecord.input_tokens).label("total_input_tokens"),
                func.sum(CostRecord.output_tokens).label("total_output_tokens"),
                func.count(CostRecord.id).label("record_count"),
            )
            .where(CostRecord.suite_id.isnot(None))
            .group_by(CostRecord.suite_id)
            .order_by(func.sum(CostRecord.cost_usd).desc())
        )

        if start_date is not None:
            stmt = stmt.where(CostRecord.timestamp >= start_date)
        if end_date is not None:
            stmt = stmt.where(CostRecord.timestamp <= end_date)

        result = await self._session.execute(stmt)
        return [
            {
                "suite_id": row.suite_id,
                "total_cost": row.total_cost,
                "total_input_tokens": row.total_input_tokens,
                "total_output_tokens": row.total_output_tokens,
                "record_count": row.record_count,
            }
            for row in result.all()
        ]

    # ==================== Budget Check Operations ====================

    async def get_budget_usage(
        self, budget: CostBudget, reference_date: datetime | None = None
    ) -> dict[str, Any]:
        """Calculate current usage for a budget.

        Args:
            budget: Budget to check.
            reference_date: Reference date for period calculation.

        Returns:
            Dict with spent, limit, remaining, percentage, is_over_threshold.
        """
        if reference_date is None:
            reference_date = datetime.now()

        # Calculate period start based on budget period
        if budget.period == "daily":
            period_start = reference_date.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif budget.period == "weekly":
            days_since_monday = reference_date.weekday()
            period_start = (reference_date - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif budget.period == "monthly":
            period_start = reference_date.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
        else:
            period_start = reference_date.replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        # Apply scope filters if defined
        kwargs: dict[str, Any] = {
            "start_date": period_start,
            "end_date": reference_date,
        }
        if budget.scope:
            if "provider" in budget.scope:
                kwargs["provider"] = budget.scope["provider"]
            if "model" in budget.scope:
                kwargs["model"] = budget.scope["model"]
            if "agent_name" in budget.scope:
                kwargs["agent_name"] = budget.scope["agent_name"]
            if "suite_id" in budget.scope:
                kwargs["suite_id"] = budget.scope["suite_id"]

        spent = await self.get_total_cost(**kwargs)
        limit = budget.limit_usd
        remaining = max(Decimal("0"), limit - spent)
        percentage = float(spent / limit) if limit > 0 else 0.0

        return {
            "budget_id": budget.id,
            "budget_name": budget.name,
            "period": budget.period,
            "period_start": period_start,
            "spent": spent,
            "limit": limit,
            "remaining": remaining,
            "percentage": percentage,
            "is_over_threshold": percentage >= budget.alert_threshold,
            "is_over_limit": spent >= limit,
        }

    async def check_all_budgets(
        self, reference_date: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Check usage for all active budgets.

        Args:
            reference_date: Reference date for period calculation.

        Returns:
            List of budget usage dicts.
        """
        budgets = await self.list_budgets(is_active=True)
        results = []

        for budget in budgets:
            usage = await self.get_budget_usage(budget, reference_date)
            results.append(usage)

        return results
