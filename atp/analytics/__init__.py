"""ATP Analytics module for cost tracking and budget management.

This module provides comprehensive cost tracking for all LLM operations,
with support for multiple providers (OpenAI, Anthropic, Google, Azure, Bedrock).

Usage:
    from atp.analytics import (
        CostRecord,
        CostBudget,
        CostRepository,
        AnalyticsDatabase,
        init_analytics_database,
        CostTracker,
        CostEvent,
        get_cost_tracker,
    )

    # Initialize the database
    db = await init_analytics_database()

    # Track costs using repository
    async with db.session() as session:
        repo = CostRepository(session)
        await repo.create_cost_record(
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=Decimal("0.015"),
        )

        # Get aggregated costs by provider
        costs = await repo.get_costs_by_provider(
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
        )

    # Or use the CostTracker for non-blocking cost tracking
    tracker = await get_cost_tracker()
    await tracker.track(CostEvent(
        timestamp=datetime.now(),
        provider="anthropic",
        model="claude-3-sonnet",
        input_tokens=1000,
        output_tokens=500,
    ))
"""

from atp.analytics.advanced import (
    AdvancedAnalyticsService,
    AnomalyDetectionResponse,
    AnomalyResult,
    AnomalyType,
    CorrelationAnalysisResponse,
    CorrelationResult,
    CorrelationStrength,
    ExportFormat,
    ScheduledReportConfig,
    ScheduledReportFrequency,
    ScheduledReportsRepository,
    ScoreTrendResponse,
    TrendAnalysisResult,
    TrendDataPoint,
    TrendDirection,
)
from atp.analytics.budgets import (
    AlertChannel,
    AlertChannelRegistry,
    AlertConfig,
    BudgetCheckResult,
    BudgetConfig,
    BudgetManager,
    BudgetPeriod,
    BudgetStatus,
    EmailAlertChannel,
    LogAlertChannel,
    WebhookAlertChannel,
    check_budget_for_cost,
    get_budget_manager,
    set_budget_manager,
)
from atp.analytics.cost import (
    CostEvent,
    CostTracker,
    ModelPricing,
    PricingConfig,
    get_cost_tracker,
    set_cost_tracker,
    shutdown_cost_tracker,
)
from atp.analytics.database import (
    AnalyticsDatabase,
    get_analytics_database,
    init_analytics_database,
    set_analytics_database,
)
from atp.analytics.models import AnalyticsBase, CostBudget, CostRecord, ScheduledReport
from atp.analytics.repository import CostRepository

__all__ = [
    # Database
    "AnalyticsBase",
    "AnalyticsDatabase",
    "get_analytics_database",
    "init_analytics_database",
    "set_analytics_database",
    # Models
    "CostBudget",
    "CostRecord",
    "ScheduledReport",
    # Repository
    "CostRepository",
    # Cost Tracking
    "CostEvent",
    "CostTracker",
    "ModelPricing",
    "PricingConfig",
    "get_cost_tracker",
    "set_cost_tracker",
    "shutdown_cost_tracker",
    # Budget Management
    "AlertChannel",
    "AlertChannelRegistry",
    "AlertConfig",
    "BudgetCheckResult",
    "BudgetConfig",
    "BudgetManager",
    "BudgetPeriod",
    "BudgetStatus",
    "EmailAlertChannel",
    "LogAlertChannel",
    "WebhookAlertChannel",
    "check_budget_for_cost",
    "get_budget_manager",
    "set_budget_manager",
    # Advanced Analytics
    "AdvancedAnalyticsService",
    "AnomalyDetectionResponse",
    "AnomalyResult",
    "AnomalyType",
    "CorrelationAnalysisResponse",
    "CorrelationResult",
    "CorrelationStrength",
    "ExportFormat",
    "ScheduledReportConfig",
    "ScheduledReportFrequency",
    "ScheduledReportsRepository",
    "ScoreTrendResponse",
    "TrendAnalysisResult",
    "TrendDataPoint",
    "TrendDirection",
]
