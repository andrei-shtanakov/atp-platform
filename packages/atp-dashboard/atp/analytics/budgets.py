"""Cost budget management and alert system for ATP Platform.

This module provides budget definition, tracking, and alerting for LLM costs.
Supports daily, weekly, and monthly budgets with configurable alert thresholds
and pluggable alert channels.

Features:
- Budget periods: daily, weekly, monthly
- Configurable alert thresholds (e.g., 80%, 100%)
- Pluggable alert channels: log, webhook, email
- Scope-based budgets (by provider, model, agent, suite)
- Integration with cost tracking system

Example usage:
    from atp.analytics.budgets import (
        BudgetManager,
        BudgetConfig,
        AlertConfig,
        get_budget_manager,
    )

    # Configure budgets
    config = BudgetConfig(
        daily=Decimal("100.00"),
        monthly=Decimal("2000.00"),
        alerts=[
            AlertConfig(threshold=0.8, channels=["log", "webhook"]),
            AlertConfig(threshold=1.0, channels=["log", "email"]),
        ],
    )

    # Get manager and check budgets
    manager = await get_budget_manager()
    status = await manager.check_budgets()
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from atp.analytics.database import get_analytics_database
from atp.analytics.models import CostBudget
from atp.analytics.repository import CostRepository

logger = logging.getLogger(__name__)


class BudgetPeriod(str, Enum):
    """Budget period types."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class AlertConfig:
    """Configuration for a budget alert.

    Attributes:
        threshold: Alert threshold as decimal (0.0-1.0, e.g., 0.8 = 80%).
        channels: List of alert channel names to notify.
    """

    threshold: float
    channels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate threshold is in valid range."""
        if not 0.0 <= self.threshold <= 2.0:
            raise ValueError(
                f"Threshold must be between 0.0 and 2.0, got {self.threshold}"
            )


@dataclass
class BudgetConfig:
    """Budget configuration for cost tracking.

    Attributes:
        daily: Daily budget limit in USD.
        weekly: Weekly budget limit in USD.
        monthly: Monthly budget limit in USD.
        alerts: List of alert configurations.
        scope: Optional scope filters (provider, model, agent_name, suite_id).
    """

    daily: Decimal | None = None
    weekly: Decimal | None = None
    monthly: Decimal | None = None
    alerts: list[AlertConfig] = field(default_factory=list)
    scope: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetConfig:
        """Create BudgetConfig from dictionary.

        Args:
            data: Dictionary with budget configuration.

        Returns:
            BudgetConfig instance.
        """
        alerts = []
        if "alerts" in data:
            for alert_data in data["alerts"]:
                alerts.append(
                    AlertConfig(
                        threshold=float(alert_data.get("threshold", 0.8)),
                        channels=alert_data.get("channels", ["log"]),
                    )
                )

        return cls(
            daily=Decimal(str(data["daily"])) if "daily" in data else None,
            weekly=Decimal(str(data["weekly"])) if "weekly" in data else None,
            monthly=Decimal(str(data["monthly"])) if "monthly" in data else None,
            alerts=alerts,
            scope=data.get("scope"),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> BudgetConfig:
        """Load budget configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            BudgetConfig instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If configuration is invalid.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Budget configuration must be a dictionary")

        # Support nested 'cost.budgets' structure or flat structure
        if "cost" in data and "budgets" in data["cost"]:
            budget_data = data["cost"]["budgets"]
            alert_data = data["cost"].get("alerts", [])
        elif "budgets" in data:
            budget_data = data["budgets"]
            alert_data = data.get("alerts", [])
        else:
            budget_data = data
            alert_data = data.get("alerts", [])

        combined = {**budget_data, "alerts": alert_data}
        return cls.from_dict(combined)


@dataclass
class BudgetStatus:
    """Status of a single budget.

    Attributes:
        budget_id: Budget ID from database.
        budget_name: Budget name.
        period: Budget period (daily, weekly, monthly).
        period_start: Start of the current period.
        limit: Budget limit in USD.
        spent: Amount spent in current period in USD.
        remaining: Remaining budget in USD.
        percentage: Percentage of budget used (0-100+).
        is_over_threshold: Whether any alert threshold has been exceeded.
        is_over_limit: Whether the budget limit has been exceeded.
        triggered_alerts: List of triggered alert thresholds.
    """

    budget_id: int | None
    budget_name: str
    period: BudgetPeriod
    period_start: datetime
    limit: Decimal
    spent: Decimal
    remaining: Decimal
    percentage: float
    is_over_threshold: bool
    is_over_limit: bool
    triggered_alerts: list[float] = field(default_factory=list)


@dataclass
class BudgetCheckResult:
    """Result of checking all budgets.

    Attributes:
        timestamp: When the check was performed.
        statuses: List of budget status objects.
        has_alerts: Whether any budget has triggered an alert.
        has_exceeded: Whether any budget has been exceeded.
    """

    timestamp: datetime
    statuses: list[BudgetStatus]
    has_alerts: bool = False
    has_exceeded: bool = False


class AlertChannel(ABC):
    """Abstract base class for alert channels.

    Implement this interface to create custom alert channels.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the channel name."""

    @abstractmethod
    async def send_alert(
        self,
        status: BudgetStatus,
        threshold: float,
        message: str,
    ) -> bool:
        """Send an alert through this channel.

        Args:
            status: Budget status that triggered the alert.
            threshold: The threshold that was exceeded.
            message: Alert message to send.

        Returns:
            True if alert was sent successfully.
        """


class LogAlertChannel(AlertChannel):
    """Alert channel that logs to the application logger."""

    @property
    def name(self) -> str:
        return "log"

    async def send_alert(
        self,
        status: BudgetStatus,
        threshold: float,
        message: str,
    ) -> bool:
        """Log the alert message."""
        level = logging.WARNING if not status.is_over_limit else logging.ERROR
        logger.log(
            level,
            f"[BUDGET ALERT] {message} "
            f"(Budget: {status.budget_name}, Period: {status.period.value}, "
            f"Spent: ${status.spent:.2f}, Limit: ${status.limit:.2f}, "
            f"Usage: {status.percentage:.1f}%)",
        )
        return True


class WebhookAlertChannel(AlertChannel):
    """Alert channel that sends HTTP webhooks."""

    def __init__(
        self,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ):
        """Initialize webhook channel.

        Args:
            url: Webhook URL to send alerts to.
            headers: Optional HTTP headers.
            timeout: Request timeout in seconds.
        """
        self._url = url
        self._headers = headers or {"Content-Type": "application/json"}
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "webhook"

    @property
    def url(self) -> str | None:
        return self._url

    @url.setter
    def url(self, value: str) -> None:
        self._url = value

    async def send_alert(
        self,
        status: BudgetStatus,
        threshold: float,
        message: str,
    ) -> bool:
        """Send alert to webhook URL."""
        if not self._url:
            logger.warning("Webhook URL not configured, skipping alert")
            return False

        try:
            import httpx

            payload = {
                "type": "budget_alert",
                "timestamp": datetime.now().isoformat(),
                "budget": {
                    "id": status.budget_id,
                    "name": status.budget_name,
                    "period": status.period.value,
                    "period_start": status.period_start.isoformat(),
                    "limit_usd": str(status.limit),
                    "spent_usd": str(status.spent),
                    "remaining_usd": str(status.remaining),
                    "percentage": status.percentage,
                },
                "alert": {
                    "threshold": threshold,
                    "is_over_limit": status.is_over_limit,
                    "message": message,
                },
            }

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._url,
                    json=payload,
                    headers=self._headers,
                )
                response.raise_for_status()
                logger.info(f"Webhook alert sent to {self._url}")
                return True

        except ImportError:
            logger.error("httpx not installed, cannot send webhook alerts")
            return False
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class EmailAlertChannel(AlertChannel):
    """Alert channel that sends emails via SMTP."""

    def __init__(
        self,
        smtp_host: str | None = None,
        smtp_port: int = 587,
        smtp_user: str | None = None,
        smtp_password: str | None = None,
        from_addr: str | None = None,
        to_addrs: list[str] | None = None,
        use_tls: bool = True,
    ):
        """Initialize email channel.

        Args:
            smtp_host: SMTP server hostname.
            smtp_port: SMTP server port.
            smtp_user: SMTP username.
            smtp_password: SMTP password.
            from_addr: Sender email address.
            to_addrs: List of recipient email addresses.
            use_tls: Whether to use STARTTLS.
        """
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._smtp_user = smtp_user
        self._smtp_password = smtp_password
        self._from_addr = from_addr
        self._to_addrs = to_addrs or []
        self._use_tls = use_tls

    @property
    def name(self) -> str:
        return "email"

    async def send_alert(
        self,
        status: BudgetStatus,
        threshold: float,
        message: str,
    ) -> bool:
        """Send alert via email."""
        if not self._smtp_host or not self._to_addrs:
            logger.warning("Email not configured, skipping alert")
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText

            subject = (
                f"[ATP Budget Alert] {status.budget_name} - "
                f"{status.percentage:.1f}% used"
            )

            body = f"""ATP Budget Alert

Budget: {status.budget_name}
Period: {status.period.value}
Period Start: {status.period_start.isoformat()}

Spent: ${status.spent:.2f}
Limit: ${status.limit:.2f}
Remaining: ${status.remaining:.2f}
Usage: {status.percentage:.1f}%

Alert Threshold: {threshold * 100:.0f}%
Status: {"EXCEEDED" if status.is_over_limit else "WARNING"}

Message: {message}

--
ATP Agent Test Platform
"""

            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self._from_addr or self._smtp_user or "atp@localhost"
            msg["To"] = ", ".join(self._to_addrs)

            # Run SMTP in thread pool to avoid blocking
            smtp_host = self._smtp_host  # Capture for closure
            smtp_port = self._smtp_port

            def _send_email() -> bool:
                assert smtp_host is not None
                try:
                    with smtplib.SMTP(smtp_host, smtp_port) as server:
                        if self._use_tls:
                            server.starttls()
                        if self._smtp_user and self._smtp_password:
                            server.login(self._smtp_user, self._smtp_password)
                        server.sendmail(
                            msg["From"],
                            self._to_addrs,
                            msg.as_string(),
                        )
                    return True
                except Exception as e:
                    logger.error(f"SMTP error: {e}")
                    return False

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, _send_email)
            if result:
                logger.info(f"Email alert sent to {', '.join(self._to_addrs)}")
            return result

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class AlertChannelRegistry:
    """Registry for alert channels."""

    def __init__(self) -> None:
        """Initialize registry with default channels."""
        self._channels: dict[str, AlertChannel] = {}
        # Register default channels
        self.register(LogAlertChannel())
        self.register(WebhookAlertChannel())
        self.register(EmailAlertChannel())

    def register(self, channel: AlertChannel) -> None:
        """Register an alert channel.

        Args:
            channel: AlertChannel instance to register.
        """
        self._channels[channel.name] = channel

    def get(self, name: str) -> AlertChannel | None:
        """Get a channel by name.

        Args:
            name: Channel name.

        Returns:
            AlertChannel if found, None otherwise.
        """
        return self._channels.get(name)

    def list_channels(self) -> list[str]:
        """List all registered channel names."""
        return list(self._channels.keys())

    def configure_webhook(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Configure the webhook channel.

        Args:
            url: Webhook URL.
            headers: Optional HTTP headers.
        """
        channel = self.get("webhook")
        if isinstance(channel, WebhookAlertChannel):
            channel.url = url
            if headers:
                channel._headers = headers

    def configure_email(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        smtp_user: str | None = None,
        smtp_password: str | None = None,
        from_addr: str | None = None,
        to_addrs: list[str] | None = None,
        use_tls: bool = True,
    ) -> None:
        """Configure the email channel.

        Args:
            smtp_host: SMTP server hostname.
            smtp_port: SMTP server port.
            smtp_user: SMTP username.
            smtp_password: SMTP password.
            from_addr: Sender email address.
            to_addrs: List of recipient email addresses.
            use_tls: Whether to use STARTTLS.
        """
        self._channels["email"] = EmailAlertChannel(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            from_addr=from_addr,
            to_addrs=to_addrs,
            use_tls=use_tls,
        )


class BudgetManager:
    """Manager for cost budgets and alerts.

    Provides methods to check budget status, send alerts, and manage budgets.

    Example:
        manager = BudgetManager()
        await manager.initialize(config)

        # Check all budgets and send alerts
        result = await manager.check_and_alert()

        # Get status without sending alerts
        result = await manager.check_budgets()
    """

    def __init__(
        self,
        channel_registry: AlertChannelRegistry | None = None,
    ):
        """Initialize budget manager.

        Args:
            channel_registry: Optional custom alert channel registry.
        """
        self._channel_registry = channel_registry or AlertChannelRegistry()
        self._config: BudgetConfig | None = None
        self._alert_history: dict[str, datetime] = {}
        self._alert_cooldown = timedelta(hours=1)

    @property
    def channel_registry(self) -> AlertChannelRegistry:
        """Get the alert channel registry."""
        return self._channel_registry

    async def initialize(self, config: BudgetConfig) -> None:
        """Initialize manager with budget configuration.

        Creates budget entries in the database if they don't exist.

        Args:
            config: Budget configuration.
        """
        self._config = config
        db = get_analytics_database()

        async with db.session() as session:
            repo = CostRepository(session)

            # Create or update budgets based on config
            for period in BudgetPeriod:
                limit = getattr(config, period.value, None)
                if limit is None:
                    continue

                budget_name = f"default_{period.value}"
                existing = await repo.get_budget_by_name(budget_name)

                if existing:
                    await repo.update_budget(
                        existing,
                        limit_usd=limit,
                        scope=config.scope,
                        is_active=True,
                    )
                else:
                    alert_channels = []
                    if config.alerts:
                        for alert in config.alerts:
                            alert_channels.extend(alert.channels)
                    alert_channels = list(set(alert_channels))

                    await repo.create_budget(
                        name=budget_name,
                        period=period.value,
                        limit_usd=limit,
                        alert_threshold=0.8,
                        scope=config.scope,
                        alert_channels=alert_channels,
                        description=f"Default {period.value} budget",
                    )

            await session.commit()

    async def check_budgets(
        self,
        reference_date: datetime | None = None,
    ) -> BudgetCheckResult:
        """Check the status of all active budgets.

        Args:
            reference_date: Reference date for period calculation.

        Returns:
            BudgetCheckResult with status of all budgets.
        """
        if reference_date is None:
            reference_date = datetime.now()

        db = get_analytics_database()
        statuses: list[BudgetStatus] = []
        has_alerts = False
        has_exceeded = False

        async with db.session() as session:
            repo = CostRepository(session)
            budget_usages = await repo.check_all_budgets(reference_date)

            for usage in budget_usages:
                triggered_alerts: list[float] = []

                # Check against configured alerts
                if self._config and self._config.alerts:
                    for alert in self._config.alerts:
                        if usage["percentage"] >= alert.threshold:
                            triggered_alerts.append(alert.threshold)
                            has_alerts = True

                if usage["is_over_limit"]:
                    has_exceeded = True

                status = BudgetStatus(
                    budget_id=usage["budget_id"],
                    budget_name=usage["budget_name"],
                    period=BudgetPeriod(usage["period"]),
                    period_start=usage["period_start"],
                    limit=usage["limit"],
                    spent=usage["spent"],
                    remaining=usage["remaining"],
                    percentage=usage["percentage"] * 100,
                    is_over_threshold=usage["is_over_threshold"],
                    is_over_limit=usage["is_over_limit"],
                    triggered_alerts=triggered_alerts,
                )
                statuses.append(status)

        return BudgetCheckResult(
            timestamp=reference_date,
            statuses=statuses,
            has_alerts=has_alerts,
            has_exceeded=has_exceeded,
        )

    async def check_and_alert(
        self,
        reference_date: datetime | None = None,
    ) -> BudgetCheckResult:
        """Check budgets and send alerts for threshold violations.

        Respects alert cooldown to avoid spamming.

        Args:
            reference_date: Reference date for period calculation.

        Returns:
            BudgetCheckResult with status of all budgets.
        """
        result = await self.check_budgets(reference_date)

        if not self._config or not self._config.alerts:
            return result

        for status in result.statuses:
            for threshold in status.triggered_alerts:
                await self._send_alerts(status, threshold)

        return result

    async def _send_alerts(
        self,
        status: BudgetStatus,
        threshold: float,
    ) -> None:
        """Send alerts for a budget threshold violation.

        Args:
            status: Budget status that triggered the alert.
            threshold: The threshold that was exceeded.
        """
        # Check cooldown
        alert_key = f"{status.budget_name}_{threshold}"
        now = datetime.now()
        last_alert = self._alert_history.get(alert_key)

        if last_alert and now - last_alert < self._alert_cooldown:
            logger.debug(
                f"Skipping alert for {alert_key} due to cooldown "
                f"(last alert: {last_alert.isoformat()})"
            )
            return

        # Find alert config for this threshold
        if not self._config:
            return

        channels: list[str] = []
        for alert in self._config.alerts:
            if alert.threshold == threshold:
                channels = alert.channels
                break

        if not channels:
            return

        # Generate message
        if status.is_over_limit:
            message = f"Budget {status.budget_name} has EXCEEDED its limit"
        else:
            message = (
                f"Budget {status.budget_name} has reached "
                f"{status.percentage:.1f}% of its limit"
            )

        # Send to each channel
        for channel_name in channels:
            channel = self._channel_registry.get(channel_name)
            if channel:
                try:
                    await channel.send_alert(status, threshold, message)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel_name}: {e}")

        # Update alert history
        self._alert_history[alert_key] = now

    async def get_budget_status(
        self,
        budget_name: str,
        reference_date: datetime | None = None,
    ) -> BudgetStatus | None:
        """Get the status of a specific budget.

        Args:
            budget_name: Name of the budget.
            reference_date: Reference date for period calculation.

        Returns:
            BudgetStatus if found, None otherwise.
        """
        result = await self.check_budgets(reference_date)
        for status in result.statuses:
            if status.budget_name == budget_name:
                return status
        return None

    async def create_budget(
        self,
        name: str,
        period: BudgetPeriod,
        limit_usd: Decimal,
        alert_threshold: float = 0.8,
        scope: dict[str, Any] | None = None,
        alert_channels: list[str] | None = None,
        description: str | None = None,
    ) -> CostBudget:
        """Create a new budget.

        Args:
            name: Budget name.
            period: Budget period.
            limit_usd: Budget limit in USD.
            alert_threshold: Alert threshold (0.0-1.0).
            scope: Optional scope filters.
            alert_channels: Alert channels to notify.
            description: Budget description.

        Returns:
            Created CostBudget instance.
        """
        db = get_analytics_database()

        async with db.session() as session:
            repo = CostRepository(session)
            budget = await repo.create_budget(
                name=name,
                period=period.value,
                limit_usd=limit_usd,
                alert_threshold=alert_threshold,
                scope=scope,
                alert_channels=alert_channels,
                description=description,
            )
            await session.commit()
            return budget

    async def update_budget(
        self,
        name: str,
        limit_usd: Decimal | None = None,
        alert_threshold: float | None = None,
        scope: dict[str, Any] | None = None,
        alert_channels: list[str] | None = None,
        is_active: bool | None = None,
    ) -> CostBudget | None:
        """Update an existing budget.

        Args:
            name: Budget name.
            limit_usd: New limit in USD.
            alert_threshold: New alert threshold.
            scope: New scope filters.
            alert_channels: New alert channels.
            is_active: New active status.

        Returns:
            Updated CostBudget if found, None otherwise.
        """
        db = get_analytics_database()

        async with db.session() as session:
            repo = CostRepository(session)
            budget = await repo.get_budget_by_name(name)
            if not budget:
                return None

            budget = await repo.update_budget(
                budget,
                limit_usd=limit_usd,
                alert_threshold=alert_threshold,
                scope=scope,
                alert_channels=alert_channels,
                is_active=is_active,
            )
            await session.commit()
            return budget

    async def delete_budget(self, name: str) -> bool:
        """Delete a budget by name.

        Args:
            name: Budget name.

        Returns:
            True if deleted, False if not found.
        """
        db = get_analytics_database()

        async with db.session() as session:
            repo = CostRepository(session)
            budget = await repo.get_budget_by_name(name)
            if not budget:
                return False

            await repo.delete_budget(budget.id)
            await session.commit()
            return True

    async def list_budgets(
        self,
        period: BudgetPeriod | None = None,
        is_active: bool | None = None,
    ) -> list[CostBudget]:
        """List all budgets.

        Args:
            period: Filter by period.
            is_active: Filter by active status.

        Returns:
            List of CostBudget instances.
        """
        db = get_analytics_database()

        async with db.session() as session:
            repo = CostRepository(session)
            return await repo.list_budgets(
                period=period.value if period else None,
                is_active=is_active,
            )


# Global budget manager instance
_budget_manager: BudgetManager | None = None


async def get_budget_manager() -> BudgetManager:
    """Get the global budget manager instance.

    Creates manager if not already initialized.

    Returns:
        BudgetManager instance.
    """
    global _budget_manager
    if _budget_manager is None:
        _budget_manager = BudgetManager()
    return _budget_manager


def set_budget_manager(manager: BudgetManager) -> None:
    """Set the global budget manager instance.

    Useful for testing.

    Args:
        manager: BudgetManager instance to use.
    """
    global _budget_manager
    _budget_manager = manager


async def check_budget_for_cost(
    cost_usd: Decimal,
    provider: str | None = None,
    model: str | None = None,
    agent_name: str | None = None,
    suite_id: str | None = None,
) -> BudgetCheckResult:
    """Check budgets after tracking a cost.

    This is a convenience function for integration with the cost tracker.
    It checks all applicable budgets and returns the result.

    Args:
        cost_usd: The cost that was just tracked.
        provider: Optional provider filter.
        model: Optional model filter.
        agent_name: Optional agent name filter.
        suite_id: Optional suite ID filter.

    Returns:
        BudgetCheckResult with current budget status.
    """
    manager = await get_budget_manager()
    return await manager.check_and_alert()
