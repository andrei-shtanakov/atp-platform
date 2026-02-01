"""Tests for ATP Analytics Budget management system."""

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.analytics.budgets import (
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


class TestAlertConfig:
    """Tests for AlertConfig dataclass."""

    def test_create_alert_config(self) -> None:
        """Test creating an alert config."""
        config = AlertConfig(threshold=0.8, channels=["log", "webhook"])

        assert config.threshold == 0.8
        assert config.channels == ["log", "webhook"]

    def test_default_channels(self) -> None:
        """Test default channels is empty list."""
        config = AlertConfig(threshold=0.9)

        assert config.channels == []

    def test_invalid_threshold_too_high(self) -> None:
        """Test that threshold > 2.0 raises ValueError."""
        with pytest.raises(ValueError, match="Threshold must be between"):
            AlertConfig(threshold=2.5)

    def test_invalid_threshold_negative(self) -> None:
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="Threshold must be between"):
            AlertConfig(threshold=-0.1)

    def test_valid_threshold_at_boundary(self) -> None:
        """Test that threshold at boundaries is valid."""
        config_low = AlertConfig(threshold=0.0)
        config_high = AlertConfig(threshold=2.0)

        assert config_low.threshold == 0.0
        assert config_high.threshold == 2.0


class TestBudgetConfig:
    """Tests for BudgetConfig class."""

    def test_create_budget_config(self) -> None:
        """Test creating a budget config."""
        config = BudgetConfig(
            daily=Decimal("100.00"),
            weekly=Decimal("500.00"),
            monthly=Decimal("2000.00"),
        )

        assert config.daily == Decimal("100.00")
        assert config.weekly == Decimal("500.00")
        assert config.monthly == Decimal("2000.00")

    def test_config_with_alerts(self) -> None:
        """Test creating config with alerts."""
        config = BudgetConfig(
            daily=Decimal("100.00"),
            alerts=[
                AlertConfig(threshold=0.8, channels=["log"]),
                AlertConfig(threshold=1.0, channels=["log", "email"]),
            ],
        )

        assert len(config.alerts) == 2
        assert config.alerts[0].threshold == 0.8
        assert config.alerts[1].channels == ["log", "email"]

    def test_config_with_scope(self) -> None:
        """Test creating config with scope filters."""
        config = BudgetConfig(
            daily=Decimal("50.00"),
            scope={"provider": "anthropic", "model": "claude-3-sonnet"},
        )

        assert config.scope == {"provider": "anthropic", "model": "claude-3-sonnet"}

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        data = {
            "daily": 100.00,
            "monthly": 2000.00,
            "alerts": [
                {"threshold": 0.8, "channels": ["log", "webhook"]},
                {"threshold": 1.0, "channels": ["email"]},
            ],
            "scope": {"provider": "openai"},
        }

        config = BudgetConfig.from_dict(data)

        assert config.daily == Decimal("100.00")
        assert config.monthly == Decimal("2000.00")
        assert config.weekly is None
        assert len(config.alerts) == 2
        assert config.scope == {"provider": "openai"}

    def test_from_dict_minimal(self) -> None:
        """Test creating config from minimal dictionary."""
        data = {"daily": 50.00}

        config = BudgetConfig.from_dict(data)

        assert config.daily == Decimal("50.00")
        assert config.weekly is None
        assert config.monthly is None
        assert config.alerts == []
        assert config.scope is None

    def test_from_yaml(self, tmp_path: Path) -> None:
        """Test loading config from YAML file."""
        yaml_content = """
cost:
  budgets:
    daily: 100.00
    monthly: 2000.00
  alerts:
    - threshold: 0.8
      channels: ["log", "webhook"]
    - threshold: 1.0
      channels: ["log", "email"]
"""
        yaml_path = tmp_path / "cost.yaml"
        yaml_path.write_text(yaml_content)

        config = BudgetConfig.from_yaml(yaml_path)

        assert config.daily == Decimal("100.00")
        assert config.monthly == Decimal("2000.00")
        assert len(config.alerts) == 2

    def test_from_yaml_flat_structure(self, tmp_path: Path) -> None:
        """Test loading config from flat YAML structure."""
        yaml_content = """
budgets:
  daily: 50.00
  weekly: 250.00
alerts:
  - threshold: 0.9
    channels: ["log"]
"""
        yaml_path = tmp_path / "cost.yaml"
        yaml_path.write_text(yaml_content)

        config = BudgetConfig.from_yaml(yaml_path)

        assert config.daily == Decimal("50.00")
        assert config.weekly == Decimal("250.00")

    def test_from_yaml_invalid_format(self, tmp_path: Path) -> None:
        """Test loading from invalid YAML raises error."""
        yaml_path = tmp_path / "invalid.yaml"
        yaml_path.write_text("not a dict")

        with pytest.raises(ValueError, match="must be a dictionary"):
            BudgetConfig.from_yaml(yaml_path)


class TestBudgetPeriod:
    """Tests for BudgetPeriod enum."""

    def test_period_values(self) -> None:
        """Test budget period values."""
        assert BudgetPeriod.DAILY.value == "daily"
        assert BudgetPeriod.WEEKLY.value == "weekly"
        assert BudgetPeriod.MONTHLY.value == "monthly"

    def test_period_from_string(self) -> None:
        """Test creating period from string."""
        assert BudgetPeriod("daily") == BudgetPeriod.DAILY
        assert BudgetPeriod("weekly") == BudgetPeriod.WEEKLY
        assert BudgetPeriod("monthly") == BudgetPeriod.MONTHLY


class TestBudgetStatus:
    """Tests for BudgetStatus dataclass."""

    def test_create_status(self) -> None:
        """Test creating a budget status."""
        now = datetime.now()
        status = BudgetStatus(
            budget_id=1,
            budget_name="daily-limit",
            period=BudgetPeriod.DAILY,
            period_start=now,
            limit=Decimal("100.00"),
            spent=Decimal("80.00"),
            remaining=Decimal("20.00"),
            percentage=80.0,
            is_over_threshold=True,
            is_over_limit=False,
        )

        assert status.budget_id == 1
        assert status.budget_name == "daily-limit"
        assert status.period == BudgetPeriod.DAILY
        assert status.spent == Decimal("80.00")
        assert status.percentage == 80.0
        assert status.is_over_threshold is True
        assert status.is_over_limit is False

    def test_status_with_triggered_alerts(self) -> None:
        """Test status with triggered alerts."""
        now = datetime.now()
        status = BudgetStatus(
            budget_id=1,
            budget_name="test",
            period=BudgetPeriod.DAILY,
            period_start=now,
            limit=Decimal("100.00"),
            spent=Decimal("90.00"),
            remaining=Decimal("10.00"),
            percentage=90.0,
            is_over_threshold=True,
            is_over_limit=False,
            triggered_alerts=[0.8, 0.9],
        )

        assert status.triggered_alerts == [0.8, 0.9]


class TestBudgetCheckResult:
    """Tests for BudgetCheckResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a check result."""
        now = datetime.now()
        result = BudgetCheckResult(
            timestamp=now,
            statuses=[],
            has_alerts=False,
            has_exceeded=False,
        )

        assert result.timestamp == now
        assert result.statuses == []
        assert result.has_alerts is False
        assert result.has_exceeded is False


class TestLogAlertChannel:
    """Tests for LogAlertChannel."""

    def test_channel_name(self) -> None:
        """Test log channel name."""
        channel = LogAlertChannel()
        assert channel.name == "log"

    @pytest.mark.anyio
    async def test_send_alert(self) -> None:
        """Test sending alert via log channel."""
        channel = LogAlertChannel()
        now = datetime.now()
        status = BudgetStatus(
            budget_id=1,
            budget_name="test",
            period=BudgetPeriod.DAILY,
            period_start=now,
            limit=Decimal("100.00"),
            spent=Decimal("80.00"),
            remaining=Decimal("20.00"),
            percentage=80.0,
            is_over_threshold=True,
            is_over_limit=False,
        )

        result = await channel.send_alert(status, 0.8, "Budget warning")

        assert result is True


class TestWebhookAlertChannel:
    """Tests for WebhookAlertChannel."""

    def test_channel_name(self) -> None:
        """Test webhook channel name."""
        channel = WebhookAlertChannel()
        assert channel.name == "webhook"

    def test_url_property(self) -> None:
        """Test URL property."""
        channel = WebhookAlertChannel(url="https://example.com/webhook")
        assert channel.url == "https://example.com/webhook"

        channel.url = "https://other.com/hook"
        assert channel.url == "https://other.com/hook"

    @pytest.mark.anyio
    async def test_send_alert_no_url(self) -> None:
        """Test sending alert without URL configured."""
        channel = WebhookAlertChannel()
        now = datetime.now()
        status = BudgetStatus(
            budget_id=1,
            budget_name="test",
            period=BudgetPeriod.DAILY,
            period_start=now,
            limit=Decimal("100.00"),
            spent=Decimal("80.00"),
            remaining=Decimal("20.00"),
            percentage=80.0,
            is_over_threshold=True,
            is_over_limit=False,
        )

        result = await channel.send_alert(status, 0.8, "Budget warning")

        assert result is False

    @pytest.mark.anyio
    async def test_send_alert_with_url(self) -> None:
        """Test sending alert with URL configured."""
        channel = WebhookAlertChannel(url="https://example.com/webhook")
        now = datetime.now()
        status = BudgetStatus(
            budget_id=1,
            budget_name="test",
            period=BudgetPeriod.DAILY,
            period_start=now,
            limit=Decimal("100.00"),
            spent=Decimal("80.00"),
            remaining=Decimal("20.00"),
            percentage=80.0,
            is_over_threshold=True,
            is_over_limit=False,
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await channel.send_alert(status, 0.8, "Budget warning")

            assert result is True
            mock_client.post.assert_called_once()


class TestEmailAlertChannel:
    """Tests for EmailAlertChannel."""

    def test_channel_name(self) -> None:
        """Test email channel name."""
        channel = EmailAlertChannel()
        assert channel.name == "email"

    @pytest.mark.anyio
    async def test_send_alert_not_configured(self) -> None:
        """Test sending alert without SMTP configured."""
        channel = EmailAlertChannel()
        now = datetime.now()
        status = BudgetStatus(
            budget_id=1,
            budget_name="test",
            period=BudgetPeriod.DAILY,
            period_start=now,
            limit=Decimal("100.00"),
            spent=Decimal("80.00"),
            remaining=Decimal("20.00"),
            percentage=80.0,
            is_over_threshold=True,
            is_over_limit=False,
        )

        result = await channel.send_alert(status, 0.8, "Budget warning")

        assert result is False


class TestAlertChannelRegistry:
    """Tests for AlertChannelRegistry."""

    def test_default_channels(self) -> None:
        """Test default channels are registered."""
        registry = AlertChannelRegistry()

        assert "log" in registry.list_channels()
        assert "webhook" in registry.list_channels()
        assert "email" in registry.list_channels()

    def test_get_channel(self) -> None:
        """Test getting a channel."""
        registry = AlertChannelRegistry()

        channel = registry.get("log")

        assert channel is not None
        assert isinstance(channel, LogAlertChannel)

    def test_get_unknown_channel(self) -> None:
        """Test getting unknown channel returns None."""
        registry = AlertChannelRegistry()

        channel = registry.get("unknown")

        assert channel is None

    def test_register_custom_channel(self) -> None:
        """Test registering a custom channel."""

        class CustomChannel(LogAlertChannel):
            @property
            def name(self) -> str:
                return "custom"

        registry = AlertChannelRegistry()
        custom = CustomChannel()
        registry.register(custom)

        assert "custom" in registry.list_channels()
        assert registry.get("custom") is custom

    def test_configure_webhook(self) -> None:
        """Test configuring webhook channel."""
        registry = AlertChannelRegistry()

        registry.configure_webhook(
            url="https://example.com/webhook",
            headers={"Authorization": "Bearer token"},
        )

        channel = registry.get("webhook")
        assert isinstance(channel, WebhookAlertChannel)
        assert channel.url == "https://example.com/webhook"

    def test_configure_email(self) -> None:
        """Test configuring email channel."""
        registry = AlertChannelRegistry()

        registry.configure_email(
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_user="user@example.com",
            smtp_password="password",
            to_addrs=["admin@example.com"],
        )

        channel = registry.get("email")
        assert isinstance(channel, EmailAlertChannel)


class TestBudgetManager:
    """Tests for BudgetManager class."""

    def _create_mock_db(self) -> tuple[MagicMock, AsyncMock]:
        """Create mock database and session."""
        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_db.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_db.session.return_value.__aexit__ = AsyncMock(return_value=None)
        return mock_db, mock_session

    @pytest.mark.anyio
    async def test_initialize_creates_budgets(self) -> None:
        """Test initialize creates budgets from config."""
        config = BudgetConfig(
            daily=Decimal("100.00"),
            monthly=Decimal("2000.00"),
            alerts=[AlertConfig(threshold=0.8, channels=["log"])],
        )

        with patch("atp.analytics.budgets.get_analytics_database") as mock_get_db:
            mock_db, mock_session = self._create_mock_db()
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.budgets.CostRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.get_budget_by_name = AsyncMock(return_value=None)
                mock_repo.create_budget = AsyncMock()
                mock_repo_cls.return_value = mock_repo

                manager = BudgetManager()
                await manager.initialize(config)

                # Should create daily and monthly budgets
                assert mock_repo.create_budget.call_count == 2

    @pytest.mark.anyio
    async def test_check_budgets(self) -> None:
        """Test checking budget status."""
        config = BudgetConfig(
            daily=Decimal("100.00"),
            alerts=[AlertConfig(threshold=0.8, channels=["log"])],
        )

        with patch("atp.analytics.budgets.get_analytics_database") as mock_get_db:
            mock_db, mock_session = self._create_mock_db()
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.budgets.CostRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.check_all_budgets = AsyncMock(
                    return_value=[
                        {
                            "budget_id": 1,
                            "budget_name": "default_daily",
                            "period": "daily",
                            "period_start": datetime.now(),
                            "spent": Decimal("50.00"),
                            "limit": Decimal("100.00"),
                            "remaining": Decimal("50.00"),
                            "percentage": 0.5,
                            "is_over_threshold": False,
                            "is_over_limit": False,
                        }
                    ]
                )
                mock_repo_cls.return_value = mock_repo

                manager = BudgetManager()
                manager._config = config

                result = await manager.check_budgets()

                assert len(result.statuses) == 1
                assert result.statuses[0].percentage == 50.0
                assert result.has_alerts is False
                assert result.has_exceeded is False

    @pytest.mark.anyio
    async def test_check_budgets_with_threshold_triggered(self) -> None:
        """Test checking budgets with threshold exceeded."""
        config = BudgetConfig(
            daily=Decimal("100.00"),
            alerts=[AlertConfig(threshold=0.8, channels=["log"])],
        )

        with patch("atp.analytics.budgets.get_analytics_database") as mock_get_db:
            mock_db, mock_session = self._create_mock_db()
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.budgets.CostRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.check_all_budgets = AsyncMock(
                    return_value=[
                        {
                            "budget_id": 1,
                            "budget_name": "default_daily",
                            "period": "daily",
                            "period_start": datetime.now(),
                            "spent": Decimal("85.00"),
                            "limit": Decimal("100.00"),
                            "remaining": Decimal("15.00"),
                            "percentage": 0.85,
                            "is_over_threshold": True,
                            "is_over_limit": False,
                        }
                    ]
                )
                mock_repo_cls.return_value = mock_repo

                manager = BudgetManager()
                manager._config = config

                result = await manager.check_budgets()

                assert result.has_alerts is True
                assert result.has_exceeded is False
                assert 0.8 in result.statuses[0].triggered_alerts

    @pytest.mark.anyio
    async def test_check_budgets_exceeded(self) -> None:
        """Test checking budgets when limit exceeded."""
        config = BudgetConfig(
            daily=Decimal("100.00"),
            alerts=[AlertConfig(threshold=1.0, channels=["log"])],
        )

        with patch("atp.analytics.budgets.get_analytics_database") as mock_get_db:
            mock_db, mock_session = self._create_mock_db()
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.budgets.CostRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.check_all_budgets = AsyncMock(
                    return_value=[
                        {
                            "budget_id": 1,
                            "budget_name": "default_daily",
                            "period": "daily",
                            "period_start": datetime.now(),
                            "spent": Decimal("110.00"),
                            "limit": Decimal("100.00"),
                            "remaining": Decimal("0.00"),
                            "percentage": 1.1,
                            "is_over_threshold": True,
                            "is_over_limit": True,
                        }
                    ]
                )
                mock_repo_cls.return_value = mock_repo

                manager = BudgetManager()
                manager._config = config

                result = await manager.check_budgets()

                assert result.has_alerts is True
                assert result.has_exceeded is True

    @pytest.mark.anyio
    async def test_create_budget(self) -> None:
        """Test creating a new budget."""
        with patch("atp.analytics.budgets.get_analytics_database") as mock_get_db:
            mock_db, mock_session = self._create_mock_db()
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.budgets.CostRepository") as mock_repo_cls:
                mock_budget = MagicMock()
                mock_budget.id = 1
                mock_budget.name = "test-budget"

                mock_repo = AsyncMock()
                mock_repo.create_budget = AsyncMock(return_value=mock_budget)
                mock_repo_cls.return_value = mock_repo

                manager = BudgetManager()

                budget = await manager.create_budget(
                    name="test-budget",
                    period=BudgetPeriod.DAILY,
                    limit_usd=Decimal("100.00"),
                    alert_threshold=0.8,
                    alert_channels=["log"],
                )

                assert budget.name == "test-budget"
                mock_repo.create_budget.assert_called_once()

    @pytest.mark.anyio
    async def test_update_budget(self) -> None:
        """Test updating an existing budget."""
        with patch("atp.analytics.budgets.get_analytics_database") as mock_get_db:
            mock_db, mock_session = self._create_mock_db()
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.budgets.CostRepository") as mock_repo_cls:
                mock_budget = MagicMock()
                mock_budget.id = 1
                mock_budget.name = "test-budget"
                mock_budget.limit_usd = Decimal("150.00")

                mock_repo = AsyncMock()
                mock_repo.get_budget_by_name = AsyncMock(return_value=mock_budget)
                mock_repo.update_budget = AsyncMock(return_value=mock_budget)
                mock_repo_cls.return_value = mock_repo

                manager = BudgetManager()

                budget = await manager.update_budget(
                    name="test-budget",
                    limit_usd=Decimal("150.00"),
                )

                assert budget is not None
                mock_repo.update_budget.assert_called_once()

    @pytest.mark.anyio
    async def test_update_budget_not_found(self) -> None:
        """Test updating a non-existent budget."""
        with patch("atp.analytics.budgets.get_analytics_database") as mock_get_db:
            mock_db, mock_session = self._create_mock_db()
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.budgets.CostRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.get_budget_by_name = AsyncMock(return_value=None)
                mock_repo_cls.return_value = mock_repo

                manager = BudgetManager()

                budget = await manager.update_budget(
                    name="nonexistent",
                    limit_usd=Decimal("100.00"),
                )

                assert budget is None

    @pytest.mark.anyio
    async def test_delete_budget(self) -> None:
        """Test deleting a budget."""
        with patch("atp.analytics.budgets.get_analytics_database") as mock_get_db:
            mock_db, mock_session = self._create_mock_db()
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.budgets.CostRepository") as mock_repo_cls:
                mock_budget = MagicMock()
                mock_budget.id = 1

                mock_repo = AsyncMock()
                mock_repo.get_budget_by_name = AsyncMock(return_value=mock_budget)
                mock_repo.delete_budget = AsyncMock()
                mock_repo_cls.return_value = mock_repo

                manager = BudgetManager()

                result = await manager.delete_budget("test-budget")

                assert result is True
                mock_repo.delete_budget.assert_called_once_with(1)

    @pytest.mark.anyio
    async def test_list_budgets(self) -> None:
        """Test listing budgets."""
        with patch("atp.analytics.budgets.get_analytics_database") as mock_get_db:
            mock_db, mock_session = self._create_mock_db()
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.budgets.CostRepository") as mock_repo_cls:
                mock_budget1 = MagicMock()
                mock_budget1.name = "budget-1"
                mock_budget2 = MagicMock()
                mock_budget2.name = "budget-2"

                mock_repo = AsyncMock()
                mock_repo.list_budgets = AsyncMock(
                    return_value=[mock_budget1, mock_budget2]
                )
                mock_repo_cls.return_value = mock_repo

                manager = BudgetManager()

                budgets = await manager.list_budgets()

                assert len(budgets) == 2

    @pytest.mark.anyio
    async def test_get_budget_status(self) -> None:
        """Test getting status for a specific budget."""
        config = BudgetConfig(
            daily=Decimal("100.00"),
        )

        with patch("atp.analytics.budgets.get_analytics_database") as mock_get_db:
            mock_db, mock_session = self._create_mock_db()
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.budgets.CostRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.check_all_budgets = AsyncMock(
                    return_value=[
                        {
                            "budget_id": 1,
                            "budget_name": "target-budget",
                            "period": "daily",
                            "period_start": datetime.now(),
                            "spent": Decimal("50.00"),
                            "limit": Decimal("100.00"),
                            "remaining": Decimal("50.00"),
                            "percentage": 0.5,
                            "is_over_threshold": False,
                            "is_over_limit": False,
                        },
                        {
                            "budget_id": 2,
                            "budget_name": "other-budget",
                            "period": "monthly",
                            "period_start": datetime.now(),
                            "spent": Decimal("100.00"),
                            "limit": Decimal("1000.00"),
                            "remaining": Decimal("900.00"),
                            "percentage": 0.1,
                            "is_over_threshold": False,
                            "is_over_limit": False,
                        },
                    ]
                )
                mock_repo_cls.return_value = mock_repo

                manager = BudgetManager()
                manager._config = config

                status = await manager.get_budget_status("target-budget")

                assert status is not None
                assert status.budget_name == "target-budget"

    @pytest.mark.anyio
    async def test_get_budget_status_not_found(self) -> None:
        """Test getting status for non-existent budget."""
        config = BudgetConfig(daily=Decimal("100.00"))

        with patch("atp.analytics.budgets.get_analytics_database") as mock_get_db:
            mock_db, mock_session = self._create_mock_db()
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.budgets.CostRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.check_all_budgets = AsyncMock(return_value=[])
                mock_repo_cls.return_value = mock_repo

                manager = BudgetManager()
                manager._config = config

                status = await manager.get_budget_status("nonexistent")

                assert status is None


class TestGlobalBudgetManager:
    """Tests for global budget manager functions."""

    @pytest.mark.anyio
    async def test_get_budget_manager(self) -> None:
        """Test getting global budget manager."""
        # Reset
        set_budget_manager(None)  # type: ignore

        manager = await get_budget_manager()

        assert manager is not None
        assert isinstance(manager, BudgetManager)

    @pytest.mark.anyio
    async def test_set_budget_manager(self) -> None:
        """Test setting custom budget manager."""
        custom_manager = BudgetManager()
        set_budget_manager(custom_manager)

        manager = await get_budget_manager()

        assert manager is custom_manager

        # Reset
        set_budget_manager(None)  # type: ignore

    @pytest.mark.anyio
    async def test_check_budget_for_cost(self) -> None:
        """Test convenience function for cost tracking integration."""
        config = BudgetConfig(daily=Decimal("100.00"))

        with patch("atp.analytics.budgets.get_analytics_database") as mock_get_db:
            mock_db = MagicMock()
            mock_session = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_db.session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_db.session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_get_db.return_value = mock_db

            with patch("atp.analytics.budgets.CostRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.check_all_budgets = AsyncMock(return_value=[])
                mock_repo_cls.return_value = mock_repo

                # Create and set custom manager
                manager = BudgetManager()
                manager._config = config
                set_budget_manager(manager)

                result = await check_budget_for_cost(
                    cost_usd=Decimal("10.00"),
                    provider="anthropic",
                )

                assert isinstance(result, BudgetCheckResult)

        # Reset
        set_budget_manager(None)  # type: ignore
