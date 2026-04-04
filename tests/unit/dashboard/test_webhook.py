"""Tests for webhook delivery module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.dashboard.webhook import (
    build_webhook_payload,
    deliver_webhook,
    schedule_webhook,
    validate_webhook_url,
)


class TestValidateWebhookUrl:
    """Tests for SSRF protection."""

    def test_allows_public_https(self) -> None:
        validate_webhook_url("https://ci.example.com/hook")

    def test_allows_public_http(self) -> None:
        validate_webhook_url("http://ci.example.com/hook")

    def test_blocks_private_10(self) -> None:
        with pytest.raises(ValueError, match="private"):
            validate_webhook_url("http://10.0.0.1/hook")

    def test_blocks_private_172(self) -> None:
        with pytest.raises(ValueError, match="private"):
            validate_webhook_url("http://172.16.0.1/hook")

    def test_blocks_private_192(self) -> None:
        with pytest.raises(ValueError, match="private"):
            validate_webhook_url("http://192.168.1.1/hook")

    def test_blocks_link_local(self) -> None:
        with pytest.raises(ValueError, match="private"):
            validate_webhook_url("http://169.254.169.254/metadata")

    def test_blocks_localhost(self) -> None:
        with pytest.raises(ValueError, match="private"):
            validate_webhook_url("http://localhost/admin")

    def test_blocks_127(self) -> None:
        with pytest.raises(ValueError, match="private"):
            validate_webhook_url("http://127.0.0.1:8080/api")

    def test_blocks_ipv6_loopback(self) -> None:
        with pytest.raises(ValueError, match="private"):
            validate_webhook_url("http://[::1]/hook")

    def test_blocks_non_http_scheme(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            validate_webhook_url("ftp://example.com/hook")

    def test_blocks_empty_url(self) -> None:
        with pytest.raises(ValueError):
            validate_webhook_url("")

    def test_allows_ip_on_public_range(self) -> None:
        validate_webhook_url("https://8.8.8.8/hook")


class TestBuildWebhookPayload:
    """Tests for payload construction."""

    def test_completed_run_payload(self) -> None:
        bm = MagicMock()
        bm.id = 1
        bm.name = "Test Benchmark"
        run = MagicMock()
        run.id = 42
        run.status = "COMPLETED"
        run.total_score = 85.5
        run.current_task_index = 10
        run.started_at = MagicMock()
        run.started_at.isoformat.return_value = "2026-04-04T11:00:00"
        run.finished_at = MagicMock()
        run.finished_at.isoformat.return_value = "2026-04-04T12:00:00"

        payload = build_webhook_payload("run.completed", bm, run, tasks_total=10)

        assert payload["event"] == "run.completed"
        assert payload["benchmark"]["id"] == 1
        assert payload["benchmark"]["name"] == "Test Benchmark"
        assert payload["run"]["id"] == 42
        assert payload["run"]["status"] == "COMPLETED"
        assert payload["run"]["total_score"] == 85.5
        assert "delivery_id" in payload
        assert "timestamp" in payload


class TestDeliverWebhook:
    """Tests for HTTP delivery with retry."""

    @pytest.mark.anyio
    async def test_successful_delivery(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("atp.dashboard.webhook.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await deliver_webhook(
                "https://example.com/hook",
                {
                    "event": "run.completed",
                    "delivery_id": "test-123",
                },
            )

        assert result is True
        mock_client.post.assert_called_once()

    @pytest.mark.anyio
    async def test_retries_on_failure(self) -> None:
        mock_success = MagicMock()
        mock_success.status_code = 200

        with patch("atp.dashboard.webhook.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [
                Exception("Connection refused"),
                mock_success,
            ]
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with patch("anyio.sleep", new_callable=AsyncMock):
                result = await deliver_webhook(
                    "https://example.com/hook",
                    {
                        "event": "run.completed",
                        "delivery_id": "test-456",
                    },
                )

        assert result is True
        assert mock_client.post.call_count == 2

    @pytest.mark.anyio
    async def test_gives_up_after_max_retries(self) -> None:
        with patch("atp.dashboard.webhook.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with patch("anyio.sleep", new_callable=AsyncMock):
                result = await deliver_webhook(
                    "https://example.com/hook",
                    {
                        "event": "run.completed",
                        "delivery_id": "test-789",
                    },
                )

        assert result is False
        assert mock_client.post.call_count == 3


class TestScheduleWebhook:
    """Tests for background task scheduling."""

    @pytest.mark.anyio
    async def test_schedule_creates_task(self) -> None:
        with patch(
            "atp.dashboard.webhook.deliver_webhook",
            new_callable=AsyncMock,
            return_value=True,
        ):
            schedule_webhook(
                "https://example.com/hook",
                {"event": "run.completed"},
            )
            await asyncio.sleep(0.01)
