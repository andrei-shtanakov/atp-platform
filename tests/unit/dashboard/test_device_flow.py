"""Tests for GitHub Device Flow logic."""

from unittest.mock import AsyncMock, patch

import pytest

from atp.dashboard.auth.device_flow import (
    DeviceCodeExpiredError,
    DeviceCodeNotFoundError,
    DeviceCodePendingError,
    DeviceFlowManager,
    DeviceFlowStore,
)


class TestDeviceFlowStore:
    """Tests for in-memory device flow store."""

    def test_create_device_code(self) -> None:
        store = DeviceFlowStore()
        entry = store.create(client_id="test-client", expires_in=900, interval=5)
        assert entry.device_code is not None
        assert entry.user_code is not None
        assert len(entry.user_code) == 8
        assert entry.client_id == "test-client"
        assert entry.interval == 5

    def test_get_by_device_code(self) -> None:
        store = DeviceFlowStore()
        entry = store.create(client_id="test", expires_in=900, interval=5)
        found = store.get_by_device_code(entry.device_code)
        assert found is not None
        assert found.device_code == entry.device_code

    def test_get_by_user_code(self) -> None:
        store = DeviceFlowStore()
        entry = store.create(client_id="test", expires_in=900, interval=5)
        found = store.get_by_user_code(entry.user_code)
        assert found is not None
        assert found.user_code == entry.user_code

    def test_get_expired_entry(self) -> None:
        store = DeviceFlowStore()
        entry = store.create(client_id="test", expires_in=0, interval=5)
        found = store.get_by_device_code(entry.device_code)
        assert found is None

    def test_remove(self) -> None:
        store = DeviceFlowStore()
        entry = store.create(client_id="test", expires_in=900, interval=5)
        store.remove(entry.device_code)
        found = store.get_by_device_code(entry.device_code)
        assert found is None

    def test_mark_authorized(self) -> None:
        store = DeviceFlowStore()
        entry = store.create(client_id="test", expires_in=900, interval=5)
        store.mark_authorized(entry.device_code, github_access_token="gho_abc123")
        found = store.get_by_device_code(entry.device_code)
        assert found is not None
        assert found.github_access_token == "gho_abc123"

    def test_cleanup_expired(self) -> None:
        store = DeviceFlowStore()
        store.create(client_id="test", expires_in=0, interval=5)
        store.create(client_id="test2", expires_in=900, interval=5)
        store.cleanup_expired()
        assert len(store._entries) == 1


class TestDeviceFlowManager:
    """Tests for Device Flow manager."""

    @pytest.mark.anyio
    async def test_initiate_creates_entry(self) -> None:
        manager = DeviceFlowManager(
            client_id="test-client", client_secret="test-secret"
        )
        result = manager.initiate()
        assert "device_code" in result
        assert "user_code" in result
        assert result["verification_uri"] == "https://github.com/login/device"
        assert result["expires_in"] == 900
        assert result["interval"] == 5

    @pytest.mark.anyio
    async def test_poll_pending(self) -> None:
        manager = DeviceFlowManager(
            client_id="test-client", client_secret="test-secret"
        )
        result = manager.initiate()
        with patch.object(
            manager,
            "_exchange_device_code",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(DeviceCodePendingError):
                await manager.poll(result["device_code"])  # type: ignore[arg-type]

    @pytest.mark.anyio
    async def test_poll_expired(self) -> None:
        manager = DeviceFlowManager(
            client_id="test-client", client_secret="test-secret"
        )
        entry = manager._store.create(client_id="test-client", expires_in=0, interval=5)
        with pytest.raises(DeviceCodeExpiredError):
            await manager.poll(entry.device_code)

    @pytest.mark.anyio
    async def test_poll_not_found(self) -> None:
        manager = DeviceFlowManager(
            client_id="test-client", client_secret="test-secret"
        )
        with pytest.raises(DeviceCodeNotFoundError):
            await manager.poll("nonexistent-code")

    @pytest.mark.anyio
    async def test_poll_authorized(self) -> None:
        manager = DeviceFlowManager(
            client_id="test-client", client_secret="test-secret"
        )
        result = manager.initiate()
        manager._store.mark_authorized(
            result["device_code"],  # type: ignore[arg-type]
            github_access_token="gho_test123",
        )
        mock_user_info = {
            "login": "testuser",
            "email": "test@example.com",
            "name": "Test User",
            "id": 12345,
            "avatar_url": "https://avatars.githubusercontent.com/u/12345",
        }
        with patch.object(
            manager,
            "_fetch_github_user",
            new_callable=AsyncMock,
            return_value=mock_user_info,
        ):
            user_info = await manager.poll(result["device_code"])  # type: ignore[arg-type]
            assert user_info["login"] == "testuser"
            assert user_info["email"] == "test@example.com"
