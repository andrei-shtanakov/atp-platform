"""Tests for GitHub Device Flow logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from atp.dashboard.auth.device_flow import (
    DeviceCodeNotFoundError,
    DeviceCodePendingError,
    DeviceFlowManager,
)
from atp.dashboard.auth.state_store import InMemoryAuthStateStore

# Mock GitHub device/code response
MOCK_GITHUB_DEVICE_RESPONSE = {
    "device_code": "github-device-code-123",
    "user_code": "ABCD-1234",
    "verification_uri": "https://github.com/login/device",
    "expires_in": 900,
    "interval": 5,
}


def _mock_github_initiate():
    """Create a patch that mocks the GitHub device/code HTTP call."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = MOCK_GITHUB_DEVICE_RESPONSE
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return patch(
        "atp.dashboard.auth.device_flow.httpx.AsyncClient",
        return_value=mock_client,
    )


class TestDeviceFlowManager:
    """Tests for Device Flow manager using AuthStateStore."""

    @pytest.fixture
    def store(self) -> InMemoryAuthStateStore:
        return InMemoryAuthStateStore()

    @pytest.fixture
    def manager(self, store: InMemoryAuthStateStore) -> DeviceFlowManager:
        return DeviceFlowManager(client_id="test-client", store=store)

    @pytest.mark.anyio
    async def test_initiate_creates_entry(
        self, manager: DeviceFlowManager, store: InMemoryAuthStateStore
    ) -> None:
        with _mock_github_initiate():
            result = await manager.initiate()
        assert "device_code" in result
        assert "user_code" in result
        assert result["verification_uri"] == "https://github.com/login/device"
        assert result["expires_in"] == 900
        assert result["interval"] == 5

        # Verify entry stored in AuthStateStore
        entry = await store.get(f"device:{result['device_code']}")
        assert entry is not None
        assert entry["user_code"] == "ABCD-1234"

    @pytest.mark.anyio
    async def test_poll_pending(self, manager: DeviceFlowManager) -> None:
        with _mock_github_initiate():
            result = await manager.initiate()
        with patch.object(
            manager,
            "_exchange_device_code",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(DeviceCodePendingError):
                await manager.poll(
                    result["device_code"]  # type: ignore[arg-type]
                )

    @pytest.mark.anyio
    async def test_poll_expired(
        self, manager: DeviceFlowManager, store: InMemoryAuthStateStore
    ) -> None:
        # Store entry with 0 TTL (immediately expired)
        await store.put("device:expired-code", {"client_id": "test"}, ttl_seconds=0)
        with pytest.raises(DeviceCodeNotFoundError):
            await manager.poll("expired-code")

    @pytest.mark.anyio
    async def test_poll_not_found(self, manager: DeviceFlowManager) -> None:
        with pytest.raises(DeviceCodeNotFoundError):
            await manager.poll("nonexistent-code")

    @pytest.mark.anyio
    async def test_poll_authorized(
        self, manager: DeviceFlowManager, store: InMemoryAuthStateStore
    ) -> None:
        with _mock_github_initiate():
            result = await manager.initiate()

        # Simulate authorization by storing token in the entry
        key = f"device:{result['device_code']}"
        entry = await store.get(key)
        assert entry is not None
        entry["github_access_token"] = "gho_test123"
        await store.put(key, entry, ttl_seconds=900)

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
            user_info = await manager.poll(
                result["device_code"]  # type: ignore[arg-type]
            )
            assert user_info["login"] == "testuser"
            assert user_info["email"] == "test@example.com"

    @pytest.mark.anyio
    async def test_poll_exchange_success(self, manager: DeviceFlowManager) -> None:
        """Test poll when exchange succeeds on first try."""
        with _mock_github_initiate():
            result = await manager.initiate()

        mock_user_info = {
            "login": "testuser",
            "email": "test@example.com",
            "name": "Test User",
            "id": 12345,
            "avatar_url": None,
        }
        with (
            patch.object(
                manager,
                "_exchange_device_code",
                new_callable=AsyncMock,
                return_value="gho_fresh_token",
            ),
            patch.object(
                manager,
                "_fetch_github_user",
                new_callable=AsyncMock,
                return_value=mock_user_info,
            ),
        ):
            user_info = await manager.poll(
                result["device_code"]  # type: ignore[arg-type]
            )
            assert user_info["login"] == "testuser"
