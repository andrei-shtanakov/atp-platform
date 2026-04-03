"""Tests for device auth route endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.v2.factory import create_test_app

# Mock GitHub device/code response
MOCK_GITHUB_DEVICE_RESPONSE = {
    "device_code": "github-device-code-test",
    "user_code": "TEST1234",
    "verification_uri": "https://github.com/login/device",
    "expires_in": 900,
    "interval": 5,
}


def _mock_github_http():
    """Patch httpx.AsyncClient to mock GitHub device/code call."""
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


@pytest.fixture
def app():
    """Create test app with device auth routes."""
    return create_test_app()


@pytest.fixture
async def client(app):
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestDeviceAuthInitiate:
    """Tests for POST /auth/device."""

    @pytest.mark.anyio
    async def test_initiate_success(self, client) -> None:
        """Test successful device flow initiation."""
        with patch("atp.dashboard.v2.routes.device_auth.get_config") as mock_config:
            cfg = MagicMock()
            cfg.github_client_id = "test-client-id"
            cfg.github_client_secret = "test-client-secret"
            mock_config.return_value = cfg
            # Reset singleton so it picks up mocked config
            import atp.dashboard.v2.routes.device_auth as mod

            mod._manager = None

            with _mock_github_http():
                resp = await client.post("/api/auth/device")
            assert resp.status_code == 200
            data = resp.json()
            assert "device_code" in data
            assert "user_code" in data
            assert "verification_uri" in data
            assert data["verification_uri"] == "https://github.com/login/device"
            assert "expires_in" in data
            assert "interval" in data

            # Reset for other tests
            mod._manager = None

    @pytest.mark.anyio
    async def test_initiate_not_configured(self, client) -> None:
        """Test initiation when GitHub OAuth is not configured."""
        with patch("atp.dashboard.v2.routes.device_auth.get_config") as mock_config:
            cfg = MagicMock()
            cfg.github_client_id = None
            cfg.github_client_secret = None
            mock_config.return_value = cfg
            import atp.dashboard.v2.routes.device_auth as mod

            mod._manager = None

            resp = await client.post("/api/auth/device")
            assert resp.status_code == 501

            mod._manager = None


class TestDeviceAuthPoll:
    """Tests for POST /auth/device/poll."""

    @pytest.mark.anyio
    async def test_poll_unknown_code(self, client) -> None:
        """Test polling with unknown device code."""
        with patch("atp.dashboard.v2.routes.device_auth.get_config") as mock_config:
            cfg = MagicMock()
            cfg.github_client_id = "test-client-id"
            cfg.github_client_secret = "test-client-secret"
            mock_config.return_value = cfg
            import atp.dashboard.v2.routes.device_auth as mod

            mod._manager = None

            resp = await client.post(
                "/api/auth/device/poll",
                json={"device_code": "nonexistent"},
            )
            assert resp.status_code == 404

            mod._manager = None
