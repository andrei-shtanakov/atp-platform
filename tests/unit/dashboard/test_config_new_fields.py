"""Test new config fields for token self-service."""

import pytest

from atp.dashboard.v2.config import DashboardConfig


class TestTokenSelfServiceConfig:
    def test_defaults(self) -> None:
        config = DashboardConfig(debug=True)
        assert config.registration_mode == "invite"
        assert config.max_agents_per_user == 10
        assert config.max_tokens_per_agent == 3
        assert config.max_user_tokens == 5
        assert config.default_token_days == 30
        assert config.max_token_days == 365

    def test_registration_mode_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATP_REGISTRATION_MODE", "open")
        config = DashboardConfig(debug=True)
        assert config.registration_mode == "open"
