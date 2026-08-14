"""Tests for the ATP_SERVER_PROFILE config field."""

import pytest
from pydantic import ValidationError

from atp.dashboard.v2.config import DashboardConfig


def test_default_profile_is_full() -> None:
    config = DashboardConfig(secret_key="x")
    assert config.server_profile == "full"


def test_env_sets_eco(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATP_SERVER_PROFILE", "eco")
    config = DashboardConfig(secret_key="x")
    assert config.server_profile == "eco"


def test_unknown_profile_rejected() -> None:
    with pytest.raises(ValidationError):
        DashboardConfig(secret_key="x", server_profile="turbo")


def test_profile_in_to_dict() -> None:
    config = DashboardConfig(secret_key="x", server_profile="eco")
    assert config.to_dict()["server_profile"] == "eco"
