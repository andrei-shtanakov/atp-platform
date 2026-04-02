"""Tests for atp-sdk package imports."""


def test_sdk_importable() -> None:
    from atp_sdk import ATPClient

    assert ATPClient is not None


def test_sdk_models_importable() -> None:
    from atp_sdk.models import LeaderboardEntry, RunStatus

    assert RunStatus is not None
    assert LeaderboardEntry is not None
