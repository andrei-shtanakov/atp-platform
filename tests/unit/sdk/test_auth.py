"""Tests for ATP SDK auth module."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from atp_sdk.auth import load_token, save_token


class TestSaveLoadToken:
    """Tests for token persistence."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading a token."""
        config_file = tmp_path / "config.json"
        with (
            patch("atp_sdk.auth.CONFIG_DIR", tmp_path),
            patch("atp_sdk.auth.CONFIG_FILE", config_file),
        ):
            save_token("test-jwt-token")
            loaded = load_token()
            assert loaded == "test-jwt-token"

    def test_save_with_platform_url(self, tmp_path: Path) -> None:
        """Test saving token keyed by platform URL."""
        config_file = tmp_path / "config.json"
        with (
            patch("atp_sdk.auth.CONFIG_DIR", tmp_path),
            patch("atp_sdk.auth.CONFIG_FILE", config_file),
        ):
            save_token("token-a", platform_url="https://a.example.com")
            save_token("token-b", platform_url="https://b.example.com")

            assert load_token(platform_url="https://a.example.com") == "token-a"
            assert load_token(platform_url="https://b.example.com") == "token-b"
            # Generic token is the last one saved
            assert load_token() == "token-b"

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Test loading when config file doesn't exist."""
        config_file = tmp_path / "nonexistent" / "config.json"
        with patch("atp_sdk.auth.CONFIG_FILE", config_file):
            assert load_token() is None

    def test_load_corrupt_file(self, tmp_path: Path) -> None:
        """Test loading when config file is corrupt."""
        config_file = tmp_path / "config.json"
        config_file.write_text("not json{{{")
        with patch("atp_sdk.auth.CONFIG_FILE", config_file):
            assert load_token() is None

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """Test that save creates the config directory."""
        config_dir = tmp_path / "newdir"
        config_file = config_dir / "config.json"
        with (
            patch("atp_sdk.auth.CONFIG_DIR", config_dir),
            patch("atp_sdk.auth.CONFIG_FILE", config_file),
        ):
            save_token("tok")
            assert config_file.exists()
            data = json.loads(config_file.read_text())
            assert data["token"] == "tok"


class TestATPClientTokenResolution:
    """Tests for ATPClient token resolution order."""

    @pytest.mark.anyio
    async def test_explicit_token(self) -> None:
        """Explicit token takes priority."""
        from atp_sdk.client import ATPClient

        with patch("atp_sdk.client.load_token", return_value="saved"):
            async with ATPClient(token="explicit") as client:
                assert client.token == "explicit"

    @pytest.mark.anyio
    async def test_env_var_token(self) -> None:
        """ATP_TOKEN env var is second priority."""
        from atp_sdk.client import ATPClient

        with (
            patch.dict("os.environ", {"ATP_TOKEN": "from-env"}),
            patch("atp_sdk.client.load_token", return_value="saved"),
        ):
            async with ATPClient() as client:
                assert client.token == "from-env"

    @pytest.mark.anyio
    async def test_saved_token(self) -> None:
        """Saved token from config file is last priority."""
        from atp_sdk.client import ATPClient

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("atp_sdk.client.load_token", return_value="saved-tok"),
        ):
            import os

            os.environ.pop("ATP_TOKEN", None)
            async with ATPClient() as client:
                assert client.token == "saved-tok"
