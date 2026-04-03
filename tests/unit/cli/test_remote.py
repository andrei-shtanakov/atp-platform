"""Tests for CLI remote utilities."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from atp.cli.commands.remote import (
    MANIFEST_FILE,
    file_sha256,
    load_manifest,
    resolve_auth_headers,
    resolve_server_url,
    save_manifest,
)


class TestResolveAuthHeaders:
    """Tests for auth header resolution."""

    def test_api_key_flag_takes_priority(self) -> None:
        headers = resolve_auth_headers(api_key="sk-test")
        assert headers["Authorization"] == "Bearer sk-test"

    def test_env_var_fallback(self) -> None:
        with patch.dict("os.environ", {"ATP_API_KEY": "sk-env"}):
            headers = resolve_auth_headers(api_key=None)
            assert headers["Authorization"] == "Bearer sk-env"

    def test_config_token_fallback(self) -> None:
        with patch(
            "atp.cli.commands.remote.load_token",
            return_value="jwt-token",
        ):
            with patch.dict("os.environ", {}, clear=True):
                headers = resolve_auth_headers(api_key=None)
                assert headers["Authorization"] == "Bearer jwt-token"

    def test_no_auth_returns_empty(self) -> None:
        with patch(
            "atp.cli.commands.remote.load_token",
            return_value=None,
        ):
            with patch.dict("os.environ", {}, clear=True):
                headers = resolve_auth_headers(api_key=None)
                assert "Authorization" not in headers


class TestResolveServerUrl:
    """Tests for server URL resolution."""

    def test_flag_takes_priority(self) -> None:
        url = resolve_server_url(
            server="https://flag.example.com",
            directory=Path("/tmp"),
        )
        assert url == "https://flag.example.com"

    def test_env_var_fallback(self) -> None:
        with patch.dict("os.environ", {"ATP_SERVER": "https://env.example.com"}):
            url = resolve_server_url(server=None, directory=Path("/tmp"))
            assert url == "https://env.example.com"

    def test_manifest_fallback(self, tmp_path: Path) -> None:
        manifest = tmp_path / MANIFEST_FILE
        manifest.write_text(
            json.dumps({"server": "https://manifest.example.com", "files": {}})
        )
        with patch.dict("os.environ", {}, clear=True):
            url = resolve_server_url(server=None, directory=tmp_path)
            assert url == "https://manifest.example.com"

    def test_no_server_returns_none(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {}, clear=True):
            url = resolve_server_url(server=None, directory=tmp_path)
            assert url is None


class TestManifest:
    """Tests for manifest I/O."""

    def test_load_empty(self, tmp_path: Path) -> None:
        manifest = load_manifest(tmp_path)
        assert manifest == {"server": "", "last_sync": "", "files": {}}

    def test_save_and_load(self, tmp_path: Path) -> None:
        data = {
            "server": "https://example.com",
            "last_sync": "2026-04-03T12:00:00Z",
            "files": {
                "suite.yaml": {
                    "sha256": "abc123",
                    "suite_id": 1,
                    "synced_at": "2026-04-03T12:00:00Z",
                }
            },
        }
        save_manifest(tmp_path, data)
        loaded = load_manifest(tmp_path)
        # last_sync is updated by save_manifest, so check other fields
        assert loaded["server"] == data["server"]
        assert loaded["files"] == data["files"]

    def test_load_corrupt_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / MANIFEST_FILE).write_text("not json{{{")
        manifest = load_manifest(tmp_path)
        assert manifest["files"] == {}


class TestFileSha256:
    """Tests for file hashing."""

    def test_hash_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text("hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert file_sha256(f) == expected
