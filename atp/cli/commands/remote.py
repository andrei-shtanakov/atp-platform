"""Shared utilities for CLI remote commands (push/pull/sync).

Provides auth resolution, server URL resolution, manifest I/O,
and file hashing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atp_sdk.auth import load_token

logger = logging.getLogger("atp.cli")

MANIFEST_FILE = ".atp-sync.json"

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_PARTIAL = 2


def resolve_auth_headers(
    api_key: str | None = None,
    server_url: str | None = None,
) -> dict[str, str]:
    """Resolve authentication headers.

    Priority: api_key flag > ATP_API_KEY env > ~/.atp/config.json token.
    """
    token = (
        api_key or os.environ.get("ATP_API_KEY") or load_token(platform_url=server_url)
    )
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def resolve_server_url(
    server: str | None,
    directory: Path,
) -> str | None:
    """Resolve server URL.

    Priority: --server flag > ATP_SERVER env > .atp-sync.json manifest.
    """
    if server:
        return server.rstrip("/")
    env_server = os.environ.get("ATP_SERVER")
    if env_server:
        return env_server.rstrip("/")
    manifest = load_manifest(directory)
    if manifest.get("server"):
        return manifest["server"]
    return None


def load_manifest(directory: Path) -> dict[str, Any]:
    """Load .atp-sync.json manifest from directory."""
    path = directory / MANIFEST_FILE
    if not path.exists():
        return {"server": "", "last_sync": "", "files": {}}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"server": "", "last_sync": "", "files": {}}


def save_manifest(directory: Path, data: dict[str, Any]) -> None:
    """Save .atp-sync.json manifest to directory."""
    data["last_sync"] = datetime.now(UTC).isoformat()
    path = directory / MANIFEST_FILE
    path.write_text(json.dumps(data, indent=2) + "\n")


def file_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_yaml_files(directory: Path) -> list[Path]:
    """Find all .yaml/.yml files in directory recursively."""
    files: list[Path] = []
    for ext in ("*.yaml", "*.yml"):
        files.extend(directory.rglob(ext))
    # Exclude manifest and hidden files
    return sorted(
        f for f in files if not f.name.startswith(".") and MANIFEST_FILE not in f.name
    )


def now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(UTC).isoformat()
