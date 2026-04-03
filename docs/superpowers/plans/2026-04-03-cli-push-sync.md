# CLI Push/Pull/Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `atp push`, `atp pull`, and `atp sync` CLI commands for managing test suites between local filesystem and remote ATP server.

**Architecture:** Shared `remote.py` module handles auth resolution, server URL, HTTP client, and manifest I/O. Three command modules (`push.py`, `pull.py`, `sync_cmd.py`) use shared utilities. Commands registered in `main.py` via `cli.add_command()`. All HTTP calls go through existing upload/export API endpoints.

**Tech Stack:** click (CLI), httpx (HTTP), hashlib (SHA256), pathlib, JSON manifest

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Create | `atp/cli/commands/remote.py` | Shared: auth, server URL, HTTP client, manifest I/O |
| Create | `atp/cli/commands/push.py` | `atp push` command |
| Create | `atp/cli/commands/pull.py` | `atp pull` command |
| Create | `atp/cli/commands/sync_cmd.py` | `atp sync` command |
| Modify | `atp/cli/main.py` | Register push, pull, sync commands |
| Create | `tests/unit/cli/test_remote.py` | Tests for shared utilities |
| Create | `tests/unit/cli/test_push.py` | Tests for push command |
| Create | `tests/unit/cli/test_pull.py` | Tests for pull command |
| Create | `tests/unit/cli/test_sync.py` | Tests for sync command |

---

### Task 1: Shared remote utilities module

**Files:**
- Create: `atp/cli/commands/remote.py`
- Create: `tests/unit/cli/test_remote.py`

- [ ] **Step 1: Write tests for remote utilities**

Create `tests/unit/cli/test_remote.py`:

```python
"""Tests for CLI remote utilities."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

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
        with patch.dict(
            "os.environ", {"ATP_SERVER": "https://env.example.com"}
        ):
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
        assert loaded == data

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
```

- [ ] **Step 2: Run tests to see them fail**

Run: `uv run python -m pytest tests/unit/cli/test_remote.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement remote.py**

Create `atp/cli/commands/remote.py`:

```python
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
        api_key
        or os.environ.get("ATP_API_KEY")
        or load_token(platform_url=server_url)
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
        f
        for f in files
        if not f.name.startswith(".") and MANIFEST_FILE not in f.name
    )


def now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(UTC).isoformat()
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/unit/cli/test_remote.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff**

Run: `uv run ruff format atp/cli/commands/remote.py tests/unit/cli/test_remote.py && uv run ruff check atp/cli/commands/remote.py tests/unit/cli/test_remote.py --fix`

- [ ] **Step 6: Commit**

```bash
git add atp/cli/commands/remote.py tests/unit/cli/test_remote.py
git commit -m "feat(cli): add shared remote utilities — auth, server URL, manifest, hashing"
```

---

### Task 2: Implement `atp push` command

**Files:**
- Create: `atp/cli/commands/push.py`
- Create: `tests/unit/cli/test_push.py`
- Modify: `atp/cli/main.py`

- [ ] **Step 1: Write tests for push**

Create `tests/unit/cli/test_push.py`:

```python
"""Tests for atp push command."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from atp.cli.commands.push import push_command


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def valid_yaml(tmp_path: Path) -> Path:
    f = tmp_path / "suite.yaml"
    f.write_text("test_suite: test\ntests:\n  - id: t1\n    name: T1\n    task:\n      description: do\n")
    return f


@pytest.fixture
def invalid_yaml(tmp_path: Path) -> Path:
    f = tmp_path / "bad.yaml"
    f.write_text("not valid yaml: [[[")
    return f


class TestPushCommand:
    """Tests for atp push."""

    def test_push_single_file_success(
        self, runner: CliRunner, valid_yaml: Path
    ) -> None:
        """Push a single valid file."""
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "suite": {"id": 1, "name": "test"},
            "validation": {"valid": True, "errors": [], "warnings": []},
            "filename": "suite.yaml",
        }

        with patch("atp.cli.commands.push.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(
                return_value=mock_client
            )
            mock_httpx.Client.return_value.__exit__ = MagicMock(
                return_value=False
            )
            mock_client.post.return_value = mock_resp

            result = runner.invoke(
                push_command,
                [str(valid_yaml), "--server", "http://test:8000"],
            )
            assert result.exit_code == 0
            assert "created" in result.output

    def test_push_no_server_fails(
        self, runner: CliRunner, valid_yaml: Path
    ) -> None:
        """Push without server URL fails."""
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(push_command, [str(valid_yaml)])
            assert result.exit_code != 0

    def test_push_validation_error(
        self, runner: CliRunner, valid_yaml: Path
    ) -> None:
        """Push with validation error returns exit code 1."""
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.json.return_value = {
            "detail": {
                "suite": None,
                "validation": {
                    "valid": False,
                    "errors": ["parse error"],
                    "warnings": [],
                },
                "filename": "suite.yaml",
            }
        }

        with patch("atp.cli.commands.push.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(
                return_value=mock_client
            )
            mock_httpx.Client.return_value.__exit__ = MagicMock(
                return_value=False
            )
            mock_client.post.return_value = mock_resp

            result = runner.invoke(
                push_command,
                [str(valid_yaml), "--server", "http://test:8000"],
            )
            assert result.exit_code == 1

    def test_push_dry_run(
        self, runner: CliRunner, valid_yaml: Path
    ) -> None:
        """Dry run shows files without uploading."""
        result = runner.invoke(
            push_command,
            [
                str(valid_yaml),
                "--server",
                "http://test:8000",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "suite.yaml" in result.output
```

- [ ] **Step 2: Implement push.py**

Create `atp/cli/commands/push.py`:

```python
"""CLI command: atp push — upload YAML test suites to remote server."""

from __future__ import annotations

from pathlib import Path

import click
import httpx

from atp.cli.commands.remote import (
    EXIT_FAILURE,
    EXIT_PARTIAL,
    EXIT_SUCCESS,
    file_sha256,
    find_yaml_files,
    load_manifest,
    now_iso,
    resolve_auth_headers,
    resolve_server_url,
    save_manifest,
)


@click.command(name="push")
@click.argument("files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--server", help="ATP server URL")
@click.option("--api-key", help="API key for authentication")
@click.option("--force", is_flag=True, help="Re-upload even if suite exists")
@click.option("--dry-run", is_flag=True, help="Show what would be pushed")
def push_command(
    files: tuple[Path, ...],
    server: str | None,
    api_key: str | None,
    force: bool,
    dry_run: bool,
) -> None:
    """Upload YAML test suite files to remote ATP server."""
    if not files:
        raise click.ClickException("No files specified")

    # Expand directories to YAML files
    all_files: list[Path] = []
    for f in files:
        if f.is_dir():
            all_files.extend(find_yaml_files(f))
        elif f.suffix in (".yaml", ".yml"):
            all_files.append(f)

    if not all_files:
        raise click.ClickException("No YAML files found")

    # Resolve server
    first_dir = all_files[0].parent
    server_url = resolve_server_url(server, first_dir)
    if not server_url:
        raise click.ClickException(
            "No server URL. Use --server, ATP_SERVER env var, "
            "or .atp-sync.json"
        )

    if dry_run:
        click.echo(f"Dry run — no changes will be made.")
        for f in all_files:
            click.echo(f"  push {f.name}")
        raise SystemExit(EXIT_SUCCESS)

    # Resolve auth
    headers = resolve_auth_headers(api_key=api_key, server_url=server_url)

    click.echo(f"Pushing {len(all_files)} file(s) to {server_url}...")

    succeeded = 0
    failed = 0

    with httpx.Client(timeout=30.0, headers=headers) as client:
        for filepath in all_files:
            result = _push_file(client, server_url, filepath, force)
            if result:
                succeeded += 1
                # Update manifest
                _update_manifest(
                    filepath, server_url, result
                )
            else:
                failed += 1

    click.echo(f"\n{succeeded} succeeded, {failed} failed")

    if failed == len(all_files):
        raise SystemExit(EXIT_FAILURE)
    elif failed > 0:
        raise SystemExit(EXIT_PARTIAL)
    raise SystemExit(EXIT_SUCCESS)


def _push_file(
    client: httpx.Client,
    server_url: str,
    filepath: Path,
    force: bool,
) -> dict | None:
    """Push a single file. Returns suite info dict or None on failure."""
    name = filepath.name
    content = filepath.read_bytes()

    resp = client.post(
        f"{server_url}/api/suite-definitions/upload",
        files={"file": (name, content, "application/yaml")},
    )

    if resp.status_code == 201:
        data = resp.json()
        suite = data.get("suite", {})
        validation = data.get("validation", {})
        warnings = validation.get("warnings", [])
        suite_id = suite.get("id", "?")
        msg = f"created (id={suite_id})"
        if warnings:
            msg += f", {len(warnings)} warning(s)"
        click.echo(f"  ✓ {name} → {msg}")
        return {"suite_id": suite_id}

    if resp.status_code == 409:
        if force:
            # TODO: delete and re-upload in sync_cmd, for push just warn
            click.echo(f"  ⚠ {name} → already exists (use atp sync for updates)")
        else:
            click.echo(f"  ⚠ {name} → already exists (skip)")
        return None

    if resp.status_code in (400, 413, 422):
        data = resp.json()
        detail = data.get("detail", data)
        if isinstance(detail, dict):
            validation = detail.get("validation", {})
            errors = validation.get("errors", [])
        else:
            errors = [str(detail)]
        click.echo(f"  ✗ {name} → {len(errors)} validation error(s)")
        for err in errors:
            click.echo(f"    - {err}")
        return None

    click.echo(f"  ✗ {name} → HTTP {resp.status_code}")
    return None


def _update_manifest(
    filepath: Path, server_url: str, result: dict
) -> None:
    """Update manifest file after successful push."""
    directory = filepath.parent
    manifest = load_manifest(directory)
    manifest["server"] = server_url
    manifest["files"][filepath.name] = {
        "sha256": file_sha256(filepath),
        "suite_id": result["suite_id"],
        "synced_at": now_iso(),
    }
    save_manifest(directory, manifest)
```

- [ ] **Step 3: Register in main.py**

In `atp/cli/main.py`, add import and registration:

```python
from atp.cli.commands.push import push_command
```

After the last `cli.add_command(...)`:

```python
cli.add_command(push_command)
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/unit/cli/test_push.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff**

Run: `uv run ruff format atp/cli/commands/push.py tests/unit/cli/test_push.py && uv run ruff check atp/cli/commands/push.py tests/unit/cli/test_push.py --fix`

- [ ] **Step 6: Commit**

```bash
git add atp/cli/commands/push.py tests/unit/cli/test_push.py atp/cli/main.py
git commit -m "feat(cli): add atp push command for uploading YAML suites"
```

---

### Task 3: Implement `atp pull` command

**Files:**
- Create: `atp/cli/commands/pull.py`
- Create: `tests/unit/cli/test_pull.py`
- Modify: `atp/cli/main.py`

- [ ] **Step 1: Write tests for pull**

Create `tests/unit/cli/test_pull.py`:

```python
"""Tests for atp pull command."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from atp.cli.commands.pull import pull_command


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestPullCommand:
    """Tests for atp pull."""

    def test_pull_all_suites(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Pull all suites from server."""
        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.json.return_value = {
            "items": [
                {"id": 1, "name": "suite-one", "version": "1.0", "description": None, "test_count": 2, "agent_count": 0},
            ],
            "total": 1, "limit": 50, "offset": 0,
        }

        yaml_resp = MagicMock()
        yaml_resp.status_code = 200
        yaml_resp.json.return_value = {"yaml_content": "test_suite: suite-one\ntests: []\n"}

        with patch("atp.cli.commands.pull.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = [list_resp, yaml_resp]

            result = runner.invoke(
                pull_command,
                ["--server", "http://test:8000", "--dir", str(tmp_path)],
            )
            assert result.exit_code == 0
            assert (tmp_path / "suite-one.yaml").exists()

    def test_pull_by_id(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Pull a specific suite by ID."""
        yaml_resp = MagicMock()
        yaml_resp.status_code = 200
        yaml_resp.json.return_value = {"yaml_content": "test_suite: my-suite\ntests: []\n"}

        with patch("atp.cli.commands.pull.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = yaml_resp

            result = runner.invoke(
                pull_command,
                ["--server", "http://test:8000", "--id", "5", "--dir", str(tmp_path)],
            )
            assert result.exit_code == 0

    def test_pull_skip_existing(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Pull skips existing files without --force."""
        (tmp_path / "suite-one.yaml").write_text("existing")

        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.json.return_value = {
            "items": [{"id": 1, "name": "suite-one", "version": "1.0", "description": None, "test_count": 2, "agent_count": 0}],
            "total": 1, "limit": 50, "offset": 0,
        }

        with patch("atp.cli.commands.pull.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = list_resp

            result = runner.invoke(
                pull_command,
                ["--server", "http://test:8000", "--dir", str(tmp_path)],
            )
            assert "skipped" in result.output.lower() or "skip" in result.output.lower()
```

- [ ] **Step 2: Implement pull.py**

Create `atp/cli/commands/pull.py`:

```python
"""CLI command: atp pull — download test suites from remote server."""

from __future__ import annotations

import re
from pathlib import Path

import click
import httpx

from atp.cli.commands.remote import (
    EXIT_FAILURE,
    EXIT_PARTIAL,
    EXIT_SUCCESS,
    file_sha256,
    load_manifest,
    now_iso,
    resolve_auth_headers,
    resolve_server_url,
    save_manifest,
)


def _sanitize_filename(name: str) -> str:
    """Sanitize suite name to valid filename."""
    sanitized = re.sub(r"[^\w\-.]", "-", name).strip("-")
    return sanitized or "unnamed"


@click.command(name="pull")
@click.option("--server", help="ATP server URL")
@click.option("--api-key", help="API key for authentication")
@click.option("--dir", "directory", type=click.Path(path_type=Path), default=".", help="Output directory")
@click.option("--id", "suite_id", type=int, help="Pull specific suite by ID")
@click.option("--all", "pull_all", is_flag=True, default=True, help="Pull all suites (default)")
@click.option("--force", is_flag=True, help="Overwrite existing files")
def pull_command(
    server: str | None,
    api_key: str | None,
    directory: Path,
    suite_id: int | None,
    pull_all: bool,
    force: bool,
) -> None:
    """Download test suites from remote ATP server."""
    server_url = resolve_server_url(server, directory)
    if not server_url:
        raise click.ClickException(
            "No server URL. Use --server, ATP_SERVER env var, "
            "or .atp-sync.json"
        )

    directory.mkdir(parents=True, exist_ok=True)
    headers = resolve_auth_headers(api_key=api_key, server_url=server_url)

    click.echo(f"Pulling suites from {server_url}...")

    pulled = 0
    skipped = 0
    failed = 0

    with httpx.Client(timeout=30.0, headers=headers) as client:
        if suite_id:
            suites = [{"id": suite_id, "name": f"suite-{suite_id}"}]
        else:
            # List all suites
            resp = client.get(
                f"{server_url}/api/suite-definitions",
                params={"limit": 100, "offset": 0},
            )
            if resp.status_code != 200:
                raise click.ClickException(
                    f"Failed to list suites: HTTP {resp.status_code}"
                )
            suites = resp.json().get("items", [])

        used_filenames: set[str] = set()

        for suite_info in suites:
            sid = suite_info["id"]
            name = suite_info.get("name", f"suite-{sid}")
            filename = _sanitize_filename(name) + ".yaml"

            # Handle filename collisions
            if filename in used_filenames:
                filename = f"{_sanitize_filename(name)}_{sid}.yaml"
            used_filenames.add(filename)

            target = directory / filename

            # Skip existing
            if target.exists() and not force:
                click.echo(f"  - skipped {filename} (exists, use --force)")
                skipped += 1
                continue

            # Export YAML
            resp = client.get(
                f"{server_url}/api/suite-definitions/{sid}/yaml"
            )
            if resp.status_code != 200:
                click.echo(f"  ✗ {filename} → HTTP {resp.status_code}")
                failed += 1
                continue

            data = resp.json()
            yaml_content = data.get("yaml_content", "")
            target.write_text(yaml_content)
            click.echo(f"  ✓ {filename} (id={sid})")
            pulled += 1

            # Update manifest
            manifest = load_manifest(directory)
            manifest["server"] = server_url
            manifest["files"][filename] = {
                "sha256": file_sha256(target),
                "suite_id": sid,
                "synced_at": now_iso(),
            }
            save_manifest(directory, manifest)

    click.echo(f"\n{pulled} pulled, {skipped} skipped, {failed} failed")

    if failed > 0 and pulled == 0:
        raise SystemExit(EXIT_FAILURE)
    elif failed > 0:
        raise SystemExit(EXIT_PARTIAL)
    raise SystemExit(EXIT_SUCCESS)
```

- [ ] **Step 3: Register in main.py**

Add to `atp/cli/main.py`:

```python
from atp.cli.commands.pull import pull_command
```

```python
cli.add_command(pull_command)
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/unit/cli/test_pull.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff**

Run: `uv run ruff format atp/cli/commands/pull.py tests/unit/cli/test_pull.py && uv run ruff check atp/cli/commands/pull.py tests/unit/cli/test_pull.py --fix`

- [ ] **Step 6: Commit**

```bash
git add atp/cli/commands/pull.py tests/unit/cli/test_pull.py atp/cli/main.py
git commit -m "feat(cli): add atp pull command for downloading suites from server"
```

---

### Task 4: Implement `atp sync` command

**Files:**
- Create: `atp/cli/commands/sync_cmd.py`
- Create: `tests/unit/cli/test_sync.py`
- Modify: `atp/cli/main.py`

- [ ] **Step 1: Write tests for sync**

Create `tests/unit/cli/test_sync.py`:

```python
"""Tests for atp sync command."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from atp.cli.commands.remote import MANIFEST_FILE, file_sha256
from atp.cli.commands.sync_cmd import sync_command


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_manifest(tmp_path: Path, server: str, files: dict) -> None:
    """Create a manifest file."""
    data = {"server": server, "last_sync": "", "files": files}
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(data))


class TestSyncCommand:
    """Tests for atp sync."""

    def test_sync_new_file(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """New file gets pushed."""
        (tmp_path / "new.yaml").write_text("test_suite: new\ntests: []\n")

        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "suite": {"id": 1, "name": "new"},
            "validation": {"valid": True, "errors": [], "warnings": []},
            "filename": "new.yaml",
        }

        with patch("atp.cli.commands.sync_cmd.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp

            result = runner.invoke(
                sync_command,
                [str(tmp_path), "--server", "http://test:8000"],
            )
            assert "created" in result.output

    def test_sync_unchanged_file(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Unchanged file gets skipped."""
        f = tmp_path / "existing.yaml"
        f.write_text("test_suite: existing\ntests: []\n")
        sha = file_sha256(f)

        _make_manifest(
            tmp_path, "http://test:8000",
            {"existing.yaml": {"sha256": sha, "suite_id": 1, "synced_at": ""}},
        )

        result = runner.invoke(
            sync_command,
            [str(tmp_path), "--server", "http://test:8000"],
        )
        assert "unchanged" in result.output

    def test_sync_deleted_file_warns(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Deleted file produces warning and is removed from manifest."""
        _make_manifest(
            tmp_path, "http://test:8000",
            {"gone.yaml": {"sha256": "old", "suite_id": 5, "synced_at": ""}},
        )

        result = runner.invoke(
            sync_command,
            [str(tmp_path), "--server", "http://test:8000"],
        )
        assert "removed locally" in result.output

        # Verify manifest updated
        manifest = json.loads((tmp_path / MANIFEST_FILE).read_text())
        assert "gone.yaml" not in manifest["files"]

    def test_sync_dry_run(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Dry run shows plan without executing."""
        (tmp_path / "new.yaml").write_text("test_suite: new\ntests: []\n")

        result = runner.invoke(
            sync_command,
            [str(tmp_path), "--server", "http://test:8000", "--dry-run"],
        )
        assert "Dry run" in result.output
        assert result.exit_code == 0
```

- [ ] **Step 2: Implement sync_cmd.py**

Create `atp/cli/commands/sync_cmd.py`:

```python
"""CLI command: atp sync — synchronize local YAML test suites with server."""

from __future__ import annotations

from pathlib import Path

import click
import httpx

from atp.cli.commands.remote import (
    EXIT_FAILURE,
    EXIT_PARTIAL,
    EXIT_SUCCESS,
    file_sha256,
    find_yaml_files,
    load_manifest,
    now_iso,
    resolve_auth_headers,
    resolve_server_url,
    save_manifest,
)


@click.command(name="sync")
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--server", help="ATP server URL")
@click.option("--api-key", help="API key for authentication")
@click.option("--dry-run", is_flag=True, help="Show what would happen")
def sync_command(
    directory: Path,
    server: str | None,
    api_key: str | None,
    dry_run: bool,
) -> None:
    """Synchronize a directory of YAML test suites with remote server."""
    server_url = resolve_server_url(server, directory)
    if not server_url:
        raise click.ClickException(
            "No server URL. Use --server, ATP_SERVER env var, "
            "or .atp-sync.json"
        )

    manifest = load_manifest(directory)
    manifest_files = manifest.get("files", {})
    local_files = find_yaml_files(directory)

    # Categorize files
    new_files: list[Path] = []
    changed_files: list[Path] = []
    unchanged_files: list[Path] = []
    deleted_names: list[str] = []

    local_names = {f.name for f in local_files}

    for f in local_files:
        entry = manifest_files.get(f.name)
        if entry is None:
            new_files.append(f)
        elif file_sha256(f) != entry.get("sha256"):
            changed_files.append(f)
        else:
            unchanged_files.append(f)

    for name in list(manifest_files.keys()):
        if name not in local_names:
            deleted_names.append(name)

    if dry_run:
        click.echo("Dry run — no changes will be made.")
        for f in new_files:
            click.echo(f"  push {f.name} (new)")
        for f in changed_files:
            click.echo(f"  push {f.name} (changed)")
        for f in unchanged_files:
            click.echo(f"  skip {f.name} (unchanged)")
        for name in deleted_names:
            click.echo(f"  warn {name} (removed locally)")
        raise SystemExit(EXIT_SUCCESS)

    click.echo(f"Syncing {directory} with {server_url}...")

    headers = resolve_auth_headers(api_key=api_key, server_url=server_url)
    created = 0
    updated = 0
    failed = 0

    with httpx.Client(timeout=30.0, headers=headers) as client:
        # Push new files
        for f in new_files:
            result = _upload_file(client, server_url, f)
            if result:
                manifest_files[f.name] = {
                    "sha256": file_sha256(f),
                    "suite_id": result["suite_id"],
                    "synced_at": now_iso(),
                }
                created += 1
                click.echo(f"  ✓ {f.name} → created (id={result['suite_id']})")
            else:
                failed += 1

        # Push changed files (delete + re-upload)
        for f in changed_files:
            entry = manifest_files.get(f.name, {})
            old_id = entry.get("suite_id")

            # Delete old
            if old_id:
                del_resp = client.delete(
                    f"{server_url}/api/suite-definitions/{old_id}"
                )
                if del_resp.status_code not in (200, 204, 404):
                    click.echo(
                        f"  ✗ {f.name} → failed to delete old "
                        f"(HTTP {del_resp.status_code})"
                    )
                    failed += 1
                    continue

            # Re-upload
            result = _upload_file(client, server_url, f)
            if result:
                manifest_files[f.name] = {
                    "sha256": file_sha256(f),
                    "suite_id": result["suite_id"],
                    "synced_at": now_iso(),
                }
                updated += 1
                click.echo(f"  ✓ {f.name} → updated (id={result['suite_id']})")
            else:
                click.echo(
                    f"  ✗ {f.name} → DELETE succeeded but upload failed!"
                    f"\n    Restore with: atp push {f.name} --force"
                )
                failed += 1

    # Handle deleted files
    for name in deleted_names:
        click.echo(f"  ⚠ {name} → removed locally (cleared from manifest)")
        manifest_files.pop(name, None)

    # Unchanged
    for f in unchanged_files:
        click.echo(f"  ✓ {f.name} → unchanged, skipped")

    # Save manifest
    manifest["server"] = server_url
    manifest["files"] = manifest_files
    save_manifest(directory, manifest)

    click.echo(
        f"\n{created} created, {updated} updated, "
        f"{len(unchanged_files)} unchanged, "
        f"{len(deleted_names)} removed locally"
    )

    if failed > 0 and created + updated == 0:
        raise SystemExit(EXIT_FAILURE)
    elif failed > 0:
        raise SystemExit(EXIT_PARTIAL)
    raise SystemExit(EXIT_SUCCESS)


def _upload_file(
    client: httpx.Client,
    server_url: str,
    filepath: Path,
) -> dict | None:
    """Upload a single file. Returns {"suite_id": N} or None."""
    resp = client.post(
        f"{server_url}/api/suite-definitions/upload",
        files={"file": (filepath.name, filepath.read_bytes(), "application/yaml")},
    )
    if resp.status_code == 201:
        data = resp.json()
        suite = data.get("suite", {})
        return {"suite_id": suite.get("id")}

    if resp.status_code in (400, 409, 413, 422):
        data = resp.json()
        detail = data.get("detail", data)
        if isinstance(detail, dict):
            errors = detail.get("validation", {}).get("errors", [])
        else:
            errors = [str(detail)]
        click.echo(f"  ✗ {filepath.name} → {len(errors)} error(s)")
        for err in errors:
            click.echo(f"    - {err}")
    else:
        click.echo(f"  ✗ {filepath.name} → HTTP {resp.status_code}")

    return None
```

- [ ] **Step 3: Register in main.py**

Add to `atp/cli/main.py`:

```python
from atp.cli.commands.sync_cmd import sync_command
```

```python
cli.add_command(sync_command)
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/unit/cli/test_sync.py -v`
Expected: All PASS

- [ ] **Step 5: Run full CLI test suite**

Run: `uv run python -m pytest tests/unit/cli/ -v`
Expected: All PASS

- [ ] **Step 6: Run ruff + pyrefly**

Run: `uv run ruff format atp/cli/commands/ tests/unit/cli/ && uv run ruff check atp/cli/commands/ tests/unit/cli/ --fix && uv run pyrefly check`

- [ ] **Step 7: Commit**

```bash
git add atp/cli/commands/sync_cmd.py tests/unit/cli/test_sync.py atp/cli/main.py
git commit -m "feat(cli): add atp sync command for directory synchronization"
```

---

### Task 5: Integration verification

- [ ] **Step 1: Verify all commands registered**

Run: `uv run atp --help | grep -E "push|pull|sync"`
Expected: All three commands listed

- [ ] **Step 2: Run full test suite**

Run: `uv run python -m pytest tests/unit/cli/ -v`
Expected: All PASS

- [ ] **Step 3: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 4: Commit any formatting changes**

```bash
git add -u && git diff --cached --stat
git commit -m "style: format and lint fixes for CLI push/pull/sync"
```
