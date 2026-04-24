# Bugfix Plan: CLIAdapter + Test Dedup + Config Drift

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix CLIAdapter file ownership bugs, deduplicate dashboard tests, clean config drift.

**Architecture:** Three independent fix groups, each committable separately.

**Source:** `atp-issues.md` findings, items 1-3 by priority.

---

## Group A: CLIAdapter file-mode fixes

### Task 1: Fix cleanup() — stop deleting user-provided files

**Files:**
- Modify: `packages/atp-adapters/atp/adapters/cli.py`

The `cleanup()` method (lines 487-522) has two bugs:
1. `_write_input_file()` sets `self._temp_input_file` even for user-provided paths (line 159), so cleanup deletes them
2. Lines 507-522 unconditionally delete user-provided `input_file`/`output_file` — the comment says "only if we created them" but the code doesn't check

**Fix:**

- [ ] **Step 1: Add ownership tracking flag**

In `_write_input_file()` (line 135), track whether we created the file. Change the method to:

```python
async def _write_input_file(self, request: ATPRequest) -> Path:
    """Write request to input file securely."""
    if self._config.input_file:
        input_path = Path(self._config.input_file)
        self._owns_input_file = False
    else:
        fd, temp_path = tempfile.mkstemp(
            prefix="atp_request_",
            suffix=".json",
        )
        input_path = Path(temp_path)
        os.close(fd)
        self._owns_input_file = True

    input_path.write_text(request.model_dump_json())
    try:
        os.chmod(input_path, 0o600)
    except OSError:
        pass

    self._temp_input_file = input_path
    return input_path
```

- [ ] **Step 2: Fix cleanup() to respect ownership**

Replace `cleanup()` (lines 487-522) with:

```python
async def cleanup(self) -> None:
    """Clean up temporary files created by this adapter.

    Only deletes files the adapter itself created. User-provided
    input_file/output_file paths are never deleted.
    """
    if hasattr(self, "_temp_input_file") and self._temp_input_file:
        if getattr(self, "_owns_input_file", True):
            try:
                if self._temp_input_file.exists():
                    self._temp_input_file.unlink(missing_ok=True)
            except OSError:
                pass
        self._temp_input_file = None

    if hasattr(self, "_temp_output_file") and self._temp_output_file:
        try:
            if self._temp_output_file.exists():
                self._temp_output_file.unlink(missing_ok=True)
        except OSError:
            pass
        self._temp_output_file = None
```

This removes the second block (lines 507-522) that unconditionally deleted user files.

- [ ] **Step 3: Remove dead code `_create_temp_output_file()`**

Delete the `_create_temp_output_file()` method (lines 184-197). It is never called anywhere.

- [ ] **Step 4: Write tests**

Add to the existing CLI adapter tests (find the test file with `grep -r "class.*CLIAdapter" tests/`). If no test file exists for cleanup, create `tests/unit/adapters/test_cli_cleanup.py`:

```python
"""Tests for CLIAdapter file cleanup behavior."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.adapters.cli import CLIAdapter, CLIAdapterConfig
from atp.protocol import ATPRequest


@pytest.fixture
def request_obj() -> ATPRequest:
    """Minimal ATPRequest for testing."""
    return ATPRequest(task_id="test-1", task={"description": "test"})


class TestCleanupOwnership:
    """Tests that cleanup respects file ownership."""

    @pytest.mark.anyio
    async def test_cleanup_deletes_temp_input_file(self, tmp_path, request_obj):
        """Adapter-created temp files ARE deleted."""
        config = CLIAdapterConfig(command="echo test", input_format="file")
        adapter = CLIAdapter(config)

        path = await adapter._write_input_file(request_obj)
        assert path.exists()
        assert adapter._owns_input_file is True

        await adapter.cleanup()
        assert not path.exists()

    @pytest.mark.anyio
    async def test_cleanup_preserves_user_input_file(self, tmp_path, request_obj):
        """User-provided input files are NOT deleted."""
        user_file = tmp_path / "my_input.json"
        user_file.touch()

        config = CLIAdapterConfig(
            command="echo test",
            input_format="file",
            input_file=str(user_file),
        )
        adapter = CLIAdapter(config)

        await adapter._write_input_file(request_obj)
        assert adapter._owns_input_file is False

        await adapter.cleanup()
        assert user_file.exists(), "User-provided file must not be deleted"

    @pytest.mark.anyio
    async def test_cleanup_idempotent(self, request_obj):
        """Calling cleanup twice does not raise."""
        config = CLIAdapterConfig(command="echo test", input_format="file")
        adapter = CLIAdapter(config)

        await adapter._write_input_file(request_obj)
        await adapter.cleanup()
        await adapter.cleanup()  # Should not raise
```

- [ ] **Step 5: Run tests**

```bash
uv run python -m pytest tests/unit/adapters/test_cli_cleanup.py -v
```

- [ ] **Step 6: Commit**

```bash
git add packages/atp-adapters/atp/adapters/cli.py tests/unit/adapters/test_cli_cleanup.py
git commit -m "fix(cli): stop deleting user-provided files in cleanup, remove dead code"
```

---

## Group B: Deduplicate dashboard tests

### Task 2: Remove duplicate tests from packages/atp-dashboard

**Files:**
- Delete: `packages/atp-dashboard/tests/unit/dashboard/` (entire directory)
- Keep: `tests/unit/dashboard/` (canonical source, has 5 extra files + conftest)

Current state:
- 35 files are identical between both locations
- 5 files have diverged: `auth/sso/test_routes.py`, `test_api.py`, `test_comparison_endpoints.py`, `test_leaderboard_endpoint.py`, `test_storage.py`
- Root `tests/` has 6 extra files not in package copy (conftest.py, test_device_auth_routes.py, test_device_flow.py, test_ui_routes.py, test_upload.py, v2/catalog/test_routes.py)
- Only root `tests/` is in pytest testpaths — package copy never runs

- [ ] **Step 1: Check diverged files for anything worth keeping**

For each of the 5 diverged files, diff root vs package version. If the package version has useful changes not in root, merge them into root first.

```bash
for f in auth/sso/test_routes.py test_api.py test_comparison_endpoints.py test_leaderboard_endpoint.py test_storage.py; do
  echo "=== $f ==="
  diff tests/unit/dashboard/$f packages/atp-dashboard/tests/unit/dashboard/$f | head -20
done
```

- [ ] **Step 2: Delete the package test directory**

```bash
rm -rf packages/atp-dashboard/tests/unit/dashboard/
```

Keep `packages/atp-dashboard/tests/` directory with just `__init__.py` if other package tests exist there, or remove entirely if empty.

- [ ] **Step 3: Verify tests still pass**

```bash
uv run python -m pytest tests/unit/dashboard/ -v --tb=short 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add -A packages/atp-dashboard/tests/
git commit -m "fix: remove duplicate dashboard tests — canonical source is tests/unit/dashboard/"
```

---

## Group C: Config drift cleanup

### Task 3: Fix .gitignore and ruff config

**Files:**
- Modify: `.gitignore`
- Modify: `pyproject.toml`

- [ ] **Step 1: Fix .gitignore patterns**

Current `.gitignore` line 10: `.atp-dashboard.db` — but actual file is `atp_dashboard.db` (no dot, underscore not hyphen). Also missing `coverage.json`.

Add/fix these lines in `.gitignore` under the SQLite section:

```
# SQLite databases (Alembic local dev)
.atp-dashboard.db
.atp-analytics.db
atp_dashboard.db
*.coverage.json
```

- [ ] **Step 2: Remove nonexistent file from ruff config**

In `pyproject.toml`, remove line 129 (`"atp/dashboard/app.py" = ["E501"]`). File doesn't exist.

- [ ] **Step 3: Remove tracked artifacts from git**

```bash
git rm --cached atp_dashboard.db 2>/dev/null || true
git rm --cached atp-games/coverage.json 2>/dev/null || true
git rm --cached game-environments/coverage.json 2>/dev/null || true
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore pyproject.toml
git commit -m "fix: clean config drift — .gitignore patterns, remove dead ruff ignore"
```

---

## Future tasks (not in this session)

### Task 4: CodeExecEvaluator sandbox (Priority: Medium)
Pass `sandboxed=True` in `_check_pytest`, `_check_npm`, `_check_custom_command`, `_check_lint`. Add tests verifying sandbox is used in public API paths.

### Task 5: Auth state → shared storage (Priority: Low for now)
Replace in-memory `DeviceFlowStore` and SSO state with SQLite-backed storage. Clean up `DeviceFlowStore.create()` dead code.

### Task 6: Package dependency cleanup (Priority: Low)
Break circular `atp-platform ↔ atp-dashboard` dependency. Requires careful extraction of shared types.
