"""Tests for sandbox manager."""

import pytest

from atp.runner.exceptions import SandboxError
from atp.runner.models import SandboxConfig
from atp.runner.sandbox import SandboxManager


class TestSandboxManager:
    """Tests for SandboxManager."""

    @pytest.fixture
    def manager(self, tmp_path) -> SandboxManager:
        """Create a sandbox manager with temp base dir."""
        return SandboxManager(base_dir=tmp_path)

    @pytest.mark.anyio
    async def test_create_sandbox(self, manager: SandboxManager) -> None:
        """Create a sandbox successfully."""
        sandbox_id = await manager.create()
        assert sandbox_id.startswith("sandbox-")
        assert manager.is_active(sandbox_id)

    @pytest.mark.anyio
    async def test_create_sandbox_with_test_id(self, manager: SandboxManager) -> None:
        """Create a sandbox with test_id."""
        sandbox_id = await manager.create(test_id="test-001")
        assert manager.is_active(sandbox_id)

    @pytest.mark.anyio
    async def test_get_workspace(self, manager: SandboxManager) -> None:
        """Get workspace path for sandbox."""
        sandbox_id = await manager.create()
        workspace = manager.get_workspace(sandbox_id)
        assert workspace.exists()
        assert workspace.name == "workspace"

    @pytest.mark.anyio
    async def test_get_logs_dir(self, manager: SandboxManager) -> None:
        """Get logs directory for sandbox."""
        sandbox_id = await manager.create()
        logs = manager.get_logs_dir(sandbox_id)
        assert logs.exists()
        assert logs.name == "logs"

    @pytest.mark.anyio
    async def test_get_artifacts_dir(self, manager: SandboxManager) -> None:
        """Get artifacts directory for sandbox."""
        sandbox_id = await manager.create()
        artifacts = manager.get_artifacts_dir(sandbox_id)
        assert artifacts.exists()
        assert artifacts.name == "artifacts"

    @pytest.mark.anyio
    async def test_cleanup_sandbox(self, manager: SandboxManager) -> None:
        """Cleanup a sandbox."""
        sandbox_id = await manager.create()
        workspace = manager.get_workspace(sandbox_id)
        assert workspace.exists()

        await manager.cleanup(sandbox_id)
        assert not manager.is_active(sandbox_id)
        assert not workspace.exists()

    @pytest.mark.anyio
    async def test_cleanup_unknown_sandbox(self, manager: SandboxManager) -> None:
        """Cleanup unknown sandbox is no-op."""
        await manager.cleanup("unknown-sandbox")

    @pytest.mark.anyio
    async def test_cleanup_all(self, manager: SandboxManager) -> None:
        """Cleanup all sandboxes."""
        ids = [await manager.create() for _ in range(3)]
        workspaces = [manager.get_workspace(id) for id in ids]

        assert all(w.exists() for w in workspaces)
        assert len(manager.list_active()) == 3

        await manager.cleanup_all()

        assert all(not w.exists() for w in workspaces)
        assert len(manager.list_active()) == 0

    def test_get_workspace_not_found(self, manager: SandboxManager) -> None:
        """Get workspace for unknown sandbox raises."""
        with pytest.raises(SandboxError, match="Sandbox not found"):
            manager.get_workspace("unknown")

    def test_get_logs_dir_not_found(self, manager: SandboxManager) -> None:
        """Get logs dir for unknown sandbox raises."""
        with pytest.raises(SandboxError):
            manager.get_logs_dir("unknown")

    def test_get_artifacts_dir_not_found(self, manager: SandboxManager) -> None:
        """Get artifacts dir for unknown sandbox raises."""
        with pytest.raises(SandboxError):
            manager.get_artifacts_dir("unknown")

    @pytest.mark.anyio
    async def test_list_active(self, manager: SandboxManager) -> None:
        """List active sandboxes."""
        assert manager.list_active() == []

        id1 = await manager.create()
        assert id1 in manager.list_active()

        id2 = await manager.create()
        assert set(manager.list_active()) == {id1, id2}

        await manager.cleanup(id1)
        assert manager.list_active() == [id2]

    @pytest.mark.anyio
    async def test_is_active(self, manager: SandboxManager) -> None:
        """Check if sandbox is active."""
        assert not manager.is_active("nonexistent")

        sandbox_id = await manager.create()
        assert manager.is_active(sandbox_id)

        await manager.cleanup(sandbox_id)
        assert not manager.is_active(sandbox_id)

    def test_with_custom_config(self, tmp_path) -> None:
        """Manager with custom config."""
        config = SandboxConfig(
            memory_limit="4Gi",
            enabled=True,
        )
        manager = SandboxManager(config=config, base_dir=tmp_path)
        assert manager.config.memory_limit == "4Gi"
        assert manager.config.enabled is True


class TestSandboxManagerContextManager:
    """Tests for SandboxManager context manager."""

    @pytest.mark.anyio
    async def test_async_context_manager(self, tmp_path) -> None:
        """Test async context manager cleanup."""
        manager = SandboxManager(base_dir=tmp_path)

        async with manager as mgr:
            sandbox_id = await mgr.create()
            workspace = mgr.get_workspace(sandbox_id)
            assert workspace.exists()

        assert not manager.is_active(sandbox_id)
        assert not workspace.exists()

    @pytest.mark.anyio
    async def test_context_manager_cleanup_on_exception(self, tmp_path) -> None:
        """Context manager cleans up even on exception."""
        manager = SandboxManager(base_dir=tmp_path)
        sandbox_id = None
        workspace = None

        with pytest.raises(RuntimeError):
            async with manager as mgr:
                sandbox_id = await mgr.create()
                workspace = mgr.get_workspace(sandbox_id)
                raise RuntimeError("Test error")

        assert sandbox_id is not None
        assert workspace is not None
        assert not manager.is_active(sandbox_id)
        assert not workspace.exists()
