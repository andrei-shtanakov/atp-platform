"""Tests for workspace fixture population."""

from pathlib import Path

import pytest

from atp.runner.exceptions import SandboxError
from atp.runner.sandbox import SandboxManager


class TestPopulateWorkspace:
    """Tests for SandboxManager.populate_workspace."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> SandboxManager:
        """Create a sandbox manager with temp base dir."""
        return SandboxManager(base_dir=tmp_path)

    @pytest.fixture
    def fixture_dir(self, tmp_path: Path) -> Path:
        """Create a sample fixture directory."""
        fixture = tmp_path / "fixtures" / "basic"
        fixture.mkdir(parents=True)
        (fixture / "readme.txt").write_text("Hello from fixture")
        (fixture / "data").mkdir()
        (fixture / "data" / "config.json").write_text('{"version": "1.0"}')
        return fixture

    def test_populate_workspace(
        self, manager: SandboxManager, fixture_dir: Path
    ) -> None:
        """Fixture files are copied into workspace."""
        sandbox_id = manager.create()
        manager.populate_workspace(sandbox_id, fixture_dir)

        workspace = manager.get_workspace(sandbox_id)
        assert (workspace / "readme.txt").exists()
        assert (workspace / "readme.txt").read_text() == "Hello from fixture"
        assert (workspace / "data" / "config.json").exists()
        assert "1.0" in (workspace / "data" / "config.json").read_text()

    def test_populate_preserves_directory_structure(
        self, manager: SandboxManager, tmp_path: Path
    ) -> None:
        """Nested directory structure is preserved."""
        fixture = tmp_path / "nested_fixture"
        fixture.mkdir()
        (fixture / "a").mkdir()
        (fixture / "a" / "b").mkdir()
        (fixture / "a" / "b" / "deep.txt").write_text("deep")

        sandbox_id = manager.create()
        manager.populate_workspace(sandbox_id, fixture)

        workspace = manager.get_workspace(sandbox_id)
        assert (workspace / "a" / "b" / "deep.txt").read_text() == "deep"

    def test_populate_nonexistent_fixture(self, manager: SandboxManager) -> None:
        """Raises SandboxError for nonexistent fixture path."""
        sandbox_id = manager.create()
        with pytest.raises(SandboxError, match="Fixture path not found"):
            manager.populate_workspace(sandbox_id, "/nonexistent/path")

    def test_populate_file_not_dir(
        self, manager: SandboxManager, tmp_path: Path
    ) -> None:
        """Raises SandboxError when fixture path is a file."""
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hello")
        sandbox_id = manager.create()
        with pytest.raises(SandboxError, match="not a directory"):
            manager.populate_workspace(sandbox_id, f)

    def test_populate_unknown_sandbox(
        self, manager: SandboxManager, fixture_dir: Path
    ) -> None:
        """Raises SandboxError for unknown sandbox."""
        with pytest.raises(SandboxError, match="Sandbox not found"):
            manager.populate_workspace("unknown-id", fixture_dir)

    def test_populate_does_not_destroy_existing_files(
        self, manager: SandboxManager, fixture_dir: Path
    ) -> None:
        """Existing workspace files are preserved."""
        sandbox_id = manager.create()
        workspace = manager.get_workspace(sandbox_id)

        # Pre-existing file
        (workspace / "existing.txt").write_text("already here")

        manager.populate_workspace(sandbox_id, fixture_dir)

        assert (workspace / "existing.txt").read_text() == "already here"
        assert (workspace / "readme.txt").exists()
