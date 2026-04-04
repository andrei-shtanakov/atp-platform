"""Tests for CLIAdapter file cleanup behavior."""

from pathlib import Path

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
    async def test_cleanup_deletes_temp_input_file(
        self, request_obj: ATPRequest
    ) -> None:
        """Adapter-created temp files ARE deleted."""
        config = CLIAdapterConfig(command="echo test", input_format="file")
        adapter = CLIAdapter(config)

        path = await adapter._write_input_file(request_obj)
        assert path.exists()
        assert adapter._owns_input_file is True

        await adapter.cleanup()
        assert not path.exists()

    @pytest.mark.anyio
    async def test_cleanup_preserves_user_input_file(
        self, tmp_path: Path, request_obj: ATPRequest
    ) -> None:
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
    async def test_cleanup_idempotent(self, request_obj: ATPRequest) -> None:
        """Calling cleanup twice does not raise."""
        config = CLIAdapterConfig(command="echo test", input_format="file")
        adapter = CLIAdapter(config)

        await adapter._write_input_file(request_obj)
        await adapter.cleanup()
        await adapter.cleanup()  # Should not raise
