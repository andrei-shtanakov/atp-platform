"""Tests for CLIAdapter cleanup behavior."""

import pytest

from atp.adapters.cli import CLIAdapter, CLIAdapterConfig
from atp.protocol import ATPRequest


@pytest.fixture
def request_obj() -> ATPRequest:
    """Minimal ATPRequest for testing."""
    return ATPRequest(task_id="test-1", task={"description": "test"})


class TestCleanupOwnership:
    """Tests that cleanup is a safe no-op in stdin/stdout mode."""

    @pytest.mark.anyio
    async def test_cleanup_is_noop(self) -> None:
        """Cleanup does nothing — stdin/stdout mode has no temp files."""
        config = CLIAdapterConfig(command="echo test")
        adapter = CLIAdapter(config)
        await adapter.cleanup()  # Should not raise

    @pytest.mark.anyio
    async def test_cleanup_idempotent(self) -> None:
        """Calling cleanup multiple times does not raise."""
        config = CLIAdapterConfig(command="echo test")
        adapter = CLIAdapter(config)
        await adapter.cleanup()
        await adapter.cleanup()  # Should not raise
