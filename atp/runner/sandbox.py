"""Sandbox management for isolated test execution."""

import logging
import shutil
import uuid
from pathlib import Path

from atp.runner.exceptions import SandboxError
from atp.runner.models import SandboxConfig

logger = logging.getLogger(__name__)


class SandboxManager:
    """
    Manages sandbox environments for isolated test execution.

    In the current implementation, provides workspace isolation through
    temporary directories. Future versions may add Docker container
    support for full isolation.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        base_dir: Path | None = None,
    ) -> None:
        """
        Initialize sandbox manager.

        Args:
            config: Sandbox configuration. Uses defaults if not provided.
            base_dir: Base directory for creating sandboxes.
                     Uses system temp dir if not specified.
        """
        self.config = config or SandboxConfig()
        self._base_dir = base_dir
        self._active_sandboxes: dict[str, Path] = {}

    @property
    def base_dir(self) -> Path:
        """Get base directory for sandboxes."""
        if self._base_dir is None:
            import tempfile

            self._base_dir = Path(tempfile.gettempdir()) / "atp-sandboxes"
        return self._base_dir

    def create(self, test_id: str | None = None) -> str:
        """
        Create a new sandbox environment.

        Args:
            test_id: Optional test identifier for the sandbox.

        Returns:
            Unique sandbox identifier.

        Raises:
            SandboxError: If sandbox creation fails.
        """
        sandbox_id = f"sandbox-{uuid.uuid4().hex[:8]}"

        try:
            sandbox_path = self.base_dir / sandbox_id
            sandbox_path.mkdir(parents=True, exist_ok=True)

            # Create workspace directory inside sandbox
            workspace = sandbox_path / "workspace"
            workspace.mkdir(exist_ok=True)

            # Create other standard directories
            (sandbox_path / "logs").mkdir(exist_ok=True)
            (sandbox_path / "artifacts").mkdir(exist_ok=True)

            self._active_sandboxes[sandbox_id] = sandbox_path

            logger.debug(
                "Created sandbox %s for test %s at %s",
                sandbox_id,
                test_id,
                sandbox_path,
            )

            return sandbox_id

        except OSError as e:
            raise SandboxError(
                f"Failed to create sandbox: {e}",
                test_id=test_id,
                sandbox_id=sandbox_id,
                cause=e,
            ) from e

    def get_workspace(self, sandbox_id: str) -> Path:
        """
        Get workspace path for a sandbox.

        Args:
            sandbox_id: Sandbox identifier.

        Returns:
            Path to workspace directory.

        Raises:
            SandboxError: If sandbox not found.
        """
        if sandbox_id not in self._active_sandboxes:
            raise SandboxError(
                f"Sandbox not found: {sandbox_id}",
                sandbox_id=sandbox_id,
            )
        return self._active_sandboxes[sandbox_id] / "workspace"

    def get_logs_dir(self, sandbox_id: str) -> Path:
        """
        Get logs directory for a sandbox.

        Args:
            sandbox_id: Sandbox identifier.

        Returns:
            Path to logs directory.

        Raises:
            SandboxError: If sandbox not found.
        """
        if sandbox_id not in self._active_sandboxes:
            raise SandboxError(
                f"Sandbox not found: {sandbox_id}",
                sandbox_id=sandbox_id,
            )
        return self._active_sandboxes[sandbox_id] / "logs"

    def get_artifacts_dir(self, sandbox_id: str) -> Path:
        """
        Get artifacts directory for a sandbox.

        Args:
            sandbox_id: Sandbox identifier.

        Returns:
            Path to artifacts directory.

        Raises:
            SandboxError: If sandbox not found.
        """
        if sandbox_id not in self._active_sandboxes:
            raise SandboxError(
                f"Sandbox not found: {sandbox_id}",
                sandbox_id=sandbox_id,
            )
        return self._active_sandboxes[sandbox_id] / "artifacts"

    def cleanup(self, sandbox_id: str) -> None:
        """
        Clean up a sandbox environment.

        Args:
            sandbox_id: Sandbox identifier.

        Raises:
            SandboxError: If cleanup fails.
        """
        if sandbox_id not in self._active_sandboxes:
            logger.warning("Attempted to cleanup unknown sandbox: %s", sandbox_id)
            return

        sandbox_path = self._active_sandboxes.pop(sandbox_id)

        try:
            if sandbox_path.exists():
                shutil.rmtree(sandbox_path)
                logger.debug("Cleaned up sandbox %s at %s", sandbox_id, sandbox_path)
        except OSError as e:
            raise SandboxError(
                f"Failed to cleanup sandbox: {e}",
                sandbox_id=sandbox_id,
                cause=e,
            ) from e

    def cleanup_all(self) -> None:
        """Clean up all active sandboxes."""
        sandbox_ids = list(self._active_sandboxes.keys())
        for sandbox_id in sandbox_ids:
            try:
                self.cleanup(sandbox_id)
            except SandboxError as e:
                logger.warning("Failed to cleanup sandbox %s: %s", sandbox_id, e)

    def is_active(self, sandbox_id: str) -> bool:
        """
        Check if a sandbox is active.

        Args:
            sandbox_id: Sandbox identifier.

        Returns:
            True if sandbox is active, False otherwise.
        """
        return sandbox_id in self._active_sandboxes

    def list_active(self) -> list[str]:
        """
        List all active sandbox identifiers.

        Returns:
            List of active sandbox IDs.
        """
        return list(self._active_sandboxes.keys())

    async def __aenter__(self) -> "SandboxManager":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit - cleanup all sandboxes."""
        self.cleanup_all()
