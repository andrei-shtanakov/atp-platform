"""Sandbox management for isolated test execution."""

import logging
import os
import shutil
import stat
import uuid
from pathlib import Path

from atp.core.security import (
    SecurityEventType,
    log_security_event,
    sanitize_filename,
    validate_path_within_workspace,
)
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

        Security: Uses full UUID and restrictive permissions to prevent
        sandbox escape and collision attacks.

        Args:
            test_id: Optional test identifier for the sandbox.

        Returns:
            Unique sandbox identifier.

        Raises:
            SandboxError: If sandbox creation fails.
        """
        # Use full UUID to prevent collision attacks
        sandbox_id = f"sandbox-{uuid.uuid4().hex}"

        try:
            sandbox_path = self.base_dir / sandbox_id
            # Create with restrictive permissions (owner only)
            sandbox_path.mkdir(parents=True, exist_ok=True, mode=0o700)

            # Create workspace directory inside sandbox
            workspace = sandbox_path / "workspace"
            workspace.mkdir(exist_ok=True, mode=0o700)

            # Create other standard directories
            (sandbox_path / "logs").mkdir(exist_ok=True, mode=0o700)
            (sandbox_path / "artifacts").mkdir(exist_ok=True, mode=0o700)

            # Ensure permissions are set correctly (mkdir mode can be affected by umask)
            try:
                os.chmod(sandbox_path, stat.S_IRWXU)  # 0o700
                os.chmod(workspace, stat.S_IRWXU)
                os.chmod(sandbox_path / "logs", stat.S_IRWXU)
                os.chmod(sandbox_path / "artifacts", stat.S_IRWXU)
            except OSError:
                pass  # May fail on some systems

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

    def populate_workspace(
        self,
        sandbox_id: str,
        fixture_path: str | Path,
    ) -> None:
        """
        Copy fixture directory contents into sandbox workspace.

        Args:
            sandbox_id: Sandbox identifier.
            fixture_path: Path to the fixture directory to copy.

        Raises:
            SandboxError: If fixture path is invalid or copy fails.
        """
        fixture = Path(fixture_path).resolve()
        if not fixture.exists():
            raise SandboxError(
                f"Fixture path not found: {fixture}",
                sandbox_id=sandbox_id,
            )
        if not fixture.is_dir():
            raise SandboxError(
                f"Fixture path is not a directory: {fixture}",
                sandbox_id=sandbox_id,
            )

        workspace = self.get_workspace(sandbox_id)
        try:
            shutil.copytree(fixture, workspace, dirs_exist_ok=True)
            logger.debug(
                "Populated workspace %s from fixture %s",
                workspace,
                fixture,
            )
        except OSError as e:
            raise SandboxError(
                f"Failed to populate workspace from fixture: {e}",
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

    def validate_path(self, sandbox_id: str, path: str | Path) -> Path:
        """
        Validate that a path is safely within the sandbox workspace.

        Security: Prevents path traversal attacks by ensuring the resolved
        path stays within the sandbox workspace directory.

        Args:
            sandbox_id: Sandbox identifier.
            path: Path to validate (relative to workspace).

        Returns:
            Resolved Path object within workspace.

        Raises:
            SandboxError: If path escapes sandbox or sandbox not found.
        """
        workspace = self.get_workspace(sandbox_id)
        try:
            return validate_path_within_workspace(path, workspace)
        except Exception as e:
            log_security_event(
                SecurityEventType.SANDBOX_VIOLATION,
                f"Invalid path in sandbox {sandbox_id}: {path}",
                field="path",
            )
            raise SandboxError(
                f"Invalid path in sandbox: {e}",
                sandbox_id=sandbox_id,
            ) from e

    def safe_write_file(
        self,
        sandbox_id: str,
        path: str | Path,
        content: str | bytes,
    ) -> Path:
        """
        Safely write a file within the sandbox workspace.

        Security: Validates path and creates parent directories safely.

        Args:
            sandbox_id: Sandbox identifier.
            path: File path relative to workspace.
            content: Content to write.

        Returns:
            Path to the written file.

        Raises:
            SandboxError: If path is invalid or write fails.
        """
        validated_path = self.validate_path(sandbox_id, path)

        try:
            # Create parent directories if needed
            validated_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file with appropriate mode
            if isinstance(content, bytes):
                validated_path.write_bytes(content)
            else:
                validated_path.write_text(content)

            # Set restrictive permissions
            os.chmod(validated_path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600

            return validated_path

        except OSError as e:
            raise SandboxError(
                f"Failed to write file in sandbox: {e}",
                sandbox_id=sandbox_id,
            ) from e

    def safe_read_file(
        self,
        sandbox_id: str,
        path: str | Path,
        binary: bool = False,
    ) -> str | bytes:
        """
        Safely read a file from the sandbox workspace.

        Security: Validates path before reading.

        Args:
            sandbox_id: Sandbox identifier.
            path: File path relative to workspace.
            binary: Whether to read as binary.

        Returns:
            File contents.

        Raises:
            SandboxError: If path is invalid or read fails.
        """
        validated_path = self.validate_path(sandbox_id, path)

        if not validated_path.exists():
            raise SandboxError(
                f"File not found in sandbox: {path}",
                sandbox_id=sandbox_id,
            )

        try:
            if binary:
                return validated_path.read_bytes()
            return validated_path.read_text()
        except OSError as e:
            raise SandboxError(
                f"Failed to read file in sandbox: {e}",
                sandbox_id=sandbox_id,
            ) from e

    def save_artifact(
        self,
        sandbox_id: str,
        name: str,
        content: str | bytes,
    ) -> Path:
        """
        Save an artifact to the sandbox artifacts directory.

        Security: Sanitizes filename and validates path.

        Args:
            sandbox_id: Sandbox identifier.
            name: Artifact name (will be sanitized).
            content: Artifact content.

        Returns:
            Path to the saved artifact.

        Raises:
            SandboxError: If save fails.
        """
        if sandbox_id not in self._active_sandboxes:
            raise SandboxError(
                f"Sandbox not found: {sandbox_id}",
                sandbox_id=sandbox_id,
            )

        # Sanitize the artifact name
        safe_name = sanitize_filename(name)
        artifacts_dir = self._active_sandboxes[sandbox_id] / "artifacts"
        artifact_path = artifacts_dir / safe_name

        try:
            if isinstance(content, bytes):
                artifact_path.write_bytes(content)
            else:
                artifact_path.write_text(content)

            os.chmod(artifact_path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
            return artifact_path

        except OSError as e:
            raise SandboxError(
                f"Failed to save artifact: {e}",
                sandbox_id=sandbox_id,
            ) from e

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
