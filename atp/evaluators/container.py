"""Container runtime for isolated code execution.

Provides Docker and Podman backends for running evaluator commands
inside containers. Falls back to subprocess + rlimits if no runtime
is available.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

from atp.evaluators.code_exec import CommandResult

logger = logging.getLogger(__name__)


class ContainerRuntime(Protocol):
    """Protocol for container runtimes."""

    async def run(
        self,
        image: str,
        command: list[str],
        workspace: Path | None = None,
        timeout: int = 300,
        env: dict[str, str] | None = None,
        memory: str = "512m",
        cpus: str = "1",
    ) -> CommandResult:
        """Run a command inside a container."""
        ...

    async def is_available(self) -> bool:
        """Check if the runtime is available on this system."""
        ...


class _CLIContainerRuntime:
    """Base class for CLI-based container runtimes (Docker/Podman)."""

    def __init__(self, executable: str) -> None:
        self._executable = executable

    async def is_available(self) -> bool:
        """Check if the CLI executable exists and works."""
        if shutil.which(self._executable) is None:
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                self._executable,
                "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.wait(), timeout=5)
            return proc.returncode == 0
        except Exception:
            return False

    async def run(
        self,
        image: str,
        command: list[str],
        workspace: Path | None = None,
        timeout: int = 300,
        env: dict[str, str] | None = None,
        memory: str = "512m",
        cpus: str = "1",
    ) -> CommandResult:
        """Run command in a container with workspace copy-in."""
        tmp_dir: Path | None = None

        try:
            # Build docker/podman run command
            cmd = [
                self._executable,
                "run",
                "--rm",
                "--network=none",
                "--read-only",
                f"--memory={memory}",
                f"--cpus={cpus}",
            ]

            # Copy workspace to temp dir and mount
            if workspace is not None and workspace.exists():
                tmp_dir = Path(tempfile.mkdtemp(prefix="atp-sandbox-"))
                shutil.copytree(workspace, tmp_dir / "work", dirs_exist_ok=True)
                work_dir = tmp_dir / "work"
                cmd.extend(
                    [
                        "-v",
                        f"{work_dir}:/work",
                        "-w",
                        "/work",
                        "--tmpfs",
                        "/tmp:rw,noexec,nosuid,size=64m",
                    ]
                )

            # Environment variables
            for key, value in (env or {}).items():
                cmd.extend(["-e", f"{key}={value}"])

            # Image and command
            cmd.append(image)
            cmd.extend(command)

            logger.debug("Container command: %s", " ".join(cmd))

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                return CommandResult(
                    return_code=process.returncode or 0,
                    stdout=stdout_bytes.decode("utf-8", errors="replace"),
                    stderr=stderr_bytes.decode("utf-8", errors="replace"),
                    timed_out=False,
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                return CommandResult(
                    return_code=-1,
                    stdout="",
                    stderr=(f"Container timed out after {timeout}s"),
                    timed_out=True,
                )

        finally:
            if tmp_dir is not None and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)


class DockerRuntime(_CLIContainerRuntime):
    """Docker-based container runtime."""

    def __init__(self) -> None:
        super().__init__("docker")


class PodmanRuntime(_CLIContainerRuntime):
    """Podman-based container runtime."""

    def __init__(self) -> None:
        super().__init__("podman")


async def detect_runtime(
    preference: str = "auto",
) -> ContainerRuntime | None:
    """Detect available container runtime.

    Args:
        preference: "auto" (try docker then podman), "docker",
                    "podman", or "none" (disable containers).

    Returns:
        A ContainerRuntime instance, or None if unavailable.
    """
    if preference == "none":
        return None

    if preference in ("docker", "auto"):
        runtime = DockerRuntime()
        if await runtime.is_available():
            logger.info("Container runtime: docker")
            return runtime
        if preference == "docker":
            logger.warning("Docker requested but not available")
            return None

    if preference in ("podman", "auto"):
        runtime = PodmanRuntime()
        if await runtime.is_available():
            logger.info("Container runtime: podman")
            return runtime
        if preference == "podman":
            logger.warning("Podman requested but not available")
            return None

    logger.info("No container runtime available, using subprocess sandbox")
    return None
