"""Container adapter for Docker-packaged agents."""

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

from pydantic import Field, field_validator

from atp.core.security import (
    sanitize_error_message,
    validate_cpu_limit,
    validate_docker_image,
    validate_docker_network,
    validate_volume_mount,
)
from atp.protocol import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
)

from .base import AdapterConfig, AgentAdapter
from .exceptions import (
    AdapterConnectionError,
    AdapterError,
    AdapterResponseError,
    AdapterTimeoutError,
)


class ContainerResources(AdapterConfig):
    """Resource limits for container."""

    memory: str = Field(default="2g", description="Memory limit (e.g., '2g', '512m')")
    cpu: str = Field(default="1", description="CPU limit (e.g., '1', '0.5')")

    @field_validator("cpu")
    @classmethod
    def validate_cpu(cls, v: str) -> str:
        """Validate CPU limit."""
        validate_cpu_limit(v)
        return v


class ContainerAdapterConfig(AdapterConfig):
    """Configuration for container adapter."""

    image: str = Field(..., description="Docker image name with tag")
    resources: ContainerResources = Field(
        default_factory=ContainerResources, description="Resource limits"
    )
    network: str = Field(
        default="none", description="Network mode (none, host, bridge)"
    )
    environment: dict[str, str] = Field(
        default_factory=dict, description="Environment variables for container"
    )
    volumes: dict[str, str] = Field(
        default_factory=dict, description="Volume mounts (host:container)"
    )
    working_dir: str | None = Field(
        None, description="Working directory inside container"
    )
    auto_remove: bool = Field(
        default=True, description="Automatically remove container after execution"
    )
    # Security settings
    allowed_volume_paths: list[str] = Field(
        default_factory=list,
        description="Allowed base paths for host volume mounts",
    )
    read_only_root: bool = Field(
        default=False,
        description="Mount root filesystem as read-only",
    )
    no_new_privileges: bool = Field(
        default=True,
        description="Prevent privilege escalation",
    )
    cap_drop: list[str] = Field(
        default_factory=lambda: ["ALL"],
        description="Linux capabilities to drop",
    )

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str) -> str:
        """Validate Docker image name."""
        return validate_docker_image(v)

    @field_validator("network")
    @classmethod
    def validate_network(cls, v: str) -> str:
        """Validate network mode."""
        return validate_docker_network(v)


class ContainerAdapter(AgentAdapter):
    """
    Adapter for Docker-packaged agents.

    Runs agents in Docker containers with:
    - ATP Request sent via stdin (JSON)
    - ATP Response received via stdout (JSON)
    - ATP Events received via stderr (JSONL)
    """

    def __init__(self, config: ContainerAdapterConfig) -> None:
        """
        Initialize container adapter.

        Args:
            config: Container adapter configuration with image name.
        """
        super().__init__(config)
        self._config: ContainerAdapterConfig = config
        self._current_container_id: str | None = None

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""
        return "container"

    def _build_docker_command(self) -> list[str]:
        """Build the docker run command arguments.

        Security: Applies resource limits, privilege restrictions,
        and capability drops for defense in depth.
        """
        cmd = ["docker", "run", "-i"]

        if self._config.auto_remove:
            cmd.append("--rm")

        # Resource limits
        cmd.extend(["--memory", self._config.resources.memory])
        cmd.extend(["--cpus", self._config.resources.cpu])

        # Memory swap limit (prevent swap abuse)
        cmd.extend(["--memory-swap", self._config.resources.memory])

        # PID limit to prevent fork bombs
        cmd.extend(["--pids-limit", "256"])

        # Network
        cmd.extend(["--network", self._config.network])

        # Security options
        if self._config.no_new_privileges:
            cmd.append("--security-opt=no-new-privileges:true")

        if self._config.read_only_root:
            cmd.append("--read-only")

        # Drop capabilities
        for cap in self._config.cap_drop:
            cmd.extend(["--cap-drop", cap])

        # Environment variables
        for key, value in self._config.environment.items():
            # Ensure proper escaping
            cmd.extend(["-e", f"{key}={value}"])

        # Volume mounts with validation
        allowed_paths = (
            [Path(p) for p in self._config.allowed_volume_paths]
            if self._config.allowed_volume_paths
            else None
        )
        for host_path, container_path in self._config.volumes.items():
            validated_host, validated_container = validate_volume_mount(
                host_path, container_path, allowed_paths
            )
            cmd.extend(["-v", f"{validated_host}:{validated_container}"])

        # Working directory
        if self._config.working_dir:
            cmd.extend(["-w", self._config.working_dir])

        # Image (already validated in config)
        cmd.append(self._config.image)

        return cmd

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """
        Execute a task in a Docker container.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse from the agent.

        Raises:
            AdapterConnectionError: If Docker is not available.
            AdapterTimeoutError: If execution times out.
            AdapterResponseError: If agent returns invalid response.
        """
        cmd = self._build_docker_command()
        request_json = request.model_dump_json()

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise AdapterConnectionError(
                "Docker command not found. Is Docker installed?",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e
        except OSError as e:
            raise AdapterConnectionError(
                f"Failed to start Docker: {e}",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=request_json.encode()),
                timeout=self._config.timeout_seconds,
            )
        except TimeoutError as e:
            process.kill()
            await process.wait()
            raise AdapterTimeoutError(
                f"Container execution timed out after {self._config.timeout_seconds}s",
                timeout_seconds=self._config.timeout_seconds,
                adapter_type=self.adapter_type,
            ) from e

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            # Sanitize error message (redacts secrets and file paths)
            error_msg = sanitize_error_message(error_msg, include_type=False)
            raise AdapterError(
                f"Container exited with code {process.returncode}: {error_msg}",
                adapter_type=self.adapter_type,
            )

        if not stdout:
            raise AdapterResponseError(
                "Container produced no output",
                adapter_type=self.adapter_type,
            )

        try:
            response_data = json.loads(stdout.decode())
            return ATPResponse.model_validate(response_data)
        except json.JSONDecodeError as e:
            raise AdapterResponseError(
                f"Invalid JSON response from container: {e}",
                response_body=stdout.decode()[:500],
                adapter_type=self.adapter_type,
            ) from e
        except ValueError as e:
            raise AdapterResponseError(
                f"Invalid ATP Response format: {e}",
                response_body=stdout.decode()[:500],
                adapter_type=self.adapter_type,
            ) from e

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """
        Execute a task with event streaming from container stderr.

        Events are read from stderr as JSONL (one JSON object per line).
        Final response is read from stdout.

        Args:
            request: ATP Request with task specification.

        Yields:
            ATPEvent objects from stderr.
            Final ATPResponse from stdout.

        Raises:
            AdapterConnectionError: If Docker is not available.
            AdapterTimeoutError: If execution times out.
            AdapterResponseError: If agent returns invalid response.
        """
        cmd = self._build_docker_command()
        request_json = request.model_dump_json()

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise AdapterConnectionError(
                "Docker command not found. Is Docker installed?",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e
        except OSError as e:
            raise AdapterConnectionError(
                f"Failed to start Docker: {e}",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

        # Write request to stdin
        if process.stdin:
            process.stdin.write(request_json.encode())
            await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()

        sequence = 0
        start_time = asyncio.get_event_loop().time()

        # Read events from stderr
        if process.stderr:
            try:
                while True:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    remaining = self._config.timeout_seconds - elapsed
                    if remaining <= 0:
                        process.kill()
                        raise AdapterTimeoutError(
                            f"Container execution timed out after "
                            f"{self._config.timeout_seconds}s",
                            timeout_seconds=self._config.timeout_seconds,
                            adapter_type=self.adapter_type,
                        )

                    try:
                        line = await asyncio.wait_for(
                            process.stderr.readline(),
                            timeout=remaining,
                        )
                    except TimeoutError:
                        process.kill()
                        raise AdapterTimeoutError(
                            f"Container execution timed out after "
                            f"{self._config.timeout_seconds}s",
                            timeout_seconds=self._config.timeout_seconds,
                            adapter_type=self.adapter_type,
                        )

                    if not line:
                        break

                    line_str = line.decode().strip()
                    if not line_str:
                        continue

                    try:
                        data = json.loads(line_str)
                        if "sequence" not in data:
                            data["sequence"] = sequence
                            sequence += 1
                        if "timestamp" not in data:
                            data["timestamp"] = datetime.now().isoformat()
                        if "task_id" not in data:
                            data["task_id"] = request.task_id
                        if "event_type" not in data:
                            data["event_type"] = EventType.PROGRESS.value

                        yield ATPEvent.model_validate(data)
                    except (json.JSONDecodeError, ValueError):
                        # Skip malformed events
                        pass

            except AdapterTimeoutError:
                await process.wait()
                raise

        # Read final response from stdout
        if process.stdout:
            stdout = await process.stdout.read()
            if stdout:
                try:
                    response_data = json.loads(stdout.decode())
                    yield ATPResponse.model_validate(response_data)
                except (json.JSONDecodeError, ValueError) as e:
                    raise AdapterResponseError(
                        f"Invalid ATP Response from container: {e}",
                        response_body=stdout.decode()[:500],
                        adapter_type=self.adapter_type,
                    ) from e

        await process.wait()
        if process.returncode != 0:
            raise AdapterError(
                f"Container exited with code {process.returncode}",
                adapter_type=self.adapter_type,
            )

    async def health_check(self) -> bool:
        """
        Check if Docker is available and image exists.

        Returns:
            True if Docker is available and image exists, False otherwise.
        """
        try:
            # Check Docker is running
            process = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
            if process.returncode != 0:
                return False

            # Check image exists (optional, image might be pulled on first run)
            process = await asyncio.create_subprocess_exec(
                "docker",
                "image",
                "inspect",
                self._config.image,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
            return process.returncode == 0
        except (FileNotFoundError, OSError):
            return False

    async def cleanup(self) -> None:
        """Clean up any running containers."""
        # Container cleanup is handled by --rm flag
        pass
