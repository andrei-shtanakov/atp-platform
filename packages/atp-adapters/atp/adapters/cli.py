"""CLI adapter for command-line agent utilities."""

import asyncio
import json
import logging
import os
import shlex
from collections.abc import AsyncIterator
from datetime import datetime

from atp.core.security import (
    filter_environment_variables,
    sanitize_error_message,
)
from atp.protocol import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
)
from pydantic import Field

from .base import AdapterConfig, AgentAdapter
from .exceptions import (
    AdapterConnectionError,
    AdapterError,
    AdapterResponseError,
    AdapterTimeoutError,
)

logger = logging.getLogger(__name__)


class CLIAdapterConfig(AdapterConfig):
    """Configuration for CLI adapter."""

    command: str = Field(..., description="Command to execute the agent")
    args: list[str] = Field(
        default_factory=list, description="Additional command arguments"
    )
    working_dir: str | None = Field(
        None, description="Working directory for command execution"
    )
    environment: dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )
    allow_shell: bool = Field(
        default=False,
        description=(
            "Allow shell execution. WARNING: enables shell injection if command "
            "comes from untrusted input."
        ),
    )
    # Security settings
    inherit_environment: bool = Field(
        default=False,
        description="Whether to inherit parent environment variables (filtered)",
    )
    allowed_env_vars: list[str] = Field(
        default_factory=list,
        description="Additional environment variables to allow when inheriting",
    )


class CLIAdapter(AgentAdapter):
    """
    Adapter for command-line agent utilities.

    Runs agents as subprocesses with:
    - ATP Request sent via stdin (JSON)
    - ATP Response received via stdout (JSON)
    - ATP Events received via stderr (JSONL)
    """

    def __init__(self, config: CLIAdapterConfig) -> None:
        """
        Initialize CLI adapter.

        Args:
            config: CLI adapter configuration with command to run.
        """
        super().__init__(config)
        self._config: CLIAdapterConfig = config
        if self._config.allow_shell:
            logger.warning(
                "CLIAdapter: allow_shell=True — commands run through shell. "
                "Ensure command source is trusted."
            )

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""
        return "cli"

    def _build_command(self, request: ATPRequest) -> list[str]:
        """Build the command to execute."""
        if self._config.allow_shell:
            # For shell execution, join command and args
            cmd_str = self._config.command
            if self._config.args:
                cmd_str += " " + " ".join(shlex.quote(arg) for arg in self._config.args)
            return [cmd_str]

        # Split command if it contains spaces and not using shell
        cmd_parts = shlex.split(self._config.command)
        return cmd_parts + list(self._config.args)

    def _get_env(self) -> dict[str, str]:
        """Get environment variables for the subprocess.

        Security: Filters sensitive environment variables to prevent
        credential leakage to subprocesses.
        """
        # Start with filtered parent environment if inheritance is enabled
        if self._config.inherit_environment:
            env = filter_environment_variables(
                additional_allowlist=set(self._config.allowed_env_vars)
            )
        else:
            # Minimal environment for subprocess execution
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": os.environ.get("HOME", "/tmp"),
                "LANG": os.environ.get("LANG", "en_US.UTF-8"),
                "TERM": os.environ.get("TERM", "xterm"),
            }

        # Apply explicit environment variables from config
        env.update(self._config.environment)
        return env

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """
        Execute a task via CLI subprocess.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse from the agent.

        Raises:
            AdapterConnectionError: If command is not found.
            AdapterTimeoutError: If execution times out.
            AdapterResponseError: If agent returns invalid response.
        """
        cmd = self._build_command(request)
        request_json = request.model_dump_json()
        stdin_data: bytes = request_json.encode()

        try:
            process = (
                await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self._config.working_dir,
                    env=self._get_env(),
                )
                if not self._config.allow_shell
                else await asyncio.create_subprocess_shell(
                    cmd[0],
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self._config.working_dir,
                    env=self._get_env(),
                )
            )
        except FileNotFoundError as e:
            raise AdapterConnectionError(
                f"Command not found: {self._config.command}",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e
        except OSError as e:
            raise AdapterConnectionError(
                f"Failed to execute command: {e}",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=stdin_data),
                timeout=self._config.timeout_seconds,
            )
        except TimeoutError as e:
            process.kill()
            await process.wait()
            raise AdapterTimeoutError(
                f"Command execution timed out after {self._config.timeout_seconds}s",
                timeout_seconds=self._config.timeout_seconds,
                adapter_type=self.adapter_type,
            ) from e

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            # Sanitize error message (redacts secrets and file paths)
            error_msg = sanitize_error_message(error_msg, include_type=False)
            raise AdapterError(
                f"Command exited with code {process.returncode}: {error_msg}",
                adapter_type=self.adapter_type,
            )

        if not stdout:
            raise AdapterResponseError(
                "Command produced no output",
                adapter_type=self.adapter_type,
            )
        response_text = stdout.decode()

        try:
            response_data = json.loads(response_text)
            return ATPResponse.model_validate(response_data)
        except json.JSONDecodeError as e:
            raise AdapterResponseError(
                f"Invalid JSON response: {e}",
                response_body=response_text[:500],
                adapter_type=self.adapter_type,
            ) from e
        except ValueError as e:
            raise AdapterResponseError(
                f"Invalid ATP Response format: {e}",
                response_body=response_text[:500],
                adapter_type=self.adapter_type,
            ) from e

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """
        Execute a task with event streaming from stderr.

        Events are read from stderr as JSONL (one JSON object per line).
        Final response is read from stdout.

        Args:
            request: ATP Request with task specification.

        Yields:
            ATPEvent objects from stderr.
            Final ATPResponse from stdout.

        Raises:
            AdapterConnectionError: If command is not found.
            AdapterTimeoutError: If execution times out.
            AdapterResponseError: If agent returns invalid response.
        """
        cmd = self._build_command(request)
        request_json = request.model_dump_json()
        stdin_data: bytes = request_json.encode()

        try:
            process = (
                await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self._config.working_dir,
                    env=self._get_env(),
                )
                if not self._config.allow_shell
                else await asyncio.create_subprocess_shell(
                    cmd[0],
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self._config.working_dir,
                    env=self._get_env(),
                )
            )
        except FileNotFoundError as e:
            raise AdapterConnectionError(
                f"Command not found: {self._config.command}",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e
        except OSError as e:
            raise AdapterConnectionError(
                f"Failed to execute command: {e}",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

        # Write stdin
        if process.stdin:
            process.stdin.write(stdin_data)
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
                            f"Command execution timed out after "
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
                            f"Command execution timed out after "
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
                        # Skip non-JSON stderr lines
                        pass

            except AdapterTimeoutError:
                await process.wait()
                raise

        # Read final response from stdout
        if process.stdout:
            stdout = await process.stdout.read()
            if stdout:
                response_text = stdout.decode()
            else:
                raise AdapterResponseError(
                    "Command produced no output",
                    adapter_type=self.adapter_type,
                )
        else:
            raise AdapterResponseError(
                "No stdout available",
                adapter_type=self.adapter_type,
            )

        await process.wait()
        if process.returncode != 0:
            raise AdapterError(
                f"Command exited with code {process.returncode}",
                adapter_type=self.adapter_type,
            )

        try:
            response_data = json.loads(response_text)
            yield ATPResponse.model_validate(response_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise AdapterResponseError(
                f"Invalid ATP Response: {e}",
                response_body=response_text[:500],
                adapter_type=self.adapter_type,
            ) from e

    async def health_check(self) -> bool:
        """
        Check if the command is available.

        Returns:
            True if command exists, False otherwise.
        """
        import shutil

        cmd_name = shlex.split(self._config.command)[0]
        return shutil.which(cmd_name) is not None

    async def cleanup(self) -> None:
        """No-op: stdin/stdout adapter has no temp files to clean up."""
        pass
