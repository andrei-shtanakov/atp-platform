"""CLI adapter for command-line agent utilities."""

import asyncio
import json
import shlex
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

from pydantic import Field

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
    shell: bool = Field(default=False, description="Execute command through shell")
    input_format: str = Field(
        default="json", description="Input format: json, file, arg"
    )
    output_format: str = Field(default="json", description="Output format: json, file")
    input_file: str | None = Field(
        None, description="File path for input when input_format=file"
    )
    output_file: str | None = Field(
        None, description="File path for output when output_format=file"
    )


class CLIAdapter(AgentAdapter):
    """
    Adapter for command-line agent utilities.

    Runs agents as subprocesses with:
    - ATP Request sent via stdin (JSON) or file
    - ATP Response received via stdout (JSON) or file
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

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""
        return "cli"

    def _build_command(self, request: ATPRequest) -> list[str]:
        """Build the command to execute."""
        if self._config.shell:
            # For shell execution, join command and args
            cmd_str = self._config.command
            if self._config.args:
                cmd_str += " " + " ".join(shlex.quote(arg) for arg in self._config.args)
            return [cmd_str]

        # Split command if it contains spaces and not using shell
        cmd_parts = shlex.split(self._config.command)
        return cmd_parts + list(self._config.args)

    def _get_env(self) -> dict[str, str] | None:
        """Get environment variables for the subprocess."""
        if not self._config.environment:
            return None

        import os

        env = os.environ.copy()
        env.update(self._config.environment)
        return env

    async def _write_input_file(self, request: ATPRequest) -> Path:
        """Write request to input file."""
        input_path = Path(self._config.input_file or "/tmp/atp_request.json")
        input_path.write_text(request.model_dump_json())
        return input_path

    async def _read_output_file(self) -> str:
        """Read response from output file."""
        output_path = Path(self._config.output_file or "/tmp/atp_response.json")
        if not output_path.exists():
            raise AdapterResponseError(
                f"Output file not found: {output_path}",
                adapter_type=self.adapter_type,
            )
        return output_path.read_text()

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

        # Prepare input
        stdin_data: bytes | None = None
        if self._config.input_format == "json":
            stdin_data = request_json.encode()
        elif self._config.input_format == "file":
            await self._write_input_file(request)

        try:
            process = (
                await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE if stdin_data else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self._config.working_dir,
                    env=self._get_env(),
                )
                if not self._config.shell
                else await asyncio.create_subprocess_shell(
                    cmd[0],
                    stdin=asyncio.subprocess.PIPE if stdin_data else None,
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
            raise AdapterError(
                f"Command exited with code {process.returncode}: {error_msg}",
                adapter_type=self.adapter_type,
            )

        # Read response
        if self._config.output_format == "file":
            response_text = await self._read_output_file()
        else:
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
        Final response is read from stdout or file.

        Args:
            request: ATP Request with task specification.

        Yields:
            ATPEvent objects from stderr.
            Final ATPResponse from stdout/file.

        Raises:
            AdapterConnectionError: If command is not found.
            AdapterTimeoutError: If execution times out.
            AdapterResponseError: If agent returns invalid response.
        """
        cmd = self._build_command(request)
        request_json = request.model_dump_json()

        # Prepare input
        stdin_data: bytes | None = None
        if self._config.input_format == "json":
            stdin_data = request_json.encode()
        elif self._config.input_format == "file":
            await self._write_input_file(request)

        try:
            process = (
                await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE if stdin_data else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self._config.working_dir,
                    env=self._get_env(),
                )
                if not self._config.shell
                else await asyncio.create_subprocess_shell(
                    cmd[0],
                    stdin=asyncio.subprocess.PIPE if stdin_data else None,
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

        # Write stdin if needed
        if stdin_data and process.stdin:
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

        # Read final response
        if self._config.output_format == "file":
            await process.wait()
            response_text = await self._read_output_file()
        else:
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
        """Clean up temporary files."""
        # Clean up input/output files if they exist
        if self._config.input_format == "file" and self._config.input_file:
            path = Path(self._config.input_file)
            if path.exists():
                path.unlink(missing_ok=True)

        if self._config.output_format == "file" and self._config.output_file:
            path = Path(self._config.output_file)
            if path.exists():
                path.unlink(missing_ok=True)
