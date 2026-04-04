"""Tests for container runtime module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.evaluators.container import (
    DockerRuntime,
    PodmanRuntime,
    detect_runtime,
)


class TestDockerRuntime:
    """Tests for DockerRuntime."""

    @pytest.mark.anyio
    async def test_is_available_when_docker_exists(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/docker"):
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock()
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                runtime = DockerRuntime()
                assert await runtime.is_available() is True

    @pytest.mark.anyio
    async def test_is_available_when_docker_missing(self) -> None:
        with patch("shutil.which", return_value=None):
            runtime = DockerRuntime()
            assert await runtime.is_available() is False

    @pytest.mark.anyio
    async def test_run_builds_correct_command(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"test output\n", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            runtime = DockerRuntime()
            result = await runtime.run(
                image="python:3.12-slim",
                command=["python", "-c", "print('hello')"],
                timeout=30,
            )

        assert result.return_code == 0
        assert result.stdout == "test output\n"
        assert not result.timed_out

        # Verify docker run command
        call_args = mock_exec.call_args[0]
        assert call_args[0] == "docker"
        assert call_args[1] == "run"
        assert "--rm" in call_args
        assert "--network=none" in call_args
        assert "--read-only" in call_args
        assert "python:3.12-slim" in call_args

    @pytest.mark.anyio
    async def test_run_with_workspace(self, tmp_path: Path) -> None:
        # Create workspace with a file
        (tmp_path / "test.py").write_text("print('hello')")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            runtime = DockerRuntime()
            result = await runtime.run(
                image="python:3.12-slim",
                command=["python", "test.py"],
                workspace=tmp_path,
                timeout=30,
            )

        assert result.return_code == 0
        call_args = mock_exec.call_args[0]
        # Should have -v mount and -w /work
        call_str = " ".join(str(a) for a in call_args)
        assert "-v" in call_str
        assert "/work" in call_str

    @pytest.mark.anyio
    async def test_run_timeout(self) -> None:
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            runtime = DockerRuntime()
            result = await runtime.run(
                image="python:3.12-slim",
                command=["sleep", "999"],
                timeout=1,
            )

        assert result.timed_out is True
        assert result.return_code == -1

    @pytest.mark.anyio
    async def test_run_with_env(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            runtime = DockerRuntime()
            await runtime.run(
                image="python:3.12-slim",
                command=["env"],
                env={"FOO": "bar"},
            )

        call_args = mock_exec.call_args[0]
        args_str = " ".join(str(a) for a in call_args)
        assert "FOO=bar" in args_str


class TestPodmanRuntime:
    """Tests for PodmanRuntime."""

    @pytest.mark.anyio
    async def test_uses_podman_executable(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            runtime = PodmanRuntime()
            await runtime.run(
                image="python:3.12-slim",
                command=["echo"],
            )

        call_args = mock_exec.call_args[0]
        assert call_args[0] == "podman"


class TestDetectRuntime:
    """Tests for runtime auto-detection."""

    @pytest.mark.anyio
    async def test_none_disables(self) -> None:
        result = await detect_runtime("none")
        assert result is None

    @pytest.mark.anyio
    async def test_auto_prefers_docker(self) -> None:
        with patch.object(DockerRuntime, "is_available", return_value=True):
            result = await detect_runtime("auto")
            assert isinstance(result, DockerRuntime)

    @pytest.mark.anyio
    async def test_auto_falls_back_to_podman(self) -> None:
        with patch.object(DockerRuntime, "is_available", return_value=False):
            with patch.object(PodmanRuntime, "is_available", return_value=True):
                result = await detect_runtime("auto")
                assert isinstance(result, PodmanRuntime)

    @pytest.mark.anyio
    async def test_auto_returns_none_when_nothing(self) -> None:
        with patch.object(DockerRuntime, "is_available", return_value=False):
            with patch.object(PodmanRuntime, "is_available", return_value=False):
                result = await detect_runtime("auto")
                assert result is None
