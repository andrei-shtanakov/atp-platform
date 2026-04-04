"""Tests for container integration in CodeExecEvaluator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.evaluators.code_exec import CodeExecEvaluator, CommandResult


class TestCodeExecContainerIntegration:
    """Tests for container execution path in _run_command."""

    @pytest.mark.anyio
    async def test_uses_container_when_enabled(self) -> None:
        mock_runtime = MagicMock()
        mock_runtime.run = AsyncMock(
            return_value=CommandResult(
                return_code=0,
                stdout="1 passed",
                stderr="",
                timed_out=False,
            )
        )

        evaluator = CodeExecEvaluator(container_runtime=mock_runtime)
        result = await evaluator._run_command(
            command=["pytest"],
            working_dir="/workspace",
            container=True,
        )

        assert result.return_code == 0
        mock_runtime.run.assert_called_once()
        call_kwargs = mock_runtime.run.call_args
        assert call_kwargs[1]["image"] == "python:3.12-slim"

    @pytest.mark.anyio
    async def test_custom_image(self) -> None:
        mock_runtime = MagicMock()
        mock_runtime.run = AsyncMock(
            return_value=CommandResult(
                return_code=0,
                stdout="ok",
                stderr="",
                timed_out=False,
            )
        )

        evaluator = CodeExecEvaluator(container_runtime=mock_runtime)
        result = await evaluator._run_command(
            command=["npm", "test"],
            container=True,
            image="node:20-slim",
        )

        assert result.return_code == 0
        call_kwargs = mock_runtime.run.call_args
        assert call_kwargs[1]["image"] == "node:20-slim"

    @pytest.mark.anyio
    async def test_falls_back_to_subprocess_when_no_runtime(self) -> None:
        evaluator = CodeExecEvaluator(container_runtime=None)
        result = await evaluator._run_command(
            command=["echo", "hello"],
            container=True,
            sandboxed=True,
        )
        # Should fall back to subprocess execution
        assert result.return_code == 0
        assert "hello" in result.stdout

    @pytest.mark.anyio
    async def test_subprocess_when_container_false(self) -> None:
        mock_runtime = MagicMock()
        mock_runtime.run = AsyncMock()

        evaluator = CodeExecEvaluator(container_runtime=mock_runtime)
        result = await evaluator._run_command(
            command=["echo", "hello"],
            container=False,
            sandboxed=True,
        )
        # Should use subprocess, not container
        assert result.return_code == 0
        mock_runtime.run.assert_not_called()

    @pytest.mark.anyio
    async def test_custom_default_image(self) -> None:
        mock_runtime = MagicMock()
        mock_runtime.run = AsyncMock(
            return_value=CommandResult(
                return_code=0,
                stdout="ok",
                stderr="",
                timed_out=False,
            )
        )

        evaluator = CodeExecEvaluator(
            container_runtime=mock_runtime,
            container_default_image="python:3.11-slim",
        )
        await evaluator._run_command(
            command=["pytest"],
            container=True,
        )

        call_kwargs = mock_runtime.run.call_args
        assert call_kwargs[1]["image"] == "python:3.11-slim"

    @pytest.mark.anyio
    async def test_container_passes_env_and_timeout(self) -> None:
        mock_runtime = MagicMock()
        mock_runtime.run = AsyncMock(
            return_value=CommandResult(
                return_code=0,
                stdout="",
                stderr="",
                timed_out=False,
            )
        )

        evaluator = CodeExecEvaluator(container_runtime=mock_runtime)
        await evaluator._run_command(
            command=["pytest"],
            timeout=60,
            env={"FOO": "bar"},
            container=True,
        )

        call_kwargs = mock_runtime.run.call_args[1]
        assert call_kwargs["timeout"] == 60
        assert call_kwargs["env"] == {"FOO": "bar"}

    @pytest.mark.anyio
    async def test_string_command_split_for_container(self) -> None:
        mock_runtime = MagicMock()
        mock_runtime.run = AsyncMock(
            return_value=CommandResult(
                return_code=0,
                stdout="",
                stderr="",
                timed_out=False,
            )
        )

        evaluator = CodeExecEvaluator(container_runtime=mock_runtime)
        await evaluator._run_command(
            command="pytest tests/ -v",
            container=True,
        )

        call_kwargs = mock_runtime.run.call_args[1]
        assert call_kwargs["command"] == ["pytest", "tests/", "-v"]
