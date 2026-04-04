"""Tests for code-exec evaluator sandbox (rlimits + safe env)."""

import sys

import pytest

from atp.evaluators.code_exec import (
    CodeExecEvaluator,
    _safe_env,
)


@pytest.fixture
def evaluator() -> CodeExecEvaluator:
    return CodeExecEvaluator()


class TestSafeEnv:
    """Tests for _safe_env()."""

    def test_safe_env_has_path(self) -> None:
        env = _safe_env()
        assert "PATH" in env
        assert "/usr/bin" in env["PATH"]

    def test_safe_env_no_secrets(self) -> None:
        env = _safe_env()
        assert "ATP_SECRET_KEY" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env

    def test_safe_env_minimal(self) -> None:
        env = _safe_env()
        assert len(env) <= 5


class TestSandboxedExecution:
    """Tests for sandboxed _run_command()."""

    @pytest.mark.anyio
    async def test_sandboxed_echo(self, evaluator: CodeExecEvaluator) -> None:
        """Basic sandboxed command works."""
        result = await evaluator._run_command(
            ["echo", "hello"],
            sandboxed=True,
        )
        assert result.return_code == 0
        assert "hello" in result.stdout

    @pytest.mark.anyio
    async def test_sandboxed_env_is_clean(self, evaluator: CodeExecEvaluator) -> None:
        """Sandboxed process does not inherit server env."""
        result = await evaluator._run_command(
            ["env"],
            sandboxed=True,
        )
        assert result.return_code == 0
        # Should not contain any ATP_ vars from parent
        assert "ATP_SECRET_KEY" not in result.stdout

    @pytest.mark.anyio
    async def test_sandboxed_timeout(self, evaluator: CodeExecEvaluator) -> None:
        """Sandboxed process respects timeout."""
        result = await evaluator._run_command(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            sandboxed=True,
            timeout=1,
        )
        assert result.timed_out

    @pytest.mark.anyio
    async def test_unsandboxed_inherits_env(self, evaluator: CodeExecEvaluator) -> None:
        """Unsandboxed process inherits parent env."""
        result = await evaluator._run_command(
            ["env"],
            sandboxed=False,
        )
        assert result.return_code == 0
        # Should contain PATH from parent env
        assert "PATH=" in result.stdout


class TestCheckMethodsSandboxed:
    """Tests that public _check_* methods use sandboxed execution."""

    @pytest.mark.anyio
    async def test_check_pytest_uses_sandbox(
        self, evaluator: CodeExecEvaluator
    ) -> None:
        """_check_pytest passes sandboxed=True to _run_command."""
        from unittest.mock import AsyncMock, patch

        mock_result = AsyncMock()
        mock_result.return_code = 0
        mock_result.stdout = "1 passed"
        mock_result.stderr = ""
        mock_result.timed_out = False

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock, return_value=mock_result
        ) as mock_run:
            await evaluator._check_pytest({"path": "tests/"})
            mock_run.assert_called_once()
            _, kwargs = mock_run.call_args
            assert kwargs.get("sandboxed") is True

    @pytest.mark.anyio
    async def test_check_npm_uses_sandbox(self, evaluator: CodeExecEvaluator) -> None:
        """_check_npm passes sandboxed=True to _run_command."""
        from unittest.mock import AsyncMock, patch

        mock_result = AsyncMock()
        mock_result.return_code = 0
        mock_result.stdout = "Tests: 5 passed"
        mock_result.stderr = ""
        mock_result.timed_out = False

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock, return_value=mock_result
        ) as mock_run:
            await evaluator._check_npm({})
            mock_run.assert_called_once()
            _, kwargs = mock_run.call_args
            assert kwargs.get("sandboxed") is True

    @pytest.mark.anyio
    async def test_check_custom_command_uses_sandbox(
        self, evaluator: CodeExecEvaluator
    ) -> None:
        """_check_custom_command passes sandboxed=True to _run_command."""
        from unittest.mock import AsyncMock, patch

        mock_result = AsyncMock()
        mock_result.return_code = 0
        mock_result.stdout = "OK"
        mock_result.stderr = ""
        mock_result.timed_out = False

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock, return_value=mock_result
        ) as mock_run:
            await evaluator._check_custom_command({"command": "echo test"})
            mock_run.assert_called_once()
            _, kwargs = mock_run.call_args
            assert kwargs.get("sandboxed") is True

    @pytest.mark.anyio
    async def test_check_lint_uses_sandbox(self, evaluator: CodeExecEvaluator) -> None:
        """_check_lint passes sandboxed=True to _run_command."""
        from unittest.mock import AsyncMock, patch

        mock_result = AsyncMock()
        mock_result.return_code = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.timed_out = False

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock, return_value=mock_result
        ) as mock_run:
            await evaluator._check_lint({"linter": "ruff"})
            mock_run.assert_called_once()
            _, kwargs = mock_run.call_args
            assert kwargs.get("sandboxed") is True
