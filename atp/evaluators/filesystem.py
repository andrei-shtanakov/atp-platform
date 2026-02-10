"""Filesystem evaluator for checking workspace state after agent execution."""

import re
from pathlib import Path
from typing import Any

from atp.core.security import validate_path_within_workspace
from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse

from .base import EvalCheck, EvalResult, Evaluator


class FilesystemEvaluator(Evaluator):
    """Evaluator for filesystem state assertions.

    Checks actual filesystem state in the sandbox workspace
    after agent execution.

    Supported assertion types:
    - file_exists: Check if a file exists at a path
    - file_not_exists: Check that a file does NOT exist
    - file_contains: Check that file content matches a pattern
    - dir_exists: Check if a directory exists
    - file_count: Check number of files in a directory
    """

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "filesystem"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Evaluate filesystem assertions."""
        workspace_path = assertion.config.get("workspace_path")
        if not workspace_path:
            return self._create_result(
                [
                    self._create_check(
                        name=assertion.type,
                        passed=False,
                        message="No workspace_path in assertion config",
                    )
                ]
            )

        workspace = Path(workspace_path)
        if not workspace.is_dir():
            return self._create_result(
                [
                    self._create_check(
                        name=assertion.type,
                        passed=False,
                        message=f"Workspace not found: {workspace}",
                    )
                ]
            )

        handlers: dict[str, Any] = {
            "file_exists": self._check_file_exists,
            "file_not_exists": self._check_file_not_exists,
            "file_contains": self._check_file_contains,
            "dir_exists": self._check_dir_exists,
            "file_count": self._check_file_count,
        }

        handler = handlers.get(assertion.type)
        if handler is None:
            check = self._create_check(
                name=assertion.type,
                passed=False,
                message=f"Unknown filesystem assertion: {assertion.type}",
            )
        else:
            check = handler(workspace, assertion.config)

        return self._create_result([check])

    def _resolve_path(self, workspace: Path, relative_path: str) -> Path | None:
        """Resolve and validate a path within workspace."""
        try:
            return validate_path_within_workspace(relative_path, workspace)
        except Exception:
            return None

    def _check_file_exists(self, workspace: Path, config: dict[str, Any]) -> EvalCheck:
        """Check if a file exists at the given path."""
        path = config.get("path", "")
        if not path:
            return self._create_check(
                name="file_exists",
                passed=False,
                message="No 'path' specified in config",
            )

        resolved = self._resolve_path(workspace, path)
        if resolved is None:
            return self._create_check(
                name="file_exists",
                passed=False,
                message=f"Invalid path: {path}",
            )

        exists = resolved.is_file()
        return self._create_check(
            name="file_exists",
            passed=exists,
            message=(f"File exists: {path}" if exists else f"File not found: {path}"),
        )

    def _check_file_not_exists(
        self, workspace: Path, config: dict[str, Any]
    ) -> EvalCheck:
        """Check that a file does NOT exist."""
        path = config.get("path", "")
        if not path:
            return self._create_check(
                name="file_not_exists",
                passed=False,
                message="No 'path' specified in config",
            )

        resolved = self._resolve_path(workspace, path)
        if resolved is None:
            return self._create_check(
                name="file_not_exists",
                passed=True,
                message=f"Path is invalid (treated as not existing): {path}",
            )

        not_exists = not resolved.exists()
        return self._create_check(
            name="file_not_exists",
            passed=not_exists,
            message=(
                f"File correctly absent: {path}"
                if not_exists
                else f"File unexpectedly exists: {path}"
            ),
        )

    def _check_file_contains(
        self, workspace: Path, config: dict[str, Any]
    ) -> EvalCheck:
        """Check that file content matches a pattern."""
        path = config.get("path", "")
        pattern = config.get("pattern", "")
        use_regex = config.get("regex", False)

        if not path:
            return self._create_check(
                name="file_contains",
                passed=False,
                message="No 'path' specified in config",
            )
        if not pattern:
            return self._create_check(
                name="file_contains",
                passed=False,
                message="No 'pattern' specified in config",
            )

        resolved = self._resolve_path(workspace, path)
        if resolved is None or not resolved.is_file():
            return self._create_check(
                name="file_contains",
                passed=False,
                message=f"File not found: {path}",
            )

        try:
            content = resolved.read_text()
        except OSError as e:
            return self._create_check(
                name="file_contains",
                passed=False,
                message=f"Cannot read file {path}: {e}",
            )

        if use_regex:
            matched = bool(re.search(pattern, content))
        else:
            matched = pattern in content

        return self._create_check(
            name="file_contains",
            passed=matched,
            message=(
                f"Pattern found in {path}"
                if matched
                else f"Pattern not found in {path}"
            ),
        )

    def _check_dir_exists(self, workspace: Path, config: dict[str, Any]) -> EvalCheck:
        """Check if a directory exists."""
        path = config.get("path", "")
        if not path:
            return self._create_check(
                name="dir_exists",
                passed=False,
                message="No 'path' specified in config",
            )

        resolved = self._resolve_path(workspace, path)
        if resolved is None:
            return self._create_check(
                name="dir_exists",
                passed=False,
                message=f"Invalid path: {path}",
            )

        exists = resolved.is_dir()
        return self._create_check(
            name="dir_exists",
            passed=exists,
            message=(
                f"Directory exists: {path}"
                if exists
                else f"Directory not found: {path}"
            ),
        )

    def _check_file_count(self, workspace: Path, config: dict[str, Any]) -> EvalCheck:
        """Check number of files in a directory."""
        path = config.get("path", ".")
        expected_count = config.get("count", 0)
        operator = config.get("operator", "eq")

        resolved = self._resolve_path(workspace, path)
        if resolved is None or not resolved.is_dir():
            return self._create_check(
                name="file_count",
                passed=False,
                message=f"Directory not found: {path}",
            )

        actual_count = sum(1 for f in resolved.iterdir() if f.is_file())

        ops: dict[str, bool] = {
            "eq": actual_count == expected_count,
            "gt": actual_count > expected_count,
            "gte": actual_count >= expected_count,
            "lt": actual_count < expected_count,
            "lte": actual_count <= expected_count,
        }

        passed = ops.get(operator, actual_count == expected_count)
        return self._create_check(
            name="file_count",
            passed=passed,
            message=(
                f"File count in {path}: {actual_count} "
                f"(expected {operator} {expected_count})"
            ),
            details={
                "actual_count": actual_count,
                "expected_count": expected_count,
                "operator": operator,
            },
        )
