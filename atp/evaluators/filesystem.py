"""Filesystem evaluator for checking workspace state after agent execution.

**The root is granted, never declared.** This evaluator used to read
`workspace_path` out of `assertion.config` and treat it as the directory to
inspect. On the CLI that is merely redundant; on the benchmark plane it is
wrong twice over, because the suite and the submission come from different
people. The suite-named directory is not where the submission was
materialized, so every check measured something unrelated — and the pass/fail
it published was an existence answer about the server's own disk.

So the root now arrives from the composition that prepared the response (the
artifact sandbox on the server, the working directory on the CLI) via
`bind_workspace_root`, and an evaluator that was never granted one refuses to
answer rather than falling back to anything. `workspace_path` survives with
its usefulness and without its authority: a *relative subpath within* the
granted root. Absolute paths and traversal are config errors — reported, not
clamped and not silently ignored, because a suite author whose intent is
dropped without a diagnostic will never learn that it was.

See ADR-008 track B.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from atp.core.security import validate_path_within_workspace
from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse

from .base import EvalCheck, EvalResult, Evaluator

logger = logging.getLogger(__name__)


class WorkspaceNotGranted(RuntimeError):
    """No composition handed this evaluator a directory to inspect.

    A wiring fault, not a suite fault, so it is raised rather than answered:
    the pipeline records it as an evaluator error with this message attached,
    which keeps it out of the coverage bucket that holds principled refusals.
    """


class WorkspaceConfigError(ValueError):
    """The assertion's `workspace_path` is not a subpath of the granted root."""


class FilesystemEvaluator(Evaluator):
    """Evaluator for filesystem state assertions.

    Checks actual filesystem state in the workspace granted to it after agent
    execution.

    Supported assertion types:
    - file_exists: Check if a file exists at a path
    - file_not_exists: Check that a file does NOT exist
    - file_contains: Check that file content matches a pattern
    - dir_exists: Check if a directory exists
    - file_count: Check number of files in a directory
    """

    def __init__(
        self, workspace_root: Path | None = None, *, trusted: bool = False
    ) -> None:
        self._root = workspace_root
        self._trusted = trusted

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "filesystem"

    def bind_workspace_root(self, root: Path, *, trusted: bool = False) -> None:
        """Receive the directory this evaluator may address, and nothing else."""
        self._root = root
        self._trusted = trusted

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Evaluate filesystem assertions against the granted workspace.

        Raises:
            WorkspaceNotGranted: if no composition bound a root.
        """
        try:
            workspace = self._workspace_for(assertion.config)
        except WorkspaceConfigError as exc:
            return self._create_result(
                [
                    self._create_check(
                        name=assertion.type, passed=False, message=str(exc)
                    )
                ]
            )

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

    def _workspace_for(self, config: dict[str, Any]) -> Path:
        """Return the directory this assertion is about.

        Raises:
            WorkspaceConfigError: `workspace_path` names something outside the
                granted root.
            WorkspaceNotGranted: nothing granted a root and the config does
                not carry a legacy absolute one.
        """
        declared = config.get("workspace_path")

        if declared and Path(declared).is_absolute():
            return self._legacy_absolute_root(str(declared))

        if self._root is None:
            raise WorkspaceNotGranted(
                "filesystem evaluator was not granted a workspace root; it is "
                "supplied by the composition that prepares the response, and "
                "an assertion may not name one of its own"
            )

        if not declared:
            return self._root

        try:
            return validate_path_within_workspace(str(declared), self._root)
        except Exception as exc:
            raise WorkspaceConfigError(
                f"Invalid 'workspace_path' {declared!r}: it must be a relative "
                f"subpath of the evaluation workspace ({exc})"
            ) from exc

    def _legacy_absolute_root(self, declared: str) -> Path:
        """Honour the pre-ADR-008 meaning of `workspace_path`, on the CLI only.

        Before the root was granted, `workspace_path` *was* the root and was
        routinely absolute. Suites written that way still exist outside this
        repository, and breaking them silently — by injecting the working
        directory and hoping it reproduces the old target, which in general it
        does not — would be the quiet failure this whole change is against.

        So the old form is converted explicitly and loudly, and only where it
        was ever meaningful: the operator's own machine, naming the operator's
        own directory. On any untrusted plane it is a config error, because
        there it never named the submission in the first place.

        Transitional. Remove once a release has shipped with the warning.
        """
        if not self._trusted:
            raise WorkspaceConfigError(
                f"Invalid 'workspace_path' {declared!r}: an absolute path is "
                "not permitted here. The workspace is supplied by the server; "
                "'workspace_path' may only name a relative subpath of it"
            )
        logger.warning(
            "Deprecated: assertion sets an absolute 'workspace_path' (%s). "
            "The evaluation workspace is now supplied by the runner, and "
            "'workspace_path' should be a relative subpath of it. Absolute "
            "paths are honoured on the local CLI only and will be removed.",
            declared,
        )
        return Path(declared)

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
            # Not "absent, therefore pass". A path that would not resolve —
            # absolute, traversing, over-long — is a question this evaluator
            # never asked, and scoring it as satisfied hands a point to the
            # malformed assertion most likely to be probing the boundary.
            # Reported as the config error it is, exactly like `file_exists`.
            return self._create_check(
                name="file_not_exists",
                passed=False,
                message=f"Invalid path: {path}",
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
