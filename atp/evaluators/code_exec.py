"""Code execution evaluator for running tests and linters."""

import asyncio
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse

from .base import EvalCheck, EvalResult, Evaluator


@dataclass
class CommandResult:
    """Result of executing a command."""

    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class CodeTestResults:
    """Parsed test results from output."""

    total: int
    passed: int
    failed: int
    skipped: int
    errors: int
    pass_rate: float
    failure_messages: list[str]


@dataclass
class LintResults:
    """Parsed lint results from output."""

    total_violations: int
    errors: int
    warnings: int
    files_checked: int
    violation_messages: list[str]


class CodeExecEvaluator(Evaluator):
    """
    Evaluator for code execution assertions.

    Supports the following assertion types:
    - pytest: Run pytest and check pass rate
    - npm: Run npm test and check pass rate
    - custom_command: Run arbitrary command and parse output
    - lint: Run ruff or eslint and check violations

    Configuration options:
    - path: Directory or file to run tests on
    - threshold: Minimum pass rate (0.0-1.0) for tests
    - command: Custom command to run
    - pattern: Regex pattern to extract results
    - linter: Linter to use (ruff, eslint)
    - max_violations: Maximum allowed violations for lint
    - timeout: Command timeout in seconds (default: 300)
    - working_dir: Working directory for command execution
    """

    def __init__(self, sandbox_manager: Any | None = None) -> None:
        """
        Initialize the evaluator.

        Args:
            sandbox_manager: Optional SandboxManager for isolated execution.
        """
        self._sandbox_manager = sandbox_manager

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "code_exec"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """
        Evaluate code execution assertions.

        Args:
            task: Test definition (unused for code exec checks).
            response: ATP Response containing artifacts.
            trace: Event trace (unused for code exec checks).
            assertion: Assertion to evaluate.

        Returns:
            EvalResult with check outcomes.
        """
        assertion_type = assertion.type
        config = assertion.config

        if assertion_type == "pytest":
            check = await self._check_pytest(config)
        elif assertion_type == "npm":
            check = await self._check_npm(config)
        elif assertion_type == "custom_command":
            check = await self._check_custom_command(config)
        elif assertion_type == "lint":
            check = await self._check_lint(config)
        else:
            check = self._create_check(
                name=f"unknown_{assertion_type}",
                passed=False,
                message=f"Unknown assertion type: {assertion_type}",
            )

        return self._create_result([check])

    async def _run_command(
        self,
        command: str | list[str],
        working_dir: str | Path | None = None,
        timeout: int = 300,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """
        Execute a command asynchronously.

        Args:
            command: Command to execute (string or list of args).
            working_dir: Working directory for execution.
            timeout: Timeout in seconds.
            env: Environment variables.

        Returns:
            CommandResult with output and return code.
        """
        if isinstance(command, str):
            cmd_args = shlex.split(command)
        else:
            cmd_args = command

        cwd = Path(working_dir) if working_dir else None

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
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
                    stderr="Command timed out",
                    timed_out=True,
                )

        except FileNotFoundError:
            return CommandResult(
                return_code=-1,
                stdout="",
                stderr=f"Command not found: {cmd_args[0]}",
            )
        except Exception as e:
            return CommandResult(
                return_code=-1,
                stdout="",
                stderr=f"Failed to execute command: {e}",
            )

    def _parse_pytest_output(self, stdout: str, stderr: str) -> CodeTestResults:
        """
        Parse pytest output to extract test results.

        Args:
            stdout: Standard output from pytest.
            stderr: Standard error from pytest.

        Returns:
            Parsed TestResults.
        """
        combined = stdout + "\n" + stderr

        # Try to match pytest summary line: "X passed, Y failed, Z skipped"
        # Examples:
        # "5 passed in 1.23s"
        # "3 passed, 2 failed, 1 skipped in 2.45s"
        # "1 passed, 1 warning in 0.12s"

        total = 0
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        failure_messages: list[str] = []

        # Match passed count
        passed_match = re.search(r"(\d+)\s+passed", combined)
        if passed_match:
            passed = int(passed_match.group(1))

        # Match failed count
        failed_match = re.search(r"(\d+)\s+failed", combined)
        if failed_match:
            failed = int(failed_match.group(1))

        # Match skipped count
        skipped_match = re.search(r"(\d+)\s+skipped", combined)
        if skipped_match:
            skipped = int(skipped_match.group(1))

        # Match error count
        errors_match = re.search(r"(\d+)\s+error", combined)
        if errors_match:
            errors = int(errors_match.group(1))

        total = passed + failed + skipped + errors

        # Extract failure messages from FAILURES section
        failures_section = re.search(
            r"={3,}\s*FAILURES\s*={3,}(.*?)(?:={3,}|$)",
            combined,
            re.DOTALL,
        )
        if failures_section:
            failure_text = failures_section.group(1)
            # Extract test names
            test_failures = re.findall(
                r"_{3,}\s*(\S+)\s*_{3,}",
                failure_text,
            )
            failure_messages = test_failures[:10]  # Limit to 10 messages

        # Calculate pass rate
        if total > 0:
            pass_rate = passed / total
        else:
            # No tests found - check if collection error
            if "no tests ran" in combined.lower() or "collected 0 items" in combined:
                pass_rate = 0.0
            else:
                pass_rate = 1.0  # No tests = pass (empty state)

        return CodeTestResults(
            total=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            pass_rate=pass_rate,
            failure_messages=failure_messages,
        )

    def _parse_npm_output(self, stdout: str, stderr: str) -> CodeTestResults:
        """
        Parse npm test output to extract test results.

        Supports common test frameworks: Jest, Mocha, Vitest.

        Args:
            stdout: Standard output from npm test.
            stderr: Standard error from npm test.

        Returns:
            Parsed TestResults.
        """
        combined = stdout + "\n" + stderr

        total = 0
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        failure_messages: list[str] = []

        # Try Jest format first: "Tests: X failed, Y passed, Z total"
        jest_match = re.search(
            r"Tests:\s*(?:(\d+)\s+failed,?\s*)?(?:(\d+)\s+passed,?\s*)?(\d+)\s+total",
            combined,
        )
        if jest_match:
            failed = int(jest_match.group(1) or 0)
            passed = int(jest_match.group(2) or 0)
            total = int(jest_match.group(3))
            skipped = total - passed - failed

        # Try Mocha format: "X passing" "Y failing"
        if total == 0:
            mocha_passing = re.search(r"(\d+)\s+passing", combined)
            mocha_failing = re.search(r"(\d+)\s+failing", combined)
            mocha_pending = re.search(r"(\d+)\s+pending", combined)

            if mocha_passing:
                passed = int(mocha_passing.group(1))
            if mocha_failing:
                failed = int(mocha_failing.group(1))
            if mocha_pending:
                skipped = int(mocha_pending.group(1))

            total = passed + failed + skipped

        # Try Vitest format: "Tests X | Y passed"
        if total == 0:
            vitest_match = re.search(
                r"Tests\s+(\d+)(?:\s+\|\s+(\d+)\s+failed)?(?:\s+\|\s+(\d+)\s+passed)?",
                combined,
            )
            if vitest_match:
                total = int(vitest_match.group(1))
                failed = int(vitest_match.group(2) or 0)
                passed = int(vitest_match.group(3) or total - failed)

        # Extract failure messages
        # Jest format: "FAIL src/test.js"
        jest_failures = re.findall(r"FAIL\s+(\S+)", combined)
        if jest_failures:
            failure_messages = jest_failures[:10]

        # Calculate pass rate
        if total > 0:
            pass_rate = passed / total
        else:
            # Check for npm script errors
            if "npm ERR!" in combined or "error" in combined.lower():
                pass_rate = 0.0
            else:
                pass_rate = 1.0

        return CodeTestResults(
            total=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            pass_rate=pass_rate,
            failure_messages=failure_messages,
        )

    def _parse_custom_output(
        self,
        stdout: str,
        stderr: str,
        pattern: str | None,
        success_pattern: str | None,
    ) -> CodeTestResults:
        """
        Parse custom command output using regex patterns.

        Args:
            stdout: Standard output.
            stderr: Standard error.
            pattern: Regex pattern to extract test counts.
                    Should have named groups: passed, failed, total
            success_pattern: Pattern that indicates success.

        Returns:
            Parsed TestResults.
        """
        combined = stdout + "\n" + stderr

        total = 0
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        pass_rate = 0.0
        failure_messages: list[str] = []

        if pattern:
            try:
                match = re.search(pattern, combined)
                if match:
                    groups = match.groupdict()
                    passed = int(groups.get("passed", 0) or 0)
                    failed = int(groups.get("failed", 0) or 0)
                    total = int(groups.get("total", 0) or passed + failed)
                    skipped = int(groups.get("skipped", 0) or 0)
                    errors = int(groups.get("errors", 0) or 0)

                    if total > 0:
                        pass_rate = passed / total
            except re.error:
                pass  # Invalid regex, continue with defaults

        # Check success pattern
        if success_pattern and pass_rate == 0.0:
            if re.search(success_pattern, combined):
                pass_rate = 1.0
                passed = 1
                total = 1

        return CodeTestResults(
            total=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            pass_rate=pass_rate,
            failure_messages=failure_messages,
        )

    def _parse_ruff_output(self, stdout: str, stderr: str) -> LintResults:
        """
        Parse ruff lint output.

        Args:
            stdout: Standard output from ruff.
            stderr: Standard error from ruff.

        Returns:
            Parsed LintResults.
        """
        combined = stdout + "\n" + stderr

        total_violations = 0
        errors = 0
        warnings = 0
        files_checked = 0
        violation_messages: list[str] = []

        # Count violations by counting lines that match ruff error format
        # Format: "path/file.py:10:5: E501 Line too long"
        violation_lines = re.findall(
            r"^(.+:\d+:\d+:\s*\w+\d+.*)$",
            combined,
            re.MULTILINE,
        )
        total_violations = len(violation_lines)
        violation_messages = violation_lines[:10]

        # Classify as errors vs warnings
        for line in violation_lines:
            if re.search(r":\s*E\d+", line):  # E prefix = error
                errors += 1
            else:
                warnings += 1

        # Try to extract "Found X errors" summary
        found_match = re.search(r"Found\s+(\d+)\s+error", combined)
        if found_match:
            total_violations = max(total_violations, int(found_match.group(1)))

        # Count files checked
        files_match = re.search(r"(\d+)\s+files?\s+checked", combined)
        if files_match:
            files_checked = int(files_match.group(1))

        # "All checks passed" means zero violations
        if "All checks passed" in combined:
            total_violations = 0
            errors = 0
            warnings = 0

        return LintResults(
            total_violations=total_violations,
            errors=errors,
            warnings=warnings,
            files_checked=files_checked,
            violation_messages=violation_messages,
        )

    def _parse_eslint_output(self, stdout: str, stderr: str) -> LintResults:
        """
        Parse eslint output.

        Args:
            stdout: Standard output from eslint.
            stderr: Standard error from eslint.

        Returns:
            Parsed LintResults.
        """
        combined = stdout + "\n" + stderr

        total_violations = 0
        errors = 0
        warnings = 0
        files_checked = 0
        violation_messages: list[str] = []

        # Count error/warning summary line
        # Format: "X problems (Y errors, Z warnings)"
        summary_match = re.search(
            r"(\d+)\s+problems?\s*\((\d+)\s+errors?,\s*(\d+)\s+warnings?\)",
            combined,
        )
        if summary_match:
            total_violations = int(summary_match.group(1))
            errors = int(summary_match.group(2))
            warnings = int(summary_match.group(3))

        # Count individual violation lines
        # Format: "  10:5  error  description  rule-name"
        violation_lines = re.findall(
            r"^\s+(\d+:\d+)\s+(error|warning)\s+(.+)$",
            combined,
            re.MULTILINE,
        )
        if not summary_match and violation_lines:
            total_violations = len(violation_lines)
            errors = sum(1 for v in violation_lines if v[1] == "error")
            warnings = total_violations - errors

        for v in violation_lines[:10]:
            violation_messages.append(f"{v[0]} {v[1]}: {v[2]}")

        # Count unique files in output
        file_pattern = r"^(/[\w/.-]+|[\w./]+\.\w+)$"
        file_patterns = re.findall(file_pattern, combined, re.MULTILINE)
        files_checked = len(set(file_patterns))

        return LintResults(
            total_violations=total_violations,
            errors=errors,
            warnings=warnings,
            files_checked=files_checked,
            violation_messages=violation_messages,
        )

    async def _check_pytest(self, config: dict[str, Any]) -> EvalCheck:
        """
        Run pytest and check pass rate.

        Config options:
        - path: Test path (default: "tests/")
        - threshold: Minimum pass rate (default: 1.0)
        - timeout: Command timeout (default: 300)
        - working_dir: Working directory
        - args: Additional pytest arguments
        """
        path = config.get("path", "tests/")
        threshold = config.get("threshold", 1.0)
        timeout = config.get("timeout", 300)
        working_dir = config.get("working_dir")
        extra_args = config.get("args", "")

        # Build command as list to prevent shell injection
        cmd_args = ["pytest", path, "-v"]
        if extra_args:
            cmd_args.extend(shlex.split(extra_args))

        result = await self._run_command(
            cmd_args,
            working_dir=working_dir,
            timeout=timeout,
        )

        if result.timed_out:
            return self._create_check(
                name="pytest",
                passed=False,
                message=f"Pytest timed out after {timeout}s",
                details={
                    "timed_out": True,
                    "timeout": timeout,
                },
            )

        test_results = self._parse_pytest_output(result.stdout, result.stderr)

        passed = test_results.pass_rate >= threshold

        return self._create_check(
            name="pytest",
            passed=passed,
            message=(
                f"Pytest: {test_results.passed}/{test_results.total} tests passed "
                f"({test_results.pass_rate:.1%}), threshold: {threshold:.1%}"
            ),
            details={
                "total": test_results.total,
                "passed": test_results.passed,
                "failed": test_results.failed,
                "skipped": test_results.skipped,
                "errors": test_results.errors,
                "pass_rate": test_results.pass_rate,
                "threshold": threshold,
                "failure_messages": test_results.failure_messages,
                "return_code": result.return_code,
            },
        )

    async def _check_npm(self, config: dict[str, Any]) -> EvalCheck:
        """
        Run npm test and check pass rate.

        Config options:
        - command: Custom npm command (default: "npm test")
        - threshold: Minimum pass rate (default: 1.0)
        - timeout: Command timeout (default: 300)
        - working_dir: Working directory
        """
        command = config.get("command", "npm test")
        threshold = config.get("threshold", 1.0)
        timeout = config.get("timeout", 300)
        working_dir = config.get("working_dir")

        # Build command as list to prevent shell injection
        cmd_args = shlex.split(command)

        result = await self._run_command(
            cmd_args,
            working_dir=working_dir,
            timeout=timeout,
        )

        if result.timed_out:
            return self._create_check(
                name="npm",
                passed=False,
                message=f"npm test timed out after {timeout}s",
                details={
                    "timed_out": True,
                    "timeout": timeout,
                },
            )

        test_results = self._parse_npm_output(result.stdout, result.stderr)

        passed = test_results.pass_rate >= threshold

        return self._create_check(
            name="npm",
            passed=passed,
            message=(
                f"npm test: {test_results.passed}/{test_results.total} tests passed "
                f"({test_results.pass_rate:.1%}), threshold: {threshold:.1%}"
            ),
            details={
                "total": test_results.total,
                "passed": test_results.passed,
                "failed": test_results.failed,
                "skipped": test_results.skipped,
                "pass_rate": test_results.pass_rate,
                "threshold": threshold,
                "failure_messages": test_results.failure_messages,
                "return_code": result.return_code,
            },
        )

    async def _check_custom_command(self, config: dict[str, Any]) -> EvalCheck:
        """
        Run a custom command and parse output.

        Config options:
        - command: Command to run (required)
        - pattern: Regex pattern with named groups (passed, failed, total)
        - success_pattern: Pattern indicating success
        - threshold: Minimum pass rate (default: 1.0)
        - timeout: Command timeout (default: 300)
        - working_dir: Working directory
        - expect_zero: Require zero return code (default: True)
        """
        command = config.get("command")
        if not command:
            return self._create_check(
                name="custom_command",
                passed=False,
                message="No command specified in config",
            )

        pattern = config.get("pattern")
        success_pattern = config.get("success_pattern")
        threshold = config.get("threshold", 1.0)
        timeout = config.get("timeout", 300)
        working_dir = config.get("working_dir")
        expect_zero = config.get("expect_zero", True)

        # Split command to prevent shell injection
        cmd_args = shlex.split(command)

        result = await self._run_command(
            cmd_args,
            working_dir=working_dir,
            timeout=timeout,
        )

        if result.timed_out:
            return self._create_check(
                name="custom_command",
                passed=False,
                message=f"Command timed out after {timeout}s",
                details={
                    "command": command,
                    "timed_out": True,
                    "timeout": timeout,
                },
            )

        # Check return code if required
        if expect_zero and result.return_code != 0:
            return self._create_check(
                name="custom_command",
                passed=False,
                message=f"Command failed with return code {result.return_code}",
                details={
                    "command": command,
                    "return_code": result.return_code,
                    "stdout": result.stdout[:500],
                    "stderr": result.stderr[:500],
                },
            )

        # Parse output if pattern provided
        if pattern or success_pattern:
            test_results = self._parse_custom_output(
                result.stdout,
                result.stderr,
                pattern,
                success_pattern,
            )

            passed = test_results.pass_rate >= threshold

            return self._create_check(
                name="custom_command",
                passed=passed,
                message=(
                    f"Command: {test_results.passed}/{test_results.total} passed "
                    f"({test_results.pass_rate:.1%}), threshold: {threshold:.1%}"
                ),
                details={
                    "command": command,
                    "total": test_results.total,
                    "passed": test_results.passed,
                    "failed": test_results.failed,
                    "pass_rate": test_results.pass_rate,
                    "threshold": threshold,
                    "return_code": result.return_code,
                },
            )

        # No pattern - just check return code was success
        return self._create_check(
            name="custom_command",
            passed=True,
            message=(
                f"Command completed successfully (return code: {result.return_code})"
            ),
            details={
                "command": command,
                "return_code": result.return_code,
                "stdout": result.stdout[:500],
            },
        )

    async def _check_lint(self, config: dict[str, Any]) -> EvalCheck:
        """
        Run a linter and check violations.

        Config options:
        - linter: Linter name (ruff, eslint) - required
        - path: Path to lint (default: ".")
        - max_violations: Maximum allowed violations (default: 0)
        - max_errors: Maximum allowed errors (default: 0)
        - timeout: Command timeout (default: 300)
        - working_dir: Working directory
        - args: Additional linter arguments
        """
        linter = config.get("linter")
        if not linter:
            return self._create_check(
                name="lint",
                passed=False,
                message="No linter specified in config",
            )

        linter = linter.lower()
        if linter not in ("ruff", "eslint"):
            return self._create_check(
                name="lint",
                passed=False,
                message=f"Unsupported linter: {linter}. Supported: ruff, eslint",
            )

        path = config.get("path", ".")
        max_violations = config.get("max_violations", 0)
        max_errors = config.get("max_errors", 0)
        timeout = config.get("timeout", 300)
        working_dir = config.get("working_dir")
        extra_args = config.get("args", "")

        # Build command as list to prevent shell injection
        if linter == "ruff":
            cmd_args = ["ruff", "check", path]
        else:  # eslint
            cmd_args = ["eslint", path]
        if extra_args:
            cmd_args.extend(shlex.split(extra_args))

        result = await self._run_command(
            cmd_args,
            working_dir=working_dir,
            timeout=timeout,
        )

        if result.timed_out:
            return self._create_check(
                name="lint",
                passed=False,
                message=f"Linter timed out after {timeout}s",
                details={
                    "linter": linter,
                    "timed_out": True,
                    "timeout": timeout,
                },
            )

        # Parse output based on linter
        if linter == "ruff":
            lint_results = self._parse_ruff_output(result.stdout, result.stderr)
        else:
            lint_results = self._parse_eslint_output(result.stdout, result.stderr)

        # Check against thresholds
        passed = (
            lint_results.total_violations <= max_violations
            and lint_results.errors <= max_errors
        )

        return self._create_check(
            name="lint",
            passed=passed,
            message=(
                f"{linter}: {lint_results.total_violations} violations "
                f"({lint_results.errors} errors, {lint_results.warnings} warnings), "
                f"max allowed: {max_violations}"
            ),
            details={
                "linter": linter,
                "total_violations": lint_results.total_violations,
                "errors": lint_results.errors,
                "warnings": lint_results.warnings,
                "max_violations": max_violations,
                "max_errors": max_errors,
                "files_checked": lint_results.files_checked,
                "violation_messages": lint_results.violation_messages,
                "return_code": result.return_code,
            },
        )

    def _create_scored_check(
        self,
        name: str,
        score: float,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> EvalCheck:
        """Helper to create an EvalCheck with custom score."""
        return EvalCheck(
            name=name,
            passed=score >= 0.5,
            score=score,
            message=message,
            details=details,
        )
