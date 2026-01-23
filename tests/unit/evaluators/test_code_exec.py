"""Unit tests for CodeExecEvaluator."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from atp.evaluators.code_exec import (
    CodeExecEvaluator,
    CodeTestResults,
    CommandResult,
    LintResults,
)
from atp.loader.models import Assertion, Constraints, TaskDefinition, TestDefinition
from atp.protocol import ATPResponse, ResponseStatus


@pytest.fixture
def evaluator() -> CodeExecEvaluator:
    """Create CodeExecEvaluator instance."""
    return CodeExecEvaluator()


@pytest.fixture
def sample_task() -> TestDefinition:
    """Create a sample test definition."""
    return TestDefinition(
        id="test-001",
        name="Sample Test",
        task=TaskDefinition(description="Test task"),
        constraints=Constraints(),
    )


@pytest.fixture
def sample_response() -> ATPResponse:
    """Create a sample response."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[],
    )


def mock_command_result(
    return_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
) -> CommandResult:
    """Helper to create mock command results."""
    return CommandResult(
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )


class TestEvaluatorProperties:
    """Tests for evaluator properties."""

    def test_evaluator_name(self, evaluator: CodeExecEvaluator) -> None:
        """Test evaluator name property."""
        assert evaluator.name == "code_exec"


class TestPytestParsing:
    """Tests for pytest output parsing."""

    def test_parse_pytest_all_passed(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing pytest output with all tests passed."""
        stdout = """
============================= test session starts ==============================
platform linux -- Python 3.12.0, pytest-8.0.0
collected 5 items

tests/test_example.py .....                                              [100%]

============================== 5 passed in 0.12s ===============================
"""
        result = evaluator._parse_pytest_output(stdout, "")
        assert result.total == 5
        assert result.passed == 5
        assert result.failed == 0
        assert result.pass_rate == 1.0

    def test_parse_pytest_with_failures(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing pytest output with failures."""
        stdout = """
============================= test session starts ==============================
collected 10 items

tests/test_example.py .....FF...                                         [100%]

================================== FAILURES ===================================
____________________________ test_something ____________________________________
...
____________________________ test_another ______________________________________
...
=========================== short test summary info ============================
FAILED tests/test_example.py::test_something
FAILED tests/test_example.py::test_another
========================= 8 passed, 2 failed in 1.23s =========================
"""
        result = evaluator._parse_pytest_output(stdout, "")
        assert result.total == 10
        assert result.passed == 8
        assert result.failed == 2
        assert result.pass_rate == 0.8
        assert len(result.failure_messages) > 0

    def test_parse_pytest_with_skipped(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing pytest output with skipped tests."""
        stdout = """
============================= test session starts ==============================
collected 5 items

tests/test_example.py ...ss                                              [100%]

========================= 3 passed, 2 skipped in 0.05s =========================
"""
        result = evaluator._parse_pytest_output(stdout, "")
        assert result.total == 5
        assert result.passed == 3
        assert result.skipped == 2
        assert result.pass_rate == 0.6

    def test_parse_pytest_with_errors(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing pytest output with errors."""
        stdout = """
============================= test session starts ==============================
collected 3 items

tests/test_example.py E..                                                [100%]

=================================== ERRORS ====================================
_________________ ERROR collecting tests/test_broken.py _______________________
...
========================= 2 passed, 1 error in 0.45s ==========================
"""
        result = evaluator._parse_pytest_output(stdout, "")
        assert result.total == 3
        assert result.passed == 2
        assert result.errors == 1

    def test_parse_pytest_no_tests(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing pytest output when no tests collected."""
        stdout = """
============================= test session starts ==============================
collected 0 items

=============================== no tests ran in 0.01s =========================
"""
        result = evaluator._parse_pytest_output(stdout, "")
        assert result.total == 0
        assert result.pass_rate == 0.0


class TestNpmParsing:
    """Tests for npm test output parsing."""

    def test_parse_jest_all_passed(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing Jest output with all tests passed."""
        stdout = """
PASS src/test.js
  ✓ should work (5 ms)
  ✓ should also work (3 ms)

Test Suites: 1 passed, 1 total
Tests:       5 passed, 5 total
Snapshots:   0 total
Time:        2.34 s
"""
        result = evaluator._parse_npm_output(stdout, "")
        assert result.total == 5
        assert result.passed == 5
        assert result.failed == 0
        assert result.pass_rate == 1.0

    def test_parse_jest_with_failures(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing Jest output with failures."""
        stdout = """
FAIL src/test.js
  ✕ should fail (12 ms)
  ✓ should pass (3 ms)

Test Suites: 1 failed, 1 total
Tests:       1 failed, 2 passed, 3 total
Snapshots:   0 total
Time:        1.23 s
"""
        result = evaluator._parse_npm_output(stdout, "")
        assert result.total == 3
        assert result.passed == 2
        assert result.failed == 1
        assert result.pass_rate == 2 / 3

    def test_parse_mocha_output(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing Mocha output."""
        stdout = """

  Example
    ✓ should do something
    ✓ should do another thing
    - should be pending


  2 passing (15ms)
  1 pending
"""
        result = evaluator._parse_npm_output(stdout, "")
        assert result.passed == 2
        assert result.skipped == 1
        assert result.total == 3

    def test_parse_npm_error(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing npm error output."""
        stderr = """
npm ERR! Test failed. See above for more details.
"""
        result = evaluator._parse_npm_output("", stderr)
        assert result.pass_rate == 0.0


class TestLintParsing:
    """Tests for lint output parsing."""

    def test_parse_ruff_clean(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing ruff output with no violations."""
        stdout = """
All checks passed!
"""
        result = evaluator._parse_ruff_output(stdout, "")
        assert result.total_violations == 0
        assert result.errors == 0
        assert result.warnings == 0

    def test_parse_ruff_with_violations(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing ruff output with violations."""
        stdout = """
src/module.py:10:5: E501 Line too long (120 > 88)
src/module.py:15:1: F401 `os` imported but unused
src/other.py:5:10: W503 Line break before binary operator
Found 3 errors.
"""
        result = evaluator._parse_ruff_output(stdout, "")
        assert result.total_violations == 3
        # Only E-prefix codes are counted as errors, others as warnings
        assert result.errors == 1  # E501
        assert result.warnings == 2  # F401, W503
        assert len(result.violation_messages) > 0

    def test_parse_eslint_clean(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing eslint output with no violations."""
        stdout = ""
        result = evaluator._parse_eslint_output(stdout, "")
        assert result.total_violations == 0

    def test_parse_eslint_with_violations(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing eslint output with violations."""
        stdout = """
/src/file.js
  10:5   error    Unexpected var, use let or const instead  no-var
  15:10  warning  Unexpected console statement              no-console

/src/other.js
   5:1   error    'foo' is defined but never used           no-unused-vars

✖ 3 problems (2 errors, 1 warning)
"""
        result = evaluator._parse_eslint_output(stdout, "")
        assert result.total_violations == 3
        assert result.errors == 2
        assert result.warnings == 1


class TestCustomParsing:
    """Tests for custom command output parsing."""

    def test_parse_with_pattern(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing with custom regex pattern."""
        stdout = "Results: passed=8, failed=2, total=10"
        pattern = (
            r"passed=(?P<passed>\d+), failed=(?P<failed>\d+), total=(?P<total>\d+)"
        )
        result = evaluator._parse_custom_output(stdout, "", pattern, None)
        assert result.total == 10
        assert result.passed == 8
        assert result.failed == 2
        assert result.pass_rate == 0.8

    def test_parse_with_success_pattern(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing with success pattern."""
        stdout = "BUILD SUCCESSFUL"
        result = evaluator._parse_custom_output(stdout, "", None, r"BUILD SUCCESSFUL")
        assert result.pass_rate == 1.0
        assert result.passed == 1

    def test_parse_invalid_pattern(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing with invalid regex pattern."""
        stdout = "some output"
        pattern = r"[invalid"  # Invalid regex
        result = evaluator._parse_custom_output(stdout, "", pattern, None)
        # Should not crash, just return defaults
        assert result.pass_rate == 0.0


class TestPytestCheck:
    """Tests for pytest check."""

    @pytest.mark.anyio
    async def test_pytest_pass(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test pytest check passes when tests pass."""
        mock_result = mock_command_result(
            stdout="5 passed in 0.12s",
            return_code=0,
        )

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            assertion = Assertion(
                type="pytest",
                config={"path": "tests/", "threshold": 1.0},
            )
            result = await evaluator.evaluate(
                sample_task, sample_response, [], assertion
            )

        assert result.passed is True
        assert result.checks[0].name == "pytest"
        assert result.checks[0].details["passed"] == 5

    @pytest.mark.anyio
    async def test_pytest_fail_threshold(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test pytest check fails when below threshold."""
        mock_result = mock_command_result(
            stdout="8 passed, 2 failed in 1.0s",
            return_code=1,
        )

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            assertion = Assertion(
                type="pytest",
                config={"path": "tests/", "threshold": 0.9},
            )
            result = await evaluator.evaluate(
                sample_task, sample_response, [], assertion
            )

        assert result.passed is False
        assert result.checks[0].details["pass_rate"] == 0.8

    @pytest.mark.anyio
    async def test_pytest_timeout(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test pytest check handles timeout."""
        mock_result = mock_command_result(timed_out=True)

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            assertion = Assertion(
                type="pytest",
                config={"path": "tests/", "timeout": 10},
            )
            result = await evaluator.evaluate(
                sample_task, sample_response, [], assertion
            )

        assert result.passed is False
        assert "timed out" in result.checks[0].message.lower()


class TestNpmCheck:
    """Tests for npm check."""

    @pytest.mark.anyio
    async def test_npm_pass(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test npm check passes when tests pass."""
        mock_result = mock_command_result(
            stdout="Tests: 10 passed, 10 total",
            return_code=0,
        )

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            assertion = Assertion(
                type="npm",
                config={"threshold": 1.0},
            )
            result = await evaluator.evaluate(
                sample_task, sample_response, [], assertion
            )

        assert result.passed is True
        assert result.checks[0].name == "npm"

    @pytest.mark.anyio
    async def test_npm_custom_command(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test npm check with custom command."""
        mock_result = mock_command_result(
            stdout="Tests: 5 passed, 5 total",
            return_code=0,
        )

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            assertion = Assertion(
                type="npm",
                config={"command": "npm run test:unit"},
            )
            await evaluator.evaluate(sample_task, sample_response, [], assertion)

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert "npm run test:unit" in str(call_args)


class TestCustomCommandCheck:
    """Tests for custom command check."""

    @pytest.mark.anyio
    async def test_custom_command_pass(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test custom command passes with zero return code."""
        mock_result = mock_command_result(
            stdout="Success!",
            return_code=0,
        )

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            assertion = Assertion(
                type="custom_command",
                config={"command": "make test"},
            )
            result = await evaluator.evaluate(
                sample_task, sample_response, [], assertion
            )

        assert result.passed is True
        assert "completed successfully" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_custom_command_no_command(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test custom command fails when no command specified."""
        assertion = Assertion(
            type="custom_command",
            config={},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)

        assert result.passed is False
        assert "no command" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_custom_command_with_pattern(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test custom command with result pattern."""
        mock_result = mock_command_result(
            stdout="Results: passed=8 failed=2",
            return_code=0,
        )

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            assertion = Assertion(
                type="custom_command",
                config={
                    "command": "run_tests.sh",
                    "pattern": r"passed=(?P<passed>\d+) failed=(?P<failed>\d+)",
                    "threshold": 0.9,
                },
            )
            result = await evaluator.evaluate(
                sample_task, sample_response, [], assertion
            )

        assert result.passed is False  # 80% < 90%
        assert result.checks[0].details["pass_rate"] == 0.8

    @pytest.mark.anyio
    async def test_custom_command_nonzero_exit(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test custom command fails on nonzero exit code."""
        mock_result = mock_command_result(
            stdout="Error occurred",
            stderr="Some error",
            return_code=1,
        )

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            assertion = Assertion(
                type="custom_command",
                config={"command": "failing_command"},
            )
            result = await evaluator.evaluate(
                sample_task, sample_response, [], assertion
            )

        assert result.passed is False
        assert "return code 1" in result.checks[0].message.lower()


class TestLintCheck:
    """Tests for lint check."""

    @pytest.mark.anyio
    async def test_lint_ruff_pass(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test lint check passes with no violations."""
        mock_result = mock_command_result(
            stdout="All checks passed!",
            return_code=0,
        )

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            assertion = Assertion(
                type="lint",
                config={"linter": "ruff", "max_violations": 0},
            )
            result = await evaluator.evaluate(
                sample_task, sample_response, [], assertion
            )

        assert result.passed is True
        assert result.checks[0].details["total_violations"] == 0

    @pytest.mark.anyio
    async def test_lint_ruff_fail(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test lint check fails with violations."""
        mock_result = mock_command_result(
            stdout="file.py:10:5: E501 Line too long\nFound 1 error.",
            return_code=1,
        )

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            assertion = Assertion(
                type="lint",
                config={"linter": "ruff", "max_violations": 0},
            )
            result = await evaluator.evaluate(
                sample_task, sample_response, [], assertion
            )

        assert result.passed is False
        assert result.checks[0].details["total_violations"] >= 1

    @pytest.mark.anyio
    async def test_lint_eslint_pass(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test eslint check passes with allowed violations."""
        mock_result = mock_command_result(
            stdout="✖ 2 problems (0 errors, 2 warnings)",
            return_code=0,
        )

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            assertion = Assertion(
                type="lint",
                config={"linter": "eslint", "max_violations": 5},
            )
            result = await evaluator.evaluate(
                sample_task, sample_response, [], assertion
            )

        assert result.passed is True

    @pytest.mark.anyio
    async def test_lint_no_linter(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test lint check fails when no linter specified."""
        assertion = Assertion(
            type="lint",
            config={},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)

        assert result.passed is False
        assert "no linter" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_lint_unsupported_linter(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test lint check fails for unsupported linter."""
        assertion = Assertion(
            type="lint",
            config={"linter": "unknown_linter"},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)

        assert result.passed is False
        assert "unsupported" in result.checks[0].message.lower()


class TestUnknownAssertionType:
    """Tests for unknown assertion types."""

    @pytest.mark.anyio
    async def test_unknown_type(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test unknown assertion type returns failure."""
        assertion = Assertion(type="unknown_type", config={})
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)

        assert result.passed is False
        assert "unknown" in result.checks[0].message.lower()


class TestCommandExecution:
    """Tests for command execution."""

    @pytest.mark.anyio
    async def test_command_not_found(self, evaluator: CodeExecEvaluator) -> None:
        """Test handling of command not found."""
        result = await evaluator._run_command("nonexistent_command_xyz")
        assert result.return_code == -1
        assert "not found" in result.stderr.lower()

    @pytest.mark.anyio
    async def test_command_with_working_dir(
        self,
        evaluator: CodeExecEvaluator,
        tmp_path: Any,
    ) -> None:
        """Test command execution with working directory."""
        # Create a test file in temp directory
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        result = await evaluator._run_command(
            "ls",
            working_dir=tmp_path,
        )
        assert result.return_code == 0
        assert "test.txt" in result.stdout

    @pytest.mark.anyio
    async def test_command_timeout(self, evaluator: CodeExecEvaluator) -> None:
        """Test command timeout handling."""
        result = await evaluator._run_command(
            "sleep 10",
            timeout=1,
        )
        assert result.timed_out is True
        assert result.return_code == -1


class TestDataClasses:
    """Tests for data classes."""

    def test_command_result(self) -> None:
        """Test CommandResult dataclass."""
        result = CommandResult(
            return_code=0,
            stdout="output",
            stderr="",
            timed_out=False,
        )
        assert result.return_code == 0
        assert result.stdout == "output"
        assert result.timed_out is False

    def test_code_test_results(self) -> None:
        """Test CodeTestResults dataclass."""
        result = CodeTestResults(
            total=10,
            passed=8,
            failed=2,
            skipped=0,
            errors=0,
            pass_rate=0.8,
            failure_messages=["test1", "test2"],
        )
        assert result.pass_rate == 0.8
        assert len(result.failure_messages) == 2

    def test_lint_results(self) -> None:
        """Test LintResults dataclass."""
        result = LintResults(
            total_violations=5,
            errors=3,
            warnings=2,
            files_checked=10,
            violation_messages=["msg1"],
        )
        assert result.total_violations == 5
        assert result.errors == 3


class TestRegistration:
    """Tests for evaluator registration."""

    def test_evaluator_registered(self) -> None:
        """Test that CodeExecEvaluator is registered."""
        from atp.evaluators import get_registry

        registry = get_registry()
        assert registry.is_registered("code_exec")

    def test_assertion_types_supported(self) -> None:
        """Test that assertion types are supported."""
        from atp.evaluators import get_registry

        registry = get_registry()
        assert registry.supports_assertion("pytest")
        assert registry.supports_assertion("npm")
        assert registry.supports_assertion("custom_command")
        assert registry.supports_assertion("lint")

    def test_create_evaluator(self) -> None:
        """Test creating evaluator from registry."""
        from atp.evaluators import create_evaluator

        evaluator = create_evaluator("code_exec")
        assert evaluator.name == "code_exec"
        assert isinstance(evaluator, CodeExecEvaluator)


class TestEdgeCases:
    """Edge case tests."""

    def test_parse_pytest_empty_output(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing empty pytest output."""
        result = evaluator._parse_pytest_output("", "")
        assert result.total == 0
        # Empty output with no indication of tests = pass (default state)
        assert result.pass_rate == 1.0

    def test_parse_npm_empty_output(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing empty npm output."""
        result = evaluator._parse_npm_output("", "")
        assert result.total == 0
        assert result.pass_rate == 1.0

    def test_parse_ruff_partial_output(self, evaluator: CodeExecEvaluator) -> None:
        """Test parsing partial ruff output."""
        stdout = "file.py:1:1: E501 message"
        result = evaluator._parse_ruff_output(stdout, "")
        assert result.total_violations == 1

    @pytest.mark.anyio
    async def test_lint_with_extra_args(
        self,
        evaluator: CodeExecEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test lint check with extra arguments."""
        mock_result = mock_command_result(
            stdout="All checks passed!",
            return_code=0,
        )

        with patch.object(
            evaluator, "_run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            assertion = Assertion(
                type="lint",
                config={
                    "linter": "ruff",
                    "path": "src/",
                    "args": "--ignore E501",
                },
            )
            await evaluator.evaluate(sample_task, sample_response, [], assertion)

            call_args = mock_run.call_args
            command = call_args[0][0]
            assert "ruff check" in command
            assert "src/" in command
            assert "--ignore E501" in command
