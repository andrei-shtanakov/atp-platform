"""Unit tests for FilesystemEvaluator."""

from pathlib import Path

import pytest

from atp.evaluators.filesystem import FilesystemEvaluator
from atp.loader.models import Assertion, Constraints, TaskDefinition, TestDefinition
from atp.protocol import ATPResponse, ResponseStatus


@pytest.fixture
def evaluator() -> FilesystemEvaluator:
    """Create FilesystemEvaluator instance."""
    return FilesystemEvaluator()


@pytest.fixture
def sample_task() -> TestDefinition:
    """Create a sample test definition."""
    return TestDefinition(
        id="test-001",
        name="Filesystem Test",
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


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a workspace with test files."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "hello.txt").write_text("Hello, world!")
    (ws / "data").mkdir()
    (ws / "data" / "config.json").write_text('{"key": "value"}')
    (ws / "data" / "notes.txt").write_text("some notes")
    return ws


class TestFilesystemEvaluatorName:
    """Test evaluator name."""

    def test_name(self, evaluator: FilesystemEvaluator) -> None:
        """Evaluator name is 'filesystem'."""
        assert evaluator.name == "filesystem"


class TestFileExists:
    """Tests for file_exists assertion."""

    @pytest.mark.anyio
    async def test_file_exists_pass(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Pass when file exists."""
        assertion = Assertion(
            type="file_exists",
            config={"path": "hello.txt", "workspace_path": str(workspace)},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed
        assert result.score == 1.0

    @pytest.mark.anyio
    async def test_file_exists_fail(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Fail when file does not exist."""
        assertion = Assertion(
            type="file_exists",
            config={
                "path": "missing.txt",
                "workspace_path": str(workspace),
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed

    @pytest.mark.anyio
    async def test_file_exists_nested(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Pass for nested file."""
        assertion = Assertion(
            type="file_exists",
            config={
                "path": "data/config.json",
                "workspace_path": str(workspace),
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_file_exists_no_path(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Fail when no path specified."""
        assertion = Assertion(
            type="file_exists",
            config={"workspace_path": str(workspace)},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed


class TestFileNotExists:
    """Tests for file_not_exists assertion."""

    @pytest.mark.anyio
    async def test_file_not_exists_pass(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Pass when file does not exist."""
        assertion = Assertion(
            type="file_not_exists",
            config={
                "path": "missing.txt",
                "workspace_path": str(workspace),
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_file_not_exists_fail(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Fail when file exists."""
        assertion = Assertion(
            type="file_not_exists",
            config={
                "path": "hello.txt",
                "workspace_path": str(workspace),
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed


class TestFileContains:
    """Tests for file_contains assertion."""

    @pytest.mark.anyio
    async def test_contains_plain_text(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Pass when file contains plain text pattern."""
        assertion = Assertion(
            type="file_contains",
            config={
                "path": "hello.txt",
                "pattern": "Hello",
                "workspace_path": str(workspace),
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_contains_plain_text_fail(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Fail when pattern not found."""
        assertion = Assertion(
            type="file_contains",
            config={
                "path": "hello.txt",
                "pattern": "Goodbye",
                "workspace_path": str(workspace),
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed

    @pytest.mark.anyio
    async def test_contains_regex(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Pass with regex pattern."""
        assertion = Assertion(
            type="file_contains",
            config={
                "path": "hello.txt",
                "pattern": r"Hello,\s+\w+!",
                "regex": True,
                "workspace_path": str(workspace),
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_contains_missing_file(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Fail when file does not exist."""
        assertion = Assertion(
            type="file_contains",
            config={
                "path": "missing.txt",
                "pattern": "anything",
                "workspace_path": str(workspace),
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed


class TestDirExists:
    """Tests for dir_exists assertion."""

    @pytest.mark.anyio
    async def test_dir_exists_pass(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Pass when directory exists."""
        assertion = Assertion(
            type="dir_exists",
            config={"path": "data", "workspace_path": str(workspace)},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_dir_exists_fail(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Fail when directory does not exist."""
        assertion = Assertion(
            type="dir_exists",
            config={
                "path": "nonexistent",
                "workspace_path": str(workspace),
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed


class TestFileCount:
    """Tests for file_count assertion."""

    @pytest.mark.anyio
    async def test_file_count_eq(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Pass when file count matches."""
        assertion = Assertion(
            type="file_count",
            config={
                "path": "data",
                "count": 2,
                "operator": "eq",
                "workspace_path": str(workspace),
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_file_count_gte(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Pass when file count is >= threshold."""
        assertion = Assertion(
            type="file_count",
            config={
                "path": "data",
                "count": 1,
                "operator": "gte",
                "workspace_path": str(workspace),
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_file_count_fail(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Fail when count doesn't match."""
        assertion = Assertion(
            type="file_count",
            config={
                "path": "data",
                "count": 10,
                "operator": "eq",
                "workspace_path": str(workspace),
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed

    @pytest.mark.anyio
    async def test_file_count_missing_dir(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Fail when directory doesn't exist."""
        assertion = Assertion(
            type="file_count",
            config={
                "path": "missing",
                "count": 0,
                "workspace_path": str(workspace),
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed


class TestNoWorkspacePath:
    """Tests for missing workspace_path."""

    @pytest.mark.anyio
    async def test_no_workspace_path(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Fail when workspace_path is missing."""
        assertion = Assertion(
            type="file_exists",
            config={"path": "hello.txt"},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed
        assert "workspace_path" in result.checks[0].message


class TestUnknownAssertionType:
    """Tests for unknown assertion type."""

    @pytest.mark.anyio
    async def test_unknown_type(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """Fail for unknown assertion type."""
        assertion = Assertion(
            type="unknown_filesystem_type",
            config={"workspace_path": str(workspace)},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed
        assert "Unknown" in result.checks[0].message
