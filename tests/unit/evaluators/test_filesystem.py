"""Unit tests for FilesystemEvaluator.

The root is granted by the composition, never declared by the suite (ADR-008
track B), so these tests bind a root the way the pipeline does rather than
passing `workspace_path` as if it were one. The tests that still pass
`workspace_path` are the ones about what it means now: a relative subpath, or
a config error.
"""

import logging
from pathlib import Path

import pytest

from atp.evaluators.filesystem import FilesystemEvaluator, WorkspaceNotGranted
from atp.loader.models import Assertion, Constraints, TaskDefinition, TestDefinition
from atp.protocol import ATPResponse, ResponseStatus


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


@pytest.fixture
def evaluator(workspace: Path) -> FilesystemEvaluator:
    """An evaluator granted the workspace, as a composition would grant it."""
    return FilesystemEvaluator(workspace)


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
    ) -> None:
        """Pass when file exists."""
        assertion = Assertion(type="file_exists", config={"path": "hello.txt"})
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed
        assert result.score == 1.0

    @pytest.mark.anyio
    async def test_file_exists_fail(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Fail when file does not exist."""
        assertion = Assertion(type="file_exists", config={"path": "missing.txt"})
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed

    @pytest.mark.anyio
    async def test_file_exists_nested(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Pass for nested file."""
        assertion = Assertion(type="file_exists", config={"path": "data/config.json"})
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_file_exists_no_path(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Fail when no path specified."""
        assertion = Assertion(type="file_exists", config={})
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
    ) -> None:
        """Pass when file does not exist."""
        assertion = Assertion(type="file_not_exists", config={"path": "missing.txt"})
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_file_not_exists_fail(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Fail when file exists."""
        assertion = Assertion(type="file_not_exists", config={"path": "hello.txt"})
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "path",
        ["../escape.txt", "/etc/passwd", "data/../../escape.txt"],
    )
    async def test_unresolvable_path_does_not_pass(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        path: str,
    ) -> None:
        """Regression (ADR-008 track B): this used to answer `passed=True`.

        "The path is invalid, so treat it as not existing" reads as reasonable
        and is backwards on a policy boundary. The evaluator never resolved the
        path, so it never asked the question — and awarding the point hands it
        to exactly the malformed assertion probing the boundary. It is a config
        error, like the same path under `file_exists`.
        """
        assertion = Assertion(type="file_not_exists", config={"path": path})
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed
        assert "Invalid path" in result.checks[0].message

    @pytest.mark.anyio
    async def test_symlink_out_of_the_workspace_does_not_pass(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
        tmp_path: Path,
    ) -> None:
        """A link is a path too, and it must not become an oracle either."""
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        (workspace / "link.txt").symlink_to(outside)

        assertion = Assertion(type="file_not_exists", config={"path": "link.txt"})
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
    ) -> None:
        """Pass when file contains plain text pattern."""
        assertion = Assertion(
            type="file_contains",
            config={"path": "hello.txt", "pattern": "Hello"},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_contains_plain_text_fail(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Fail when pattern not found."""
        assertion = Assertion(
            type="file_contains",
            config={"path": "hello.txt", "pattern": "Goodbye"},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed

    @pytest.mark.anyio
    async def test_contains_regex(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Pass with regex pattern."""
        assertion = Assertion(
            type="file_contains",
            config={
                "path": "hello.txt",
                "pattern": r"Hello,\s+\w+!",
                "regex": True,
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
    ) -> None:
        """Fail when file does not exist."""
        assertion = Assertion(
            type="file_contains",
            config={"path": "missing.txt", "pattern": "anything"},
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
    ) -> None:
        """Pass when directory exists."""
        assertion = Assertion(type="dir_exists", config={"path": "data"})
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_dir_exists_fail(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Fail when directory does not exist."""
        assertion = Assertion(type="dir_exists", config={"path": "nonexistent"})
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
    ) -> None:
        """Pass when file count matches."""
        assertion = Assertion(
            type="file_count",
            config={"path": "data", "count": 2, "operator": "eq"},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_file_count_gte(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Pass when file count is >= threshold."""
        assertion = Assertion(
            type="file_count",
            config={"path": "data", "count": 1, "operator": "gte"},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_file_count_fail(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Fail when count doesn't match."""
        assertion = Assertion(
            type="file_count",
            config={"path": "data", "count": 10, "operator": "eq"},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed

    @pytest.mark.anyio
    async def test_file_count_missing_dir(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Fail when directory doesn't exist."""
        assertion = Assertion(
            type="file_count",
            config={"path": "missing", "count": 0},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed


class TestTheRootIsGranted:
    """Where this evaluator may look is decided by its composition."""

    @pytest.mark.anyio
    async def test_an_ungranted_evaluator_refuses_rather_than_guessing(
        self,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Fail-closed: no root means no fallback, not the working directory.

        Raised rather than answered `False`, so it lands in the pipeline's
        evaluator-error bucket with a message. A wiring fault reported as a
        failed assertion would read as "the agent did not produce the file".
        """
        evaluator = FilesystemEvaluator()
        assertion = Assertion(type="file_exists", config={"path": "hello.txt"})
        with pytest.raises(WorkspaceNotGranted):
            await evaluator.evaluate(sample_task, sample_response, [], assertion)

    @pytest.mark.anyio
    async def test_binding_grants_the_root(
        self,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
    ) -> None:
        """The pipeline binds; the evaluator then answers about that directory."""
        evaluator = FilesystemEvaluator()
        evaluator.bind_workspace_root(workspace)
        assertion = Assertion(type="file_exists", config={"path": "hello.txt"})
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_workspace_path_is_a_subpath_of_the_granted_root(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """The field keeps its usefulness: address a subdirectory of the root."""
        assertion = Assertion(
            type="file_exists",
            config={"path": "config.json", "workspace_path": "data"},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    @pytest.mark.parametrize("declared", ["/etc", "../..", "data/../../.."])
    async def test_a_workspace_path_outside_the_root_is_a_config_error(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        declared: str,
    ) -> None:
        """Reported, not clamped and not ignored.

        Silently ignoring the field would drop the suite author's intent with
        no diagnostic, which is how a suite ends up measuring something nobody
        asked for and nobody notices.
        """
        assertion = Assertion(
            type="file_exists",
            config={"path": "hello.txt", "workspace_path": declared},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed
        assert "workspace_path" in result.checks[0].message

    @pytest.mark.anyio
    async def test_the_untrusted_plane_never_reads_a_suite_named_directory(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        tmp_path: Path,
    ) -> None:
        """The defect this track exists to close, stated as a behaviour.

        An absolute `workspace_path` naming a real directory full of real
        files must not turn the benchmark into an existence oracle for the
        server's own disk.
        """
        elsewhere = tmp_path / "server-secrets"
        elsewhere.mkdir()
        (elsewhere / "hello.txt").write_text("not the submission")

        assertion = Assertion(
            type="file_exists",
            config={"path": "hello.txt", "workspace_path": str(elsewhere)},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed
        assert "absolute path is not permitted" in result.checks[0].message


class TestLegacyAbsoluteWorkspacePath:
    """The pre-ADR-008 form, converted explicitly and only where it was valid."""

    @pytest.mark.anyio
    async def test_honoured_on_the_trusted_plane_with_a_warning(
        self,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A CLI suite written against the old meaning keeps working, audibly."""
        evaluator = FilesystemEvaluator(tmp_path, trusted=True)
        assertion = Assertion(
            type="file_exists",
            config={"path": "hello.txt", "workspace_path": str(workspace)},
        )
        with caplog.at_level(logging.WARNING, logger="atp.evaluators.filesystem"):
            result = await evaluator.evaluate(
                sample_task, sample_response, [], assertion
            )
        assert result.passed
        assert "Deprecated" in caplog.text

    @pytest.mark.anyio
    async def test_trust_is_bound_not_assumed(
        self,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        workspace: Path,
        tmp_path: Path,
    ) -> None:
        """Re-binding without trust withdraws the legacy form with it."""
        evaluator = FilesystemEvaluator(tmp_path, trusted=True)
        evaluator.bind_workspace_root(tmp_path)
        assertion = Assertion(
            type="file_exists",
            config={"path": "hello.txt", "workspace_path": str(workspace)},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed


class TestUnknownAssertionType:
    """Tests for unknown assertion type."""

    @pytest.mark.anyio
    async def test_unknown_type(
        self,
        evaluator: FilesystemEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Fail for unknown assertion type."""
        assertion = Assertion(type="unknown_filesystem_type", config={})
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert not result.passed
        assert "Unknown" in result.checks[0].message
