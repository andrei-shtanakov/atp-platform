"""Adversarial tests for the artifact workspace.

The input is a path chosen by whoever submitted the run. The CLI writes such
paths straight into the working directory, which is an arbitrary file write
the moment the submitter is not the operator. Each test here corresponds to a
way that goes wrong, and asserts two things: nothing was written outside the
sandbox, and the attempt was *recorded* rather than silently dropped.

Rejection is deliberately not an exception. One malformed artifact must not
suppress the whole evaluation, or a submitter could avoid a bad score by
sending a poisoned path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from atp.evaluation.workspace import (
    ArtifactWorkspace,
    RejectReason,
    WorkspaceLimits,
)
from atp.protocol import ATPResponse
from atp.protocol.models import ArtifactFile


def _response(*artifacts: ArtifactFile) -> ATPResponse:
    """A completed response carrying the given artifacts."""
    return ATPResponse(task_id="t1", status="completed", artifacts=list(artifacts))


def _artifact(path: str | None, content: str = "x") -> ArtifactFile:
    """One inline file artifact."""
    return ArtifactFile(type="file", path=path, content=content)


def _reasons(workspace: ArtifactWorkspace) -> list[str]:
    """Rejection reasons, in order."""
    return [r.reason for r in workspace.report.rejected]


class TestProtocolAlreadyBlocksTheseAtParseTime:
    """Absolute paths, `..` and empty paths never reach the workspace.

    `ArtifactFile.validate_artifact_path` rejects them when the response is
    parsed, so these cannot be exercised by building a response — the model
    refuses to construct. The workspace checks them anyway, and the checks are
    driven directly here: a control that lives in exactly one layer is one
    refactor away from not existing, and `model_construct` skips validation.
    """

    @pytest.mark.parametrize("bad", ["/etc/passwd", "a/../../escaped.txt", "   "])
    def test_model_refuses_to_build_such_an_artifact(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _artifact(bad)

    def test_workspace_still_rejects_an_absolute_path(self, tmp_path: Path) -> None:
        workspace = ArtifactWorkspace()
        assert workspace._check(tmp_path, "/etc/passwd", "x", set()) is None
        assert _reasons(workspace) == [RejectReason.ABSOLUTE_PATH]

    def test_workspace_still_rejects_traversal_after_resolution(
        self, tmp_path: Path
    ) -> None:
        """`a/../../x` only escapes once resolved, so resolution comes first."""
        workspace = ArtifactWorkspace()
        assert workspace._check(tmp_path, "a/../../escaped.txt", "x", set()) is None
        assert _reasons(workspace) == [RejectReason.ESCAPES_WORKSPACE]

    def test_workspace_still_rejects_an_empty_path(self, tmp_path: Path) -> None:
        workspace = ArtifactWorkspace()
        assert workspace._check(tmp_path, "   ", "x", set()) is None
        assert _reasons(workspace) == [RejectReason.EMPTY_PATH]


class TestPathEscapes:
    """What the protocol cannot know about: links, reserved names, the sandbox."""

    def test_traversal_that_lands_back_inside_is_allowed(self, tmp_path: Path) -> None:
        """The rule is containment, not shape — driven directly, since the
        protocol refuses to build an artifact containing `..` at all."""
        workspace = ArtifactWorkspace()
        assert workspace._check(tmp_path, "a/../b.txt", "x", set()) == Path("b.txt")
        assert workspace.report.rejected == []

    def test_symlink_component_is_rejected(self, tmp_path: Path) -> None:
        """A link planted by an earlier artifact must not be written through.

        Driven directly against a prepared directory, since a fresh temp dir
        never contains a symlink of its own.
        """
        (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
        workspace = ArtifactWorkspace()
        assert workspace._check(tmp_path, "link/file.txt", "x", set()) is None
        assert _reasons(workspace) == [RejectReason.SYMLINK_IN_PATH]

    @pytest.mark.parametrize("name", [".atp/config", ".atp-workspace/x.txt"])
    def test_reserved_names_are_rejected(self, name: str) -> None:
        """The workspace's own bookkeeping is not writable by a submission."""
        workspace = ArtifactWorkspace()
        with workspace.prepare(_response(_artifact(name))):
            pass
        assert _reasons(workspace) == [RejectReason.RESERVED_NAME]


class TestBounds:
    """An unbounded payload is a denial-of-service, not an evaluation."""

    def test_oversized_file_is_rejected(self) -> None:
        workspace = ArtifactWorkspace(WorkspaceLimits(max_file_bytes=10))
        with workspace.prepare(_response(_artifact("big.txt", "y" * 11))):
            pass
        assert _reasons(workspace) == [RejectReason.FILE_TOO_LARGE]

    def test_file_count_limit_is_enforced(self) -> None:
        workspace = ArtifactWorkspace(WorkspaceLimits(max_files=2))
        artifacts = [_artifact(f"f{n}.txt") for n in range(4)]
        with workspace.prepare(_response(*artifacts)):
            pass
        assert len(workspace.report.written) == 2
        assert _reasons(workspace) == [RejectReason.TOO_MANY_FILES] * 2

    def test_total_size_budget_is_enforced(self) -> None:
        """Many small files can exceed a budget no single file violates."""
        workspace = ArtifactWorkspace(
            WorkspaceLimits(max_file_bytes=100, max_total_bytes=10)
        )
        artifacts = [_artifact(f"f{n}.txt", "abcd") for n in range(4)]
        with workspace.prepare(_response(*artifacts)):
            pass
        assert workspace.report.total_bytes <= 10
        assert RejectReason.TOTAL_TOO_LARGE in _reasons(workspace)

    def test_duplicate_path_is_rejected_rather_than_overwriting(self) -> None:
        """Otherwise a later artifact silently replaces what was scored."""
        workspace = ArtifactWorkspace()
        with workspace.prepare(
            _response(_artifact("a.txt", "first"), _artifact("a.txt", "second"))
        ):
            assert workspace.root is not None
            assert (workspace.root / "a.txt").read_text() == "first"
        assert _reasons(workspace) == [RejectReason.DUPLICATE_PATH]


class TestNothingEscapesAsAnException:
    """The module promises rejection, not throwing. These used to throw."""

    def test_a_path_naming_the_workspace_root_is_rejected(self) -> None:
        """`.` resolves to a directory; writing it raised IsADirectoryError."""
        workspace = ArtifactWorkspace()
        with workspace.prepare(_response(_artifact(".", "x"))):
            pass
        assert _reasons(workspace) == [RejectReason.NOT_A_FILE]

    def test_file_then_directory_of_the_same_name_is_rejected(self) -> None:
        """`a` as a file, then `a/b.txt`: mkdir used to raise FileExistsError."""
        workspace = ArtifactWorkspace()
        with workspace.prepare(
            _response(_artifact("a", "file"), _artifact("a/b.txt", "child"))
        ):
            pass
        assert workspace.report.written == ["a"]
        assert _reasons(workspace) == [RejectReason.PATH_COLLISION]

    def test_directory_then_file_of_the_same_name_is_rejected(self) -> None:
        """The reverse order raised IsADirectoryError instead."""
        workspace = ArtifactWorkspace()
        with workspace.prepare(
            _response(_artifact("a/b.txt", "child"), _artifact("a", "file"))
        ):
            pass
        assert workspace.report.written == ["a/b.txt"]
        assert _reasons(workspace) == [RejectReason.PATH_COLLISION]

    def test_a_collision_does_not_stop_later_artifacts(self) -> None:
        """The whole point: a poisoned path costs points, not the run."""
        workspace = ArtifactWorkspace()
        with workspace.prepare(
            _response(
                _artifact("a", "file"),
                _artifact("a/b.txt", "child"),
                _artifact("later.txt", "kept"),
            )
        ):
            assert workspace.root is not None
            assert (workspace.root / "later.txt").read_text() == "kept"


class TestReuse:
    """A reused instance must not spend another submission's budget."""

    def test_report_is_reset_between_evaluations(self) -> None:
        workspace = ArtifactWorkspace()
        with workspace.prepare(_response(_artifact("first.txt", "aaaa"))):
            pass
        with workspace.prepare(_response(_artifact("second.txt", "bb"))):
            pass
        assert workspace.report.written == ["second.txt"]
        assert workspace.report.total_bytes == 2

    def test_rejections_do_not_leak_into_the_next_report(self) -> None:
        workspace = ArtifactWorkspace(WorkspaceLimits(max_file_bytes=2))
        with workspace.prepare(_response(_artifact("big.txt", "toolong"))):
            pass
        assert workspace.report.rejected
        with workspace.prepare(_response(_artifact("ok.txt", "x"))):
            pass
        assert workspace.report.clean is True

    def test_file_count_budget_does_not_accumulate(self) -> None:
        """Otherwise the second submission is judged by the first one's usage."""
        workspace = ArtifactWorkspace(WorkspaceLimits(max_files=2))
        for _ in range(3):
            with workspace.prepare(_response(_artifact("a.txt"), _artifact("b.txt"))):
                pass
            assert workspace.report.clean is True


class TestLifecycle:
    """A sandbox that outlives the evaluation is not a sandbox."""

    def test_directory_is_removed_on_success(self) -> None:
        workspace = ArtifactWorkspace()
        with workspace.prepare(_response(_artifact("a.txt"))):
            root = workspace.root
            assert root is not None and root.exists()
        assert not root.exists()

    def test_directory_is_removed_when_the_body_raises(self) -> None:
        workspace = ArtifactWorkspace()
        captured: Path | None = None
        with pytest.raises(RuntimeError):
            with workspace.prepare(_response(_artifact("a.txt"))):
                captured = workspace.root
                raise RuntimeError("evaluator exploded")
        assert captured is not None and not captured.exists()

    def test_each_evaluation_gets_its_own_directory(self) -> None:
        first, second = ArtifactWorkspace(), ArtifactWorkspace()
        with first.prepare(_response()):
            with second.prepare(_response()):
                assert first.root != second.root


class TestPreparedResponse:
    """Evaluators must never see a path the submitter chose."""

    def test_paths_are_rewritten_to_sandbox_relative(self) -> None:
        workspace = ArtifactWorkspace()
        with workspace.prepare(_response(_artifact("src/main.py", "code"))) as prepared:
            assert prepared.response.artifacts[0].path == "src/main.py"
            assert not Path(prepared.response.artifacts[0].path).is_absolute()

    def test_rejected_artifact_keeps_content_but_nothing_is_written(self) -> None:
        """It is still readable by a content assertion; there is just no file."""
        workspace = ArtifactWorkspace(WorkspaceLimits(max_file_bytes=1))
        with workspace.prepare(_response(_artifact("big.txt", "toolong"))) as prepared:
            assert prepared.response.artifacts[0].content == "toolong"
            assert workspace.root is not None
            assert not (workspace.root / "big.txt").exists()

    def test_the_sandbox_is_offered_as_the_root_evaluators_may_address(self) -> None:
        """Without this the filesystem evaluator has nowhere legitimate to look."""
        workspace = ArtifactWorkspace()
        with workspace.prepare(_response(_artifact("out.txt", "x"))) as prepared:
            assert prepared.root == workspace.root
            assert prepared.root is not None
            assert (prepared.root / "out.txt").is_file()

    def test_the_offered_root_is_gone_once_the_block_exits(self) -> None:
        """The grant is for one evaluation; a root outliving it would be a leak."""
        workspace = ArtifactWorkspace()
        with workspace.prepare(_response(_artifact("out.txt", "x"))) as prepared:
            escaped = prepared.root
        assert escaped is not None
        assert not escaped.exists()

    def test_the_original_response_is_left_untouched(self) -> None:
        """Rewriting in place would corrupt the record of what the agent sent."""
        original = _response(_artifact("kept/name.txt", "x"))
        workspace = ArtifactWorkspace()
        with workspace.prepare(original):
            pass
        assert original.artifacts[0].path == "kept/name.txt"


class TestReporting:
    """A refusal nobody can see is indistinguishable from a bad score."""

    def test_report_lists_written_and_rejected(self) -> None:
        workspace = ArtifactWorkspace(WorkspaceLimits(max_file_bytes=4))
        with workspace.prepare(
            _response(_artifact("ok.txt", "fine"), _artifact("nope.txt", "far too big"))
        ):
            pass
        assert workspace.report.written == ["ok.txt"]
        assert workspace.report.rejected[0].path == "nope.txt"
        assert workspace.report.clean is False

    def test_clean_report_when_everything_was_accepted(self) -> None:
        workspace = ArtifactWorkspace()
        with workspace.prepare(_response(_artifact("ok.txt"))):
            pass
        assert workspace.report.clean is True

    def test_a_bad_artifact_does_not_stop_the_good_ones(self) -> None:
        """One poisoned artifact must not suppress the whole evaluation."""
        workspace = ArtifactWorkspace(WorkspaceLimits(max_file_bytes=8))
        with workspace.prepare(
            _response(_artifact("bad.txt", "x" * 20), _artifact("good.txt", "kept"))
        ):
            assert workspace.root is not None
            assert (workspace.root / "good.txt").read_text() == "kept"
