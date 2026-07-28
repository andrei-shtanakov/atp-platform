"""Bounded artifact workspace for evaluating an untrusted submission.

The CLI writes response artifacts straight into the working directory, using
the path the agent supplied. That is fine on an operator's own machine and
unacceptable on a server: the path is attacker-controlled, so it is an
arbitrary file write, and the payload is unbounded.

This workspace is the server-side answer. Every evaluation gets its own
temporary directory; every artifact is checked before it is written; anything
that fails a check is *rejected and recorded*, never silently dropped and
never partially applied. Evaluators then see a response whose artifact paths
are sandbox-relative, so no name chosen by the agent is ever used to address
anything outside the sandbox.

Rejection is not an error: a submission full of bad paths should still be
scored on the assertions that do not need them, with the rejections visible.
An exception here would let one malformed artifact suppress an entire result.

`ArtifactFile.validate_artifact_path` already rejects empty paths, null bytes,
`..` traversal and absolute paths at parse time, so those checks here are
defence in depth — they still run because a response can be built without
validation (`model_construct`) and because a control that exists in one layer
only is one refactor away from not existing. What this workspace adds on top
is everything the protocol cannot know about: a per-evaluation directory,
symlink components, reserved names, and bounds on count and size.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from atp.evaluation.pipeline import PreparedResponse
from atp.protocol import ATPResponse
from atp.protocol.models import ArtifactFile

#: Names the workspace reserves for itself. An artifact may not take them.
RESERVED_NAMES = frozenset({".atp", ".atp-workspace"})


class RejectReason:
    """Why an artifact was not materialized. Stable strings; consumers match."""

    ABSOLUTE_PATH = "absolute_path"
    ESCAPES_WORKSPACE = "escapes_workspace"
    SYMLINK_IN_PATH = "symlink_in_path"
    RESERVED_NAME = "reserved_name"
    TOO_MANY_FILES = "too_many_files"
    FILE_TOO_LARGE = "file_too_large"
    TOTAL_TOO_LARGE = "total_size_exceeded"
    DUPLICATE_PATH = "duplicate_path"
    EMPTY_PATH = "empty_path"
    NOT_A_FILE = "not_a_file"
    PATH_COLLISION = "path_collision"
    WRITE_FAILED = "write_failed"


@dataclass(frozen=True)
class WorkspaceLimits:
    """Bounds on what one submission may materialize.

    Defaults are deliberately small: a benchmark submission is a handful of
    files, and a limit nobody notices is a limit that stops nothing.
    """

    max_files: int = 64
    max_file_bytes: int = 1_048_576  # 1 MiB
    max_total_bytes: int = 8_388_608  # 8 MiB


@dataclass(frozen=True)
class RejectedArtifact:
    """One artifact that was not written, and why."""

    path: str
    reason: str
    detail: str | None = None


@dataclass
class MaterializationReport:
    """What the workspace accepted and what it refused."""

    written: list[str] = field(default_factory=list)
    rejected: list[RejectedArtifact] = field(default_factory=list)
    total_bytes: int = 0

    @property
    def clean(self) -> bool:
        """True when every artifact was accepted."""
        return not self.rejected


def _has_symlink_component(root: Path, relative: Path) -> bool:
    """True if any existing component of the path *as given* is a symlink.

    Must be checked before resolution, not after: `resolve()` follows links,
    so a path through a link that points back into the workspace comes out the
    other side looking ordinary. A first draft checked the resolved path and
    happily wrote through `link/file.txt` where `link -> .` — the test named
    for that case is what caught it.

    A link pointing *outside* is separately caught by containment; this covers
    the ones that resolve back inside. Legitimate artifacts never need either.
    """
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


class ArtifactWorkspace:
    """A throwaway directory that materializes artifacts under strict rules.

    Use via :meth:`prepare`, which is shaped to be the pipeline's injected
    artifact context. The report survives the context so the caller can record
    what was refused alongside the evaluation outcome.
    """

    def __init__(self, limits: WorkspaceLimits | None = None) -> None:
        self.limits = limits or WorkspaceLimits()
        self.report = MaterializationReport()
        self.root: Path | None = None

    @contextlib.contextmanager
    def prepare(self, response: ATPResponse) -> Iterator[PreparedResponse]:
        """Materialize artifacts into a fresh temp dir; yield a rewritten response.

        The yielded response carries sandbox-relative paths, so an evaluator
        cannot be handed an agent-chosen absolute path by accident. The
        directory is removed on the way out, including when the body raises —
        an evaluator that crashes must not leave a submission's files on disk.

        The temp dir travels with the response as the root a filesystem
        evaluator may address. It is the *only* directory such an evaluator
        gets on this plane, which is what makes `filesystem` admissible here
        at all: it now inspects the submission instead of the server's disk.
        """
        # Fresh report per evaluation: a reused instance must not carry another
        # submission's rejections, nor spend its budget.
        self.report = MaterializationReport()
        root = Path(tempfile.mkdtemp(prefix="atp-eval-"))
        self.root = root
        try:
            prepared = self._materialize(root, response)
            yield PreparedResponse(prepared, root)
        finally:
            shutil.rmtree(root, ignore_errors=True)
            self.root = None

    def _materialize(self, root: Path, response: ATPResponse) -> ATPResponse:
        """Write what is allowed, record what is not, return a safe response."""
        accepted: dict[int, str] = {}
        seen: set[str] = set()

        for index, artifact in enumerate(response.artifacts):
            # Only file artifacts have a path; structured ones carry no
            # filesystem footprint and are passed through untouched.
            if not isinstance(artifact, ArtifactFile):
                continue
            raw_path, content = artifact.path, artifact.content
            if not raw_path or content is None:
                continue

            relative = self._check(root, raw_path, content, seen)
            if relative is None:
                continue

            target = root / relative
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            except OSError as exc:
                # Belt and braces. The collision checks above catch the cases
                # we know of; this catches the ones we do not, because the
                # module promises that a bad artifact is recorded, never
                # thrown -- otherwise one poisoned path kills a whole run.
                self.report.rejected.append(
                    RejectedArtifact(raw_path, RejectReason.WRITE_FAILED, str(exc))
                )
                continue

            seen.add(str(relative))
            accepted[index] = str(relative)
            self.report.written.append(str(relative))
            self.report.total_bytes += len(content.encode("utf-8"))

        return self._rewrite(response, accepted)

    def _check(
        self, root: Path, raw_path: str, content: str, seen: set[str]
    ) -> Path | None:
        """Return the safe relative path, or None after recording a rejection."""

        def reject(reason: str, detail: str | None = None) -> None:
            self.report.rejected.append(RejectedArtifact(raw_path, reason, detail))

        candidate = Path(raw_path)
        if not raw_path.strip():
            reject(RejectReason.EMPTY_PATH)
            return None
        if candidate.is_absolute():
            reject(RejectReason.ABSOLUTE_PATH)
            return None
        if any(part in RESERVED_NAMES for part in candidate.parts):
            reject(RejectReason.RESERVED_NAME)
            return None

        # Before resolution: resolution erases the evidence.
        if _has_symlink_component(root, candidate):
            reject(RejectReason.SYMLINK_IN_PATH, str(candidate))
            return None

        resolved = (root / candidate).resolve()
        root_resolved = root.resolve()
        if resolved != root_resolved and root_resolved not in resolved.parents:
            reject(RejectReason.ESCAPES_WORKSPACE, str(candidate))
            return None

        relative = resolved.relative_to(root_resolved)
        if relative == Path("."):
            # `.` and anything resolving to the workspace root name a
            # directory, not a file; writing it raised IsADirectoryError.
            reject(RejectReason.NOT_A_FILE, "path names the workspace root")
            return None
        if self._collides(root, relative):
            return None
        if str(relative) in seen:
            reject(RejectReason.DUPLICATE_PATH, str(relative))
            return None

        size = len(content.encode("utf-8"))
        if size > self.limits.max_file_bytes:
            reject(RejectReason.FILE_TOO_LARGE, f"{size} bytes")
            return None
        if len(self.report.written) >= self.limits.max_files:
            reject(RejectReason.TOO_MANY_FILES, f"limit {self.limits.max_files}")
            return None
        if self.report.total_bytes + size > self.limits.max_total_bytes:
            reject(RejectReason.TOTAL_TOO_LARGE, f"{size} bytes would exceed budget")
            return None

        return relative

    def _collides(self, root: Path, relative: Path) -> bool:
        """True after recording a rejection when the path fights the tree.

        Two artifacts can describe incompatible shapes -- `a` as a file and
        then `a/b.txt`, or the reverse. Both are attacker-controlled and both
        used to raise mid-materialization, aborting an evaluation that should
        merely have scored lower.
        """

        def reject(detail: str) -> None:
            self.report.rejected.append(
                RejectedArtifact(str(relative), RejectReason.PATH_COLLISION, detail)
            )

        target = root / relative
        if target.is_dir():
            reject("a directory already exists at this path")
            return True

        current = root
        for part in relative.parts[:-1]:
            current = current / part
            if current.exists() and not current.is_dir():
                reject(f"'{current.relative_to(root)}' already exists as a file")
                return True
        return False

    @staticmethod
    def _rewrite(response: ATPResponse, accepted: dict[int, str]) -> ATPResponse:
        """Copy the response with written artifacts' paths normalized.

        A rejected artifact keeps its declared path — the protocol has already
        constrained it to a relative, traversal-free string — but nothing was
        written there, so an evaluator that opens it simply finds nothing.
        Blanking the field is not an option: `path` is required by the model,
        and inventing a placeholder would only move the confusion.

        The original response is never mutated: it is the record of what the
        agent actually sent, and rewriting it in place would destroy that.
        """
        prepared = response.model_copy(deep=True)
        for index, artifact in enumerate(prepared.artifacts):
            if isinstance(artifact, ArtifactFile) and index in accepted:
                artifact.path = accepted[index]
        return prepared
