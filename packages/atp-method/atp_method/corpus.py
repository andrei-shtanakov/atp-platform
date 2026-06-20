"""Corpus resolution, verification, and per-run materialization."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from atp_method.schema import ArtifactCorpus, CorpusMetadata

TEXT_SUFFIXES = frozenset({".md", ".txt"})


@dataclass(frozen=True)
class ResolvedCorpusFile:
    """A selected source file under a corpus root."""

    relative_path: str
    path: Path


@dataclass(frozen=True)
class ResolvedCorpus:
    """Corpus paths after include/exclude expansion."""

    corpus_id: str
    root: Path
    files: tuple[ResolvedCorpusFile, ...]
    manifest_path: Path
    metadata_path: Path | None


@dataclass(frozen=True)
class VerifiedCorpusFile:
    """A selected file verified against the hash manifest."""

    relative_path: str
    path: Path
    sha256: str
    normalized_text: str
    lines: dict[int, str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CorpusVerificationResult:
    """A corpus verified by manifest and ready to materialize."""

    corpus_id: str
    root: Path
    files: tuple[VerifiedCorpusFile, ...]
    manifest_path: Path
    metadata_path: Path | None
    metadata: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class MaterializedCorpusFile:
    """A verified file copied into the run workspace."""

    relative_path: str
    path: Path
    sha256: str
    lines: dict[int, str]
    metadata: dict[str, Any]

    @property
    def line_count(self) -> int:
        """Return the number of normalized text lines."""
        return len(self.lines)


@dataclass(frozen=True)
class MaterializedCorpus:
    """A read-only corpus copy rooted in a run workspace."""

    corpus_id: str
    root: Path
    files: tuple[MaterializedCorpusFile, ...]
    metadata: dict[str, dict[str, Any]]


def normalize_lf(text: str) -> str:
    """Normalize text newlines to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _safe_manifest_path(path: str) -> str:
    if not path or "\x00" in path:
        raise ValueError(f"unsafe manifest path: {path!r}")
    normalized = path.replace("\\", "/")
    p = Path(normalized)
    if p.is_absolute() or normalized.startswith("~") or "//" in normalized:
        raise ValueError(f"unsafe manifest path: {path!r}")
    if any(part in ("", ".", "..") for part in normalized.split("/")):
        raise ValueError(f"unsafe manifest path: {path!r}")
    return normalized


def _line_index(text: str) -> dict[int, str]:
    return {i: line for i, line in enumerate(text.splitlines(), start=1)}


def _read_normalized_text(path: Path) -> str:
    try:
        return normalize_lf(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"corpus file is not UTF-8 text: {path}") from exc


class CorpusResolver:
    """Resolve corpus include/exclude patterns relative to a case file."""

    def resolve(self, case_path: Path, corpus: ArtifactCorpus) -> ResolvedCorpus:
        """Expand include/exclude patterns and return canonical sorted files."""
        case_dir = case_path.parent
        root = (case_dir / corpus.root).resolve()
        if not root.is_dir():
            raise ValueError(f"corpus root does not exist: {root}")

        selected: set[Path] = set()
        for pattern in corpus.include:
            for path in root.glob(pattern):
                if path.is_symlink():
                    raise ValueError(f"corpus file must not be a symlink: {path}")
                if path.is_file():
                    selected.add(path)
        for pattern in corpus.exclude:
            for path in root.glob(pattern):
                selected.discard(path)

        files: list[ResolvedCorpusFile] = []
        for path in selected:
            if path.is_symlink():
                raise ValueError(f"corpus file must not be a symlink: {path}")
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"corpus file escapes root: {path}")
            if resolved.suffix.lower() not in TEXT_SUFFIXES:
                raise ValueError(f"unsupported corpus file type: {path}")
            rel = resolved.relative_to(root).as_posix()
            files.append(ResolvedCorpusFile(relative_path=rel, path=resolved))

        files.sort(key=lambda f: f.relative_path)
        manifest_path = (root / corpus.digest.manifest_path).resolve()
        if not manifest_path.is_relative_to(root):
            raise ValueError("manifest path escapes corpus root")
        metadata_path = None
        if corpus.metadata_path:
            metadata_path = (root / corpus.metadata_path).resolve()
            if not metadata_path.is_relative_to(root):
                raise ValueError("metadata path escapes corpus root")
        return ResolvedCorpus(
            corpus_id=corpus.id,
            root=root,
            files=tuple(files),
            manifest_path=manifest_path,
            metadata_path=metadata_path,
        )


class CorpusIntegrityVerifier:
    """Verify selected corpus files against a SHA-256 manifest."""

    def verify(self, resolved: ResolvedCorpus) -> CorpusVerificationResult:
        """Hash LF-normalized content and compare exactly with manifest entries."""
        manifest = self._load_manifest(resolved.manifest_path)
        selected = {file.relative_path for file in resolved.files}
        manifest_paths = set(manifest)
        if selected != manifest_paths:
            raise ValueError(
                "selected corpus files do not match manifest paths: "
                f"selected={sorted(selected)} manifest={sorted(manifest_paths)}"
            )

        metadata = self._load_metadata(resolved.metadata_path)
        verified: list[VerifiedCorpusFile] = []
        for file in resolved.files:
            text = _read_normalized_text(file.path)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            expected = manifest[file.relative_path]
            if digest != expected:
                raise ValueError(
                    f"sha256 hash mismatch for {file.relative_path}: "
                    f"expected {expected}, got {digest}"
                )
            file_metadata = metadata.get(file.relative_path, {})
            verified.append(
                VerifiedCorpusFile(
                    relative_path=file.relative_path,
                    path=file.path,
                    sha256=digest,
                    normalized_text=text,
                    lines=_line_index(text),
                    metadata=dict(file_metadata),
                )
            )

        return CorpusVerificationResult(
            corpus_id=resolved.corpus_id,
            root=resolved.root,
            files=tuple(verified),
            manifest_path=resolved.manifest_path,
            metadata_path=resolved.metadata_path,
            metadata=metadata,
        )

    def _load_manifest(self, path: Path) -> dict[str, str]:
        if not path.is_file():
            raise ValueError(f"manifest file does not exist: {path}")
        entries: dict[str, str] = {}
        for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                digest, rel = line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"invalid manifest line {line_number}") from exc
            rel = _safe_manifest_path(rel.strip())
            if rel in entries:
                raise ValueError(f"duplicate manifest path: {rel}")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"invalid sha256 digest for {rel}")
            entries[rel] = digest
        return entries

    def _load_metadata(self, path: Path | None) -> dict[str, dict[str, Any]]:
        if path is None or not path.exists():
            return {}
        data = yaml.safe_load(path.read_text()) or {}
        metadata = CorpusMetadata.model_validate(data)
        return {
            path: item.model_dump(exclude_none=True)
            for path, item in metadata.files.items()
        }


class CorpusMaterializer:
    """Copy verified corpus files into an isolated run workspace."""

    def materialize(
        self, resolved: CorpusVerificationResult, workspace: Path
    ) -> MaterializedCorpus:
        """Materialize selected files under ``workspace/<corpus_id>``."""
        root = workspace / resolved.corpus_id
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

        files: list[MaterializedCorpusFile] = []
        for file in resolved.files:
            destination = root / file.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(file.normalized_text, encoding="utf-8")
            files.append(
                MaterializedCorpusFile(
                    relative_path=file.relative_path,
                    path=destination,
                    sha256=file.sha256,
                    lines=file.lines,
                    metadata=file.metadata,
                )
            )

        return MaterializedCorpus(
            corpus_id=resolved.corpus_id,
            root=root,
            files=tuple(files),
            metadata=resolved.metadata,
        )
