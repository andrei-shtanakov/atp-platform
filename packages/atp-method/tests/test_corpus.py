"""Tests for corpus resolution, verification, and materialization."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _artifact_corpus(root: str = "assets/corpus"):
    from atp_method.schema import ArtifactCorpus

    return ArtifactCorpus.model_validate(
        {
            "id": "req-corpus",
            "root": root,
            "include": ["**/*.md", "**/*.txt"],
            "exclude": ["README.md"],
            "digest": {
                "algorithm": "sha256",
                "normalization": "lf",
                "manifest_path": "manifest.sha256",
            },
            "metadata_path": "corpus.meta.yaml",
        }
    )


def _sha256_lf(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _write_corpus(case_dir: Path, manifest_lines: list[str] | None = None) -> Path:
    root = case_dir / "assets" / "corpus"
    (root / "archive").mkdir(parents=True)
    (root / "policy-current.md").write_text("Title\r\nDeadline line\r\n")
    (root / "vendor-addendum.txt").write_text("Vendor must attest.\n")
    (root / "archive" / "policy-2023.md").write_text("Obsolete deadline.\n")
    (root / "README.md").write_text("excluded\n")
    (root / "corpus.meta.yaml").write_text(
        "files:\n"
        "  policy-current.md:\n"
        "    role: source\n"
        "    status: current\n"
        "  vendor-addendum.txt:\n"
        "    role: source\n"
        "    status: current\n"
        "  archive/policy-2023.md:\n"
        "    role: distractor\n"
        "    status: obsolete\n"
    )
    lines = manifest_lines or [
        f"{_sha256_lf('Title\nDeadline line\n')}  policy-current.md",
        f"{_sha256_lf('Vendor must attest.\n')}  vendor-addendum.txt",
        f"{_sha256_lf('Obsolete deadline.\n')}  archive/policy-2023.md",
    ]
    (root / "manifest.sha256").write_text("\n".join(lines) + "\n")
    return root


def _verified_corpus(corpus_id: str):
    from atp_method.corpus import CorpusVerificationResult, VerifiedCorpusFile

    return CorpusVerificationResult(
        corpus_id=corpus_id,
        root=Path("/unused/source"),
        files=(
            VerifiedCorpusFile(
                relative_path="policy-current.md",
                path=Path("/unused/source/policy-current.md"),
                sha256=_sha256_lf("Title\nDeadline line\n"),
                normalized_text="Title\nDeadline line\n",
                lines={1: "Title", 2: "Deadline line"},
                metadata={"role": "source"},
            ),
        ),
        manifest_path=Path("/unused/source/manifest.sha256"),
        metadata_path=None,
        metadata={"policy-current.md": {"role": "source"}},
    )


def test_resolver_selects_canonical_sorted_included_files(tmp_path: Path) -> None:
    from atp_method.corpus import CorpusResolver

    case_path = tmp_path / "cases" / "case.yaml"
    case_path.parent.mkdir()
    _write_corpus(case_path.parent)

    resolved = CorpusResolver().resolve(case_path, _artifact_corpus())

    assert [f.relative_path for f in resolved.files] == [
        "archive/policy-2023.md",
        "policy-current.md",
        "vendor-addendum.txt",
    ]
    assert resolved.root == case_path.parent / "assets" / "corpus"
    assert resolved.manifest_path == resolved.root / "manifest.sha256"
    assert resolved.metadata_path == resolved.root / "corpus.meta.yaml"


def test_resolver_rejects_symlink_selected_file(tmp_path: Path) -> None:
    from atp_method.corpus import CorpusResolver

    case_path = tmp_path / "case.yaml"
    root = _write_corpus(tmp_path)
    (root / "linked.md").symlink_to(tmp_path / "outside.md")

    with pytest.raises(ValueError, match="symlink"):
        CorpusResolver().resolve(case_path, _artifact_corpus())


def test_verifier_requires_manifest_paths_to_match_selected_set(tmp_path: Path) -> None:
    from atp_method.corpus import CorpusIntegrityVerifier, CorpusResolver

    case_path = tmp_path / "case.yaml"
    _write_corpus(
        tmp_path,
        manifest_lines=[
            f"{_sha256_lf('Title\nDeadline line\n')}  policy-current.md",
        ],
    )
    resolved = CorpusResolver().resolve(case_path, _artifact_corpus())

    with pytest.raises(ValueError, match="manifest|selected"):
        CorpusIntegrityVerifier().verify(resolved)


def test_verifier_rejects_duplicate_manifest_paths(tmp_path: Path) -> None:
    from atp_method.corpus import CorpusIntegrityVerifier, CorpusResolver

    case_path = tmp_path / "case.yaml"
    digest = _sha256_lf("Title\nDeadline line\n")
    _write_corpus(
        tmp_path,
        manifest_lines=[
            f"{digest}  policy-current.md",
            f"{digest}  policy-current.md",
            f"{_sha256_lf('Vendor must attest.\n')}  vendor-addendum.txt",
            f"{_sha256_lf('Obsolete deadline.\n')}  archive/policy-2023.md",
        ],
    )
    resolved = CorpusResolver().resolve(case_path, _artifact_corpus())

    with pytest.raises(ValueError, match="duplicate"):
        CorpusIntegrityVerifier().verify(resolved)


def test_verifier_hashes_lf_normalized_content_and_builds_line_index(
    tmp_path: Path,
) -> None:
    from atp_method.corpus import CorpusIntegrityVerifier, CorpusResolver

    case_path = tmp_path / "case.yaml"
    _write_corpus(tmp_path)
    resolved = CorpusResolver().resolve(case_path, _artifact_corpus())

    verified = CorpusIntegrityVerifier().verify(resolved)

    current = next(f for f in verified.files if f.relative_path == "policy-current.md")
    assert current.sha256 == _sha256_lf("Title\nDeadline line\n")
    assert current.normalized_text == "Title\nDeadline line\n"
    assert current.lines[1] == "Title"
    assert current.lines[2] == "Deadline line"


def test_verifier_rejects_hash_mismatch(tmp_path: Path) -> None:
    from atp_method.corpus import CorpusIntegrityVerifier, CorpusResolver

    case_path = tmp_path / "case.yaml"
    _write_corpus(
        tmp_path,
        manifest_lines=[
            f"{'0' * 64}  policy-current.md",
            f"{_sha256_lf('Vendor must attest.\n')}  vendor-addendum.txt",
            f"{_sha256_lf('Obsolete deadline.\n')}  archive/policy-2023.md",
        ],
    )
    resolved = CorpusResolver().resolve(case_path, _artifact_corpus())

    with pytest.raises(ValueError, match="hash|sha256"):
        CorpusIntegrityVerifier().verify(resolved)


def test_materializer_copies_verified_files_preserving_relative_paths(
    tmp_path: Path,
) -> None:
    from atp_method.corpus import (
        CorpusIntegrityVerifier,
        CorpusMaterializer,
        CorpusResolver,
    )

    case_path = tmp_path / "case.yaml"
    _write_corpus(tmp_path)
    resolved = CorpusResolver().resolve(case_path, _artifact_corpus())
    verified = CorpusIntegrityVerifier().verify(resolved)

    materialized = CorpusMaterializer().materialize(verified, tmp_path / "workspace")

    assert materialized.corpus_id == "req-corpus"
    assert [f.relative_path for f in materialized.files] == [
        "archive/policy-2023.md",
        "policy-current.md",
        "vendor-addendum.txt",
    ]
    assert (materialized.root / "policy-current.md").read_text() == (
        "Title\nDeadline line\n"
    )
    assert (materialized.root / "archive" / "policy-2023.md").is_file()


def test_materializer_accepts_safe_single_corpus_id(tmp_path: Path) -> None:
    from atp_method.corpus import CorpusMaterializer

    workspace = tmp_path / "workspace"
    verified = _verified_corpus("req-corpus")

    materialized = CorpusMaterializer().materialize(verified, workspace)

    assert materialized.root == workspace / "req-corpus"
    assert (workspace / "req-corpus" / "policy-current.md").read_text() == (
        "Title\nDeadline line\n"
    )


@pytest.mark.parametrize("corpus_id", ["../outside", "nested/corpus"])
def test_materializer_rejects_unsafe_relative_corpus_id_without_touching_outside_paths(
    tmp_path: Path, corpus_id: str
) -> None:
    from atp_method.corpus import CorpusMaterializer

    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep\n")
    verified = _verified_corpus(corpus_id)

    with pytest.raises(ValueError, match="corpus_id"):
        CorpusMaterializer().materialize(verified, workspace)

    assert sentinel.read_text() == "keep\n"


def test_materializer_rejects_absolute_corpus_id_without_touching_outside_paths(
    tmp_path: Path,
) -> None:
    from atp_method.corpus import CorpusMaterializer

    workspace = tmp_path / "workspace"
    outside = tmp_path / "absolute-target"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep\n")
    verified = _verified_corpus(str(outside))

    with pytest.raises(ValueError, match="corpus_id"):
        CorpusMaterializer().materialize(verified, workspace)

    assert sentinel.read_text() == "keep\n"
