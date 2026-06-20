"""Loader tests for corpus-backed method cases."""

import shutil
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_CASE_PATH = (
    REPO_ROOT
    / "method"
    / "cases"
    / "req-extraction"
    / "case-req-extraction-fabricated-deadline-corpus-clean-001.yaml"
)
CORPUS_ASSETS_PATH = CORPUS_CASE_PATH.parent / "assets"


def _case_doc() -> dict:
    return {
        "id": "case-demo-001",
        "version": 1,
        "family": "demo",
        "status": "active",
        "suite_type": "probe",
        "capability": "calibration",
        "construction_axis": "information_conditions",
        "axis_level": "clean",
        "instruction": "Extract requirements.",
        "environment": {"tools": ["file_read"], "side_effects": "none"},
        "expected_failure_mode": "misses citation grounding",
        "grader": {
            "type": "programmatic",
            "checker": "citation_grounding",
            "critical_check": "ground citations",
            "scoring": "fail if critical fails",
            "config": {
                "artifact_name": "answer",
                "corpus_id": "req-corpus",
                "expected": [
                    {
                        "output_path": "$.requirements[0].citations.deadline",
                        "source_path": "policy.md",
                        "line_start": 2,
                        "line_end": 2,
                    }
                ],
            },
        },
        "output_contract": {
            "artifact_name": "answer",
            "schema": {"type": "object"},
        },
        "run_mode": "read_only_corpus",
        "artifact_corpus": {
            "id": "req-corpus",
            "root": "assets/req-corpus",
            "include": ["**/*.md"],
            "exclude": [],
            "digest": {
                "algorithm": "sha256",
                "normalization": "lf",
                "manifest_path": "manifest.sha256",
            },
        },
        "provenance": {"author": "test", "created": "2026-06-20"},
    }


def _copy_corpus_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "req-extraction"
    case_dir.mkdir()
    shutil.copy2(CORPUS_CASE_PATH, case_dir / CORPUS_CASE_PATH.name)
    shutil.copytree(CORPUS_ASSETS_PATH, case_dir / "assets")
    return case_dir / CORPUS_CASE_PATH.name


def test_loader_preserves_corpus_metadata_without_inlining_contents(
    tmp_path: Path,
) -> None:
    from atp_method.loader import load_case

    case_path = tmp_path / "case.yaml"
    case_path.write_text(yaml.safe_dump(_case_doc()))

    td = load_case(case_path)

    assert td.task.input_data["case_path"] == str(case_path)
    assert td.task.input_data["run_mode"] == "read_only_corpus"
    assert td.task.input_data["artifact_corpus"]["id"] == "req-corpus"
    assert td.task.input_data["request_preparer"] == "corpus"
    assert "policy.md" not in yaml.safe_dump(td.task.input_data)


def test_loader_loads_corpus_backed_req_extraction_fixture() -> None:
    from atp_method.loader import load_case

    td = load_case(CORPUS_CASE_PATH)

    assert td.id == "case-req-extraction-fabricated-deadline-corpus-clean-001"
    assert td.task.input_data["case_path"] == str(CORPUS_CASE_PATH)
    assert td.task.input_data["run_mode"] == "read_only_corpus"
    assert td.task.input_data["request_preparer"] == "corpus"
    assert td.task.input_data["artifact_corpus"] == {
        "id": "fabricated-deadline-clean-corpus",
        "root": "assets/fabricated-deadline-clean-corpus-001",
        "include": ["**/*.md"],
        "exclude": [],
        "digest": {
            "algorithm": "sha256",
            "manifest_path": "manifest.sha256",
            "normalization": "lf",
        },
        "metadata_path": "corpus.meta.yaml",
    }
    assert td.constraints.allowed_tools == ["file_read"]
    assert td.assertions[0].config["checker"] == "citation_grounding"
    assert td.assertions[0].config["expected"][0]["source_path"] == (
        "policy-current.md"
    )
    requirements_items = td.task.input_data["output_contract"]["schema"]["properties"][
        "requirements"
    ]["items"]
    citations_schema = requirements_items["properties"]["citations"]
    assert citations_schema["required"] == ["deadline"]
    assert citations_schema["properties"]["deadline"]["type"] == "object"
    assert citations_schema["properties"]["deadline"]["required"] == [
        "path",
        "page",
        "line_start",
        "line_end",
        "field",
    ]
    assert (
        "citations.deadline as a single citation object"
        in td.task.input_data["output_contract"]["format_instruction"]
    )
    assert "within 30 days of onboarding" not in yaml.safe_dump(td.task.input_data)


@pytest.mark.anyio
async def test_corpus_preparation_rejects_copied_fixture_with_corrupt_manifest(
    tmp_path: Path,
) -> None:
    from atp.protocol import ATPRequest, Task

    from atp_method.loader import load_case
    from atp_method.runtime import CorpusRunPreparer

    case_path = _copy_corpus_case(tmp_path)
    manifest = (
        case_path.parent
        / "assets"
        / "fabricated-deadline-clean-corpus-001"
        / "manifest.sha256"
    )
    lines = manifest.read_text().splitlines()
    lines = [
        f"{'0' * 64}  policy-current.md"
        if line.endswith("  policy-current.md")
        else line
        for line in lines
    ]
    manifest.write_text("\n".join(lines) + "\n")
    td = load_case(case_path)
    request = ATPRequest(
        task_id=td.id,
        task=Task(description=td.task.description, input_data=td.task.input_data),
    )

    with pytest.raises(ValueError, match="hash|sha256"):
        await CorpusRunPreparer(workspace=tmp_path / "workspace").prepare(td, request)
