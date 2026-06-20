"""Deterministic citation-grounding checker for corpus-backed JSON text."""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from atp.core.results import CaseVerdict
from atp.evaluators.json_path.resolver import InvalidPath, resolve

CITATION_GROUNDING_CHECKER_VERSION = "citation_grounding@1"


def citation_grounding_check(config: dict[str, Any], text: str | None) -> CaseVerdict:
    """Validate JSON-text citations against configured corpus files."""
    if text is None:
        return _malformed("no agent output")
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return _malformed("output is not valid JSON")

    schema = config.get("schema")
    if schema:
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as exc:
            return _malformed(f"schema violation: {exc.message}")
        except jsonschema.SchemaError as exc:
            return _malformed(f"invalid schema: {exc.message}")

    expected = config.get("expected")
    if not isinstance(expected, list) or not expected:
        return _malformed("no expected citations configured")

    results: list[dict[str, Any]] = []
    for expectation in expected:
        ok, reason, malformed = _check_expected(data, config, expectation)
        if malformed:
            return _malformed(reason)
        results.append({"expected": expectation, "ok": ok, "reason": reason})

    forbidden_results = [
        _check_forbidden(data, config, item) for item in config.get("forbidden", [])
    ]
    results.extend(forbidden_results)
    passed = all(result["ok"] for result in results)
    return CaseVerdict(
        critical_pass=passed,
        malformed=False,
        details={"results": results, "n": len(results)},
        grader_version=CITATION_GROUNDING_CHECKER_VERSION,
    )


def _check_expected(
    data: Any, config: dict[str, Any], expectation: dict[str, Any]
) -> tuple[bool, str, bool]:
    path = expectation.get("output_path")
    if not isinstance(path, str):
        return False, "output_path must be a string", False
    try:
        found, citation = resolve(data, path)
    except InvalidPath:
        return False, f"invalid output_path: {path}", False
    if not found:
        return False, f"output_path not found: {path}", False
    if not isinstance(citation, dict):
        return False, "citation node is not an object", True

    ok, reason, malformed = _validate_citation_shape(citation)
    if not ok:
        return False, reason, malformed

    source_path = expectation.get("source_path")
    if not isinstance(source_path, str):
        return False, "source_path must be a string", False
    if citation["path"] != source_path:
        return False, f"expected source {source_path}, got {citation['path']}", False
    if citation.get("page") != expectation.get("page"):
        return False, "citation page does not match expected page", False
    if citation["line_start"] != expectation.get("line_start") or citation[
        "line_end"
    ] != expectation.get("line_end"):
        return False, "citation line range does not match expected range", False

    return _validate_file_and_metadata(config, citation, expectation)


def _check_forbidden(
    data: Any, config: dict[str, Any], forbidden: dict[str, Any]
) -> dict[str, Any]:
    source_path = forbidden.get("source_path")
    if not isinstance(source_path, str):
        return {"forbidden": forbidden, "ok": False, "reason": "invalid source_path"}
    found_forbidden = _contains_source(data, source_path)
    if not found_forbidden:
        return {"forbidden": forbidden, "ok": True, "reason": "not cited"}
    file_info = _files(config).get(source_path, {})
    metadata = file_info.get("metadata", {}) if isinstance(file_info, dict) else {}
    for key, expected in forbidden.items():
        if key == "source_path":
            continue
        if metadata.get(key) != expected:
            return {
                "forbidden": forbidden,
                "ok": True,
                "reason": "metadata did not match forbidden rule",
            }
    return {
        "forbidden": forbidden,
        "ok": False,
        "reason": f"forbidden source cited: {source_path}",
    }


def _contains_source(node: Any, source_path: Any) -> bool:
    if isinstance(node, dict):
        if node.get("path") == source_path:
            return True
        return any(_contains_source(value, source_path) for value in node.values())
    if isinstance(node, list):
        return any(_contains_source(value, source_path) for value in node)
    return False


def _validate_citation_shape(citation: dict[str, Any]) -> tuple[bool, str, bool]:
    path = citation.get("path")
    if not isinstance(path, str) or not path:
        return False, "citation.path must be a non-empty string", True
    if citation.get("page") is not None:
        return False, "citation.page must be null for text corpus files", False
    line_start = citation.get("line_start")
    line_end = citation.get("line_end")
    if not isinstance(line_start, int) or not isinstance(line_end, int):
        return False, "citation line range must be integer", True
    if line_start < 1 or line_end < line_start:
        return False, "citation line range is invalid", False
    return True, "ok", False


def _validate_file_and_metadata(
    config: dict[str, Any], citation: dict[str, Any], expectation: dict[str, Any]
) -> tuple[bool, str, bool]:
    files = _files(config)
    source_path = citation["path"]
    file_info = files.get(source_path)
    if not isinstance(file_info, dict):
        return False, f"citation source is not in corpus files: {source_path}", False
    line_count = file_info.get("line_count")
    if isinstance(line_count, int) and citation["line_end"] > line_count:
        return False, "citation line range exceeds file length", False

    metadata = file_info.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    for key, expected in expectation.items():
        if key in {"output_path", "source_path", "page", "line_start", "line_end"}:
            continue
        if metadata.get(key) != expected:
            return False, f"metadata mismatch for {source_path}: {key}", False
    return True, "ok", False


def _files(config: dict[str, Any]) -> dict[str, Any]:
    files = config.get("files")
    return files if isinstance(files, dict) else {}


def _malformed(reason: str) -> CaseVerdict:
    return CaseVerdict(
        critical_pass=False,
        malformed=True,
        details={"reason": reason},
        grader_version=CITATION_GROUNDING_CHECKER_VERSION,
    )
