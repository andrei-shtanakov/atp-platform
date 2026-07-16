"""Unit tests for openprose_receipts: canonical form, ledger reader, checker.

Contract: method/contract/openprose/receipt.md (openprose.receipt.v1).
"""

import json
from pathlib import Path
from typing import Any

import pytest

from atp.evaluators.openprose_receipts.canonical import canonical_json, content_hash
from atp.evaluators.openprose_receipts.reader import (  # noqa: F401
    LoadedLedger,
    VerifyResult,
    load_ledger,
    verify_run,
)


class TestCanonicalJson:
    def test_sorts_object_keys(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'

    def test_no_whitespace_nested_containers(self) -> None:
        value = {"outer": {"z": [1, 2], "a": None}, "flag": True}
        assert canonical_json(value) == b'{"flag":true,"outer":{"a":null,"z":[1,2]}}'

    def test_utf8_not_ascii_escaped(self) -> None:
        assert canonical_json({"k": "приём"}) == '{"k":"приём"}'.encode()

    def test_string_json_escapes_preserved(self) -> None:
        assert canonical_json('a"b\n') == b'"a\\"b\\n"'

    def test_rejects_float(self) -> None:
        with pytest.raises(ValueError):
            canonical_json({"tokens": 1.5})

    def test_rejects_nan_and_infinity(self) -> None:
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError):
                canonical_json(bad)

    def test_bool_is_not_an_int(self) -> None:
        assert canonical_json(True) == b"true"

    def test_rejects_non_string_keys(self) -> None:
        with pytest.raises(TypeError):
            canonical_json({1: "x"})


class TestContentHash:
    def test_excludes_content_hash_field_itself(self) -> None:
        receipt = {"v": "openprose.receipt.v1", "prev": None}
        h = content_hash(receipt)
        assert h == content_hash({**receipt, "content_hash": "sha256:bogus"})
        assert h.startswith("sha256:")
        assert len(h) == 71  # "sha256:" + 64 hex

    def test_unknown_fields_participate_in_hash(self) -> None:
        base = {"v": "openprose.receipt.v1", "prev": None}
        assert content_hash(base) != content_hash({**base, "future_field": 1})

    def test_hash_covers_prev(self) -> None:
        base = {"v": "openprose.receipt.v1", "prev": None}
        assert content_hash(base) != content_hash({**base, "prev": "sha256:aa"})


RUN_ID = "20260101-000000-testrn"


def _receipt(statement_id: str, prev: str | None, **overrides: Any) -> dict[str, Any]:
    """A minimal structurally-valid receipt, hashed last (mirrors the contract)."""
    body: dict[str, Any] = {
        "v": "openprose.receipt.v1",
        "run_id": RUN_ID,
        "statement_id": statement_id,
        "kind": "session",
        "agent": "worker",
        "input_fingerprints": {},
        "output_fingerprint": None,
        "status": "rendered",
        "surprise_cause": "self",
        "usage": {
            "basis": "estimated",
            "input_tokens": 100,
            "output_tokens": 10,
            "model": "haiku",
        },
        "error": None,
        "detail": None,
        "reused_from": None,
        "prev": prev,
        "hash_algorithm": "sha256",
    }
    body.update(overrides)
    body["content_hash"] = content_hash(body)
    return body


def _chain(n: int) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    prev: str | None = None
    for i in range(1, n + 1):
        r = _receipt(f"s{i:03d}", prev)
        receipts.append(r)
        prev = r["content_hash"]
    return receipts


def _manifest(
    receipts: list[dict[str, Any]],
    count: int | None = None,
    head: str | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "v": "openprose.run.v1",
        "run_id": RUN_ID,
        "program": "fixture.prose",
        "state_backend": "filesystem",
        "status": status,
        "receipt_count": count if count is not None else len(receipts),
        "ledger_head": head if head is not None else receipts[-1]["content_hash"],
    }


def _write_run(
    run_dir: Path,
    receipts: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
    trailing_garbage: str | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, sort_keys=True) for r in receipts]
    if trailing_garbage is not None:
        lines.append(trailing_garbage)
    (run_dir / "receipts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if manifest is not None:
        (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


class TestLoadLedger:
    def test_loads_valid_ledger(self, tmp_path: Path) -> None:
        run = _write_run(tmp_path / "r", _chain(3), None)
        loaded = load_ledger(run / "receipts.jsonl")
        assert len(loaded.receipts) == 3
        assert loaded.errors == [] and loaded.warnings == []

    def test_torn_final_line_is_warning_with_prefix(self, tmp_path: Path) -> None:
        run = _write_run(
            tmp_path / "r", _chain(2), None, trailing_garbage='{"v": "openprose.re'
        )
        loaded = load_ledger(run / "receipts.jsonl")
        assert len(loaded.receipts) == 2
        assert loaded.errors == []
        assert loaded.warnings[0].code == "torn_write_line"
        assert "torn write" in loaded.warnings[0].message

    def test_invalid_json_mid_ledger_is_error_and_stops(self, tmp_path: Path) -> None:
        chain = _chain(3)
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        lines = [
            json.dumps(chain[0], sort_keys=True),
            "NOT JSON",
            json.dumps(chain[2], sort_keys=True),
        ]
        (run_dir / "receipts.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        loaded = load_ledger(run_dir / "receipts.jsonl")
        # Loader stops at the parsed prefix: line 3 is never parsed.
        assert len(loaded.receipts) == 1
        assert loaded.errors[0].code == "invalid_json"
        assert loaded.errors[0].line_no == 2

    def test_non_object_line_is_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        (run_dir / "receipts.jsonl").write_text('[1,2]\n"x"\n', encoding="utf-8")
        loaded = load_ledger(run_dir / "receipts.jsonl")
        assert loaded.receipts == []
        assert loaded.errors[0].code == "invalid_json"

    def test_empty_file_loads_empty(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        (run_dir / "receipts.jsonl").write_text("", encoding="utf-8")
        loaded = load_ledger(run_dir / "receipts.jsonl")
        assert loaded.receipts == [] and loaded.errors == [] and loaded.warnings == []


class TestVerifyRun:
    def test_valid_run_is_ok(self, tmp_path: Path) -> None:
        receipts = _chain(3)
        run = _write_run(tmp_path / "r", receipts, _manifest(receipts))
        result = verify_run(run)
        assert result.ok is True
        assert result.receipt_count == 3
        assert result.errors == [] and result.warnings == []

    def test_empty_ledger_is_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        (run_dir / "receipts.jsonl").write_text("", encoding="utf-8")
        result = verify_run(run_dir)
        assert result.ok is False
        assert result.errors[0].code == "empty_ledger"

    def test_unknown_version_is_refused(self, tmp_path: Path) -> None:
        receipts = _chain(1)
        receipts[0]["v"] = "openprose.receipt.v99"
        run = _write_run(tmp_path / "r", receipts, None)
        result = verify_run(run)
        assert any(e.code == "unknown_version" for e in result.errors)

    def test_tampered_content_fails_hash(self, tmp_path: Path) -> None:
        receipts = _chain(2)
        receipts[1]["agent"] = "tampered"  # content changed, hash not recomputed
        run = _write_run(tmp_path / "r", receipts, _manifest(receipts))
        result = verify_run(run)
        assert result.ok is False
        assert any(
            e.code == "content_hash_mismatch" and "content_hash mismatch" in e.message
            for e in result.errors
        )

    def test_broken_prev_fails_chain(self, tmp_path: Path) -> None:
        receipts = _chain(2)
        receipts[1]["prev"] = "sha256:" + "0" * 64
        receipts[1]["content_hash"] = content_hash(receipts[1])  # rehash honestly
        run = _write_run(tmp_path / "r", receipts, _manifest(receipts))
        result = verify_run(run)
        assert result.ok is False
        assert any(
            e.code == "chain_break" and "prev broken" in e.message
            for e in result.errors
        )

    def test_float_tokens_rejected(self, tmp_path: Path) -> None:
        receipts = _chain(1)
        receipts[0]["usage"]["input_tokens"] = 1.5
        receipts[0]["content_hash"] = "sha256:" + "0" * 64  # unhashable anyway
        run = _write_run(tmp_path / "r", receipts, None)
        result = verify_run(run)
        assert any(e.code == "invalid_number" for e in result.errors)

    def test_missing_manifest_is_warning(self, tmp_path: Path) -> None:
        run = _write_run(tmp_path / "r", _chain(2), None)
        result = verify_run(run)
        assert result.ok is True
        assert result.warnings[0].code == "no_anchor"

    def test_receipt_count_mismatch_with_matching_head(self, tmp_path: Path) -> None:
        receipts = _chain(2)
        run = _write_run(tmp_path / "r", receipts, _manifest(receipts, count=5))
        result = verify_run(run)
        assert result.ok is False
        assert any(e.code == "receipt_count_mismatch" for e in result.errors)

    def test_torn_manifest_trailing_by_one_is_warning(self, tmp_path: Path) -> None:
        receipts = _chain(2)
        manifest = _manifest(
            receipts, count=1, head=receipts[0]["content_hash"], status="running"
        )
        run = _write_run(tmp_path / "r", receipts, manifest)
        result = verify_run(run)
        assert result.ok is True
        assert any(
            w.code == "torn_write_manifest" and "torn write" in w.message
            for w in result.warnings
        )
        assert result.receipt_count == 1  # the anchored prefix

    def test_manifest_trailing_by_two_is_error(self, tmp_path: Path) -> None:
        receipts = _chain(3)
        manifest = _manifest(receipts, count=1, head=receipts[0]["content_hash"])
        run = _write_run(tmp_path / "r", receipts, manifest)
        result = verify_run(run)
        assert result.ok is False
        assert any(
            e.code == "ledger_head_mismatch" and "ledger_head" in e.message
            for e in result.errors
        )

    def test_head_matching_nothing_is_error(self, tmp_path: Path) -> None:
        receipts = _chain(2)
        manifest = _manifest(receipts, head="sha256:" + "f" * 64)
        run = _write_run(tmp_path / "r", receipts, manifest)
        result = verify_run(run)
        assert result.ok is False
        assert any(e.code == "ledger_head_mismatch" for e in result.errors)

    def test_torn_line_with_anchored_prefix_is_ok(self, tmp_path: Path) -> None:
        receipts = _chain(2)
        run = _write_run(
            tmp_path / "r",
            receipts,
            _manifest(receipts),
            trailing_garbage='{"v": "openprose.rec',
        )
        result = verify_run(run)
        assert result.ok is True
        assert result.receipt_count == 2
        assert any(w.code == "torn_write_line" for w in result.warnings)
