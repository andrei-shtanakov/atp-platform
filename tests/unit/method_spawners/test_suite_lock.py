"""Tests for the golden-suite lock (ADR-ECO-003a D3).

The lock freezes the exact case set + byte content of a case dir so a re-sweep
or future model A/B runs on a byte-identical suite. run_pipe_check verifies the
live dir against the lock and fails loud on drift.
"""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_pipe_check",
    Path(__file__).resolve().parents[3] / "method" / "run_pipe_check.py",
)
assert _SPEC and _SPEC.loader
rpc = importlib.util.module_from_spec(_SPEC)
sys.modules["run_pipe_check"] = rpc
_SPEC.loader.exec_module(rpc)


def _case(case_dir: Path, cid: str, version: int = 1, body: str = "x") -> None:
    (case_dir / f"{cid}.yaml").write_text(
        f"id: {cid}\nversion: {version}\ninstruction: {body}\n", encoding="utf-8"
    )


def _mk_suite(tmp_path: Path) -> Path:
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    _case(case_dir, "case-a-001")
    _case(case_dir, "case-b-001")
    return case_dir


def test_write_then_verify_roundtrips_clean(tmp_path: Path) -> None:
    case_dir = _mk_suite(tmp_path)
    path = rpc._write_suite_lock(case_dir, "code-review")
    assert path.name == rpc.SUITE_LOCK_NAME
    lock = rpc._load_lock(case_dir)
    assert lock is not None
    assert lock["benchmark_id"] == "code-review"
    assert lock["case_count"] == 2
    assert lock["suite_hash"].startswith("sha256:")
    assert rpc._diff_against_lock(case_dir, lock) == []


def test_absent_lock_loads_as_none(tmp_path: Path) -> None:
    case_dir = _mk_suite(tmp_path)
    assert rpc._load_lock(case_dir) is None


def test_changed_case_content_is_drift(tmp_path: Path) -> None:
    case_dir = _mk_suite(tmp_path)
    rpc._write_suite_lock(case_dir, "code-review")
    lock = rpc._load_lock(case_dir)
    assert lock is not None
    # Edit the body — same id, different bytes.
    _case(case_dir, "case-a-001", body="MUTATED")
    drift = rpc._diff_against_lock(case_dir, lock)
    assert any("changed case content" in d and "case-a-001" in d for d in drift)


def test_added_case_is_drift(tmp_path: Path) -> None:
    case_dir = _mk_suite(tmp_path)
    rpc._write_suite_lock(case_dir, "code-review")
    lock = rpc._load_lock(case_dir)
    _case(case_dir, "case-c-001")
    drift = rpc._diff_against_lock(case_dir, lock)
    assert any("new case" in d and "case-c-001" in d for d in drift)


def test_removed_case_is_drift(tmp_path: Path) -> None:
    case_dir = _mk_suite(tmp_path)
    rpc._write_suite_lock(case_dir, "code-review")
    lock = rpc._load_lock(case_dir)
    (case_dir / "case-b-001.yaml").unlink()
    drift = rpc._diff_against_lock(case_dir, lock)
    assert any("missing case" in d and "case-b-001" in d for d in drift)


def test_fingerprint_is_order_independent(tmp_path: Path) -> None:
    case_dir = _mk_suite(tmp_path)
    entries = rpc._case_entries(case_dir)
    assert rpc._suite_fingerprint(entries) == rpc._suite_fingerprint(
        list(reversed(entries))
    )
