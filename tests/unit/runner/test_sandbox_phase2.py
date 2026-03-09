"""Tests for Phase 2 sandbox enhancements."""

import os
from pathlib import Path

from atp.runner.sandbox import (
    SandboxManager,
    _is_process_alive,
    detect_scope_conflicts,
)


class TestProcessAlive:
    def test_current_process_alive(self) -> None:
        assert _is_process_alive(os.getpid()) is True

    def test_dead_pid(self) -> None:
        # PID 99999999 is almost certainly not running
        assert _is_process_alive(99999999) is False

    def test_zero_pid(self) -> None:
        assert _is_process_alive(0) is False

    def test_negative_pid(self) -> None:
        assert _is_process_alive(-1) is False


class TestScopeConflicts:
    def test_no_conflicts(self) -> None:
        tests = [
            {"test_id": "a", "artifact_paths": ["out/a.txt"]},
            {"test_id": "b", "artifact_paths": ["out/b.txt"]},
        ]
        assert detect_scope_conflicts(tests) == []

    def test_detects_overlap(self) -> None:
        tests = [
            {"test_id": "a", "artifact_paths": ["shared.txt", "a.txt"]},
            {"test_id": "b", "artifact_paths": ["shared.txt", "b.txt"]},
        ]
        conflicts = detect_scope_conflicts(tests)
        assert len(conflicts) == 1
        assert conflicts[0] == ("a", "b", "shared.txt")

    def test_multiple_overlaps(self) -> None:
        tests = [
            {"test_id": "a", "artifact_paths": ["x.txt", "y.txt"]},
            {"test_id": "b", "artifact_paths": ["x.txt", "y.txt"]},
        ]
        conflicts = detect_scope_conflicts(tests)
        assert len(conflicts) == 2

    def test_empty_tests(self) -> None:
        assert detect_scope_conflicts([]) == []

    def test_no_artifact_paths(self) -> None:
        tests = [
            {"test_id": "a"},
            {"test_id": "b"},
        ]
        assert detect_scope_conflicts(tests) == []


class TestProcessTracking:
    def test_state_file_path(self) -> None:
        mgr = SandboxManager()
        path = mgr._state_file()
        assert path.name == "running-tests.json"
        assert ".atp" in str(path)

    def test_record_and_unrecord(self, tmp_path: Path) -> None:
        mgr = SandboxManager(base_dir=tmp_path)
        mgr._record_process("test-sandbox-1", "test-1")

        state = mgr._load_state()
        assert "test-sandbox-1" in state
        assert state["test-sandbox-1"]["pid"] == os.getpid()

        mgr._unrecord_process("test-sandbox-1")
        state = mgr._load_state()
        assert "test-sandbox-1" not in state
