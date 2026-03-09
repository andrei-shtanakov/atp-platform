"""Tests for trace retention policy."""

import json
import time
from pathlib import Path

from atp.tracing.storage import FileTraceStorage


def _write_trace(base: Path, trace_id: str, age_days: int = 0) -> Path:
    """Write a minimal trace JSON file with a specific age."""
    path = base / f"{trace_id}.json"
    data = {
        "trace_id": trace_id,
        "test_id": "test-1",
        "test_name": "Test",
        "status": "completed",
        "started_at": "2025-01-01T00:00:00Z",
        "steps": [],
        "metadata": {
            "adapter_type": "http",
            "total_tokens": 0,
        },
        "total_events": 0,
    }
    path.write_text(json.dumps(data))
    if age_days > 0:
        old_time = time.time() - (age_days * 86400)
        import os

        os.utime(path, (old_time, old_time))
    return path


class TestRetentionPolicy:
    def test_delete_old_traces(self, tmp_path: Path) -> None:
        storage = FileTraceStorage(base_dir=tmp_path)
        _write_trace(tmp_path, "old-1", age_days=60)
        _write_trace(tmp_path, "old-2", age_days=45)
        _write_trace(tmp_path, "recent", age_days=5)

        deleted = storage.apply_retention(max_age_days=30)
        assert deleted == 2
        assert not (tmp_path / "old-1.json").exists()
        assert not (tmp_path / "old-2.json").exists()
        assert (tmp_path / "recent.json").exists()

    def test_no_deletions_if_all_recent(self, tmp_path: Path) -> None:
        storage = FileTraceStorage(base_dir=tmp_path)
        _write_trace(tmp_path, "t1", age_days=1)
        _write_trace(tmp_path, "t2", age_days=10)

        deleted = storage.apply_retention(max_age_days=30)
        assert deleted == 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        storage = FileTraceStorage(base_dir=tmp_path)
        deleted = storage.apply_retention(max_age_days=30)
        assert deleted == 0

    def test_count(self, tmp_path: Path) -> None:
        storage = FileTraceStorage(base_dir=tmp_path)
        assert storage.count() == 0
        _write_trace(tmp_path, "t1")
        _write_trace(tmp_path, "t2")
        assert storage.count() == 2

    def test_custom_retention_period(self, tmp_path: Path) -> None:
        storage = FileTraceStorage(base_dir=tmp_path)
        _write_trace(tmp_path, "t1", age_days=8)
        _write_trace(tmp_path, "t2", age_days=3)

        deleted = storage.apply_retention(max_age_days=7)
        assert deleted == 1
        assert not (tmp_path / "t1.json").exists()
        assert (tmp_path / "t2.json").exists()
