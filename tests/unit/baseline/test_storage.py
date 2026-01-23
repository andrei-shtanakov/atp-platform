"""Tests for baseline storage module."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from atp.baseline.models import Baseline, TestBaseline
from atp.baseline.storage import load_baseline, save_baseline


class TestSaveBaseline:
    """Tests for save_baseline function."""

    def test_save_baseline(self, tmp_path: Path) -> None:
        """Test saving a baseline to a file."""
        baseline = Baseline(
            suite_name="test-suite",
            agent_name="test-agent",
            runs_per_test=5,
            tests={
                "test-1": TestBaseline(
                    test_id="test-1",
                    test_name="Test One",
                    mean_score=85.0,
                    std=3.0,
                    n_runs=5,
                    ci_95=(82.0, 88.0),
                    success_rate=1.0,
                ),
            },
        )

        output_path = tmp_path / "baseline.json"
        save_baseline(baseline, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["suite_name"] == "test-suite"
        assert data["agent_name"] == "test-agent"
        assert "test-1" in data["tests"]

    def test_save_baseline_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that save_baseline creates parent directories."""
        baseline = Baseline(
            suite_name="test-suite",
            agent_name="test-agent",
            runs_per_test=5,
            tests={},
        )

        output_path = tmp_path / "nested" / "dir" / "baseline.json"
        save_baseline(baseline, output_path)

        assert output_path.exists()


class TestLoadBaseline:
    """Tests for load_baseline function."""

    def test_load_baseline(self, tmp_path: Path) -> None:
        """Test loading a baseline from a file."""
        data = {
            "version": "1.0",
            "created_at": "2024-01-15T10:30:00+00:00",
            "suite_name": "test-suite",
            "agent_name": "test-agent",
            "runs_per_test": 5,
            "tests": {
                "test-1": {
                    "test_id": "test-1",
                    "test_name": "Test One",
                    "mean_score": 85.0,
                    "std": 3.0,
                    "n_runs": 5,
                    "ci_95": [82.0, 88.0],
                    "success_rate": 1.0,
                },
            },
        }

        input_path = tmp_path / "baseline.json"
        with open(input_path, "w") as f:
            json.dump(data, f)

        baseline = load_baseline(input_path)

        assert baseline.suite_name == "test-suite"
        assert baseline.agent_name == "test-agent"
        assert baseline.runs_per_test == 5
        assert "test-1" in baseline.tests
        assert baseline.tests["test-1"].mean_score == 85.0

    def test_load_baseline_file_not_found(self, tmp_path: Path) -> None:
        """Test loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_baseline(tmp_path / "nonexistent.json")

    def test_load_baseline_invalid_json(self, tmp_path: Path) -> None:
        """Test loading invalid JSON raises JSONDecodeError."""
        input_path = tmp_path / "invalid.json"
        with open(input_path, "w") as f:
            f.write("not valid json")

        with pytest.raises(json.JSONDecodeError):
            load_baseline(input_path)

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Test save then load preserves data."""
        original = Baseline(
            suite_name="test-suite",
            agent_name="test-agent",
            runs_per_test=5,
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            tests={
                "test-1": TestBaseline(
                    test_id="test-1",
                    test_name="Test One",
                    mean_score=85.5,
                    std=3.2,
                    n_runs=5,
                    ci_95=(82.0, 89.0),
                    success_rate=0.95,
                    mean_duration=1.5,
                    mean_tokens=1000.0,
                    mean_cost=0.001,
                ),
                "test-2": TestBaseline(
                    test_id="test-2",
                    test_name="Test Two",
                    mean_score=70.0,
                    std=5.0,
                    n_runs=5,
                    ci_95=(65.0, 75.0),
                    success_rate=0.8,
                ),
            },
        )

        path = tmp_path / "baseline.json"
        save_baseline(original, path)
        loaded = load_baseline(path)

        assert loaded.suite_name == original.suite_name
        assert loaded.agent_name == original.agent_name
        assert loaded.runs_per_test == original.runs_per_test
        assert len(loaded.tests) == len(original.tests)

        for test_id in original.tests:
            orig_test = original.tests[test_id]
            load_test = loaded.tests[test_id]
            assert load_test.mean_score == orig_test.mean_score
            assert load_test.std == orig_test.std
            assert load_test.n_runs == orig_test.n_runs
            assert load_test.success_rate == orig_test.success_rate
