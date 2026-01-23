"""Tests for baseline models."""

import pytest

from atp.baseline.models import Baseline, ChangeType, TestBaseline


class TestTestBaselineModel:
    """Tests for TestBaseline model."""

    def test_creation(self) -> None:
        """Test creating a TestBaseline."""
        baseline = TestBaseline(
            test_id="test-1",
            test_name="Test One",
            mean_score=85.5,
            std=3.2,
            n_runs=5,
            ci_95=(82.0, 89.0),
            success_rate=0.9,
        )

        assert baseline.test_id == "test-1"
        assert baseline.test_name == "Test One"
        assert baseline.mean_score == 85.5
        assert baseline.std == 3.2
        assert baseline.n_runs == 5
        assert baseline.ci_95 == (82.0, 89.0)
        assert baseline.success_rate == 0.9

    def test_optional_fields(self) -> None:
        """Test optional fields in TestBaseline."""
        baseline = TestBaseline(
            test_id="test-1",
            test_name="Test One",
            mean_score=85.5,
            std=3.2,
            n_runs=5,
            ci_95=(82.0, 89.0),
            success_rate=1.0,
            mean_duration=1.5,
            mean_tokens=1000.0,
            mean_cost=0.001,
        )

        assert baseline.mean_duration == 1.5
        assert baseline.mean_tokens == 1000.0
        assert baseline.mean_cost == 0.001

    def test_to_dict(self) -> None:
        """Test converting TestBaseline to dict."""
        baseline = TestBaseline(
            test_id="test-1",
            test_name="Test One",
            mean_score=85.5555,
            std=3.2222,
            n_runs=5,
            ci_95=(82.0111, 89.0111),
            success_rate=0.95,
            mean_duration=1.5555,
        )

        result = baseline.to_dict()

        assert result["test_id"] == "test-1"
        assert result["test_name"] == "Test One"
        assert result["mean_score"] == 85.5555  # rounded to 4 decimals
        assert result["std"] == 3.2222
        assert result["n_runs"] == 5
        assert result["ci_95"] == [82.0111, 89.0111]
        assert result["success_rate"] == 0.95
        assert result["mean_duration"] == 1.5555

    def test_validation_score_range(self) -> None:
        """Test that mean_score must be 0-100."""
        with pytest.raises(ValueError):
            TestBaseline(
                test_id="test-1",
                test_name="Test One",
                mean_score=101.0,  # Invalid: > 100
                std=3.2,
                n_runs=5,
                ci_95=(82.0, 89.0),
                success_rate=1.0,
            )

    def test_validation_std_non_negative(self) -> None:
        """Test that std must be non-negative."""
        with pytest.raises(ValueError):
            TestBaseline(
                test_id="test-1",
                test_name="Test One",
                mean_score=85.0,
                std=-1.0,  # Invalid: negative
                n_runs=5,
                ci_95=(82.0, 89.0),
                success_rate=1.0,
            )


class TestBaselineModel:
    """Tests for Baseline model."""

    def test_creation(self) -> None:
        """Test creating a Baseline."""
        test_baseline = TestBaseline(
            test_id="test-1",
            test_name="Test One",
            mean_score=85.0,
            std=3.0,
            n_runs=5,
            ci_95=(82.0, 88.0),
            success_rate=1.0,
        )

        baseline = Baseline(
            suite_name="my-suite",
            agent_name="my-agent",
            runs_per_test=5,
            tests={"test-1": test_baseline},
        )

        assert baseline.suite_name == "my-suite"
        assert baseline.agent_name == "my-agent"
        assert baseline.runs_per_test == 5
        assert "test-1" in baseline.tests

    def test_to_dict(self) -> None:
        """Test converting Baseline to dict."""
        test_baseline = TestBaseline(
            test_id="test-1",
            test_name="Test One",
            mean_score=85.0,
            std=3.0,
            n_runs=5,
            ci_95=(82.0, 88.0),
            success_rate=1.0,
        )

        baseline = Baseline(
            suite_name="my-suite",
            agent_name="my-agent",
            runs_per_test=5,
            tests={"test-1": test_baseline},
        )

        result = baseline.to_dict()

        assert result["version"] == "1.0"
        assert result["suite_name"] == "my-suite"
        assert result["agent_name"] == "my-agent"
        assert result["runs_per_test"] == 5
        assert "test-1" in result["tests"]
        assert "created_at" in result

    def test_from_dict(self) -> None:
        """Test creating Baseline from dict."""
        data = {
            "version": "1.0",
            "created_at": "2024-01-15T10:30:00+00:00",
            "suite_name": "my-suite",
            "agent_name": "my-agent",
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
                }
            },
        }

        baseline = Baseline.from_dict(data)

        assert baseline.suite_name == "my-suite"
        assert baseline.agent_name == "my-agent"
        assert baseline.runs_per_test == 5
        assert "test-1" in baseline.tests
        assert baseline.tests["test-1"].mean_score == 85.0

    def test_from_dict_default_values(self) -> None:
        """Test from_dict with missing optional fields."""
        data = {
            "suite_name": "my-suite",
            "agent_name": "my-agent",
            "tests": {
                "test-1": {
                    "mean_score": 85.0,
                    "std": 3.0,
                    "n_runs": 5,
                    "ci_95": [82.0, 88.0],
                }
            },
        }

        baseline = Baseline.from_dict(data)

        assert baseline.version == "1.0"
        assert baseline.runs_per_test == 1
        assert baseline.tests["test-1"].success_rate == 1.0


class TestChangeTypeEnum:
    """Tests for ChangeType enum."""

    def test_values(self) -> None:
        """Test ChangeType enum values."""
        assert ChangeType.REGRESSION.value == "regression"
        assert ChangeType.IMPROVEMENT.value == "improvement"
        assert ChangeType.NO_CHANGE.value == "no_change"
        assert ChangeType.NEW_TEST.value == "new_test"
        assert ChangeType.REMOVED_TEST.value == "removed_test"
