"""Unit tests for evaluator base classes and models."""

import pytest

from atp.evaluators.base import EvalCheck, EvalResult


class TestEvalCheck:
    """Tests for EvalCheck model."""

    def test_create_passed_check(self) -> None:
        """Test creating a passed check."""
        check = EvalCheck(
            name="test_check",
            passed=True,
            score=1.0,
            message="Check passed",
        )
        assert check.name == "test_check"
        assert check.passed is True
        assert check.score == 1.0
        assert check.message == "Check passed"
        assert check.details is None

    def test_create_failed_check(self) -> None:
        """Test creating a failed check."""
        check = EvalCheck(
            name="test_check",
            passed=False,
            score=0.0,
            message="Check failed",
            details={"reason": "missing artifact"},
        )
        assert check.passed is False
        assert check.score == 0.0
        assert check.details == {"reason": "missing artifact"}

    def test_partial_score(self) -> None:
        """Test check with partial score."""
        check = EvalCheck(
            name="test_check",
            passed=True,
            score=0.75,
        )
        assert check.score == 0.75

    def test_score_validation_min(self) -> None:
        """Test score minimum validation."""
        with pytest.raises(ValueError):
            EvalCheck(name="test", passed=True, score=-0.1)

    def test_score_validation_max(self) -> None:
        """Test score maximum validation."""
        with pytest.raises(ValueError):
            EvalCheck(name="test", passed=True, score=1.1)

    def test_name_required(self) -> None:
        """Test name is required and non-empty."""
        with pytest.raises(ValueError):
            EvalCheck(name="", passed=True, score=1.0)


class TestEvalResult:
    """Tests for EvalResult model."""

    def test_empty_result(self) -> None:
        """Test empty result properties."""
        result = EvalResult(evaluator="test")
        assert result.evaluator == "test"
        assert result.checks == []
        assert result.passed is True
        assert result.score == 1.0
        assert result.total_checks == 0
        assert result.passed_checks == 0
        assert result.failed_checks == 0

    def test_all_passed(self) -> None:
        """Test result with all passing checks."""
        result = EvalResult(
            evaluator="test",
            checks=[
                EvalCheck(name="check1", passed=True, score=1.0),
                EvalCheck(name="check2", passed=True, score=0.8),
            ],
        )
        assert result.passed is True
        assert result.score == 0.9
        assert result.total_checks == 2
        assert result.passed_checks == 2
        assert result.failed_checks == 0

    def test_some_failed(self) -> None:
        """Test result with mixed pass/fail checks."""
        result = EvalResult(
            evaluator="test",
            checks=[
                EvalCheck(name="check1", passed=True, score=1.0),
                EvalCheck(name="check2", passed=False, score=0.0),
            ],
        )
        assert result.passed is False
        assert result.score == 0.5
        assert result.total_checks == 2
        assert result.passed_checks == 1
        assert result.failed_checks == 1

    def test_all_failed(self) -> None:
        """Test result with all failing checks."""
        result = EvalResult(
            evaluator="test",
            checks=[
                EvalCheck(name="check1", passed=False, score=0.0),
                EvalCheck(name="check2", passed=False, score=0.0),
            ],
        )
        assert result.passed is False
        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.failed_checks == 2

    def test_add_check(self) -> None:
        """Test adding a check to result."""
        result = EvalResult(evaluator="test")
        check = EvalCheck(name="check1", passed=True, score=1.0)
        result.add_check(check)
        assert result.total_checks == 1
        assert result.checks[0] == check

    def test_merge_results(self) -> None:
        """Test merging two results."""
        result1 = EvalResult(
            evaluator="test1",
            checks=[EvalCheck(name="check1", passed=True, score=1.0)],
        )
        result2 = EvalResult(
            evaluator="test2",
            checks=[EvalCheck(name="check2", passed=False, score=0.0)],
        )
        merged = result1.merge(result2)
        assert merged.evaluator == "test1"
        assert merged.total_checks == 2
        assert merged.passed_checks == 1
        assert merged.failed_checks == 1

    def test_aggregate_results(self) -> None:
        """Test aggregating multiple results."""
        results = [
            EvalResult(
                evaluator="test1",
                checks=[EvalCheck(name="check1", passed=True, score=1.0)],
            ),
            EvalResult(
                evaluator="test2",
                checks=[
                    EvalCheck(name="check2", passed=True, score=0.8),
                    EvalCheck(name="check3", passed=False, score=0.0),
                ],
            ),
        ]
        aggregated = EvalResult.aggregate(results)
        assert aggregated.evaluator == "aggregate"
        assert aggregated.total_checks == 3
        assert aggregated.passed_checks == 2
        assert aggregated.failed_checks == 1

    def test_aggregate_empty_list(self) -> None:
        """Test aggregating empty list."""
        aggregated = EvalResult.aggregate([])
        assert aggregated.evaluator == "aggregate"
        assert aggregated.total_checks == 0
        assert aggregated.passed is True

    def test_score_calculation_precision(self) -> None:
        """Test score calculation with various values."""
        result = EvalResult(
            evaluator="test",
            checks=[
                EvalCheck(name="check1", passed=True, score=1.0),
                EvalCheck(name="check2", passed=True, score=0.5),
                EvalCheck(name="check3", passed=False, score=0.0),
            ],
        )
        assert result.score == pytest.approx(0.5)
