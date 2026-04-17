"""Tests for pre-evaluation guardrails."""

from atp.evaluators.guardrails import (
    check_not_silently_failed,
    check_timeout_not_exceeded,
    check_within_budget,
    run_guardrails,
    should_skip_evaluation,
)
from atp.loader.models import Constraints
from atp.loader.models import TestDefinition as _TestDefinition
from atp.protocol import ATPResponse, Metrics, ResponseStatus
from atp.protocol.models import ArtifactFile


def _make_response(
    status: str = "completed",
    artifacts: list | None = None,
    cost_usd: float | None = None,
) -> ATPResponse:
    return ATPResponse(
        task_id="test-1",
        status=ResponseStatus(status),
        artifacts=artifacts or [],
        metrics=Metrics(cost_usd=cost_usd) if cost_usd is not None else None,
    )


def _make_test(
    timeout: int | None = 60,
    budget: float | None = None,
) -> _TestDefinition:
    return _TestDefinition(
        id="test-1",
        name="Test One",
        task={"description": "Do something"},
        constraints=Constraints(
            timeout_seconds=timeout,
            budget_usd=budget,
        ),
    )


class TestNotSilentlyFailed:
    def test_completed_passes(self) -> None:
        r = _make_response(status="completed")
        assert check_not_silently_failed(r).passed is True

    def test_failed_no_artifacts_fails(self) -> None:
        r = _make_response(status="failed")
        result = check_not_silently_failed(r)
        assert result.passed is False
        assert "empty" in result.reason
        assert result.name == "not_silently_failed"

    def test_failed_with_artifacts_passes(self) -> None:
        artifact = ArtifactFile(type="file", path="out.txt")
        r = _make_response(status="failed", artifacts=[artifact])
        assert check_not_silently_failed(r).passed is True

    def test_completed_with_empty_artifacts_still_passes(self) -> None:
        """Completed but empty-artifacts is a valid shape (e.g. event-scored
        tests); guardrail must not short-circuit it."""
        r = _make_response(status="completed", artifacts=[])
        assert check_not_silently_failed(r).passed is True


class TestTimeoutNotExceeded:
    def test_completed_passes(self) -> None:
        t = _make_test(timeout=60)
        r = _make_response(status="completed")
        assert check_timeout_not_exceeded(t, r).passed is True

    def test_timeout_fails(self) -> None:
        t = _make_test(timeout=60)
        r = _make_response(status="timeout")
        result = check_timeout_not_exceeded(t, r)
        assert result.passed is False
        assert "timed out" in result.reason


class TestWithinBudget:
    def test_no_budget_passes(self) -> None:
        t = _make_test(budget=None)
        r = _make_response(cost_usd=100.0)
        assert check_within_budget(t, r).passed is True

    def test_within_budget_passes(self) -> None:
        t = _make_test(budget=1.0)
        r = _make_response(cost_usd=0.5)
        assert check_within_budget(t, r).passed is True

    def test_over_budget_fails(self) -> None:
        t = _make_test(budget=1.0)
        r = _make_response(cost_usd=1.5)
        result = check_within_budget(t, r)
        assert result.passed is False
        assert "exceeded" in result.reason

    def test_no_metrics_passes(self) -> None:
        t = _make_test(budget=1.0)
        r = _make_response()
        assert check_within_budget(t, r).passed is True


class TestRunGuardrailsIntegration:
    def test_all_pass(self) -> None:
        t = _make_test()
        r = _make_response()
        checks = run_guardrails(t, r)
        assert all(c.passed for c in checks)
        assert should_skip_evaluation(checks) is None

    def test_skip_on_failure(self) -> None:
        t = _make_test()
        r = _make_response(status="timeout")
        checks = run_guardrails(t, r)
        reason = should_skip_evaluation(checks)
        assert reason is not None
        assert "timed out" in reason
