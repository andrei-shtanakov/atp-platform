"""Tests for runner models."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from atp.loader.models import TaskDefinition, TestDefinition
from atp.protocol import ATPResponse, Metrics, ResponseStatus
from atp.runner.models import (
    ProgressEvent,
    ProgressEventType,
    RunResult,
    SandboxConfig,
    SuiteResult,
    TestResult,
)


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_default_values(self) -> None:
        """SandboxConfig has sensible defaults."""
        config = SandboxConfig()
        assert config.memory_limit == "2Gi"
        assert config.cpu_limit == "2"
        assert config.network_mode == "none"
        assert config.allowed_hosts == []
        assert config.workspace_path == Path("/workspace")
        assert config.readonly_mounts == []
        assert config.hard_timeout_seconds == 600
        assert config.enabled is False

    def test_custom_values(self) -> None:
        """SandboxConfig with custom values."""
        config = SandboxConfig(
            memory_limit="4Gi",
            cpu_limit="4",
            network_mode="custom",
            allowed_hosts=["api.example.com"],
            workspace_path=Path("/work"),
            hard_timeout_seconds=1200,
            enabled=True,
        )
        assert config.memory_limit == "4Gi"
        assert config.cpu_limit == "4"
        assert config.network_mode == "custom"
        assert config.allowed_hosts == ["api.example.com"]
        assert config.workspace_path == Path("/work")
        assert config.hard_timeout_seconds == 1200
        assert config.enabled is True

    def test_hard_timeout_must_be_positive(self) -> None:
        """hard_timeout_seconds must be positive."""
        with pytest.raises(ValueError):
            SandboxConfig(hard_timeout_seconds=0)
        with pytest.raises(ValueError):
            SandboxConfig(hard_timeout_seconds=-1)


class TestProgressEvent:
    """Tests for ProgressEvent."""

    def test_minimal_event(self) -> None:
        """ProgressEvent with minimal fields."""
        event = ProgressEvent(event_type=ProgressEventType.TEST_STARTED)
        assert event.event_type == ProgressEventType.TEST_STARTED
        assert event.timestamp is not None
        assert event.suite_name is None
        assert event.test_id is None

    def test_test_started_event(self) -> None:
        """ProgressEvent for test started."""
        event = ProgressEvent(
            event_type=ProgressEventType.TEST_STARTED,
            test_id="test-001",
            test_name="Sample Test",
            total_runs=3,
        )
        assert event.event_type == ProgressEventType.TEST_STARTED
        assert event.test_id == "test-001"
        assert event.test_name == "Sample Test"
        assert event.total_runs == 3

    def test_test_completed_event(self) -> None:
        """ProgressEvent for test completed."""
        event = ProgressEvent(
            event_type=ProgressEventType.TEST_COMPLETED,
            test_id="test-001",
            success=True,
            details={"duration_seconds": 1.5},
        )
        assert event.event_type == ProgressEventType.TEST_COMPLETED
        assert event.success is True
        assert event.details["duration_seconds"] == 1.5

    def test_suite_events(self) -> None:
        """ProgressEvent for suite start/complete."""
        start = ProgressEvent(
            event_type=ProgressEventType.SUITE_STARTED,
            suite_name="regression",
            total_tests=5,
        )
        assert start.suite_name == "regression"
        assert start.total_tests == 5

        complete = ProgressEvent(
            event_type=ProgressEventType.SUITE_COMPLETED,
            suite_name="regression",
            completed_tests=5,
            success=True,
        )
        assert complete.completed_tests == 5


class TestRunResult:
    """Tests for RunResult."""

    @pytest.fixture
    def success_response(self) -> ATPResponse:
        """Create a successful response."""
        return ATPResponse(
            task_id="task-001",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(total_tokens=1000, wall_time_seconds=1.5),
        )

    @pytest.fixture
    def failed_response(self) -> ATPResponse:
        """Create a failed response."""
        return ATPResponse(
            task_id="task-001",
            status=ResponseStatus.FAILED,
            error="Agent error",
        )

    @pytest.fixture
    def timeout_response(self) -> ATPResponse:
        """Create a timeout response."""
        return ATPResponse(
            task_id="task-001",
            status=ResponseStatus.TIMEOUT,
            error="Timeout",
        )

    def test_successful_run(self, success_response: ATPResponse) -> None:
        """RunResult for successful run."""
        result = RunResult(
            test_id="test-001",
            run_number=1,
            response=success_response,
        )
        assert result.test_id == "test-001"
        assert result.run_number == 1
        assert result.success is True
        assert result.events == []
        assert result.error is None

    def test_failed_run(self, failed_response: ATPResponse) -> None:
        """RunResult for failed run."""
        result = RunResult(
            test_id="test-001",
            run_number=1,
            response=failed_response,
            error="Agent crashed",
        )
        assert result.success is False
        assert result.error == "Agent crashed"

    def test_duration_calculation(self, success_response: ATPResponse) -> None:
        """RunResult calculates duration correctly."""
        start = datetime.now()
        end = start + timedelta(seconds=2.5)
        result = RunResult(
            test_id="test-001",
            run_number=1,
            response=success_response,
            start_time=start,
            end_time=end,
        )
        assert result.duration_seconds is not None
        assert abs(result.duration_seconds - 2.5) < 0.01

    def test_duration_none_without_end_time(
        self, success_response: ATPResponse
    ) -> None:
        """RunResult duration is None without end_time."""
        result = RunResult(
            test_id="test-001",
            run_number=1,
            response=success_response,
        )
        result.end_time = None
        assert result.duration_seconds is None


class TestTestResult:
    """Tests for TestResult."""

    @pytest.fixture
    def test_def(self) -> TestDefinition:
        """Create a test definition."""
        return TestDefinition(
            id="test-001",
            name="Sample Test",
            task=TaskDefinition(description="Do something"),
        )

    @pytest.fixture
    def success_run(self) -> RunResult:
        """Create a successful run result."""
        return RunResult(
            test_id="test-001",
            run_number=1,
            response=ATPResponse(
                task_id="task-001",
                status=ResponseStatus.COMPLETED,
            ),
        )

    @pytest.fixture
    def failed_run(self) -> RunResult:
        """Create a failed run result."""
        return RunResult(
            test_id="test-001",
            run_number=1,
            response=ATPResponse(
                task_id="task-001",
                status=ResponseStatus.FAILED,
            ),
        )

    @pytest.fixture
    def timeout_run(self) -> RunResult:
        """Create a timeout run result."""
        return RunResult(
            test_id="test-001",
            run_number=1,
            response=ATPResponse(
                task_id="task-001",
                status=ResponseStatus.TIMEOUT,
            ),
        )

    def test_empty_result(self, test_def: TestDefinition) -> None:
        """TestResult with no runs."""
        result = TestResult(test=test_def)
        assert result.success is False
        assert result.status == ResponseStatus.FAILED
        assert result.total_runs == 0
        assert result.successful_runs == 0

    def test_all_successful(
        self, test_def: TestDefinition, success_run: RunResult
    ) -> None:
        """TestResult with all successful runs."""
        success_run2 = RunResult(
            test_id="test-001",
            run_number=2,
            response=ATPResponse(
                task_id="task-002",
                status=ResponseStatus.COMPLETED,
            ),
        )
        result = TestResult(test=test_def, runs=[success_run, success_run2])
        assert result.success is True
        assert result.status == ResponseStatus.COMPLETED
        assert result.total_runs == 2
        assert result.successful_runs == 2

    def test_some_failed(
        self,
        test_def: TestDefinition,
        success_run: RunResult,
        failed_run: RunResult,
    ) -> None:
        """TestResult with mixed results."""
        failed_run.run_number = 2
        result = TestResult(test=test_def, runs=[success_run, failed_run])
        assert result.success is False
        assert result.status == ResponseStatus.PARTIAL
        assert result.successful_runs == 1

    def test_all_failed(self, test_def: TestDefinition, failed_run: RunResult) -> None:
        """TestResult with all failed runs."""
        result = TestResult(test=test_def, runs=[failed_run])
        assert result.success is False
        assert result.status == ResponseStatus.FAILED
        assert result.successful_runs == 0

    def test_timeout_status(
        self, test_def: TestDefinition, timeout_run: RunResult
    ) -> None:
        """TestResult with timeout takes priority."""
        result = TestResult(test=test_def, runs=[timeout_run])
        assert result.status == ResponseStatus.TIMEOUT

    def test_timeout_mixed(
        self,
        test_def: TestDefinition,
        success_run: RunResult,
        timeout_run: RunResult,
    ) -> None:
        """TestResult with timeout mixed with success."""
        timeout_run.run_number = 2
        result = TestResult(test=test_def, runs=[success_run, timeout_run])
        assert result.status == ResponseStatus.TIMEOUT

    def test_duration_calculation(
        self, test_def: TestDefinition, success_run: RunResult
    ) -> None:
        """TestResult calculates duration."""
        start = datetime.now()
        end = start + timedelta(seconds=5.0)
        result = TestResult(
            test=test_def,
            runs=[success_run],
            start_time=start,
            end_time=end,
        )
        assert result.duration_seconds is not None
        assert abs(result.duration_seconds - 5.0) < 0.01


class TestSuiteResult:
    """Tests for SuiteResult."""

    @pytest.fixture
    def test_def(self) -> TestDefinition:
        """Create a test definition."""
        return TestDefinition(
            id="test-001",
            name="Sample Test",
            task=TaskDefinition(description="Do something"),
        )

    @pytest.fixture
    def passed_test(self, test_def: TestDefinition) -> TestResult:
        """Create a passed test result."""
        return TestResult(
            test=test_def,
            runs=[
                RunResult(
                    test_id="test-001",
                    run_number=1,
                    response=ATPResponse(
                        task_id="task-001",
                        status=ResponseStatus.COMPLETED,
                    ),
                )
            ],
        )

    @pytest.fixture
    def failed_test(self, test_def: TestDefinition) -> TestResult:
        """Create a failed test result."""
        test = TestDefinition(
            id="test-002",
            name="Failed Test",
            task=TaskDefinition(description="Fail"),
        )
        return TestResult(
            test=test,
            runs=[
                RunResult(
                    test_id="test-002",
                    run_number=1,
                    response=ATPResponse(
                        task_id="task-002",
                        status=ResponseStatus.FAILED,
                    ),
                )
            ],
        )

    def test_empty_suite(self) -> None:
        """SuiteResult with no tests."""
        result = SuiteResult(suite_name="empty", agent_name="test-agent")
        assert result.total_tests == 0
        assert result.passed_tests == 0
        assert result.failed_tests == 0
        assert result.success is False
        assert result.success_rate == 0.0

    def test_all_passed(self, passed_test: TestResult) -> None:
        """SuiteResult with all tests passed."""
        result = SuiteResult(
            suite_name="test",
            agent_name="agent",
            tests=[passed_test],
        )
        assert result.total_tests == 1
        assert result.passed_tests == 1
        assert result.failed_tests == 0
        assert result.success is True
        assert result.success_rate == 1.0

    def test_mixed_results(
        self, passed_test: TestResult, failed_test: TestResult
    ) -> None:
        """SuiteResult with mixed results."""
        result = SuiteResult(
            suite_name="test",
            agent_name="agent",
            tests=[passed_test, failed_test],
        )
        assert result.total_tests == 2
        assert result.passed_tests == 1
        assert result.failed_tests == 1
        assert result.success is False
        assert result.success_rate == 0.5

    def test_duration_calculation(self, passed_test: TestResult) -> None:
        """SuiteResult calculates duration."""
        start = datetime.now()
        end = start + timedelta(seconds=10.0)
        result = SuiteResult(
            suite_name="test",
            agent_name="agent",
            tests=[passed_test],
            start_time=start,
            end_time=end,
        )
        assert result.duration_seconds is not None
        assert abs(result.duration_seconds - 10.0) < 0.01
