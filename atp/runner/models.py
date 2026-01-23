"""Data models for test runner."""

from collections.abc import Callable
from dataclasses import field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

from atp.loader.models import TestDefinition
from atp.protocol import ATPEvent, ATPResponse, ResponseStatus


class SandboxConfig(BaseModel):
    """Configuration for test sandbox environment."""

    # Resource limits
    memory_limit: str = Field(default="2Gi", description="Memory limit (e.g., '2Gi')")
    cpu_limit: str = Field(default="2", description="CPU limit")

    # Network settings
    network_mode: str = Field(
        default="none",
        description="Network mode: 'none', 'host', or 'custom'",
    )
    allowed_hosts: list[str] = Field(
        default_factory=list,
        description="Allowed hosts when network_mode='custom'",
    )

    # Filesystem settings
    workspace_path: Path = Field(
        default=Path("/workspace"),
        description="Path to workspace inside sandbox",
    )
    readonly_mounts: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Read-only mounts as (host_path, container_path) tuples",
    )

    # Timeout
    hard_timeout_seconds: int = Field(
        default=600,
        description="Hard timeout in seconds - forcefully kills execution",
        gt=0,
    )

    # Enabled flag - allows disabling sandbox for local testing
    enabled: bool = Field(
        default=False,
        description="Whether to use sandbox isolation",
    )


class ProgressEventType(str, Enum):
    """Types of progress events."""

    TEST_STARTED = "test_started"
    TEST_COMPLETED = "test_completed"
    TEST_FAILED = "test_failed"
    TEST_TIMEOUT = "test_timeout"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    SUITE_STARTED = "suite_started"
    SUITE_COMPLETED = "suite_completed"
    AGENT_EVENT = "agent_event"


class ProgressEvent(BaseModel):
    """Event emitted during test execution for progress tracking."""

    event_type: ProgressEventType = Field(..., description="Type of progress event")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Event timestamp",
    )

    # Context fields - not all are used for every event type
    suite_name: str | None = Field(None, description="Suite name")
    test_id: str | None = Field(None, description="Test identifier")
    test_name: str | None = Field(None, description="Human-readable test name")
    run_number: int | None = Field(None, description="Current run number (1-indexed)")
    total_runs: int | None = Field(None, description="Total number of runs")
    total_tests: int | None = Field(None, description="Total number of tests")
    completed_tests: int | None = Field(None, description="Number of completed tests")

    # Result info for completion events
    success: bool | None = Field(None, description="Whether operation succeeded")
    error: str | None = Field(None, description="Error message if failed")

    # Agent event data
    agent_event: ATPEvent | None = Field(
        None, description="Original ATP event from agent"
    )

    # Additional context
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event-specific details",
    )


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class RunResult:
    """Result of a single test run (one execution)."""

    test_id: str
    run_number: int
    response: ATPResponse
    events: list[ATPEvent] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if run completed successfully."""
        return self.response.status == ResponseStatus.COMPLETED

    @property
    def duration_seconds(self) -> float | None:
        """Calculate run duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class TestResult:
    """Result of a test (possibly multiple runs)."""

    test: TestDefinition
    runs: list[RunResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if all runs completed successfully."""
        if not self.runs:
            return False
        return all(run.success for run in self.runs)

    @property
    def status(self) -> ResponseStatus:
        """Get aggregate status from all runs."""
        if not self.runs:
            return ResponseStatus.FAILED

        # If any run timed out, return timeout
        for run in self.runs:
            if run.response.status == ResponseStatus.TIMEOUT:
                return ResponseStatus.TIMEOUT

        # If all completed, return completed
        if all(run.response.status == ResponseStatus.COMPLETED for run in self.runs):
            return ResponseStatus.COMPLETED

        # If any failed but some completed, return partial
        completed = sum(
            1 for run in self.runs if run.response.status == ResponseStatus.COMPLETED
        )
        if completed > 0:
            return ResponseStatus.PARTIAL

        return ResponseStatus.FAILED

    @property
    def total_runs(self) -> int:
        """Get total number of runs."""
        return len(self.runs)

    @property
    def successful_runs(self) -> int:
        """Get number of successful runs."""
        return sum(1 for run in self.runs if run.success)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate total test duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    def get_run_durations(self) -> list[float]:
        """Get list of run durations (excluding None values)."""
        return [
            run.duration_seconds
            for run in self.runs
            if run.duration_seconds is not None
        ]

    def get_run_steps(self) -> list[int]:
        """Get list of steps from each run (excluding None values)."""
        return [
            run.response.metrics.total_steps
            for run in self.runs
            if run.response.metrics and run.response.metrics.total_steps is not None
        ]

    def get_run_tokens(self) -> list[int]:
        """Get list of tokens from each run (excluding None values)."""
        return [
            run.response.metrics.total_tokens
            for run in self.runs
            if run.response.metrics and run.response.metrics.total_tokens is not None
        ]

    def get_run_costs(self) -> list[float]:
        """Get list of costs from each run (excluding None values)."""
        return [
            run.response.metrics.cost_usd
            for run in self.runs
            if run.response.metrics and run.response.metrics.cost_usd is not None
        ]


@dataclass
class SuiteResult:
    """Result of running a complete test suite."""

    suite_name: str
    agent_name: str
    tests: list[TestResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    error: str | None = None

    @property
    def total_tests(self) -> int:
        """Get total number of tests."""
        return len(self.tests)

    @property
    def passed_tests(self) -> int:
        """Get number of passed tests."""
        return sum(1 for t in self.tests if t.success)

    @property
    def failed_tests(self) -> int:
        """Get number of failed tests."""
        return sum(1 for t in self.tests if not t.success)

    @property
    def success(self) -> bool:
        """Check if all tests passed."""
        return self.total_tests > 0 and all(t.success for t in self.tests)

    @property
    def success_rate(self) -> float:
        """Calculate test success rate (0.0 - 1.0)."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests

    @property
    def duration_seconds(self) -> float | None:
        """Calculate total suite duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def total_runs(self) -> int:
        """Get total number of runs across all tests."""
        return sum(t.total_runs for t in self.tests)

    @property
    def runs_per_test(self) -> int:
        """Get the number of runs per test (assumes uniform)."""
        if not self.tests:
            return 0
        return self.tests[0].total_runs if self.tests else 0
