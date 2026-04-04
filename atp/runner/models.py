"""Data models for test runner."""

from pathlib import Path

from pydantic import BaseModel, Field

from atp.core.results import (  # noqa: F401 — re-export
    ProgressCallback,
    ProgressEvent,
    ProgressEventType,
    RunResult,
    SuiteResult,
    TestResult,
)


class SandboxConfig(BaseModel):
    """Configuration for test sandbox environment."""

    memory_limit: str = Field(default="2Gi", description="Memory limit (e.g., '2Gi')")
    cpu_limit: str = Field(default="2", description="CPU limit")
    network_mode: str = Field(
        default="none",
        description="Network mode: 'none', 'host', or 'custom'",
    )
    allowed_hosts: list[str] = Field(
        default_factory=list,
        description="Allowed hosts when network_mode='custom'",
    )
    workspace_path: Path = Field(
        default=Path("/workspace"),
        description="Path to workspace inside sandbox",
    )
    readonly_mounts: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Read-only mounts as (host_path, container_path) tuples",
    )
    hard_timeout_seconds: int = Field(
        default=600,
        description="Hard timeout in seconds - forcefully kills execution",
        gt=0,
    )
    enabled: bool = Field(
        default=False,
        description="Whether to use sandbox isolation",
    )
