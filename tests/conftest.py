"""Shared pytest fixtures for ATP tests."""

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_suites_dir(fixtures_dir: Path) -> Path:
    """Return path to the test_suites fixtures directory."""
    return fixtures_dir / "test_suites"


@pytest.fixture
def valid_suite_path(test_suites_dir: Path) -> Path:
    """Return path to the valid test suite YAML file."""
    return test_suites_dir / "valid_suite.yaml"


@pytest.fixture
def with_vars_suite_path(test_suites_dir: Path) -> Path:
    """Return path to the test suite with variables YAML file."""
    return test_suites_dir / "with_vars.yaml"


@pytest.fixture
def sample_env_vars() -> dict[str, str]:
    """Return sample environment variables for testing."""
    return {
        "API_ENDPOINT": "http://example.com",
        "TEST_VAR": "test_value",
        "API_KEY": "test_key_123",
    }


@pytest.fixture
def empty_env_vars() -> dict[str, str]:
    """Return empty environment variables dict."""
    return {}


@pytest.fixture
def sample_atp_request() -> dict[str, Any]:
    """Return sample ATP request data."""
    return {
        "task": {
            "description": "Test task description",
            "constraints": {
                "max_steps": 10,
                "timeout": 300,
                "allowed_tools": ["python", "bash"],
            },
        },
        "context": {"key": "value"},
    }


@pytest.fixture
def sample_atp_response() -> dict[str, Any]:
    """Return sample ATP response data."""
    return {
        "status": "success",
        "artifacts": [{"path": "/output/result.txt", "type": "text"}],
        "metrics": {"tokens": 1000, "steps": 5, "cost": 0.05},
    }


@pytest.fixture
def sample_atp_event() -> dict[str, Any]:
    """Return sample ATP event data."""
    return {
        "type": "tool_call",
        "timestamp": "2024-01-01T00:00:00Z",
        "data": {"tool": "python", "args": ["script.py"]},
    }


@pytest.fixture
def tmp_work_dir(tmp_path: Path) -> Path:
    """Create and return a temporary working directory."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    return work_dir


@pytest.fixture
def sample_test_definition() -> dict[str, Any]:
    """Return sample test definition data."""
    return {
        "id": "test-001",
        "name": "Sample Test",
        "task": {
            "description": "Perform sample task",
            "constraints": {"max_steps": 5, "timeout": 60},
        },
        "evaluators": [
            {"type": "artifact", "config": {"expected_files": ["output.txt"]}}
        ],
    }


@pytest.fixture
def sample_agent_config() -> dict[str, Any]:
    """Return sample agent configuration."""
    return {
        "type": "http",
        "config": {
            "endpoint": "http://localhost:8000",
            "timeout": 30,
            "headers": {"Content-Type": "application/json"},
        },
    }


@pytest.fixture
def mock_agent_response(sample_atp_response: dict[str, Any]) -> dict[str, Any]:
    """Return mock agent response for testing."""
    return sample_atp_response
