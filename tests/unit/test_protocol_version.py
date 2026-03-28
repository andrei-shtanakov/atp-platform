"""Tests for ATP Protocol version handling."""

import pytest

from atp.protocol import (
    PROTOCOL_VERSION,
    ATPEvent,
    ATPRequest,
    ATPResponse,
)


def test_protocol_version_constant_exists() -> None:
    """PROTOCOL_VERSION is exported and matches expected format."""
    assert isinstance(PROTOCOL_VERSION, str)
    assert PROTOCOL_VERSION == "1.0"


def test_request_default_version() -> None:
    """ATPRequest defaults to current PROTOCOL_VERSION."""
    request = ATPRequest(
        task_id="test-1",
        task={"description": "Test task"},
    )
    assert request.version == PROTOCOL_VERSION


def test_response_default_version() -> None:
    """ATPResponse defaults to current PROTOCOL_VERSION."""
    response = ATPResponse(
        task_id="test-1",
        status="completed",
    )
    assert response.version == PROTOCOL_VERSION


def test_event_default_version() -> None:
    """ATPEvent defaults to current PROTOCOL_VERSION."""
    event = ATPEvent(
        task_id="test-1",
        sequence=0,
        event_type="progress",
        payload={"message": "test"},
    )
    assert event.version == PROTOCOL_VERSION


def test_request_rejects_unsupported_version() -> None:
    """ATPRequest rejects versions outside supported set."""
    with pytest.raises(ValueError, match="version"):
        ATPRequest(
            task_id="test-1",
            task={"description": "Test task"},
            version="99.0",
        )


def test_response_rejects_unsupported_version() -> None:
    """ATPResponse rejects unsupported versions."""
    with pytest.raises(ValueError, match="version"):
        ATPResponse(
            task_id="test-1",
            status="completed",
            version="99.0",
        )


def test_event_rejects_unsupported_version() -> None:
    """ATPEvent rejects unsupported versions."""
    with pytest.raises(ValueError, match="version"):
        ATPEvent(
            task_id="test-1",
            sequence=0,
            event_type="progress",
            payload={},
            version="99.0",
        )


def test_request_accepts_supported_version() -> None:
    """ATPRequest accepts explicitly set supported version."""
    request = ATPRequest(
        task_id="test-1",
        task={"description": "Test task"},
        version="1.0",
    )
    assert request.version == "1.0"
