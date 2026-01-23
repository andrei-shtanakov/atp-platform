"""JSON Schema generation for ATP Protocol models."""

from typing import Any

from atp.protocol.models import ATPEvent, ATPRequest, ATPResponse


def generate_request_schema() -> dict[str, Any]:
    """Generate JSON Schema for ATPRequest."""
    return ATPRequest.model_json_schema()


def generate_response_schema() -> dict[str, Any]:
    """Generate JSON Schema for ATPResponse."""
    return ATPResponse.model_json_schema()


def generate_event_schema() -> dict[str, Any]:
    """Generate JSON Schema for ATPEvent."""
    return ATPEvent.model_json_schema()


def generate_all_schemas() -> dict[str, dict[str, Any]]:
    """Generate all ATP Protocol schemas."""
    return {
        "ATPRequest": generate_request_schema(),
        "ATPResponse": generate_response_schema(),
        "ATPEvent": generate_event_schema(),
    }
