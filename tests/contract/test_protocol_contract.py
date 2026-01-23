"""Contract tests for ATP Protocol models.

These tests verify that the ATP Protocol models correctly validate,
serialize, and deserialize according to the protocol specification.
Tests use JSON fixture files to ensure comprehensive coverage.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from atp.protocol.models import (
    ArtifactFile,
    ArtifactReference,
    ArtifactStructured,
    ATPEvent,
    ATPRequest,
    ATPResponse,
    ErrorPayload,
    EventType,
    LLMRequestPayload,
    ProgressPayload,
    ReasoningPayload,
    ResponseStatus,
    ToolCallPayload,
)
from atp.protocol.schema import (
    generate_all_schemas,
    generate_event_schema,
    generate_request_schema,
    generate_response_schema,
)

# Fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "protocol"


def load_fixture(category: str, name: str) -> dict[str, Any]:
    """Load a JSON fixture file."""
    fixture_path = FIXTURES_DIR / category / f"{name}.json"
    with open(fixture_path) as f:
        return json.load(f)


class TestATPRequestContract:
    """Contract tests for ATPRequest model."""

    @pytest.fixture
    def valid_fixtures(self) -> dict[str, Any]:
        """Load valid request fixtures."""
        return load_fixture("requests", "valid")

    @pytest.fixture
    def invalid_fixtures(self) -> dict[str, Any]:
        """Load invalid request fixtures."""
        return load_fixture("requests", "invalid")

    def test_valid_requests_parse_successfully(
        self, valid_fixtures: dict[str, Any]
    ) -> None:
        """All valid request fixtures should parse without errors."""
        for name, fixture in valid_fixtures.items():
            try:
                request = ATPRequest(**fixture["data"])
                assert request.task_id is not None
                assert request.task is not None
                assert request.task.description is not None
            except ValidationError as e:
                pytest.fail(f"Valid fixture '{name}' failed to parse: {e}")

    def test_minimal_request(self, valid_fixtures: dict[str, Any]) -> None:
        """Test minimal valid request with only required fields."""
        fixture = valid_fixtures["minimal_request"]
        request = ATPRequest(**fixture["data"])
        assert request.version == "1.0"  # default version
        assert request.task_id == "task-001"
        assert request.task.description == "Simple task description"
        assert request.constraints == {}
        assert request.context is None
        assert request.metadata is None

    def test_full_request(self, valid_fixtures: dict[str, Any]) -> None:
        """Test full request with all optional fields."""
        fixture = valid_fixtures["full_request"]
        request = ATPRequest(**fixture["data"])
        assert request.version == "1.0"
        assert request.task_id == "task-008-full"
        assert request.task.description == "Complete market analysis for tech sector"
        assert request.task.input_data is not None
        assert request.task.expected_artifacts is not None
        assert request.constraints["max_steps"] == 100
        assert request.context is not None
        assert request.context.tools_endpoint is not None
        assert request.metadata is not None

    def test_invalid_requests_fail_validation(
        self, invalid_fixtures: dict[str, Any]
    ) -> None:
        """All invalid request fixtures should fail validation."""
        for name, fixture in invalid_fixtures.items():
            data = fixture["data"]
            expected_error = fixture.get("expected_error", "")

            # Skip array test case which needs special handling
            if isinstance(data, list):
                with pytest.raises((ValidationError, TypeError)):
                    ATPRequest(**data)  # type: ignore[arg-type]
                continue

            with pytest.raises(ValidationError) as exc_info:
                ATPRequest(**data)

            error_str = str(exc_info.value).lower()
            if expected_error:
                assert expected_error.lower() in error_str, (
                    f"Fixture '{name}': expected error containing "
                    f"'{expected_error}', got: {error_str}"
                )

    def test_request_roundtrip_serialization(
        self, valid_fixtures: dict[str, Any]
    ) -> None:
        """Test that requests can be serialized to JSON and back."""
        for name, fixture in valid_fixtures.items():
            # Parse from fixture
            original = ATPRequest(**fixture["data"])

            # Serialize to JSON string
            json_str = original.model_dump_json()

            # Parse JSON back
            data = json.loads(json_str)

            # Reconstruct model
            reconstructed = ATPRequest(**data)

            # Verify equality
            assert original.task_id == reconstructed.task_id
            assert original.task.description == reconstructed.task.description
            assert original.version == reconstructed.version
            assert original.constraints == reconstructed.constraints


class TestATPResponseContract:
    """Contract tests for ATPResponse model."""

    @pytest.fixture
    def valid_fixtures(self) -> dict[str, Any]:
        """Load valid response fixtures."""
        return load_fixture("responses", "valid")

    @pytest.fixture
    def invalid_fixtures(self) -> dict[str, Any]:
        """Load invalid response fixtures."""
        return load_fixture("responses", "invalid")

    def test_valid_responses_parse_successfully(
        self, valid_fixtures: dict[str, Any]
    ) -> None:
        """All valid response fixtures should parse without errors."""
        for name, fixture in valid_fixtures.items():
            try:
                response = ATPResponse(**fixture["data"])
                assert response.task_id is not None
                assert response.status is not None
            except ValidationError as e:
                pytest.fail(f"Valid fixture '{name}' failed to parse: {e}")

    def test_all_status_values(self, valid_fixtures: dict[str, Any]) -> None:
        """Test responses with all valid status values."""
        status_fixtures = {
            "minimal_completed": ResponseStatus.COMPLETED,
            "failed_response": ResponseStatus.FAILED,
            "timeout_response": ResponseStatus.TIMEOUT,
            "cancelled_response": ResponseStatus.CANCELLED,
            "partial_response": ResponseStatus.PARTIAL,
        }
        for fixture_name, expected_status in status_fixtures.items():
            fixture = valid_fixtures[fixture_name]
            response = ATPResponse(**fixture["data"])
            assert response.status == expected_status

    def test_artifact_types(self, valid_fixtures: dict[str, Any]) -> None:
        """Test responses with different artifact types."""
        # File artifact
        file_fixture = valid_fixtures["completed_with_file_artifact"]
        response = ATPResponse(**file_fixture["data"])
        assert len(response.artifacts) == 1
        assert response.artifacts[0].type == "file"
        assert isinstance(response.artifacts[0], ArtifactFile)

        # Structured artifact
        struct_fixture = valid_fixtures["completed_with_structured_artifact"]
        response = ATPResponse(**struct_fixture["data"])
        assert len(response.artifacts) == 1
        assert response.artifacts[0].type == "structured"
        assert isinstance(response.artifacts[0], ArtifactStructured)

        # Reference artifact
        ref_fixture = valid_fixtures["completed_with_reference_artifact"]
        response = ATPResponse(**ref_fixture["data"])
        assert len(response.artifacts) == 1
        assert response.artifacts[0].type == "reference"
        assert isinstance(response.artifacts[0], ArtifactReference)

    def test_invalid_responses_fail_validation(
        self, invalid_fixtures: dict[str, Any]
    ) -> None:
        """All invalid response fixtures should fail validation."""
        for name, fixture in invalid_fixtures.items():
            data = fixture["data"]
            expected_error = fixture.get("expected_error", "")

            with pytest.raises(ValidationError) as exc_info:
                ATPResponse(**data)

            error_str = str(exc_info.value).lower()
            if expected_error:
                assert expected_error.lower() in error_str, (
                    f"Fixture '{name}': expected error containing "
                    f"'{expected_error}', got: {error_str}"
                )

    def test_response_roundtrip_serialization(
        self, valid_fixtures: dict[str, Any]
    ) -> None:
        """Test that responses can be serialized to JSON and back."""
        for name, fixture in valid_fixtures.items():
            # Parse from fixture
            original = ATPResponse(**fixture["data"])

            # Serialize to JSON string
            json_str = original.model_dump_json()

            # Parse JSON back
            data = json.loads(json_str)

            # Reconstruct model
            reconstructed = ATPResponse(**data)

            # Verify equality
            assert original.task_id == reconstructed.task_id
            assert original.status == reconstructed.status
            assert len(original.artifacts) == len(reconstructed.artifacts)


class TestATPEventContract:
    """Contract tests for ATPEvent model."""

    @pytest.fixture
    def valid_fixtures(self) -> dict[str, Any]:
        """Load valid event fixtures."""
        return load_fixture("events", "valid")

    @pytest.fixture
    def invalid_fixtures(self) -> dict[str, Any]:
        """Load invalid event fixtures."""
        return load_fixture("events", "invalid")

    def test_valid_events_parse_successfully(
        self, valid_fixtures: dict[str, Any]
    ) -> None:
        """All valid event fixtures should parse without errors."""
        for name, fixture in valid_fixtures.items():
            try:
                event = ATPEvent(**fixture["data"])
                assert event.task_id is not None
                assert event.sequence >= 0
                assert event.event_type is not None
            except ValidationError as e:
                pytest.fail(f"Valid fixture '{name}' failed to parse: {e}")

    def test_all_event_types(self, valid_fixtures: dict[str, Any]) -> None:
        """Test events with all valid event types."""
        event_type_fixtures = {
            "tool_call_complete": EventType.TOOL_CALL,
            "llm_request_basic": EventType.LLM_REQUEST,
            "reasoning_full": EventType.REASONING,
            "error_recoverable": EventType.ERROR,
            "progress_percentage": EventType.PROGRESS,
        }
        for fixture_name, expected_type in event_type_fixtures.items():
            fixture = valid_fixtures[fixture_name]
            event = ATPEvent(**fixture["data"])
            assert event.event_type == expected_type

    def test_invalid_events_fail_validation(
        self, invalid_fixtures: dict[str, Any]
    ) -> None:
        """All invalid event fixtures should fail validation."""
        for name, fixture in invalid_fixtures.items():
            data = fixture["data"]
            expected_error = fixture.get("expected_error", "")

            with pytest.raises(ValidationError) as exc_info:
                ATPEvent(**data)

            error_str = str(exc_info.value).lower()
            if expected_error:
                assert expected_error.lower() in error_str, (
                    f"Fixture '{name}': expected error containing "
                    f"'{expected_error}', got: {error_str}"
                )

    def test_event_roundtrip_serialization(
        self, valid_fixtures: dict[str, Any]
    ) -> None:
        """Test that events can be serialized to JSON and back."""
        for name, fixture in valid_fixtures.items():
            # Parse from fixture
            original = ATPEvent(**fixture["data"])

            # Serialize to JSON string
            json_str = original.model_dump_json()

            # Parse JSON back
            data = json.loads(json_str)

            # Reconstruct model
            reconstructed = ATPEvent(**data)

            # Verify equality
            assert original.task_id == reconstructed.task_id
            assert original.sequence == reconstructed.sequence
            assert original.event_type == reconstructed.event_type
            assert original.payload == reconstructed.payload


class TestEventPayloadsContract:
    """Contract tests for event payload models."""

    def test_tool_call_payload_validation(self) -> None:
        """Test ToolCallPayload validates correctly."""
        # Valid payload
        payload = ToolCallPayload(
            tool="web_search",
            input={"query": "test"},
            output={"results": []},
            duration_ms=100.5,
            status="success",
        )
        assert payload.tool == "web_search"
        assert payload.duration_ms == 100.5

        # Empty tool name should fail
        with pytest.raises(ValidationError):
            ToolCallPayload(tool="")

        # Negative duration should fail
        with pytest.raises(ValidationError):
            ToolCallPayload(tool="search", duration_ms=-10.0)

    def test_llm_request_payload_validation(self) -> None:
        """Test LLMRequestPayload validates correctly."""
        # Valid payload
        payload = LLMRequestPayload(
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            duration_ms=2000.0,
        )
        assert payload.model == "claude-sonnet-4-20250514"
        assert payload.input_tokens == 1000

        # Empty model name should fail
        with pytest.raises(ValidationError):
            LLMRequestPayload(model="")

        # Negative tokens should fail
        with pytest.raises(ValidationError):
            LLMRequestPayload(model="gpt-4", input_tokens=-100)

    def test_reasoning_payload_validation(self) -> None:
        """Test ReasoningPayload validates correctly."""
        # Valid payload with all fields
        payload = ReasoningPayload(
            thought="Analyzing the data",
            plan="Step 1: Gather info, Step 2: Process",
            step="Analysis",
        )
        assert payload.thought == "Analyzing the data"

        # Valid payload with optional fields
        payload = ReasoningPayload()
        assert payload.thought is None
        assert payload.plan is None
        assert payload.step is None

    def test_error_payload_validation(self) -> None:
        """Test ErrorPayload validates correctly."""
        # Valid payload
        payload = ErrorPayload(
            error_type="ConnectionError",
            message="Connection failed",
            recoverable=True,
        )
        assert payload.error_type == "ConnectionError"
        assert payload.recoverable is True

        # Empty error_type should fail
        with pytest.raises(ValidationError):
            ErrorPayload(error_type="", message="Error")

        # Empty message should fail
        with pytest.raises(ValidationError):
            ErrorPayload(error_type="Error", message="")

    def test_progress_payload_validation(self) -> None:
        """Test ProgressPayload validates correctly."""
        # Valid payload
        payload = ProgressPayload(
            current_step=5,
            percentage=50.0,
            message="Processing",
        )
        assert payload.current_step == 5
        assert payload.percentage == 50.0

        # Percentage over 100 should fail
        with pytest.raises(ValidationError):
            ProgressPayload(percentage=101.0)

        # Negative percentage should fail
        with pytest.raises(ValidationError):
            ProgressPayload(percentage=-10.0)

        # Negative current_step should fail
        with pytest.raises(ValidationError):
            ProgressPayload(current_step=-1)


class TestJSONSchemaGeneration:
    """Tests for JSON Schema generation."""

    def test_request_schema_structure(self) -> None:
        """Test request schema has correct structure."""
        schema = generate_request_schema()

        # Check top-level properties
        assert "properties" in schema
        assert "required" in schema

        # Check required fields
        assert "task_id" in schema["required"]
        assert "task" in schema["required"]

        # Check property definitions
        props = schema["properties"]
        assert "version" in props
        assert "task_id" in props
        assert "task" in props
        assert "constraints" in props
        assert "context" in props
        assert "metadata" in props

    def test_response_schema_structure(self) -> None:
        """Test response schema has correct structure."""
        schema = generate_response_schema()

        # Check required fields
        assert "task_id" in schema["required"]
        assert "status" in schema["required"]

        # Check property definitions
        props = schema["properties"]
        assert "version" in props
        assert "task_id" in props
        assert "status" in props
        assert "artifacts" in props
        assert "metrics" in props
        assert "error" in props
        assert "trace_id" in props

    def test_event_schema_structure(self) -> None:
        """Test event schema has correct structure."""
        schema = generate_event_schema()

        # Check required fields
        assert "task_id" in schema["required"]
        assert "sequence" in schema["required"]
        assert "event_type" in schema["required"]
        assert "payload" in schema["required"]

        # Check property definitions
        props = schema["properties"]
        assert "version" in props
        assert "task_id" in props
        assert "timestamp" in props
        assert "sequence" in props
        assert "event_type" in props
        assert "payload" in props

    def test_all_schemas_json_serializable(self) -> None:
        """Test all generated schemas are valid JSON."""
        schemas = generate_all_schemas()

        for name, schema in schemas.items():
            try:
                json_str = json.dumps(schema)
                # Verify it can be parsed back
                parsed = json.loads(json_str)
                assert parsed == schema
            except (TypeError, ValueError) as e:
                pytest.fail(f"Schema {name} is not JSON serializable: {e}")

    def test_schema_includes_definitions(self) -> None:
        """Test schemas include type definitions."""
        schema = generate_request_schema()
        # Pydantic v2 uses $defs
        assert "$defs" in schema or "definitions" in schema


class TestProtocolVersionHandling:
    """Tests for protocol version handling."""

    def test_default_version_request(self) -> None:
        """Test request uses default version when not specified."""
        request = ATPRequest(
            task_id="test-task",
            task={"description": "Test"},  # type: ignore[arg-type]
        )
        assert request.version == "1.0"

    def test_explicit_version_request(self) -> None:
        """Test request accepts explicit version."""
        request = ATPRequest(
            version="1.0",
            task_id="test-task",
            task={"description": "Test"},  # type: ignore[arg-type]
        )
        assert request.version == "1.0"

    def test_default_version_response(self) -> None:
        """Test response uses default version when not specified."""
        response = ATPResponse(
            task_id="test-task",
            status=ResponseStatus.COMPLETED,
        )
        assert response.version == "1.0"

    def test_explicit_version_response(self) -> None:
        """Test response accepts explicit version."""
        response = ATPResponse(
            version="1.0",
            task_id="test-task",
            status=ResponseStatus.COMPLETED,
        )
        assert response.version == "1.0"

    def test_default_version_event(self) -> None:
        """Test event uses default version when not specified."""
        event = ATPEvent(
            task_id="test-task",
            sequence=0,
            event_type=EventType.PROGRESS,
            payload={},
        )
        assert event.version == "1.0"

    def test_explicit_version_event(self) -> None:
        """Test event accepts explicit version."""
        event = ATPEvent(
            version="1.0",
            task_id="test-task",
            sequence=0,
            event_type=EventType.PROGRESS,
            payload={},
        )
        assert event.version == "1.0"

    def test_version_preserved_in_serialization(self) -> None:
        """Test version is preserved through serialization."""
        # Request
        request = ATPRequest(
            version="1.0",
            task_id="test",
            task={"description": "Test"},  # type: ignore[arg-type]
        )
        data = json.loads(request.model_dump_json())
        assert data["version"] == "1.0"

        # Response
        response = ATPResponse(
            version="1.0",
            task_id="test",
            status=ResponseStatus.COMPLETED,
        )
        data = json.loads(response.model_dump_json())
        assert data["version"] == "1.0"

        # Event
        event = ATPEvent(
            version="1.0",
            task_id="test",
            sequence=0,
            event_type=EventType.PROGRESS,
            payload={},
        )
        data = json.loads(event.model_dump_json())
        assert data["version"] == "1.0"


class TestPydanticModelRoundtrip:
    """Tests for Pydantic model roundtrip serialization."""

    def test_request_dict_roundtrip(self) -> None:
        """Test request model_dump and reconstruction."""
        original = ATPRequest(
            version="1.0",
            task_id="roundtrip-test",
            task={  # type: ignore[arg-type]
                "description": "Test roundtrip",
                "input_data": {"key": "value"},
                "expected_artifacts": ["output.txt"],
            },
            constraints={"max_steps": 100, "timeout_seconds": 300},
            context={  # type: ignore[arg-type]
                "tools_endpoint": "http://localhost:8000",
                "workspace_path": "/workspace",
            },
            metadata={"test": True},
        )

        # Convert to dict
        data = original.model_dump()

        # Reconstruct
        reconstructed = ATPRequest(**data)

        # Verify all fields
        assert original.version == reconstructed.version
        assert original.task_id == reconstructed.task_id
        assert original.task.description == reconstructed.task.description
        assert original.task.input_data == reconstructed.task.input_data
        assert original.task.expected_artifacts == reconstructed.task.expected_artifacts
        assert original.constraints == reconstructed.constraints
        assert original.context is not None
        assert reconstructed.context is not None
        assert original.context.tools_endpoint == reconstructed.context.tools_endpoint
        assert original.metadata == reconstructed.metadata

    def test_response_dict_roundtrip(self) -> None:
        """Test response model_dump and reconstruction."""
        original = ATPResponse(
            version="1.0",
            task_id="roundtrip-test",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(path="output.txt", content="Hello"),
                ArtifactStructured(name="data", data={"count": 42}),
            ],
            metrics={  # type: ignore[arg-type]
                "total_tokens": 1000,
                "wall_time_seconds": 30.5,
            },
            trace_id="trace-123",
        )

        # Convert to dict
        data = original.model_dump()

        # Reconstruct
        reconstructed = ATPResponse(**data)

        # Verify all fields
        assert original.version == reconstructed.version
        assert original.task_id == reconstructed.task_id
        assert original.status == reconstructed.status
        assert len(original.artifacts) == len(reconstructed.artifacts)
        assert original.artifacts[0].type == reconstructed.artifacts[0].type
        assert original.trace_id == reconstructed.trace_id

    def test_event_dict_roundtrip(self) -> None:
        """Test event model_dump and reconstruction."""
        original = ATPEvent(
            version="1.0",
            task_id="roundtrip-test",
            sequence=5,
            event_type=EventType.TOOL_CALL,
            payload={
                "tool": "search",
                "input": {"query": "test"},
                "output": {"results": []},
                "duration_ms": 100.5,
            },
        )

        # Convert to dict
        data = original.model_dump()

        # Reconstruct
        reconstructed = ATPEvent(**data)

        # Verify all fields
        assert original.version == reconstructed.version
        assert original.task_id == reconstructed.task_id
        assert original.sequence == reconstructed.sequence
        assert original.event_type == reconstructed.event_type
        assert original.payload == reconstructed.payload

    def test_json_roundtrip_preserves_types(self) -> None:
        """Test JSON roundtrip preserves correct types."""
        # Create response with mixed artifact types
        original = ATPResponse(
            task_id="type-test",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(path="file.txt"),
                ArtifactStructured(name="data", data={"x": 1}),
                ArtifactReference(path="s3://bucket/file"),
            ],
        )

        # Roundtrip through JSON
        json_str = original.model_dump_json()
        data = json.loads(json_str)
        reconstructed = ATPResponse(**data)

        # Verify artifact types
        assert isinstance(reconstructed.artifacts[0], ArtifactFile)
        assert isinstance(reconstructed.artifacts[1], ArtifactStructured)
        assert isinstance(reconstructed.artifacts[2], ArtifactReference)

    def test_enum_serialization(self) -> None:
        """Test enums serialize to string values."""
        response = ATPResponse(
            task_id="enum-test",
            status=ResponseStatus.COMPLETED,
        )

        data = json.loads(response.model_dump_json())
        assert data["status"] == "completed"

        event = ATPEvent(
            task_id="enum-test",
            sequence=0,
            event_type=EventType.TOOL_CALL,
            payload={},
        )

        data = json.loads(event.model_dump_json())
        assert data["event_type"] == "tool_call"

    def test_optional_fields_null_handling(self) -> None:
        """Test optional fields handle null correctly."""
        response = ATPResponse(
            task_id="null-test",
            status=ResponseStatus.COMPLETED,
            metrics=None,
            error=None,
            trace_id=None,
        )

        data = response.model_dump()
        assert data["metrics"] is None
        assert data["error"] is None
        assert data["trace_id"] is None

        # Roundtrip
        reconstructed = ATPResponse(**data)
        assert reconstructed.metrics is None
        assert reconstructed.error is None
        assert reconstructed.trace_id is None
