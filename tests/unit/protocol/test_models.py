"""Unit tests for ATP Protocol models."""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from atp.protocol.models import (
    ArtifactFile,
    ArtifactReference,
    ArtifactStructured,
    ATPEvent,
    ATPRequest,
    ATPResponse,
    Context,
    ErrorPayload,
    EventType,
    LLMRequestPayload,
    Metrics,
    ProgressPayload,
    ReasoningPayload,
    ResponseStatus,
    Task,
    ToolCallPayload,
)


class TestTask:
    """Tests for Task model."""

    def test_valid_task(self) -> None:
        """Test valid task creation."""
        task = Task(description="Test task")
        assert task.description == "Test task"
        assert task.input_data is None
        assert task.expected_artifacts is None

    def test_task_with_input_data(self) -> None:
        """Test task with input data."""
        task = Task(
            description="Process data",
            input_data={"key": "value", "numbers": [1, 2, 3]},
            expected_artifacts=["output.json", "report.md"],
        )
        assert task.description == "Process data"
        assert task.input_data == {"key": "value", "numbers": [1, 2, 3]}
        assert task.expected_artifacts == ["output.json", "report.md"]

    def test_empty_description_invalid(self) -> None:
        """Test that empty description is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            Task(description="")
        assert "at least 1 character" in str(exc_info.value).lower()

    def test_task_serialization(self) -> None:
        """Test task serialization to JSON."""
        task = Task(description="Test", input_data={"x": 1})
        json_str = task.model_dump_json()
        data = json.loads(json_str)
        assert data["description"] == "Test"
        assert data["input_data"] == {"x": 1}

    def test_task_deserialization(self) -> None:
        """Test task deserialization from JSON."""
        data = {"description": "Test task", "input_data": {"key": "value"}}
        task = Task(**data)
        assert task.description == "Test task"
        assert task.input_data == {"key": "value"}


class TestContext:
    """Tests for Context model."""

    def test_empty_context(self) -> None:
        """Test context with all None values."""
        context = Context()
        assert context.tools_endpoint is None
        assert context.workspace_path is None
        assert context.environment is None

    def test_full_context(self) -> None:
        """Test context with all fields."""
        context = Context(
            tools_endpoint="http://localhost:8000/tools",
            workspace_path="/workspace",
            environment={"API_KEY": "secret", "DEBUG": "true"},
        )
        assert context.tools_endpoint == "http://localhost:8000/tools"
        assert context.workspace_path == "/workspace"
        assert context.environment == {"API_KEY": "secret", "DEBUG": "true"}


class TestATPRequest:
    """Tests for ATPRequest model."""

    def test_minimal_request(self) -> None:
        """Test minimal valid request."""
        request = ATPRequest(task_id="test-123", task=Task(description="Do something"))
        assert request.version == "1.0"
        assert request.task_id == "test-123"
        assert request.task.description == "Do something"
        assert request.constraints == {}
        assert request.context is None
        assert request.metadata is None

    def test_full_request(self) -> None:
        """Test request with all fields."""
        request = ATPRequest(
            version="1.0",
            task_id="task-456",
            task=Task(
                description="Analyze data",
                input_data={"file": "data.csv"},
                expected_artifacts=["report.pdf"],
            ),
            constraints={
                "max_steps": 50,
                "max_tokens": 10000,
                "timeout_seconds": 300,
                "allowed_tools": ["web_search", "calculator"],
            },
            context=Context(
                tools_endpoint="http://tools.local",
                workspace_path="/tmp/workspace",
                environment={"ENV": "test"},
            ),
            metadata={"user": "test_user", "priority": "high"},
        )
        assert request.task_id == "task-456"
        assert request.constraints["max_steps"] == 50
        assert request.context.tools_endpoint == "http://tools.local"
        assert request.metadata["user"] == "test_user"

    def test_empty_task_id_invalid(self) -> None:
        """Test that empty task_id is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ATPRequest(task_id="", task=Task(description="Test"))
        assert "task_id cannot be empty" in str(exc_info.value)

    def test_whitespace_task_id_invalid(self) -> None:
        """Test that whitespace-only task_id is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ATPRequest(task_id="   ", task=Task(description="Test"))
        assert "task_id cannot be empty" in str(exc_info.value)

    def test_request_serialization(self) -> None:
        """Test request serialization to JSON."""
        request = ATPRequest(
            task_id="test-123",
            task=Task(description="Test"),
            constraints={"max_steps": 10},
        )
        json_str = request.model_dump_json()
        data = json.loads(json_str)
        assert data["task_id"] == "test-123"
        assert data["task"]["description"] == "Test"
        assert data["constraints"]["max_steps"] == 10

    def test_request_deserialization(self) -> None:
        """Test request deserialization from JSON."""
        data = {
            "task_id": "test-789",
            "task": {"description": "Process"},
            "constraints": {"timeout_seconds": 120},
        }
        request = ATPRequest(**data)
        assert request.task_id == "test-789"
        assert request.task.description == "Process"
        assert request.constraints["timeout_seconds"] == 120


class TestMetrics:
    """Tests for Metrics model."""

    def test_empty_metrics(self) -> None:
        """Test metrics with all None values."""
        metrics = Metrics()
        assert metrics.total_tokens is None
        assert metrics.wall_time_seconds is None

    def test_full_metrics(self) -> None:
        """Test metrics with all fields."""
        metrics = Metrics(
            total_tokens=1500,
            input_tokens=1000,
            output_tokens=500,
            total_steps=10,
            tool_calls=5,
            llm_calls=3,
            wall_time_seconds=45.5,
            cost_usd=0.025,
        )
        assert metrics.total_tokens == 1500
        assert metrics.input_tokens == 1000
        assert metrics.output_tokens == 500
        assert metrics.total_steps == 10
        assert metrics.tool_calls == 5
        assert metrics.llm_calls == 3
        assert metrics.wall_time_seconds == 45.5
        assert metrics.cost_usd == 0.025

    def test_negative_values_invalid(self) -> None:
        """Test that negative values are invalid."""
        with pytest.raises(ValidationError):
            Metrics(total_tokens=-100)

        with pytest.raises(ValidationError):
            Metrics(wall_time_seconds=-1.0)

        with pytest.raises(ValidationError):
            Metrics(cost_usd=-0.01)


class TestArtifacts:
    """Tests for Artifact models."""

    def test_file_artifact(self) -> None:
        """Test file artifact."""
        artifact = ArtifactFile(
            path="output.txt",
            content_type="text/plain",
            size_bytes=1024,
            content_hash="sha256:abc123",
            content="Hello, world!",
        )
        assert artifact.type == "file"
        assert artifact.path == "output.txt"
        assert artifact.content_type == "text/plain"
        assert artifact.size_bytes == 1024
        assert artifact.content == "Hello, world!"

    def test_file_artifact_minimal(self) -> None:
        """Test file artifact with minimal fields."""
        artifact = ArtifactFile(path="data.bin")
        assert artifact.type == "file"
        assert artifact.path == "data.bin"
        assert artifact.content_type is None

    def test_structured_artifact(self) -> None:
        """Test structured artifact."""
        artifact = ArtifactStructured(
            name="results",
            data={"status": "success", "count": 42, "items": [1, 2, 3]},
            content_type="application/json",
        )
        assert artifact.type == "structured"
        assert artifact.name == "results"
        assert artifact.data["status"] == "success"
        assert artifact.data["count"] == 42

    def test_reference_artifact(self) -> None:
        """Test reference artifact."""
        artifact = ArtifactReference(
            path="s3://bucket/large-file.dat",
            content_type="application/octet-stream",
            size_bytes=1024000,
        )
        assert artifact.type == "reference"
        assert artifact.path == "s3://bucket/large-file.dat"
        assert artifact.size_bytes == 1024000

    def test_empty_path_invalid(self) -> None:
        """Test that empty path is invalid."""
        with pytest.raises(ValidationError):
            ArtifactFile(path="")

        with pytest.raises(ValidationError):
            ArtifactReference(path="")

    def test_empty_name_invalid(self) -> None:
        """Test that empty name is invalid."""
        with pytest.raises(ValidationError):
            ArtifactStructured(name="", data={})


class TestATPResponse:
    """Tests for ATPResponse model."""

    def test_minimal_response(self) -> None:
        """Test minimal valid response."""
        response = ATPResponse(task_id="test-123", status=ResponseStatus.COMPLETED)
        assert response.version == "1.0"
        assert response.task_id == "test-123"
        assert response.status == ResponseStatus.COMPLETED
        assert response.artifacts == []
        assert response.metrics is None
        assert response.error is None

    def test_completed_response(self) -> None:
        """Test successful completed response."""
        response = ATPResponse(
            task_id="task-456",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(path="output.txt", content="Result"),
                ArtifactStructured(name="data", data={"key": "value"}),
            ],
            metrics=Metrics(total_tokens=1000, wall_time_seconds=30.5),
        )
        assert response.status == ResponseStatus.COMPLETED
        assert len(response.artifacts) == 2
        assert response.metrics.total_tokens == 1000
        assert response.error is None

    def test_failed_response(self) -> None:
        """Test failed response with error."""
        response = ATPResponse(
            task_id="task-789",
            status=ResponseStatus.FAILED,
            error="Failed to connect to database",
            metrics=Metrics(wall_time_seconds=5.0),
        )
        assert response.status == ResponseStatus.FAILED
        assert response.error == "Failed to connect to database"
        assert response.metrics.wall_time_seconds == 5.0

    def test_timeout_response(self) -> None:
        """Test timeout response."""
        response = ATPResponse(
            task_id="task-999",
            status=ResponseStatus.TIMEOUT,
            artifacts=[ArtifactFile(path="partial.txt")],
            metrics=Metrics(wall_time_seconds=300.0),
        )
        assert response.status == ResponseStatus.TIMEOUT
        assert len(response.artifacts) == 1

    def test_all_status_values(self) -> None:
        """Test all status values are valid."""
        for status in ResponseStatus:
            response = ATPResponse(task_id="test", status=status)
            assert response.status == status

    def test_empty_task_id_invalid(self) -> None:
        """Test that empty task_id is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ATPResponse(task_id="", status=ResponseStatus.COMPLETED)
        assert "task_id cannot be empty" in str(exc_info.value)

    def test_response_serialization(self) -> None:
        """Test response serialization to JSON."""
        response = ATPResponse(
            task_id="test-123",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="out.txt")],
        )
        json_str = response.model_dump_json()
        data = json.loads(json_str)
        assert data["task_id"] == "test-123"
        assert data["status"] == "completed"
        assert data["artifacts"][0]["type"] == "file"

    def test_response_deserialization(self) -> None:
        """Test response deserialization from JSON."""
        data = {
            "task_id": "test-456",
            "status": "failed",
            "error": "Something went wrong",
            "artifacts": [],
        }
        response = ATPResponse(**data)
        assert response.task_id == "test-456"
        assert response.status == ResponseStatus.FAILED
        assert response.error == "Something went wrong"


class TestEventPayloads:
    """Tests for event payload models."""

    def test_tool_call_payload(self) -> None:
        """Test tool call payload."""
        payload = ToolCallPayload(
            tool="web_search",
            input={"query": "Python testing"},
            output={"results": [{"title": "PyTest"}]},
            duration_ms=150.5,
            status="success",
        )
        assert payload.tool == "web_search"
        assert payload.input["query"] == "Python testing"
        assert payload.duration_ms == 150.5

    def test_llm_request_payload(self) -> None:
        """Test LLM request payload."""
        payload = LLMRequestPayload(
            model="claude-sonnet-4",
            input_tokens=500,
            output_tokens=300,
            duration_ms=2000.0,
        )
        assert payload.model == "claude-sonnet-4"
        assert payload.input_tokens == 500
        assert payload.output_tokens == 300

    def test_reasoning_payload(self) -> None:
        """Test reasoning payload."""
        payload = ReasoningPayload(
            thought="I need to search for information",
            plan="1. Search web, 2. Analyze results",
            step="Searching web for Python testing",
        )
        assert payload.thought == "I need to search for information"
        assert payload.plan == "1. Search web, 2. Analyze results"

    def test_error_payload(self) -> None:
        """Test error payload."""
        payload = ErrorPayload(
            error_type="ConnectionError",
            message="Failed to connect to API",
            recoverable=True,
        )
        assert payload.error_type == "ConnectionError"
        assert payload.message == "Failed to connect to API"
        assert payload.recoverable is True

    def test_progress_payload(self) -> None:
        """Test progress payload."""
        payload = ProgressPayload(
            current_step=5, percentage=50.0, message="Processing data"
        )
        assert payload.current_step == 5
        assert payload.percentage == 50.0
        assert payload.message == "Processing data"

    def test_progress_percentage_validation(self) -> None:
        """Test progress percentage bounds."""
        # Valid values
        ProgressPayload(percentage=0.0)
        ProgressPayload(percentage=50.0)
        ProgressPayload(percentage=100.0)

        # Invalid values
        with pytest.raises(ValidationError):
            ProgressPayload(percentage=-1.0)

        with pytest.raises(ValidationError):
            ProgressPayload(percentage=101.0)


class TestATPEvent:
    """Tests for ATPEvent model."""

    def test_minimal_event(self) -> None:
        """Test minimal valid event."""
        event = ATPEvent(
            task_id="test-123",
            sequence=0,
            event_type=EventType.PROGRESS,
            payload={"message": "Started"},
        )
        assert event.version == "1.0"
        assert event.task_id == "test-123"
        assert event.sequence == 0
        assert event.event_type == EventType.PROGRESS
        assert isinstance(event.timestamp, datetime)

    def test_tool_call_event(self) -> None:
        """Test tool call event."""
        event = ATPEvent(
            task_id="task-456",
            sequence=1,
            event_type=EventType.TOOL_CALL,
            payload={
                "tool": "calculator",
                "input": {"expr": "2+2"},
                "output": {"result": 4},
                "duration_ms": 10.0,
                "status": "success",
            },
        )
        assert event.event_type == EventType.TOOL_CALL
        assert event.payload["tool"] == "calculator"

    def test_llm_request_event(self) -> None:
        """Test LLM request event."""
        event = ATPEvent(
            task_id="task-789",
            sequence=2,
            event_type=EventType.LLM_REQUEST,
            payload={
                "model": "gpt-4",
                "input_tokens": 100,
                "output_tokens": 50,
                "duration_ms": 1500.0,
            },
        )
        assert event.event_type == EventType.LLM_REQUEST
        assert event.payload["model"] == "gpt-4"

    def test_reasoning_event(self) -> None:
        """Test reasoning event."""
        event = ATPEvent(
            task_id="task-111",
            sequence=3,
            event_type=EventType.REASONING,
            payload={"thought": "Need to verify results", "step": "Verification"},
        )
        assert event.event_type == EventType.REASONING
        assert event.payload["thought"] == "Need to verify results"

    def test_error_event(self) -> None:
        """Test error event."""
        event = ATPEvent(
            task_id="task-222",
            sequence=4,
            event_type=EventType.ERROR,
            payload={
                "error_type": "ValueError",
                "message": "Invalid input",
                "recoverable": False,
            },
        )
        assert event.event_type == EventType.ERROR
        assert event.payload["error_type"] == "ValueError"

    def test_all_event_types(self) -> None:
        """Test all event types are valid."""
        for i, event_type in enumerate(EventType):
            event = ATPEvent(
                task_id="test", sequence=i, event_type=event_type, payload={}
            )
            assert event.event_type == event_type

    def test_empty_task_id_invalid(self) -> None:
        """Test that empty task_id is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ATPEvent(
                task_id="",
                sequence=0,
                event_type=EventType.PROGRESS,
                payload={},
            )
        assert "task_id cannot be empty" in str(exc_info.value)

    def test_negative_sequence_invalid(self) -> None:
        """Test that negative sequence is invalid."""
        with pytest.raises(ValidationError):
            ATPEvent(
                task_id="test",
                sequence=-1,
                event_type=EventType.PROGRESS,
                payload={},
            )

    def test_event_serialization(self) -> None:
        """Test event serialization to JSON."""
        event = ATPEvent(
            task_id="test-123",
            sequence=5,
            event_type=EventType.PROGRESS,
            payload={"percentage": 75.0},
        )
        json_str = event.model_dump_json()
        data = json.loads(json_str)
        assert data["task_id"] == "test-123"
        assert data["sequence"] == 5
        assert data["event_type"] == "progress"

    def test_event_deserialization(self) -> None:
        """Test event deserialization from JSON."""
        data = {
            "task_id": "test-456",
            "timestamp": "2024-01-15T10:30:00",
            "sequence": 10,
            "event_type": "tool_call",
            "payload": {"tool": "search"},
        }
        event = ATPEvent(**data)
        assert event.task_id == "test-456"
        assert event.sequence == 10
        assert event.event_type == EventType.TOOL_CALL


class TestSecurityValidations:
    """Tests for security-related validations."""

    def test_task_id_rejects_special_chars(self) -> None:
        """Test that task_id rejects special characters."""
        with pytest.raises(ValidationError) as exc_info:
            ATPRequest(task_id="task/with/slashes", task=Task(description="Test"))
        assert "alphanumeric" in str(exc_info.value).lower()

    def test_task_id_rejects_dots(self) -> None:
        """Test that task_id rejects path traversal attempts."""
        with pytest.raises(ValidationError) as exc_info:
            ATPRequest(task_id="task..id", task=Task(description="Test"))
        assert "alphanumeric" in str(exc_info.value).lower()

    def test_task_id_too_long(self) -> None:
        """Test that overly long task_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ATPRequest(task_id="a" * 200, task=Task(description="Test"))
        assert "too long" in str(exc_info.value).lower()

    def test_artifact_path_rejects_traversal(self) -> None:
        """Test that artifact path rejects path traversal."""
        with pytest.raises(ValidationError) as exc_info:
            ArtifactFile(path="../../../etc/passwd")
        assert "traversal" in str(exc_info.value).lower()

    def test_artifact_path_rejects_absolute(self) -> None:
        """Test that artifact path rejects absolute paths."""
        with pytest.raises(ValidationError) as exc_info:
            ArtifactFile(path="/etc/passwd")
        assert "absolute" in str(exc_info.value).lower()

    def test_artifact_path_rejects_null_bytes(self) -> None:
        """Test that artifact path rejects null bytes."""
        with pytest.raises(ValidationError) as exc_info:
            ArtifactFile(path="file\x00.txt")
        assert "null" in str(exc_info.value).lower()

    def test_artifact_name_rejects_slashes(self) -> None:
        """Test that artifact name rejects path separators."""
        with pytest.raises(ValidationError) as exc_info:
            ArtifactStructured(name="path/to/file", data={})
        assert "separator" in str(exc_info.value).lower()

    def test_context_tools_endpoint_validates_scheme(self) -> None:
        """Test that tools endpoint requires HTTP/HTTPS."""
        with pytest.raises(ValidationError) as exc_info:
            Context(tools_endpoint="file:///etc/passwd")
        assert "http" in str(exc_info.value).lower()

    def test_context_workspace_rejects_null_bytes(self) -> None:
        """Test that workspace path rejects null bytes."""
        with pytest.raises(ValidationError) as exc_info:
            Context(workspace_path="/tmp/test\x00path")
        assert "null" in str(exc_info.value).lower()

    def test_context_environment_rejects_null_bytes(self) -> None:
        """Test that environment rejects null bytes."""
        with pytest.raises(ValidationError) as exc_info:
            Context(environment={"VAR\x00NAME": "value"})
        assert "null" in str(exc_info.value).lower()

    def test_expected_artifacts_rejects_traversal(self) -> None:
        """Test that expected artifacts reject path traversal."""
        with pytest.raises(ValidationError) as exc_info:
            Task(description="Test", expected_artifacts=["../secret.txt"])
        assert "invalid" in str(exc_info.value).lower()

    def test_expected_artifacts_rejects_absolute(self) -> None:
        """Test that expected artifacts reject absolute paths."""
        with pytest.raises(ValidationError) as exc_info:
            Task(description="Test", expected_artifacts=["/etc/passwd"])
        assert "invalid" in str(exc_info.value).lower()

    def test_description_max_length(self) -> None:
        """Test that description respects max length."""
        # Should succeed at reasonable length
        Task(description="x" * 10000)

    def test_error_message_max_length(self) -> None:
        """Test that error messages are limited."""
        # Error messages should be accepted up to limit
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.FAILED,
            error="x" * 5000,
        )
        assert len(response.error) == 5000

    def test_reference_path_rejects_null_bytes(self) -> None:
        """Test that reference path rejects null bytes."""
        with pytest.raises(ValidationError) as exc_info:
            ArtifactReference(path="s3://bucket/file\x00.dat")
        assert "null" in str(exc_info.value).lower()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_large_input_data(self) -> None:
        """Test handling of large input data."""
        large_data = {"items": list(range(10000)), "text": "x" * 100000}
        task = Task(description="Process large data", input_data=large_data)
        assert len(task.input_data["items"]) == 10000
        assert len(task.input_data["text"]) == 100000

    def test_large_payload(self) -> None:
        """Test event with large payload."""
        large_payload = {"data": "x" * 100000, "numbers": list(range(1000))}
        event = ATPEvent(
            task_id="test",
            sequence=0,
            event_type=EventType.PROGRESS,
            payload=large_payload,
        )
        assert len(event.payload["data"]) == 100000

    def test_unicode_content(self) -> None:
        """Test handling of unicode content."""
        task = Task(
            description="Процесс данных с 中文 и émojis 🚀",
            input_data={"text": "Hello 世界 🌍"},
        )
        assert "🚀" in task.description
        assert "世界" in task.input_data["text"]

    def test_nested_structures(self) -> None:
        """Test deeply nested data structures."""
        nested = {
            "level1": {
                "level2": {
                    "level3": {"level4": {"level5": {"value": "deep"}}},
                },
            },
        }
        artifact = ArtifactStructured(name="nested", data=nested)
        assert (
            artifact.data["level1"]["level2"]["level3"]["level4"]["level5"]["value"]
            == "deep"
        )

    def test_empty_collections(self) -> None:
        """Test handling of empty collections."""
        request = ATPRequest(
            task_id="test",
            task=Task(
                description="Test",
                input_data={},
                expected_artifacts=[],
            ),
            constraints={},
        )
        assert request.task.input_data == {}
        assert request.task.expected_artifacts == []
        assert request.constraints == {}

    def test_null_values(self) -> None:
        """Test handling of null/None values."""
        response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            metrics=None,
            error=None,
            trace_id=None,
        )
        assert response.metrics is None
        assert response.error is None
        assert response.trace_id is None

    def test_special_characters_in_strings(self) -> None:
        """Test special characters in string fields."""
        task = Task(
            description=(
                "Task with \"quotes\" and 'apostrophes' and\nnewlines\tand\ttabs"
            )
        )
        assert '"quotes"' in task.description
        assert "\n" in task.description

    def test_zero_values(self) -> None:
        """Test zero values in numeric fields."""
        metrics = Metrics(
            total_tokens=0,
            total_steps=0,
            wall_time_seconds=0.0,
            cost_usd=0.0,
        )
        assert metrics.total_tokens == 0
        assert metrics.wall_time_seconds == 0.0

    def test_max_safe_integer(self) -> None:
        """Test handling of very large integers."""
        metrics = Metrics(
            total_tokens=9007199254740991,  # JavaScript MAX_SAFE_INTEGER
            total_steps=1000000000,
        )
        assert metrics.total_tokens == 9007199254740991

    def test_float_precision(self) -> None:
        """Test float precision handling."""
        metrics = Metrics(
            wall_time_seconds=123.456789012345,
            cost_usd=0.001234567890,
        )
        assert isinstance(metrics.wall_time_seconds, float)
        assert isinstance(metrics.cost_usd, float)
