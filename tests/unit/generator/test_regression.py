"""Unit tests for regression test generator."""

import json
import re
import tempfile
from pathlib import Path

import pytest

from atp.generator.regression import (
    AnonymizationLevel,
    DataAnonymizer,
    RecordedEvent,
    Recording,
    RecordingStatus,
    RegressionTestGenerator,
    TestDeduplicator,
    TestParameterizer,
    create_recording_session,
    load_recordings_from_file,
    save_recordings_to_file,
)
from atp.protocol.models import (
    ArtifactFile,
    ATPRequest,
    ATPResponse,
    EventType,
    ResponseStatus,
    Task,
)


class TestRecordedEvent:
    """Tests for RecordedEvent class."""

    def test_create_event(self) -> None:
        """Test creating a recorded event."""
        event = RecordedEvent(
            event_type=EventType.TOOL_CALL,
            payload={"tool": "read_file", "input": {"path": "test.txt"}},
            sequence=0,
        )

        assert event.event_type == EventType.TOOL_CALL
        assert event.payload["tool"] == "read_file"
        assert event.sequence == 0
        assert event.timestamp is not None

    def test_to_atp_event(self) -> None:
        """Test converting to ATP event."""
        event = RecordedEvent(
            event_type=EventType.PROGRESS,
            payload={"message": "Working..."},
            sequence=5,
        )

        atp_event = event.to_atp_event("task-123")

        assert atp_event.task_id == "task-123"
        assert atp_event.event_type == EventType.PROGRESS
        assert atp_event.payload["message"] == "Working..."
        assert atp_event.sequence == 5


class TestRecording:
    """Tests for Recording class."""

    def test_create_recording(self) -> None:
        """Test creating a recording."""
        recording = Recording(id="rec-001", name="Test Recording")

        assert recording.id == "rec-001"
        assert recording.name == "Test Recording"
        assert recording.status == RecordingStatus.ACTIVE
        assert recording.events == []
        assert recording.request is None
        assert recording.response is None

    def test_add_event(self) -> None:
        """Test adding events to a recording."""
        recording = Recording(id="rec-001")

        recording.add_event(EventType.TOOL_CALL, {"tool": "write_file"})
        recording.add_event(EventType.PROGRESS, {"message": "Done"})

        assert len(recording.events) == 2
        assert recording.events[0].event_type == EventType.TOOL_CALL
        assert recording.events[0].sequence == 0
        assert recording.events[1].event_type == EventType.PROGRESS
        assert recording.events[1].sequence == 1

    def test_complete_recording(self) -> None:
        """Test completing a recording."""
        recording = Recording(id="rec-001")
        response = ATPResponse(
            task_id="task-001",
            status=ResponseStatus.COMPLETED,
        )

        recording.complete(response)

        assert recording.status == RecordingStatus.COMPLETED
        assert recording.response == response

    def test_fail_recording(self) -> None:
        """Test failing a recording."""
        recording = Recording(id="rec-001")

        recording.fail("Connection timeout")

        assert recording.status == RecordingStatus.FAILED
        assert recording.metadata["error"] == "Connection timeout"

    def test_get_tool_calls(self) -> None:
        """Test extracting tool calls from events."""
        recording = Recording(id="rec-001")
        recording.add_event(EventType.TOOL_CALL, {"tool": "read_file", "input": {}})
        recording.add_event(EventType.PROGRESS, {"message": "Working"})
        recording.add_event(EventType.TOOL_CALL, {"tool": "write_file", "input": {}})

        tool_calls = recording.get_tool_calls()

        assert len(tool_calls) == 2
        assert tool_calls[0]["tool"] == "read_file"
        assert tool_calls[1]["tool"] == "write_file"

    def test_get_content_hash(self) -> None:
        """Test generating content hash for deduplication."""
        request = ATPRequest(
            task_id="task-001",
            task=Task(description="Write hello world"),
        )
        recording = Recording(id="rec-001", request=request)
        recording.add_event(EventType.TOOL_CALL, {"tool": "write_file"})

        hash1 = recording.get_content_hash()

        # Same content should produce same hash
        recording2 = Recording(id="rec-002", request=request)
        recording2.add_event(EventType.TOOL_CALL, {"tool": "write_file"})

        hash2 = recording2.get_content_hash()

        assert hash1 == hash2
        assert len(hash1) == 16  # SHA-256 truncated to 16 chars


class TestRecordingSession:
    """Tests for RecordingSession class."""

    def test_create_session(self) -> None:
        """Test creating a recording session."""
        session = create_recording_session("test-session")

        assert session.session_id == "test-session"
        assert session.recordings == []
        assert session.active_recording is None

    def test_create_session_auto_id(self) -> None:
        """Test creating a session with auto-generated ID."""
        session = create_recording_session()

        assert session.session_id.startswith("session-")

    def test_start_recording(self) -> None:
        """Test starting a recording."""
        session = create_recording_session("test-session")
        request = ATPRequest(
            task_id="task-001",
            task=Task(description="Test task"),
        )

        recording_id = session.start_recording(request, "My Recording")

        assert recording_id == "test-session-001"
        assert len(session.recordings) == 1
        assert session.active_recording is not None
        assert session.active_recording.name == "My Recording"
        assert session.active_recording.request == request

    def test_record_event(self) -> None:
        """Test recording an event."""
        session = create_recording_session("test-session")
        request = ATPRequest(
            task_id="task-001",
            task=Task(description="Test"),
        )
        session.start_recording(request)

        session.record_event(EventType.TOOL_CALL, {"tool": "test_tool"})

        assert session.active_recording is not None
        assert len(session.active_recording.events) == 1

    def test_record_event_without_active_raises(self) -> None:
        """Test that recording without active raises error."""
        session = create_recording_session("test-session")

        with pytest.raises(RuntimeError, match="No active recording"):
            session.record_event(EventType.TOOL_CALL, {})

    def test_complete_recording(self) -> None:
        """Test completing a recording."""
        session = create_recording_session("test-session")
        request = ATPRequest(
            task_id="task-001",
            task=Task(description="Test"),
        )
        session.start_recording(request)

        response = ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        session.complete_recording(response)

        assert session.active_recording is None
        assert session.recordings[0].status == RecordingStatus.COMPLETED

    def test_get_recordings_only_completed(self) -> None:
        """Test that get_recordings returns only completed recordings."""
        session = create_recording_session("test-session")

        # Add a completed recording
        request1 = ATPRequest(task_id="task-001", task=Task(description="Test 1"))
        session.start_recording(request1)
        session.complete_recording(
            ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        )

        # Add a failed recording
        request2 = ATPRequest(task_id="task-002", task=Task(description="Test 2"))
        session.start_recording(request2)
        session.fail_recording("Error")

        # Add active recording
        request3 = ATPRequest(task_id="task-003", task=Task(description="Test 3"))
        session.start_recording(request3)

        completed = session.get_recordings()

        assert len(completed) == 1
        assert completed[0].request is not None
        assert completed[0].request.task_id == "task-001"


class TestDataAnonymizer:
    """Tests for DataAnonymizer class."""

    def test_anonymize_none_level(self) -> None:
        """Test that none level returns data unchanged."""
        anonymizer = DataAnonymizer(level=AnonymizationLevel.NONE)
        text = "Email: test@example.com, Phone: 555-123-4567"

        result = anonymizer.anonymize_string(text)

        assert result == text

    def test_anonymize_email_basic(self) -> None:
        """Test email anonymization at basic level."""
        anonymizer = DataAnonymizer(level=AnonymizationLevel.BASIC)
        text = "Contact: user@example.com and admin@test.org"

        result = anonymizer.anonymize_string(text)

        assert "@" not in result
        assert "{EMAIL_1}" in result
        assert "{EMAIL_2}" in result

    def test_anonymize_phone_basic(self) -> None:
        """Test phone anonymization at basic level."""
        anonymizer = DataAnonymizer(level=AnonymizationLevel.BASIC)
        text = "Call me at (555) 123-4567"

        result = anonymizer.anonymize_string(text)

        assert "(555) 123-4567" not in result
        assert "{PHONE_1}" in result

    def test_anonymize_api_key_basic(self) -> None:
        """Test API key anonymization at basic level."""
        anonymizer = DataAnonymizer(level=AnonymizationLevel.BASIC)
        text = "API key: sk_live_abcdefghij1234"

        result = anonymizer.anonymize_string(text)

        assert "sk_live_abcdefghij1234" not in result
        assert "{API_KEY_1}" in result

    def test_anonymize_ssn_strict(self) -> None:
        """Test SSN anonymization at strict level."""
        anonymizer = DataAnonymizer(level=AnonymizationLevel.STRICT)
        text = "SSN: 123-45-6789"

        result = anonymizer.anonymize_string(text)

        assert "123-45-6789" not in result
        assert "{SSN_1}" in result

    def test_anonymize_credit_card_strict(self) -> None:
        """Test credit card anonymization at strict level."""
        anonymizer = DataAnonymizer(level=AnonymizationLevel.STRICT)
        text = "Card: 4111 1111 1111 1111"

        result = anonymizer.anonymize_string(text)

        assert "4111 1111 1111 1111" not in result
        assert "{CREDIT_CARD_1}" in result

    def test_anonymize_dict(self) -> None:
        """Test dictionary anonymization."""
        anonymizer = DataAnonymizer(level=AnonymizationLevel.BASIC)
        data = {
            "email": "test@example.com",
            "nested": {
                "phone": "555-123-4567",
            },
            "number": 42,
        }

        result = anonymizer.anonymize_dict(data)

        assert "{EMAIL_1}" in result["email"]
        assert "{PHONE_1}" in result["nested"]["phone"]
        assert result["number"] == 42

    def test_anonymize_list(self) -> None:
        """Test list anonymization."""
        anonymizer = DataAnonymizer(level=AnonymizationLevel.BASIC)
        data = ["test@example.com", "regular text", {"email": "other@test.com"}]

        result = anonymizer.anonymize_list(data)

        assert "{EMAIL_" in result[0]
        assert result[1] == "regular text"
        assert "{EMAIL_" in result[2]["email"]

    def test_anonymize_recording(self) -> None:
        """Test recording anonymization."""
        anonymizer = DataAnonymizer(level=AnonymizationLevel.BASIC)
        request = ATPRequest(
            task_id="task-001",
            task=Task(description="Send email to user@example.com"),
        )
        recording = Recording(id="rec-001", request=request)

        result = anonymizer.anonymize_recording(recording)

        assert result.request is not None
        assert "user@example.com" not in result.request.task.description
        assert "{EMAIL_1}" in result.request.task.description

    def test_custom_patterns(self) -> None:
        """Test custom anonymization patterns."""

        custom = {"project_id": re.compile(r"proj_[a-z0-9]+")}
        anonymizer = DataAnonymizer(
            level=AnonymizationLevel.BASIC, custom_patterns=custom
        )
        text = "Project: proj_abc123xyz"

        result = anonymizer.anonymize_string(text)

        assert "proj_abc123xyz" not in result
        assert "{PROJECT_ID_1}" in result


class TestTestDeduplicator:
    """Tests for TestDeduplicator class."""

    def _create_recording(
        self, task_id: str, description: str, tools: list[str] | None = None
    ) -> Recording:
        """Helper to create a test recording."""
        request = ATPRequest(
            task_id=task_id,
            task=Task(description=description),
        )
        recording = Recording(id=f"rec-{task_id}", request=request)
        for tool in tools or []:
            recording.add_event(EventType.TOOL_CALL, {"tool": tool})
        recording.complete(
            ATPResponse(task_id=task_id, status=ResponseStatus.COMPLETED)
        )
        return recording

    def test_is_duplicate_exact(self) -> None:
        """Test detecting exact duplicates by hash."""
        deduplicator = TestDeduplicator()

        rec1 = self._create_recording("task-001", "Write hello", ["write_file"])
        rec2 = self._create_recording("task-002", "Write hello", ["write_file"])

        assert not deduplicator.is_duplicate(rec1)
        assert deduplicator.is_duplicate(rec2)  # Same content hash

    def test_is_duplicate_different(self) -> None:
        """Test that different recordings are not duplicates."""
        deduplicator = TestDeduplicator()

        rec1 = self._create_recording("task-001", "Write hello", ["write_file"])
        rec2 = self._create_recording("task-002", "Read file", ["read_file"])

        assert not deduplicator.is_duplicate(rec1)
        assert not deduplicator.is_duplicate(rec2)

    def test_deduplicate_removes_duplicates(self) -> None:
        """Test deduplication removes duplicate recordings."""
        deduplicator = TestDeduplicator()

        recordings = [
            self._create_recording("task-001", "Write hello", ["write_file"]),
            self._create_recording("task-002", "Write hello", ["write_file"]),
            self._create_recording("task-003", "Read data", ["read_file"]),
        ]

        unique = deduplicator.deduplicate(recordings)

        assert len(unique) == 2

    def test_similarity_threshold(self) -> None:
        """Test similarity threshold for fuzzy deduplication."""
        # Use high threshold to catch similar recordings
        deduplicator = TestDeduplicator(similarity_threshold=0.5)

        recordings = [
            self._create_recording(
                "task-001", "Write hello world file", ["write_file"]
            ),
            self._create_recording(
                "task-002", "Write goodbye world file", ["write_file"]
            ),
        ]

        unique = deduplicator.deduplicate(recordings)

        # With high similarity (same tools, similar description), one should be removed
        assert len(unique) == 1


class TestTestParameterizer:
    """Tests for TestParameterizer class."""

    def test_extract_file_paths(self) -> None:
        """Test extracting file paths."""
        parameterizer = TestParameterizer(extract_file_paths=True)
        text = "Create output.json and save to report.md"

        params = parameterizer.extract_parameters(text)

        assert "file_path" in params
        assert "output.json" in params["file_path"]
        assert "report.md" in params["file_path"]

    def test_extract_urls(self) -> None:
        """Test extracting URLs."""
        parameterizer = TestParameterizer(extract_urls=True)
        text = "Fetch data from https://api.example.com/data"

        params = parameterizer.extract_parameters(text)

        assert "url" in params
        assert "https://api.example.com/data" in params["url"]

    def test_extract_numbers(self) -> None:
        """Test extracting numbers."""
        parameterizer = TestParameterizer(extract_numbers=True)
        text = "Process 100 items with batch size 10"

        params = parameterizer.extract_parameters(text)

        assert "number" in params
        assert "100" in params["number"]
        assert "10" in params["number"]

    def test_parameterize_recording(self) -> None:
        """Test parameterizing a recording."""
        parameterizer = TestParameterizer()
        request = ATPRequest(
            task_id="task-001",
            task=Task(description="Save results to output.json"),
        )
        recording = Recording(id="rec-001", request=request)

        param_recording, params = parameterizer.parameterize_recording(recording)

        assert param_recording.request is not None
        assert "{file_path_1}" in param_recording.request.task.description
        assert params["file_path_1"] == "output.json"

    def test_custom_extractors(self) -> None:
        """Test custom parameter extractors."""

        custom = {"version": re.compile(r"v\d+\.\d+\.\d+")}
        parameterizer = TestParameterizer(custom_extractors=custom)
        text = "Upgrade to v2.0.1"

        params = parameterizer.extract_parameters(text)

        assert "version" in params
        assert "v2.0.1" in params["version"]


class TestRegressionTestGenerator:
    """Tests for RegressionTestGenerator class."""

    def _create_completed_recording(self) -> Recording:
        """Helper to create a completed recording."""
        request = ATPRequest(
            task_id="task-001",
            task=Task(
                description="Write a file named output.txt with content 'Hello'",
                expected_artifacts=["output.txt"],
            ),
            constraints={"max_steps": 5, "timeout_seconds": 60},
        )
        recording = Recording(id="rec-001", name="Test Recording", request=request)
        recording.add_event(EventType.TOOL_CALL, {"tool": "write_file"})
        recording.complete(
            ATPResponse(
                task_id="task-001",
                status=ResponseStatus.COMPLETED,
                artifacts=[ArtifactFile(path="output.txt", content="Hello")],
            )
        )
        return recording

    def test_generate_from_recordings_basic(self) -> None:
        """Test basic test generation from recordings."""
        generator = RegressionTestGenerator(
            anonymization_level=AnonymizationLevel.NONE,
            extract_parameters=False,
        )
        recordings = [self._create_completed_recording()]

        suite, params = generator.generate_from_recordings(
            recordings=recordings,
            suite_name="test-suite",
            suite_description="Test suite",
        )

        assert suite.name == "test-suite"
        assert len(suite.tests) == 1
        test = suite.tests[0]
        assert test.id.startswith("reg-")
        assert "regression" in test.tags
        assert "recorded" in test.tags
        assert "Write a file" in test.task.description

    def test_generate_with_anonymization(self) -> None:
        """Test test generation with anonymization."""
        generator = RegressionTestGenerator(
            anonymization_level=AnonymizationLevel.BASIC,
            extract_parameters=False,
        )
        request = ATPRequest(
            task_id="task-001",
            task=Task(description="Send email to user@example.com"),
        )
        recording = Recording(id="rec-001", request=request)
        recording.complete(
            ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        )

        suite, _ = generator.generate_from_recordings(
            recordings=[recording],
            suite_name="test-suite",
        )

        assert "user@example.com" not in suite.tests[0].task.description
        assert "{EMAIL_" in suite.tests[0].task.description

    def test_generate_with_parameterization(self) -> None:
        """Test test generation with parameterization."""
        generator = RegressionTestGenerator(
            anonymization_level=AnonymizationLevel.NONE,
            extract_parameters=True,
        )
        request = ATPRequest(
            task_id="task-001",
            task=Task(description="Create file output.json with data"),
        )
        recording = Recording(id="rec-001", request=request)
        recording.complete(
            ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        )

        suite, params = generator.generate_from_recordings(
            recordings=[recording],
            suite_name="test-suite",
        )

        # Parameters should be extracted
        assert len(params) > 0
        test_id = suite.tests[0].id
        assert test_id in params
        assert "file_path_1" in params[test_id]

    def test_generate_with_deduplication(self) -> None:
        """Test test generation with deduplication."""
        generator = RegressionTestGenerator(
            anonymization_level=AnonymizationLevel.NONE,
            extract_parameters=False,
        )

        # Create duplicate recordings
        recordings = []
        for i in range(3):
            request = ATPRequest(
                task_id=f"task-{i:03d}",
                task=Task(description="Write hello world"),
            )
            recording = Recording(id=f"rec-{i:03d}", request=request)
            recording.add_event(EventType.TOOL_CALL, {"tool": "write_file"})
            recording.complete(
                ATPResponse(task_id=f"task-{i:03d}", status=ResponseStatus.COMPLETED)
            )
            recordings.append(recording)

        suite, _ = generator.generate_from_recordings(
            recordings=recordings,
            suite_name="test-suite",
            deduplicate=True,
        )

        # Should deduplicate to 1 test
        assert len(suite.tests) == 1

    def test_generate_assertions(self) -> None:
        """Test that assertions are generated from response."""
        generator = RegressionTestGenerator(
            anonymization_level=AnonymizationLevel.NONE,
            extract_parameters=False,
        )
        recording = self._create_completed_recording()

        suite, _ = generator.generate_from_recordings(
            recordings=[recording],
            suite_name="test-suite",
        )

        test = suite.tests[0]
        # Should have behavior assertion for completed status
        behavior_assertions = [a for a in test.assertions if a.type == "behavior"]
        assert len(behavior_assertions) > 0

        # Should have artifact_exists assertion
        artifact_assertions = [
            a for a in test.assertions if a.type == "artifact_exists"
        ]
        assert len(artifact_assertions) > 0

    def test_generate_with_tags(self) -> None:
        """Test adding custom tags to generated tests."""
        generator = RegressionTestGenerator(
            anonymization_level=AnonymizationLevel.NONE,
            extract_parameters=False,
        )
        recording = self._create_completed_recording()

        suite, _ = generator.generate_from_recordings(
            recordings=[recording],
            suite_name="test-suite",
            tags=["ci", "smoke"],
        )

        test = suite.tests[0]
        assert "ci" in test.tags
        assert "smoke" in test.tags
        assert "regression" in test.tags

    def test_generate_tool_tags(self) -> None:
        """Test that tool-based tags are added."""
        generator = RegressionTestGenerator(
            anonymization_level=AnonymizationLevel.NONE,
            extract_parameters=False,
        )
        recording = self._create_completed_recording()

        suite, _ = generator.generate_from_recordings(
            recordings=[recording],
            suite_name="test-suite",
        )

        test = suite.tests[0]
        assert "uses:write_file" in test.tags

    def test_generate_from_session(self) -> None:
        """Test generating from a recording session."""
        generator = RegressionTestGenerator(
            anonymization_level=AnonymizationLevel.NONE,
            extract_parameters=False,
        )

        session = create_recording_session("test-session")
        request = ATPRequest(
            task_id="task-001",
            task=Task(description="Test task"),
        )
        session.start_recording(request, "Test Recording")
        session.complete_recording(
            ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        )

        suite, _ = generator.generate_from_session(session)

        assert suite.name == "regression-test-session"
        assert len(suite.tests) == 1

    def test_to_yaml(self) -> None:
        """Test YAML generation."""
        generator = RegressionTestGenerator(
            anonymization_level=AnonymizationLevel.NONE,
            extract_parameters=False,
        )
        recording = self._create_completed_recording()

        suite, _ = generator.generate_from_recordings(
            recordings=[recording],
            suite_name="test-suite",
        )

        yaml_content = generator.to_yaml(suite)

        assert "test_suite:" in yaml_content
        assert "test-suite" in yaml_content
        assert "regression" in yaml_content


class TestRecordingPersistence:
    """Tests for recording file operations."""

    def test_save_and_load_recordings(self) -> None:
        """Test saving and loading recordings."""
        request = ATPRequest(
            task_id="task-001",
            task=Task(description="Test task"),
        )
        recording = Recording(id="rec-001", name="Test", request=request)
        recording.add_event(EventType.TOOL_CALL, {"tool": "test_tool"})
        recording.complete(
            ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        )

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = Path(f.name)

        try:
            # Save
            save_recordings_to_file([recording], path)

            # Load
            loaded = load_recordings_from_file(path)

            assert len(loaded) == 1
            assert loaded[0].id == "rec-001"
            assert loaded[0].name == "Test"
            assert loaded[0].status == RecordingStatus.COMPLETED
            assert len(loaded[0].events) == 1
        finally:
            path.unlink()

    def test_load_recordings_array_format(self) -> None:
        """Test loading recordings from array format."""
        data = [
            {
                "id": "rec-001",
                "name": "Test 1",
                "status": "completed",
                "request": {
                    "task_id": "task-001",
                    "task": {"description": "Test"},
                },
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)

        try:
            loaded = load_recordings_from_file(path)
            assert len(loaded) == 1
            assert loaded[0].id == "rec-001"
        finally:
            path.unlink()

    def test_load_recordings_object_format(self) -> None:
        """Test loading recordings from object format with 'recordings' key."""
        data = {
            "recordings": [
                {
                    "id": "rec-001",
                    "status": "completed",
                    "request": {
                        "task_id": "task-001",
                        "task": {"description": "Test"},
                    },
                }
            ]
        }

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)

        try:
            loaded = load_recordings_from_file(path)
            assert len(loaded) == 1
        finally:
            path.unlink()

    def test_load_recordings_invalid_format(self) -> None:
        """Test loading recordings from invalid format raises error."""
        data = {"invalid": "format"}

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid recordings file format"):
                load_recordings_from_file(path)
        finally:
            path.unlink()


class TestRegressionTestGeneratorSaveOutput:
    """Tests for RegressionTestGenerator save functionality."""

    def test_save_suite(self) -> None:
        """Test saving generated suite to file."""
        generator = RegressionTestGenerator(
            anonymization_level=AnonymizationLevel.NONE,
            extract_parameters=False,
        )
        request = ATPRequest(
            task_id="task-001",
            task=Task(description="Test task"),
        )
        recording = Recording(id="rec-001", request=request)
        recording.complete(
            ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        )

        suite, params = generator.generate_from_recordings(
            recordings=[recording],
            suite_name="test-suite",
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            output_path = Path(f.name)

        try:
            generator.save_suite(suite, output_path)

            # Verify file was created
            assert output_path.exists()

            # Verify content
            content = output_path.read_text()
            assert "test_suite:" in content
            assert "test-suite" in content
        finally:
            output_path.unlink()

    def test_save_suite_with_params(self) -> None:
        """Test saving suite with parameters file."""
        generator = RegressionTestGenerator(
            anonymization_level=AnonymizationLevel.NONE,
            extract_parameters=True,
        )
        request = ATPRequest(
            task_id="task-001",
            task=Task(description="Create file output.json"),
        )
        recording = Recording(id="rec-001", request=request)
        recording.complete(
            ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        )

        suite, params = generator.generate_from_recordings(
            recordings=[recording],
            suite_name="test-suite",
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            output_path = Path(f.name)

        params_path = output_path.with_suffix(".params.yaml")

        try:
            generator.save_suite(suite, output_path, parameters=params)

            # Verify both files were created
            assert output_path.exists()
            assert params_path.exists()

            # Verify params content
            params_content = params_path.read_text()
            assert "file_path_1" in params_content
        finally:
            output_path.unlink()
            if params_path.exists():
                params_path.unlink()
