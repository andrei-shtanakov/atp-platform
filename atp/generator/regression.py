"""Regression test generator from recorded agent interactions.

This module provides functionality to:
1. Record agent interactions during test runs
2. Generate YAML test suites from recordings
3. Parameterize recorded tests for reuse
4. Anonymize sensitive data in recordings
5. Deduplicate similar tests
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from atp.generator.core import TestGenerator, TestSuiteData
from atp.loader.models import (
    Assertion,
    Constraints,
    TaskDefinition,
    TestDefinition,
)
from atp.protocol.models import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
    ResponseStatus,
)


class AnonymizationLevel(str, Enum):
    """Level of data anonymization."""

    NONE = "none"
    BASIC = "basic"  # Emails, phone numbers, common PII
    STRICT = "strict"  # All potentially sensitive data


class RecordingStatus(str, Enum):
    """Status of a recording session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class RecordedEvent(BaseModel):
    """A single recorded event from agent interaction."""

    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    sequence: int = 0

    def to_atp_event(self, task_id: str) -> ATPEvent:
        """Convert to ATP event."""
        return ATPEvent(
            task_id=task_id,
            timestamp=self.timestamp,
            sequence=self.sequence,
            event_type=self.event_type,
            payload=self.payload,
        )


class Recording(BaseModel):
    """A complete recording of an agent interaction session."""

    id: str = Field(..., description="Unique recording identifier")
    name: str | None = Field(None, description="Human-readable name")
    timestamp: datetime = Field(default_factory=datetime.now)
    status: RecordingStatus = RecordingStatus.ACTIVE
    request: ATPRequest | None = None
    response: ATPResponse | None = None
    events: list[RecordedEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_event(
        self,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> None:
        """Add an event to the recording."""
        event = RecordedEvent(
            event_type=event_type,
            payload=payload,
            sequence=len(self.events),
        )
        self.events.append(event)

    def complete(self, response: ATPResponse) -> None:
        """Mark recording as completed with response."""
        self.response = response
        self.status = RecordingStatus.COMPLETED

    def fail(self, error: str) -> None:
        """Mark recording as failed."""
        self.status = RecordingStatus.FAILED
        self.metadata["error"] = error

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """Extract tool calls from events."""
        tool_calls = []
        for event in self.events:
            if event.event_type == EventType.TOOL_CALL:
                tool_calls.append(event.payload)
        return tool_calls

    def get_content_hash(self) -> str:
        """Generate a hash for deduplication based on key content."""
        content_parts = []

        # Include task description
        if self.request:
            content_parts.append(self.request.task.description)
            if self.request.task.input_data:
                content_parts.append(str(sorted(self.request.task.input_data.items())))

        # Include tool call sequence (tool names only for similarity)
        tool_calls = self.get_tool_calls()
        tool_sequence = [tc.get("tool", "") for tc in tool_calls]
        content_parts.append("|".join(tool_sequence))

        # Include response status
        if self.response:
            content_parts.append(self.response.status.value)

        content_str = "::".join(content_parts)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]


@dataclass
class RecordingSession:
    """Manages a recording session that can capture multiple interactions."""

    session_id: str
    recordings: list[Recording] = field(default_factory=list)
    active_recording: Recording | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def start_recording(self, request: ATPRequest, name: str | None = None) -> str:
        """Start a new recording for a request.

        Args:
            request: The ATP request to record.
            name: Optional name for the recording.

        Returns:
            Recording ID.
        """
        recording_id = f"{self.session_id}-{len(self.recordings) + 1:03d}"
        recording = Recording(
            id=recording_id,
            name=name or f"Recording {len(self.recordings) + 1}",
            request=request,
        )
        self.recordings.append(recording)
        self.active_recording = recording
        return recording_id

    def record_event(self, event_type: EventType, payload: dict[str, Any]) -> None:
        """Record an event in the active recording."""
        if self.active_recording is None:
            raise RuntimeError("No active recording. Call start_recording first.")
        self.active_recording.add_event(event_type, payload)

    def complete_recording(self, response: ATPResponse) -> None:
        """Complete the active recording with a response."""
        if self.active_recording is None:
            raise RuntimeError("No active recording to complete.")
        self.active_recording.complete(response)
        self.active_recording = None

    def fail_recording(self, error: str) -> None:
        """Mark active recording as failed."""
        if self.active_recording is None:
            raise RuntimeError("No active recording to fail.")
        self.active_recording.fail(error)
        self.active_recording = None

    def get_recordings(self) -> list[Recording]:
        """Get all completed recordings."""
        return [r for r in self.recordings if r.status == RecordingStatus.COMPLETED]


class DataAnonymizer:
    """Anonymizes sensitive data in recordings."""

    # Common PII patterns - ordered from most specific to least specific
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    # Phone: require boundary or punctuation before and full 10-digit format
    PHONE_PATTERN = re.compile(
        r"(?<![a-zA-Z0-9_-])(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?![0-9])"
    )
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    # Credit card pattern - more specific with word boundaries
    CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b|\b\d{16}\b")
    IP_ADDRESS_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
    # API key pattern - match secret/key prefixes with underscore separator
    API_KEY_PATTERN = re.compile(
        r"\b(sk_|pk_|api_|key_|token_|secret_|auth_)[a-zA-Z0-9_]{8,}\b",
        re.IGNORECASE,
    )

    # Replacement placeholders
    REPLACEMENTS = {
        "email": "{EMAIL}",
        "phone": "{PHONE}",
        "ssn": "{SSN}",
        "credit_card": "{CREDIT_CARD}",
        "ip_address": "{IP_ADDRESS}",
        "api_key": "{API_KEY}",
    }

    def __init__(
        self,
        level: AnonymizationLevel = AnonymizationLevel.BASIC,
        custom_patterns: dict[str, re.Pattern[str]] | None = None,
    ) -> None:
        """Initialize anonymizer.

        Args:
            level: Level of anonymization to apply.
            custom_patterns: Additional patterns to anonymize.
        """
        self.level = level
        self.custom_patterns = custom_patterns or {}
        self._counter: dict[str, int] = {}

    def _get_replacement(self, pattern_name: str) -> str:
        """Get a unique replacement for a pattern."""
        self._counter[pattern_name] = self._counter.get(pattern_name, 0) + 1
        base = self.REPLACEMENTS.get(pattern_name, f"{{{pattern_name.upper()}}}")
        return f"{base[:-1]}_{self._counter[pattern_name]}}}"

    def anonymize_string(self, text: str) -> str:
        """Anonymize a single string value."""
        if self.level == AnonymizationLevel.NONE:
            return text

        result = text

        # Basic patterns
        if self.level in (AnonymizationLevel.BASIC, AnonymizationLevel.STRICT):
            result = self.EMAIL_PATTERN.sub(
                lambda _: self._get_replacement("email"), result
            )
            result = self.PHONE_PATTERN.sub(
                lambda _: self._get_replacement("phone"), result
            )
            result = self.API_KEY_PATTERN.sub(
                lambda _: self._get_replacement("api_key"), result
            )

        # Strict patterns
        if self.level == AnonymizationLevel.STRICT:
            result = self.SSN_PATTERN.sub(
                lambda _: self._get_replacement("ssn"), result
            )
            result = self.CREDIT_CARD_PATTERN.sub(
                lambda _: self._get_replacement("credit_card"), result
            )
            result = self.IP_ADDRESS_PATTERN.sub(
                lambda _: self._get_replacement("ip_address"), result
            )

        # Custom patterns
        for name, pattern in self.custom_patterns.items():
            result = pattern.sub(lambda _: self._get_replacement(name), result)

        return result

    def anonymize_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively anonymize a dictionary."""
        if self.level == AnonymizationLevel.NONE:
            return data

        result: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.anonymize_string(value)
            elif isinstance(value, dict):
                result[key] = self.anonymize_dict(value)
            elif isinstance(value, list):
                result[key] = self.anonymize_list(value)
            else:
                result[key] = value
        return result

    def anonymize_list(self, data: list[Any]) -> list[Any]:
        """Recursively anonymize a list."""
        if self.level == AnonymizationLevel.NONE:
            return data

        result: list[Any] = []
        for item in data:
            if isinstance(item, str):
                result.append(self.anonymize_string(item))
            elif isinstance(item, dict):
                result.append(self.anonymize_dict(item))
            elif isinstance(item, list):
                result.append(self.anonymize_list(item))
            else:
                result.append(item)
        return result

    def anonymize_recording(self, recording: Recording) -> Recording:
        """Anonymize all data in a recording."""
        if self.level == AnonymizationLevel.NONE:
            return recording

        # Create a copy for modification
        data = recording.model_dump()

        # Anonymize request
        if data.get("request"):
            if data["request"].get("task"):
                task = data["request"]["task"]
                if task.get("description"):
                    task["description"] = self.anonymize_string(task["description"])
                if task.get("input_data"):
                    task["input_data"] = self.anonymize_dict(task["input_data"])

            if data["request"].get("context"):
                context = data["request"]["context"]
                if context.get("environment"):
                    context["environment"] = self.anonymize_dict(context["environment"])

            if data["request"].get("metadata"):
                data["request"]["metadata"] = self.anonymize_dict(
                    data["request"]["metadata"]
                )

        # Anonymize events
        if data.get("events"):
            for event in data["events"]:
                if event.get("payload"):
                    event["payload"] = self.anonymize_dict(event["payload"])

        # Anonymize response
        if data.get("response"):
            if data["response"].get("error"):
                data["response"]["error"] = self.anonymize_string(
                    data["response"]["error"]
                )

        # Anonymize metadata
        if data.get("metadata"):
            data["metadata"] = self.anonymize_dict(data["metadata"])

        return Recording(**data)


class TestDeduplicator:
    """Detects and removes duplicate or very similar tests."""

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        compare_tool_sequence: bool = True,
        compare_response_status: bool = True,
    ) -> None:
        """Initialize deduplicator.

        Args:
            similarity_threshold: Minimum similarity ratio to consider duplicates.
            compare_tool_sequence: Include tool sequence in similarity comparison.
            compare_response_status: Include response status in similarity comparison.
        """
        self.similarity_threshold = similarity_threshold
        self.compare_tool_sequence = compare_tool_sequence
        self.compare_response_status = compare_response_status
        self._seen_hashes: set[str] = set()

    def _compute_similarity(
        self, recording1: Recording, recording2: Recording
    ) -> float:
        """Compute similarity between two recordings."""
        score = 0.0
        weights_sum = 0.0

        # Compare task descriptions
        if recording1.request and recording2.request:
            desc1 = recording1.request.task.description.lower()
            desc2 = recording2.request.task.description.lower()

            # Simple word overlap similarity
            words1 = set(desc1.split())
            words2 = set(desc2.split())
            if words1 or words2:
                overlap = len(words1 & words2)
                union = len(words1 | words2)
                desc_similarity = overlap / union if union > 0 else 0
                score += desc_similarity * 0.5
                weights_sum += 0.5

        # Compare tool sequences
        if self.compare_tool_sequence:
            tools1 = [tc.get("tool", "") for tc in recording1.get_tool_calls()]
            tools2 = [tc.get("tool", "") for tc in recording2.get_tool_calls()]

            if tools1 or tools2:
                # Sequence similarity based on common subsequence
                common = 0
                for t1 in tools1:
                    if t1 in tools2:
                        common += 1
                total = max(len(tools1), len(tools2))
                tool_similarity = common / total if total > 0 else 1.0
                score += tool_similarity * 0.3
                weights_sum += 0.3

        # Compare response status
        if self.compare_response_status:
            if recording1.response and recording2.response:
                status_match = (
                    1.0
                    if recording1.response.status == recording2.response.status
                    else 0.0
                )
                score += status_match * 0.2
                weights_sum += 0.2

        return score / weights_sum if weights_sum > 0 else 0.0

    def is_duplicate(self, recording: Recording) -> bool:
        """Check if a recording is a duplicate of previously seen recordings.

        Args:
            recording: Recording to check.

        Returns:
            True if the recording is considered a duplicate.
        """
        content_hash = recording.get_content_hash()
        if content_hash in self._seen_hashes:
            return True
        self._seen_hashes.add(content_hash)
        return False

    def deduplicate(self, recordings: list[Recording]) -> list[Recording]:
        """Remove duplicate recordings from a list.

        Args:
            recordings: List of recordings to deduplicate.

        Returns:
            List of unique recordings.
        """
        unique_recordings: list[Recording] = []
        self._seen_hashes.clear()

        for recording in recordings:
            if not self.is_duplicate(recording):
                # Also check similarity against existing unique recordings
                is_similar = False
                for existing in unique_recordings:
                    similarity = self._compute_similarity(recording, existing)
                    if similarity >= self.similarity_threshold:
                        is_similar = True
                        break

                if not is_similar:
                    unique_recordings.append(recording)

        return unique_recordings


class TestParameterizer:
    """Extracts and parameterizes variable values from recordings."""

    # Common parameter patterns
    FILE_PATH_PATTERN = re.compile(r"[a-zA-Z0-9_./\\-]+\.(txt|json|yaml|yml|md|py|js)")
    URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")
    NUMBER_PATTERN = re.compile(r"\b\d+\b")

    def __init__(
        self,
        extract_file_paths: bool = True,
        extract_urls: bool = True,
        extract_numbers: bool = False,
        custom_extractors: dict[str, re.Pattern[str]] | None = None,
    ) -> None:
        """Initialize parameterizer.

        Args:
            extract_file_paths: Extract file paths as parameters.
            extract_urls: Extract URLs as parameters.
            extract_numbers: Extract numbers as parameters.
            custom_extractors: Additional extraction patterns.
        """
        self.extract_file_paths = extract_file_paths
        self.extract_urls = extract_urls
        self.extract_numbers = extract_numbers
        self.custom_extractors = custom_extractors or {}

    def extract_parameters(self, text: str) -> dict[str, list[str]]:
        """Extract potential parameters from text.

        Args:
            text: Text to extract parameters from.

        Returns:
            Dictionary mapping parameter types to found values.
        """
        parameters: dict[str, list[str]] = {}

        if self.extract_file_paths:
            # File extension is captured, reconstruct full matches
            full_paths = []
            for match in self.FILE_PATH_PATTERN.finditer(text):
                full_paths.append(match.group(0))
            if full_paths:
                parameters["file_path"] = list(set(full_paths))

        if self.extract_urls:
            urls = self.URL_PATTERN.findall(text)
            if urls:
                parameters["url"] = list(set(urls))

        if self.extract_numbers:
            numbers = self.NUMBER_PATTERN.findall(text)
            if numbers:
                parameters["number"] = list(set(numbers))

        for name, pattern in self.custom_extractors.items():
            matches = pattern.findall(text)
            if matches:
                parameters[name] = list(set(matches))

        return parameters

    def parameterize_recording(
        self, recording: Recording
    ) -> tuple[Recording, dict[str, Any]]:
        """Convert literal values to parameters in a recording.

        Args:
            recording: Recording to parameterize.

        Returns:
            Tuple of (parameterized recording, extracted parameters).
        """
        all_parameters: dict[str, Any] = {}

        if recording.request is None:
            return recording, all_parameters

        # Extract from task description
        task_desc = recording.request.task.description
        params = self.extract_parameters(task_desc)

        # Create parameterized description
        parameterized_desc = task_desc
        param_counter: dict[str, int] = {}

        for param_type, values in params.items():
            for value in values:
                param_counter[param_type] = param_counter.get(param_type, 0) + 1
                param_name = f"{param_type}_{param_counter[param_type]}"
                parameterized_desc = parameterized_desc.replace(
                    value, f"{{{param_name}}}"
                )
                all_parameters[param_name] = value

        # Create modified recording
        data = recording.model_dump()
        if data.get("request") and data["request"].get("task"):
            data["request"]["task"]["description"] = parameterized_desc

        return Recording(**data), all_parameters


class RegressionTestGenerator:
    """Generates regression tests from recorded agent interactions."""

    def __init__(
        self,
        anonymization_level: AnonymizationLevel = AnonymizationLevel.BASIC,
        similarity_threshold: float = 0.8,
        extract_parameters: bool = True,
    ) -> None:
        """Initialize regression test generator.

        Args:
            anonymization_level: Level of data anonymization to apply.
            similarity_threshold: Threshold for deduplication.
            extract_parameters: Whether to extract parameters from recordings.
        """
        self.anonymizer = DataAnonymizer(level=anonymization_level)
        self.deduplicator = TestDeduplicator(similarity_threshold=similarity_threshold)
        self.parameterizer = TestParameterizer() if extract_parameters else None
        self.generator = TestGenerator()

    def _recording_to_test(
        self,
        recording: Recording,
        test_id: str,
        tags: list[str] | None = None,
    ) -> tuple[TestDefinition, dict[str, Any]]:
        """Convert a recording to a test definition.

        Args:
            recording: Recording to convert.
            test_id: ID for the generated test.
            tags: Optional tags to add to the test.

        Returns:
            Tuple of (test definition, extracted parameters).
        """
        parameters: dict[str, Any] = {}

        # Parameterize if enabled
        if self.parameterizer:
            recording, parameters = self.parameterizer.parameterize_recording(recording)

        # Anonymize the recording
        recording = self.anonymizer.anonymize_recording(recording)

        # Build test definition
        if recording.request is None:
            raise ValueError("Recording has no request")

        task_desc = recording.request.task.description
        input_data = recording.request.task.input_data
        expected_artifacts = recording.request.task.expected_artifacts

        # Extract constraints from request
        constraints_data = recording.request.constraints
        constraints = Constraints(
            max_steps=constraints_data.get("max_steps"),
            max_tokens=constraints_data.get("max_tokens"),
            timeout_seconds=constraints_data.get("timeout_seconds", 300),
            allowed_tools=constraints_data.get("allowed_tools"),
            budget_usd=constraints_data.get("budget_usd"),
        )

        # Build assertions based on response
        assertions: list[Assertion] = []

        if recording.response:
            # Assert expected status
            if recording.response.status == ResponseStatus.COMPLETED:
                assertions.append(
                    Assertion(
                        type="behavior",
                        config={"check": "completed_successfully"},
                    )
                )

            # Assert expected artifacts
            for artifact in recording.response.artifacts:
                if hasattr(artifact, "path"):
                    assertions.append(
                        Assertion(
                            type="artifact_exists",
                            config={"path": artifact.path},
                        )
                    )

        # Build tags - use a set to avoid duplicates and copy the input list
        all_tags: set[str] = set(tags) if tags else set()
        all_tags.add("regression")
        all_tags.add("recorded")

        # Add tool-based tags
        tool_calls = recording.get_tool_calls()
        tools_used = set(tc.get("tool", "") for tc in tool_calls if tc.get("tool"))
        for tool in tools_used:
            all_tags.add(f"uses:{tool}")

        # Create test name from recording
        test_name = recording.name or f"Regression test {test_id}"

        test = TestDefinition(
            id=test_id,
            name=test_name,
            description=f"Generated from recording {recording.id}",
            tags=sorted(all_tags),
            task=TaskDefinition(
                description=task_desc,
                input_data=input_data,
                expected_artifacts=expected_artifacts,
            ),
            constraints=constraints,
            assertions=assertions,
        )

        return test, parameters

    def generate_from_recordings(
        self,
        recordings: list[Recording],
        suite_name: str,
        suite_description: str | None = None,
        deduplicate: bool = True,
        tags: list[str] | None = None,
    ) -> tuple[TestSuiteData, dict[str, dict[str, Any]]]:
        """Generate a test suite from recordings.

        Args:
            recordings: List of recordings to convert.
            suite_name: Name for the generated suite.
            suite_description: Optional suite description.
            deduplicate: Whether to remove duplicate recordings.
            tags: Optional tags to add to all tests.

        Returns:
            Tuple of (test suite data, parameters by test ID).
        """
        # Filter to completed recordings
        completed = [r for r in recordings if r.status == RecordingStatus.COMPLETED]

        # Deduplicate if requested
        if deduplicate:
            completed = self.deduplicator.deduplicate(completed)

        # Create suite
        suite = self.generator.create_suite(
            name=suite_name,
            description=suite_description
            or f"Regression tests generated from {len(completed)} recordings",
        )

        # Convert recordings to tests
        all_parameters: dict[str, dict[str, Any]] = {}

        for recording in completed:
            test_id = self.generator.generate_test_id(suite, prefix="reg")
            test, params = self._recording_to_test(recording, test_id, tags)
            self.generator.add_test(suite, test)

            if params:
                all_parameters[test_id] = params

        return suite, all_parameters

    def generate_from_session(
        self,
        session: RecordingSession,
        suite_name: str | None = None,
        suite_description: str | None = None,
        deduplicate: bool = True,
        tags: list[str] | None = None,
    ) -> tuple[TestSuiteData, dict[str, dict[str, Any]]]:
        """Generate a test suite from a recording session.

        Args:
            session: Recording session to convert.
            suite_name: Name for the generated suite (defaults to session ID).
            suite_description: Optional suite description.
            deduplicate: Whether to remove duplicate recordings.
            tags: Optional tags to add to all tests.

        Returns:
            Tuple of (test suite data, parameters by test ID).
        """
        return self.generate_from_recordings(
            recordings=session.get_recordings(),
            suite_name=suite_name or f"regression-{session.session_id}",
            suite_description=suite_description,
            deduplicate=deduplicate,
            tags=tags,
        )

    def save_suite(
        self,
        suite: TestSuiteData,
        output_path: str | Path,
        parameters: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Save generated suite to YAML file.

        Args:
            suite: Test suite to save.
            output_path: Path to output file.
            parameters: Optional parameters to save alongside.
        """
        from atp.generator.writer import YAMLWriter

        writer = YAMLWriter()
        writer.save(suite, output_path)

        # Save parameters file if present
        if parameters:
            import yaml

            params_path = Path(output_path).with_suffix(".params.yaml")
            with open(params_path, "w") as f:
                yaml.dump(parameters, f, default_flow_style=False)

    def to_yaml(self, suite: TestSuiteData) -> str:
        """Convert suite to YAML string.

        Args:
            suite: Test suite to convert.

        Returns:
            YAML formatted string.
        """
        return self.generator.to_yaml(suite)


def create_recording_session(session_id: str | None = None) -> RecordingSession:
    """Create a new recording session.

    Args:
        session_id: Optional session ID (auto-generated if not provided).

    Returns:
        A new RecordingSession instance.
    """
    if session_id is None:
        session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return RecordingSession(session_id=session_id)


def load_recordings_from_file(file_path: str | Path) -> list[Recording]:
    """Load recordings from a JSON file.

    Args:
        file_path: Path to recordings file.

    Returns:
        List of Recording objects.
    """
    import json

    with open(file_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [Recording(**r) for r in data]
    elif isinstance(data, dict) and "recordings" in data:
        return [Recording(**r) for r in data["recordings"]]
    else:
        raise ValueError("Invalid recordings file format")


def save_recordings_to_file(recordings: list[Recording], file_path: str | Path) -> None:
    """Save recordings to a JSON file.

    Args:
        recordings: List of recordings to save.
        file_path: Path to output file.
    """
    import json

    data = [r.model_dump(mode="json") for r in recordings]

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
