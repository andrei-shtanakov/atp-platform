"""Tests for mock tools models."""

import pytest
from pydantic import ValidationError

from atp.mock_tools.models import (
    MatchType,
    MockDefinition,
    MockResponse,
    MockTool,
    PatternMatcher,
    ToolCall,
    ToolCallRecord,
)


class TestPatternMatcher:
    """Tests for PatternMatcher model."""

    def test_any_match_always_true(self) -> None:
        """Test ANY match type always matches."""
        matcher = PatternMatcher(type=MatchType.ANY)

        assert matcher.matches(None)
        assert matcher.matches("anything")
        assert matcher.matches({"key": "value"})
        assert matcher.matches("")

    def test_exact_match_string(self) -> None:
        """Test EXACT match with string input."""
        matcher = PatternMatcher(type=MatchType.EXACT, pattern="test value")

        assert matcher.matches("test value")
        assert not matcher.matches("test")
        assert not matcher.matches("test value extra")

    def test_exact_match_with_field(self) -> None:
        """Test EXACT match with specific field."""
        matcher = PatternMatcher(
            type=MatchType.EXACT,
            pattern="expected",
            field="query",
        )

        assert matcher.matches({"query": "expected"})
        assert not matcher.matches({"query": "other"})
        assert not matcher.matches({"other_field": "expected"})

    def test_contains_match(self) -> None:
        """Test CONTAINS match type."""
        matcher = PatternMatcher(type=MatchType.CONTAINS, pattern="python")

        assert matcher.matches("learn python programming")
        assert matcher.matches("python")
        assert not matcher.matches("java programming")

    def test_contains_match_with_field(self) -> None:
        """Test CONTAINS match with specific field."""
        matcher = PatternMatcher(
            type=MatchType.CONTAINS,
            pattern="test",
            field="name",
        )

        assert matcher.matches({"name": "test_function"})
        assert matcher.matches({"name": "my_test"})
        assert not matcher.matches({"name": "function"})

    def test_regex_match(self) -> None:
        """Test REGEX match type."""
        matcher = PatternMatcher(type=MatchType.REGEX, pattern=r"^test-\d+$")

        assert matcher.matches("test-123")
        assert matcher.matches("test-1")
        assert not matcher.matches("test-abc")
        assert not matcher.matches("prefix-test-123")

    def test_regex_match_invalid_pattern(self) -> None:
        """Test REGEX match with invalid pattern returns False."""
        matcher = PatternMatcher(type=MatchType.REGEX, pattern="[invalid")

        assert not matcher.matches("anything")

    def test_matches_none_input(self) -> None:
        """Test matching with None input."""
        matcher = PatternMatcher(type=MatchType.CONTAINS, pattern="test")

        assert not matcher.matches(None)

    def test_matches_dict_converted_to_string(self) -> None:
        """Test dict input is converted to string when no field specified."""
        matcher = PatternMatcher(type=MatchType.CONTAINS, pattern="key")

        assert matcher.matches({"key": "value"})

    def test_matches_no_pattern_returns_true(self) -> None:
        """Test matcher with no pattern returns True."""
        matcher = PatternMatcher(type=MatchType.EXACT, pattern=None)

        assert matcher.matches("anything")


class TestMockResponse:
    """Tests for MockResponse model."""

    def test_default_values(self) -> None:
        """Test default values."""
        response = MockResponse()

        assert response.output is None
        assert response.error is None
        assert response.delay_ms == 0
        assert response.status == "success"

    def test_custom_values(self) -> None:
        """Test custom values."""
        response = MockResponse(
            output={"result": 42},
            error="test error",
            delay_ms=100,
            status="error",
        )

        assert response.output == {"result": 42}
        assert response.error == "test error"
        assert response.delay_ms == 100
        assert response.status == "error"

    def test_delay_ms_validation(self) -> None:
        """Test delay_ms must be non-negative."""
        with pytest.raises(ValidationError):
            MockResponse(delay_ms=-1)


class TestMockTool:
    """Tests for MockTool model."""

    def test_minimal_tool(self) -> None:
        """Test creating tool with minimal data."""
        tool = MockTool(name="test_tool")

        assert tool.name == "test_tool"
        assert tool.description is None
        assert tool.responses == []
        assert tool.default_response is not None

    def test_tool_name_required(self) -> None:
        """Test tool name is required."""
        with pytest.raises(ValidationError):
            MockTool(name="")

    def test_get_response_default(self) -> None:
        """Test get_response returns default when no matches."""
        tool = MockTool(
            name="test",
            default_response=MockResponse(output={"default": True}),
        )

        response = tool.get_response({"any": "input"})

        assert response.output == {"default": True}

    def test_get_response_matching(self) -> None:
        """Test get_response returns matching response."""
        matcher = PatternMatcher(type=MatchType.EXACT, pattern="match")
        matched_response = MockResponse(output={"matched": True})

        tool = MockTool(
            name="test",
            responses=[(matcher, matched_response)],
            default_response=MockResponse(output={"default": True}),
        )

        # Matching input
        response = tool.get_response("match")
        assert response.output == {"matched": True}

        # Non-matching input
        response = tool.get_response("no match")
        assert response.output == {"default": True}

    def test_get_response_first_match_wins(self) -> None:
        """Test get_response returns first matching response."""
        matcher1 = PatternMatcher(type=MatchType.CONTAINS, pattern="test")
        matcher2 = PatternMatcher(type=MatchType.CONTAINS, pattern="test")

        tool = MockTool(
            name="test",
            responses=[
                (matcher1, MockResponse(output={"first": True})),
                (matcher2, MockResponse(output={"second": True})),
            ],
        )

        response = tool.get_response("test")
        assert response.output == {"first": True}

    def test_responses_from_dict_format(self) -> None:
        """Test responses can be provided in dict format."""
        tool = MockTool(
            name="test",
            responses=[
                {
                    "matcher": {"type": "exact", "pattern": "test"},
                    "response": {"output": {"matched": True}},
                }
            ],
        )

        assert len(tool.responses) == 1
        matcher, response = tool.responses[0]
        assert matcher is not None
        assert matcher.type == MatchType.EXACT


class TestToolCall:
    """Tests for ToolCall model."""

    def test_minimal_call(self) -> None:
        """Test creating call with minimal data."""
        call = ToolCall(tool="test_tool")

        assert call.tool == "test_tool"
        assert call.input is None
        assert call.task_id is None

    def test_full_call(self) -> None:
        """Test creating call with all fields."""
        call = ToolCall(
            tool="test_tool",
            input={"query": "test"},
            task_id="task-001",
        )

        assert call.tool == "test_tool"
        assert call.input == {"query": "test"}
        assert call.task_id == "task-001"

    def test_tool_name_required(self) -> None:
        """Test tool name is required."""
        with pytest.raises(ValidationError):
            ToolCall(tool="")


class TestToolCallRecord:
    """Tests for ToolCallRecord model."""

    def test_record_creation(self) -> None:
        """Test creating a call record."""
        record = ToolCallRecord(
            tool="test_tool",
            input={"query": "test"},
            output={"result": "value"},
            error=None,
            status="success",
            duration_ms=50.5,
            task_id="task-001",
        )

        assert record.tool == "test_tool"
        assert record.input == {"query": "test"}
        assert record.output == {"result": "value"}
        assert record.error is None
        assert record.status == "success"
        assert record.duration_ms == 50.5
        assert record.task_id == "task-001"
        assert record.timestamp is not None

    def test_duration_ms_validation(self) -> None:
        """Test duration_ms must be non-negative."""
        with pytest.raises(ValidationError):
            ToolCallRecord(
                tool="test",
                status="success",
                duration_ms=-1,
            )


class TestMockDefinition:
    """Tests for MockDefinition model."""

    def test_minimal_definition(self) -> None:
        """Test creating definition with minimal data."""
        definition = MockDefinition(name="test_mock")

        assert definition.name == "test_mock"
        assert definition.description is None
        assert definition.tools == []
        assert definition.default_delay_ms == 0

    def test_full_definition(self) -> None:
        """Test creating definition with all fields."""
        tool = MockTool(name="tool1")
        definition = MockDefinition(
            name="test_mock",
            description="Test description",
            tools=[tool],
            default_delay_ms=100,
        )

        assert definition.name == "test_mock"
        assert definition.description == "Test description"
        assert len(definition.tools) == 1
        assert definition.default_delay_ms == 100

    def test_get_tool_found(self) -> None:
        """Test get_tool returns tool when found."""
        tool = MockTool(name="my_tool")
        definition = MockDefinition(name="test", tools=[tool])

        result = definition.get_tool("my_tool")

        assert result is not None
        assert result.name == "my_tool"

    def test_get_tool_not_found(self) -> None:
        """Test get_tool returns None when not found."""
        definition = MockDefinition(name="test", tools=[])

        result = definition.get_tool("nonexistent")

        assert result is None

    def test_name_required(self) -> None:
        """Test name is required."""
        with pytest.raises(ValidationError):
            MockDefinition(name="")
