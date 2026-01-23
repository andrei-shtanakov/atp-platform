"""Models for mock tool definitions and responses."""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class MatchType(str, Enum):
    """Match type for pattern matching."""

    EXACT = "exact"
    REGEX = "regex"
    CONTAINS = "contains"
    ANY = "any"


class PatternMatcher(BaseModel):
    """Pattern matcher for tool input matching."""

    type: MatchType = Field(default=MatchType.ANY, description="Type of matching")
    pattern: str | None = Field(
        default=None, description="Pattern to match (for exact, regex, contains)"
    )
    field: str | None = Field(
        default=None, description="Specific field to match (for dict inputs)"
    )

    def matches(self, input_data: dict[str, Any] | str | None) -> bool:
        """Check if input matches this pattern.

        Args:
            input_data: Input data to check

        Returns:
            True if input matches pattern
        """
        if self.type == MatchType.ANY:
            return True

        # Get value to match
        if self.field and isinstance(input_data, dict):
            value = input_data.get(self.field, "")
        elif isinstance(input_data, dict):
            value = str(input_data)
        elif input_data is None:
            value = ""
        else:
            value = str(input_data)

        if self.pattern is None:
            return True

        if self.type == MatchType.EXACT:
            return value == self.pattern

        if self.type == MatchType.CONTAINS:
            return self.pattern in value

        if self.type == MatchType.REGEX:
            try:
                return bool(re.search(self.pattern, value))
            except re.error:
                return False

        return False


class MockResponse(BaseModel):
    """Mock response definition."""

    output: dict[str, Any] | str | None = Field(
        default=None, description="Response output data"
    )
    error: str | None = Field(
        default=None, description="Error message if this response is an error"
    )
    delay_ms: int = Field(
        default=0, ge=0, description="Simulated delay in milliseconds"
    )
    status: Literal["success", "error"] = Field(
        default="success", description="Response status"
    )


class MockTool(BaseModel):
    """Mock tool definition with matchers and responses."""

    name: str = Field(..., min_length=1, description="Tool name")
    description: str | None = Field(default=None, description="Tool description")
    responses: list[tuple[PatternMatcher | None, MockResponse]] = Field(
        default_factory=list,
        description="List of (matcher, response) pairs",
    )
    default_response: MockResponse = Field(
        default_factory=lambda: MockResponse(output={"result": "mock response"}),
        description="Default response when no patterns match",
    )

    @field_validator("responses", mode="before")
    @classmethod
    def validate_responses(
        cls, v: list[Any]
    ) -> list[tuple[PatternMatcher | None, MockResponse]]:
        """Convert dict format to tuple format if needed."""
        if not v:
            return []

        result = []
        for item in v:
            if isinstance(item, dict):
                matcher = item.get("matcher")
                response = item.get("response", MockResponse())
                if matcher and isinstance(matcher, dict):
                    matcher = PatternMatcher(**matcher)
                if isinstance(response, dict):
                    response = MockResponse(**response)
                result.append((matcher, response))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                matcher, response = item
                if isinstance(matcher, dict):
                    matcher = PatternMatcher(**matcher)
                if isinstance(response, dict):
                    response = MockResponse(**response)
                result.append((matcher, response))
            else:
                result.append(item)

        return result

    def get_response(self, input_data: dict[str, Any] | str | None) -> MockResponse:
        """Get response for given input.

        Args:
            input_data: Tool input data

        Returns:
            Matching MockResponse
        """
        for matcher, response in self.responses:
            if matcher is None or matcher.matches(input_data):
                return response
        return self.default_response


class ToolCall(BaseModel):
    """Incoming tool call request."""

    tool: str = Field(..., min_length=1, description="Tool name")
    input: dict[str, Any] | str | None = Field(
        default=None, description="Tool input data"
    )
    task_id: str | None = Field(default=None, description="Associated task ID")


class ToolCallRecord(BaseModel):
    """Record of a tool call for auditing."""

    timestamp: datetime = Field(
        default_factory=datetime.now, description="Call timestamp"
    )
    tool: str = Field(..., description="Tool name")
    input: dict[str, Any] | str | None = Field(default=None, description="Tool input")
    output: dict[str, Any] | str | None = Field(default=None, description="Tool output")
    error: str | None = Field(default=None, description="Error if any")
    status: str = Field(..., description="Call status (success/error)")
    duration_ms: float = Field(..., ge=0, description="Call duration in ms")
    task_id: str | None = Field(default=None, description="Associated task ID")


class MockDefinition(BaseModel):
    """Complete mock definition with multiple tools."""

    name: str = Field(..., min_length=1, description="Mock definition name")
    description: str | None = Field(default=None, description="Description")
    tools: list[MockTool] = Field(
        default_factory=list, description="List of mock tools"
    )
    default_delay_ms: int = Field(
        default=0, ge=0, description="Default delay for all tools"
    )

    def get_tool(self, tool_name: str) -> MockTool | None:
        """Get tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            MockTool if found, None otherwise
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
