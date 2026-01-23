"""YAML loader for mock tool definitions."""

from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import MarkedYAMLError

from atp.core.exceptions import ParseError, ValidationError
from atp.mock_tools.models import (
    MatchType,
    MockDefinition,
    MockResponse,
    MockTool,
    PatternMatcher,
)


class MockDefinitionLoader:
    """Load mock definitions from YAML files."""

    def __init__(self) -> None:
        """Initialize the loader."""
        self._yaml = YAML()
        self._yaml.preserve_quotes = True

    def load_file(self, file_path: str | Path) -> MockDefinition:
        """Load mock definition from YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            MockDefinition object

        Raises:
            ParseError: If YAML parsing fails
            ValidationError: If validation fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ParseError(f"File not found: {file_path}")

        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = self._yaml.load(f)

            if data is None:
                raise ParseError(f"Empty YAML file: {file_path}")

            return self._build_definition(data, str(file_path))

        except MarkedYAMLError as e:
            line = e.problem_mark.line + 1 if e.problem_mark else None
            column = e.problem_mark.column + 1 if e.problem_mark else None
            raise ParseError(
                f"YAML parsing error at line {line}, column {column}: {e.problem}"
            ) from e

    def load_string(self, content: str) -> MockDefinition:
        """Load mock definition from YAML string.

        Args:
            content: YAML content as string

        Returns:
            MockDefinition object

        Raises:
            ParseError: If YAML parsing fails
            ValidationError: If validation fails
        """
        try:
            data = self._yaml.load(content)

            if data is None:
                raise ParseError("Empty YAML content")

            return self._build_definition(data, None)

        except MarkedYAMLError as e:
            line = e.problem_mark.line + 1 if e.problem_mark else None
            column = e.problem_mark.column + 1 if e.problem_mark else None
            raise ParseError(
                f"YAML parsing error at line {line}, column {column}: {e.problem}"
            ) from e

    def _build_definition(
        self, data: dict[str, Any], file_path: str | None
    ) -> MockDefinition:
        """Build MockDefinition from parsed YAML data.

        Args:
            data: Parsed YAML data
            file_path: Optional file path for error messages

        Returns:
            MockDefinition object

        Raises:
            ValidationError: If validation fails
        """
        # Convert tools format
        tools_data = data.get("tools", [])
        tools = []

        for tool_data in tools_data:
            tool = self._build_tool(tool_data)
            tools.append(tool)

        try:
            return MockDefinition(
                name=data.get("name", "unnamed"),
                description=data.get("description"),
                tools=tools,
                default_delay_ms=data.get("default_delay_ms", 0),
            )
        except PydanticValidationError as e:
            errors = []
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")

            error_msg = "Validation failed:\n  " + "\n  ".join(errors)
            raise ValidationError(error_msg, file_path=file_path) from e

    def _build_tool(self, tool_data: dict[str, Any]) -> MockTool:
        """Build MockTool from parsed data.

        Args:
            tool_data: Tool data from YAML

        Returns:
            MockTool object
        """
        responses: list[tuple[PatternMatcher | None, MockResponse]] = []

        for resp_data in tool_data.get("responses", []):
            matcher = self._build_matcher(resp_data.get("when"))
            response = self._build_response(resp_data.get("then", {}))
            responses.append((matcher, response))

        default_response_data = tool_data.get("default", {})
        default_response = self._build_response(default_response_data)

        return MockTool(
            name=tool_data.get("name", "unnamed"),
            description=tool_data.get("description"),
            responses=responses,
            default_response=default_response,
        )

    def _build_matcher(self, when_data: dict[str, Any] | None) -> PatternMatcher | None:
        """Build PatternMatcher from 'when' clause.

        Args:
            when_data: When clause data

        Returns:
            PatternMatcher or None if no matcher
        """
        if not when_data:
            return None

        match_type = when_data.get("type", "any")
        try:
            match_type_enum = MatchType(match_type)
        except ValueError:
            match_type_enum = MatchType.ANY

        return PatternMatcher(
            type=match_type_enum,
            pattern=when_data.get("pattern"),
            field=when_data.get("field"),
        )

    def _build_response(self, then_data: dict[str, Any]) -> MockResponse:
        """Build MockResponse from 'then' clause.

        Args:
            then_data: Then clause data

        Returns:
            MockResponse object
        """
        return MockResponse(
            output=then_data.get("output"),
            error=then_data.get("error"),
            delay_ms=then_data.get("delay_ms", 0),
            status=then_data.get("status", "success"),
        )
