"""Tests for mock definition loader."""

import pytest

from atp.core.exceptions import ParseError
from atp.mock_tools.loader import MockDefinitionLoader
from atp.mock_tools.models import MatchType


class TestMockDefinitionLoader:
    """Tests for MockDefinitionLoader class."""

    def test_load_string_minimal(self) -> None:
        """Test loading minimal YAML definition."""
        loader = MockDefinitionLoader()
        yaml_content = """
name: test_mock
tools: []
"""
        definition = loader.load_string(yaml_content)

        assert definition.name == "test_mock"
        assert definition.tools == []
        assert definition.default_delay_ms == 0

    def test_load_string_with_tools(self) -> None:
        """Test loading definition with tools."""
        loader = MockDefinitionLoader()
        yaml_content = """
name: test_mock
description: Test mock definition
default_delay_ms: 50
tools:
  - name: web_search
    description: Search the web
    default:
      output:
        results: []
"""
        definition = loader.load_string(yaml_content)

        assert definition.name == "test_mock"
        assert definition.description == "Test mock definition"
        assert definition.default_delay_ms == 50
        assert len(definition.tools) == 1

        tool = definition.tools[0]
        assert tool.name == "web_search"
        assert tool.description == "Search the web"
        assert tool.default_response.output == {"results": []}

    def test_load_string_with_pattern_matching(self) -> None:
        """Test loading definition with pattern matching."""
        loader = MockDefinitionLoader()
        yaml_content = """
name: test_mock
tools:
  - name: search
    responses:
      - when:
          type: exact
          field: query
          pattern: "python"
        then:
          output:
            found: true
      - when:
          type: contains
          pattern: "error"
        then:
          error: "Search failed"
          status: error
      - when:
          type: regex
          pattern: '^test-[0-9]+$'
        then:
          output:
            matched: true
    default:
      output:
        found: false
"""
        definition = loader.load_string(yaml_content)

        tool = definition.tools[0]
        assert len(tool.responses) == 3

        # Check exact matcher
        matcher1, response1 = tool.responses[0]
        assert matcher1 is not None
        assert matcher1.type == MatchType.EXACT
        assert matcher1.field == "query"
        assert matcher1.pattern == "python"
        assert response1.output == {"found": True}

        # Check contains matcher
        matcher2, response2 = tool.responses[1]
        assert matcher2 is not None
        assert matcher2.type == MatchType.CONTAINS
        assert matcher2.pattern == "error"
        assert response2.status == "error"

        # Check regex matcher
        matcher3, response3 = tool.responses[2]
        assert matcher3 is not None
        assert matcher3.type == MatchType.REGEX

    def test_load_string_empty_content(self) -> None:
        """Test loading empty YAML content raises error."""
        loader = MockDefinitionLoader()

        with pytest.raises(ParseError, match="Empty YAML content"):
            loader.load_string("")

    def test_load_string_invalid_yaml(self) -> None:
        """Test loading invalid YAML raises error."""
        loader = MockDefinitionLoader()
        # Invalid YAML: unclosed quote
        invalid_yaml = """
name: "unclosed string
tools: []
"""
        with pytest.raises(ParseError, match="YAML parsing error"):
            loader.load_string(invalid_yaml)

    def test_load_string_missing_name(self) -> None:
        """Test loading without name uses default."""
        loader = MockDefinitionLoader()
        yaml_content = """
tools: []
"""
        definition = loader.load_string(yaml_content)
        assert definition.name == "unnamed"

    def test_load_file_not_found(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test loading non-existent file raises error."""
        loader = MockDefinitionLoader()
        fake_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(ParseError, match="File not found"):
            loader.load_file(fake_path)

    def test_load_file_valid(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test loading valid YAML file."""
        loader = MockDefinitionLoader()
        yaml_content = """
name: file_mock
tools:
  - name: test_tool
    default:
      output:
        result: ok
"""
        yaml_file = tmp_path / "mock.yaml"
        yaml_file.write_text(yaml_content)

        definition = loader.load_file(yaml_file)

        assert definition.name == "file_mock"
        assert len(definition.tools) == 1

    def test_load_file_empty(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test loading empty YAML file raises error."""
        loader = MockDefinitionLoader()
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        with pytest.raises(ParseError, match="Empty YAML file"):
            loader.load_file(empty_file)

    def test_load_string_with_delay(self) -> None:
        """Test loading definition with delay settings."""
        loader = MockDefinitionLoader()
        yaml_content = """
name: delay_mock
tools:
  - name: slow_tool
    responses:
      - when:
          type: any
        then:
          output: {}
          delay_ms: 500
    default:
      output: {}
      delay_ms: 100
"""
        definition = loader.load_string(yaml_content)

        tool = definition.tools[0]
        matcher, response = tool.responses[0]
        assert response.delay_ms == 500
        assert tool.default_response.delay_ms == 100

    def test_load_string_no_when_clause(self) -> None:
        """Test loading response without when clause."""
        loader = MockDefinitionLoader()
        yaml_content = """
name: test_mock
tools:
  - name: tool
    responses:
      - then:
          output:
            always: true
    default:
      output: {}
"""
        definition = loader.load_string(yaml_content)

        tool = definition.tools[0]
        matcher, _ = tool.responses[0]
        # No when clause means None matcher
        assert matcher is None

    def test_load_string_unknown_match_type(self) -> None:
        """Test loading with unknown match type defaults to ANY."""
        loader = MockDefinitionLoader()
        yaml_content = """
name: test_mock
tools:
  - name: tool
    responses:
      - when:
          type: unknown_type
        then:
          output: {}
    default:
      output: {}
"""
        definition = loader.load_string(yaml_content)

        tool = definition.tools[0]
        matcher, _ = tool.responses[0]
        assert matcher is not None
        assert matcher.type == MatchType.ANY
