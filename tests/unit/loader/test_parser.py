"""Unit tests for YAML parser and variable substitution."""

import pytest

from atp.core.exceptions import ParseError, ValidationError
from atp.loader.parser import VariableSubstitution, YAMLParser


class TestYAMLParser:
    """Test YAML parser functionality."""

    def test_parse_valid_yaml_string(self):
        """Test parsing valid YAML string."""
        parser = YAMLParser()
        yaml_content = """
        key1: value1
        key2: 123
        nested:
          inner: value
        """

        result = parser.parse_string(yaml_content)

        assert result["key1"] == "value1"
        assert result["key2"] == 123
        assert result["nested"]["inner"] == "value"

    def test_parse_empty_yaml_raises_error(self):
        """Test that empty YAML raises ParseError."""
        parser = YAMLParser()

        with pytest.raises(ParseError, match="Empty YAML content"):
            parser.parse_string("")

    def test_parse_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises ParseError with line number."""
        parser = YAMLParser()
        invalid_yaml = """
        key1: value1
        key2: [unclosed bracket
        """

        with pytest.raises(ParseError, match="line"):
            parser.parse_string(invalid_yaml)

    def test_parse_file_not_found(self, tmp_path):
        """Test that missing file raises ParseError."""
        parser = YAMLParser()
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(ParseError, match="File not found"):
            parser.parse_file(nonexistent)

    def test_parse_file_success(self, tmp_path):
        """Test parsing valid YAML file."""
        parser = YAMLParser()
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnumber: 42")

        result = parser.parse_file(yaml_file)

        assert result["key"] == "value"
        assert result["number"] == 42


class TestVariableSubstitution:
    """Test variable substitution functionality."""

    def test_substitute_simple_variable(self):
        """Test substituting a simple variable."""
        sub = VariableSubstitution(env={"API_KEY": "secret123"})

        result = sub.substitute("Token: ${API_KEY}")

        assert result == "Token: secret123"

    def test_substitute_with_default(self):
        """Test substitution with default value when variable not found."""
        sub = VariableSubstitution(env={})

        result = sub.substitute("Value: ${MISSING_VAR:default_value}")

        assert result == "Value: default_value"

    def test_substitute_missing_required_variable_raises_error(self):
        """Test that missing required variable raises ValidationError."""
        sub = VariableSubstitution(env={})

        with pytest.raises(ValidationError, match="REQUIRED_VAR"):
            sub.substitute("Value: ${REQUIRED_VAR}")

    def test_substitute_in_dict(self):
        """Test variable substitution in nested dictionary."""
        sub = VariableSubstitution(env={"HOST": "localhost", "PORT": "8000"})

        data = {"endpoint": "http://${HOST}:${PORT}", "nested": {"key": "${HOST}"}}

        result = sub.substitute(data)

        assert result["endpoint"] == "http://localhost:8000"
        assert result["nested"]["key"] == "localhost"

    def test_substitute_in_list(self):
        """Test variable substitution in list."""
        sub = VariableSubstitution(env={"TOOL": "web_search"})

        data = ["tool1", "${TOOL}", "tool3"]

        result = sub.substitute(data)

        assert result == ["tool1", "web_search", "tool3"]

    def test_substitute_no_variables(self):
        """Test that strings without variables are unchanged."""
        sub = VariableSubstitution(env={"KEY": "value"})

        result = sub.substitute("plain string")

        assert result == "plain string"

    def test_substitute_multiple_variables_in_string(self):
        """Test multiple variables in single string."""
        sub = VariableSubstitution(env={"USER": "admin", "PASS": "secret"})

        result = sub.substitute("${USER}:${PASS}")

        assert result == "admin:secret"

    def test_find_unresolved_variables(self):
        """Test finding unresolved variable references."""
        sub = VariableSubstitution(env={"EXISTING": "value"})

        data = {
            "resolved": "${EXISTING}",
            "with_default": "${MISSING:default}",
            "unresolved": "${REQUIRED_VAR}",
        }

        unresolved = sub._find_unresolved(data)

        assert unresolved == {"REQUIRED_VAR"}

    def test_validate_no_unresolved_success(self):
        """Test validation passes when all variables resolved."""
        sub = VariableSubstitution(env={"KEY": "value"})

        data = {"field": "${KEY}"}
        substituted = sub.substitute(data)

        # Should not raise
        sub.validate_no_unresolved(substituted)

    def test_substitute_preserves_types(self):
        """Test that non-string types are preserved."""
        sub = VariableSubstitution(env={"VAR": "value"})

        data = {"string": "${VAR}", "number": 123, "bool": True, "null": None}

        result = sub.substitute(data)

        assert result["string"] == "value"
        assert result["number"] == 123
        assert result["bool"] is True
        assert result["null"] is None

    def test_variable_name_validation(self):
        """Test that only valid variable names are matched."""
        sub = VariableSubstitution(env={"VALID_VAR_123": "value"})

        # Valid variable names (uppercase, underscores, numbers)
        result1 = sub.substitute("${VALID_VAR_123}")
        assert result1 == "value"

        # Invalid patterns should not be substituted
        result2 = sub.substitute("${lowercase}")  # lowercase not valid
        assert "${lowercase}" in result2  # Should remain unchanged
