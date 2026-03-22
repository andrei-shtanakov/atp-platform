"""YAML parser with ruamel.yaml for line number tracking."""

import os
import re
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import MarkedYAMLError

from atp.core.exceptions import ParseError, ValidationError


class YAMLParser:
    """YAML parser with variable substitution and line number tracking."""

    def __init__(self):
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.default_flow_style = False

    def parse_file(self, file_path: str | Path) -> dict[str, Any]:
        """Parse YAML file with error handling.

        Args:
            file_path: Path to YAML file

        Returns:
            Parsed YAML data as dictionary

        Raises:
            ParseError: If YAML parsing fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ParseError(f"File not found: {file_path}")

        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = self.yaml.load(f)

            if data is None:
                raise ParseError(f"Empty YAML file: {file_path}")

            return data

        except MarkedYAMLError as e:
            # Extract line and column from ruamel.yaml error
            line = e.problem_mark.line + 1 if e.problem_mark else None
            column = e.problem_mark.column + 1 if e.problem_mark else None
            raise ParseError(
                f"YAML parsing error at line {line}, column {column}: {e.problem}"
            ) from e

        except Exception as e:
            raise ParseError(f"Failed to parse YAML file {file_path}: {e}") from e

    def parse_string(self, content: str) -> dict[str, Any]:
        """Parse YAML from string.

        Args:
            content: YAML content as string

        Returns:
            Parsed YAML data as dictionary

        Raises:
            ParseError: If YAML parsing fails
        """
        try:
            data = self.yaml.load(content)

            if data is None:
                raise ParseError("Empty YAML content")

            return data

        except MarkedYAMLError as e:
            line = e.problem_mark.line + 1 if e.problem_mark else None
            column = e.problem_mark.column + 1 if e.problem_mark else None
            raise ParseError(
                f"YAML parsing error at line {line}, column {column}: {e.problem}"
            ) from e

        except Exception as e:
            raise ParseError(f"Failed to parse YAML content: {e}") from e


class VariableSubstitution:
    """Handle variable substitution in test definitions."""

    # Pattern to match ${VAR} or ${VAR:default}
    VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*?)(?::([^}]*))?\}")

    def __init__(self, env: dict[str, str] | None = None):
        """Initialize with environment variables.

        Args:
            env: Custom environment dict, defaults to os.environ
        """
        self.env = env if env is not None else dict(os.environ)

    def substitute(self, data: Any) -> Any:
        """Recursively substitute variables in data structure.

        Args:
            data: Data structure (dict, list, str, or primitive)

        Returns:
            Data with variables substituted

        Raises:
            ValidationError: If required variable is not found
        """
        if isinstance(data, dict):
            return {key: self.substitute(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.substitute(item) for item in data]
        elif isinstance(data, str):
            return self._substitute_string(data)
        else:
            return data

    def _substitute_string(self, text: str) -> str:
        """Substitute variables in a string.

        Args:
            text: String potentially containing ${VAR} placeholders

        Returns:
            String with variables substituted

        Raises:
            ValidationError: If required variable is not found
        """

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2)

            if var_name in self.env:
                return self.env[var_name]
            elif default_value is not None:
                return default_value
            else:
                raise ValidationError(
                    f"Required environment variable not found: {var_name}"
                )

        return self.VAR_PATTERN.sub(replacer, text)

    def validate_no_unresolved(self, data: Any) -> None:
        """Check that no unresolved variables remain.

        Args:
            data: Data structure to check

        Raises:
            ValidationError: If unresolved variables are found
        """
        unresolved = self._find_unresolved(data)
        if unresolved:
            raise ValidationError(
                f"Unresolved variables found: {', '.join(sorted(unresolved))}"
            )

    def _find_unresolved(self, data: Any) -> set[str]:
        """Find all unresolved variable references.

        Args:
            data: Data structure to search

        Returns:
            Set of unresolved variable names
        """
        unresolved = set()

        if isinstance(data, dict):
            for value in data.values():
                unresolved.update(self._find_unresolved(value))
        elif isinstance(data, list):
            for item in data:
                unresolved.update(self._find_unresolved(item))
        elif isinstance(data, str):
            for match in self.VAR_PATTERN.finditer(data):
                var_name = match.group(1)
                default_value = match.group(2)
                # Only unresolved if no default and not in env
                if default_value is None and var_name not in self.env:
                    unresolved.add(var_name)

        return unresolved
