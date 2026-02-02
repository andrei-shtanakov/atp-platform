"""Unit tests for atp.core.exceptions module."""

import pytest

from atp.core.exceptions import (
    ATPError,
    LoaderError,
    ParseError,
    ValidationError,
)


class TestATPError:
    """Tests for ATPError base class."""

    def test_atp_error_is_exception(self):
        """Test that ATPError is an Exception subclass."""
        assert issubclass(ATPError, Exception)

    def test_atp_error_message(self):
        """Test ATPError with message."""
        error = ATPError("Test error message")
        assert str(error) == "Test error message"


class TestLoaderError:
    """Tests for LoaderError class."""

    def test_loader_error_is_atp_error(self):
        """Test that LoaderError is an ATPError subclass."""
        assert issubclass(LoaderError, ATPError)


class TestParseError:
    """Tests for ParseError class."""

    def test_parse_error_is_loader_error(self):
        """Test that ParseError is a LoaderError subclass."""
        assert issubclass(ParseError, LoaderError)


class TestValidationError:
    """Tests for ValidationError class."""

    def test_validation_error_is_loader_error(self):
        """Test that ValidationError is a LoaderError subclass."""
        assert issubclass(ValidationError, LoaderError)

    def test_validation_error_message_only(self):
        """Test ValidationError with message only."""
        error = ValidationError("Invalid value")
        assert str(error) == "Invalid value"
        assert error.message == "Invalid value"
        assert error.line is None
        assert error.column is None
        assert error.file_path is None

    def test_validation_error_with_file_path(self):
        """Test ValidationError with file path."""
        error = ValidationError("Invalid value", file_path="test.yaml")
        assert "File: test.yaml" in str(error)
        assert "Invalid value" in str(error)
        assert error.file_path == "test.yaml"

    def test_validation_error_with_line(self):
        """Test ValidationError with line number."""
        error = ValidationError("Invalid value", line=42)
        assert "Line: 42" in str(error)
        assert "Invalid value" in str(error)
        assert error.line == 42

    def test_validation_error_with_column(self):
        """Test ValidationError with column number."""
        error = ValidationError("Invalid value", column=10)
        assert "Column: 10" in str(error)
        assert "Invalid value" in str(error)
        assert error.column == 10

    def test_validation_error_with_line_and_column(self):
        """Test ValidationError with both line and column."""
        error = ValidationError("Invalid value", line=42, column=10)
        assert "Line: 42" in str(error)
        assert "Column: 10" in str(error)
        assert "Invalid value" in str(error)
        assert error.line == 42
        assert error.column == 10

    def test_validation_error_with_all_params(self):
        """Test ValidationError with all parameters."""
        error = ValidationError(
            "Invalid value",
            line=42,
            column=10,
            file_path="test.yaml",
        )
        error_str = str(error)
        assert "File: test.yaml" in error_str
        assert "Line: 42" in error_str
        assert "Column: 10" in error_str
        assert "Invalid value" in error_str
        assert error.message == "Invalid value"
        assert error.line == 42
        assert error.column == 10
        assert error.file_path == "test.yaml"

    def test_validation_error_can_be_raised(self):
        """Test that ValidationError can be raised and caught."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Test error", line=1, column=5)

        assert exc_info.value.line == 1
        assert exc_info.value.column == 5
