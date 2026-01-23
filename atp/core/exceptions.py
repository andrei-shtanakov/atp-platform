"""ATP exceptions."""


class ATPError(Exception):
    """Base exception for all ATP errors."""


class LoaderError(ATPError):
    """Base exception for loader errors."""


class ValidationError(LoaderError):
    """Validation error with line number information."""

    def __init__(
        self,
        message: str,
        line: int | None = None,
        column: int | None = None,
        file_path: str | None = None,
    ):
        self.message = message
        self.line = line
        self.column = column
        self.file_path = file_path

        location_parts = []
        if file_path:
            location_parts.append(f"File: {file_path}")
        if line is not None:
            location_parts.append(f"Line: {line}")
        if column is not None:
            location_parts.append(f"Column: {column}")

        if location_parts:
            full_message = f"{', '.join(location_parts)}: {message}"
        else:
            full_message = message

        super().__init__(full_message)


class ParseError(LoaderError):
    """YAML parsing error."""
