"""Directory-backed file tools for corpus runs."""

from __future__ import annotations

from pathlib import Path

from atp.mock_tools.models import MockResponse, ToolCall


def _normalize_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _content_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".md":
        return "text/markdown"
    return "text/plain"


def _line_count(text: str) -> int:
    return len(text.splitlines())


class DirectoryFileRead:
    """Serve read-only UTF-8 text files from a fixed directory."""

    def __init__(self, root: Path, *, allowed_paths: set[str]) -> None:
        """Initialize with a root directory and allowed relative paths."""
        self.root = root.resolve()
        self.allowed_paths = {path.replace("\\", "/") for path in allowed_paths}

    async def __call__(self, call: ToolCall) -> MockResponse:
        """Handle a ``file_read`` call."""
        input_data = call.input
        if not isinstance(input_data, dict):
            return self._error("file_read input must be an object")
        path = input_data.get("path")
        if not isinstance(path, str) or not path:
            return self._error("file_read requires input.path")
        if "\x00" in path:
            return self._error("invalid path")
        normalized = path.replace("\\", "/")
        path_obj = Path(normalized)
        if path_obj.is_absolute() or normalized.startswith("~"):
            return self._error("absolute paths are not allowed")
        if "//" in normalized or any(
            part in ("", ".", "..") for part in normalized.split("/")
        ):
            return self._error("unsafe path")
        if normalized not in self.allowed_paths:
            return self._error(f"path is not allowed: {normalized}")

        target = (self.root / normalized).resolve()
        if not target.is_relative_to(self.root):
            return self._error("path escapes root")
        if target.is_dir():
            return self._error("directory reads are not allowed")
        if not target.is_file():
            return self._error(f"file not found: {normalized}")
        if target.suffix.lower() not in {".md", ".txt"}:
            return self._error("unsupported file type")

        try:
            content = _normalize_lf(target.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            return self._error("file is not UTF-8 text")
        return MockResponse(
            output={
                "path": normalized,
                "content": content,
                "metadata": {
                    "content_type": _content_type(normalized),
                    "line_count": _line_count(content),
                },
            }
        )

    def _error(self, message: str) -> MockResponse:
        return MockResponse(status="error", error=message)
