"""Tests for directory-backed file_read mock tool."""

from pathlib import Path

import pytest

from atp.mock_tools.models import ToolCall


def _handler(root: Path, allowed: set[str] | None = None):
    from atp.mock_tools.file_tools import DirectoryFileRead

    return DirectoryFileRead(root, allowed_paths=allowed or {"policy.md", "notes.txt"})


@pytest.mark.anyio
async def test_file_read_valid_path_returns_normalized_text_and_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "policy.md").write_text("Line 1\r\nLine 2\r\n")
    handler = _handler(tmp_path)

    response = await handler(ToolCall(tool="file_read", input={"path": "policy.md"}))

    assert response.status == "success"
    assert response.output == {
        "path": "policy.md",
        "content": "Line 1\nLine 2\n",
        "metadata": {"content_type": "text/markdown", "line_count": 2},
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "input_data",
    [
        {},
        {"path": "missing.md"},
        {"path": "../policy.md"},
        {"path": "/tmp/policy.md"},
        {"path": "policy\x00.md"},
        {"path": "nested"},
        {"path": "secret.md"},
        {"path": "image.bin"},
    ],
)
async def test_file_read_rejects_invalid_or_disallowed_paths(
    tmp_path: Path, input_data: dict
) -> None:
    (tmp_path / "policy.md").write_text("ok\n")
    (tmp_path / "secret.md").write_text("no\n")
    (tmp_path / "image.bin").write_bytes(b"\xff\x00\x01")
    (tmp_path / "nested").mkdir()
    handler = _handler(tmp_path, allowed={"policy.md", "nested", "image.bin"})

    response = await handler(ToolCall(tool="file_read", input=input_data))

    assert response.status == "error"
    assert response.error
