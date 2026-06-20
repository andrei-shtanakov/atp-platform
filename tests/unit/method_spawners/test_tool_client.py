"""Tests for method spawner tool-client HTTP calls."""

import pytest


@pytest.mark.anyio
async def test_call_tool_posts_to_tools_call_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from method.spawners import _tool_client

    calls: list[dict] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "tool": "file_read",
                "status": "success",
                "output": {"content": "policy\n"},
                "duration_ms": 1.0,
            }

    class _Client:
        async def __aenter__(self):  # type: ignore[no-untyped-def]
            return self

        async def __aexit__(self, *exc):  # type: ignore[no-untyped-def]
            return None

        async def post(self, url: str, json: dict):  # type: ignore[no-untyped-def]
            calls.append({"url": url, "json": json})
            return _Response()

    monkeypatch.setattr(_tool_client.httpx, "AsyncClient", _Client)

    response = await _tool_client.call_tool(
        "http://tools.local",
        "file_read",
        {"path": "policy.md"},
        task_id="task-1",
    )

    assert response["status"] == "success"
    assert response["output"]["content"] == "policy\n"
    assert calls == [
        {
            "url": "http://tools.local/tools/call",
            "json": {
                "tool": "file_read",
                "input": {"path": "policy.md"},
                "task_id": "task-1",
            },
        }
    ]
