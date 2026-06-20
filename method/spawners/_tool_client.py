"""HTTP client for ATP tools endpoint calls."""

from __future__ import annotations

from typing import Any

import httpx


async def call_tool(
    tools_endpoint: str,
    tool: str,
    input_data: dict[str, Any] | str | None,
    *,
    task_id: str | None = None,
) -> dict[str, Any]:
    """POST one tool call to ``<tools_endpoint>/tools/call``."""
    url = tools_endpoint.rstrip("/") + "/tools/call"
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json={"tool": tool, "input": input_data, "task_id": task_id},
        )
        response.raise_for_status()
        return response.json()
