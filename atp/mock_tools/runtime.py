"""Runtime helpers for serving mock tools on localhost."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn

from atp.mock_tools.server import MockToolServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@asynccontextmanager
async def serve_mock_tools(server: MockToolServer) -> AsyncIterator[str]:
    """Serve ``server`` on an ephemeral localhost port."""
    port = _free_port()
    config = uvicorn.Config(
        server.get_app(),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    uvicorn_server = uvicorn.Server(config)
    task = asyncio.create_task(uvicorn_server.serve())
    try:
        while not uvicorn_server.started:
            if task.done():
                raise RuntimeError("mock tool server failed to start")
            await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        uvicorn_server.should_exit = True
        await task
