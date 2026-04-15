"""Shared fixtures for E2E tests."""

import json
import time
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest


class MockAgentHandler(BaseHTTPRequestHandler):
    """Mock HTTP handler for testing agent responses.

    Behavior is controlled by request path:
    - /agent: Returns successful response
    - /agent/fail: Returns failed response
    - /agent/slow: Delays response by configured time
    - /health: Returns 200 OK for health check
    """

    # Class-level configuration
    delay_seconds: float = 0.0
    fail_mode: bool = False
    error_mode: bool = False

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress logging."""
        pass

    def do_GET(self) -> None:
        """Handle GET requests (health check)."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests (agent execution)."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Check if SSE is requested
        accept = self.headers.get("Accept", "")
        is_sse = "text/event-stream" in accept

        # Handle /agent endpoint
        if self.path == "/agent":
            if is_sse:
                self._handle_agent_request_sse(body)
            else:
                self._handle_agent_request(body)
        elif self.path == "/agent/fail":
            if is_sse:
                self._handle_fail_request_sse(body)
            else:
                self._handle_fail_request(body)
        elif self.path == "/agent/slow":
            if is_sse:
                self._handle_slow_request_sse(body)
            else:
                self._handle_slow_request(body)
        elif self.path == "/agent/error":
            self._handle_error_request()
        else:
            self.send_response(404)
            self.end_headers()

    def _build_success_response(self, task_id: str) -> dict[str, Any]:
        """Build a successful response dict."""
        return {
            "version": "1.0",
            "task_id": task_id,
            "status": "completed",
            "artifacts": [
                {
                    "type": "structured",
                    "name": "result",
                    "data": {"message": "Task completed successfully"},
                }
            ],
            "metrics": {
                "total_tokens": 150,
                "total_steps": 3,
                "wall_time_seconds": 1.5,
            },
        }

    def _handle_agent_request(self, body: bytes) -> None:
        """Handle successful agent response (JSON)."""
        try:
            request_data = json.loads(body)
            task_id = request_data.get("task_id", "unknown")
            response = self._build_success_response(task_id)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON"}')

    def _handle_agent_request_sse(self, body: bytes) -> None:
        """Handle successful agent response (SSE stream)."""
        try:
            request_data = json.loads(body)
            task_id = request_data.get("task_id", "unknown")

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            # Send progress event
            event_data = {
                "task_id": task_id,
                "event_type": "progress",
                "sequence": 0,
                "payload": {"message": "Processing task"},
            }
            self.wfile.write(b"event: progress\n")
            self.wfile.write(f"data: {json.dumps(event_data)}\n\n".encode())
            self.wfile.flush()

            # Send final response
            response = self._build_success_response(task_id)
            self.wfile.write(b"event: response\n")
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()

        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()

    def _build_fail_response(self, task_id: str) -> dict[str, Any]:
        """Build a failed response dict."""
        return {
            "version": "1.0",
            "task_id": task_id,
            "status": "failed",
            "error": "Test deliberately failed",
            "metrics": {
                "total_tokens": 50,
                "total_steps": 1,
            },
        }

    def _handle_fail_request(self, body: bytes) -> None:
        """Handle request that should result in a failed test (JSON)."""
        try:
            request_data = json.loads(body)
            task_id = request_data.get("task_id", "unknown")
            response = self._build_fail_response(task_id)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()

    def _handle_fail_request_sse(self, body: bytes) -> None:
        """Handle request that should result in a failed test (SSE)."""
        try:
            request_data = json.loads(body)
            task_id = request_data.get("task_id", "unknown")

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            # Send final failed response
            response = self._build_fail_response(task_id)
            self.wfile.write(b"event: response\n")
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()

        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()

    def _build_slow_response(self, task_id: str, delay: float) -> dict[str, Any]:
        """Build a slow response dict."""
        return {
            "version": "1.0",
            "task_id": task_id,
            "status": "completed",
            "artifacts": [],
            "metrics": {
                "total_tokens": 100,
                "total_steps": 2,
                "wall_time_seconds": delay,
            },
        }

    def _handle_slow_request(self, body: bytes) -> None:
        """Handle slow request that may timeout (JSON)."""
        # Use class-level delay setting
        delay = getattr(self.__class__, "delay_seconds", 5.0)
        time.sleep(delay)

        try:
            request_data = json.loads(body)
            task_id = request_data.get("task_id", "unknown")
            response = self._build_slow_response(task_id, delay)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()

    def _handle_slow_request_sse(self, body: bytes) -> None:
        """Handle slow request that may timeout (SSE)."""
        delay = getattr(self.__class__, "delay_seconds", 5.0)
        time.sleep(delay)

        try:
            request_data = json.loads(body)
            task_id = request_data.get("task_id", "unknown")

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            # Send final response
            response = self._build_slow_response(task_id, delay)
            self.wfile.write(b"event: response\n")
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()

        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()

    def _handle_error_request(self) -> None:
        """Handle request that returns a server error."""
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error": "Internal server error"}')


class MockAgentServer:
    """A mock HTTP server for testing.

    Provides endpoints for simulating different agent behaviors.
    """

    def __init__(self) -> None:
        """Initialize the mock server."""
        self._server: HTTPServer | None = None
        self._thread: Thread | None = None
        self._port: int = 0

    def start(self) -> str:
        """Start the mock server and return the base URL."""
        self._server = HTTPServer(("127.0.0.1", 0), MockAgentHandler)
        self._port = self._server.server_address[1]

        self._thread = Thread(target=self._serve)
        self._thread.daemon = True
        self._thread.start()

        return f"http://127.0.0.1:{self._port}"

    def _serve(self) -> None:
        """Server thread main loop."""
        if self._server:
            self._server.serve_forever()

    def stop(self) -> None:
        """Stop the mock server."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    def set_delay(self, seconds: float) -> None:
        """Set the delay for slow endpoint."""
        MockAgentHandler.delay_seconds = seconds


@pytest.fixture(scope="module")
def mock_agent_server() -> Generator[MockAgentServer, None, None]:
    """Create and start a mock agent server for the test module."""
    server = MockAgentServer()
    url = server.start()

    # Store URL as attribute for tests to access
    server.url = url  # type: ignore[attr-defined]

    yield server

    server.stop()


@pytest.fixture
def mock_server_url(mock_agent_server: MockAgentServer) -> str:
    """Get the mock server URL."""
    return mock_agent_server.url  # type: ignore[attr-defined]


@pytest.fixture
def e2e_fixtures_dir() -> Path:
    """Return the path to E2E test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def valid_suite_path(e2e_fixtures_dir: Path) -> Path:
    """Return path to a valid test suite for E2E tests."""
    return e2e_fixtures_dir / "valid_e2e_suite.yaml"


@pytest.fixture
def failing_suite_path(e2e_fixtures_dir: Path) -> Path:
    """Return path to a test suite that should fail."""
    return e2e_fixtures_dir / "failing_suite.yaml"


@pytest.fixture
def timeout_suite_path(e2e_fixtures_dir: Path) -> Path:
    """Return path to a test suite with timeout tests."""
    return e2e_fixtures_dir / "timeout_suite.yaml"


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# ---------------------------------------------------------------------------
# MCP tournament e2e fixtures (Phase 8 of the MCP tournament vertical slice)
# ---------------------------------------------------------------------------


_E2E_SECRET = "e2e-test-secret-32-bytes-long-pad"


@pytest.fixture
async def e2e_mcp_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Boot a real uvicorn instance on a free port with an ephemeral
    SQLite database, yielding ``(base_url, port)``.

    The fixture resets the ``get_config`` LRU cache and patches
    ``atp.dashboard.auth.SECRET_KEY`` so JWTs minted with the e2e
    secret decode correctly in-process.
    """
    import asyncio
    import socket

    import uvicorn

    db_path = tmp_path / "e2e.db"
    monkeypatch.setenv("ATP_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("ATP_SECRET_KEY", _E2E_SECRET)
    monkeypatch.setenv("ATP_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("ATP_DEBUG", "false")

    # SECRET_KEY is bound at import time — patch the live module value
    # so the JWT middleware decodes the tokens we mint.
    import atp.dashboard.auth as auth_module

    monkeypatch.setattr(auth_module, "SECRET_KEY", _E2E_SECRET)

    # Drop any cached config so the new env vars take effect.
    from atp.dashboard.v2.config import get_config

    get_config.cache_clear()

    from atp.dashboard.v2.factory import create_app

    app = create_app()

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    loop = asyncio.get_event_loop()
    deadline = loop.time() + 5.0
    started = False
    while loop.time() < deadline:
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            started = True
            break
        except OSError:
            await asyncio.sleep(0.05)
    if not started:
        server.should_exit = True
        await server_task
        raise RuntimeError(f"uvicorn did not come up on port {port}")

    # TCP-accept readiness is necessary but not sufficient: FastMCP's SSE
    # session manager mounts during the inner lifespan_context which can
    # race the first SSE GET in CI. Do a FULL authenticated SSE handshake
    # as the readiness probe — any response before ``event: endpoint``
    # means the session manager isn't fully warmed yet. Keeps retrying
    # until a complete endpoint frame is observed OR the deadline passes.
    import httpx as _httpx

    probe_jwt = _make_e2e_jwt(0, "probe")
    mcp_deadline = loop.time() + 15.0
    mcp_ready = False
    last_err: Exception | None = None
    while loop.time() < mcp_deadline:
        try:
            async with _httpx.AsyncClient(timeout=3.0) as probe:
                async with probe.stream(
                    "GET",
                    f"http://127.0.0.1:{port}/mcp/sse",
                    headers={
                        "Accept": "text/event-stream",
                        "Authorization": f"Bearer {probe_jwt}",
                    },
                ) as resp:
                    if resp.status_code != 200:
                        last_err = RuntimeError(f"/mcp/sse status={resp.status_code}")
                        await asyncio.sleep(0.1)
                        continue
                    async for line in resp.aiter_lines():
                        if line.startswith("event: endpoint"):
                            mcp_ready = True
                            break
                    if mcp_ready:
                        break
                    last_err = RuntimeError("/mcp/sse closed without endpoint frame")
        except _httpx.RequestError as e:
            last_err = e
        await asyncio.sleep(0.1)
    if not mcp_ready:
        server.should_exit = True
        await server_task
        raise RuntimeError(
            f"MCP /mcp/sse never emitted endpoint frame on port {port}"
            + (f": {last_err}" if last_err else "")
        )

    try:
        yield (f"http://127.0.0.1:{port}", port)
    finally:
        server.should_exit = True
        await server_task
        get_config.cache_clear()


def _make_e2e_jwt(user_id: int, username: str) -> str:
    from datetime import UTC, datetime, timedelta

    import jwt

    return jwt.encode(
        {
            "sub": username,
            "user_id": user_id,
            "exp": datetime.now(tz=UTC) + timedelta(hours=1),
        },
        _E2E_SECRET,
        algorithm="HS256",
    )


@pytest.fixture
async def mcp_seeded_users(e2e_mcp_server) -> dict[str, dict]:
    """Insert admin + alice + bob into the e2e database, return JWTs.

    Depends on ``e2e_mcp_server`` so that the runtime database has
    been initialised via ``init_database`` in the app's lifespan.
    """
    from atp.dashboard.database import get_database
    from atp.dashboard.models import User

    db = get_database()
    async with db.session() as session:
        admin = User(
            username="admin",
            email="admin@e2e",
            hashed_password="x",
            is_admin=True,
            is_active=True,
        )
        alice = User(
            username="alice",
            email="alice@e2e",
            hashed_password="x",
            is_admin=False,
            is_active=True,
        )
        bob = User(
            username="bob",
            email="bob@e2e",
            hashed_password="x",
            is_admin=False,
            is_active=True,
        )
        session.add_all([admin, alice, bob])
        await session.commit()
        await session.refresh(admin)
        await session.refresh(alice)
        await session.refresh(bob)

        return {
            "admin": {
                "id": admin.id,
                "username": "admin",
                "jwt": _make_e2e_jwt(admin.id, "admin"),
            },
            "alice": {
                "id": alice.id,
                "username": "alice",
                "jwt": _make_e2e_jwt(alice.id, "alice"),
            },
            "bob": {
                "id": bob.id,
                "username": "bob",
                "jwt": _make_e2e_jwt(bob.id, "bob"),
            },
        }
