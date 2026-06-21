"""Tests for the anthropic_api spawner shim (offline, via a fake anthropic SDK)."""

import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from atp_method.envelopes import build_prompt, get_envelope

from atp.protocol import ATPEvent

SHIM = (
    Path(__file__).resolve().parents[3]
    / "method"
    / "spawners"
    / "anthropic_api_shim.py"
)

# A stand-in `anthropic` module dropped on PYTHONPATH so the shim's lazy
# `import anthropic` resolves to this instead of the real SDK (no network/key).
_FAKE_ANTHROPIC = """
import json as _json

_FINDINGS = _json.dumps(
    [{"rule_id": "sql-injection", "anchor": 'f"SELECT', "severity": "critical"}]
)


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Usage:
    input_tokens = 800
    output_tokens = 120


class _Msg:
    content = [_Block(_FINDINGS)]
    usage = _Usage()


class _Messages:
    def create(self, *a, **k):
        return _Msg()


class Anthropic:
    def __init__(self, *a, **k):
        self.messages = _Messages()
"""


def _run_shim(request: dict, env: dict) -> dict:
    proc = _run_shim_raw(request, env)
    return json.loads(proc.stdout.decode())


def _run_shim_raw(request: dict, env: dict) -> subprocess.CompletedProcess[bytes]:
    proc = subprocess.run(
        [sys.executable, str(SHIM)],
        input=json.dumps(request).encode(),
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    return proc


def _stderr_json_lines(proc: subprocess.CompletedProcess[bytes]) -> list[dict]:
    return [
        json.loads(line) for line in proc.stderr.decode().splitlines() if line.strip()
    ]


class _ToolHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))
        assert self.path == "/tools/call"
        assert body["tool"] == "file_read"
        response = {
            "tool": "file_read",
            "status": "success",
            "output": {"content": "policy line\n"},
            "duration_ms": 1.0,
        }
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return None


class _ToolErrorHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))
        assert self.path == "/tools/call"
        assert body["tool"] == "file_read"
        response = {
            "tool": "file_read",
            "status": "error",
            "error": "policy not found",
            "duration_ms": 1.0,
        }
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return None


def _serve_tools(
    handler: type[BaseHTTPRequestHandler] = _ToolHandler,
) -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


def _debug_files_by_suffix(
    debug_dir: Path, safe_task_id: str, expected_suffixes: set[str]
) -> dict[str, Path]:
    timestamp_pattern = r"\d{8}-\d{6}-\d{6}"
    pattern = re.compile(
        rf"^(?P<timestamp>{timestamp_pattern})-{re.escape(safe_task_id)}-(?P<suffix>.+)$"
    )
    matches: dict[str, tuple[str, Path]] = {}
    unexpected = []
    for path in debug_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if not match:
            unexpected.append(path.name)
            continue
        matches[match.group("suffix")] = (match.group("timestamp"), path)

    assert unexpected == []
    assert set(matches) == expected_suffixes
    timestamps = {timestamp for timestamp, _path in matches.values()}
    assert len(timestamps) == 1
    return {suffix: path for suffix, (_timestamp, path) in matches.items()}


def test_missing_key_emits_failed() -> None:
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    resp = _run_shim(
        {"version": "1.0", "task_id": "t1", "task": {"description": "x"}}, env
    )
    assert resp["status"] == "failed"
    assert "ANTHROPIC_API_KEY" in resp["error"]


def test_success_with_fake_sdk(tmp_path: Path) -> None:
    (tmp_path / "anthropic.py").write_text(_FAKE_ANTHROPIC)
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": "test-key",
        "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    env.pop("ATP_METHOD_DEBUG_IO_DIR", None)
    request = {
        "version": "1.0",
        "task_id": "t2",
        "task": {"description": "Review the diff against the rules."},
        "context": {"artifacts": []},
    }
    resp = _run_shim(request, env)
    assert list(tmp_path.glob("*-t2-*.txt")) == []
    assert list(tmp_path.glob("*-t2-*.json")) == []
    assert resp["task_id"] == "t2"
    assert resp["status"] == "completed"
    arts = resp["artifacts"]
    assert len(arts) == 1
    findings = json.loads(arts[0]["content"])
    assert findings[0]["rule_id"] == "sql-injection"
    assert "SELECT" in findings[0]["anchor"]
    assert resp["metrics"]["total_tokens"] == 920
    # raw API response carries no cost field; the baseline leaves it null
    assert resp["metrics"]["cost_usd"] is None


def test_success_writes_debug_io_files_when_enabled(tmp_path: Path) -> None:
    (tmp_path / "anthropic.py").write_text(_FAKE_ANTHROPIC)
    debug_dir = tmp_path / "debug"
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": "test-key",
        "ATP_METHOD_DEBUG_IO_DIR": str(debug_dir),
        "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    request = {
        "version": "1.0",
        "task_id": "debug/task",
        "task": {"description": "Review the diff against the rules."},
        "context": {"artifacts": []},
    }

    proc = _run_shim_raw(request, env)

    stdout_response = json.loads(proc.stdout.decode())
    debug_files = _debug_files_by_suffix(
        debug_dir,
        "debug_task",
        {
            "prompt.txt",
            "raw_response.json",
            "final_output.txt",
            "atp_response.json",
        },
    )
    atp_response = json.loads(debug_files["atp_response.json"].read_text())
    assert stdout_response == atp_response
    assert debug_files["prompt.txt"].read_text() == build_prompt(
        request, get_envelope("review")
    )
    assert (
        debug_files["final_output.txt"].read_text()
        == (stdout_response["artifacts"][0]["content"])
    )
    raw_response = json.loads(debug_files["raw_response.json"].read_text())
    assert raw_response["content"] == [
        {"type": "text", "text": stdout_response["artifacts"][0]["content"]}
    ]
    assert raw_response["usage"] == {"input_tokens": 800, "output_tokens": 120}


def test_invalid_stdin_emits_failed_not_crash() -> None:
    # Empty/garbage stdin must still produce a contract-shaped failed response.
    proc = subprocess.run(
        [sys.executable, str(SHIM)],
        input=b"not json at all",
        capture_output=True,
        env=os.environ.copy(),
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    resp = json.loads(proc.stdout.decode())
    assert resp["status"] == "failed"
    assert "invalid" in resp["error"].lower()


# Fake SDK whose message has NO usage attribute — exercises the token fallback.
_FAKE_ANTHROPIC_NO_USAGE = """
class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Msg:
    content = [_Block("[]")]
    usage = None


class _Messages:
    def create(self, *a, **k):
        return _Msg()


class Anthropic:
    def __init__(self, *a, **k):
        self.messages = _Messages()
"""


def test_missing_usage_defaults_tokens_zero(tmp_path: Path) -> None:
    (tmp_path / "anthropic.py").write_text(_FAKE_ANTHROPIC_NO_USAGE)
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": "test-key",
        "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    resp = _run_shim(
        {"version": "1.0", "task_id": "t3", "task": {"description": "x"}}, env
    )
    assert resp["status"] == "completed"
    assert resp["metrics"]["total_tokens"] == 0
    assert resp["metrics"]["input_tokens"] == 0


_FAKE_ANTHROPIC_TOOL_LOOP = """
import json as _json
import os as _os
from pathlib import Path as _Path

_LOG = _Path(_os.environ["FAKE_ANTHROPIC_LOG"])


class _Block:
    def __init__(self, type, text=None, id=None, name=None, input=None):
        self.type = type
        self.text = text
        self.id = id
        self.name = name
        self.input = input


class _Usage:
    input_tokens = 3
    output_tokens = 4


class _Msg:
    def __init__(self, content):
        self.content = content
        self.usage = _Usage()


class _Messages:
    def __init__(self):
        self.calls = 0

    def create(self, *a, **k):
        self.calls += 1
        _LOG.write_text(_json.dumps({"calls": self.calls, "kwargs": k}, default=str))
        if self.calls == 1:
            assert k["tools"][0]["name"] == "file_read"
            return _Msg(
                [
                    _Block(
                        "tool_use",
                        id="toolu_1",
                        name="file_read",
                        input={"path": "policy.md"},
                    )
                ]
            )
        assert any(
            block.get("type") == "tool_result"
            and block.get("tool_use_id") == "toolu_1"
            and "content" in block
            for block in k["messages"][-1]["content"]
        )
        return _Msg([_Block("text", text='{"ok": true}')])


class Anthropic:
    def __init__(self, *a, **k):
        self.messages = _Messages()
"""


def test_file_read_tool_loop_when_constraints_allow_endpoint(
    tmp_path: Path,
) -> None:
    (tmp_path / "anthropic.py").write_text(_FAKE_ANTHROPIC_TOOL_LOOP)
    log_path = tmp_path / "anthropic-log.json"
    debug_dir = tmp_path / "debug"
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": "test-key",
        "ATP_METHOD_DEBUG_IO_DIR": str(debug_dir),
        "FAKE_ANTHROPIC_LOG": str(log_path),
        "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    server, endpoint = _serve_tools()
    try:
        request = {
            "version": "1.0",
            "task_id": "t-tools",
            "task": {"description": "Extract", "input_data": {}},
            "constraints": {"allowed_tools": ["file_read"]},
            "context": {"tools_endpoint": endpoint, "workspace_path": "/w"},
        }

        proc = _run_shim_raw(request, env)
        resp = json.loads(proc.stdout.decode())
    finally:
        server.shutdown()

    assert resp["status"] == "completed"
    assert json.loads(resp["artifacts"][0]["content"]) == {"ok": True}
    stderr_events = _stderr_json_lines(proc)
    assert len(stderr_events) == 1
    event = ATPEvent.model_validate({"sequence": 0, **stderr_events[0]})
    assert event.event_type == "tool_call"
    assert event.task_id == "t-tools"
    assert event.payload == {
        "tool": "file_read",
        "input": {"path": "policy.md"},
        "status": "success",
        "output": {"content": "policy line\n"},
    }
    logged = json.loads(log_path.read_text())
    assert logged["calls"] == 2
    assert logged["kwargs"]["tools"][0]["name"] == "file_read"
    debug_files = _debug_files_by_suffix(
        debug_dir,
        "t-tools",
        {"prompt.txt", "raw_tool_loop.json", "final_output.txt", "atp_response.json"},
    )
    raw_tool_loop = json.loads(debug_files["raw_tool_loop.json"].read_text())
    assert raw_tool_loop["steps"][0]["assistant"]["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "file_read",
            "input": {"path": "policy.md"},
        }
    ]
    assert raw_tool_loop["steps"][0]["tool_results"][0]["response"]["output"] == {
        "content": "policy line\n"
    }
    assert raw_tool_loop["steps"][1]["assistant"]["content"] == [
        {"type": "text", "text": '{"ok": true}'}
    ]
    assert raw_tool_loop["steps"][1]["tool_results"] == []


def test_file_read_tool_loop_stderr_event_includes_tool_error(
    tmp_path: Path,
) -> None:
    (tmp_path / "anthropic.py").write_text(_FAKE_ANTHROPIC_TOOL_LOOP)
    log_path = tmp_path / "anthropic-log.json"
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": "test-key",
        "FAKE_ANTHROPIC_LOG": str(log_path),
        "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    server, endpoint = _serve_tools(_ToolErrorHandler)
    try:
        request = {
            "version": "1.0",
            "task_id": "t-tool-error",
            "task": {"description": "Extract", "input_data": {}},
            "constraints": {"allowed_tools": ["file_read"]},
            "context": {"tools_endpoint": endpoint, "workspace_path": "/w"},
        }

        proc = _run_shim_raw(request, env)
    finally:
        server.shutdown()

    stderr_events = _stderr_json_lines(proc)
    assert len(stderr_events) == 1
    event = ATPEvent.model_validate({"sequence": 0, **stderr_events[0]})
    assert event.event_type == "tool_call"
    assert event.task_id == "t-tool-error"
    assert event.payload == {
        "tool": "file_read",
        "input": {"path": "policy.md"},
        "status": "error",
        "error": "policy not found",
    }


_FAKE_ANTHROPIC_ALWAYS_TOOL = """
class _Block:
    def __init__(self):
        self.type = "tool_use"
        self.id = "toolu_loop"
        self.name = "file_read"
        self.input = {"path": "policy.md"}


class _Msg:
    content = [_Block()]
    usage = None


class _Messages:
    def create(self, *a, **k):
        return _Msg()


class Anthropic:
    def __init__(self, *a, **k):
        self.messages = _Messages()
"""


def test_file_read_tool_loop_fails_contract_shaped_on_max_iterations(
    tmp_path: Path,
) -> None:
    (tmp_path / "anthropic.py").write_text(_FAKE_ANTHROPIC_ALWAYS_TOOL)
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": "test-key",
        "ANTHROPIC_TOOL_MAX_ITERATIONS": "2",
        "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    server, endpoint = _serve_tools()
    try:
        request = {
            "version": "1.0",
            "task_id": "t-loop",
            "task": {"description": "Extract", "input_data": {}},
            "constraints": {"allowed_tools": ["file_read"]},
            "context": {"tools_endpoint": endpoint},
        }

        resp = _run_shim(request, env)
    finally:
        server.shutdown()

    assert resp["task_id"] == "t-loop"
    assert resp["status"] == "failed"
    assert "tool" in resp["error"].lower()
    assert "iteration" in resp["error"].lower()
