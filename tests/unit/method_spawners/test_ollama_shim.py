"""Tests for the ollama spawner shim (offline; urllib.urlopen is mocked)."""

import importlib.util
import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

SHIM_PATH = (
    Path(__file__).resolve().parents[3] / "method" / "spawners" / "ollama_shim.py"
)


def _load_shim() -> Any:
    """Import the shim module by path (it lives outside the package tree)."""
    spec = importlib.util.spec_from_file_location("ollama_shim", SHIM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shim = _load_shim()

_CANNED_TEXT = json.dumps(
    [{"rule_id": "sql-injection", "anchor": 'f"SELECT', "severity": "critical"}]
)
_CANNED_CHAT = {
    "model": "qwen2.5:7b",
    "message": {"role": "assistant", "content": _CANNED_TEXT},
    "prompt_eval_count": 800,
    "eval_count": 120,
    "done": True,
}


class _FakeResponse:
    """Context-manager stand-in for urllib's HTTPResponse."""

    def __init__(self, payload: dict) -> None:
        self._buf = io.BytesIO(json.dumps(payload).encode())

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._buf.read()


def test_build_response_success() -> None:
    request = {"version": "1.0", "task_id": "t2"}
    resp = shim.build_response(request, _CANNED_CHAT)
    assert resp["task_id"] == "t2"
    assert resp["status"] == "completed"
    arts = resp["artifacts"]
    assert len(arts) == 1
    assert arts[0]["path"] == "review.md"
    assert arts[0]["content"] == _CANNED_TEXT
    findings = json.loads(arts[0]["content"])
    assert findings[0]["rule_id"] == "sql-injection"
    assert resp["metrics"]["total_tokens"] == 920
    assert resp["metrics"]["input_tokens"] == 800
    assert resp["metrics"]["output_tokens"] == 120
    assert resp["metrics"]["cost_usd"] is None


def test_build_response_missing_token_counts() -> None:
    request = {"version": "1.0", "task_id": "t9"}
    resp = shim.build_response(request, {"message": {"content": "[]"}})
    assert resp["status"] == "completed"
    # Neither count present => total is unknown (None), per-field default to 0.
    assert resp["metrics"]["total_tokens"] is None
    assert resp["metrics"]["input_tokens"] == 0
    assert resp["metrics"]["output_tokens"] == 0


def test_main_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    monkeypatch.setattr(
        shim.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(_CANNED_CHAT)
    )
    request = {
        "version": "1.0",
        "task_id": "t2",
        "task": {"description": "Review the diff against the rules."},
        "context": {"artifacts": []},
    }
    monkeypatch.setattr(shim.sys, "stdin", io.StringIO(json.dumps(request)))
    out = io.StringIO()
    monkeypatch.setattr(shim.sys, "stdout", out)

    assert shim.main() == 0
    resp = json.loads(out.getvalue())
    assert resp["task_id"] == "t2"
    assert resp["status"] == "completed"
    assert resp["artifacts"][0]["content"] == _CANNED_TEXT
    assert resp["metrics"]["total_tokens"] == 920
    assert resp["metrics"]["cost_usd"] is None


def test_main_missing_model_emits_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    def _boom(*a: object, **k: object) -> None:  # pragma: no cover - must not run
        raise AssertionError("urlopen must not be called when OLLAMA_MODEL is unset")

    monkeypatch.setattr(shim.urllib.request, "urlopen", _boom)
    request = {"version": "1.0", "task_id": "t1", "task": {"description": "x"}}
    monkeypatch.setattr(shim.sys, "stdin", io.StringIO(json.dumps(request)))
    out = io.StringIO()
    monkeypatch.setattr(shim.sys, "stdout", out)

    assert shim.main() == 0
    resp = json.loads(out.getvalue())
    assert resp["status"] == "failed"
    assert "OLLAMA_MODEL" in resp["error"]


def test_main_urllib_error_emits_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")

    def _raise(*a: object, **k: object) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(shim.urllib.request, "urlopen", _raise)
    request = {"version": "1.0", "task_id": "t4", "task": {"description": "x"}}
    monkeypatch.setattr(shim.sys, "stdin", io.StringIO(json.dumps(request)))
    out = io.StringIO()
    monkeypatch.setattr(shim.sys, "stdout", out)

    assert shim.main() == 0
    resp = json.loads(out.getvalue())
    assert resp["task_id"] == "t4"
    assert resp["status"] == "failed"
    assert "ollama request error" in resp["error"].lower()


def test_main_invalid_stdin_emits_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    monkeypatch.setattr(shim.sys, "stdin", io.StringIO("not json at all"))
    out = io.StringIO()
    monkeypatch.setattr(shim.sys, "stdout", out)

    assert shim.main() == 0
    resp = json.loads(out.getvalue())
    assert resp["status"] == "failed"
    assert "invalid" in resp["error"].lower()
