import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SPAWNERS = Path(__file__).resolve().parents[3] / "method" / "spawners"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SPAWNERS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_response_normalizes_openai_compat_payload() -> None:
    mod = _load("_openai_compat")
    raw = {
        "choices": [{"message": {"content": "[]"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }
    resp = mod.build_response({"task_id": "t1"}, raw)
    assert resp["status"] == "completed"
    assert resp["task_id"] == "t1"
    assert resp["artifacts"][0]["content"] == "[]"
    assert resp["metrics"]["total_tokens"] == 12
    assert resp["metrics"]["cost_usd"] is None


def test_build_response_missing_usage_is_none() -> None:
    mod = _load("_openai_compat")
    resp = mod.build_response(
        {"task_id": "t"}, {"choices": [{"message": {"content": "x"}}]}
    )
    assert resp["metrics"]["total_tokens"] is None


def _run_shim(shim: str, env_extra: dict[str, str]) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(_SPAWNERS / f"{shim}.py")],
        input=json.dumps({"task_id": "t"}).encode(),
        capture_output=True,
        env=env,
        timeout=30,
    )
    return json.loads(proc.stdout.decode())


def test_mimo_shim_fails_clearly_without_key() -> None:
    out = _run_shim("mimo_shim", {})
    assert out["status"] == "failed"
    assert "MIMO_API_KEY not set" in out["error"]


def test_qwen_shim_fails_clearly_without_key() -> None:
    out = _run_shim("qwen_shim", {})
    assert out["status"] == "failed"
    assert "QWEN_API_KEY not set" in out["error"]


def test_mimo_shim_fails_clearly_without_model() -> None:
    out = _run_shim("mimo_shim", {"MIMO_API_KEY": "x"})
    assert out["status"] == "failed"
    assert "MIMO_MODEL not set" in out["error"]


def test_call_builds_single_v1_chat_completions_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """URL must not duplicate the version segment when host already contains /v1."""
    mod = _load("_openai_compat")
    captured: dict[str, str] = {}

    class _Resp:
        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"[]"}}]}'

    def _fake_urlopen(req: object, timeout: float | None = None) -> _Resp:
        captured["url"] = req.full_url  # type: ignore[attr-defined]
        return _Resp()

    monkeypatch.setattr(mod.urllib.request, "urlopen", _fake_urlopen)
    mod._call("https://token-plan-sgp.xiaomimimo.com/v1", "k", "m", "p")
    assert (
        captured["url"] == "https://token-plan-sgp.xiaomimimo.com/v1/chat/completions"
    )
    assert "/v1/v1/" not in captured["url"]
