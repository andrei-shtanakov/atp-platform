import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

_SPAWNERS = Path(__file__).resolve().parents[3] / "method" / "spawners"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SPAWNERS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Add spawners dir to sys.path so relative imports work
    sys.path.insert(0, str(_SPAWNERS))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def test_model_arg_prefixes_provider_when_bare() -> None:
    cli = _load("_cli_common")
    assert cli.model_arg("gpt-5", "openai") == "openai/gpt-5"
    assert cli.model_arg("glm-5.1", "opencode") == "opencode/glm-5.1"
    # already provider-qualified → unchanged
    assert cli.model_arg("openai/gpt-5", "openai") == "openai/gpt-5"


def test_build_response_shape_and_token_total() -> None:
    cli = _load("_cli_common")
    r = cli.build_response("t1", "[]", 10, 4)
    assert r["status"] == "completed"
    assert r["task_id"] == "t1"
    assert r["version"] == "1.0"
    assert r["artifacts"][0]["content"] == "[]"
    assert r["metrics"]["total_tokens"] == 14
    assert r["metrics"]["cost_usd"] is None
    r2 = cli.build_response("t", "x", None, 4)  # partial → total None
    assert r2["metrics"]["total_tokens"] is None


def test_opencode_parse_output_text_and_tokens() -> None:
    oc = _load("opencode_shim")
    stdout = (
        '{"type":"text","part":{"text":"["}}\n'
        '{"type":"text","part":{"text":"]"}}\n'
        '{"type":"step_finish","part":{"tokens":{"total":20,"input":15,"output":5}}}\n'
    )
    text, in_tok, out_tok = oc._parse(stdout)
    assert text == "[]"
    assert (in_tok, out_tok) == (15, 5)


def _run_shim(shim: str, env_extra: dict[str, str]) -> dict:
    # Force the binary to a non-existent command so the subprocess errors out
    # without any network/CLI dependency.
    env = dict(os.environ)
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(_SPAWNERS / f"{shim}.py")],
        input=json.dumps(
            {"task_id": "t", "task": {"description": "x", "input_data": {}}}
        ).encode(),
        capture_output=True,
        env=env,
        timeout=30,
    )
    return json.loads(proc.stdout.decode())


def test_opencode_shim_fails_when_binary_missing() -> None:
    out = _run_shim(
        "opencode_shim",
        {"OPENCODE_BIN": "definitely-not-a-real-bin-xyz", "OPENCODE_MODEL": "glm-5.1"},
    )
    assert out["status"] == "failed"
    assert "invocation error" in out["error"] or "failed" in out["error"]


def test_opencode_shim_fails_without_model() -> None:
    out = _run_shim("opencode_shim", {"OPENCODE_MODEL": ""})
    assert out["status"] == "failed"
    assert "OPENCODE_MODEL not set" in out["error"]


def test_run_timeout_yields_failed(monkeypatch) -> None:
    import io

    cli = _load("_cli_common")

    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(cli.subprocess, "run", _raise)
    monkeypatch.setenv("X_MODEL", "m")
    monkeypatch.setattr(
        cli.sys,
        "stdin",
        io.StringIO('{"task_id":"t","task":{"description":"x","input_data":{}}}'),
    )
    buf = io.StringIO()
    monkeypatch.setattr(cli.sys, "stdout", buf)
    rc = cli.run(
        bin_env="X_BIN",
        default_bin="true",
        model_env="X_MODEL",
        default_provider="openai",
        argv=lambda b, m, p: [*b, p],
        parse_output=lambda s: ("", None, None),
    )
    import json as _json

    out = _json.loads(buf.getvalue())
    assert rc == 0
    assert out["status"] == "failed"
    assert "timed out" in out["error"]


def test_pi_parse_output_assistant_text_and_usage() -> None:
    pi = _load("pi_shim")
    stdout = (
        '{"type":"message_start","message":{"role":"assistant","content":[],'
        '"usage":{"input":0,"output":0,"totalTokens":0}}}\n'
        '{"type":"message_end","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"[]"}],'
        '"usage":{"input":12,"output":3,"totalTokens":15}}}\n'
    )
    text, in_tok, out_tok = pi._parse(stdout)
    assert text == "[]"
    assert (in_tok, out_tok) == (12, 3)


def test_pi_shim_fails_when_binary_missing() -> None:
    out = _run_shim(
        "pi_shim", {"PI_BIN": "definitely-not-a-real-bin-xyz", "PI_MODEL": "gpt-5"}
    )
    assert out["status"] == "failed"
    assert "invocation error" in out["error"] or "failed" in out["error"]


def test_opencode_parse_tolerates_non_dict_jsonl_line() -> None:
    # A valid-JSON line that is not an object (e.g. "[]") must be skipped,
    # not crash the parser via .get() on a list.
    oc = _load("opencode_shim")
    stdout = '[]\n42\n"plain"\n{"type":"text","part":{"text":"ok"}}\n'
    text, in_tok, out_tok = oc._parse(stdout)
    assert text == "ok"
    assert (in_tok, out_tok) == (None, None)


def test_pi_parse_tolerates_non_dict_jsonl_line() -> None:
    pi = _load("pi_shim")
    stdout = (
        "[]\n"
        '{"type":"message_end","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"ok"}],"usage":{"input":1,"output":2}}}\n'
    )
    text, in_tok, out_tok = pi._parse(stdout)
    assert text == "ok"
    assert (in_tok, out_tok) == (1, 2)


def test_run_parse_error_yields_failed(monkeypatch) -> None:
    # parse_output raising must become a contract-shaped failed response,
    # never an uncaught crash.
    import io

    cli = _load("_cli_common")

    class _Proc:
        returncode = 0
        stdout = b"whatever"
        stderr = b""

    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _Proc())
    monkeypatch.setenv("X_MODEL", "m")
    monkeypatch.setattr(
        cli.sys,
        "stdin",
        io.StringIO('{"task_id":"t","task":{"description":"x","input_data":{}}}'),
    )
    buf = io.StringIO()
    monkeypatch.setattr(cli.sys, "stdout", buf)

    def _boom(_s: str):
        raise ValueError("kaboom")

    rc = cli.run(
        bin_env="X_BIN",
        default_bin="true",
        model_env="X_MODEL",
        default_provider="openai",
        argv=lambda b, m, p: [*b, p],
        parse_output=_boom,
    )
    out = json.loads(buf.getvalue())
    assert rc == 0
    assert out["status"] == "failed"
    assert "output parse error" in out["error"]
