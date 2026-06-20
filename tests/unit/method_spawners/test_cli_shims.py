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
