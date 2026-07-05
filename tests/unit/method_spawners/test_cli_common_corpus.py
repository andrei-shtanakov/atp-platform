"""Corpus-mode helpers shared by CLI spawner shims (Path A)."""

import importlib.util
from pathlib import Path

_SPAWNERS = Path(__file__).resolve().parents[3] / "method" / "spawners"

_spec = importlib.util.spec_from_file_location(
    "_cli_common_corpus_under_test", _SPAWNERS / "_cli_common.py"
)
assert _spec and _spec.loader
_cli_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cli_common)

corpus_workspace = _cli_common.corpus_workspace
normalize_citation_paths = _cli_common.normalize_citation_paths


def test_corpus_workspace_returns_root_for_corpus_run() -> None:
    request = {
        "task": {"input_data": {"run_mode": "read_only_corpus"}},
        "context": {"workspace_path": "/tmp/x/.atp-runs/t1/corpus-1"},
    }
    assert corpus_workspace(request) == "/tmp/x/.atp-runs/t1/corpus-1"


def test_corpus_workspace_none_without_run_mode() -> None:
    # workspace_path alone is not enough — only read_only_corpus runs
    # switch the shim into corpus mode.
    request = {
        "task": {"input_data": {}},
        "context": {"workspace_path": "/tmp/x"},
    }
    assert corpus_workspace(request) is None


def test_corpus_workspace_none_without_workspace() -> None:
    request = {
        "task": {"input_data": {"run_mode": "read_only_corpus"}},
        "context": {},
    }
    assert corpus_workspace(request) is None


def test_corpus_workspace_none_on_missing_keys() -> None:
    assert corpus_workspace({}) is None


def test_corpus_workspace_none_on_null_context() -> None:
    # The CLI adapter serializes context: null when the request has none.
    request = {
        "task": {"input_data": {"run_mode": "read_only_corpus"}},
        "context": None,
    }
    assert corpus_workspace(request) is None


def test_normalize_strips_workspace_prefix() -> None:
    text = '{"path": "/ws/root/policy-current.md", "n": 1}'
    assert normalize_citation_paths(text, "/ws/root") == (
        '{"path": "policy-current.md", "n": 1}'
    )


def test_normalize_strips_nested_and_multiple() -> None:
    text = '"/ws/root/archive/policy-2023.md" and "/ws/root/policy-current.md"'
    out = normalize_citation_paths(text, "/ws/root")
    assert out == '"archive/policy-2023.md" and "policy-current.md"'


def test_normalize_handles_trailing_slash_workspace() -> None:
    assert normalize_citation_paths('"/ws/root/a.md"', "/ws/root/") == '"a.md"'


def test_normalize_leaves_relative_paths_alone() -> None:
    text = '{"path": "policy-current.md"}'
    assert normalize_citation_paths(text, "/ws/root") == text


def _corpus_request(workspace: str) -> dict:
    return {
        "version": "1.0",
        "task_id": "corpus-1",
        "task": {
            "description": "Extract requirements with citations.",
            "input_data": {
                "run_mode": "read_only_corpus",
                "artifact_corpus": {
                    "id": "c1",
                    "files": ["policy-current.md", "archive/policy-2023.md"],
                },
            },
        },
        "context": {"workspace_path": workspace},
        "constraints": {},
    }


def _run_pi_shim(request: dict, extra_env: dict) -> dict:
    import json
    import os
    import subprocess
    import sys

    shim = _SPAWNERS / "pi_shim.py"
    fake = Path(__file__).resolve().parent / "fixtures" / "fake_pi.py"
    env = {
        **os.environ,
        "PI_BIN": f"{sys.executable} {fake}",
        "PI_MODEL": "gpt-5",
        **extra_env,
    }
    proc = subprocess.run(
        [sys.executable, str(shim)],
        input=json.dumps(request).encode(),
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    return json.loads(proc.stdout.decode())


def test_pi_corpus_run_confine_flags_before_prompt_and_cwd(tmp_path) -> None:
    import json

    workspace = tmp_path / "corpus"
    workspace.mkdir()
    (workspace / "policy-current.md").write_text("deadline: 2026-08-01\n")
    log_path = tmp_path / "invocation.json"

    resp = _run_pi_shim(
        _corpus_request(str(workspace)),
        extra_env={"ATP_FAKE_PI_LOG": str(log_path)},
    )

    invocation = json.loads(log_path.read_text())
    assert invocation["cwd"] == str(workspace)
    argv = invocation["argv"]
    # Read-only confinement: only the `read` tool stays enabled.
    assert "--tools" in argv
    assert argv[argv.index("--tools") + 1] == "read"
    # The argv contract puts the prompt LAST; confinement flags precede it.
    assert argv.index("--tools") < len(argv) - 1
    assert "Read-only corpus" in argv[-1]
    assert resp["status"] == "completed"
    # Absolute citation path normalized to corpus-relative.
    content = resp["artifacts"][0]["content"]
    assert str(workspace) not in content
    assert "policy-current.md" in content


def test_pi_non_corpus_run_has_no_tools_flag(tmp_path) -> None:
    import json

    log_path = tmp_path / "invocation.json"
    request = {
        "version": "1.0",
        "task_id": "t1",
        "task": {"description": "Review the diff.", "input_data": {}},
        "constraints": {},
    }
    resp = _run_pi_shim(request, extra_env={"ATP_FAKE_PI_LOG": str(log_path)})
    invocation = json.loads(log_path.read_text())
    assert "--tools" not in invocation["argv"]
    import os

    assert invocation["cwd"] == os.getcwd()
    assert resp["status"] == "completed"


def _run_opencode_shim(request: dict, extra_env: dict) -> dict:
    import json
    import os
    import subprocess
    import sys

    shim = _SPAWNERS / "opencode_shim.py"
    fake = Path(__file__).resolve().parent / "fixtures" / "fake_opencode.py"
    env = {
        **os.environ,
        "OPENCODE_BIN": f"{sys.executable} {fake}",
        "OPENCODE_MODEL": "glm-5.1",
        **extra_env,
    }
    proc = subprocess.run(
        [sys.executable, str(shim)],
        input=json.dumps(request).encode(),
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    return json.loads(proc.stdout.decode())


def test_opencode_corpus_run_injects_readonly_config_and_cwd(tmp_path) -> None:
    import json

    workspace = tmp_path / "corpus"
    workspace.mkdir()
    (workspace / "policy-current.md").write_text("deadline: 2026-08-01\n")
    log_path = tmp_path / "invocation.json"

    resp = _run_opencode_shim(
        _corpus_request(str(workspace)),
        extra_env={"ATP_FAKE_OPENCODE_LOG": str(log_path)},
    )

    invocation = json.loads(log_path.read_text())
    assert invocation["cwd"] == str(workspace)
    # Confinement is config-based: the subprocess must see an INJECTED
    # XDG_CONFIG_HOME whose opencode.json is the read-only profile.
    cfg_home = invocation["xdg_config_home"]
    assert cfg_home, "corpus run must inject an isolated XDG_CONFIG_HOME"
    # The shim removes its temp config home after the run; fake_opencode
    # snapshots the config content into the log at invocation time.
    config = invocation["config"]
    assert config is not None, "injected opencode.json missing at invocation"
    assert config["permission"]["edit"] == "deny"
    assert config["permission"]["write"] == "deny"
    assert config["permission"]["bash"] == "deny"
    assert config["permission"]["webfetch"] == "deny"
    assert config["permission"]["read"] == "allow"
    assert resp["status"] == "completed"
    # Absolute citation path normalized to corpus-relative.
    content = resp["artifacts"][0]["content"]
    assert str(workspace) not in content
    assert "policy-current.md" in content


def test_opencode_non_corpus_run_keeps_user_config_surface(tmp_path) -> None:
    import json
    import os

    log_path = tmp_path / "invocation.json"
    request = {
        "version": "1.0",
        "task_id": "t1",
        "task": {"description": "Review the diff.", "input_data": {}},
        "constraints": {},
    }
    resp = _run_opencode_shim(
        request, extra_env={"ATP_FAKE_OPENCODE_LOG": str(log_path)}
    )
    invocation = json.loads(log_path.read_text())
    # Non-corpus runs must NOT swap the config home (inline behavior and
    # provider config untouched); cwd inherited from the caller.
    assert invocation["xdg_config_home"] == os.environ.get("XDG_CONFIG_HOME")
    assert invocation["cwd"] == os.getcwd()
    assert resp["status"] == "completed"
