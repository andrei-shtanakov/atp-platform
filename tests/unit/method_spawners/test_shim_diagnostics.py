"""Failure-path diagnostics: error taxonomy + raw stream persistence.

Observability layer added after the glm-5.1 empty-run incident (~40% of
agentic runs reaped by the hard timeout with zero evidence left behind):
the shim now persists raw subprocess streams (``ATP_SHIM_RAW_DIR``) and
keeps the stderr tail in empty-output failures; the harness classifies
shim failure text onto the report_benchmark-v1 ``error_class`` enum so a
reaped hang (timeout) is separable from capability failure (test_failure).
"""

import importlib.util
import io
import json
import sys
from pathlib import Path

_SPAWNERS = Path(__file__).resolve().parents[3] / "method" / "spawners"

_REQUEST = '{"task_id":"t-1","task":{"description":"x","input_data":{}}}'


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SPAWNERS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_SPAWNERS))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def _drive_run(cli, monkeypatch, argv, parse_output):
    """Run cli.run() with a canned ATPRequest; return the parsed ATPResponse."""
    monkeypatch.setenv("X_MODEL", "m")
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(_REQUEST))
    buf = io.StringIO()
    monkeypatch.setattr(cli.sys, "stdout", buf)
    rc = cli.run(
        bin_env="X_BIN",
        default_bin=sys.executable,
        model_env="X_MODEL",
        default_provider="openai",
        argv=argv,
        parse_output=parse_output,
    )
    assert rc == 0
    return json.loads(buf.getvalue())


# --------------------------------------------------------------------------- #
#  Harness-side classifier (report_benchmark-v1 error_class enum)
# --------------------------------------------------------------------------- #


def test_classify_shim_error_taxonomy() -> None:
    from method.run_pipe_check import _classify_shim_error

    assert _classify_shim_error(None) is None
    assert _classify_shim_error("") is None
    assert _classify_shim_error("opencode timed out after 600.0s") == "timeout"
    assert _classify_shim_error("opencode invocation error: boom") == "crash"
    assert _classify_shim_error("opencode failed (rc=2): stderr text") == "crash"
    # Every remaining stable _cli_common infra prefix is crash too — shim bugs
    # must not masquerade as capability failures (v1 has no finer infra class).
    assert _classify_shim_error("opencode command build error: bad arg") == "crash"
    assert (
        _classify_shim_error("opencode output parse error: Expecting value") == "crash"
    )
    assert _classify_shim_error("invalid ATPRequest JSON on stdin: boom") == "crash"
    assert _classify_shim_error("OPENCODE_MODEL not set") == "crash"
    # Empty output is a capability signal, not infra — stays test_failure
    # via the status fallback (classifier returns None).
    assert _classify_shim_error("opencode produced no output text") is None
    assert _classify_shim_error("something else entirely") is None


def test_classifier_ignores_provider_words_inside_stderr_tail() -> None:
    """The empty-output message embeds a raw stderr tail; provider log lines
    like 'request timed out' inside it must NOT flip the class to timeout."""
    from method.run_pipe_check import _classify_shim_error

    msg = (
        "opencode produced no output text; "
        "stderr tail: 'upstream request timed out, failed (rc=1)'"
    )
    assert _classify_shim_error(msg) is None


# --------------------------------------------------------------------------- #
#  Shim-side diagnostics (_cli_common)
# --------------------------------------------------------------------------- #


def test_empty_output_failure_carries_stderr_tail(monkeypatch) -> None:
    cli = _load("_cli_common")
    script = "import sys; sys.stderr.write('HTTP 429 quota hint')"
    out = _drive_run(
        cli,
        monkeypatch,
        argv=lambda b, m, p: [*b, "-c", script],
        parse_output=lambda s: ("", None, None),
    )
    assert out["status"] == "failed"
    assert "produced no output text" in out["error"]
    assert "HTTP 429 quota hint" in out["error"]


def test_raw_streams_persisted_when_dir_set(monkeypatch, tmp_path) -> None:
    cli = _load("_cli_common")
    raw_dir = tmp_path / "raw"
    monkeypatch.setenv("ATP_SHIM_RAW_DIR", str(raw_dir))
    script = "import sys; sys.stdout.write('OUT'); sys.stderr.write('ERR')"
    out = _drive_run(
        cli,
        monkeypatch,
        argv=lambda b, m, p: [*b, "-c", script],
        parse_output=lambda s: (s, 1, 1),
    )
    assert out["status"] == "completed"
    assert (raw_dir / "t-1.stdout").read_bytes() == b"OUT"
    assert (raw_dir / "t-1.stderr").read_bytes() == b"ERR"


def test_raw_streams_not_written_without_dir(monkeypatch, tmp_path) -> None:
    cli = _load("_cli_common")
    monkeypatch.delenv("ATP_SHIM_RAW_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    out = _drive_run(
        cli,
        monkeypatch,
        argv=lambda b, m, p: [*b, "-c", "print('hi')"],
        parse_output=lambda s: (s, None, None),
    )
    assert out["status"] == "completed"
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
#  opencode state isolation (SQLite "database is locked" contention fix)
# --------------------------------------------------------------------------- #


def test_opencode_isolated_data_home_seeds_auth(monkeypatch, tmp_path) -> None:
    """Each invocation gets a fresh XDG_DATA_HOME with only auth.json copied
    from the operator's real data dir — concurrent runs must not share
    opencode's SQLite state ("database is locked" killed 20-40% of runs)."""
    oc = _load("opencode_shim")
    real = tmp_path / "real-data-home"
    (real / "opencode").mkdir(parents=True)
    (real / "opencode" / "auth.json").write_text('{"zen": "cred"}')
    (real / "opencode" / "sessions.db").write_text("shared state")
    monkeypatch.setenv("XDG_DATA_HOME", str(real))

    iso = Path(oc._isolated_data_home())
    try:
        assert iso != real
        assert (iso / "opencode" / "auth.json").read_text() == '{"zen": "cred"}'
        # Session/DB state must NOT leak into the isolated dir.
        assert not (iso / "opencode" / "sessions.db").exists()
    finally:
        import shutil

        shutil.rmtree(iso, ignore_errors=True)


def test_opencode_isolation_tolerates_missing_auth(monkeypatch, tmp_path) -> None:
    oc = _load("opencode_shim")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "empty"))
    iso = Path(oc._isolated_data_home())
    try:
        assert (iso / "opencode").is_dir()
        assert not (iso / "opencode" / "auth.json").exists()
    finally:
        import shutil

        shutil.rmtree(iso, ignore_errors=True)


def test_timeout_persists_partial_streams(monkeypatch, tmp_path) -> None:
    """A reaped hang must leave its partial streams behind — that is the
    only evidence distinguishing a provider stall from a silent agent."""
    cli = _load("_cli_common")
    raw_dir = tmp_path / "raw"
    monkeypatch.setenv("ATP_SHIM_RAW_DIR", str(raw_dir))
    monkeypatch.setattr(cli, "REQUEST_TIMEOUT_S", 0.5)
    script = "import sys,time; sys.stderr.write('init ok'); time.sleep(5)"
    out = _drive_run(
        cli,
        monkeypatch,
        argv=lambda b, m, p: [*b, "-u", "-c", script],
        parse_output=lambda s: (s, None, None),
    )
    assert out["status"] == "failed"
    assert "timed out after 0.5s" in out["error"]
    # Partial stderr captured up to the kill.
    assert (raw_dir / "t-1.stderr").exists()
