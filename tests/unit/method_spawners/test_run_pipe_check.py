"""Tests for the pipe-check harness CLI error paths (Phase A-2)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[3] / "method" / "run_pipe_check.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(HARNESS), *args],
        capture_output=True,
        env=os.environ.copy(),
        timeout=60,
    )


def _any_valid_agent_id() -> str:
    # Error-path tests need a VALID --agents value but must not pin a specific
    # model string: catalog lifecycle (a model going retired) would break them
    # in the wrong place. Pick any id from the catalog-projected registry.
    from method.run_pipe_check import AGENTS

    assert AGENTS, "catalog-projected AGENTS registry is empty"
    return next(iter(AGENTS))


def test_unknown_task_type_exits_2_with_stderr() -> None:
    # Fail fast on an unknown --task-type: one-line stderr + exit 2 (not a
    # traceback), before any agent runs.
    proc = _run(
        [
            "--agents",
            _any_valid_agent_id(),
            "--task-type",
            "bogus",
            "--dry-run",
        ]
    )
    assert proc.returncode == 2
    err = proc.stderr.decode()
    assert "unknown task_type" in err
    assert "Traceback" not in err


def test_unknown_agent_exits_2() -> None:
    proc = _run(["--agents", "nope", "--task-type", "review", "--dry-run"])
    assert proc.returncode == 2
    assert "Unknown agent" in proc.stderr.decode()


def test_dashboard_replace_without_to_dashboard_exits_2() -> None:
    # --dashboard-replace alone is a misconfiguration: fail fast (exit 2),
    # not silently no-op.
    proc = _run(["--dashboard-replace", "--agents", "claude_code", "--dry-run"])
    assert proc.returncode == 2
    err = proc.stderr.decode()
    assert "--dashboard-replace requires --to-dashboard" in err
    assert "Traceback" not in err


def test_task_type_disagreeing_with_suite_exits_2() -> None:
    # Running the req-extraction suite without --task-type (defaults to
    # 'review') would stamp benchmark_id=code-review into the lock and
    # mislabel the arbiter export. Guard fails fast (exit 2).
    proc = _run(
        [
            "--case-dir",
            "method/cases/req-extraction",
            "--agents",
            _any_valid_agent_id(),
            "--dry-run",
        ]
    )
    assert proc.returncode == 2
    err = proc.stderr.decode()
    assert "disagrees with the suite" in err
    assert "req-extraction" in err
    assert "Traceback" not in err


def test_suite_task_type_reads_homogeneous_cases(tmp_path: Path) -> None:
    from method.run_pipe_check import _suite_task_type

    (tmp_path / "a.yaml").write_text("id: a\ntask_type: req-extraction\n")
    (tmp_path / "b.yaml").write_text("id: b\ntask_type: req-extraction\n")
    assert _suite_task_type(tmp_path) == "req-extraction"


def test_suite_task_type_none_when_absent(tmp_path: Path) -> None:
    from method.run_pipe_check import _suite_task_type

    (tmp_path / "a.yaml").write_text("id: a\n")
    assert _suite_task_type(tmp_path) is None


def test_suite_task_type_mixed_raises(tmp_path: Path) -> None:
    from method.run_pipe_check import _suite_task_type

    (tmp_path / "a.yaml").write_text("id: a\ntask_type: review\n")
    (tmp_path / "b.yaml").write_text("id: b\ntask_type: req-extraction\n")
    with pytest.raises(ValueError, match="mixes task_types"):
        _suite_task_type(tmp_path)


def test_suite_task_type_malformed_yaml_raises_valueerror(tmp_path: Path) -> None:
    # A malformed case must surface as a ValueError (→ one-line CLI error +
    # exit 2 upstream), never a raw YAMLError traceback.
    from method.run_pipe_check import _suite_task_type

    (tmp_path / "bad.yaml").write_text("id: a\n  bad: : indent\n:::\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        _suite_task_type(tmp_path)


def _completed_test_result() -> object:
    """Build a minimal completed TestResult for a real code-review case."""
    from datetime import UTC, datetime

    from atp_method.loader import load_case

    from atp.core.results import RunResult, TestResult
    from atp.protocol import ArtifactFile, ATPResponse, ResponseStatus

    case_path = (
        Path(__file__).resolve().parents[3]
        / "method"
        / "cases"
        / "code-review"
        / "case-code-review-sqli-moderate-001.yaml"
    )
    test_def = load_case(case_path)
    response = ATPResponse(
        task_id=test_def.id,
        status=ResponseStatus.COMPLETED,
        artifacts=[ArtifactFile(path="review.md", content="[]")],
    )
    run = RunResult(
        test_id=test_def.id,
        run_number=1,
        response=response,
        events=[],
        end_time=datetime.now(tz=UTC),
    )
    return TestResult(test=test_def, runs=[run])


def _multi_run_test_result(n: int, status: str = "completed") -> object:
    """A TestResult with ``n`` runs of the given response status.

    Each run carries a Metrics with 100 tokens + $0.01 so aggregation of
    spend across runs is assertable.
    """
    from datetime import UTC, datetime

    from atp_method.loader import load_case

    from atp.core.results import RunResult, TestResult
    from atp.protocol import ArtifactFile, ATPResponse, Metrics, ResponseStatus

    case_path = (
        Path(__file__).resolve().parents[3]
        / "method"
        / "cases"
        / "code-review"
        / "case-code-review-sqli-moderate-001.yaml"
    )
    test_def = load_case(case_path)
    runs = []
    for i in range(n):
        response = ATPResponse(
            task_id=test_def.id,
            status=ResponseStatus(status),
            artifacts=[ArtifactFile(path="review.md", content="[]")],
            metrics=Metrics(total_tokens=100, cost_usd=0.01),
        )
        runs.append(
            RunResult(
                test_id=test_def.id,
                run_number=i + 1,
                response=response,
                events=[],
                end_time=datetime.now(tz=UTC),
            )
        )
    return TestResult(test=test_def, runs=runs)


class _QueueEval:
    """Stub evaluator returning a scripted (passed, malformed) per call."""

    def __init__(self, verdicts: list[tuple[bool, bool]]) -> None:
        self._verdicts = list(verdicts)

    async def evaluate(self, test_def, response, events, assertion):  # type: ignore[no-untyped-def]
        from atp.evaluators.base import EvalCheck, EvalResult

        passed, malformed = self._verdicts.pop(0)
        return EvalResult(
            evaluator="stub",
            checks=[
                EvalCheck(
                    name="critical_check",
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    details={"malformed": malformed},
                )
            ],
        )


def test_grade_case_surfaces_continuous_metrics() -> None:
    import anyio

    from atp.evaluators.base import EvalCheck, EvalResult
    from method.run_pipe_check import _grade_case

    class _StubEval:
        async def evaluate(self, test_def, response, events, assertion):  # type: ignore[no-untyped-def]
            return EvalResult(
                evaluator="stub",
                checks=[
                    EvalCheck(
                        name="critical_check",
                        passed=True,
                        score=1.0,
                        details={
                            "malformed": False,
                            "recall": 0.5,
                            "precision": 0.75,
                            "fp_count": 1,
                        },
                    )
                ],
            )

    tr = _completed_test_result()
    base = anyio.run(_grade_case, _StubEval(), tr, "moderate", False)
    assert base["recall"] == 0.5
    assert base["precision"] == 0.75
    assert base["fp_count"] == 1


def test_grade_case_majority_pass_across_runs() -> None:
    # 2 of 3 runs pass → the case passes (majority vote), and spend is
    # summed across all 3 runs (runs=3 costs 3x — the bug this fixes).
    import anyio

    from method.run_pipe_check import _grade_case

    tr = _multi_run_test_result(3)
    ev = _QueueEval([(True, False), (False, False), (True, False)])
    base = anyio.run(_grade_case, ev, tr, "severe", False)
    assert base["critical_pass"] is True
    assert base["runs_graded"] == 3
    assert base["run_pass_count"] == 2
    assert base["tokens"] == 300
    assert base["cost_usd"] == pytest.approx(0.03)


def test_grade_case_minority_pass_fails() -> None:
    # 1 of 3 runs passes → the case fails (not a trustworthy routing signal).
    import anyio

    from method.run_pipe_check import _grade_case

    tr = _multi_run_test_result(3)
    ev = _QueueEval([(True, False), (False, False), (False, False)])
    base = anyio.run(_grade_case, ev, tr, "severe", False)
    assert base["critical_pass"] is False
    assert base["run_pass_count"] == 1


def test_grade_case_majority_malformed_flags_malformed() -> None:
    import anyio

    from method.run_pipe_check import _grade_case

    tr = _multi_run_test_result(3)
    ev = _QueueEval([(False, True), (False, True), (True, False)])
    base = anyio.run(_grade_case, ev, tr, "severe", False)
    assert base["critical_pass"] is False
    assert base["malformed"] is True


def test_grade_case_partial_infra_grades_completed_runs() -> None:
    # 1 run infra-failed (timeout), 2 completed and passed → graded on the
    # 2 good runs; the case is NOT marked infra (some output was gradeable).
    import anyio

    from atp.protocol import ResponseStatus
    from method.run_pipe_check import _grade_case

    tr = _multi_run_test_result(3)
    tr.runs[0].response.status = ResponseStatus("timeout")
    ev = _QueueEval([(True, False), (True, False)])  # only completed runs graded
    base = anyio.run(_grade_case, ev, tr, "severe", False)
    assert base["critical_pass"] is True
    assert base["runs_graded"] == 2
    assert base["error_class"] is None
    # Spend still summed across ALL executed runs (the timeout burned tokens too).
    assert base["tokens"] == 300


def test_grade_case_all_runs_infra_fail_sets_error_class() -> None:
    import anyio

    from method.run_pipe_check import _grade_case

    tr = _multi_run_test_result(3, status="timeout")
    base = anyio.run(_grade_case, object(), tr, "severe", False)
    assert base["critical_pass"] is False
    assert base["runs_graded"] == 0
    assert base["error_class"] == "timeout"


@pytest.mark.parametrize(
    ("status", "expected_error_class"),
    [
        ("failed", "test_failure"),
        ("timeout", "timeout"),
        ("cancelled", "other"),
        ("partial", "other"),
    ],
)
def test_grade_case_normalizes_non_completed_status_error_class(
    status: str, expected_error_class: str
) -> None:
    import anyio

    from atp.protocol import ResponseStatus
    from method.run_pipe_check import _grade_case

    tr = _completed_test_result()
    tr.runs[0].response.status = ResponseStatus(status)

    base = anyio.run(_grade_case, object(), tr, "moderate", False)

    assert base["critical_pass"] is False
    assert base["error_class"] == expected_error_class


def test_grade_case_normalizes_missing_run_error_class() -> None:
    import anyio

    from method.run_pipe_check import _grade_case

    tr = _completed_test_result()
    tr.runs = []

    base = anyio.run(_grade_case, object(), tr, "moderate", False)

    assert base["critical_pass"] is False
    assert base["error_class"] == "test_failure"


def test_export_to_dashboard_imports_written_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--to-dashboard wiring reuses the bridge to land reports in the store."""
    import json

    import anyio

    from method.run_pipe_check import _export_to_dashboard

    # Point the bridge's default DB at a throwaway sqlite (init_database reads
    # ATP_DATABASE_URL when no url is passed) so the dev DB is untouched.
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    monkeypatch.setenv("ATP_DATABASE_URL", db_url)
    (tmp_path / "report_benchmark_claude_code.json").write_text(
        json.dumps(
            {
                "payload_version": "1.0.0",
                "run_id": "rpc-run-1",
                "benchmark_id": "code-review",
                "agent_id": "claude_code",
                "ts": "2026-06-19T10:00:00+00:00",
                "score": 0.9,
                "score_components": {"critical_pass_rate": 0.9},
                "duration_seconds": 1.0,
            }
        )
    )

    path_before = list(sys.path)
    anyio.run(_export_to_dashboard, tmp_path, False)
    assert sys.path == path_before  # helper must not mutate global sys.path

    from sqlalchemy import select

    from atp.dashboard import init_database
    from atp.dashboard.models import SuiteExecution

    async def _check() -> list[str]:
        db = await init_database(url=db_url)
        async with db.session() as session:
            return list(
                (await session.execute(select(SuiteExecution.run_uuid))).scalars().all()
            )

    assert anyio.run(_check) == ["rpc-run-1"]


def test_write_case_details_one_line_per_case(tmp_path: Path) -> None:
    import json

    from method.run_pipe_check import _write_case_details

    case_results = [
        {"case_id": "a", "recall": 0.5, "precision": 0.75, "fp_count": 1},
        {"case_id": "b", "recall": 1.0, "precision": 1.0, "fp_count": 0},
    ]
    out = tmp_path / "case_details_stub.jsonl"
    _write_case_details(out, case_results)
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    for line, expected in zip(lines, case_results, strict=True):
        obj = json.loads(line)
        assert obj == expected
        assert "recall" in obj
        assert "precision" in obj
        assert "fp_count" in obj


def test_agents_registry_builds_harness_at_model_ids() -> None:
    from method.run_pipe_check import AGENTS

    assert "claude_code@claude-sonnet-4-6" in AGENTS
    assert "anthropic_api@claude-sonnet-4-6" in AGENTS
    assert "deepseek@deepseek-chat" in AGENTS
    assert "ollama@qwen2.5:14b" in AGENTS
    # codex pinned to gpt-5.5 (gpt-5-codex is unavailable on a ChatGPT account) —
    # it is arbiter's second routable key.
    assert "codex_cli@gpt-5.5" in AGENTS
    spec = AGENTS["ollama@qwen2.5:14b"]
    assert spec["model"] == "qwen2.5:14b"
    assert spec["model_env"] == "OLLAMA_MODEL"
    assert spec["harness"] == "ollama"
    assert spec["shim"].endswith("ollama_shim.py")


def test_default_registry_has_no_safe_id_collisions() -> None:
    from method.run_pipe_check import AGENTS, _safe_id_collision

    assert _safe_id_collision(list(AGENTS)) is None


def test_safe_id_collision_detects_collapsing_ids() -> None:
    from method.run_pipe_check import _safe_id_collision

    # Two distinct ids that collapse to the same file stem (ollama_qwen2_5_14b).
    pair = _safe_id_collision(["ollama@qwen2.5:14b", "ollama@qwen2_5_14b"])
    assert pair == ("ollama@qwen2.5:14b", "ollama@qwen2_5_14b")


def test_safe_agent_id_renders_filesystem_safe() -> None:
    from method.run_pipe_check import safe_agent_id

    assert safe_agent_id("ollama@qwen2.5:14b") == "ollama_qwen2_5_14b"
    assert (
        safe_agent_id("claude_code@claude-sonnet-4-6")
        == "claude_code_claude-sonnet-4-6"
    )


def test_legacy_bare_harness_id_is_unknown() -> None:
    # The old harness-only id no longer resolves; must exit 2.
    proc = _run(["--agents", "claude_code", "--task-type", "review", "--dry-run"])
    assert proc.returncode == 2
    assert "Unknown agent" in proc.stderr.decode()


def test_registry_has_sonnet_and_new_api_agents_no_opus() -> None:
    from method.run_pipe_check import AGENTS

    assert "claude_code@claude-sonnet-4-6" in AGENTS
    assert "anthropic_api@claude-sonnet-4-6" in AGENTS
    assert "mimo@mimo-v2.5-pro" in AGENTS
    assert "qwen@qwen3.6-plus" in AGENTS
    # opus fully retired
    assert not any("claude-opus-4-8" in a for a in AGENTS)
    assert AGENTS["mimo@mimo-v2.5-pro"]["model_env"] == "MIMO_MODEL"
    assert AGENTS["mimo@mimo-v2.5-pro"]["shim"].endswith("mimo_shim.py")
    assert AGENTS["qwen@qwen3.6-plus"]["shim"].endswith("qwen_shim.py")


def test_preflight_skips_mimo_qwen_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from method.run_pipe_check import _preflight

    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    assert _preflight("mimo@mimo-v2.5-pro") == "MIMO_API_KEY not set"
    assert _preflight("qwen@qwen3.6-plus") == "QWEN_API_KEY not set"


def test_registry_has_pi_and_opencode() -> None:
    from method.run_pipe_check import AGENTS

    assert "pi@gpt-5" in AGENTS
    assert "opencode@glm-5.1" in AGENTS
    assert AGENTS["pi@gpt-5"]["model_env"] == "PI_MODEL"
    assert AGENTS["pi@gpt-5"]["shim"].endswith("pi_shim.py")
    assert AGENTS["opencode@glm-5.1"]["shim"].endswith("opencode_shim.py")


def test_preflight_skips_pi_opencode_without_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from method.run_pipe_check import _preflight

    monkeypatch.setenv("PI_BIN", "definitely-not-a-real-bin-xyz")
    monkeypatch.setenv("OPENCODE_BIN", "definitely-not-a-real-bin-xyz")
    assert "pi binary not found" in (_preflight("pi@gpt-5") or "")
    assert "opencode binary not found" in (_preflight("opencode@glm-5.1") or "")


def test_vendored_catalog_is_the_roster_source() -> None:
    # ADR-ECO-003: HARNESSES + AGENT_MODELS are projected from the vendored SSOT
    # catalog, not literals. The vendored file must exist and drive the sweep.
    from method.run_pipe_check import CATALOG_PATH, _load_agent_catalog

    assert CATALOG_PATH.exists(), "vendored agents-catalog.toml missing from repo"
    harnesses, agent_models = _load_agent_catalog()
    # Both arbiter routable join keys must be present in the tested sweep set.
    assert ("claude_code", "claude-sonnet-4-6") in agent_models
    assert ("codex_cli", "gpt-5.5") in agent_models
    # Harness map carries (shim, model_env) tuples used to build AGENTS.
    assert harnesses["claude_code"] == (
        "method/spawners/claude_code_shim.py",
        "CLAUDE_MODEL",
    )
    # No retired/opus model leaks into the roster.
    assert not any("opus" in model for _, model in agent_models)


def test_load_agent_catalog_only_includes_tested(tmp_path: Path) -> None:
    # An [[agents]] row with tested = false is excluded from the sweep set;
    # its harness is still registered (so AGENTS can reference it if promoted).
    from method.run_pipe_check import _load_agent_catalog

    catalog = (
        "[harnesses.foo]\n"
        'shim = "method/spawners/foo_shim.py"\n'
        'model_env = "FOO_MODEL"\n'
        "routable = false\n"
        "[[agents]]\n"
        'harness = "foo"\n'
        'model = "foo-1"\n'
        "tested = true\n"
        "[[agents]]\n"
        'harness = "foo"\n'
        'model = "foo-2"\n'
        "tested = false\n"
    )
    path = tmp_path / "cat.toml"
    path.write_text(catalog)
    harnesses, agent_models = _load_agent_catalog(path)
    assert harnesses == {"foo": ("method/spawners/foo_shim.py", "FOO_MODEL")}
    assert agent_models == [("foo", "foo-1")]


def test_load_agent_catalog_undeclared_harness_fails_loudly(tmp_path: Path) -> None:
    # A tested agent referencing a harness with no [harnesses.*] block must raise
    # a path-qualified error at load time, not a cryptic KeyError building AGENTS.
    from method.run_pipe_check import _load_agent_catalog

    catalog = (
        "[harnesses.foo]\n"
        'shim = "method/spawners/foo_shim.py"\n'
        'model_env = "FOO_MODEL"\n'
        "[[agents]]\n"
        'harness = "bar"\n'  # bar is never declared under [harnesses]
        'model = "bar-1"\n'
        "tested = true\n"
    )
    path = tmp_path / "cat.toml"
    path.write_text(catalog)
    with pytest.raises(ValueError, match="undeclared"):
        _load_agent_catalog(path)


def test_axis_by_id_skips_read_only_corpus_cases(tmp_path: Path) -> None:
    # The CLI-adapter path can't run read_only_corpus cases (no file_read/corpus
    # wiring); _axis_by_id must exclude them and _corpus_case_ids must find them.
    from method.run_pipe_check import _axis_by_id, _corpus_case_ids

    (tmp_path / "inline.yaml").write_text(
        "id: case-inline-001\naxis_level: severe\ntask_type: req-extraction\n"
    )
    (tmp_path / "corpus.yaml").write_text(
        "id: case-corpus-001\naxis_level: clean\nrun_mode: read_only_corpus\n"
        "task_type: req-extraction\n"
    )
    assert _corpus_case_ids(tmp_path) == {"case-corpus-001"}
    axis = _axis_by_id(tmp_path)
    assert axis == {"case-inline-001": "severe"}


def test_axis_by_id_includes_corpus_when_requested(tmp_path: Path) -> None:
    # Path A: corpus-capable harnesses run read_only_corpus cases; the
    # exclusion is now caller-controlled instead of unconditional.
    from method.run_pipe_check import _axis_by_id

    (tmp_path / "inline.yaml").write_text(
        "id: case-inline-001\naxis_level: severe\ntask_type: req-extraction\n"
    )
    (tmp_path / "corpus.yaml").write_text(
        "id: case-corpus-001\naxis_level: clean\nrun_mode: read_only_corpus\n"
        "task_type: req-extraction\n"
    )
    axis = _axis_by_id(tmp_path, include_corpus=True)
    assert axis == {"case-inline-001": "severe", "case-corpus-001": "clean"}
    # default stays corpus-free (existing behavior, #217)
    assert _axis_by_id(tmp_path) == {"case-inline-001": "severe"}


def test_corpus_capable_harnesses_lists_wired_clis_only() -> None:
    from method.run_pipe_check import CORPUS_CAPABLE_HARNESSES

    assert "claude_code" in CORPUS_CAPABLE_HARNESSES
    assert "codex_cli" in CORPUS_CAPABLE_HARNESSES
    assert "pi" in CORPUS_CAPABLE_HARNESSES
    assert "opencode" in CORPUS_CAPABLE_HARNESSES
    # deepseek/mimo/qwen/ollama/anthropic_api are not native-fs CLIs —
    # they stay out by design (anthropic_api uses the HTTP tool loop).
    assert "deepseek" not in CORPUS_CAPABLE_HARNESSES
    assert "anthropic_api" not in CORPUS_CAPABLE_HARNESSES


def test_register_corpus_preparer_is_idempotent() -> None:
    from atp.runner.preparation import (
        get_request_preparer,
        register_request_preparer,
        unregister_request_preparer,
    )
    from method.run_pipe_check import _register_corpus_preparer

    # Restore whatever was registered before (e.g. by the atp-method plugin
    # in another test) so this test can't create order-dependent failures.
    previous = get_request_preparer("corpus")
    try:
        _register_corpus_preparer()
        first = get_request_preparer("corpus")
        assert first is not None
        _register_corpus_preparer()
        assert get_request_preparer("corpus") is not None
    finally:
        if previous is not None:
            register_request_preparer("corpus", previous)
        else:
            unregister_request_preparer("corpus")
