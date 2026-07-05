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
