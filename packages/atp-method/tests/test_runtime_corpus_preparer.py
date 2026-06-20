"""Tests for the atp-method corpus run preparer."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from atp.loader.models import Assertion, TaskDefinition, TestDefinition
from atp.protocol import ATPRequest, Task


@pytest.mark.anyio
async def test_corpus_run_preparer_materializes_and_attaches_tool_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from atp_method.runtime import CorpusRunPreparer

    materialized_root = tmp_path / "workspace" / "req-corpus"
    materialized_root.mkdir(parents=True)

    class _Resolver:
        def resolve(self, case_path, corpus):  # type: ignore[no-untyped-def]
            return "resolved"

    class _Verifier:
        def verify(self, resolved):  # type: ignore[no-untyped-def]
            assert resolved == "resolved"
            return "verified"

    class _Materializer:
        def materialize(self, verified, workspace):  # type: ignore[no-untyped-def]
            assert verified == "verified"
            assert workspace == tmp_path / "workspace"
            return type(
                "Materialized",
                (),
                {
                    "corpus_id": "req-corpus",
                    "root": materialized_root,
                    "files": (type("File", (), {"relative_path": "policy.md"})(),),
                },
            )()

    class _ToolServer:
        endpoint = "http://127.0.0.1:9999"
        cleanup = AsyncMock()

    monkeypatch.setattr("atp_method.runtime.CorpusResolver", _Resolver)
    monkeypatch.setattr("atp_method.runtime.CorpusIntegrityVerifier", _Verifier)
    monkeypatch.setattr("atp_method.runtime.CorpusMaterializer", _Materializer)
    monkeypatch.setattr(
        "atp_method.runtime.serve_corpus_tools",
        AsyncMock(return_value=_ToolServer()),
    )

    test = TestDefinition(
        id="t1",
        name="corpus",
        task=TaskDefinition(
            description="extract",
            input_data={
                "case_path": str(tmp_path / "case.yaml"),
                "run_mode": "read_only_corpus",
                "artifact_corpus": {"id": "req-corpus"},
            },
        ),
    )
    request = ATPRequest(task_id="t1", task=Task(description="extract"))

    prepared = await CorpusRunPreparer(workspace=tmp_path / "workspace").prepare(
        test, request
    )

    assert prepared.request.context.workspace_path == str(materialized_root)
    assert prepared.request.context.tools_endpoint == "http://127.0.0.1:9999"
    assert prepared.request.constraints["allowed_tools"] == ["file_read"]
    await prepared.cleanup()
    _ToolServer.cleanup.assert_awaited_once()


@pytest.mark.anyio
async def test_corpus_run_preparer_adds_file_metadata_to_citation_grounding_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from atp_method.runtime import CorpusRunPreparer

    materialized_root = tmp_path / "workspace" / "req-corpus"
    materialized_root.mkdir(parents=True)

    class _Resolver:
        def resolve(self, case_path, corpus):  # type: ignore[no-untyped-def]
            return "resolved"

    class _Verifier:
        def verify(self, resolved):  # type: ignore[no-untyped-def]
            assert resolved == "resolved"
            return "verified"

    class _Materializer:
        def materialize(self, verified, workspace):  # type: ignore[no-untyped-def]
            assert verified == "verified"
            assert workspace == tmp_path / "workspace"
            return type(
                "Materialized",
                (),
                {
                    "corpus_id": "req-corpus",
                    "root": materialized_root,
                    "files": (
                        type(
                            "File",
                            (),
                            {
                                "relative_path": "policy.md",
                                "line_count": 3,
                                "metadata": {
                                    "document_type": "policy",
                                    "effective_date": "2026-01-01",
                                },
                            },
                        )(),
                    ),
                },
            )()

    class _ToolServer:
        endpoint = "http://127.0.0.1:9999"
        cleanup = AsyncMock()

    monkeypatch.setattr("atp_method.runtime.CorpusResolver", _Resolver)
    monkeypatch.setattr("atp_method.runtime.CorpusIntegrityVerifier", _Verifier)
    monkeypatch.setattr("atp_method.runtime.CorpusMaterializer", _Materializer)
    monkeypatch.setattr(
        "atp_method.runtime.serve_corpus_tools",
        AsyncMock(return_value=_ToolServer()),
    )

    assertion = Assertion(
        type="method_critical_check",
        config={
            "checker": "citation_grounding",
            "expected": [
                {
                    "output_path": "$.requirements[0].citation",
                    "source_path": "policy.md",
                    "line_start": 1,
                    "line_end": 2,
                }
            ],
        },
    )
    test = TestDefinition(
        id="t1",
        name="corpus",
        task=TaskDefinition(
            description="extract",
            input_data={
                "case_path": str(tmp_path / "case.yaml"),
                "run_mode": "read_only_corpus",
                "artifact_corpus": {"id": "req-corpus"},
            },
        ),
        assertions=[assertion],
    )
    request = ATPRequest(task_id="t1", task=Task(description="extract"))

    prepared = await CorpusRunPreparer(workspace=tmp_path / "workspace").prepare(
        test, request
    )

    assert prepared.request.task.input_data["artifact_corpus"]["files"] == ["policy.md"]
    assert assertion.config["files"] == {
        "policy.md": {
            "line_count": 3,
            "metadata": {
                "document_type": "policy",
                "effective_date": "2026-01-01",
            },
        }
    }
