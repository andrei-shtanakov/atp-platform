"""Runtime preparation for corpus-backed method cases."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from atp.loader.models import TestDefinition
from atp.mock_tools.file_tools import DirectoryFileRead
from atp.mock_tools.runtime import serve_mock_tools
from atp.mock_tools.server import MockToolServer
from atp.protocol import ATPRequest, Context
from atp.runner.preparation import PreparedRequest

from atp_method.corpus import (
    CorpusIntegrityVerifier,
    CorpusMaterializer,
    CorpusResolver,
    MaterializedCorpus,
)
from atp_method.schema import ArtifactCorpus


@dataclass
class ServedCorpusTools:
    """A started corpus tool server."""

    endpoint: str
    cleanup: Callable[[], Awaitable[None]]


async def serve_corpus_tools(materialized: MaterializedCorpus) -> ServedCorpusTools:
    """Serve ``file_read`` for a materialized corpus."""
    server = MockToolServer(record_calls=True)
    server.add_handler(
        "file_read",
        DirectoryFileRead(
            materialized.root,
            allowed_paths={file.relative_path for file in materialized.files},
        ),
    )
    manager = serve_mock_tools(server)
    endpoint = await manager.__aenter__()

    async def cleanup() -> None:
        await manager.__aexit__(None, None, None)

    return ServedCorpusTools(endpoint=endpoint, cleanup=cleanup)


class CorpusRunPreparer:
    """Prepare a read-only corpus run before adapter execution."""

    def __init__(self, workspace: Path | None = None) -> None:
        """Initialize with an optional fixed workspace for tests."""
        self.workspace = workspace

    async def prepare(
        self, test: TestDefinition, request: ATPRequest
    ) -> PreparedRequest:
        """Resolve, verify, materialize, and attach corpus tool context."""
        input_data = dict(test.task.input_data or {})
        case_path_raw = input_data.get("case_path")
        corpus = input_data.get("artifact_corpus")
        if not case_path_raw or corpus is None:
            raise ValueError(
                "corpus request preparation requires case_path and artifact_corpus"
            )

        workspace = self.workspace or Path.cwd() / ".atp-runs" / request.task_id
        if isinstance(corpus, dict) and "root" in corpus:
            corpus_for_resolver = ArtifactCorpus.model_validate(corpus)
        else:
            corpus_for_resolver = corpus
        resolved = CorpusResolver().resolve(
            Path(str(case_path_raw)), corpus_for_resolver
        )
        verified = CorpusIntegrityVerifier().verify(resolved)
        materialized = CorpusMaterializer().materialize(verified, workspace)
        served = await serve_corpus_tools(materialized)

        request.context = request.context or Context()
        request.context.workspace_path = str(materialized.root)
        request.context.tools_endpoint = served.endpoint
        request.constraints = {**request.constraints, "allowed_tools": ["file_read"]}

        request_input = dict(request.task.input_data or {})
        artifact_corpus = dict(request_input.get("artifact_corpus") or corpus)
        artifact_corpus["files"] = [file.relative_path for file in materialized.files]
        request_input["artifact_corpus"] = artifact_corpus
        request_input["run_mode"] = "read_only_corpus"
        request.task.input_data = request_input

        citation_assertions = [
            assertion
            for assertion in test.assertions
            if assertion.config.get("checker") == "citation_grounding"
        ]
        if citation_assertions:
            files_config = {
                file.relative_path: {
                    "line_count": file.line_count,
                    "metadata": dict(file.metadata),
                }
                for file in materialized.files
            }
            for assertion in citation_assertions:
                assertion.config["files"] = files_config

        return PreparedRequest(request=request, cleanup=served.cleanup)
