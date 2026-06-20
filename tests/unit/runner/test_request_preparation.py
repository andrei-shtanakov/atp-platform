"""Tests for runner request-preparation hooks."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from atp.adapters.base import AgentAdapter
from atp.loader.models import Constraints, TaskDefinition, TestDefinition
from atp.protocol import ATPRequest, ATPResponse, ResponseStatus
from atp.runner.orchestrator import TestOrchestrator


@dataclass
class _RecordingPreparer:
    cleanup_called: bool = False

    async def prepare(self, test: TestDefinition, request: ATPRequest):
        from atp.runner.preparation import PreparedRequest

        request.metadata = {**(request.metadata or {}), "prepared": True}
        return PreparedRequest(request=request, cleanup=self.cleanup)

    async def cleanup(self) -> None:
        self.cleanup_called = True


class _Adapter(AgentAdapter):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.requests: list[ATPRequest] = []

    @property
    def adapter_type(self) -> str:
        return "recording"

    async def execute(self, request: ATPRequest) -> ATPResponse:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("adapter failed")
        return ATPResponse(task_id=request.task_id, status=ResponseStatus.COMPLETED)

    async def stream_events(self, request: ATPRequest):  # type: ignore[no-untyped-def]
        yield await self.execute(request)


def _test_definition() -> TestDefinition:
    return TestDefinition(
        id="t1",
        name="prepared test",
        task=TaskDefinition(
            description="run",
            input_data={"request_preparer": "corpus"},
        ),
        constraints=Constraints(timeout_seconds=5),
    )


def test_request_preparer_registry_resolves_named_preparer() -> None:
    from atp.runner.preparation import (
        get_request_preparer,
        register_request_preparer,
        unregister_request_preparer,
    )

    preparer = _RecordingPreparer()
    register_request_preparer("corpus", preparer)
    try:
        assert get_request_preparer("corpus") is preparer
    finally:
        unregister_request_preparer("corpus")


@pytest.mark.anyio
async def test_orchestrator_applies_named_preparer_before_adapter_execution() -> None:
    from atp.runner.preparation import (
        register_request_preparer,
        unregister_request_preparer,
    )

    preparer = _RecordingPreparer()
    adapter = _Adapter()
    register_request_preparer("corpus", preparer)
    try:
        result = await TestOrchestrator(adapter).run_single_test(_test_definition())
    finally:
        unregister_request_preparer("corpus")

    assert result.success is True
    assert adapter.requests[0].metadata["prepared"] is True
    assert preparer.cleanup_called is True


@pytest.mark.anyio
async def test_orchestrator_cleans_up_prepared_request_after_adapter_failure() -> None:
    from atp.runner.preparation import (
        register_request_preparer,
        unregister_request_preparer,
    )

    preparer = _RecordingPreparer()
    register_request_preparer("corpus", preparer)
    try:
        result = await TestOrchestrator(_Adapter(fail=True)).run_single_test(
            _test_definition()
        )
    finally:
        unregister_request_preparer("corpus")

    assert result.success is False
    assert preparer.cleanup_called is True
