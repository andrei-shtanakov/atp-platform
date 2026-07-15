"""Orchestrator emits one UsageRecord per adapter call (003e M0 seam)."""

from collections.abc import AsyncIterator

import pytest

from atp.adapters.base import AgentAdapter
from atp.cost.capture import UsageRecord
from atp.loader.models import Constraints, TaskDefinition, TestDefinition
from atp.protocol import ATPEvent, ATPRequest, ATPResponse, Metrics, ResponseStatus
from atp.runner.orchestrator import TestOrchestrator


class RecordingCapture:
    """Test double collecting every UsageRecord it is handed."""

    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    def record_usage(self, record: UsageRecord) -> None:
        self.records.append(record)


class StubAdapter(AgentAdapter):
    """Minimal adapter returning a canned completed response with metrics."""

    @property
    def adapter_type(self) -> str:
        return "stub"

    async def execute(self, request: ATPRequest) -> ATPResponse:
        return ATPResponse(
            task_id=request.task_id,
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(input_tokens=11, output_tokens=7),
        )

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        yield await self.execute(request)


def make_test() -> TestDefinition:
    return TestDefinition(
        id="t-usage-capture",
        name="usage capture smoke",
        task=TaskDefinition(description="say hi"),
        constraints=Constraints(timeout_seconds=10),
    )


@pytest.mark.anyio
async def test_execute_run_records_usage() -> None:
    capture = RecordingCapture()
    orch = TestOrchestrator(adapter=StubAdapter(), usage_capture=capture)
    result = await orch.run_single_test(make_test())

    assert result is not None
    assert len(capture.records) == 1
    rec = capture.records[0]
    assert rec.adapter_type == "stub"
    assert rec.status == "completed"
    assert rec.usage is not None
    assert rec.usage.input_tokens == 11
    assert rec.model is None
    assert rec.reported_cost_usd is None
    assert rec.test_id == "t-usage-capture"
