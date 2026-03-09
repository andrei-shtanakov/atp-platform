"""Tests for FallbackAdapter."""

from collections.abc import AsyncIterator

import pytest

from atp.adapters.base import AgentAdapter
from atp.adapters.exceptions import AdapterError
from atp.adapters.fallback import FallbackAdapter
from atp.protocol import ATPEvent, ATPRequest, ATPResponse, ResponseStatus


class _StubAdapter(AgentAdapter):
    """Adapter stub for testing."""

    def __init__(self, name: str, fail: bool = False) -> None:
        super().__init__()
        self._name = name
        self._fail = fail
        self.called = False

    @property
    def adapter_type(self) -> str:
        return self._name

    async def execute(self, request: ATPRequest) -> ATPResponse:
        self.called = True
        if self._fail:
            raise AdapterError(f"{self._name} failed", adapter_type=self._name)
        return ATPResponse(
            task_id=request.task_id,
            status=ResponseStatus.COMPLETED,
            artifacts=[],
        )

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        self.called = True
        if self._fail:
            raise AdapterError(f"{self._name} failed", adapter_type=self._name)
        yield ATPResponse(
            task_id=request.task_id,
            status=ResponseStatus.COMPLETED,
            artifacts=[],
        )


def _request() -> ATPRequest:
    return ATPRequest(task_id="t1", task={"description": "do something"})


class TestFallbackAdapter:
    def test_empty_chain_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            FallbackAdapter(chain=[])

    @pytest.mark.anyio
    async def test_primary_succeeds(self) -> None:
        primary = _StubAdapter("a")
        fallback = _StubAdapter("b")
        adapter = FallbackAdapter(chain=[primary, fallback])

        resp = await adapter.execute(_request())
        assert resp.status == ResponseStatus.COMPLETED
        assert primary.called is True
        assert fallback.called is False

    @pytest.mark.anyio
    async def test_fallback_on_failure(self) -> None:
        primary = _StubAdapter("a", fail=True)
        fallback = _StubAdapter("b")
        adapter = FallbackAdapter(chain=[primary, fallback])

        resp = await adapter.execute(_request())
        assert resp.status == ResponseStatus.COMPLETED
        assert primary.called is True
        assert fallback.called is True

    @pytest.mark.anyio
    async def test_all_fail_raises(self) -> None:
        a = _StubAdapter("a", fail=True)
        b = _StubAdapter("b", fail=True)
        adapter = FallbackAdapter(chain=[a, b])

        with pytest.raises(AdapterError, match="All adapters failed"):
            await adapter.execute(_request())

    @pytest.mark.anyio
    async def test_stream_fallback(self) -> None:
        primary = _StubAdapter("a", fail=True)
        fallback = _StubAdapter("b")
        adapter = FallbackAdapter(chain=[primary, fallback])

        items = []
        async for item in adapter.stream_events(_request()):
            items.append(item)
        assert len(items) == 1
        assert isinstance(items[0], ATPResponse)

    @pytest.mark.anyio
    async def test_health_check_any_healthy(self) -> None:
        a = _StubAdapter("a")
        b = _StubAdapter("b")
        adapter = FallbackAdapter(chain=[a, b])
        assert await adapter.health_check() is True

    def test_adapter_type(self) -> None:
        adapter = FallbackAdapter(chain=[_StubAdapter("a")])
        assert adapter.adapter_type == "fallback"
