"""Unit tests for SDKAdapter."""

import asyncio

import pytest

from atp.adapters.sdk_adapter import SDKAdapter, SDKAdapterConfig
from atp.protocol import ATPRequest, ATPResponse, ResponseStatus, Task


@pytest.fixture
def adapter() -> SDKAdapter:
    """Create an SDKAdapter with short timeout for tests."""
    return SDKAdapter(SDKAdapterConfig(timeout_seconds=5.0))


@pytest.fixture
def sample_request() -> ATPRequest:
    """Create a sample ATP request for testing."""
    return ATPRequest(
        task_id="test-task-123",
        task=Task(description="Test task"),
        constraints={"max_steps": 10},
    )


@pytest.fixture
def sample_response() -> ATPResponse:
    """Create a sample ATP response for testing."""
    return ATPResponse(
        task_id="test-task-123",
        status=ResponseStatus.COMPLETED,
        artifacts=[],
    )


class TestSDKAdapterConfig:
    """Tests for SDKAdapterConfig."""

    def test_default_timeout(self) -> None:
        """Test default timeout is 3600 seconds."""
        config = SDKAdapterConfig()
        assert config.timeout_seconds == 3600.0

    def test_custom_timeout(self) -> None:
        """Test custom timeout value."""
        config = SDKAdapterConfig(timeout_seconds=60.0)
        assert config.timeout_seconds == 60.0


class TestSDKAdapter:
    """Tests for SDKAdapter."""

    @pytest.mark.anyio
    async def test_adapter_type(self, adapter: SDKAdapter) -> None:
        """Test adapter type is 'sdk'."""
        assert adapter.adapter_type == "sdk"

    @pytest.mark.anyio
    async def test_enqueue_and_resolve(
        self,
        adapter: SDKAdapter,
        sample_request: ATPRequest,
        sample_response: ATPResponse,
    ) -> None:
        """Test full flow: execute enqueues, pull retrieves, resolve unblocks."""

        async def agent_side() -> None:
            """Simulate agent pulling and resolving a task."""
            await asyncio.sleep(0.05)
            task = adapter.pull_task()
            assert task is not None
            assert task.task_id == "test-task-123"
            adapter.resolve_task(task.task_id, sample_response)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(agent_side())
            result = await asyncio.ensure_future(adapter.execute(sample_request))

        assert result.task_id == "test-task-123"
        assert result.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_timeout_raises(
        self,
        sample_request: ATPRequest,
    ) -> None:
        """Test that execute raises TimeoutError when not resolved."""
        adapter = SDKAdapter(SDKAdapterConfig(timeout_seconds=0.1))
        with pytest.raises(TimeoutError):
            await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_timeout_cleans_up(
        self,
        sample_request: ATPRequest,
    ) -> None:
        """Test that pending state is cleaned up after timeout."""
        adapter = SDKAdapter(SDKAdapterConfig(timeout_seconds=0.1))
        with pytest.raises(TimeoutError):
            await adapter.execute(sample_request)

        # Verify cleanup: no pending tasks or events remain
        assert adapter.pull_task() is None

    def test_pull_returns_none_when_empty(self, adapter: SDKAdapter) -> None:
        """Test pull_task returns None when no tasks are pending."""
        assert adapter.pull_task() is None

    @pytest.mark.anyio
    async def test_stream_events_yields_response(
        self,
        adapter: SDKAdapter,
        sample_request: ATPRequest,
        sample_response: ATPResponse,
    ) -> None:
        """Test stream_events yields the final response."""

        async def agent_side() -> None:
            await asyncio.sleep(0.05)
            task = adapter.pull_task()
            assert task is not None
            adapter.resolve_task(task.task_id, sample_response)

        items: list[ATPResponse] = []

        async def collect() -> None:
            async for item in adapter.stream_events(sample_request):
                items.append(item)  # type: ignore[arg-type]

        async with asyncio.TaskGroup() as tg:
            tg.create_task(agent_side())
            tg.create_task(collect())

        assert len(items) == 1
        assert items[0].task_id == "test-task-123"

    @pytest.mark.anyio
    async def test_multiple_tasks_fifo(
        self,
        adapter: SDKAdapter,
    ) -> None:
        """Test that pull_task returns tasks in FIFO order."""
        req1 = ATPRequest(
            task_id="task-1",
            task=Task(description="First"),
        )
        req2 = ATPRequest(
            task_id="task-2",
            task=Task(description="Second"),
        )

        resp1 = ATPResponse(task_id="task-1", status=ResponseStatus.COMPLETED)
        resp2 = ATPResponse(task_id="task-2", status=ResponseStatus.COMPLETED)

        async def agent_side() -> None:
            await asyncio.sleep(0.05)
            t1 = adapter.pull_task()
            assert t1 is not None
            assert t1.task_id == "task-1"
            adapter.resolve_task(t1.task_id, resp1)

            await asyncio.sleep(0.05)
            t2 = adapter.pull_task()
            assert t2 is not None
            assert t2.task_id == "task-2"
            adapter.resolve_task(t2.task_id, resp2)

        async def platform_side() -> list[ATPResponse]:
            results = []
            r1 = await adapter.execute(req1)
            results.append(r1)
            r2 = await adapter.execute(req2)
            results.append(r2)
            return results

        async with asyncio.TaskGroup() as tg:
            tg.create_task(agent_side())
            results_task = tg.create_task(platform_side())

        results = results_task.result()
        assert results[0].task_id == "task-1"
        assert results[1].task_id == "task-2"
