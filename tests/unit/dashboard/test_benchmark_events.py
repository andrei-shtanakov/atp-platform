"""Tests for benchmark run event streaming endpoint."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException


class TestEmitEvents:
    """Tests for POST /runs/{run_id}/events."""

    @pytest.mark.anyio
    async def test_emit_events_appends(self) -> None:
        """Events are appended to run.events."""
        from atp.dashboard.benchmark.models import Run, RunStatus
        from atp.dashboard.v2.routes.benchmark_api import emit_events

        # Use __wrapped__ to bypass the slowapi limiter decorator
        fn = emit_events.__wrapped__

        mock_run = MagicMock(spec=Run)
        mock_run.status = RunStatus.IN_PROGRESS
        mock_run.events = [{"event_type": "existing"}]

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_run
        mock_session.flush = AsyncMock()

        result = await fn(
            request=MagicMock(),
            run_id=1,
            data={"events": [{"event_type": "tool_call", "data": {}}]},
            session=mock_session,
        )

        assert result["accepted"] == 1
        assert result["total"] == 2
        assert len(mock_run.events) == 2

    @pytest.mark.anyio
    async def test_emit_events_rejects_over_limit(self) -> None:
        """Returns 422 when event limit exceeded."""
        from atp.dashboard.benchmark.models import Run, RunStatus
        from atp.dashboard.v2.routes.benchmark_api import emit_events

        fn = emit_events.__wrapped__

        mock_run = MagicMock(spec=Run)
        mock_run.status = RunStatus.IN_PROGRESS
        mock_run.events = [{"e": i} for i in range(999)]

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_run

        with pytest.raises(HTTPException) as exc_info:
            await fn(
                request=MagicMock(),
                run_id=1,
                data={"events": [{"e": 1}, {"e": 2}]},
                session=mock_session,
            )
        assert exc_info.value.status_code == 422
        assert "limit" in exc_info.value.detail.lower()

    @pytest.mark.anyio
    async def test_emit_events_rejects_not_in_progress(self) -> None:
        """Returns 400 when run is not IN_PROGRESS."""
        from atp.dashboard.benchmark.models import Run, RunStatus
        from atp.dashboard.v2.routes.benchmark_api import emit_events

        fn = emit_events.__wrapped__

        mock_run = MagicMock(spec=Run)
        mock_run.status = RunStatus.COMPLETED

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_run

        with pytest.raises(HTTPException) as exc_info:
            await fn(
                request=MagicMock(),
                run_id=1,
                data={"events": [{"e": 1}]},
                session=mock_session,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_emit_events_rejects_missing_run(self) -> None:
        """Returns 404 when run not found."""
        from atp.dashboard.v2.routes.benchmark_api import emit_events

        fn = emit_events.__wrapped__

        mock_session = AsyncMock()
        mock_session.get.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await fn(
                request=MagicMock(),
                run_id=999,
                data={"events": []},
                session=mock_session,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.anyio
    async def test_emit_events_rejects_non_list(self) -> None:
        """Returns 400 when events is not a list."""
        from atp.dashboard.benchmark.models import Run, RunStatus
        from atp.dashboard.v2.routes.benchmark_api import emit_events

        fn = emit_events.__wrapped__

        mock_run = MagicMock(spec=Run)
        mock_run.status = RunStatus.IN_PROGRESS

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_run

        with pytest.raises(HTTPException) as exc_info:
            await fn(
                request=MagicMock(),
                run_id=1,
                data={"events": "not-a-list"},
                session=mock_session,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_emit_events_empty_list(self) -> None:
        """Empty events list is accepted."""
        from atp.dashboard.benchmark.models import Run, RunStatus
        from atp.dashboard.v2.routes.benchmark_api import emit_events

        fn = emit_events.__wrapped__

        mock_run = MagicMock(spec=Run)
        mock_run.status = RunStatus.IN_PROGRESS
        mock_run.events = []

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_run
        mock_session.flush = AsyncMock()

        result = await fn(
            request=MagicMock(),
            run_id=1,
            data={"events": []},
            session=mock_session,
        )

        assert result["accepted"] == 0
        assert result["total"] == 0
