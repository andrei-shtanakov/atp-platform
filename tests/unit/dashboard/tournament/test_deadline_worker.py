"""Unit tests for the deadline worker tick logic."""

import asyncio
import logging
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.tournament.deadlines import _tick, run_deadline_worker


def _make_session_factory(
    expired_rounds: list[int],
    expired_pending: list[int],
):
    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=[
            MagicMock(__iter__=lambda self: iter([(r,) for r in expired_rounds])),
            MagicMock(__iter__=lambda self: iter([(t,) for t in expired_pending])),
        ]
    )
    # Async context manager support
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return factory, session


@pytest.mark.anyio
async def test_tick_empty_paths_logs_zero_counts(caplog):
    factory, session = _make_session_factory([], [])
    bus = MagicMock()
    log = logging.getLogger("tournament.deadlines")

    with caplog.at_level(logging.INFO, logger="tournament.deadlines"):
        await _tick(factory, bus, log)

    records = [r for r in caplog.records if "tick_complete" in r.message]
    assert len(records) == 1


@pytest.mark.anyio
async def test_tick_inner_guard_isolates_poison_row(caplog, monkeypatch):
    factory, session = _make_session_factory([1, 2, 3], [])
    bus = MagicMock()

    # Patch TournamentService so force_resolve_round raises on the middle row
    from atp.dashboard.tournament import deadlines

    call_count = {"n": 0}

    class _FakeService:
        def __init__(self, *a, **k):
            pass

        async def force_resolve_round(self, round_id):
            call_count["n"] += 1
            if round_id == 2:
                raise RuntimeError("simulated poison")

        async def cancel_tournament_system(self, *a, **k):
            pass

    monkeypatch.setattr(deadlines, "TournamentService", _FakeService)
    log = logging.getLogger("tournament.deadlines")

    with caplog.at_level(logging.ERROR, logger="tournament.deadlines"):
        await _tick(factory, bus, log)

    # All three rounds attempted — inner guard absorbed the middle failure
    assert call_count["n"] == 3
    failed_records = [r for r in caplog.records if "round_resolve_failed" in r.message]
    assert len(failed_records) == 1
    outer_records = [r for r in caplog.records if "tick_failed" in r.message]
    assert len(outer_records) == 0


@pytest.mark.anyio
async def test_run_deadline_worker_hard_cancel_exits_fast(monkeypatch):
    """Shutdown sequence: shutdown_event.set() + task.cancel() must
    terminate the worker even if it's mid-tick in a slow call."""
    factory = MagicMock()

    slow_session = MagicMock()

    async def slow_execute(*a, **k):
        await asyncio.sleep(5)
        return MagicMock(__iter__=lambda self: iter([]))

    slow_session.execute = slow_execute
    factory.return_value.__aenter__ = AsyncMock(return_value=slow_session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)

    bus = MagicMock()
    shutdown = asyncio.Event()
    task = asyncio.create_task(
        run_deadline_worker(factory, bus, shutdown_event=shutdown)
    )
    await asyncio.sleep(0.1)  # let the worker enter the slow execute
    shutdown.set()
    task.cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert task.done()  # must not hang waiting for the 5s sleep
