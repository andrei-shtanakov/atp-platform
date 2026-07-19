"""Tests for cooperative signal handling in the CLI runner."""

import asyncio
import os
import signal

import pytest

from atp.cli.main import _install_signal_handlers


class FakeOrchestrator:
    def __init__(self) -> None:
        self.stopped = False

    def request_shutdown(self) -> None:
        self.stopped = True


@pytest.mark.anyio
async def test_sigint_triggers_request_shutdown() -> None:
    orchestrator = FakeOrchestrator()
    remove = _install_signal_handlers(orchestrator)
    try:
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.sleep(0.05)  # let the loop dispatch the handler
        assert orchestrator.stopped is True
    finally:
        remove()
