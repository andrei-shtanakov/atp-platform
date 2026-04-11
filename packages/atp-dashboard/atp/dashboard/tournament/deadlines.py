"""Plan 2a deadline worker. Single asyncio task running inside the
FastAPI lifespan. Two-path scan per tick:
1. Expired round deadlines -> force_resolve_round.
2. Expired pending tournaments -> cancel_tournament_system(PENDING_TIMEOUT).

Per-iteration outer try/except + per-row inner try/except; one poisoned
row cannot kill the tick. Hard cancel on shutdown — correctness is
preserved by SQLAlchemy transaction rollback on CancelledError and
session_sync on subscriber reconnect.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from atp.dashboard.tournament.events import TournamentEventBus
from atp.dashboard.tournament.models import (
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.reasons import CancelReason
from atp.dashboard.tournament.service import TournamentService

POLL_INTERVAL_S: float = float(
    os.environ.get("ATP_DEADLINE_WORKER_POLL_INTERVAL_S", "5")
)


async def run_deadline_worker(
    session_factory: async_sessionmaker[AsyncSession],
    bus: TournamentEventBus,
    *,
    shutdown_event: asyncio.Event,
) -> None:
    """Main loop. Runs until shutdown_event is set or the task is cancelled."""
    log = logging.getLogger("tournament.deadlines")
    log.info("deadline_worker.started", extra={"poll_interval_s": POLL_INTERVAL_S})

    while not shutdown_event.is_set():
        try:
            await _tick(session_factory, bus, log)
        except Exception:
            log.exception("deadline_worker.tick_failed")

        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=POLL_INTERVAL_S)
        except TimeoutError:
            pass  # normal path — interval elapsed

    log.info("deadline_worker.shutting_down")


async def _tick(
    session_factory: async_sessionmaker[AsyncSession],
    bus: TournamentEventBus,
    log: logging.Logger,
) -> None:
    """One scan pass across both paths."""
    t_start = time.monotonic()
    async with session_factory() as session:
        service = TournamentService(session, bus)

        # Path 1: expired round deadlines
        expired_rounds_result = await session.execute(
            select(Round.id)
            .join(Tournament, Tournament.id == Round.tournament_id)
            .where(Round.status == RoundStatus.WAITING_FOR_ACTIONS)
            .where(Round.deadline < datetime.utcnow())
            .where(Tournament.status == TournamentStatus.ACTIVE)
        )
        round_ids = [row[0] for row in expired_rounds_result]
        for round_id in round_ids:
            try:
                await service.force_resolve_round(round_id)
            except Exception:
                log.exception(
                    "deadline_worker.round_resolve_failed",
                    extra={"round_id": round_id},
                )

        # Path 2: expired PENDING tournaments
        expired_pending_result = await session.execute(
            select(Tournament.id)
            .where(Tournament.status == TournamentStatus.PENDING)
            .where(Tournament.pending_deadline < datetime.utcnow())
        )
        tournament_ids = [row[0] for row in expired_pending_result]
        for tournament_id in tournament_ids:
            try:
                await service.cancel_tournament_system(
                    tournament_id,
                    reason=CancelReason.PENDING_TIMEOUT,
                )
            except Exception:
                log.exception(
                    "deadline_worker.pending_cancel_failed",
                    extra={"tournament_id": tournament_id},
                )

    log.info(
        "deadline_worker.tick_complete",
        extra={
            "rounds_processed": len(round_ids),
            "pending_cancelled": len(tournament_ids),
            "elapsed_ms": int((time.monotonic() - t_start) * 1000),
        },
    )
