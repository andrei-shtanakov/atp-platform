"""Plan 2a deadline worker. Single asyncio task running inside the
FastAPI lifespan. Two-path scan per tick:
1. Expired round deadlines -> force_resolve_round.
2. Expired pending tournaments -> shrink/start for El Farol & Public Goods,
   plain cancel_tournament_system(PENDING_TIMEOUT) for the rest.

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
from datetime import UTC, datetime

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


def _utc_now() -> datetime:
    """Current UTC time as naive datetime (SQLite-compatible)."""
    return datetime.now(UTC).replace(tzinfo=None)


_DEFAULT_POLL_INTERVAL_S = 5.0


def _get_poll_interval() -> float:
    """Read poll interval from env at call time (not at module import).

    Reading at call time (not module-level) ensures that monkeypatched
    environment variables in tests take effect even when the module was
    imported before the patch was applied.
    """
    return float(
        os.environ.get(
            "ATP_DEADLINE_WORKER_POLL_INTERVAL_S", str(_DEFAULT_POLL_INTERVAL_S)
        )
    )


async def run_deadline_worker(
    session_factory: async_sessionmaker[AsyncSession],
    bus: TournamentEventBus,
    *,
    shutdown_event: asyncio.Event,
) -> None:
    """Main loop. Runs until shutdown_event is set or the task is cancelled."""
    log = logging.getLogger("tournament.deadlines")
    poll_interval_s = _get_poll_interval()
    log.info("deadline_worker.started poll_interval_s=%s", poll_interval_s)

    while not shutdown_event.is_set():
        try:
            await _tick(session_factory, bus, log)
        except Exception:
            log.exception("deadline_worker.tick_failed")

        poll_interval_s = _get_poll_interval()
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=poll_interval_s)
        except TimeoutError:
            pass  # normal path — interval elapsed

    log.info("deadline_worker.shutting_down")


async def _tick(
    session_factory: async_sessionmaker[AsyncSession],
    bus: TournamentEventBus,
    log: logging.Logger,
) -> None:
    """One scan pass across both paths.

    Each round and tournament is processed in its own session so that
    a commit (and bus.publish inside service methods) occurs atomically
    per row — the forwarder tasks that respond to bus events need the DB
    state to already be visible when they open their own sessions.
    """
    t_start = time.monotonic()

    # Collect expired round IDs and pending tournament routing data in a
    # read-only session.
    async with session_factory() as scan_session:
        expired_rounds_result = await scan_session.execute(
            select(Round.id)
            .join(Tournament, Tournament.id == Round.tournament_id)
            .where(Round.status == RoundStatus.WAITING_FOR_ACTIONS)
            .where(Round.deadline < _utc_now())
            .where(Tournament.status == TournamentStatus.ACTIVE)
        )
        round_ids = [row[0] for row in expired_rounds_result]

        expired_pending_result = await scan_session.execute(
            select(Tournament.id, Tournament.game_type)
            .where(Tournament.status == TournamentStatus.PENDING)
            .where(Tournament.pending_deadline < _utc_now())
        )
        pending_tournaments = [(row[0], row[1]) for row in expired_pending_result]

    # Path 1: expired round deadlines — one session per round so that
    # service.force_resolve_round() can commit before bus.publish fires.
    # Note: force_resolve_round() commits internally before publishing, so
    # no explicit commit here — the session context manager handles cleanup.
    for round_id in round_ids:
        try:
            async with session_factory() as session:
                service = TournamentService(session, bus)
                await service.force_resolve_round(round_id)
        except Exception:
            log.exception(
                "deadline_worker.round_resolve_failed",
                extra={"round_id": round_id},
            )

    # Path 2: expired PENDING tournaments
    for tournament_id, game_type in pending_tournaments:
        try:
            async with session_factory() as session:
                service = TournamentService(session, bus)
                if game_type in {"el_farol", "public_goods"}:
                    await service.try_shrink_and_start_or_cancel(tournament_id)
                else:
                    await service.cancel_tournament_system(
                        tournament_id,
                        reason=CancelReason.PENDING_TIMEOUT,
                    )
                # cancel_tournament_system/_cancel_impl does not commit, and
                # try_shrink_and_start_or_cancel only commits on the start
                # branch via _start_tournament(). A trailing commit here is
                # therefore required for cancel/no-op branches and harmless
                # for already-committed start branches.
                await session.commit()
        except Exception:
            log.exception(
                "deadline_worker.pending_transition_failed",
                extra={"tournament_id": tournament_id, "game_type": game_type},
            )

    log.info(
        "deadline_worker.tick_complete rounds=%d pending_processed=%d elapsed_ms=%d",
        len(round_ids),
        len(pending_tournaments),
        int((time.monotonic() - t_start) * 1000),
    )
