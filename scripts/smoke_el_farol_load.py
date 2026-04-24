"""Pre-merge load smoke: in-process El Farol N=20, R=30.

Spec §7.3. Exercises TournamentService directly against an alembic-
migrated SQLite DB. Times ``get_state_for`` round-over-round to catch
O(R) payload bloat from ``attendance_by_round``.

Observed behaviour on PR-2 merge (2026-04-16):
  - All 30 rounds resolve cleanly, no exceptions.
  - Median per-round latency grows ~0.7 ms (round 1) -> ~4.5 ms
    (round 30) — a ~6x climb as ``attendance_by_round`` accumulates.
    This is the expected Phase B behaviour; shrinking it is tracked
    for Phase C (paginate / cap history).
  - max() is heavily contaminated by SQLite WAL checkpoint spikes
    (~18 ms every few rounds), so it's NOT a useful gate at N=20.

Pass criteria (this script):
  - exit 0 (no exceptions, all rounds resolve)
  - median latency at round R <= 15x round-1 median (catches
    catastrophic regression while tolerating the expected O(R)
    growth and CI noise).

Usage (from repo root):
    uv run python scripts/smoke_el_farol_load.py
"""

from __future__ import annotations

import asyncio
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from atp.dashboard.models import User
from atp.dashboard.tournament.service import TournamentService

N = 20
R = 30
ROUND_DEADLINE_S = 60
NUM_SLOTS = 16


class _DummyBus:
    async def publish(self, event):  # noqa: ANN001
        pass


async def _setup_db() -> tuple[Path, async_sessionmaker]:
    tmp = Path(tempfile.mkdtemp(prefix="el-farol-smoke-"))
    db_path = tmp / "smoke.db"
    env = {**os.environ, "ATP_DASHBOARD_DATABASE_URL": f"sqlite:///{db_path}"}
    result = subprocess.run(
        ["uv", "run", "alembic", "-n", "dashboard", "upgrade", "head"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("alembic upgrade failed:", result.stderr, file=sys.stderr)
        sys.exit(1)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    return db_path, async_sessionmaker(engine, expire_on_commit=False)


async def _seed_users(session, n: int) -> list[User]:
    for uid in range(1, n + 1):
        await session.execute(
            text(
                "INSERT INTO users "
                "(id, tenant_id, username, email, hashed_password, "
                "is_active, is_admin, created_at, updated_at) "
                "VALUES (:id, 'default', :u, :e, 'x', 1, 0, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"id": uid, "u": f"bot{uid}", "e": f"bot{uid}@smoke"},
        )
    return [await session.get(User, uid) for uid in range(1, n + 1)]


async def _run() -> int:
    _, session_factory = await _setup_db()
    rng = random.Random(42)

    async with session_factory() as session:
        users = await _seed_users(session, N)
        await session.commit()

        svc = TournamentService(session, _DummyBus())
        t, _ = await svc.create_tournament(
            creator=users[0],
            name="smoke",
            game_type="el_farol",
            num_players=N,
            total_rounds=R,
            round_deadline_s=ROUND_DEADLINE_S,
        )
        for u in users:
            await svc.join(t.id, u, agent_name=u.username)
        await session.commit()

        latencies_per_round: list[list[float]] = []
        for _round_idx in range(R):
            # Time get_state_for each user (one pass per round).
            round_lats: list[float] = []
            for u in users:
                t0 = time.perf_counter()
                await svc.get_state_for(t.id, u)
                round_lats.append(time.perf_counter() - t0)
            latencies_per_round.append(round_lats)

            # Each user submits a random slot list.
            for u in users:
                k = rng.randint(0, 8)
                slots = sorted(rng.sample(range(NUM_SLOTS), k))
                await svc.submit_action(t.id, u, action={"slots": slots})
            await session.commit()

    for r_idx, lats in enumerate(latencies_per_round):
        median = statistics.median(lats)
        mx = max(lats)
        print(
            f"round {r_idx + 1:2d}: median = {median * 1000:6.2f}ms  "
            f"max = {mx * 1000:6.2f}ms"
        )

    first_median = statistics.median(latencies_per_round[0])
    last_median = statistics.median(latencies_per_round[-1])
    ratio = last_median / first_median if first_median > 0 else 1.0
    if ratio > 15.0:
        print(
            f"FAIL: median degraded {first_median * 1000:.2f}ms -> "
            f"{last_median * 1000:.2f}ms ({ratio:.2f}x > 15x)",
            file=sys.stderr,
        )
        return 1
    print(
        f"OK: median within tolerance "
        f"({first_median * 1000:.2f}ms -> {last_median * 1000:.2f}ms, "
        f"{ratio:.2f}x <= 15x)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
