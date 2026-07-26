"""Canonical server entrypoint: the composition root for a running dashboard.

This module is the one place that sees both halves — evaluator implementations
from atp-platform and the application factory from atp-dashboard — and it is
deliberately in atp-platform, because that is the package allowed to import
both. The dashboard receives an already-restricted resolver and never learns
where it came from.

Everything that starts a production server goes through here:
`atp dashboard`, and docker-compose, which runs `atp dashboard`. Pointing
uvicorn at `atp.dashboard.v2.factory:app` instead yields a completion-only
application — legitimate, but only when that is what you meant. Which of the
two you got is visible at `/api/evaluation/capabilities`, so it is never a
guess.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atp.evaluation import (
    DETERMINISTIC_ALLOWLIST,
    UNTRUSTED_SUBMISSION,
    FilteredResolver,
)

if TYPE_CHECKING:  # pragma: no cover - import cost only matters at runtime
    from fastapi import FastAPI


def build_server_resolver() -> FilteredResolver:
    """The evaluator resolver a public server may hold.

    Restricted before it leaves this function: what crosses the package
    boundary cannot produce an evaluator that executes code or calls out, so
    the restriction does not depend on the far side honouring it.
    """
    from atp.evaluators.registry import get_registry

    return FilteredResolver(get_registry(), UNTRUSTED_SUBMISSION)


def create_server_app() -> FastAPI:
    """Build the dashboard with deterministic evaluation wired in.

    Used as a uvicorn factory (`atp.server:create_server_app`), so reload and
    multi-worker launches construct it the same way a single process does —
    a composition that only works in one launch mode is a composition waiting
    to differ in production.
    """
    from atp.dashboard.v2.factory import create_dashboard_app

    return create_dashboard_app(
        evaluator_resolver=build_server_resolver(),
        evaluation_mode=DETERMINISTIC_ALLOWLIST,
    )


def run_server(host: str = "127.0.0.1", port: int = 8080, reload: bool = False) -> None:
    """Serve the canonical application."""  # pragma: no cover - thin wrapper
    import uvicorn

    uvicorn.run(
        "atp.server:create_server_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )
