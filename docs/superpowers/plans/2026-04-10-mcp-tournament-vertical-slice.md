# MCP Tournament Server — Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get two `MCPAdapter` bots playing a 3-round Prisoner's Dilemma tournament end-to-end through a real FastMCP SSE endpoint, with payoffs computed by `game-environments`, persisted in SQLite, and verified by an e2e test.

**Architecture:** Protocol-agnostic `TournamentService` over existing SQLAlchemy `Tournament/Participant/Round/Action` models, an in-process `asyncio.Queue` event bus, and a thin FastMCP server mounted under `/mcp` in the existing FastAPI app. Round resolution is synchronous in the last `make_move`, no deadline worker, no constraints (yet). PD only, 2 players, 3 rounds fixed.

**Tech Stack:** Python 3.12, FastAPI, FastMCP, SQLAlchemy 2.x async, Alembic, anyio/asyncio, pytest + pytest-anyio, httpx, in-house `MCPAdapter` SSE client, `game-environments` PD module.

**Source spec:** `docs/superpowers/specs/2026-04-10-mcp-tournament-server-design.md`. Decisions referenced as AD-N below come from there.

**Phase 0 status (updated 2026-04-10 after verification):**

- ✅ Task 0.1 PASSED (commit `87054c2` on branch `feat/mcp-tournament-vertical-slice`) — `MCPAdapter` can receive server-pushed notifications via `adapter._transport.stream_events()`. TWO latent bugs discovered in `SSETransport` (see §Phase Pre-8 added below).
- ✅ Task 0.2 PASSED (commit `713d7f4`) — FastMCP mount under FastAPI correctly triggers outer middleware. Three corrections to FastMCP API pattern applied throughout plan (see below).

**FastMCP 3.x API corrections** (verified empirically in Task 0.2, apply these patterns throughout the plan):

1. **Mount API:** `mcp_server.sse_app()` → `mcp_server.http_app(transport="sse")`. There is no `sse_app()` in FastMCP 3.x. Default mounted paths: `/sse` (SSE handshake) and `/messages/` (POST for JSON-RPC). Under `app.mount("/mcp", mcp_app)` the full URL is `/mcp/sse`.
2. **Request access from tool handlers:** use the public helper `from fastmcp.server.dependencies import get_http_request; request = get_http_request()` — NOT `ctx.request_context.request` (internal, version-fragile). `get_http_request()` resolves the underlying Starlette `Request` via a ContextVar managed by FastMCP.
3. **Mounted sub-app lifespan is NOT propagated automatically by Starlette.** The FastMCP sub-app has its own lifespan that must be composed into the outer FastAPI lifespan, e.g.:
   ```python
   mcp_app = mcp_server.http_app(transport="sse")
   @asynccontextmanager
   async def combined_lifespan(app):
       async with original_lifespan(app):
           async with mcp_app.router.lifespan_context(app):
               yield
   app = FastAPI(lifespan=combined_lifespan)
   app.mount("/mcp", mcp_app)
   ```
   Without this composition, FastMCP's internal session manager is never initialized and SSE connections hang or 500.
4. **Integration test harness:** `TestClient.stream()` hangs on SSE endpoints because the in-memory `ASGITransport` drives the app to completion. Use either `httpx.AsyncClient(transport=ASGITransport(app)) + anyio.move_on_after(timeout)` for quick in-memory checks, or a real uvicorn instance (Phase 8 pattern) for full end-to-end.
5. **Dependency constraint:** Task 6.1 pins `fastmcp >= 3.0`.

**Scope of THIS plan:**
- Phase 0 verification (MCPAdapter + FastMCP/middleware integration)
- Additive schema columns ONLY (no UNIQUE constraints, no NOT NULL flips — those are in Plan 2)
- `TournamentEventBus`
- `TournamentService` minimal: `create_tournament`, `join`, `get_state_for`, `submit_action`, plus private `_start_tournament`, `_resolve_round`, `_complete_tournament`
- `format_state_for_player` on the PD game class
- FastMCP server mount + auth middleware
- 3 MCP tools: `join_tournament`, `get_current_state`, `make_move`
- 2 notifications: `round_started`, `tournament_completed`
- E2E test with 2 `MCPAdapter` bots

**Out of scope (handled in Plan 2):**
- Deadline worker, `force_resolve_round`
- `leave_tournament`, `get_history`, `list_tournaments`, `get_tournament`, `cancel_tournament` tools
- `session_sync` notification, idempotent `join`
- AD-9 token expiry cap, AD-10 `join_token`, `MAX_ACTIVE_TOURNAMENTS_PER_USER`
- Schema constraints: `uq_participant_tournament_user`, `uq_action_round_participant`, `uq_round_tournament_number`, `Index(status, deadline)`, `Participant.user_id NOT NULL`
- REST admin endpoints + dashboard UI
- `round_ended`, `tournament_cancelled` notifications

**NEW: Added to scope after Phase 0 verification** (see §Phase Pre-8 below):
- Fixing two latent bugs in `packages/atp-adapters/atp/adapters/mcp/transport.py` that Phase 0.1 surfaced:
  1. `SSETransport._read_sse_events` cannot parse the `event: endpoint` frame from spec-compliant MCP servers (bare path string, not JSON) → any real MCP server → 405 on first POST.
  2. `SSETransport` has a single `_message_queue` shared between `_wait_for_response` and `stream_events`; notifications arriving during a pending request are silently dropped.
  Neither bug affects Phases 1-7 (service layer + unit tests are isolated from `SSETransport`), but both must be fixed before Phase 8 (e2e with real `MCPAdapter` ↔ real `FastMCP`).

---

## File structure

**New files:**

```
packages/atp-dashboard/atp/dashboard/tournament/events.py    — TournamentEvent + TournamentEventBus
packages/atp-dashboard/atp/dashboard/tournament/errors.py    — ValidationError, ConflictError, NotFoundError
packages/atp-dashboard/atp/dashboard/tournament/service.py   — TournamentService
packages/atp-dashboard/atp/dashboard/tournament/state.py     — RoundState dataclass
packages/atp-dashboard/atp/dashboard/mcp/__init__.py         — FastMCP instance + module-level event bus binding
packages/atp-dashboard/atp/dashboard/mcp/auth.py             — MCPAuthMiddleware
packages/atp-dashboard/atp/dashboard/mcp/notifications.py    — _forward_events_to_session, _format_notification_for_user
packages/atp-dashboard/atp/dashboard/mcp/tools.py            — join_tournament, get_current_state, make_move tools

migrations/dashboard/versions/<hash>_tournament_slice_columns.py — additive columns only

tests/unit/dashboard/tournament/__init__.py
tests/unit/dashboard/tournament/test_event_bus.py
tests/unit/dashboard/tournament/test_service_join.py
tests/unit/dashboard/tournament/test_service_resolve.py
tests/unit/dashboard/tournament/test_state_formatter.py
tests/unit/dashboard/tournament/conftest.py                  — shared fixtures (in-memory session, test event bus)

tests/unit/dashboard/mcp/__init__.py
tests/unit/dashboard/mcp/test_auth_middleware.py
tests/unit/dashboard/mcp/test_tools.py
tests/unit/dashboard/mcp/test_notification_format.py

tests/integration/test_mcp_handshake.py
tests/integration/test_mcp_pd_flow.py

tests/e2e/test_mcp_pd_tournament.py                          — acceptance test for the slice

docs/notes/phase0-fastmcp-findings.md                        — Phase 0 verification report (what we learned about FastMCP API)
```

**Modified files:**

```
packages/atp-dashboard/atp/dashboard/tournament/models.py    — add num_players, total_rounds, round_deadline_s, Action.payoff
packages/atp-dashboard/atp/dashboard/v2/factory.py           — mount FastMCP under /mcp
game-environments/game_envs/games/prisoners_dilemma.py       — add format_state_for_player method
pyproject.toml                                                — add fastmcp dependency
packages/atp-adapters/atp/adapters/mcp/...                   — IF Phase 0 reveals MCPAdapter cannot subscribe to notifications, expanded here. Specifics deferred to Phase 0 findings.
```

**Untouched in this plan:**

- `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py` (501 stubs stay until Plan 2)
- `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py` (no UI in slice)
- All existing benchmark, auth, RBAC, dashboard code

---

## Phase 0 — Verification (must run first)

**Goal of phase:** Determine before writing any service-layer code whether two unverified assumptions in the spec hold:

1. `packages/atp-adapters/atp/adapters/mcp/MCPAdapter` (existing 1900+ line in-house client) can sit on a long-running SSE connection and surface server-pushed `notifications/message` to the caller via some callback or async iterator.
2. Mounting a FastMCP `http_app(transport="sse")` under FastAPI as a Starlette sub-app correctly triggers the outer app's middleware stack — specifically, `JWTUserStateMiddleware` must populate `request.state.user_id` on the SSE handshake request.

**Failure mode if skipped:** Either failure surfaces deep in Phase 8 (e2e test) and forces 1+ week of rework or pivot to a different MCP client library.

**Estimated duration:** 2-4 hours total. Both tasks gate the rest of the plan.

### Task 0.1: MCPAdapter notification subscription capability

**Files:**
- Read: `packages/atp-adapters/atp/adapters/mcp/adapter.py`
- Read: `packages/atp-adapters/atp/adapters/mcp/transport.py`
- Create (throwaway): `tests/scratch/test_mcp_adapter_notifications.py`
- Create: `docs/notes/phase0-fastmcp-findings.md`

- [ ] **Step 1: Find the notification surface in MCPAdapter**

Read `packages/atp-adapters/atp/adapters/mcp/adapter.py` and `transport.py`. Search for any of the following patterns and document what you find in `docs/notes/phase0-fastmcp-findings.md` under a section "MCPAdapter notification API":

```
grep -nE "notification|subscribe|on_message|callback|listener|handler" \
    packages/atp-adapters/atp/adapters/mcp/adapter.py \
    packages/atp-adapters/atp/adapters/mcp/transport.py
```

Specifically, you are looking for ONE of:
- A public method like `register_notification_handler(method_name, callback)`
- An async iterator interface like `async for msg in adapter.notifications()`
- A field like `adapter.on_notification` you assign a callable to
- A subclass hook (override a method to receive notifications)

Write your findings in the doc as plain prose: "MCPAdapter exposes notifications via X. To subscribe, you do Y."

- [ ] **Step 2: Write a throwaway integration test**

Create `tests/scratch/test_mcp_adapter_notifications.py`:

```python
"""Throwaway Phase 0 verification — DO NOT keep after Phase 0 passes."""
from __future__ import annotations

import asyncio

import pytest
from fastmcp import FastMCP


@pytest.mark.anyio
async def test_mcp_adapter_can_receive_server_pushed_notifications(
    unused_tcp_port: int,
) -> None:
    """Goal: prove MCPAdapter (in subscription mode) receives 10
    server-pushed notifications/message events sent over 10 seconds.

    If this test cannot be written against the current MCPAdapter API,
    Phase 0 has failed and the plan must be revised before continuing.
    """
    from atp.adapters.mcp.adapter import MCPAdapter
    from atp.adapters.mcp.transport import SSETransport, SSETransportConfig

    # 1. Stand up a trivial FastMCP server that pushes notifications.
    mcp = FastMCP("phase0-test")

    received: list[dict] = []

    @mcp.tool()
    async def start_pushing(ctx) -> str:
        """When called, push 10 notifications over 10 seconds."""
        for i in range(10):
            await ctx.session.send_notification({
                "method": "notifications/message",
                "params": {
                    "level": "info",
                    "data": {"event": "test", "n": i},
                },
            })
            await asyncio.sleep(1)
        return "done"

    # NOTE: actual FastMCP server bootstrap goes here. Look up the
    # current FastMCP "run as ASGI app under uvicorn" pattern in the
    # FastMCP README. Document the exact pattern in
    # docs/notes/phase0-fastmcp-findings.md as you discover it.

    # 2. Connect MCPAdapter to it via SSE.
    transport = SSETransport(
        SSETransportConfig(url=f"http://localhost:{unused_tcp_port}/sse")
    )
    adapter = MCPAdapter(transport=transport)
    await adapter.connect()

    # 3. Register notification handler — the EXACT API depends on what
    # you found in Step 1. Use whichever shape you documented.
    #
    # Example for a callback-style API:
    #     adapter.on_notification = lambda n: received.append(n)
    #
    # Example for an async iterator style:
    #     async def collector():
    #         async for n in adapter.notifications():
    #             received.append(n)
    #     collector_task = asyncio.create_task(collector())
    #
    # Example for a method-based registration:
    #     adapter.register_notification_handler(
    #         "notifications/message",
    #         lambda n: received.append(n),
    #     )
    #
    # Pick the one that matches the actual MCPAdapter API.

    # 4. Trigger the push.
    await adapter.call_tool("start_pushing", {})

    # 5. Wait long enough for all 10 to arrive (with timeout).
    deadline = asyncio.get_event_loop().time() + 15
    while len(received) < 10 and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.1)

    await adapter.disconnect()

    # 6. Assert all 10 received.
    assert len(received) == 10, (
        f"expected 10 notifications, got {len(received)}. "
        f"MCPAdapter notification subscription may not be supported."
    )
    assert [r["params"]["data"]["n"] for r in received] == list(range(10))
```

- [ ] **Step 3: Run the test**

```bash
uv run python -m pytest tests/scratch/test_mcp_adapter_notifications.py -v -s
```

Expected outcomes (one of):
- **PASS**: MCPAdapter can subscribe. Document the exact API in `docs/notes/phase0-fastmcp-findings.md` as the canonical pattern. **Phase 0.1 verified.**
- **FAIL: cannot register handler / no notification API**: MCPAdapter cannot subscribe in current form. Stop. Document the gap in the findings doc, then revise this plan: add tasks under Phase 0 to extend `MCPAdapter` with notification subscription support, OR add a Phase 0.1.bis to evaluate switching to the upstream `mcp` Python package as the participant client. Do NOT continue to Task 0.2 until this is decided.
- **FAIL: timeout, fewer than 10 received**: Some notifications arriving but not all. This is more nuanced — could be flow control, queue overflow, transport bug. Document specifics. May still be usable with workarounds.

- [ ] **Step 4: Document findings**

Write to `docs/notes/phase0-fastmcp-findings.md`:

```markdown
# Phase 0 Findings (2026-04-10)

## Task 0.1: MCPAdapter notification subscription

**Result:** [PASS / FAIL with details]

**MCPAdapter notification API:**

[Concrete code example of how to subscribe to notifications. Copy from
the test code above with the actual API used. This becomes the
canonical pattern referenced from notifications.py later.]

**Implications for the rest of the plan:**

[If PASS: "Use the X pattern in mcp/notifications.py".
 If FAIL: "Add tasks 0.1a, 0.1b to extend MCPAdapter, OR pivot to
 upstream mcp package — decision pending."]
```

- [ ] **Step 5: Commit findings (NOT the throwaway test)**

```bash
git add docs/notes/phase0-fastmcp-findings.md
git commit -m "docs(phase0): MCPAdapter notification capability findings"
```

The scratch test is intentionally NOT committed — it served its purpose as an investigation tool.

---

### Task 0.2: FastMCP + JWTUserStateMiddleware integration

**Files:**
- Modify (temporarily): `packages/atp-dashboard/atp/dashboard/v2/factory.py`
- Create (throwaway): `tests/scratch/test_phase0_mcp_mount.py`
- Modify: `docs/notes/phase0-fastmcp-findings.md`

- [ ] **Step 1: Write the throwaway test**

Create `tests/scratch/test_phase0_mcp_mount.py`:

```python
"""Throwaway Phase 0 verification — verify FastMCP mount triggers
JWTUserStateMiddleware on SSE handshake.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import FastMCP

from atp.dashboard.v2.rate_limit import JWTUserStateMiddleware


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    import atp.dashboard.auth as auth_module
    monkeypatch.setattr(auth_module, "SECRET_KEY", "test-secret-32-bytes-long-padding")


def _make_jwt(user_id: int) -> str:
    return jwt.encode(
        {
            "sub": "alice",
            "user_id": user_id,
            "exp": datetime.now(tz=UTC) + timedelta(hours=1),
        },
        "test-secret-32-bytes-long-padding",
        algorithm="HS256",
    )


def test_fastmcp_mount_sees_jwt_user_state_middleware() -> None:
    """When FastMCP is mounted under FastAPI and the outer app has
    JWTUserStateMiddleware, the SSE handshake request must reach the
    handshake handler with request.state.user_id populated.
    """
    app = FastAPI()
    app.add_middleware(JWTUserStateMiddleware)

    mcp = FastMCP("phase0-mount-test")

    captured_user_id: list[int | None] = []

    @mcp.tool()
    async def whoami(ctx) -> dict:
        # Read user_id from the wrapped request state. Exact API for
        # accessing the underlying Request from a FastMCP Context
        # depends on FastMCP version — discover and document in
        # docs/notes/phase0-fastmcp-findings.md.
        from fastmcp.server.dependencies import get_http_request
        request = get_http_request()
        captured_user_id.append(getattr(request.state, "user_id", None))
        return {"user_id": captured_user_id[-1]}

    # Mount FastMCP's HTTP/SSE app under /mcp (FastMCP 3.x API).
    app.mount("/mcp", mcp.http_app(transport="sse"))

    token = _make_jwt(user_id=42)

    client = TestClient(app)

    # Two checks:
    #   (a) handshake itself accepts the bearer token (200 / SSE stream open)
    #   (b) when whoami tool is called via the SSE channel, captured user_id == 42

    # The exact API for "open SSE session, call tool, read result via
    # FastMCP test client" needs to be discovered. FastMCP may ship a
    # `Client` or `TestClient`. Check the FastMCP docs and document in
    # phase0-fastmcp-findings.md.

    # Alternative if no high-level test client exists: use httpx to
    # POST to the SSE endpoint with Authorization header and verify
    # the handshake at least returns 200/text-event-stream — the
    # captured_user_id verification falls to the next phase.

    # Minimal acceptable check for this throwaway test:
    response = client.get(
        "/mcp/sse",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "text/event-stream",
        },
    )
    assert response.status_code in (200, 405), (
        f"SSE handshake failed unexpectedly: {response.status_code} {response.text}"
    )
```

- [ ] **Step 2: Run the test**

```bash
uv run python -m pytest tests/scratch/test_phase0_mcp_mount.py -v -s
```

Expected: PASS for the handshake-status check. The deeper "tool call sees user_id == 42" check may need a real running uvicorn, which is fine to defer to Phase 8 e2e.

If the handshake itself returns 401 or 500: there is a real integration problem. Stop and investigate before proceeding. Likely causes: middleware order, FastMCP not honoring outer app middleware, missing ASGI lifespan propagation.

- [ ] **Step 3: Discover and document the correct ctx → request access pattern**

While running the test, find by experimentation or by reading FastMCP source how to access the underlying Starlette `Request` (and thus `request.state`) from a `Context` object inside a tool handler. Document the exact attribute path in `docs/notes/phase0-fastmcp-findings.md`:

```markdown
## Task 0.2: FastMCP + JWTUserStateMiddleware integration

**Result:** [PASS / FAIL]

**Accessing request.state from a tool handler:**
```python
# Inside @mcp.tool async def my_tool(ctx: Context, ...):
from fastmcp.server.dependencies import get_http_request
request = get_http_request()
user_id = request.state.user_id
```

**Mount pattern in factory.py (verified working):**
```python
app.mount("/mcp", mcp.http_app(transport="sse"))
```

**Middleware ordering (verified working):**
[List the exact add_middleware order that produces the right outcome.]
```

- [ ] **Step 4: Commit findings**

```bash
git add docs/notes/phase0-fastmcp-findings.md
git commit -m "docs(phase0): FastMCP mount + middleware integration verified"
```

Throwaway test is NOT committed.

- [ ] **Step 5: Phase 0 gate decision**

Re-read both findings sections in `docs/notes/phase0-fastmcp-findings.md`. If both are PASS:
- Continue to Phase 1.
- The findings doc is the canonical reference for FastMCP API patterns used throughout the rest of this plan. Tasks below may say "use the pattern from phase0-fastmcp-findings.md §X.Y".

If either is FAIL, do NOT continue. Revise this plan:
- For 0.1 fail: insert tasks to extend `MCPAdapter` notification API, OR re-do the plan against the upstream `mcp` Python package.
- For 0.2 fail: investigate FastMCP middleware integration, possibly add a custom ASGI wrapper, or use a different mount pattern.

---

## Phase 1 — Schema additive columns (no constraints)

**Goal:** Add only the columns the slice needs (`num_players`, `total_rounds`, `round_deadline_s`, `Action.payoff`). NO `UNIQUE`, NO `NOT NULL` flips, NO new indexes — those are in Plan 2.

**Why this minimum:** the slice runs with `num_players=2` and `total_rounds=3` hardcoded in the test, and stores payoffs per action. That is all the schema needs to do for the e2e to pass.

### Task 1.1: Add additive columns to Tournament and Action models

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/models.py`

- [ ] **Step 1: Add columns to `Tournament` and `Action` SQLAlchemy classes**

Open `packages/atp-dashboard/atp/dashboard/tournament/models.py`. Find the `Tournament` class definition (line ~29). Add three new columns immediately after `rules`:

```python
class Tournament(Base):
    """A tournament definition for game-theoretic evaluation."""

    __tablename__ = "tournaments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=DEFAULT_TENANT_ID,
        index=True,
    )
    game_type: Mapped[str] = mapped_column(String(100), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=TournamentStatus.PENDING,
    )
    starts_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ends_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    rules: Mapped[dict] = mapped_column(JSON, default=dict)
    # NEW columns for v1 vertical slice
    num_players: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="2"
    )
    total_rounds: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="1"
    )
    round_deadline_s: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="30"
    )
    # END new columns
    created_by: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id"),
        nullable=True,
    )
    # ... rest of the class unchanged
```

Find the `Action` class (line ~155). Add `payoff` column:

```python
class Action(Base):
    """An action submitted by a participant in a round."""

    __tablename__ = "tournament_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tournament_rounds.id"),
        nullable=False,
    )
    participant_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tournament_participants.id"),
        nullable=False,
    )
    action_data: Mapped[dict] = mapped_column(JSON, default=dict)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    # NEW column for v1 vertical slice — denormalized payoff per action
    payoff: Mapped[float | None] = mapped_column(Float, nullable=True)
    # END new column

    # Relationships unchanged
```

`Float` is already imported at the top. Confirm with:

```bash
grep -n "from sqlalchemy import" packages/atp-dashboard/atp/dashboard/tournament/models.py
```

Expected: `Float` is in the imports. If not, add it.

- [ ] **Step 2: Run the existing tournament model unit tests to confirm no regression**

```bash
uv run python -m pytest tests/unit/dashboard/ -k "tournament" -v 2>&1 | tail -20
```

Expected: existing tests pass (no new tests yet — they may all be skipped if there are none, that is OK).

### Task 1.2: Generate Alembic migration for the new columns

**Files:**
- Create: `migrations/dashboard/versions/<auto-hash>_tournament_slice_columns.py`

- [ ] **Step 1: Generate the migration**

```bash
cd /Users/Andrei_Shtanakov/labs/all_ai_orchestrators/atp-platform
uv run alembic -c alembic.ini --name dashboard revision --autogenerate -m "tournament_slice_columns"
```

This creates a file in `migrations/dashboard/versions/` named like `abc123_tournament_slice_columns.py`. Open it.

- [ ] **Step 2: Inspect and tighten the autogenerated migration**

The autogenerated migration should add 4 columns:
- `tournaments.num_players` Integer NOT NULL server_default="2"
- `tournaments.total_rounds` Integer NOT NULL server_default="1"
- `tournaments.round_deadline_s` Integer NOT NULL server_default="30"
- `tournament_actions.payoff` Float NULLABLE

If autogenerate produced anything else (renames, drops, type changes), DELETE those lines manually — they are autogenerate noise. The migration must be **only** these four `op.add_column` calls.

The final migration should look like:

```python
"""tournament_slice_columns

Revision ID: <hash>
Revises: c8d5f2a91234
Create Date: 2026-04-10 ...
"""
from alembic import op
import sqlalchemy as sa


revision = "<hash>"
down_revision = "c8d5f2a91234"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tournaments",
        sa.Column("num_players", sa.Integer(), nullable=False, server_default="2"),
    )
    op.add_column(
        "tournaments",
        sa.Column("total_rounds", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "tournaments",
        sa.Column(
            "round_deadline_s", sa.Integer(), nullable=False, server_default="30"
        ),
    )
    op.add_column(
        "tournament_actions",
        sa.Column("payoff", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tournament_actions", "payoff")
    op.drop_column("tournaments", "round_deadline_s")
    op.drop_column("tournaments", "total_rounds")
    op.drop_column("tournaments", "num_players")
```

Verify `down_revision` matches the latest existing migration. Check with:

```bash
ls migrations/dashboard/versions/ | sort
```

The newest before yours should be `c8d5f2a91234_enforce_run_user_id_not_null.py`.

- [ ] **Step 3: Apply migration to a fresh test SQLite DB and verify**

```bash
TMPDB=$(mktemp -t atp-test-XXXXX.db)
DASHBOARD_DATABASE_URL="sqlite:///$TMPDB" \
    uv run alembic -c alembic.ini --name dashboard upgrade head
```

Expected: no error.

Verify the columns landed:

```bash
sqlite3 "$TMPDB" "PRAGMA table_info(tournaments);"
sqlite3 "$TMPDB" "PRAGMA table_info(tournament_actions);"
rm "$TMPDB"
```

Expected: `num_players`, `total_rounds`, `round_deadline_s` in tournaments table; `payoff` in tournament_actions table.

- [ ] **Step 4: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/models.py \
        migrations/dashboard/versions/*_tournament_slice_columns.py
git commit -m "feat(tournament): add slice columns (num_players, total_rounds, round_deadline_s, payoff)"
```

---

## Phase 2 — TournamentEventBus

**Goal:** A minimal in-process pub/sub the service publishes to and the MCP notification layer subscribes from.

### Task 2.1: TournamentEvent dataclass + event_type literal

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/tournament/events.py`
- Create: `tests/unit/dashboard/tournament/__init__.py`
- Create: `tests/unit/dashboard/tournament/test_event_bus.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/tournament/__init__.py` as an empty file.

Create `tests/unit/dashboard/tournament/test_event_bus.py`:

```python
"""Tests for TournamentEventBus and TournamentEvent."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest


def test_tournament_event_dataclass_holds_required_fields() -> None:
    from atp.dashboard.tournament.events import TournamentEvent

    event = TournamentEvent(
        event_type="round_started",
        tournament_id=7,
        round_number=1,
        data={"foo": "bar"},
        timestamp=datetime(2026, 4, 10, 12, 0, 0, tzinfo=UTC),
    )
    assert event.event_type == "round_started"
    assert event.tournament_id == 7
    assert event.round_number == 1
    assert event.data == {"foo": "bar"}
    assert event.timestamp.year == 2026
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_event_bus.py::test_tournament_event_dataclass_holds_required_fields -v
```

Expected: `ImportError: cannot import name 'TournamentEvent' from 'atp.dashboard.tournament.events'` (module does not exist yet).

- [ ] **Step 3: Implement TournamentEvent**

Create `packages/atp-dashboard/atp/dashboard/tournament/events.py`:

```python
"""Tournament event bus and event dataclass.

Protocol-agnostic in-process pub/sub. Used by TournamentService to
broadcast state-change events; consumed by the MCP notification layer
to push notifications/message to connected clients.

Events are ephemeral: not persisted, not replayed on subscriber
reconnect (per AD-7 in the design spec).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

EventType = Literal[
    "round_started",
    "round_ended",
    "tournament_completed",
    "tournament_cancelled",
]


@dataclass(frozen=True)
class TournamentEvent:
    """One state-change event in a tournament's lifecycle.

    The ``data`` field is a generic dict; per-player personalization
    happens at the subscriber level (the MCP notification layer calls
    TournamentService.get_state_for to format a player-private view).
    """

    event_type: EventType
    tournament_id: int
    round_number: int | None
    data: dict[str, Any]
    timestamp: datetime
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_event_bus.py::test_tournament_event_dataclass_holds_required_fields -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/events.py \
        tests/unit/dashboard/tournament/__init__.py \
        tests/unit/dashboard/tournament/test_event_bus.py
git commit -m "feat(tournament): TournamentEvent dataclass"
```

### Task 2.2: TournamentEventBus.publish + subscribe basic flow

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/events.py`
- Modify: `tests/unit/dashboard/tournament/test_event_bus.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/dashboard/tournament/test_event_bus.py`:

```python
@pytest.mark.anyio
async def test_publish_delivers_to_single_subscriber() -> None:
    from atp.dashboard.tournament.events import TournamentEvent, TournamentEventBus

    bus = TournamentEventBus()
    event = TournamentEvent(
        event_type="round_started",
        tournament_id=1,
        round_number=1,
        data={},
        timestamp=datetime.now(tz=UTC),
    )

    async with bus.subscribe(tournament_id=1) as queue:
        await bus.publish(event)
        received = await queue.get()
        assert received is event


@pytest.mark.anyio
async def test_publish_to_other_tournament_does_not_reach_subscriber() -> None:
    from atp.dashboard.tournament.events import TournamentEvent, TournamentEventBus

    bus = TournamentEventBus()
    other_event = TournamentEvent(
        event_type="round_started",
        tournament_id=2,
        round_number=1,
        data={},
        timestamp=datetime.now(tz=UTC),
    )

    async with bus.subscribe(tournament_id=1) as queue:
        await bus.publish(other_event)
        # Subscriber for tournament 1 should NOT receive an event for
        # tournament 2. Verify by trying to read with a tiny timeout.
        import asyncio
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.get(), timeout=0.1)


@pytest.mark.anyio
async def test_publish_to_no_subscribers_is_noop() -> None:
    from atp.dashboard.tournament.events import TournamentEvent, TournamentEventBus

    bus = TournamentEventBus()
    event = TournamentEvent(
        event_type="round_started",
        tournament_id=99,
        round_number=1,
        data={},
        timestamp=datetime.now(tz=UTC),
    )
    # Should not raise.
    await bus.publish(event)


@pytest.mark.anyio
async def test_unsubscribe_on_context_exit_removes_queue() -> None:
    from atp.dashboard.tournament.events import TournamentEventBus

    bus = TournamentEventBus()
    async with bus.subscribe(tournament_id=1):
        pass
    # After exit, internal subscribers dict should not have tournament 1.
    assert 1 not in bus._subscribers
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_event_bus.py -v
```

Expected: 4 new tests fail with `cannot import name 'TournamentEventBus'`. The first test still passes.

- [ ] **Step 3: Implement TournamentEventBus**

Append to `packages/atp-dashboard/atp/dashboard/tournament/events.py`:

```python
import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger("atp.dashboard.tournament.events")

_QUEUE_MAXSIZE = 100


class TournamentEventBus:
    """In-process pub/sub for tournament events.

    Subscribers register per-tournament_id and receive only events for
    that tournament. Each subscriber gets its own asyncio.Queue
    (maxsize=100) so a slow consumer cannot block others.

    Events are ephemeral. Missed events on disconnect are NOT replayed.
    See AD-4 and AD-7 in the design spec.
    """

    def __init__(self) -> None:
        self._subscribers: dict[int, set[asyncio.Queue[TournamentEvent]]] = {}

    async def publish(self, event: TournamentEvent) -> None:
        """Fan out an event to all subscribers of event.tournament_id.

        Best-effort: if a subscriber's queue is full, the event is
        dropped for that subscriber and a warning is logged. Never
        raises — bus.publish is always fire-and-forget.
        """
        queues = self._subscribers.get(event.tournament_id, set())
        for queue in list(queues):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "subscriber queue full for tournament %d, dropping event %s",
                    event.tournament_id,
                    event.event_type,
                )

    @asynccontextmanager
    async def subscribe(
        self, tournament_id: int
    ) -> AsyncIterator[asyncio.Queue[TournamentEvent]]:
        """Subscribe to events for one tournament.

        Usage:
            async with bus.subscribe(tournament_id=7) as queue:
                while True:
                    event = await queue.get()
                    ...

        On context exit (including via task cancellation), the queue
        is removed from the subscribers set.
        """
        queue: asyncio.Queue[TournamentEvent] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._subscribers.setdefault(tournament_id, set()).add(queue)
        try:
            yield queue
        finally:
            subs = self._subscribers.get(tournament_id)
            if subs is not None:
                subs.discard(queue)
                if not subs:
                    del self._subscribers[tournament_id]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_event_bus.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/events.py \
        tests/unit/dashboard/tournament/test_event_bus.py
git commit -m "feat(tournament): TournamentEventBus pub/sub"
```

### Task 2.3: Multi-subscriber fan-out + queue-full drop

**Files:**
- Modify: `tests/unit/dashboard/tournament/test_event_bus.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/dashboard/tournament/test_event_bus.py`:

```python
@pytest.mark.anyio
async def test_publish_fans_out_to_multiple_subscribers() -> None:
    from atp.dashboard.tournament.events import TournamentEvent, TournamentEventBus

    bus = TournamentEventBus()
    event = TournamentEvent(
        event_type="round_started",
        tournament_id=1,
        round_number=1,
        data={},
        timestamp=datetime.now(tz=UTC),
    )

    async with bus.subscribe(tournament_id=1) as q1:
        async with bus.subscribe(tournament_id=1) as q2:
            await bus.publish(event)
            assert (await q1.get()) is event
            assert (await q2.get()) is event


@pytest.mark.anyio
async def test_publish_drops_when_subscriber_queue_full(caplog) -> None:
    import logging

    from atp.dashboard.tournament.events import (
        TournamentEvent,
        TournamentEventBus,
        _QUEUE_MAXSIZE,
    )

    bus = TournamentEventBus()

    async with bus.subscribe(tournament_id=1) as queue:
        # Fill the queue to maxsize without consuming.
        for i in range(_QUEUE_MAXSIZE):
            await bus.publish(
                TournamentEvent(
                    event_type="round_started",
                    tournament_id=1,
                    round_number=i,
                    data={},
                    timestamp=datetime.now(tz=UTC),
                )
            )

        # The next publish should be dropped (with a warning), not raise.
        with caplog.at_level(logging.WARNING, logger="atp.dashboard.tournament.events"):
            await bus.publish(
                TournamentEvent(
                    event_type="round_started",
                    tournament_id=1,
                    round_number=999,
                    data={},
                    timestamp=datetime.now(tz=UTC),
                )
            )

        assert any(
            "queue full" in rec.message for rec in caplog.records
        ), "expected queue-full warning to be logged"
```

- [ ] **Step 2: Run the tests to verify they pass (no implementation change needed)**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_event_bus.py -v
```

Expected: 7 PASSED. The behavior these tests check is already implemented in Task 2.2; this task adds explicit coverage.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/dashboard/tournament/test_event_bus.py
git commit -m "test(tournament): event bus fan-out and queue-full drop"
```

---

## Phase 3 — Tournament errors module

**Goal:** Centralize the three exception classes used throughout the service so test code and route handlers can both import them.

### Task 3.1: errors.py with ValidationError, ConflictError, NotFoundError

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/tournament/errors.py`

- [ ] **Step 1: Create the module**

Create `packages/atp-dashboard/atp/dashboard/tournament/errors.py`:

```python
"""Tournament service exceptions.

These are the only error channel out of TournamentService. Service
methods raise these; transport layers (MCP tool handlers, REST routes)
catch them and translate to ToolError / HTTPException.
"""
from __future__ import annotations


class TournamentError(Exception):
    """Base for all tournament service errors."""


class ValidationError(TournamentError):
    """Invalid input shape, missing required field, unknown game_type,
    action that doesn't match the game's action schema, etc.

    Maps to HTTP 422 / MCP ToolError(422).
    """


class ConflictError(TournamentError):
    """State machine violation: join in ACTIVE, double make_move,
    leave during ACTIVE, etc.

    Maps to HTTP 409 / MCP ToolError(409).
    """


class NotFoundError(TournamentError):
    """Resource does not exist OR is not owned by the requesting user.

    Per the enumeration-guard pattern (Issue 1 fix, design spec
    §Error handling), 404 is returned regardless of which case
    applies. Clients cannot distinguish 'doesn't exist' from
    'exists but not yours'.

    Maps to HTTP 404 / MCP ToolError(404).
    """
```

- [ ] **Step 2: Verify it imports**

```bash
uv run python -c "from atp.dashboard.tournament.errors import ValidationError, ConflictError, NotFoundError; print('ok')"
```

Expected output: `ok`

- [ ] **Step 3: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/errors.py
git commit -m "feat(tournament): error types"
```

---

## Phase 4 — RoundState dataclass + PD format_state_for_player

**Goal:** A typed shape for what a player sees on their turn, and the PD-specific formatter that produces it.

### Task 4.1: RoundState dataclass

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/tournament/state.py`
- Create: `tests/unit/dashboard/tournament/test_state_formatter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/tournament/test_state_formatter.py`:

```python
"""Tests for RoundState dataclass."""
from __future__ import annotations


def test_round_state_to_dict_serializes_all_fields() -> None:
    from atp.dashboard.tournament.state import RoundState

    state = RoundState(
        tournament_id=7,
        round_number=5,
        game_type="prisoners_dilemma",
        your_history=["cooperate", "cooperate", "defect"],
        opponent_history=["cooperate", "defect", "cooperate"],
        your_cumulative_score=8,
        opponent_cumulative_score=10,
        action_schema={"type": "choice", "options": ["cooperate", "defect"]},
        your_turn=True,
        total_rounds=100,
    )

    d = state.to_dict()
    assert d["tournament_id"] == 7
    assert d["round_number"] == 5
    assert d["game_type"] == "prisoners_dilemma"
    assert d["your_history"] == ["cooperate", "cooperate", "defect"]
    assert d["opponent_history"] == ["cooperate", "defect", "cooperate"]
    assert d["your_cumulative_score"] == 8
    assert d["opponent_cumulative_score"] == 10
    assert d["action_schema"] == {"type": "choice", "options": ["cooperate", "defect"]}
    assert d["your_turn"] is True
    assert d["total_rounds"] == 100
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_state_formatter.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement RoundState**

Create `packages/atp-dashboard/atp/dashboard/tournament/state.py`:

```python
"""Player-private RoundState — what one participant sees on their turn.

The shape is intentionally generic enough to fit any 2-player game in
the v1 slice (PD now, Stag Hunt and Battle of Sexes nearly free in
Plan 2). For N-player games (El Farol etc.), the formatter on the
game class returns the same RoundState shape with N-aware fields
populated; this stays out of scope for the v1 slice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RoundState:
    tournament_id: int
    round_number: int
    game_type: str
    your_history: list[str]
    opponent_history: list[str]
    your_cumulative_score: float
    opponent_cumulative_score: float
    action_schema: dict[str, Any]
    your_turn: bool
    total_rounds: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_state_formatter.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/state.py \
        tests/unit/dashboard/tournament/test_state_formatter.py
git commit -m "feat(tournament): RoundState dataclass"
```

### Task 4.2: PD format_state_for_player on the game class

**Files:**
- Modify: `game-environments/game_envs/games/prisoners_dilemma.py`
- Modify: `tests/unit/dashboard/tournament/test_state_formatter.py`

- [ ] **Step 1: Read the existing PD module to find the game class**

```bash
grep -n "^class \|payoff\|cooperate\|defect" game-environments/game_envs/games/prisoners_dilemma.py | head -30
```

Document the class name and any existing helper for payoffs. The new method `format_state_for_player` will go on whichever class represents "the game" (likely `PrisonersDilemma`).

- [ ] **Step 2: Write the failing test**

Append to `tests/unit/dashboard/tournament/test_state_formatter.py`:

```python
def test_pd_format_state_for_player_first_round_no_history() -> None:
    from game_envs.games.prisoners_dilemma import PrisonersDilemma

    game = PrisonersDilemma()
    # Round 1, no actions yet, two participants, formatter should return
    # an empty-history RoundState for participant_idx=0.
    state = game.format_state_for_player(
        round_number=1,
        total_rounds=3,
        participant_idx=0,
        action_history=[],  # list[list[str]] — outer = round, inner = action per participant
        cumulative_scores=[0.0, 0.0],
    )
    assert state["round_number"] == 1
    assert state["game_type"] == "prisoners_dilemma"
    assert state["your_history"] == []
    assert state["opponent_history"] == []
    assert state["your_cumulative_score"] == 0.0
    assert state["opponent_cumulative_score"] == 0.0
    assert state["action_schema"]["options"] == ["cooperate", "defect"]
    assert state["your_turn"] is True
    assert state["total_rounds"] == 3


def test_pd_format_state_for_player_with_history() -> None:
    from game_envs.games.prisoners_dilemma import PrisonersDilemma

    game = PrisonersDilemma()
    # Three rounds played: A=[C,C,D], B=[C,D,C]
    history = [
        ["cooperate", "cooperate"],
        ["cooperate", "defect"],
        ["defect", "cooperate"],
    ]
    # From player A's perspective (idx=0), opponent is B (idx=1).
    state_a = game.format_state_for_player(
        round_number=4,
        total_rounds=10,
        participant_idx=0,
        action_history=history,
        cumulative_scores=[8.0, 10.0],
    )
    assert state_a["your_history"] == ["cooperate", "cooperate", "defect"]
    assert state_a["opponent_history"] == ["cooperate", "defect", "cooperate"]
    assert state_a["your_cumulative_score"] == 8.0
    assert state_a["opponent_cumulative_score"] == 10.0

    # From player B's perspective (idx=1), opponent is A (idx=0).
    state_b = game.format_state_for_player(
        round_number=4,
        total_rounds=10,
        participant_idx=1,
        action_history=history,
        cumulative_scores=[8.0, 10.0],
    )
    assert state_b["your_history"] == ["cooperate", "defect", "cooperate"]
    assert state_b["opponent_history"] == ["cooperate", "cooperate", "defect"]
    assert state_b["your_cumulative_score"] == 10.0
    assert state_b["opponent_cumulative_score"] == 8.0
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_state_formatter.py -v
```

Expected: 2 new tests fail with `AttributeError: 'PrisonersDilemma' object has no attribute 'format_state_for_player'` (or similar).

- [ ] **Step 4: Implement format_state_for_player**

Add to `game-environments/game_envs/games/prisoners_dilemma.py` inside the `PrisonersDilemma` class (or whichever class your Step 1 grep identified):

```python
    def format_state_for_player(
        self,
        round_number: int,
        total_rounds: int,
        participant_idx: int,
        action_history: list[list[str]],
        cumulative_scores: list[float],
    ) -> dict:
        """Build a player-private RoundState dict for the given player.

        Args:
            round_number: The 1-indexed round about to be played.
            total_rounds: Total rounds in the tournament.
            participant_idx: Which participant we are formatting for
                (0 or 1 in 2-player PD).
            action_history: List per round of [player0_action, player1_action]
                strings. Empty list = no rounds played yet.
            cumulative_scores: Per-participant cumulative scores so far.

        Returns:
            Dict matching the RoundState shape, with this player's view
            (your_history vs opponent_history correctly oriented).
        """
        opponent_idx = 1 - participant_idx
        your_history = [round_actions[participant_idx] for round_actions in action_history]
        opponent_history = [round_actions[opponent_idx] for round_actions in action_history]
        return {
            "tournament_id": -1,  # caller fills this in
            "round_number": round_number,
            "game_type": "prisoners_dilemma",
            "your_history": your_history,
            "opponent_history": opponent_history,
            "your_cumulative_score": cumulative_scores[participant_idx],
            "opponent_cumulative_score": cumulative_scores[opponent_idx],
            "action_schema": {
                "type": "choice",
                "options": ["cooperate", "defect"],
            },
            "your_turn": True,
            "total_rounds": total_rounds,
            "extra": {},
        }
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_state_formatter.py -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Run the existing PD test suite to confirm no regression**

```bash
uv run python -m pytest game-environments/tests/test_prisoners_dilemma.py -v 2>&1 | tail -20
```

Expected: existing tests still pass. We added a method, did not modify existing behavior.

- [ ] **Step 7: Commit**

```bash
git add game-environments/game_envs/games/prisoners_dilemma.py \
        tests/unit/dashboard/tournament/test_state_formatter.py
git commit -m "feat(game-envs): PD.format_state_for_player for tournament integration"
```

---

## Phase 5 — TournamentService

**Goal:** A protocol-agnostic service implementing `create_tournament`, `join`, `get_state_for`, `submit_action`, and the private `_start_tournament`, `_resolve_round`, `_complete_tournament`. No deadlines, no constraints, no leave/get_history/list — just enough for the slice.

### Task 5.1: TournamentService skeleton + create_tournament

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Create: `tests/unit/dashboard/tournament/conftest.py`
- Create: `tests/unit/dashboard/tournament/test_service_join.py`

- [ ] **Step 1: Create test fixtures**

Create `tests/unit/dashboard/tournament/conftest.py`:

```python
"""Shared fixtures for tournament service tests."""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from atp.dashboard.models import Base, User
from atp.dashboard.tournament.events import TournamentEventBus

# Tournament models import side-effects: ensure their classes are registered
import atp.dashboard.tournament.models  # noqa: F401


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Fresh in-memory SQLite + all tables, one per test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    async with SessionLocal() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def admin_user(session: AsyncSession) -> User:
    user = User(
        username="admin",
        email="admin@example.com",
        hashed_password="x",
        is_admin=True,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    return user


@pytest_asyncio.fixture
async def alice(session: AsyncSession) -> User:
    user = User(
        username="alice",
        email="alice@example.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    return user


@pytest_asyncio.fixture
async def bob(session: AsyncSession) -> User:
    user = User(
        username="bob",
        email="bob@example.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    return user


@pytest_asyncio.fixture
def event_bus() -> TournamentEventBus:
    return TournamentEventBus()
```

Note: this assumes the existing `atp.dashboard.models.User` schema does not enforce additional NOT NULL columns beyond what we provide. If it does, expand the fixtures by reading the User model and providing values.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/dashboard/tournament/test_service_join.py`:

```python
"""Tests for TournamentService.create_tournament and join lifecycle."""
from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEventBus


@pytest.mark.anyio
async def test_create_tournament_persists_basic_fields(
    session: AsyncSession, admin_user: User, event_bus: TournamentEventBus
) -> None:
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    tournament = await svc.create_tournament(
        admin=admin_user,
        name="slice-test",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )

    assert tournament.id is not None
    assert tournament.game_type == "prisoners_dilemma"
    assert tournament.num_players == 2
    assert tournament.total_rounds == 3
    assert tournament.round_deadline_s == 30
    assert tournament.created_by == admin_user.id
    # Status reused from the existing enum: PENDING = "accepting joins"
    assert tournament.status == "pending"
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_join.py -v
```

Expected: ImportError on `TournamentService`.

- [ ] **Step 4: Implement TournamentService skeleton + create_tournament**

Create `packages/atp-dashboard/atp/dashboard/tournament/service.py`:

```python
"""TournamentService — protocol-agnostic core for tournament gameplay.

This module knows about SQLAlchemy and game-environments. It does NOT
know about FastAPI, FastMCP, or HTTP. Unit-tested via direct calls
with an in-memory session and a test event bus.

This is the v1 vertical slice version: only the methods needed for a
2-player 3-round PD e2e test. Plan 2 expands the surface (deadline
worker, leave/get_history/list, AD-9/AD-10 enforcement, etc.).
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEventBus
from atp.dashboard.tournament.errors import ValidationError
from atp.dashboard.tournament.models import (
    Tournament,
    TournamentStatus,
)

logger = logging.getLogger("atp.dashboard.tournament.service")

_SUPPORTED_GAMES = frozenset({"prisoners_dilemma"})


class TournamentService:
    def __init__(self, session: AsyncSession, bus: TournamentEventBus) -> None:
        self._session = session
        self._bus = bus

    async def create_tournament(
        self,
        admin: User,
        *,
        name: str,
        game_type: str,
        num_players: int,
        total_rounds: int,
        round_deadline_s: int,
    ) -> Tournament:
        """Create a new tournament in PENDING (accepting-joins) state.

        Caller is responsible for verifying admin authorization at the
        transport layer; this method trusts that admin.is_admin == True.
        """
        if game_type not in _SUPPORTED_GAMES:
            raise ValidationError(
                f"unsupported game_type {game_type!r}; "
                f"v1 slice supports: {sorted(_SUPPORTED_GAMES)}"
            )
        if num_players < 2:
            raise ValidationError("num_players must be >= 2")
        if total_rounds < 1:
            raise ValidationError("total_rounds must be >= 1")
        if round_deadline_s < 1:
            raise ValidationError("round_deadline_s must be >= 1")

        tournament = Tournament(
            game_type=game_type,
            status=TournamentStatus.PENDING,
            num_players=num_players,
            total_rounds=total_rounds,
            round_deadline_s=round_deadline_s,
            created_by=admin.id,
            config={"name": name},
        )
        self._session.add(tournament)
        await self._session.flush()
        await self._session.refresh(tournament)
        return tournament
```

Note on the `name` field: the slice does not introduce a separate `name` column on Tournament — that is in Plan 2. We stash it under `config["name"]` for now; the test asserts on the structured fields, not on `name` directly. This keeps Phase 1 schema additive-only.

- [ ] **Step 5: Run the test to verify it passes**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_join.py::test_create_tournament_persists_basic_fields -v
```

Expected: PASS.

- [ ] **Step 6: Add validation tests**

Append to `tests/unit/dashboard/tournament/test_service_join.py`:

```python
@pytest.mark.anyio
async def test_create_tournament_rejects_unknown_game_type(
    session: AsyncSession, admin_user: User, event_bus: TournamentEventBus
) -> None:
    from atp.dashboard.tournament.errors import ValidationError
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    with pytest.raises(ValidationError, match="unsupported game_type"):
        await svc.create_tournament(
            admin=admin_user,
            name="bad-game",
            game_type="chess",
            num_players=2,
            total_rounds=3,
            round_deadline_s=30,
        )


@pytest.mark.anyio
async def test_create_tournament_rejects_invalid_num_players(
    session: AsyncSession, admin_user: User, event_bus: TournamentEventBus
) -> None:
    from atp.dashboard.tournament.errors import ValidationError
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    with pytest.raises(ValidationError, match="num_players"):
        await svc.create_tournament(
            admin=admin_user,
            name="single-player-pd",
            game_type="prisoners_dilemma",
            num_players=1,
            total_rounds=3,
            round_deadline_s=30,
        )
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_join.py -v
```

Expected: 3 PASSED.

- [ ] **Step 8: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/conftest.py \
        tests/unit/dashboard/tournament/test_service_join.py
git commit -m "feat(tournament): TournamentService.create_tournament with validation"
```

### Task 5.2: join — first participant

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Modify: `tests/unit/dashboard/tournament/test_service_join.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/dashboard/tournament/test_service_join.py`:

```python
@pytest.mark.anyio
async def test_join_first_player_creates_participant(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    tournament = await svc.create_tournament(
        admin=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )

    participant = await svc.join(
        tournament_id=tournament.id, user=alice, agent_name="alice-tft"
    )
    assert participant.id is not None
    assert participant.tournament_id == tournament.id
    assert participant.user_id == alice.id
    assert participant.agent_name == "alice-tft"
    # Tournament still PENDING because only 1/2 joined.
    await session.refresh(tournament)
    assert tournament.status == "pending"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_join.py::test_join_first_player_creates_participant -v
```

Expected: AttributeError, `'TournamentService' object has no attribute 'join'`.

- [ ] **Step 3: Implement join (first participant only — start logic in next task)**

Add to `packages/atp-dashboard/atp/dashboard/tournament/service.py` (add the import for `Participant` first):

At the top of the file, change the existing model import to:

```python
from atp.dashboard.tournament.models import (
    Participant,
    Tournament,
    TournamentStatus,
)
```

Add to the imports near `from atp.dashboard.tournament.errors import ValidationError`:

```python
from atp.dashboard.tournament.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
```

Add to the imports for SQLAlchemy:

```python
from sqlalchemy import func, select
```

Add a new method to the `TournamentService` class (after `create_tournament`):

```python
    async def join(
        self,
        tournament_id: int,
        user: User,
        agent_name: str,
    ) -> Participant:
        """Join an open tournament.

        v1 slice: open-join only, no join_token, no
        MAX_ACTIVE_TOURNAMENTS_PER_USER (those land in Plan 2 per
        AD-10). When the join brings participant count to num_players,
        the tournament starts immediately and round 1 is created.
        """
        tournament = await self._session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id} not found")
        if tournament.status != TournamentStatus.PENDING:
            raise ConflictError(
                f"tournament {tournament_id} is {tournament.status}, not accepting joins"
            )

        participant = Participant(
            tournament_id=tournament_id,
            user_id=user.id,
            agent_name=agent_name,
        )
        self._session.add(participant)
        await self._session.flush()
        await self._session.refresh(participant)
        return participant
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_join.py::test_join_first_player_creates_participant -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_join.py
git commit -m "feat(tournament): TournamentService.join (first participant)"
```

### Task 5.3: join — second participant triggers _start_tournament

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Modify: `tests/unit/dashboard/tournament/test_service_join.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/dashboard/tournament/test_service_join.py`:

```python
@pytest.mark.anyio
async def test_join_filling_last_slot_starts_tournament_and_creates_round_1(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.models import Round
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    tournament = await svc.create_tournament(
        admin=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )

    await svc.join(tournament.id, alice, "alice-tft")
    await svc.join(tournament.id, bob, "bob-random")

    await session.refresh(tournament)
    # Status moved from PENDING (pending) to ACTIVE.
    assert tournament.status == "active"

    # Exactly one Round exists, with round_number=1.
    rounds = (
        await session.execute(
            select(Round).where(Round.tournament_id == tournament.id)
        )
    ).scalars().all()
    assert len(rounds) == 1
    assert rounds[0].round_number == 1
    assert rounds[0].status == "waiting_for_actions"
```

You will need this import at the top of the test file (it is already implicitly available via SQLAlchemy):

```python
from sqlalchemy import select
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_join.py::test_join_filling_last_slot_starts_tournament_and_creates_round_1 -v
```

Expected: FAIL with `AssertionError: assert 'pending' == 'active'` or no Round created.

- [ ] **Step 3: Implement _start_tournament and wire join → start**

Add the import for `Round` to `service.py`:

```python
from atp.dashboard.tournament.models import (
    Participant,
    Round,
    Tournament,
    TournamentStatus,
)
```

Also add `datetime` and `timedelta`:

```python
from datetime import datetime, timedelta
```

Modify the `join` method to count participants after insert and trigger start:

```python
    async def join(
        self,
        tournament_id: int,
        user: User,
        agent_name: str,
    ) -> Participant:
        """Join an open tournament.

        v1 slice: open-join only, no join_token, no
        MAX_ACTIVE_TOURNAMENTS_PER_USER (Plan 2). When the join brings
        participant count to num_players, the tournament starts
        immediately and round 1 is created in the same transaction.
        """
        tournament = await self._session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id} not found")
        if tournament.status != TournamentStatus.PENDING:
            raise ConflictError(
                f"tournament {tournament_id} is {tournament.status}, not accepting joins"
            )

        participant = Participant(
            tournament_id=tournament_id,
            user_id=user.id,
            agent_name=agent_name,
        )
        self._session.add(participant)
        await self._session.flush()
        await self._session.refresh(participant)

        # Count participants (including this one).
        count = await self._session.scalar(
            select(func.count(Participant.id)).where(
                Participant.tournament_id == tournament_id
            )
        )
        if count == tournament.num_players:
            await self._start_tournament(tournament)

        return participant

    async def _start_tournament(self, tournament: Tournament) -> None:
        """Transition a PENDING tournament to ACTIVE and create round 1."""
        tournament.status = TournamentStatus.ACTIVE
        tournament.starts_at = datetime.now()
        round_1 = Round(
            tournament_id=tournament.id,
            round_number=1,
            status="waiting_for_actions",
            started_at=datetime.now(),
            deadline=datetime.now() + timedelta(seconds=tournament.round_deadline_s),
            state={},
        )
        self._session.add(round_1)
        await self._session.flush()
        # Event publish happens AFTER caller commits (see commit barrier
        # invariant in spec). For the slice we publish on flush since
        # the test fixtures use a session that auto-commits on success.
        # In Plan 2 we add an explicit pending-events queue committed
        # after session.commit().
```

Note on the publish-after-commit invariant: the slice cuts a corner. The proper pattern (queue events, publish after `session.commit()`) lands in Plan 2. For the slice's e2e test, the caller will explicitly `await session.commit()` and the bus publish happens on the next service-method boundary. This is good enough to make the e2e green and is documented as an explicit follow-up.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_join.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_join.py
git commit -m "feat(tournament): _start_tournament transitions to ACTIVE and creates round 1"
```

### Task 5.4: get_state_for — happy path on round 1

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Create: `tests/unit/dashboard/tournament/test_service_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/tournament/test_service_state.py`:

```python
"""Tests for TournamentService.get_state_for."""
from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEventBus


@pytest.mark.anyio
async def test_get_state_for_round_1_no_history(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    tournament = await svc.create_tournament(
        admin=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(tournament.id, alice, "alice")
    await svc.join(tournament.id, bob, "bob")

    state = await svc.get_state_for(tournament.id, alice)

    assert state.tournament_id == tournament.id
    assert state.round_number == 1
    assert state.game_type == "prisoners_dilemma"
    assert state.your_history == []
    assert state.opponent_history == []
    assert state.your_cumulative_score == 0.0
    assert state.opponent_cumulative_score == 0.0
    assert state.action_schema["options"] == ["cooperate", "defect"]
    assert state.your_turn is True
    assert state.total_rounds == 3
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_state.py -v
```

Expected: AttributeError, `'TournamentService' object has no attribute 'get_state_for'`.

- [ ] **Step 3: Implement get_state_for**

Add this method to `TournamentService` in `service.py` (after `_start_tournament`). First add imports:

```python
from sqlalchemy.orm import selectinload

from atp.dashboard.tournament.models import Action
from atp.dashboard.tournament.state import RoundState
```

And the game registry:

```python
from game_envs.games.prisoners_dilemma import PrisonersDilemma

_GAME_INSTANCES: dict[str, Any] = {
    "prisoners_dilemma": PrisonersDilemma(),
}
```

Then the method:

```python
    async def get_state_for(
        self,
        tournament_id: int,
        user: User,
    ) -> RoundState:
        """Build a player-private RoundState for the current round.

        v1 slice raises NotFoundError if user is not a participant of
        the tournament.
        """
        tournament = await self._session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id} not found")

        # Find this user's participant row + collect all participants
        # ordered by id (id order == seat order in the slice; Plan 2
        # introduces explicit seat_index).
        participants = (
            await self._session.execute(
                select(Participant)
                .where(Participant.tournament_id == tournament_id)
                .order_by(Participant.id)
            )
        ).scalars().all()
        my_idx = next(
            (i for i, p in enumerate(participants) if p.user_id == user.id),
            None,
        )
        if my_idx is None:
            raise NotFoundError(f"tournament {tournament_id} not found")

        # Load all rounds with their actions, ordered.
        rounds = (
            await self._session.execute(
                select(Round)
                .where(Round.tournament_id == tournament_id)
                .order_by(Round.round_number)
                .options(selectinload(Round.actions))
            )
        ).scalars().all()

        # Build action history: list per completed round of
        # [participant0_action_str, participant1_action_str, ...].
        # Cumulative scores are summed payoffs from completed rounds.
        action_history: list[list[str]] = []
        cumulative_scores: list[float] = [0.0] * len(participants)

        current_round_number = 1
        for r in rounds:
            if r.status == "completed":
                row: list[str] = [""] * len(participants)
                for action in r.actions:
                    p_idx = next(
                        i for i, p in enumerate(participants)
                        if p.id == action.participant_id
                    )
                    row[p_idx] = action.action_data.get("choice", "")
                    cumulative_scores[p_idx] += action.payoff or 0.0
                action_history.append(row)
            else:
                current_round_number = r.round_number
                break
        else:
            # Tournament has no in-progress rounds (all completed).
            current_round_number = len(rounds)

        game = _GAME_INSTANCES[tournament.game_type]
        formatted = game.format_state_for_player(
            round_number=current_round_number,
            total_rounds=tournament.total_rounds,
            participant_idx=my_idx,
            action_history=action_history,
            cumulative_scores=cumulative_scores,
        )
        return RoundState(
            tournament_id=tournament_id,
            round_number=formatted["round_number"],
            game_type=formatted["game_type"],
            your_history=formatted["your_history"],
            opponent_history=formatted["opponent_history"],
            your_cumulative_score=formatted["your_cumulative_score"],
            opponent_cumulative_score=formatted["opponent_cumulative_score"],
            action_schema=formatted["action_schema"],
            your_turn=formatted["your_turn"],
            total_rounds=formatted["total_rounds"],
            extra=formatted.get("extra", {}),
        )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_state.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_state.py
git commit -m "feat(tournament): get_state_for builds RoundState from rounds + actions"
```

### Task 5.5: submit_action — first action of a round

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Create: `tests/unit/dashboard/tournament/test_service_resolve.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/tournament/test_service_resolve.py`:

```python
"""Tests for TournamentService.submit_action and round resolution."""
from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEventBus


@pytest.mark.anyio
async def test_submit_action_first_player_returns_waiting(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t = await svc.create_tournament(
        admin=admin_user, name="t", game_type="prisoners_dilemma",
        num_players=2, total_rounds=3, round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    result = await svc.submit_action(
        t.id, alice, action={"choice": "cooperate"}
    )
    assert result["status"] == "waiting"
    assert result["round_number"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_resolve.py::test_submit_action_first_player_returns_waiting -v
```

Expected: AttributeError, `'TournamentService' object has no attribute 'submit_action'`.

- [ ] **Step 3: Implement submit_action (waiting branch only)**

Add to `TournamentService` in `service.py`:

```python
    async def submit_action(
        self,
        tournament_id: int,
        user: User,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit one player's action for the current round.

        Returns:
            {"status": "waiting", "round_number": N} if more actions
            are still expected this round.

            {"status": "round_resolved", ...} if this was the last
            action and the round resolved synchronously (handled in
            Task 5.6).
        """
        tournament = await self._session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id} not found")
        if tournament.status != TournamentStatus.ACTIVE:
            raise ConflictError(
                f"tournament {tournament_id} is {tournament.status}, not active"
            )

        # Find this user's participant.
        my_participant = (
            await self._session.execute(
                select(Participant).where(
                    Participant.tournament_id == tournament_id,
                    Participant.user_id == user.id,
                )
            )
        ).scalar_one_or_none()
        if my_participant is None:
            raise NotFoundError(f"tournament {tournament_id} not found")

        # Find the current waiting_for_actions round.
        current_round = (
            await self._session.execute(
                select(Round)
                .where(
                    Round.tournament_id == tournament_id,
                    Round.status == "waiting_for_actions",
                )
                .order_by(Round.round_number.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if current_round is None:
            raise ConflictError(
                f"tournament {tournament_id} has no round accepting actions"
            )

        # Validate action shape against the game schema.
        game = _GAME_INSTANCES[tournament.game_type]
        if action.get("choice") not in game.format_state_for_player(
            round_number=1, total_rounds=1, participant_idx=0,
            action_history=[], cumulative_scores=[0.0, 0.0],
        )["action_schema"]["options"]:
            raise ValidationError(
                f"invalid action {action!r} for game {tournament.game_type}"
            )

        # Check whether this player already submitted in this round.
        existing = (
            await self._session.execute(
                select(Action).where(
                    Action.round_id == current_round.id,
                    Action.participant_id == my_participant.id,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise ConflictError(
                f"participant {my_participant.id} already submitted in round "
                f"{current_round.round_number}"
            )

        # Insert the action.
        new_action = Action(
            round_id=current_round.id,
            participant_id=my_participant.id,
            action_data={"choice": action["choice"]},
        )
        self._session.add(new_action)
        await self._session.flush()

        # Count actions for this round.
        action_count = await self._session.scalar(
            select(func.count(Action.id)).where(Action.round_id == current_round.id)
        )

        if action_count < tournament.num_players:
            return {
                "status": "waiting",
                "round_number": current_round.round_number,
            }

        # Last action — resolve the round (Task 5.6 implements this).
        return await self._resolve_round(current_round, tournament)
```

You will get a NameError on `_resolve_round` until Task 5.6, but the test for "waiting" branch should pass because it returns before reaching that line. Verify:

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_resolve.py::test_submit_action_first_player_returns_waiting -v
```

Expected: PASS (the waiting branch returns early).

- [ ] **Step 5: Add stub _resolve_round so other call sites do not break**

Add a stub in `service.py`:

```python
    async def _resolve_round(
        self, round_obj: Round, tournament: Tournament
    ) -> dict[str, Any]:
        """Resolve a round when all actions are present (Task 5.6)."""
        raise NotImplementedError("implemented in Task 5.6")
```

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_resolve.py
git commit -m "feat(tournament): submit_action waiting branch"
```

### Task 5.6: _resolve_round + last-action triggers resolve

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Modify: `tests/unit/dashboard/tournament/test_service_resolve.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/dashboard/tournament/test_service_resolve.py`:

```python
@pytest.mark.anyio
async def test_submit_action_last_player_resolves_round(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """Alice cooperates, Bob defects → Bob gets 5, Alice gets 0 (PD payoff)."""
    from atp.dashboard.tournament.models import Action, Round
    from atp.dashboard.tournament.service import TournamentService
    from sqlalchemy import select

    svc = TournamentService(session, event_bus)
    t = await svc.create_tournament(
        admin=admin_user, name="t", game_type="prisoners_dilemma",
        num_players=2, total_rounds=3, round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
    result = await svc.submit_action(t.id, bob, action={"choice": "defect"})

    assert result["status"] == "round_resolved"
    assert result["round_number"] == 1

    # Round 1 is now completed.
    rounds = (
        await session.execute(
            select(Round).where(Round.tournament_id == t.id).order_by(Round.round_number)
        )
    ).scalars().all()
    assert len(rounds) == 2  # round 1 completed + round 2 created
    assert rounds[0].status == "completed"
    assert rounds[1].status == "waiting_for_actions"
    assert rounds[1].round_number == 2

    # Action payoffs are persisted.
    actions = (
        await session.execute(
            select(Action).where(Action.round_id == rounds[0].id)
        )
    ).scalars().all()
    by_user = {}
    for a in actions:
        # Map to user via participant
        from atp.dashboard.tournament.models import Participant
        p = await session.get(Participant, a.participant_id)
        by_user[p.user_id] = a
    assert by_user[alice.id].action_data["choice"] == "cooperate"
    assert by_user[alice.id].payoff == 0.0
    assert by_user[bob.id].action_data["choice"] == "defect"
    assert by_user[bob.id].payoff == 5.0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_resolve.py::test_submit_action_last_player_resolves_round -v
```

Expected: NotImplementedError on _resolve_round stub.

- [ ] **Step 3: Implement _resolve_round**

Replace the stub `_resolve_round` in `service.py` with:

```python
    async def _resolve_round(
        self, round_obj: Round, tournament: Tournament
    ) -> dict[str, Any]:
        """Resolve a round: compute payoffs, write Action.payoff,
        mark round completed, create the next round (or finish the
        tournament if this was the last).
        """
        # Atomic guard against double-resolve.
        round_obj.status = "resolving"
        await self._session.flush()

        # Load actions in deterministic seat order.
        actions = (
            await self._session.execute(
                select(Action)
                .where(Action.round_id == round_obj.id)
                .order_by(Action.participant_id)
            )
        ).scalars().all()

        # Map to participant index — id order == seat order in the slice.
        participants = (
            await self._session.execute(
                select(Participant)
                .where(Participant.tournament_id == tournament.id)
                .order_by(Participant.id)
            )
        ).scalars().all()
        idx_by_pid = {p.id: i for i, p in enumerate(participants)}

        # Build action vector ordered by seat.
        action_vec: list[str] = [""] * len(participants)
        actions_by_idx: dict[int, Action] = {}
        for a in actions:
            i = idx_by_pid[a.participant_id]
            action_vec[i] = a.action_data["choice"]
            actions_by_idx[i] = a

        # PD payoff matrix (hardcoded in slice; in Plan 2 this comes
        # from the game-environments PD class).
        # CC = 3,3 ; CD = 0,5 ; DC = 5,0 ; DD = 1,1
        a0, a1 = action_vec[0], action_vec[1]
        if a0 == "cooperate" and a1 == "cooperate":
            payoffs = [3.0, 3.0]
        elif a0 == "cooperate" and a1 == "defect":
            payoffs = [0.0, 5.0]
        elif a0 == "defect" and a1 == "cooperate":
            payoffs = [5.0, 0.0]
        else:  # both defect
            payoffs = [1.0, 1.0]

        # Write payoffs into actions.
        for i, action in actions_by_idx.items():
            action.payoff = payoffs[i]

        round_obj.status = "completed"
        await self._session.flush()

        # Determine if this was the last round.
        if round_obj.round_number >= tournament.total_rounds:
            await self._complete_tournament(tournament)
            await self._session.flush()
            return {
                "status": "round_resolved",
                "round_number": round_obj.round_number,
                "tournament_completed": True,
                "payoffs": payoffs,
            }

        # Create the next round.
        next_round = Round(
            tournament_id=tournament.id,
            round_number=round_obj.round_number + 1,
            status="waiting_for_actions",
            started_at=datetime.now(),
            deadline=datetime.now() + timedelta(seconds=tournament.round_deadline_s),
            state={},
        )
        self._session.add(next_round)
        await self._session.flush()

        return {
            "status": "round_resolved",
            "round_number": round_obj.round_number,
            "tournament_completed": False,
            "payoffs": payoffs,
            "next_round_number": next_round.round_number,
        }

    async def _complete_tournament(self, tournament: Tournament) -> None:
        """Mark tournament COMPLETED, write final scores."""
        tournament.status = TournamentStatus.COMPLETED
        tournament.ends_at = datetime.now()

        participants = (
            await self._session.execute(
                select(Participant).where(
                    Participant.tournament_id == tournament.id
                )
            )
        ).scalars().all()
        for p in participants:
            total = await self._session.scalar(
                select(func.coalesce(func.sum(Action.payoff), 0.0))
                .where(Action.participant_id == p.id)
            )
            p.total_score = float(total or 0.0)
        await self._session.flush()
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_resolve.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Add the full 3-round game test**

Append to `tests/unit/dashboard/tournament/test_service_resolve.py`:

```python
@pytest.mark.anyio
async def test_full_3_round_pd_tournament_completes(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """End-to-end at the service level: alice always cooperates, bob
    always defects, 3 rounds. Final scores: alice=0, bob=15."""
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t = await svc.create_tournament(
        admin=admin_user, name="t", game_type="prisoners_dilemma",
        num_players=2, total_rounds=3, round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    for round_n in range(1, 4):
        await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
        result = await svc.submit_action(t.id, bob, action={"choice": "defect"})
        assert result["round_number"] == round_n

    await session.refresh(t)
    assert t.status == "completed"

    # Final scores.
    from atp.dashboard.tournament.models import Participant
    from sqlalchemy import select
    parts = (
        await session.execute(
            select(Participant).where(Participant.tournament_id == t.id)
        )
    ).scalars().all()
    by_user = {p.user_id: p for p in parts}
    assert by_user[alice.id].total_score == 0.0
    assert by_user[bob.id].total_score == 15.0
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_resolve.py::test_full_3_round_pd_tournament_completes -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_resolve.py
git commit -m "feat(tournament): _resolve_round + _complete_tournament + full PD scoring"
```

### Task 5.7: Wire event bus publishes into _start_tournament, _resolve_round, _complete_tournament

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Modify: `tests/unit/dashboard/tournament/test_service_resolve.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/dashboard/tournament/test_service_resolve.py`:

```python
@pytest.mark.anyio
async def test_full_3_round_publishes_round_started_and_tournament_completed(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """Verify the slice's two notification events flow through the bus."""
    from atp.dashboard.tournament.events import TournamentEvent
    from atp.dashboard.tournament.service import TournamentService

    received: list[TournamentEvent] = []

    svc = TournamentService(session, event_bus)

    t = await svc.create_tournament(
        admin=admin_user, name="t", game_type="prisoners_dilemma",
        num_players=2, total_rounds=3, round_deadline_s=30,
    )

    # Subscribe BEFORE joins (which trigger _start_tournament → first
    # round_started publish).
    import asyncio
    async def collect():
        async with event_bus.subscribe(t.id) as queue:
            for _ in range(4):  # 3 round_started + 1 tournament_completed
                event = await queue.get()
                received.append(event)

    collector = asyncio.create_task(collect())
    # Yield once so the subscription registers before publish.
    await asyncio.sleep(0)

    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")
    for _ in range(3):
        await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
        await svc.submit_action(t.id, bob, action={"choice": "defect"})

    await asyncio.wait_for(collector, timeout=2.0)

    assert [e.event_type for e in received] == [
        "round_started",
        "round_started",
        "round_started",
        "tournament_completed",
    ]
    assert [e.round_number for e in received[:3]] == [1, 2, 3]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_resolve.py::test_full_3_round_publishes_round_started_and_tournament_completed -v
```

Expected: TimeoutError or empty `received` — service does not yet publish.

- [ ] **Step 3: Add publishes to service methods**

In `service.py`, find `_start_tournament` and append to it after the `Round` flush:

```python
        await self._bus.publish(
            TournamentEvent(
                event_type="round_started",
                tournament_id=tournament.id,
                round_number=1,
                data={"total_rounds": tournament.total_rounds},
                timestamp=datetime.now(),
            )
        )
```

In `_resolve_round`, after creating `next_round` and the flush:

```python
        await self._bus.publish(
            TournamentEvent(
                event_type="round_started",
                tournament_id=tournament.id,
                round_number=next_round.round_number,
                data={"total_rounds": tournament.total_rounds},
                timestamp=datetime.now(),
            )
        )
```

In `_complete_tournament`, after the participant scoring loop and flush:

```python
        await self._bus.publish(
            TournamentEvent(
                event_type="tournament_completed",
                tournament_id=tournament.id,
                round_number=None,
                data={
                    "final_scores": {
                        p.user_id: p.total_score for p in participants
                    },
                },
                timestamp=datetime.now(),
            )
        )
```

Also add the import at the top of `service.py`:

```python
from atp.dashboard.tournament.events import TournamentEvent, TournamentEventBus
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/test_service_resolve.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Run all tournament unit tests as a regression check**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/ -v
```

Expected: all tests PASS (event bus, state formatter, service join/state/resolve).

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_resolve.py
git commit -m "feat(tournament): publish round_started + tournament_completed events"
```

---

## Phase 6 — MCP server scaffold

**Goal:** Mount FastMCP under FastAPI at `/mcp`, with auth middleware that reads `request.state.user_id` (already populated by `JWTUserStateMiddleware`).

### Task 6.1: Add fastmcp dependency

**Files:**
- Modify: `pyproject.toml` (root or `packages/atp-dashboard/pyproject.toml`, depending on where dashboard deps live)

- [ ] **Step 1: Find which pyproject.toml lists FastAPI**

```bash
grep -l "fastapi" pyproject.toml packages/*/pyproject.toml
```

The match is the file to add `fastmcp` to.

- [ ] **Step 2: Add fastmcp using uv with version constraint**

```bash
uv add 'fastmcp>=3.0'
```

(If the grep showed it should go into a workspace package, run `uv add --package atp-dashboard 'fastmcp>=3.0'` instead.)

The `>=3.0` constraint is mandatory: the plan is verified against FastMCP 3.x API (`http_app(transport="sse")`, `get_http_request()`). 2.x uses different patterns (`sse_app()`, different ctx access) that would break every Phase 6-7 task.

- [ ] **Step 3: Verify install**

```bash
uv run python -c "import fastmcp; print(fastmcp.__version__)"
```

Expected: a version string, e.g. `2.x.x`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add fastmcp dependency"
```

### Task 6.2: Create MCPAuthMiddleware

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/mcp/__init__.py` (empty namespace marker)
- Create: `packages/atp-dashboard/atp/dashboard/mcp/auth.py`
- Create: `tests/unit/dashboard/mcp/__init__.py`
- Create: `tests/unit/dashboard/mcp/test_auth_middleware.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/mcp/__init__.py` (empty).

Create `tests/unit/dashboard/mcp/test_auth_middleware.py`:

```python
"""Tests for MCPAuthMiddleware: rejects requests without request.state.user_id."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from atp.dashboard.mcp.auth import MCPAuthMiddleware


def _make_app(set_user_id: int | None) -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def _set_state(request, call_next):
        if set_user_id is not None:
            request.state.user_id = set_user_id
        return await call_next(request)

    app.add_middleware(MCPAuthMiddleware)

    @app.get("/_test")
    async def _test() -> dict:
        return {"ok": True}

    return app


def test_mcp_auth_rejects_when_no_user_id() -> None:
    client = TestClient(_make_app(set_user_id=None))
    response = client.get("/_test")
    assert response.status_code == 401


def test_mcp_auth_accepts_when_user_id_present() -> None:
    client = TestClient(_make_app(set_user_id=42))
    response = client.get("/_test")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
```

Note: middleware order in the test app — the `_set_state` http middleware MUST run before `MCPAuthMiddleware` (Starlette LIFO: added later runs first). In the test app `MCPAuthMiddleware` is added after, so on incoming request it runs FIRST and sees state empty. We reverse this with the order, so we need to add `_set_state` AFTER `MCPAuthMiddleware`. Re-write:

```python
def _make_app(set_user_id: int | None) -> FastAPI:
    app = FastAPI()
    # MCPAuthMiddleware first, so _set_state wraps it on request path.
    app.add_middleware(MCPAuthMiddleware)

    @app.middleware("http")
    async def _set_state(request, call_next):
        if set_user_id is not None:
            request.state.user_id = set_user_id
        return await call_next(request)

    @app.get("/_test")
    async def _test() -> dict:
        return {"ok": True}

    return app
```

- [ ] **Step 2: Create the empty mcp namespace and run the failing test**

Create `packages/atp-dashboard/atp/dashboard/mcp/__init__.py`:

```python
"""MCP server module — FastMCP instance, tools, auth, notifications."""
```

Run:

```bash
uv run python -m pytest tests/unit/dashboard/mcp/test_auth_middleware.py -v
```

Expected: ImportError on `atp.dashboard.mcp.auth`.

- [ ] **Step 3: Implement MCPAuthMiddleware**

Create `packages/atp-dashboard/atp/dashboard/mcp/auth.py`:

```python
"""Authentication gate for the MCP SSE endpoint.

Sits in front of the FastMCP mount and rejects any request that does
not have `request.state.user_id` populated by the outer
JWTUserStateMiddleware. The actual JWT decoding is done upstream;
this module only enforces presence.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class MCPAuthMiddleware(BaseHTTPMiddleware):
    """Reject MCP requests without an authenticated user_id."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        user_id = getattr(request.state, "user_id", None)
        if user_id is None:
            return JSONResponse(
                {"error": "unauthorized", "detail": "Bearer JWT required"},
                status_code=401,
            )
        return await call_next(request)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run python -m pytest tests/unit/dashboard/mcp/test_auth_middleware.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/mcp/__init__.py \
        packages/atp-dashboard/atp/dashboard/mcp/auth.py \
        tests/unit/dashboard/mcp/__init__.py \
        tests/unit/dashboard/mcp/test_auth_middleware.py
git commit -m "feat(mcp): MCPAuthMiddleware enforces request.state.user_id"
```

### Task 6.3: Mount FastMCP under FastAPI in factory.py

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/factory.py`
- Modify: `packages/atp-dashboard/atp/dashboard/mcp/__init__.py`

- [ ] **Step 1: Define a module-level FastMCP instance**

Replace `packages/atp-dashboard/atp/dashboard/mcp/__init__.py` with:

```python
"""MCP server module — FastMCP instance, tools, auth, notifications."""
from __future__ import annotations

from fastmcp import FastMCP

from atp.dashboard.tournament.events import TournamentEventBus

# Module-level singletons. The bus is shared between the service
# layer (publish) and the MCP notification layer (subscribe). The
# FastMCP instance is mounted under /mcp in factory.py.
mcp_server: FastMCP = FastMCP("atp-platform-tournaments")
tournament_event_bus: TournamentEventBus = TournamentEventBus()
```

- [ ] **Step 2: Mount it in factory.py (with lifespan composition — critical gotcha from Phase 0)**

Open `packages/atp-dashboard/atp/dashboard/v2/factory.py`. This task has TWO parts: (a) mount the FastMCP sub-app, and (b) compose its lifespan into the outer FastAPI lifespan. Part (b) is non-negotiable — Phase 0.2 verified that Starlette does NOT propagate sub-app lifespans automatically, and without composition FastMCP's internal session manager never initializes.

First, find the existing `lifespan` function and the `create_app` body. Before `app = FastAPI(...)`, compose lifespans:

```python
from contextlib import asynccontextmanager

# ... inside create_app, AFTER computing config but BEFORE creating app ...
from atp.dashboard.mcp import mcp_server
from atp.dashboard.mcp.auth import MCPAuthMiddleware
from atp.dashboard.mcp import tools  # noqa: F401 — registers tools as side effect

mcp_app = mcp_server.http_app(transport="sse")

# Compose outer lifespan with FastMCP's inner lifespan. The inner
# lifespan is what initializes FastMCP's session manager; Starlette
# does NOT propagate sub-app lifespans under mount(), so we have to
# drive it ourselves from the outer FastAPI lifespan.
_original_lifespan = lifespan  # the existing lifespan defined higher in this file

@asynccontextmanager
async def _combined_lifespan(app_):
    async with _original_lifespan(app_):
        async with mcp_app.router.lifespan_context(app_):
            yield

app_settings["lifespan"] = _combined_lifespan
```

Then the `FastAPI(**app_settings)` call picks up the combined lifespan. After `app = FastAPI(**app_settings)`, add the mount + middleware (still before `app.include_router(api_router, prefix="/api")`):

```python
    # Mount the MCP tournament server under /mcp.
    # MCPAuthMiddleware sits between JWTUserStateMiddleware (which
    # populates request.state.user_id) and FastMCP, rejecting
    # unauthenticated handshakes with 401.
    mcp_app.add_middleware(MCPAuthMiddleware)
    app.mount("/mcp", mcp_app)
```

The `tools` import is a side-effect import: importing it registers all `@mcp_server.tool()` decorators. We will create that module in the next task.

**Note on API:** use `mcp_server.http_app(transport="sse")`, NOT `mcp_server.sse_app()` — the latter does not exist in FastMCP 3.x. See the Phase 0 FastMCP corrections at the top of this plan.

- [ ] **Step 3: Create a stub `tools.py` so the import does not fail**

Create `packages/atp-dashboard/atp/dashboard/mcp/tools.py`:

```python
"""MCP tool handlers — populated in subsequent tasks."""
from __future__ import annotations

from atp.dashboard.mcp import mcp_server

# Tools registered in tasks 7.1-7.3.
# Importing this module registers them via the @mcp_server.tool() decorator.
```

- [ ] **Step 4: Run the existing factory tests to confirm no regression**

```bash
uv run python -m pytest tests/unit/dashboard/test_factory.py tests/unit/dashboard/test_rate_limit.py -v 2>&1 | tail -10
```

Expected: existing tests pass.

- [ ] **Step 5: Smoke-test that the app boots**

```bash
uv run python -c "
from atp.dashboard.v2.factory import create_app
app = create_app()
print('routes:', [r.path for r in app.routes])
" 2>&1 | head -20
```

Expected: a list of routes including `/mcp` somewhere.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/mcp/__init__.py \
        packages/atp-dashboard/atp/dashboard/mcp/tools.py \
        packages/atp-dashboard/atp/dashboard/v2/factory.py
git commit -m "feat(mcp): mount FastMCP under /mcp with MCPAuthMiddleware"
```

---

## Phase 7 — MCP tools

**Goal:** Three tool handlers that wrap `TournamentService` calls. Notification subscription on `join_tournament`. Per-player formatted notifications via `_format_notification_for_user`.

> **NOTE:** The exact way `Context` (`ctx`) provides access to the underlying `Request` and to `send_notification` differs between FastMCP versions. **Use the patterns documented in `docs/notes/phase0-fastmcp-findings.md` from Task 0.2.** The code below uses placeholder attribute paths (`ctx.request_context.request`, `ctx.session.send_notification`) — adjust to match the verified API before running the tests.

### Task 7.1: join_tournament tool

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/mcp/tools.py`
- Create: `tests/unit/dashboard/mcp/test_tools.py`

- [ ] **Step 1: Write the failing test (mock-based unit test)**

Create `tests/unit/dashboard/mcp/test_tools.py`:

```python
"""Tests for MCP tool handlers (mocked service layer)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.anyio
async def test_join_tournament_calls_service_and_returns_participant_info() -> None:
    """The tool should call TournamentService.join with the right
    arguments and return a dict with tournament_id and participant_id.
    """
    from atp.dashboard.mcp import tools

    # Build a fake context that exposes user_id and a session.
    fake_user = MagicMock(id=42, is_admin=False)
    fake_participant = MagicMock(id=99)
    fake_service = MagicMock()
    fake_service.join = AsyncMock(return_value=fake_participant)

    # Patch the helper that resolves user + service from ctx.
    result = await tools._join_tournament_impl(
        tournament_id=7,
        agent_name="alice-tft",
        user=fake_user,
        service=fake_service,
    )

    fake_service.join.assert_awaited_once_with(
        tournament_id=7, user=fake_user, agent_name="alice-tft"
    )
    assert result == {
        "tournament_id": 7,
        "participant_id": 99,
        "agent_name": "alice-tft",
        "status": "joined",
    }
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run python -m pytest tests/unit/dashboard/mcp/test_tools.py::test_join_tournament_calls_service_and_returns_participant_info -v
```

Expected: AttributeError on `tools._join_tournament_impl`.

- [ ] **Step 3: Implement _join_tournament_impl + register tool**

Replace `packages/atp-dashboard/atp/dashboard/mcp/tools.py` with:

```python
"""MCP tool handlers for tournament gameplay.

This module is imported for side effects (registering tools on the
mcp_server FastMCP instance). The actual logic is in private
``_*_impl`` helpers that take user and service as arguments to make
unit testing trivial.

Phase 0 verification (docs/notes/phase0-fastmcp-findings.md) is the
source of truth for the exact ctx attribute paths used here.
"""
from __future__ import annotations

import logging
from typing import Any

from atp.dashboard.mcp import mcp_server, tournament_event_bus
from atp.dashboard.tournament.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from atp.dashboard.tournament.service import TournamentService

logger = logging.getLogger("atp.dashboard.mcp.tools")


# ---------------------------------------------------------------------------
# Pure-impl helpers (unit-tested directly with mocks)
# ---------------------------------------------------------------------------


async def _join_tournament_impl(
    *,
    tournament_id: int,
    agent_name: str,
    user: Any,
    service: TournamentService,
) -> dict[str, Any]:
    participant = await service.join(
        tournament_id=tournament_id, user=user, agent_name=agent_name
    )
    return {
        "tournament_id": tournament_id,
        "participant_id": participant.id,
        "agent_name": agent_name,
        "status": "joined",
    }


# ---------------------------------------------------------------------------
# FastMCP-registered tool entry points
# ---------------------------------------------------------------------------


@mcp_server.tool()
async def join_tournament(
    ctx,  # type: ignore[no-untyped-def]  # FastMCP Context — type from API discovery
    tournament_id: int,
    agent_name: str,
) -> dict:
    """Join an open tournament.

    Starts an event subscription for this MCP session — round_started
    and tournament_completed notifications are pushed automatically
    until the session disconnects.
    """
    from atp.dashboard.mcp.notifications import (
        forward_events_to_session,
        resolve_user_from_ctx,
        with_service,
    )

    user = await resolve_user_from_ctx(ctx)
    async with with_service(ctx, tournament_event_bus) as service:
        result = await _join_tournament_impl(
            tournament_id=tournament_id,
            agent_name=agent_name,
            user=user,
            service=service,
        )

    # Spawn the per-session notification forwarder for this tournament.
    await forward_events_to_session(ctx, tournament_id, user)
    return result
```

The two helpers `resolve_user_from_ctx`, `with_service`, and `forward_events_to_session` will be implemented in `notifications.py` (Task 7.4). For now they don't exist — but the test uses `_join_tournament_impl` directly so it doesn't need them.

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run python -m pytest tests/unit/dashboard/mcp/test_tools.py::test_join_tournament_calls_service_and_returns_participant_info -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/mcp/tools.py \
        tests/unit/dashboard/mcp/test_tools.py
git commit -m "feat(mcp): join_tournament tool + _impl helper"
```

### Task 7.2: get_current_state and make_move tools

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/mcp/tools.py`
- Modify: `tests/unit/dashboard/mcp/test_tools.py`

- [ ] **Step 1: Write failing tests for both impl helpers**

Append to `tests/unit/dashboard/mcp/test_tools.py`:

```python
@pytest.mark.anyio
async def test_get_current_state_impl_returns_state_dict() -> None:
    from atp.dashboard.mcp import tools

    fake_user = MagicMock(id=42)
    fake_state = MagicMock()
    fake_state.to_dict.return_value = {
        "tournament_id": 7, "round_number": 5, "your_turn": True
    }
    fake_service = MagicMock()
    fake_service.get_state_for = AsyncMock(return_value=fake_state)

    result = await tools._get_current_state_impl(
        tournament_id=7, user=fake_user, service=fake_service
    )
    fake_service.get_state_for.assert_awaited_once_with(7, fake_user)
    assert result == {"tournament_id": 7, "round_number": 5, "your_turn": True}


@pytest.mark.anyio
async def test_make_move_impl_passes_action_to_service() -> None:
    from atp.dashboard.mcp import tools

    fake_user = MagicMock(id=42)
    fake_service = MagicMock()
    fake_service.submit_action = AsyncMock(
        return_value={"status": "waiting", "round_number": 1}
    )

    result = await tools._make_move_impl(
        tournament_id=7,
        action={"choice": "cooperate"},
        user=fake_user,
        service=fake_service,
    )
    fake_service.submit_action.assert_awaited_once_with(
        7, fake_user, action={"choice": "cooperate"}
    )
    assert result == {"status": "waiting", "round_number": 1}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run python -m pytest tests/unit/dashboard/mcp/test_tools.py -v
```

Expected: 2 new tests fail with AttributeError.

- [ ] **Step 3: Implement the helpers + tool wrappers**

Append to `packages/atp-dashboard/atp/dashboard/mcp/tools.py`:

```python
async def _get_current_state_impl(
    *,
    tournament_id: int,
    user: Any,
    service: TournamentService,
) -> dict[str, Any]:
    state = await service.get_state_for(tournament_id, user)
    return state.to_dict()


async def _make_move_impl(
    *,
    tournament_id: int,
    action: dict[str, Any],
    user: Any,
    service: TournamentService,
) -> dict[str, Any]:
    return await service.submit_action(tournament_id, user, action=action)


@mcp_server.tool()
async def get_current_state(
    ctx,  # type: ignore[no-untyped-def]
    tournament_id: int,
) -> dict:
    """Return a player-private RoundState for the current round."""
    from atp.dashboard.mcp.notifications import resolve_user_from_ctx, with_service

    user = await resolve_user_from_ctx(ctx)
    async with with_service(ctx, tournament_event_bus) as service:
        return await _get_current_state_impl(
            tournament_id=tournament_id, user=user, service=service
        )


@mcp_server.tool()
async def make_move(
    ctx,  # type: ignore[no-untyped-def]
    tournament_id: int,
    action: dict,
) -> dict:
    """Submit an action for the current round."""
    from atp.dashboard.mcp.notifications import resolve_user_from_ctx, with_service

    user = await resolve_user_from_ctx(ctx)
    async with with_service(ctx, tournament_event_bus) as service:
        return await _make_move_impl(
            tournament_id=tournament_id,
            action=action,
            user=user,
            service=service,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run python -m pytest tests/unit/dashboard/mcp/test_tools.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/mcp/tools.py \
        tests/unit/dashboard/mcp/test_tools.py
git commit -m "feat(mcp): get_current_state and make_move tools"
```

### Task 7.3: notifications.py — resolve_user_from_ctx, with_service, _format_notification_for_user

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/mcp/notifications.py`
- Create: `tests/unit/dashboard/mcp/test_notification_format.py`

- [ ] **Step 1: Write the failing test for notification formatting**

Create `tests/unit/dashboard/mcp/test_notification_format.py`:

```python
"""Tests for the per-player notification formatter."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.tournament.events import TournamentEvent


@pytest.mark.anyio
async def test_format_round_started_calls_get_state_for_and_wraps_payload() -> None:
    from atp.dashboard.mcp.notifications import _format_notification_for_user

    fake_user = MagicMock(id=42)
    fake_state = MagicMock()
    fake_state.to_dict.return_value = {"round_number": 5, "your_turn": True}
    fake_service = MagicMock()
    fake_service.get_state_for = AsyncMock(return_value=fake_state)

    event = TournamentEvent(
        event_type="round_started",
        tournament_id=7,
        round_number=5,
        data={"total_rounds": 10},
        timestamp=datetime.now(tz=UTC),
    )

    notification = await _format_notification_for_user(event, fake_user, fake_service)

    fake_service.get_state_for.assert_awaited_once_with(7, fake_user)
    assert notification is not None
    assert notification["method"] == "notifications/message"
    assert notification["params"]["data"]["event"] == "round_started"
    assert notification["params"]["data"]["tournament_id"] == 7
    assert notification["params"]["data"]["round_number"] == 5
    assert notification["params"]["data"]["state"] == {
        "round_number": 5,
        "your_turn": True,
    }


@pytest.mark.anyio
async def test_format_tournament_completed_does_not_call_get_state_for() -> None:
    from atp.dashboard.mcp.notifications import _format_notification_for_user

    fake_user = MagicMock(id=42)
    fake_service = MagicMock()
    fake_service.get_state_for = AsyncMock()

    event = TournamentEvent(
        event_type="tournament_completed",
        tournament_id=7,
        round_number=None,
        data={"final_scores": {42: 0.0, 43: 15.0}},
        timestamp=datetime.now(tz=UTC),
    )

    notification = await _format_notification_for_user(event, fake_user, fake_service)

    fake_service.get_state_for.assert_not_awaited()
    assert notification is not None
    assert notification["params"]["data"]["event"] == "tournament_completed"
    assert notification["params"]["data"]["final_scores"] == {42: 0.0, 43: 15.0}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run python -m pytest tests/unit/dashboard/mcp/test_notification_format.py -v
```

Expected: ImportError on `atp.dashboard.mcp.notifications`.

- [ ] **Step 3: Implement notifications.py**

Create `packages/atp-dashboard/atp/dashboard/mcp/notifications.py`:

```python
"""Per-session event forwarder and notification formatter.

When a player joins a tournament via the join_tournament tool, this
module spawns a background asyncio.Task that subscribes to the
TournamentEventBus for that tournament and forwards each event as an
MCP notifications/message to the session.

Per-player personalization is done HERE, not in the service layer
(see Service layer §Invariants in the spec).

Phase 0 verification (docs/notes/phase0-fastmcp-findings.md) provides
the exact ctx → request, ctx → session.send_notification, and tool
context patterns used in this module.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import get_database
from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEvent, TournamentEventBus
from atp.dashboard.tournament.service import TournamentService

logger = logging.getLogger("atp.dashboard.mcp.notifications")

# session_id → tournament_id → asyncio.Task
_session_tasks: dict[str, dict[int, asyncio.Task[None]]] = {}


# ---------------------------------------------------------------------------
# ctx helpers — exact attribute paths come from Phase 0 findings
# ---------------------------------------------------------------------------


async def resolve_user_from_ctx(ctx: Any) -> User:
    """Look up the User row corresponding to ctx's session.

    Reads request.state.user_id from the underlying Starlette request
    via FastMCP's public get_http_request() helper. This is the
    verified pattern from Phase 0.2 findings — do NOT use
    ctx.request_context.request (internal, version-fragile).
    """
    from fastmcp.server.dependencies import get_http_request

    request = get_http_request()
    user_id: int | None = getattr(request.state, "user_id", None)
    if user_id is None:
        raise RuntimeError("MCP session has no authenticated user_id")

    db = get_database()
    async with db.session() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise RuntimeError(f"User {user_id} not found in database")
        return user


@asynccontextmanager
async def with_service(
    ctx: Any, bus: TournamentEventBus
) -> AsyncIterator[TournamentService]:
    """Yield a TournamentService bound to a fresh DB session.

    The session is closed on context exit. Used by tool handlers so
    each tool call is one transactional unit.
    """
    db = get_database()
    async with db.session() as session:
        yield TournamentService(session, bus)
        await session.commit()


# ---------------------------------------------------------------------------
# Notification formatting and forwarding
# ---------------------------------------------------------------------------


async def _format_notification_for_user(
    event: TournamentEvent,
    user: User,
    service: TournamentService,
) -> dict[str, Any] | None:
    """Convert a TournamentEvent into a per-player MCP notifications/message.

    For round_started, calls service.get_state_for to build the
    player-private RoundState. For tournament_completed, the
    leaderboard is global so no per-player call is needed.
    """
    if event.event_type == "round_started":
        state = await service.get_state_for(event.tournament_id, user)
        return {
            "method": "notifications/message",
            "params": {
                "level": "info",
                "logger": "atp.tournament",
                "data": {
                    "event": "round_started",
                    "tournament_id": event.tournament_id,
                    "round_number": event.round_number,
                    "state": state.to_dict(),
                },
            },
        }
    if event.event_type == "tournament_completed":
        return {
            "method": "notifications/message",
            "params": {
                "level": "info",
                "logger": "atp.tournament",
                "data": {
                    "event": "tournament_completed",
                    "tournament_id": event.tournament_id,
                    "final_scores": event.data.get("final_scores", {}),
                },
            },
        }
    return None


async def forward_events_to_session(
    ctx: Any, tournament_id: int, user: User
) -> None:
    """Spawn a background task that forwards events for one tournament
    to one MCP session. Cancelled when the session disconnects or
    leaves the tournament.
    """
    from atp.dashboard.mcp import tournament_event_bus

    # Discover session_id from ctx — exact path is in phase0 findings.
    session_id = getattr(ctx, "session_id", None) or id(ctx)

    async def _forward() -> None:
        try:
            async with tournament_event_bus.subscribe(tournament_id) as queue:
                while True:
                    event = await queue.get()
                    db = get_database()
                    async with db.session() as session:
                        service = TournamentService(session, tournament_event_bus)
                        notification = await _format_notification_for_user(
                            event, user, service
                        )
                    if notification is None:
                        continue
                    # PLACEHOLDER — replace with verified send_notification path.
                    await ctx.session.send_notification(notification)  # type: ignore[attr-defined]
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "notification forwarder crashed for session=%s tournament=%d",
                session_id, tournament_id,
            )

    task = asyncio.create_task(_forward())
    _session_tasks.setdefault(session_id, {})[tournament_id] = task
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run python -m pytest tests/unit/dashboard/mcp/test_notification_format.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/mcp/notifications.py \
        tests/unit/dashboard/mcp/test_notification_format.py
git commit -m "feat(mcp): notification formatter and per-session forwarder"
```

---

## Phase Pre-8 — MCPAdapter SSETransport remediation

**Goal:** Fix the two latent bugs in `packages/atp-adapters/atp/adapters/mcp/transport.py` that Phase 0.1 surfaced, so that Phase 8's e2e test can use stock `MCPAdapter` against stock `FastMCP` without monkey-patches.

**Why this is a phase, not a simple task.** The fix touches live production code (`MCPAdapter`) that is used by every existing SSE-based MCP adapter in the ATP platform. It affects other test suites and has non-trivial concurrency implications. Each task in this phase is independently committable and regression-safe for existing users.

**Scope guardrail.** If implementation in this phase starts to exceed ~1.5 days or the changes cascade into other adapter modules, STOP and split this phase into its own standalone plan (call it `docs/superpowers/plans/2026-04-XX-mcpadapter-sse-demux.md`). Do not let the scope expand inside the tournament slice plan.

**Source of the bug descriptions:** `docs/notes/phase0-fastmcp-findings.md` §"Critical caveats — two latent bugs in SSETransport" — READ IT BEFORE STARTING.

### Task Pre-8.1: Fix `SSETransport._read_sse_events` endpoint frame parsing

**Files:**
- Modify: `packages/atp-adapters/atp/adapters/mcp/transport.py` (around line 820)
- Modify or create: `tests/unit/adapters/mcp/test_sse_transport.py` (find the existing location — there are some SSE tests in the repo)

**The bug** (verbatim from `_read_sse_events`, transport.py:820):

```python
try:
    data = json.loads(event_data)
    if event_type == "endpoint" and "uri" in data:
        self._session_id = data.get("sessionId")
    await self._message_queue.put(data)
except json.JSONDecodeError:
    pass  # Skip malformed events
```

MCP-compliant servers (FastMCP, reference SDK) send the endpoint frame as a bare path:

```
event: endpoint
data: /messages/?session_id=e519ec7f97674620865b363add0d9da9
```

`json.loads("/messages/?session_id=...")` raises `JSONDecodeError`, the `except: pass` swallows it, `_session_id` is never set, and `self._config.post_endpoint` is never updated. Every subsequent `send()` POSTs to the `/sse` GET URL → HTTP 405.

- [ ] **Step 1: Find existing SSETransport tests**

```bash
find tests -name "test_*sse*" -o -name "*transport*test*" 2>&1 | head
grep -rnl "SSETransport\|_read_sse_events" tests/ 2>&1 | head
```

If there is an existing test file for `SSETransport`, extend it. If not, create `tests/unit/adapters/mcp/test_sse_transport.py`.

- [ ] **Step 2: Write failing tests**

Test 1: when the transport receives a spec-compliant endpoint frame (`event: endpoint\ndata: /messages/?session_id=abc123\n\n`), `_session_id` is set to `"abc123"` and `self._config.post_endpoint` is set to the absolute URL `"{base_url}/messages/?session_id=abc123"`.

Test 2: legacy JSON-shaped endpoint frame (`data: {"uri": "/messages/", "sessionId": "xyz"}\n\n`) ALSO works (backwards compatibility — don't break any existing user who relied on the old, wrong, interpretation).

Test 3: the `_message_queue` does NOT receive the endpoint frame as a regular message (endpoint frames are metadata, not messages).

The exact test harness depends on what exists — most likely feed a `httpx.MockTransport` or stub the SSE stream with an `async def` that yields lines from a list.

- [ ] **Step 3: Implement the fix**

Replace the `_read_sse_events` event-completion block with:

```python
elif line == "" and event_data:
    # Empty line signals end of event
    if event_type == "endpoint":
        # MCP spec: endpoint frame data is a bare path, not JSON.
        # Legacy: some older stubs sent JSON with {"uri": ..., "sessionId": ...}.
        # Support both for backwards compatibility.
        endpoint_path: str | None = None
        session_id: str | None = None
        try:
            parsed = json.loads(event_data)
            if isinstance(parsed, dict) and "uri" in parsed:
                endpoint_path = parsed["uri"]
                session_id = parsed.get("sessionId")
        except json.JSONDecodeError:
            # Spec-compliant: bare path
            endpoint_path = event_data.strip()

        if endpoint_path:
            # Resolve against the SSE GET URL to get the absolute POST URL.
            from urllib.parse import urljoin, urlparse, parse_qs
            self._config.post_endpoint = urljoin(self._config.url, endpoint_path)
            if session_id is None:
                qs = parse_qs(urlparse(endpoint_path).query)
                session_id = qs.get("session_id", [None])[0]
            if session_id:
                self._session_id = session_id
        # Do NOT enqueue the endpoint frame — it is metadata, not a message.
    else:
        try:
            data = json.loads(event_data)
            await self._message_queue.put(data)
        except json.JSONDecodeError:
            pass  # Skip malformed data events

    event_type = ""
    event_data = ""
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run python -m pytest tests/unit/adapters/mcp/test_sse_transport.py -v
```

Expected: all new tests PASS, existing tests unchanged.

- [ ] **Step 5: Run the full adapter test suite as a regression check**

```bash
uv run python -m pytest tests/unit/adapters/ -v 2>&1 | tail -20
```

Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-adapters/atp/adapters/mcp/transport.py \
        tests/unit/adapters/mcp/test_sse_transport.py
git commit -m "fix(adapters-mcp): parse spec-compliant SSE endpoint frame as bare path"
```

### Task Pre-8.2: Demux `SSETransport` responses and notifications into separate queues

**Files:**
- Modify: `packages/atp-adapters/atp/adapters/mcp/transport.py` (`_read_sse_events`, `_wait_for_response`, `stream_events`, `receive`)
- Modify: `tests/unit/adapters/mcp/test_sse_transport.py`

**The bug** (verbatim from `_wait_for_response`, transport.py:308):

```python
async def _wait_for_response(self, request_id: int | str) -> dict[str, Any]:
    while True:
        response = await self.receive()
        if response.get("id") == request_id:
            return response
        # Could buffer other messages here for out-of-order responses
```

`receive()` pulls from `_message_queue` which is the ONLY queue. When a notification arrives during a pending `send_request()`, it is popped, its `id` does not match, it is silently dropped. When a consumer task runs `stream_events()` in parallel with `send_request()`, they race for every message — and since the consumer cannot look at `id`, responses get stolen.

**The fix.** Split `_message_queue` into two queues inside `SSETransport`:

- `_response_futures: dict[int | str, asyncio.Future[dict]]` — pending `send_request` call lookups by request_id
- `_notification_queue: asyncio.Queue[dict]` — server-pushed notifications

`_read_sse_events` becomes the single routing point: for each incoming JSON-RPC dict, if it has an `id` that matches a pending future, resolve the future; otherwise (no `id`, or unknown `id`), enqueue into `_notification_queue`.

`send_request` registers a future under its request_id before sending, awaits the future (with timeout), and cleans up on return.

`stream_events` drains `_notification_queue` only.

- [ ] **Step 1: Write failing tests**

Test 1 — notification during in-flight request: start a concurrent notification stream, issue `send_request("tools/call", ...)` while 3 notifications arrive during the request, verify that (a) `send_request` gets exactly the matching response, (b) all 3 notifications arrive via `stream_events()`, none are dropped.

Test 2 — multiple concurrent requests: issue two `send_request` calls in parallel, each gets the correct response by id.

Test 3 — orphan notification before any consumer: publish 2 notifications before any `stream_events()` starts, then start a consumer, verify it receives both (queue was buffering).

Test 4 — response without matching future: a response with an unknown `id` goes into `_notification_queue` (graceful fallback, not an error) — document and accept this behavior.

- [ ] **Step 2: Run tests to verify they fail**

Expected: current shared-queue implementation fails test 1 (notifications dropped) and test 2 (race condition).

- [ ] **Step 3: Implement demux**

Refactor roughly along these lines (exact shape depends on existing code — do NOT blindly replace; integrate carefully):

```python
class SSETransport(MCPTransport):
    def __init__(self, config: SSETransportConfig) -> None:
        super().__init__(config)
        self._config = config
        # ... existing fields ...
        self._response_futures: dict[
            int | str, asyncio.Future[dict[str, Any]]
        ] = {}
        self._notification_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._futures_lock = asyncio.Lock()

    async def _read_sse_events(self) -> None:
        # ... existing event parsing into `data` dict ...
        async def _route(data: dict[str, Any]) -> None:
            if "id" in data and "method" not in data:
                # This is a response — try to deliver to a waiting future.
                async with self._futures_lock:
                    fut = self._response_futures.pop(data["id"], None)
                if fut is not None and not fut.done():
                    fut.set_result(data)
                    return
                # Orphan response — fall through to notification queue.
            # Everything else (notifications, orphan responses) → notifications.
            await self._notification_queue.put(data)

        # Inside the line-parsing loop, replace `await self._message_queue.put(data)`
        # with `await _route(data)`.

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | list[Any] | None = None,
        request_id: int | str | None = None,
    ) -> dict[str, Any]:
        # ... existing request_id generation ...
        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        async with self._futures_lock:
            self._response_futures[request_id] = future
        try:
            await self.send(request)
            return await asyncio.wait_for(future, timeout=self._config.request_timeout)
        finally:
            async with self._futures_lock:
                self._response_futures.pop(request_id, None)

    async def receive(self) -> dict[str, Any]:
        """Blocking receive from the notification queue.

        Deprecated for new code — prefer `stream_events()` or the
        response-future path via `send_request()`. Kept for any
        existing caller that polls.
        """
        return await self._notification_queue.get()

    async def stream_events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield server-pushed notifications as they arrive."""
        while self._state == TransportState.CONNECTED:
            msg = await self._notification_queue.get()
            yield msg
```

**Critical:** keep the OLD `_message_queue` attribute in place and keep `_wait_for_response` as a back-compat shim that raises `NotImplementedError("replaced by send_request future path")` — OR leave it untouched and route old callers transparently. Existing tests and existing adapter code MUST keep working.

- [ ] **Step 4: Run the new demux tests**

```bash
uv run python -m pytest tests/unit/adapters/mcp/test_sse_transport.py -v
```

Expected: all new tests PASS.

- [ ] **Step 5: Full regression sweep across adapters**

```bash
uv run python -m pytest tests/unit/adapters/ tests/integration/ -v 2>&1 | tail -30
```

Expected: no regressions. If existing tests break, it means the old `_message_queue` path is still in use somewhere. Do one of:
- Add a back-compat shim in `receive()` / `_wait_for_response()` that pulls from `_notification_queue` with request-id filtering (temporary compromise).
- Update the calling code.
Prefer a back-compat shim to keep the blast radius small.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-adapters/atp/adapters/mcp/transport.py \
        tests/unit/adapters/mcp/test_sse_transport.py
git commit -m "fix(adapters-mcp): demux SSE responses and notifications into separate queues"
```

### Task Pre-8.3: Verify Phase 0.1 scratch test passes without monkey-patch

**Files:**
- Modify (temporarily): `tests/scratch/test_mcp_adapter_notifications.py`

- [ ] **Step 1: Remove the monkey-patch**

The scratch test from Task 0.1 at `tests/scratch/test_mcp_adapter_notifications.py` contains a monkey-patch of `_read_sse_events` to work around Bug #1. Remove it. The scratch test should now rely on stock `SSETransport`.

- [ ] **Step 2: Run the scratch test**

```bash
uv run --with fastmcp python -m pytest tests/scratch/test_mcp_adapter_notifications.py -v -s
```

Expected: PASS without monkey-patch. All 10 notifications received via stock `MCPAdapter._transport.stream_events()`. If this passes, both bugs are genuinely fixed from MCPAdapter's perspective.

If it fails, one of the two fixes is incomplete. Debug and iterate.

- [ ] **Step 3: Leave the scratch test untracked**

Do not commit the scratch test. It remains an investigation artifact.

- [ ] **Step 4: No commit** — this is verification only.

---

## Phase 8 — End-to-end test

**Goal:** Two `MCPAdapter` bots play a 3-round PD tournament against a real uvicorn instance, and the test asserts both the protocol behavior (notifications received, tools return correct shapes) and the persisted DB state.

> **Prerequisite:** Phase 0 verification AND Phase Pre-8 (`SSETransport` remediation) must have PASSED. The exact MCPAdapter notification API comes from `docs/notes/phase0-fastmcp-findings.md` — use `adapter._transport.stream_events()` pattern, OR a public method added in a follow-up if Phase Pre-8 also adds one.

### Task 8.1: e2e fixtures — uvicorn server, test users, JWT helpers

**Files:**
- Create: `tests/e2e/__init__.py` (empty)
- Create: `tests/e2e/conftest.py`
- Create: `tests/e2e/test_mcp_pd_tournament.py`

- [ ] **Step 1: Create e2e fixtures**

Create `tests/e2e/__init__.py` empty.

Create `tests/e2e/conftest.py`:

```python
"""Fixtures for MCP tournament e2e tests."""
from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import jwt
import pytest_asyncio
import uvicorn
from sqlalchemy.ext.asyncio import async_sessionmaker

from atp.dashboard.models import User
from atp.dashboard.v2.factory import create_app


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest_asyncio.fixture
async def e2e_server(tmp_path, monkeypatch) -> AsyncIterator[tuple[str, int]]:
    """Boot a real uvicorn instance on a free port with an
    ephemeral SQLite database."""
    db_path = tmp_path / "e2e.db"
    monkeypatch.setenv("ATP_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("ATP_SECRET_KEY", "e2e-test-secret-32-bytes-long-pad")
    monkeypatch.setenv("ATP_RATE_LIMIT_ENABLED", "false")

    import atp.dashboard.auth as auth_module
    monkeypatch.setattr(auth_module, "SECRET_KEY", "e2e-test-secret-32-bytes-long-pad")

    port = _free_port()
    app = create_app()

    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Wait for the port to accept connections.
    deadline = asyncio.get_event_loop().time() + 5
    while asyncio.get_event_loop().time() < deadline:
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            break
        except OSError:
            await asyncio.sleep(0.05)
    else:
        raise RuntimeError(f"uvicorn did not come up on port {port}")

    yield (f"http://127.0.0.1:{port}", port)

    server.should_exit = True
    await server_task


def _make_jwt(user_id: int, username: str) -> str:
    return jwt.encode(
        {
            "sub": username,
            "user_id": user_id,
            "exp": datetime.now(tz=UTC) + timedelta(hours=1),
        },
        "e2e-test-secret-32-bytes-long-pad",
        algorithm="HS256",
    )


@pytest_asyncio.fixture
async def seeded_users(e2e_server) -> dict[str, dict]:
    """Insert admin + alice + bob in the e2e database, return their JWTs."""
    from atp.dashboard.database import get_database

    db = get_database()
    async with db.session() as session:
        admin = User(
            username="admin", email="admin@e2e", hashed_password="x",
            is_admin=True, is_active=True,
        )
        alice = User(
            username="alice", email="alice@e2e", hashed_password="x",
            is_admin=False, is_active=True,
        )
        bob = User(
            username="bob", email="bob@e2e", hashed_password="x",
            is_admin=False, is_active=True,
        )
        session.add_all([admin, alice, bob])
        await session.commit()
        await session.refresh(admin)
        await session.refresh(alice)
        await session.refresh(bob)

        return {
            "admin": {"id": admin.id, "username": "admin", "jwt": _make_jwt(admin.id, "admin")},
            "alice": {"id": alice.id, "username": "alice", "jwt": _make_jwt(alice.id, "alice")},
            "bob":   {"id": bob.id,   "username": "bob",   "jwt": _make_jwt(bob.id, "bob")},
        }
```

- [ ] **Step 2: Verify fixtures work in isolation**

Create a tiny smoke test in `tests/e2e/test_mcp_pd_tournament.py`:

```python
"""End-to-end PD tournament test (acceptance criterion for v1 slice)."""
from __future__ import annotations

import pytest


@pytest.mark.anyio
async def test_e2e_server_boots(e2e_server) -> None:
    base_url, port = e2e_server
    assert base_url.startswith("http://127.0.0.1:")


@pytest.mark.anyio
async def test_seeded_users_have_jwts(e2e_server, seeded_users) -> None:
    assert "alice" in seeded_users
    assert seeded_users["alice"]["jwt"]
```

Run:

```bash
uv run python -m pytest tests/e2e/test_mcp_pd_tournament.py -v
```

Expected: 2 PASSED. If anything fails, the most likely culprits are: env var name mismatch (the test app may read different env var names than CLAUDE.md documents — verify), database migration not applied at startup, or `SECRET_KEY` patching not taking effect because auth_module already cached it.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/__init__.py tests/e2e/conftest.py tests/e2e/test_mcp_pd_tournament.py
git commit -m "test(e2e): MCP tournament server bootstrap fixtures"
```

### Task 8.2: e2e — admin creates tournament via direct service call

**Files:**
- Modify: `tests/e2e/test_mcp_pd_tournament.py`

- [ ] **Step 1: Add the test**

Append to `tests/e2e/test_mcp_pd_tournament.py`:

```python
@pytest.mark.anyio
async def test_admin_creates_tournament_directly(
    e2e_server, seeded_users
) -> None:
    """Until Plan 2 adds the REST admin endpoint, the test creates the
    tournament by calling the service directly inside the same process.
    The MCP path (for participants) is exercised separately.
    """
    from atp.dashboard.database import get_database
    from atp.dashboard.models import User
    from atp.dashboard.tournament.events import TournamentEventBus
    from atp.dashboard.tournament.service import TournamentService

    db = get_database()
    async with db.session() as session:
        admin = await session.get(User, seeded_users["admin"]["id"])
        svc = TournamentService(session, TournamentEventBus())
        t = await svc.create_tournament(
            admin=admin,
            name="e2e-pd",
            game_type="prisoners_dilemma",
            num_players=2,
            total_rounds=3,
            round_deadline_s=30,
        )
        await session.commit()
        assert t.id is not None
```

- [ ] **Step 2: Run the test**

```bash
uv run python -m pytest tests/e2e/test_mcp_pd_tournament.py::test_admin_creates_tournament_directly -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_mcp_pd_tournament.py
git commit -m "test(e2e): admin creates tournament via direct service call"
```

### Task 8.3: e2e — two MCPAdapter bots play 3-round PD via SSE

**Files:**
- Modify: `tests/e2e/test_mcp_pd_tournament.py`

- [ ] **Step 1: Add the acceptance test**

> **Phase 0 dependency.** This test uses placeholder MCPAdapter API. Replace with the patterns documented in `docs/notes/phase0-fastmcp-findings.md` §Task 0.1 before running.

Append to `tests/e2e/test_mcp_pd_tournament.py`:

```python
@pytest.mark.anyio
async def test_two_bots_play_pd_tournament_end_to_end(
    e2e_server, seeded_users
) -> None:
    """ACCEPTANCE TEST FOR THE V1 SLICE.

    Two MCPAdapter bots connect to the platform via SSE, join a 3-round
    Prisoner's Dilemma tournament, play 'alice always cooperates' vs
    'bob always defects', and verify final scores match the PD payoff
    matrix (alice=0, bob=15).
    """
    import asyncio

    from atp.adapters.mcp.adapter import MCPAdapter
    from atp.adapters.mcp.transport import SSETransport, SSETransportConfig
    from atp.dashboard.database import get_database
    from atp.dashboard.models import User
    from atp.dashboard.tournament.events import TournamentEventBus
    from atp.dashboard.tournament.models import Participant, Round, Tournament
    from atp.dashboard.tournament.service import TournamentService
    from sqlalchemy import select

    base_url, _ = e2e_server

    # 1. Admin creates the tournament directly.
    db = get_database()
    async with db.session() as session:
        admin = await session.get(User, seeded_users["admin"]["id"])
        svc = TournamentService(session, TournamentEventBus())
        t = await svc.create_tournament(
            admin=admin,
            name="e2e-pd-acceptance",
            game_type="prisoners_dilemma",
            num_players=2,
            total_rounds=3,
            round_deadline_s=30,
        )
        await session.commit()
        tournament_id = t.id

    # 2. Two MCPAdapter bots connect via SSE.
    sse_url = f"{base_url}/mcp/sse"
    alice_bot = MCPAdapter(
        transport=SSETransport(SSETransportConfig(
            url=sse_url,
            headers={"Authorization": f"Bearer {seeded_users['alice']['jwt']}"},
        ))
    )
    bob_bot = MCPAdapter(
        transport=SSETransport(SSETransportConfig(
            url=sse_url,
            headers={"Authorization": f"Bearer {seeded_users['bob']['jwt']}"},
        ))
    )

    alice_received: list[dict] = []
    bob_received: list[dict] = []

    # Replace the next two lines with the verified MCPAdapter
    # notification subscription pattern from phase0 findings.
    alice_bot.on_notification = lambda n: alice_received.append(n)  # PLACEHOLDER
    bob_bot.on_notification = lambda n: bob_received.append(n)      # PLACEHOLDER

    await alice_bot.connect()
    await bob_bot.connect()

    try:
        # 3. Both join the tournament.
        await alice_bot.call_tool("join_tournament", {
            "tournament_id": tournament_id,
            "agent_name": "alice-always-cooperate",
        })
        await bob_bot.call_tool("join_tournament", {
            "tournament_id": tournament_id,
            "agent_name": "bob-always-defect",
        })

        # 4. Wait for the first round_started notification on both sides.
        async def _wait_for_round_started(received: list[dict], target_round: int) -> None:
            deadline = asyncio.get_event_loop().time() + 5
            while asyncio.get_event_loop().time() < deadline:
                for n in received:
                    data = n.get("params", {}).get("data", {})
                    if (
                        data.get("event") == "round_started"
                        and data.get("round_number") == target_round
                    ):
                        return
                await asyncio.sleep(0.05)
            raise TimeoutError(
                f"never got round_started for round {target_round}; "
                f"received: {received}"
            )

        # 5. Play 3 rounds.
        for round_n in range(1, 4):
            await _wait_for_round_started(alice_received, round_n)
            await _wait_for_round_started(bob_received, round_n)

            await alice_bot.call_tool("make_move", {
                "tournament_id": tournament_id,
                "action": {"choice": "cooperate"},
            })
            await bob_bot.call_tool("make_move", {
                "tournament_id": tournament_id,
                "action": {"choice": "defect"},
            })

        # 6. Wait for tournament_completed.
        async def _wait_for_tournament_completed(received: list[dict]) -> dict:
            deadline = asyncio.get_event_loop().time() + 5
            while asyncio.get_event_loop().time() < deadline:
                for n in received:
                    data = n.get("params", {}).get("data", {})
                    if data.get("event") == "tournament_completed":
                        return data
                await asyncio.sleep(0.05)
            raise TimeoutError("never got tournament_completed notification")

        completed = await _wait_for_tournament_completed(alice_received)
        final_scores = completed["final_scores"]
        # Convert keys from str (JSON-serialized) to int for assertion.
        final_scores = {int(k): v for k, v in final_scores.items()}
        assert final_scores[seeded_users["alice"]["id"]] == 0.0
        assert final_scores[seeded_users["bob"]["id"]] == 15.0

    finally:
        await alice_bot.disconnect()
        await bob_bot.disconnect()

    # 7. Direct DB sanity check.
    db = get_database()
    async with db.session() as session:
        tournament = await session.get(Tournament, tournament_id)
        assert tournament.status == "completed"

        rounds = (
            await session.execute(
                select(Round)
                .where(Round.tournament_id == tournament_id)
                .order_by(Round.round_number)
            )
        ).scalars().all()
        assert len(rounds) == 3
        assert all(r.status == "completed" for r in rounds)

        parts = (
            await session.execute(
                select(Participant).where(Participant.tournament_id == tournament_id)
            )
        ).scalars().all()
        assert len(parts) == 2
        by_user = {p.user_id: p for p in parts}
        assert by_user[seeded_users["alice"]["id"]].total_score == 0.0
        assert by_user[seeded_users["bob"]["id"]].total_score == 15.0
```

- [ ] **Step 2: Run the acceptance test**

```bash
uv run python -m pytest tests/e2e/test_mcp_pd_tournament.py::test_two_bots_play_pd_tournament_end_to_end -v -s
```

Expected: PASS.

If it fails, the most likely culprits in order of likelihood:
1. **MCPAdapter API mismatch** — `on_notification` is a placeholder. Replace with the verified pattern from phase0 findings.
2. **`ctx.request_context.request` placeholder in `notifications.py`** — replace with the verified ctx attribute path.
3. **`ctx.session.send_notification` placeholder** — same.
4. **Module-level `tournament_event_bus` not shared between service and notifications** — verify both `TournamentService` instances use the same bus instance from `atp.dashboard.mcp.tournament_event_bus`.
5. **Database session for the in-test admin create vs the in-tool service create using different engines** — verify `get_database()` returns a singleton.

Iterate by reading the test output, fixing the placeholder, and re-running. Each fix should be one targeted edit.

- [ ] **Step 3: Commit when green**

```bash
git add tests/e2e/test_mcp_pd_tournament.py
git commit -m "test(e2e): two MCPAdapter bots play 3-round PD acceptance test"
```

---

## Phase 9 — Final regression sweep + slice acceptance

### Task 9.1: Run the full suite

- [ ] **Step 1: Run all dashboard unit tests**

```bash
uv run python -m pytest tests/unit/dashboard/ -q 2>&1 | tail -10
```

Expected: all PASS, no regressions in existing dashboard tests.

- [ ] **Step 2: Run the new tournament + mcp test directories**

```bash
uv run python -m pytest tests/unit/dashboard/tournament/ tests/unit/dashboard/mcp/ -v 2>&1 | tail -30
```

Expected: all PASS.

- [ ] **Step 3: Run the e2e test**

```bash
uv run python -m pytest tests/e2e/test_mcp_pd_tournament.py -v 2>&1 | tail -20
```

Expected: all PASS.

- [ ] **Step 4: Format and lint everything we touched**

```bash
uv run ruff format \
    packages/atp-dashboard/atp/dashboard/tournament/ \
    packages/atp-dashboard/atp/dashboard/mcp/ \
    packages/atp-dashboard/atp/dashboard/v2/factory.py \
    game-environments/game_envs/games/prisoners_dilemma.py \
    tests/unit/dashboard/tournament/ \
    tests/unit/dashboard/mcp/ \
    tests/e2e/

uv run ruff check \
    packages/atp-dashboard/atp/dashboard/tournament/ \
    packages/atp-dashboard/atp/dashboard/mcp/ \
    tests/unit/dashboard/tournament/ \
    tests/unit/dashboard/mcp/ \
    tests/e2e/

uv run pyrefly check \
    packages/atp-dashboard/atp/dashboard/tournament/ \
    packages/atp-dashboard/atp/dashboard/mcp/
```

Expected: no errors. If pyrefly catches type issues, fix them inline (the most likely fix area is the `ctx` Any-typed parameters — once you have FastMCP imported, replace `Any` with the real `Context` type from FastMCP).

- [ ] **Step 5: Commit any format/lint cleanup**

```bash
git add -A
git commit -m "chore(tournament): format and lint cleanup after vertical slice"
```

### Task 9.2: Acceptance gate — vertical slice DONE

- [ ] **Step 1: Manual acceptance check**

The slice is DONE when ALL of the following are true:

1. `tests/e2e/test_mcp_pd_tournament.py::test_two_bots_play_pd_tournament_end_to_end` passes locally on a fresh checkout of this branch.
2. All `tests/unit/dashboard/tournament/` tests pass (event bus, state, join, resolve).
3. All `tests/unit/dashboard/mcp/` tests pass (auth middleware, tools, notification format).
4. The full existing dashboard test suite passes (`tests/unit/dashboard/` minus the new dirs) — no regressions in benchmark, auth, RBAC, or rate-limiting tests.
5. `uv run ruff check` and `uv run pyrefly check` are clean on all changed files.
6. `docs/notes/phase0-fastmcp-findings.md` is committed and contains verified API patterns for both MCPAdapter notifications and FastMCP request access.

When all six are true, **the v1 vertical slice is complete**. Two LLM bots can play a 3-round PD tournament through real MCP SSE end-to-end. The infrastructure for Plan 2 (deadlines, full tool set, AD-9/AD-10, REST admin, dashboard UI) is unblocked.

- [ ] **Step 2: Push the slice branch**

```bash
git push origin HEAD
```

If working in a feature branch off main, also open a PR for review. If working directly on main (this project's typical pattern), the push goes straight to remote.

---

## Self-review checklist

**Spec coverage** — every requirement from the design spec that this plan claims to cover:

| Spec section | Plan task |
|---|---|
| AD-1: SSE-only transport | Phase 6 (FastMCP mount over SSE) |
| AD-2: MCP-first gameplay | Phases 6, 7 (no REST gameplay touched) |
| AD-3: N-player capable | Phase 5 (`num_players` everywhere, no hardcoded 2) |
| AD-4: In-process event bus | Phase 2 |
| AD-5: Synchronous round resolution | Task 5.5 + 5.6 |
| AD-6: Deadline worker | **NOT IN SLICE** — Plan 2 |
| AD-7: No reconnect replay | Slice ships without it (consistent with AD-7) |
| AD-8: 409 on double make_move | Task 5.5 (existing-action check) |
| AD-9: Token expiry hard cap | **NOT IN SLICE** — Plan 2 |
| AD-10: join_token + 1-active limit | **NOT IN SLICE** — Plan 2 |
| §Service layer API (full) | Phase 5 partial — only `create_tournament`, `join`, `get_state_for`, `submit_action`, private resolvers |
| §Event bus | Phase 2 |
| §MCP server | Phases 6, 7 |
| §Notification delivery | Task 7.3 |
| §Persistence (full) | Phase 1 partial — additive columns only, no constraints |
| §Error handling | Phase 3 (errors module) |
| §Testing strategy unit + e2e | Phases 2-9 (every task is TDD) |
| §Phase 0 verification | Phase 0 ✅ (both tasks PASSED, commits `87054c2` + `713d7f4`) |
| **(New) MCPAdapter SSETransport bugs discovered during Phase 0.1** | Phase Pre-8 — added post-verification |
| §Backlog | Plan 2 will reference items from spec backlog |

**Placeholders** — searched plan for `TODO`, `TBD`, `FIXME`, `XXX`. None present except deliberate `PLACEHOLDER` notes pointing engineers to the phase 0 findings doc, which is intentional and explicit.

**Type / signature consistency** — checked that:
- `TournamentService.__init__(session, bus)` signature is consistent across all task code blocks
- `TournamentEventBus.publish(event)` and `subscribe(tournament_id)` are consistent
- `RoundState.to_dict()` is the only serialization method (no `dict()` or `json()` variants slipped in)
- The `_session_tasks` dict shape `dict[session_id, dict[tournament_id, Task]]` is consistent in `notifications.py`
- `_join_tournament_impl`, `_get_current_state_impl`, `_make_move_impl` all take keyword-only `user` and `service`

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-10-mcp-tournament-vertical-slice.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
