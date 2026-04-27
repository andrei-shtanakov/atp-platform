# MCP Server Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/mcp/sse` cold-start reliable for clients that race tool registration against authentication (Claude SDK in particular), reduce DB pressure on every MCP request, give clients an explicit warmup signal, and improve observability so future regressions are visible. The motivating incident was tournament 30 (2026-04-27): three Claude-SDK-based bots opened parallel SSE handshakes, the first 1–2 lost tool registration, the model aborted on the protocol-defined "MCP tools unavailable" path, and the tournament cancelled with 1/3 participants joined.

**Architecture:** Eight independent improvements, each shippable as its own PR. Tasks 1–3 are user-visible reliability wins (high payoff, low risk); Tasks 4–7 are observability and platform hygiene; Task 8 is documentation. Tasks are ordered by impact-per-effort, not by dependency — most can be parallelised across contributors.

**Tech Stack:** Python 3.12, FastAPI 0.128, FastMCP 3.2.3, sse-starlette, SQLAlchemy async (`asyncpg` on prod, `aiosqlite` in tests), `cachetools` (already a dep), Alembic, pytest with anyio. `uv` for package management; `uv run ruff format / ruff check / pyrefly check / pytest` as the standard quality gate.

**Source of truth for current state (as of 2026-04-27, commit `1e9d285`):**
- FastMCP version: `3.2.3` (verify via `uv tree fastmcp`).
- MCP mount path: `/mcp` → `app.mount` in `packages/atp-dashboard/atp/dashboard/v2/factory.py:205` (transport=`sse`, line 118).
- Auth chain: `JWTUserStateMiddleware` (`packages/atp-dashboard/atp/dashboard/v2/rate_limit.py:102-153`) → `MCPAuthMiddleware` (`packages/atp-dashboard/atp/dashboard/mcp/auth.py:25-92`) → FastMCP.
- Token resolution: `JWTUserStateMiddleware._resolve_api_token` (`rate_limit.py:192-270`) — DB SELECT + UPDATE on every request, no in-process cache for the SELECT path.
- DB pool: `pool_size=5, max_overflow=10` (`packages/atp-dashboard/atp/dashboard/database.py:70-71`).
- Existing in-process cache pattern: `_legacy_purpose_cache` (`rate_limit.py:241`).
- MCP tools registered via `@mcp_server.tool()` decorators in `packages/atp-dashboard/atp/dashboard/mcp/tools.py` (no `ping` tool exists today).
- `join_tournament` is **not** idempotent: re-issuing for an already-joined `(tournament_id, agent_name)` returns a server error.
- No structured metrics on MCP handshake; only ad-hoc `logger.info` lines in `MCPAuthMiddleware`.
- `claude_el_farol_3bots.py` (driver, lives in `../agents-for-game/`) ships with its own retry-on-failed-join workaround (`JOIN_RETRY_ATTEMPTS = 3`); this plan reduces the need for that workaround on the server side.

**Out of scope:**
- Replacing FastMCP entirely with another MCP server framework.
- Reworking JWT issuance / token format.
- Migrating off SSE — Task 5 *adds* streamable HTTP alongside SSE, does not remove SSE.
- Changes to participant-kit-en bot code (it uses raw `mcp` lib, is not affected by the cold-start race).

---

## Task 0: Bootstrap shared state for the work

**Files:** none (infra).

- [ ] **Step 0.1: Confirm starting commit**

Run:
```bash
cd <repo-root>
git fetch origin main
git log -1 --format='%h %s' origin/main
```

Expected: top commit on main is `1e9d285 Wire El Farol tournament telemetry end-to-end (#98)` or later.

- [ ] **Step 0.2: Verify baseline tests pass**

Run: `uv sync --group dev && uv run pytest tests/unit/dashboard/ -q --no-cov` from repo root.

Expected: all tests pass. If anything fails, fix it before starting Task 1.

- [ ] **Step 0.3: Sanity-check FastMCP version**

Run: `uv tree fastmcp 2>/dev/null | head -3`.

Expected: `fastmcp v3.2.3`. If a newer version is available, note it for Task 6 — but do not bump yet.

---

## Task 1: Cache API token resolution (high priority, low risk)

**PR scope:** One merged PR titled `feat(auth): in-process LRU cache for API-token resolution`. Eliminates the per-request DB SELECT in `JWTUserStateMiddleware._resolve_api_token` for the hot path: same token used repeatedly within 30 seconds (e.g. SSE keepalives, polling clients, multi-handshake bursts).

**Files in this task:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/rate_limit.py` (`_resolve_api_token`, add `_token_auth_cache`)
- Test: `tests/unit/dashboard/test_token_resolution_cache.py` (new)

- [ ] **Step 1.1: Branch off main**

Run:
```bash
git fetch origin main
git checkout -b feat/mcp-cache-token-resolution origin/main
uv sync --group dev
```

- [ ] **Step 1.2: Add the cache**

Add at module level in `rate_limit.py`, near `_legacy_purpose_cache`:

```python
from cachetools import TTLCache

_TOKEN_AUTH_CACHE_TTL_S = 30.0
_TOKEN_AUTH_CACHE_MAX = 1024
# Keyed by token_hash (sha256 hex). Value = (user_id, agent_id, agent_purpose).
# TTL is short enough that token revocation propagates within seconds without
# explicit invalidation. Bounded size protects against memory growth under
# token churn.
_token_auth_cache: TTLCache[str, tuple[int, int | None, str | None]] = TTLCache(
    maxsize=_TOKEN_AUTH_CACHE_MAX, ttl=_TOKEN_AUTH_CACHE_TTL_S
)
```

In `_resolve_api_token`, before the DB block:

```python
cached = _token_auth_cache.get(token_hash)
if cached is not None:
    state["user_id"], state["agent_id"], state["agent_purpose"] = cached
    state["token_type"] = "api"
    return
```

After successful DB resolution, populate the cache with the resolved triple.

- [ ] **Step 1.3: Keep `last_used_at` UPDATE on the miss path**

The 60-second debounce on `last_used_at` is already in the SELECT/UPDATE block. With a 30 s cache TTL, the cache evicts before the debounce window closes, so on the next miss the UPDATE either fires (debounce expired) or no-ops (debounce window). No behaviour change is needed — confirm by inspection.

- [ ] **Step 1.4: Add unit tests**

Create `tests/unit/dashboard/test_token_resolution_cache.py`:

- `test_cache_hit_skips_db` — patch `db.session()` to raise; verify `_resolve_api_token` populates state correctly when token is in cache.
- `test_cache_miss_falls_through_to_db` — clear cache; verify normal flow.
- `test_cache_expiry_refetches_from_db` — populate cache with a stale entry (`monkeypatch` `cachetools` clock or wait); confirm DB hit on second call.
- `test_cache_population_after_db_resolution` — call once with empty cache; confirm cache is populated; call again; confirm DB not hit.
- `test_revoked_token_not_returned_from_cache` (acceptable lag) — note in docstring: revocation has up to 30 s propagation lag; this is intentional pre-1.0.

- [ ] **Step 1.5: Run quality gates**

```bash
uv run ruff format packages/atp-dashboard/atp/dashboard/v2/rate_limit.py tests/unit/dashboard/test_token_resolution_cache.py
uv run ruff check .
uv run pyrefly check
uv run pytest tests/unit/dashboard/test_token_resolution_cache.py tests/unit/dashboard/auth/ -v --no-cov
```

Expected: 0 errors, all tests pass.

- [ ] **Step 1.6: Smoke-test the existing auth path end-to-end**

```bash
uv run pytest tests/integration/dashboard/test_tsa_auth_gating.py tests/unit/dashboard/mcp/test_auth_middleware.py -v --no-cov
```

Expected: pass. These cover the JWT/MCPAuth chain that wraps the cache.

- [ ] **Step 1.7: Commit and open PR**

```bash
git add -A
git commit -m "feat(auth): in-process LRU cache for API-token resolution"
git push -u origin HEAD
gh pr create --base main --title "feat(auth): in-process LRU cache for API-token resolution" --body "..."
```

PR body must include: motivation (MCP cold-start race, tournament 30 incident on 2026-04-27), 30 s revocation lag note, manual verification checklist.

---

## Task 2: Make `join_tournament` idempotent (high priority, low risk)

**PR scope:** `feat(tournament): make join_tournament idempotent`. Re-issuing `join_tournament` for an `(tournament_id, agent_name)` pair that is already joined by the same owner returns the existing participant row with a `status: "already_joined"` field instead of erroring. Reduces blast radius of client-side retry loops.

**Files in this task:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` (the `join` method — find via `grep -n 'def join' packages/atp-dashboard/atp/dashboard/tournament/service.py`)
- Modify: `packages/atp-dashboard/atp/dashboard/mcp/tools.py` (the `join_tournament` shim — confirm response shape)
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py` (REST `POST /api/v1/tournaments/{id}/join`)
- Test: `tests/unit/dashboard/tournament/test_service_join.py` (extend; add idempotency case)
- Test: `tests/unit/dashboard/tournament/test_service_join_idempotent.py` (new — or extend existing)
- Test: `tests/unit/dashboard/mcp/test_tools.py` (shim assertion)

- [ ] **Step 2.1: Branch and baseline.** Same as Step 1.1 but `feat/mcp-idempotent-join`.
- [ ] **Step 2.2: Define idempotency contract.** Same `(tournament_id, agent_name, owner_user_id)` returns existing `participant_id` with `status: "already_joined"`. Different `owner_user_id` for same `agent_name` still errors with `409 conflict` (security boundary — agent names are not globally unique).
- [ ] **Step 2.3: Implement in `service.py`.** Wrap the unique-constraint catch (or `select-then-insert` race) into the idempotent return path. Preserve semantics for the not-yet-joined case.
- [ ] **Step 2.4: Update MCP shim and REST route** to surface `already_joined` flag without breaking existing `participant_id` consumers.
- [ ] **Step 2.5: Add tests** covering: fresh join, double-join same owner, double-join different owner (must reject), join into completed/cancelled tournament (must still reject).
- [ ] **Step 2.6: Quality gates** (`ruff`, `pyrefly`, targeted `pytest`).
- [ ] **Step 2.7: Commit, push, PR.**

---

## Task 3: Add `mcp__atp-tournaments__ping` tool (high priority, low risk)

**PR scope:** `feat(mcp): add ping tool for client warmup gating`. Lets clients (Claude SDK, raw `mcp` lib, or anything else) call a trivial DB-free tool first to confirm tool registration succeeded before issuing real ops. Cold-start clients can `ping` then proceed; if `ping` is missing from the tool list, MCP transport is broken and the client knows to retry the handshake instead of attempting real operations.

**Files in this task:**
- Modify: `packages/atp-dashboard/atp/dashboard/mcp/tools.py` (add `@mcp_server.tool() async def ping(...)`)
- Test: `tests/unit/dashboard/mcp/test_tools.py` (assert ping returns the expected shape)
- Test: `tests/integration/dashboard/test_tsa_runner_builtins.py` or similar (verify ping is reachable via the mounted server)
- Update: `participant-kit-el-farol-en/README.md` — *optional* mention of ping as a connectivity check (low priority — raw lib doesn't need it).

- [ ] **Step 3.1: Branch and baseline.** `feat/mcp-ping-tool`.
- [ ] **Step 3.2: Define payload.** Return `{"ok": True, "server_version": <package version>, "ts": <iso8601>}`. No DB access. No auth-state inspection beyond what `MCPAuthMiddleware` already enforced.
- [ ] **Step 3.3: Implement.** Mirror the existing tool decorator style; document that this is a warmup endpoint, not a health-check (auth is still required).
- [ ] **Step 3.4: Add tests** for shape, that it works without any DB session, that it respects the same auth gating as other tools (no leak to unauthenticated callers).
- [ ] **Step 3.5: Quality gates + PR.**

---

## Task 4: MCP handshake metrics (medium priority, medium effort)

**PR scope:** `feat(mcp): structured handshake observability`. Emit structured log events for the SSE handshake lifecycle so future cold-start incidents can be diagnosed from logs alone. No metrics backend dependency — start with structured `logger.info` (already using `structlog`).

**Files in this task:**
- Modify: `packages/atp-dashboard/atp/dashboard/mcp/auth.py` — emit `mcp_handshake_started` and `mcp_handshake_authorized` with `request_id`, `token_id`, `client_ip`, `user_agent`.
- Modify: `packages/atp-dashboard/atp/dashboard/mcp/__init__.py` or `tools.py` — hook tool dispatch to emit `mcp_first_tool_call` per session.
- Add: a small helper module that assigns a `request_id` (uuid4) per MCP scope and threads it through.
- Test: `tests/unit/dashboard/mcp/test_handshake_observability.py` (new) — capture structured log events.

- [ ] **Step 4.1: Branch + baseline.** `feat/mcp-handshake-metrics`.
- [ ] **Step 4.2: Plumb `request_id`** into scope state via the middleware chain.
- [ ] **Step 4.3: Add events at the four checkpoints** (started → auth'd → tools-list-served → first-tool-call). Use field names that match an eventual Prometheus/OpenTelemetry mapping.
- [ ] **Step 4.4: Add tests** (capture events with `caplog`; assert ordering and required fields).
- [ ] **Step 4.5: Document the event schema** in `docs/runbooks/mcp-observability.md` (new) so on-call knows what to grep for.
- [ ] **Step 4.6: Quality gates + PR.**

---

## Task 5: Streamable HTTP transport alongside SSE (medium priority, medium effort)

**PR scope:** `feat(mcp): mount streamable HTTP transport at /mcp/http`. Adds the newer FastMCP transport in parallel with SSE; new clients can opt into it. Streamable HTTP is less prone to handshake-races (no long-lived unidirectional channel). SSE remains the default; participant-kit-en is unchanged.

**Files in this task:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/factory.py` — second `mcp_app = mcp_server.http_app(transport="streamable-http")` mount at `/mcp/http`.
- Modify: rate-limit bypass logic in `factory.py:176-185` — extend `/mcp/*` skip to cover `/mcp/http/*` too.
- Test: `tests/integration/dashboard/test_mcp_streamable_http.py` (new) — handshake + tool list + tool call.
- Update: `docs/reference/api-reference.md` — mention both transports.

- [ ] **Step 5.1: Branch.** `feat/mcp-streamable-http`.
- [ ] **Step 5.2: Investigate FastMCP API** for parallel transport mounts; confirm `mcp_server.http_app(transport="streamable-http")` returns a separate ASGI app that can co-mount.
- [ ] **Step 5.3: Add the mount** + extend lifespan composition (`_combined_lifespan` already drives one mcp_app's lifespan; verify second mount needs the same).
- [ ] **Step 5.4: Confirm auth chain** still applies (`MCPAuthMiddleware` should wrap the new transport too).
- [ ] **Step 5.5: Integration test** that exercises the new endpoint with a real MCP client (raw `mcp` lib, streamable transport).
- [ ] **Step 5.6: Quality gates + PR.**

---

## Task 6: Check FastMCP version for upstream fixes (low priority, low effort)

**PR scope:** `chore(deps): bump fastmcp` *if* changelog shows handshake/tool-registration fixes since 3.2.3. Bumping must be conservative — FastMCP 3.x has had minor breaking changes between minors. Read changelog first, decide later.

- [ ] **Step 6.1:** `uv tree fastmcp` → confirm current version locked.
- [ ] **Step 6.2:** Check FastMCP repo / changelog for releases since 3.2.3 (https://github.com/jlowin/fastmcp/releases).
- [ ] **Step 6.3:** If a relevant fix exists: `uv add fastmcp@^X.Y.Z`, run full test suite, smoke-test MCP handshake locally.
- [ ] **Step 6.4:** If no relevant fix: close out this task with a comment noting which version was checked and on what date — this avoids re-investigation on the next round.

---

## Task 7: SQLAlchemy pool sizing review (low priority, low effort)

**PR scope:** Either no change (document that 5+10 is intentional and adequate) or `chore(db): bump pool sizing`. Decision driven by data, not by feel.

- [ ] **Step 7.1: Measure current pool utilization on prod** for one week. Add an asyncpg/SQLAlchemy event listener (or run-time sampler) that logs `pool.checkedin()` / `pool.checkedout()` / `pool.overflow()` once per minute. Capture peak.
- [ ] **Step 7.2: Decide.**
  - If peak overflow > 5 frequently → bump `pool_size` to 10, leave overflow at 10.
  - If peak overflow rare → no change; document the headroom in `database.py` comment.
- [ ] **Step 7.3:** Whichever path, commit a comment in `database.py:70-71` so the choice is auditable.

---

## Task 8: Documentation for Claude-SDK bot developers (low priority, low effort)

**PR scope:** `docs(mcp): cold-start guidance for Claude-SDK bots`. Captures the lessons from tournament 30 (2026-04-27) so the next developer who builds a multi-bot Claude-SDK driver doesn't rediscover the race from first principles. Lives next to existing dev docs.

**Files in this task:**
- New: `docs/guides/claude-sdk-mcp-bots.md`
- Update: `docs/reference/api-reference.md` (cross-link to the new guide).

Content checklist:
- [ ] Problem statement: ToolSearch retries vs FastMCP cold-start tool registration.
- [ ] Recommended pattern: open SDK → REST verify the join landed → retry on miss with fresh client.
- [ ] Reference implementation: link to `claude_el_farol_3bots.py` once it lives in this repo or a public sibling repo.
- [ ] Why ping tool (Task 3) helps: SDK can issue a ping before the real operation.
- [ ] When the server-side fixes (Tasks 1, 4) might let you skip the workaround.
- [ ] Quality gates + PR.

---

## Verification across all tasks

After Tasks 1–3 land:
- [ ] **End-to-end smoke:** run `claude_el_farol_3bots.py` against prod with `JOIN_RETRY_ATTEMPTS=1` (the workaround disabled). All three bots should join on the first attempt. If they do — Tasks 1+3 sufficiently mitigate the race for this driver. If they don't — Task 5 (streamable HTTP) becomes higher priority.

After Task 4 lands:
- [ ] **Log review:** trigger a 5-bot parallel handshake; confirm structured events surface every checkpoint with correlated `request_id`.

---

## Sequencing recommendation

| Task | Priority | Effort | Dependencies | Suggested PR order |
|---|---|---|---|---|
| 1 — Token cache | High | ~1 day | None | 1st |
| 2 — Idempotent join | High | ~1 day | None | 2nd |
| 3 — Ping tool | High | ~½ day | None | 3rd |
| 4 — Metrics | Medium | ~2 days | None (independent) | 4th |
| 5 — Streamable HTTP | Medium | ~3 days | None (independent) | 5th — only if 1–4 are insufficient |
| 6 — FastMCP bump | Low | ~½ day | None | After 4 (so we have metrics to confirm no regression) |
| 7 — Pool sizing | Low | ~1 day measure + ½ day decide | None | After 4 (so we have metrics) |
| 8 — Docs | Low | ~½ day | After 1, 3 (so the recommended pattern reflects what shipped) | Last |

Total: ~10 person-days across 8 PRs. Tasks 1–3 can ship the same week; Tasks 4–7 can be parallelised across contributors.
