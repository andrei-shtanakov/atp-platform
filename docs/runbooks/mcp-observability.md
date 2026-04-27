# Runbook: MCP Handshake Observability

## Context

Tournament 30 (2026-04-27) cancelled with 1/3 participants joined because two Claude-SDK bots failed the SSE handshake → tool-list → first-tool-call sequence. Diagnosing it required scraping ad-hoc `logger.info` lines because the MCP server had no structured trace through that lifecycle.

Task 4 of `docs/superpowers/plans/2026-04-27-mcp-server-reliability.md` adds four structured events at the four checkpoints we care about. This runbook is the single source of truth for what each event means and what fields it carries — keep it in sync with `packages/atp-dashboard/atp/dashboard/mcp/observability.py`, which holds the public schema constants.

## Event schema

All events log via the `atp.mcp.observability` logger at `INFO` level. Each record's `LogRecord.extra` carries the structured fields below; the message body is the event name (`record.message == record.event`).

### `mcp_handshake_started`

Fires at the top of `MCPAuthMiddleware.__call__` for every HTTP request hitting `/mcp/*`.

| Field | Type | Notes |
|---|---|---|
| `request_id` | str (16 hex) | Per-request correlation id; threaded into `request.state.mcp_request_id` for downstream tool wrappers. |
| `client_ip` | str \| null | First entry of `X-Forwarded-For`, falls back to `scope["client"][0]`. |
| `user_agent` | str \| null | `User-Agent` request header verbatim. |
| `path` | str | The ASGI scope `path` (e.g. `/mcp/sse/`). |

### `mcp_handshake_authorized`

Fires on auth-pass exit from `MCPAuthMiddleware`. Indicates the request will reach FastMCP.

| Field | Type | Notes |
|---|---|---|
| `request_id` | str | Same as the matching `started` event. |
| `user_id` | int | Authenticated user. |
| `agent_id` | int \| null | Agent-scoped token's owned `Agent.id` (always set for `atp_a_*` tokens that pass purpose gating). |
| `agent_purpose` | str | `"tournament"` (other purposes are rejected). |
| `duration_ms` | float | Wall time spent inside the middleware (auth lookup + cache check). Pin this for the cold-start race. |

### `mcp_handshake_rejected`

Fires on auth-fail exit. Three possible reasons:

| `reason` value | HTTP status | Trigger |
|---|---|---|
| `"unauthenticated"` | 401 | `request.state.user_id` not set (no/invalid token). |
| `"user_level_token"` | 403 | Token has no `agent_purpose` (e.g. user-level `atp_u_*` token or browser session). |
| `"benchmark_token"` | 403 | Token's `agent_purpose != "tournament"` (typically `"benchmark"`). |

Common fields:

| Field | Type | Notes |
|---|---|---|
| `request_id` | str | Matches the `started` event. |
| `reason` | str | One of the three values above. |
| `status` | int | `401` or `403` — same as the response. |
| `duration_ms` | float | Time to reject. |

### `mcp_first_tool_call`

Fires the first time a given FastMCP **session** invokes any tool. Subsequent calls within the same session do not re-emit. Session identity comes from `ctx.session_id` (FastMCP-assigned, stable across the SSE connection).

| Field | Type | Notes |
|---|---|---|
| `session_id` | str | FastMCP session id; stringified for log uniformity. |
| `request_id` | str | The HTTP request that carried the tool invocation (different from the SSE-handshake `request_id`). |
| `tool` | str | Tool name as registered with FastMCP (e.g. `"join_tournament"`, `"ping"`). |

The interval `mcp_handshake_authorized.duration_ms` → `mcp_first_tool_call` (timestamps) is the **cold-start race window** — the period between successful auth and first useful work. If a client's `mcp_first_tool_call` does not appear within ~30 s of `mcp_handshake_authorized`, either the SDK is still doing its tool-discovery dance or the client gave up.

## Operator queries

Logs are JSON-formatted; substitute your log-aggregator's syntax.

**Mean handshake duration over the last hour:**

```
event:mcp_handshake_authorized | timeshift -1h | aggregate avg(duration_ms)
```

**Reject rate by reason:**

```
event:mcp_handshake_rejected | aggregate count by reason
```

**Sessions that never made a tool call (truncated SDK init):**

```
SELECT session_id
  FROM logs
 WHERE event = 'mcp_handshake_authorized'
   AND request_id NOT IN (
     SELECT request_id FROM logs WHERE event = 'mcp_first_tool_call'
   )
```

## Known gaps

- **`mcp_tools_list_responded`** is documented in the plan but not yet emitted. FastMCP handles `tools/list` internally and exposes no public hook; an in-PR proof-of-concept would need either an ASGI response middleware that peeks JSON-RPC bodies or a fork of the FastMCP transport. Deferred to a follow-up.
- **Per-session correlation across MCP transports** (streamable HTTP vs SSE) is not yet normalized — `session_id` shape may differ once Task 5 lands.
- **Memory growth under attack** — `_first_tool_call_seen` is bounded at 4096 sessions / 1 h TTL. A flood of unique sessions could push older entries out, causing a re-emit of `mcp_first_tool_call` for the same session if it returns after eviction. Acceptable for diagnostics; tighten if it becomes ambiguous.

## Adding new events

1. Define the constant in `packages/atp-dashboard/atp/dashboard/mcp/observability.py` with a docstring listing every field and its type.
2. Emit via `emit_event(MY_EVENT, request_id=..., ...)`.
3. Add a row to this runbook describing the event.
4. Add a test in `tests/unit/dashboard/mcp/test_handshake_observability.py` that pins both presence and field types.
