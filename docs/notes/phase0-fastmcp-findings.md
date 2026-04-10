# Phase 0 Findings (2026-04-10)

## Task 0.1: MCPAdapter notification subscription

**Result:** PASS, with significant caveats. Notifications can be received, but
only via the transport layer and only under specific ordering constraints. Two
latent bugs in `SSETransport` were uncovered along the way and must be fixed
before the MCP tournament server's programmatic Python participant path is
usable against any real MCP SSE server.

### Environment

- Verified against `packages/atp-adapters/atp/adapters/mcp/{adapter,transport}.py`
  on branch `feat/mcp-tournament-vertical-slice` (at commit 8a22538 + uncommitted
  Phase 0 findings).
- FastMCP version: whatever `uv run --with fastmcp` resolved on 2026-04-10
  (PrefectHQ/fastmcp, pulled into the throwaway uv env, NOT added to
  `pyproject.toml`). Tested `mcp.run(transport="sse", ...)`.

### MCPAdapter notification API

There is **no public `MCPAdapter` method** for subscribing to server-pushed
notifications. `MCPAdapter.stream_events(request)` exists but is misleading —
it synthesises ATP events around a single tool call, it does NOT surface
pushed `notifications/message` or any other server-originated notification.

The only notification surface lives on the transport:

```python
# Canonical working pattern (from the scratch verification test).
# Assumes the adapter has already completed its initialize() handshake
# so the transport is connected and the MCP session is live.

transport = adapter._transport          # MCPTransport (SSETransport in our case)
iterator = transport.stream_events()    # _SSEEventIterator over _message_queue

async for msg in iterator:
    method = msg.get("method", "")
    if method.startswith("notifications/message"):
        handle(msg)                     # msg is a raw JSON-RPC dict
```

`transport.receive()` is an equivalent single-shot version.

Both `stream_events()` and `receive()` pull from the same underlying
`asyncio.Queue` (`SSETransport._message_queue`), which is populated by a
background `_reader_task` (`_read_sse_events`) started during `connect()`.

**Access is via a private attribute (`adapter._transport`)**. Either Phase 7
(`mcp/notifications.py`) must expose a small public subscription helper on
`MCPAdapter`, or we live with reaching into `_transport` — not ideal, but
workable for the vertical slice.

### FastMCP server-side notification API

The scratch test's `start_pushing` tool uses `ctx.info(...)`:

```python
from fastmcp import Context, FastMCP

mcp = FastMCP("phase0-notify-probe")

@mcp.tool
async def start_pushing(ctx: Context) -> dict:
    async def _pusher() -> None:
        for i in range(10):
            await ctx.info(f"push-{i}")      # sends notifications/message
            await asyncio.sleep(0.4)

    # Fire-and-forget so the tool response unblocks send_request() BEFORE
    # the first notification hits the queue (see caveat #2 below).
    asyncio.create_task(_pusher())
    return {"scheduled": 10}
```

Observed on-wire notification shape:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/message",
  "params": {"level": "info", "data": {"msg": "push-0", "extra": null}}
}
```

So the message string lives at `params.data.msg`, not `params.content` or
`params.message`. Anything we build in `mcp/notifications.py` that translates
these into tournament events must read `params.data`. `ctx.debug/warning/error`
behave identically with a different `level`.

There is also a lower-level `ctx.log(level, message, extra=None, logger_name=None)`
and under the hood the MCP Python SDK's `ctx.session.send_log_message(...)`.
For the tournament server we should prefer `ctx.info/error/...` for anything
human-readable and reserve custom notification methods (e.g.
`notifications/tournament/match_complete`) for structured domain events — which
will require the raw `ctx.session.send_notification(...)` path, not `ctx.info`.

### Critical caveats — two latent bugs in SSETransport

#### Bug #1: SSE endpoint discovery is broken against MCP-compliant servers

`SSETransport._read_sse_events` (transport.py:820) handles the `event: endpoint`
SSE frame like this:

```python
data = json.loads(event_data)
if event_type == "endpoint" and "uri" in data:
    self._session_id = data.get("sessionId")
```

FastMCP (and the reference MCP SDK) send the endpoint frame as:

```
event: endpoint
data: /messages/?session_id=e519ec7f97674620865b363add0d9da9
```

That `data` line is a **bare path string**, not JSON. `json.loads` raises
`JSONDecodeError`, the `except json.JSONDecodeError: pass` swallows it, and the
transport never learns that POSTs need to go to `/messages/?session_id=...`.
Every `send()` instead POSTs to the `/sse` GET URL and gets HTTP 405.

Verified empirically: without a patch, `MCPAdapter.initialize()` fails with
`AdapterConnectionError: POST request failed with status 405` on the very
first `initialize` request. **The stock SSETransport cannot currently talk
to any MCP-spec-compliant SSE server.** The only reason this hasn't shown up
in the existing test suite is that the existing SSE tests appear to stub or
not exercise this flow end-to-end.

The scratch test monkey-patches `_read_sse_events` with a reader that parses
the endpoint path directly and assigns `self._config.post_endpoint` + extracts
`session_id`. That patch is preserved inline in the test file for reference.

**Fix required before Phase 1:** update `_read_sse_events` in
`packages/atp-adapters/atp/adapters/mcp/transport.py` to:
1. Detect `event_type == "endpoint"` and treat `event_data` as a URL/path.
2. Resolve it against `self._config.url` into `self._config.post_endpoint`.
3. Parse `session_id` query param into `self._session_id`.
4. Skip the `json.loads` for endpoint frames.

This should land as a separate fix PR, not bundled into the tournament slice —
it affects every existing SSE user of MCPAdapter.

#### Bug #2: Shared queue between request/response and notification streams

`SSETransport` has a single `asyncio.Queue` (`_message_queue`) for **everything**
coming off the SSE connection — JSON-RPC responses, notifications, all of it.
Two consumers pull from it:

1. `send_request() -> _wait_for_response()` which loops
   `response = await self.receive()` and discards non-matching messages
   (transport.py:308–312). There is an explicit TODO in the code:
   `# Could buffer other messages here for out-of-order responses`.
2. `stream_events() -> _SSEEventIterator` which does `await self._queue.get()`.

Consequences:

- **If a consumer task is running `stream_events()` in parallel with a caller
  running `send_request()`, they race for every message off the queue.** During
  the scratch test, starting the consumer before calling `tools/call` caused
  the consumer to steal the tool-call response, and `send_request()` timed
  out after 10 s.
- **Notifications that arrive while a `send_request()` is mid-flight are
  silently dropped** by `_wait_for_response`'s "skip messages whose id !=
  my request_id" loop. So even without a concurrent consumer, we lose any
  notification the server happens to push during a pending request.

The scratch test works around both problems by (a) arranging for the FastMCP
tool to return immediately and push notifications from a background task
started _after_ the response is sent, and (b) only starting the consumer
after `send_request("tools/call", ...)` has returned. Those workarounds are
fine for a one-shot verification but **fundamentally incompatible with a
long-running tournament participant loop** that needs to intersperse
`tools/call` requests with continuous notification consumption.

### Implications for the rest of the plan

1. **Phase 7 (`mcp/notifications.py`) must NOT simply wrap
   `adapter._transport.stream_events()`**. It needs proper message demuxing:
   a single reader task that routes responses by `id` into per-request futures
   and notifications into a separate queue / callback dispatch. Options:
   - **Option A (fix SSETransport in place):** refactor `SSETransport` so
     `_reader_task` classifies each incoming dict:
     - has `id` matching an in-flight request → resolve that request's future
     - otherwise (no `id` or unknown `id`) → push onto a separate
       `_notification_queue`
     Then `send_request()` awaits its future and `stream_events()` drains
     only the notification queue. This is the right fix and roughly 80–120
     lines. It also subsumes Bug #2's TODO.
   - **Option B (pivot to upstream `mcp` Python SDK client):** the official
     `mcp` Python package already has a properly-demuxed
     `ClientSession` that supports notification handlers via
     `session.set_notification_callback(...)` / the `read_stream` / `write_stream`
     abstraction. Using it for the tournament Python participant path would
     sidestep both bugs entirely. Downside: introduces a second MCP client
     stack alongside `MCPAdapter`, and the rest of the ATP platform remains
     on the buggy in-house one.
   - **Option C (extend MCPAdapter as a subclass for tournaments only):**
     least invasive but perpetuates the bug.

   **Recommendation: Option A.** Fix `SSETransport` demuxing as a prerequisite
   to Phase 7, before writing `mcp/notifications.py`. Cost: ~half a day.
   Benefit: every existing MCPAdapter SSE user gets reliable push support,
   and `mcp/notifications.py` becomes a thin public wrapper.

2. **Bug #1 (endpoint discovery) must be fixed before any real SSE client
   code in this plan runs**, including in Task 6.x where we spin up the
   tournament server and self-test it with `MCPAdapter`. Either land a fix
   early in Phase 1 or carry the monkey-patch forward in the slice and open
   a tracking issue to fix properly.

3. **MCPAdapter currently has no public notification API.** Phase 7 should
   add one (e.g. `adapter.notifications() -> AsyncIterator[dict]` or
   `adapter.on_notification(method: str, callback)`) rather than having
   `mcp/notifications.py` poke at `adapter._transport`. This is a
   prerequisite gate for Phase 7, not a nice-to-have.

4. **FastMCP SSE transport is documented as "legacy"** by upstream. If we
   stay on it for Phase 6+, we should note this in the spec and consider
   streamable-HTTP as a follow-up. The `MCPAdapter` transport layer only
   understands SSE today, so a streamable-HTTP migration is non-trivial.

### Gate status

Task 0.1 gate: **PASS** — server-pushed notifications can be received by
`MCPAdapter` via the transport layer, so the core "programmatic Python
participant consumes pushed events" premise is not dead. **But** the two
bugs above mean the plan needs one extra remediation task in Phase 1 (fix
`SSETransport._read_sse_events` + demux the queue) before Phase 7 can land
cleanly. Recommend adding Task 1.X: "Fix SSETransport endpoint discovery
and response/notification demuxing" as a dependency of Phase 7.

Proceed to Task 0.2 (FastMCP + JWTUserStateMiddleware).
