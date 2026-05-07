# Migration: `POST /api/agents` is gone — use `POST /api/v1/agents`

**Affected versions:** before PR #53 (LABS-54 Phase 2, merged 2026-04) → after
**Affected components:** clients calling the legacy ownerless agent-creation
endpoint (`POST /api/agents`).
**PR / commit:** #53 (`e1ab0436`)

## What changed

The legacy `POST /api/agents` endpoint that worked at v1.0.0 has been
retired. It now returns `410 Gone` and emits the deprecation header trio:

- `Deprecation: true`
- `Sunset: Fri, 17 Apr 2026 12:00:00 GMT`
- `Link: </api/v1/agents>; rel="successor-version"`

The replacement is `POST /api/v1/agents`. The replacement resolves
ownership from the caller's JWT (no anonymous agents) and enforces
per-user, per-purpose quotas.

## Why

The ownerless agent-creation path was the foundation for the agent-quota
and per-purpose-token security work (LABS-54, LABS-TSA). Keeping it
around as a 201-returning shadow path would have undermined those
controls — every quota and ownership check would need a "but not via the
legacy endpoint" carve-out. Returning 410 + a successor URL lets stale
clients fail loudly and points them at the correct route.

## Before

```http
POST /api/agents HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "my-agent",
  "agent_type": "http"
}
```

```http
HTTP/1.1 201 Created
Content-Type: application/json

{"id": "...", "name": "my-agent", "agent_type": "http", ...}
```

## After

```http
POST /api/v1/agents HTTP/1.1
Host: api.example.com
Authorization: Bearer <user-or-admin-token>
Content-Type: application/json

{
  "name": "my-agent",
  "purpose": "tournament"
}
```

```http
HTTP/1.1 201 Created
Content-Type: application/json

{
  "agent_id": "...",
  "name": "my-agent",
  "purpose": "tournament",
  "owner_id": "user_12345",
  ...
}
```

A request to the old URL responds:

```http
HTTP/1.1 410 Gone
Deprecation: true
Sunset: Fri, 17 Apr 2026 12:00:00 GMT
Link: </api/v1/agents>; rel="successor-version"
Content-Type: application/json

{
  "detail": "POST /api/agents is deprecated. Use POST /api/v1/agents (resolves owner from your JWT)."
}
```

## How to migrate

1. Change the URL from `POST /api/agents` to `POST /api/v1/agents`.
2. Make sure the caller's bearer token resolves to a real user (user-level
   `atp_u_*` tokens or admin JWTs work; legacy unowned API keys do not).
3. Update the request body: replace `agent_type` with `purpose` (e.g.,
   `"tournament"` or other valid purpose values). The `name` field remains
   unchanged.
4. The response now includes `owner_id` derived from the JWT and `agent_id`
   as the primary identifier.
5. If you hit `429 Too Many Requests` after migrating, you've hit the
   per-user / per-purpose quota — see `ATP_MAX_AGENTS_PER_USER` in CLAUDE.md
   for configuration.

## Backward compatibility

None. The 410 response is permanent. No grace period, no flag to
re-enable the old endpoint. Clients must update.

## Timeline

- **Sunset date:** 2026-04-17 12:00 GMT (retirement target; actual removal was 2026-04-18)
- **Deprecation status:** The Deprecation header was set to `true` at the time
  of the v410 response.
- **Successor:** The replacement endpoint is active immediately at the same API
  version and host. No parallel operation period.

## References

- Endpoint definition (legacy, now 410):
  `packages/atp-dashboard/atp/dashboard/v2/routes/agents.py`.
- Endpoint definition (new):
  `packages/atp-dashboard/atp/dashboard/v2/routes/agent_management_api.py`.
- Commit: `e1ab0436` (PR #53, LABS-54 Phase 2, 2026-04-18).
- Follow-up fix: `ef0f3656` (corrected HTTP-date weekday in Sunset header,
  added OpenAPI documentation for deprecation).
