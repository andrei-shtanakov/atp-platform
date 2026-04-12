# Token Self-Service & Agent Ownership Design

**Date:** 2026-04-12
**Status:** Draft
**Scope:** Scope #2 — self-service token management, agent ownership model, invite-based registration

## Problem

The ATP platform currently requires an admin to manually create bot users and issue tokens via CLI (`atp admin create-bot-user`, `atp admin issue-token`). This doesn't scale beyond a single operator. Users cannot:

- Create and manage their own agents
- Issue API tokens for autonomous agent operation
- View their tournament/test history per agent
- Register without admin intervention

## Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Agent model | Agent = config record owned by User (not a separate User) | Clean separation: User authenticates, Agent is tested/evaluated |
| Auth tokens | User-level + agent-scoped tokens | User-level for UI/CLI, agent-scoped limits blast radius per bot |
| Versioning | String label (like docker tag), separate field | Simple, no extra tables, filterable |
| Token scope | Full access, no granularity (`["*"]`) | YAGNI; `scopes` column exists for future use |
| Limits | 10 agents/user, 3 tokens/agent, 5 user-tokens, 30-day default | Reasonable defaults, all env-var configurable |
| Registration | Invite-only now (`ATP_REGISTRATION_MODE=invite`), open later | Pre-release platform, controlled access |

## Data Models

### Changes to `Agent`

```python
class Agent(Base):
    # existing: id, tenant_id, name, agent_type, config, description, created_at, updated_at

    # new fields:
    owner_id: int | None  # FK -> users.id, nullable for backwards compat
    version: str = "latest"  # string label, like docker tag

    # new constraint (replaces existing unique(tenant_id, name)):
    # unique(tenant_id, owner_id, name, version)
```

### New table: `APIToken`

```python
class APIToken(Base):
    __tablename__ = "api_tokens"

    id: int  # PK autoincrement
    tenant_id: str = "default"
    user_id: int  # FK -> users.id, token owner
    agent_id: int | None  # FK -> agents.id, None = user-level token

    name: str  # human-readable ("my-bot-prod", "ci-runner")
    token_prefix: str  # first 8 chars of token for UI display ("atp_u_3f")
    token_hash: str  # sha256(full_token), indexed, unique

    scopes: list[str] = ["*"]  # JSON, unused for now
    expires_at: datetime | None  # None = never expires
    last_used_at: datetime | None
    revoked_at: datetime | None  # not None = revoked

    created_at: datetime
```

**Token format:**
- User-level: `atp_u_<32 hex chars>` (total 38 chars)
- Agent-scoped: `atp_a_<32 hex chars>` (total 38 chars)
- Stored as `sha256(token)` — token shown once at creation, never retrievable

### New table: `Invite`

```python
class Invite(Base):
    __tablename__ = "invites"

    id: int  # PK
    code: str  # unique, "atp_inv_<16 hex chars>"
    created_by_id: int  # FK -> users.id (admin who created)

    used_by_id: int | None  # FK -> users.id, who redeemed
    used_at: datetime | None
    expires_at: datetime | None
    max_uses: int = 1  # 1 = single-use
    use_count: int = 0

    created_at: datetime
```

### Changes to `Participant`

```python
class Participant(Base):
    # existing: tournament_id, user_id, agent_name, joined_at, total_score, released_at

    # new field:
    agent_id: int | None  # FK -> agents.id, nullable for old records
```

## API Endpoints

### Agent Management (`/api/v1/agents`)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/api/v1/agents` | Create agent | user token |
| `GET` | `/api/v1/agents` | List my agents | user token |
| `GET` | `/api/v1/agents/{id}` | Agent details | owner or admin |
| `PATCH` | `/api/v1/agents/{id}` | Update agent | owner or admin |
| `DELETE` | `/api/v1/agents/{id}` | Delete agent + revoke all its tokens | owner or admin |

**Create agent request:**
```json
{
  "name": "tit-for-tat",
  "version": "v1",
  "agent_type": "mcp",
  "config": {"endpoint": "http://..."},
  "description": "Classic TFT strategy"
}
```

Validations: limit 10 agents per user, unique(owner_id, name, version). Delete is blocked if agent is participating in an active (pending/in_progress) tournament — returns 409 "Agent is in an active tournament".

### Token Management (`/api/v1/tokens`)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/api/v1/tokens` | Create token | user token / session |
| `GET` | `/api/v1/tokens` | List my tokens (no secrets) | user token / session |
| `DELETE` | `/api/v1/tokens/{id}` | Revoke token | owner or admin |

**Create user-level token:**
```json
POST /api/v1/tokens
{
  "name": "ci-runner",
  "expires_in_days": 90
}
// -> 201 {id, name, token_prefix, token: "atp_u_3f8a...", expires_at}
//   token shown ONCE
```

**Create agent-scoped token:**
```json
POST /api/v1/tokens
{
  "name": "tft-prod",
  "agent_id": 42,
  "expires_in_days": null
}
// -> 201 {id, name, token_prefix, token: "atp_a_9c1b...", expires_at: null}
```

Validations: limit 5 user-tokens, 3 agent-tokens per agent, agent must belong to user.

**List tokens response:**
```json
[
  {"id": 1, "name": "ci-runner", "token_prefix": "atp_u_3f", "agent_id": null,
   "created_at": "...", "expires_at": "...", "last_used_at": "...", "revoked_at": null},
  {"id": 2, "name": "tft-prod", "token_prefix": "atp_a_9c", "agent_id": 42, ...}
]
```

### Invite Management (`/api/v1/invites`)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/api/v1/invites` | Create invite | admin only |
| `GET` | `/api/v1/invites` | List invites | admin only |
| `DELETE` | `/api/v1/invites/{id}` | Deactivate invite | admin only |

**Registration with invite:**
```json
POST /api/auth/register
{
  "username": "alice",
  "email": "alice@example.com",
  "password": "...",
  "invite_code": "atp_inv_a1b2c3..."
}
```

When `ATP_REGISTRATION_MODE=invite`: invite_code is required. When `=open`: ignored.

## Auth Middleware

### Token resolution order

```
1. Extract token from Authorization header or atp_token cookie
2. If starts with "atp_u_" or "atp_a_":
   -> sha256(token) -> SELECT FROM api_tokens WHERE token_hash = ? AND revoked_at IS NULL
   -> check expires_at
   -> UPDATE last_used_at (debounced, max once per 60 sec)
   -> request.state.user_id = token.user_id
   -> request.state.agent_id = token.agent_id  (None for user-level)
   -> request.state.token_type = "api"
3. Else: existing JWT decode (session tokens)
   -> request.state.token_type = "session"
```

### Agent-scoped tokens in tournaments

When a bot with agent-scoped token calls `join_tournament`:
- `agent_id` comes from `request.state.agent_id` (from token)
- `agent_name` resolved from `Agent.name` (lookup by agent_id)
- Bot **cannot** specify a different agent_id — token is bound
- User-level token: bot passes `agent_id` explicitly, ownership verified

### GitHub OAuth with invite-only mode

```
1. /ui/login -> "Sign in with GitHub"
2. GitHub callback -> post_auth.py -> user doesn't exist yet
3. invite mode: redirect to /ui/register?github=1&username=...&email=...
   (pre-fill from GitHub profile, invite_code required)
4. open mode: create user automatically (current behavior)
```

## Dashboard UI Pages

All pages use HTMX + Pico CSS (existing stack).

### `/ui/agents` — My Agents

Table: Name, Version, Type, Active Tokens count, Created, Actions (Edit/Delete).
Button: "New Agent" -> inline form (name, version, type, config JSON, description).

### `/ui/agents/{id}` — Agent Detail

Sections:
- **Info**: name, version, type, description, config (readonly JSON)
- **Tokens**: list of this agent's tokens + "Create Token" button + "Revoke" per token
- **Tournament History**: table of tournaments this agent participated in (from Participant)

Token creation shows modal with warning "Token is shown only once" + copy button.

### `/ui/tokens` — My Tokens

Combined table of all user's tokens (user-level and agent-scoped):
Name, Type (user/agent), Agent name, Prefix, Expires, Last Used, Status, Actions (Revoke).

Button: "New Token" -> form (name, type selector, agent dropdown if agent type, expiry: 7/30/90/never).

### `/ui/invites` — Invite Management (admin only)

Table: Code (truncated), Created By, Used By, Status, Created, Actions (Deactivate).
Button: "Generate Invite" -> shows code + copy button.

### Changes to existing pages

- **Sidebar**: add "My Agents", "My Tokens". "Invites" for admins only.
- **`/ui/login`**: when `ATP_REGISTRATION_MODE=invite`, registration form shows invite code field.
- **Tournament detail**: participant agent names link to `/ui/agents/{id}` when agent_id is set.

## Error Handling

| Situation | HTTP | Message |
|-----------|------|---------|
| Token revoked | 401 | "Token has been revoked" |
| Token expired | 401 | "Token has expired" |
| Agent limit reached (10) | 409 | "Agent limit reached (max 10)" |
| Token limit reached | 409 | "Token limit reached for this agent (max 3)" |
| Duplicate agent (name+version) | 409 | "Agent 'tit-for-tat' version 'v1' already exists" |
| Invalid invite code | 400 | "Invalid or expired invite code" |
| Agent-scoped token wrong agent | 403 | "Token is scoped to agent {id}" |
| Not owner | 403 | "You don't own this agent" |
| Delete agent in active tournament | 409 | "Agent is in an active tournament" |
| Registration needs invite | 400 | "Invite code required" |

## Database Migration

New tables (`APIToken`, `Invite`): created via `create_all()` + Alembic migration.

Changes to existing tables:
- `Agent`: add `owner_id`, `version` via `_add_missing_columns()` + Alembic
- `Participant`: add `agent_id` via `_add_missing_columns()` + Alembic
- Unique constraint `(tenant_id, owner_id, name, version)` on Agent: **Alembic only** (`_add_missing_columns` doesn't handle constraints)

Backwards compatibility: existing agents get `owner_id = NULL`, existing participants get `agent_id = NULL`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ATP_REGISTRATION_MODE` | `invite` | `invite` or `open` |
| `ATP_MAX_AGENTS_PER_USER` | `10` | Max agents per user |
| `ATP_MAX_TOKENS_PER_AGENT` | `3` | Max active tokens per agent |
| `ATP_MAX_USER_TOKENS` | `5` | Max user-level tokens |
| `ATP_DEFAULT_TOKEN_DAYS` | `30` | Default token expiry |
| `ATP_MAX_TOKEN_DAYS` | `365` | Max allowed expiry (0 = allow "never") |

## Out of Scope

- Granular token scopes (column exists, logic deferred)
- Test history per agent (Agent model ready, test runner integration separate)
- Agent monitoring/control via platform (future scope)
- Multi-use invites (max_uses field exists, UI shows single-use only for now)
- Rate limiting per token type (current per-user_id limits apply equally)
