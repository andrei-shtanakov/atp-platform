# Auth UX — Design Spec

**Goal:** Add login/logout flow to the dashboard UI so users can authenticate via GitHub OAuth and access auth-protected features (admin timeline, raw JSON API).

**Scope:** UI auth flow only (scope #1). Token self-service and agent ownership are separate future specs.

## What Already Works

- GitHub Device Flow at `/api/auth/device` + `/api/auth/device/poll`
- Local credential login at `/api/auth/token`
- Login page template at `/ui/login` with both flows (GitHub + local)
- Register page at `/ui/register` (first user = admin)
- JWT issuance with `sub` + `user_id` claims, 60-min TTL
- `atp_token` cookie set by login.html JavaScript on successful auth
- `JWTUserStateMiddleware` reads `Authorization: Bearer` header → sets `request.state.user_id`
- `get_current_user` FastAPI dependency reads Bearer token → returns User

## What's Missing

1. **No link to login page** — `/ui/login` exists but nothing in the dashboard links to it
2. **No logout** — no way to clear the session
3. **Cookie not read by auth** — browser sends `atp_token` cookie to `/api/*` (same origin) but `JWTUserStateMiddleware` and `get_current_user` only read `Authorization: Bearer` header. So "View raw JSON →" returns 401 even when logged in.
4. **Sidebar doesn't show auth state** — always shows username or nothing, no login/logout controls

## Changes

### 1. Sidebar auth controls (`base_ui.html`)

**Anonymous state** (no `user` in context):
```
[sidebar footer]
  Sign in   ← link to /ui/login
```

**Logged-in state** (`user` present):
```
[sidebar footer]
  andrei-shtanakov   Sign out
```

"Sign out" is a link to `POST /ui/logout` (via HTMX or a small form).

### 2. Logout route (`ui.py`)

New route `POST /ui/logout`:
- Clear `atp_token` cookie (set to empty, `Max-Age=0`)
- Redirect to `/ui/` (302)

### 3. Cookie fallback in auth middleware

**`JWTUserStateMiddleware`** (`rate_limit.py`): after checking `Authorization: Bearer`, also check `request.cookies.get("atp_token")`. Same JWT decode logic.

**`get_current_user`** (`auth/__init__.py`): the `OAuth2PasswordBearer` dependency only reads `Authorization` header. Add a secondary source: if Bearer token is None, try `request.cookies.get("atp_token")`. This requires changing the dependency signature to also accept `Request`.

### 4. Pass `user` to all UI templates

Currently `base_ui.html` checks `{% if user %}` but most routes don't pass `user` to the template context. The sidebar needs `user` to show login/logout state.

**Approach:** Add a helper that resolves the current user from cookie/header and returns `User | None`. Call it in each UI route handler and add `user` to the template context. This is lightweight — one `session.get(User, user_id)` call, and `user_id` is already available from `request.state.user_id` (set by middleware).

### 5. Login page: GitHub primary, credentials secondary

Existing `/ui/login` template already has both flows. Adjust visual hierarchy:
- "Sign in with GitHub" button — prominent, primary action
- "Or sign in with credentials" — small text link below that reveals the form

No functional change to the login logic — just CSS/layout reordering.

## Access Model

| Content | Anonymous | Logged-in | Admin |
|---------|-----------|-----------|-------|
| Tournament list (public) | ✓ | ✓ | ✓ |
| Tournament detail (public) | ✓ | ✓ | ✓ |
| Tournament detail — event timeline | ✗ | ✗ | ✓ |
| Private tournaments | ✗ | owner/participant only | ✓ |
| "View raw JSON →" (API) | ✗ (401) | ✓ | ✓ |
| Benchmarks, runs, leaderboard, analytics | ✓ | ✓ | ✓ |

## File Changes

### Modified files

| File | Change |
|------|--------|
| `templates/ui/base_ui.html` | Sidebar footer: login button (anon) or username + sign-out (logged in) |
| `templates/ui/login.html` | Reorder: GitHub primary, credentials secondary |
| `routes/ui.py` | Add `POST /ui/logout` route. Add `_get_ui_user()` helper. Pass `user` to all template contexts. |
| `auth/__init__.py` | `get_current_user`: fallback to `atp_token` cookie if no Bearer token |
| `v2/rate_limit.py` | `JWTUserStateMiddleware`: also read `atp_token` cookie |

### New files

None.

## Non-goals

- User self-registration flow changes (works as-is via `/ui/register`)
- Token self-service UI ("My Tokens")
- Agent ownership model
- Session management / token revocation
- Password reset flow
- 2FA/MFA
