# Dashboard Frontend Phase 1: Shell + Benchmarks

**Date:** 2026-04-03
**Status:** Approved (rev.2)

## Overview

Add a server-rendered web UI to the ATP Dashboard using HTMX + Jinja2 + Pico CSS. Phase 1 delivers the app shell (sidebar navigation, layout) and two working pages (Home, Benchmarks).

## Tech Stack

- **HTMX 2.0** (CDN) — partial page updates, no custom JS
- **Jinja2** — server-side templates, integrated with FastAPI
- **Pico CSS** (CDN) — classless CSS framework, minimal custom styles
- **FastAPI** — serves both API (`/api/...`) and UI (`/ui/...`)
- **Zero build step** — no node_modules, no bundler, no TypeScript

## Pages

### Phase 1 (this spec)

| Route | Page | Data source |
|-------|------|-------------|
| `/ui/login` | GitHub Device Flow login | `POST /api/auth/device` |
| `/ui/` | Home — stats + recent activity | DB via service layer |
| `/ui/benchmarks` | Benchmark list (table) | DB via service layer |
| `/ui/benchmarks/{id}` | Benchmark detail | DB via service layer |

### Placeholders (future phases)

| Route | Section |
|-------|---------|
| `/ui/games` | Games & Tournaments |
| `/ui/runs` | Run history |
| `/ui/leaderboard` | Global leaderboard |
| `/ui/suites` | Suite management + YAML upload |
| `/ui/analytics` | Trends & analytics |

All placeholder routes are registered immediately (sidebar links never 404). Each renders `placeholder.html` with a `page_title` variable.

## Layout

Sidebar + Content layout:
- **Left sidebar** (220px, dark background `#1a1a2e`): logo, nav links, active item highlighted, user info at bottom
- **Content area**: full remaining width, light background
- Sidebar is server-rendered in `base.html`, active item set via Jinja2 variable

## Sidebar Navigation

1. Home
2. Benchmarks
3. Games
4. Runs
5. Leaderboard
6. Suites
7. Analytics
8. _(separator)_
9. Settings
10. User (email / logout)

## Architecture

### File Structure

```
packages/atp-dashboard/atp/dashboard/
├── templates/
│   ├── base.html              # Layout: sidebar + content block, CDN links
│   ├── login.html             # Login page (Device Flow)
│   ├── home.html              # Home with stats + activity
│   ├── benchmarks.html        # Benchmark list table
│   ├── benchmark_detail.html  # Single benchmark detail
│   ├── placeholder.html       # "Coming soon" for future pages
│   ├── error.html             # Error page (404, 500)
│   └── partials/
│       └── benchmark_table.html  # <tbody> partial for HTMX updates
├── static/
│   └── style.css              # Custom CSS overrides (sidebar, layout)
└── v2/routes/
    └── ui.py                  # UI routes (GET /ui/...)
```

### Router

New `ui.py` router with `prefix="/ui"`, serves HTML responses via Jinja2 `TemplateResponse`.

### Data Flow

UI handlers call the **service layer** directly (not HTTP internal API, not raw DB queries). This avoids HTTP round-trip overhead while maintaining separation of concerns.

```
Browser → ui.py handler → service layer (same process) → DB → Jinja2 template → HTML
```

Where no dedicated service exists (e.g., stats counts), the handler uses SQLAlchemy queries directly via the `DBSession` dependency — same pattern as API routes.

### HTMX Partial Update Pattern

All pages follow a standard pattern for HTMX partial updates:

```
GET /ui/benchmarks           → full page (base.html + content)
GET /ui/benchmarks?partial=1 → only the content fragment (e.g., <tbody>)
```

Handler checks `request.query_params.get("partial")`. If set, returns the partial template from `templates/partials/`. If not, renders the full page.

HTMX requests include `hx-get="/ui/benchmarks?partial=1"` with `hx-target="#benchmark-table-body"` and `hx-swap="innerHTML"`.

### Authentication

- UI routes check for JWT in httpOnly cookie (`atp_token`)
- If no valid token, redirect to `/ui/login?next={current_path}`
- Login page shows Device Flow instructions, polls via HTMX
- On success, sets httpOnly cookie and redirects to `next` param or `/ui/`
- Dependency: `get_current_user_or_redirect(request)` — returns User or raises redirect

**CSRF protection:** HTMX adds a custom header `HX-Request: true` on all requests. UI POST/DELETE handlers verify this header is present. Since custom headers cannot be set by cross-origin forms, this provides CSRF protection without a separate token. Base template sets `hx-headers='{"X-CSRF": "1"}'` on the `<body>` tag as an additional signal.

**Token expiry:** When JWT expires during a session, the next request returns a redirect to `/ui/login?next={path}&expired=1`. Login page shows "Session expired, please log in again" if `expired` param is set. No silent re-auth in Phase 1.

**Open redirect prevention:** The `next` parameter is validated to be a relative path starting with `/ui/`. Any other value is replaced with `/ui/`.

## Home Page

Three stat cards:
- Total benchmarks (count)
- Total runs (count)
- Active runs (in_progress count)

Recent activity feed (last 10 events): run completions, benchmark creations.

## Benchmarks Page

Table with columns: Name, Version, Tasks, Tags, Created. Each row links to detail page. Uses Pico CSS `<table>` (styled automatically).

**Pagination:** Server-side, 50 items per page. Rendered as page links at bottom. HTMX loads next page via `hx-get="/ui/benchmarks?partial=1&page=2"`. No infinite scroll — explicit page numbers.

## Benchmark Detail Page

Sections:
- Header: name, description, version, family_tag
- Tests list: table with id, name, description, assertion count
- Recent runs: last 10 runs, table with agent, status, score, date
- Leaderboard preview: top 5 agents

## Error Handling

| Scenario | Response |
|----------|----------|
| Page not found (404) | `error.html` with "Page not found" message |
| Server error (500) | `error.html` with "Something went wrong" (no stack trace in production) |
| Benchmark not found | `error.html` with "Benchmark #{id} not found" |
| Session expired (401) | Redirect to `/ui/login?next={path}&expired=1` |
| DB unavailable | `error.html` with "Service temporarily unavailable" |

FastAPI exception handlers registered for 404 and 500 on UI routes.

## Scope

- No custom JavaScript beyond HTMX
- No dark mode (Pico CSS default light theme)
- No responsive/mobile layout (desktop-first, dashboard use case)
- No WebSocket real-time updates (future phase)
- No pagination on benchmark detail sub-tables (runs, tests) — data volumes are small in Phase 1
- Templates are minimal — Pico CSS handles most styling, `style.css` covers sidebar and layout grid
