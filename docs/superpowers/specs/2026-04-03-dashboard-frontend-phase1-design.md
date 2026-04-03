# Dashboard Frontend Phase 1: Shell + Benchmarks

**Date:** 2026-04-03
**Status:** Approved

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
| `/ui/` | Home — stats + recent activity | `GET /api/home/summary` |
| `/ui/benchmarks` | Benchmark list (table) | `GET /api/v1/benchmarks` |
| `/ui/benchmarks/{id}` | Benchmark detail — tests, runs, leaderboard preview | `GET /api/v1/benchmarks/{id}`, `/api/v1/benchmarks/{id}/leaderboard` |

### Placeholders (future phases)

| Route | Section |
|-------|---------|
| `/ui/games` | Games & Tournaments |
| `/ui/runs` | Run history |
| `/ui/leaderboard` | Global leaderboard |
| `/ui/suites` | Suite management + YAML upload |
| `/ui/analytics` | Trends & analytics |

Placeholder pages show the sidebar + a "Coming soon" message.

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
│   ├── base.html          # Layout: sidebar + content block, CDN links
│   ├── login.html         # Login page (Device Flow)
│   ├── home.html          # Home with stats + activity
│   ├── benchmarks.html    # Benchmark list table
│   ├── benchmark_detail.html  # Single benchmark detail
│   └── placeholder.html   # "Coming soon" placeholder
├── static/
│   └── style.css          # Custom CSS overrides (minimal)
└── v2/routes/
    └── ui.py              # UI routes (GET /ui/...)
```

### Router

New `ui.py` router with `prefix="/ui"`, serves HTML responses via Jinja2 `TemplateResponse`.

### Data Flow

1. User navigates to `/ui/benchmarks`
2. `ui.py` handler queries DB directly (same pattern as API routes) or calls internal API
3. Renders Jinja2 template with data
4. HTMX handles partial updates (e.g., sorting, filtering) via `hx-get` to API endpoints

### Authentication

- UI routes check for JWT in cookie (`atp_token`)
- If no valid token, redirect to `/ui/login`
- Login page shows Device Flow instructions, polls via HTMX
- On success, sets httpOnly cookie and redirects to `/ui/`
- Middleware or dependency: `get_current_user_or_redirect()`

## Home Page

Three stat cards:
- Total benchmarks (count)
- Total runs (count)
- Active runs (in_progress count)

Recent activity feed (last 10 events): run completions, benchmark creations.

Data comes from existing `/api/home/summary` endpoint.

## Benchmarks Page

Table with columns: Name, Version, Tasks, Tags, Created. Each row links to detail page. Uses Pico CSS `<table>` (styled automatically).

HTMX: table loads via `hx-get` for instant filter/sort without full page reload.

## Benchmark Detail Page

Sections:
- Header: name, description, version, family_tag
- Tests list: table with id, name, description, assertion count
- Recent runs: table with agent, status, score, date
- Leaderboard preview: top 5 agents

## Scope

- No custom JavaScript beyond HTMX
- No dark mode (Pico CSS default light theme)
- No responsive/mobile layout (desktop-first, dashboard use case)
- No WebSocket real-time updates (future phase)
- Templates are minimal — Pico CSS handles most styling
