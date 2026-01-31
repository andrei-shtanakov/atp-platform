"""ATP Dashboard v2 routes package.

This package contains modular route definitions for the ATP Dashboard API.
Each module handles a specific domain of functionality:

- home: Dashboard summary and home page routes
- agents: Agent CRUD operations
- suites: Suite execution queries and management
- tests: Test execution queries
- comparison: Agent comparison and side-by-side views
- auth: Authentication routes (login, register, user info)
- trends: Historical trend analysis
- timeline: Event timeline visualization
- leaderboard: Performance leaderboard matrix
- definitions: Suite definition CRUD and YAML export
- templates: Test template discovery
"""

from fastapi import APIRouter

from atp.dashboard.v2.routes.agents import router as agents_router
from atp.dashboard.v2.routes.auth import router as auth_router
from atp.dashboard.v2.routes.comparison import router as comparison_router
from atp.dashboard.v2.routes.definitions import router as definitions_router
from atp.dashboard.v2.routes.home import router as home_router
from atp.dashboard.v2.routes.leaderboard import router as leaderboard_router
from atp.dashboard.v2.routes.suites import router as suites_router
from atp.dashboard.v2.routes.templates import router as templates_router
from atp.dashboard.v2.routes.tests import router as tests_router
from atp.dashboard.v2.routes.timeline import router as timeline_router
from atp.dashboard.v2.routes.trends import router as trends_router

# Create the main API router that aggregates all sub-routers
router = APIRouter()

# Include all domain-specific routers
router.include_router(auth_router)
router.include_router(home_router)
router.include_router(agents_router)
router.include_router(suites_router)
router.include_router(tests_router)
router.include_router(trends_router)
router.include_router(comparison_router)
router.include_router(leaderboard_router)
router.include_router(timeline_router)
router.include_router(definitions_router)
router.include_router(templates_router)

__all__ = [
    "router",
    "agents_router",
    "auth_router",
    "comparison_router",
    "definitions_router",
    "home_router",
    "leaderboard_router",
    "suites_router",
    "templates_router",
    "tests_router",
    "timeline_router",
    "trends_router",
]
