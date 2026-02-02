"""ATP Dashboard v2 routes package.

This package contains modular route definitions for the ATP Dashboard API.
Each module handles a specific domain of functionality:

- home: Dashboard summary and home page routes
- agents: Agent CRUD operations
- suites: Suite execution queries and management
- tests: Test execution queries
- comparison: Agent comparison and side-by-side views
- auth: Authentication routes (login, register, user info)
- sso: SSO/OIDC authentication routes
- saml: SAML 2.0 authentication routes
- trends: Historical trend analysis
- timeline: Event timeline visualization
- leaderboard: Performance leaderboard matrix
- public_leaderboard: Public leaderboard with agent profiles (TASK-802)
- marketplace: Test suite marketplace (TASK-803)
- definitions: Suite definition CRUD and YAML export
- templates: Test template discovery
- traces: Debug endpoint for OpenTelemetry traces (dev mode only)
- metrics: Prometheus metrics endpoint
- costs: Cost analytics and breakdowns
- budgets: Budget management and monitoring
- analytics: Advanced analytics (trends, anomalies, correlations, export)
- experiments: A/B testing experiment management (TASK-506)
- tenants: Tenant management (admin-only CRUD, quotas, settings)
- roles: Role-based access control management
- audit: Audit logging (query, filter, export, retention)
- websocket: Real-time updates via WebSocket (TASK-801)
"""

from fastapi import APIRouter

from atp.dashboard.v2.routes.agents import router as agents_router
from atp.dashboard.v2.routes.analytics import router as analytics_router
from atp.dashboard.v2.routes.audit import router as audit_router
from atp.dashboard.v2.routes.auth import router as auth_router
from atp.dashboard.v2.routes.budgets import router as budgets_router
from atp.dashboard.v2.routes.comparison import router as comparison_router
from atp.dashboard.v2.routes.costs import router as costs_router
from atp.dashboard.v2.routes.definitions import router as definitions_router
from atp.dashboard.v2.routes.experiments import router as experiments_router
from atp.dashboard.v2.routes.home import router as home_router
from atp.dashboard.v2.routes.leaderboard import router as leaderboard_router
from atp.dashboard.v2.routes.marketplace import router as marketplace_router
from atp.dashboard.v2.routes.metrics import router as metrics_router
from atp.dashboard.v2.routes.public_leaderboard import (
    router as public_leaderboard_router,
)
from atp.dashboard.v2.routes.roles import router as roles_router
from atp.dashboard.v2.routes.saml import router as saml_router
from atp.dashboard.v2.routes.sso import router as sso_router
from atp.dashboard.v2.routes.suites import router as suites_router
from atp.dashboard.v2.routes.templates import router as templates_router
from atp.dashboard.v2.routes.tenants import router as tenants_router
from atp.dashboard.v2.routes.tests import router as tests_router
from atp.dashboard.v2.routes.timeline import router as timeline_router
from atp.dashboard.v2.routes.traces import router as traces_router
from atp.dashboard.v2.routes.trends import router as trends_router
from atp.dashboard.v2.routes.websocket import router as websocket_router

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
router.include_router(public_leaderboard_router)
router.include_router(marketplace_router)
router.include_router(timeline_router)
router.include_router(definitions_router)
router.include_router(templates_router)
router.include_router(traces_router)
router.include_router(metrics_router)
router.include_router(costs_router)
router.include_router(budgets_router)
router.include_router(analytics_router)
router.include_router(experiments_router)
router.include_router(tenants_router)
router.include_router(roles_router)
router.include_router(sso_router)
router.include_router(saml_router)
router.include_router(audit_router)
router.include_router(websocket_router)

__all__ = [
    "router",
    "agents_router",
    "analytics_router",
    "audit_router",
    "auth_router",
    "budgets_router",
    "comparison_router",
    "costs_router",
    "definitions_router",
    "experiments_router",
    "home_router",
    "leaderboard_router",
    "marketplace_router",
    "metrics_router",
    "public_leaderboard_router",
    "roles_router",
    "saml_router",
    "sso_router",
    "suites_router",
    "templates_router",
    "tenants_router",
    "tests_router",
    "timeline_router",
    "traces_router",
    "trends_router",
    "websocket_router",
]
