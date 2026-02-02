# Dashboard v2 Migration Guide

Guide for migrating custom extensions and integrations from Dashboard v1 to v2.

## Overview

Dashboard v2 introduces a modular architecture that separates concerns and improves maintainability. This guide helps you migrate custom routes, services, and templates from v1 to v2.

---

## Architecture Comparison

### v1 Architecture (Deprecated)

```
atp/dashboard/
├── app.py           # Monolithic: routes, HTML, all logic (237KB)
├── api.py           # Additional API routes
└── ...
```

### v2 Architecture (Recommended)

```
atp/dashboard/
├── v2/
│   ├── factory.py        # App factory & lifespan (~100 lines)
│   ├── config.py         # Configuration management
│   ├── dependencies.py   # Dependency injection
│   ├── routes/           # Modular route handlers
│   │   ├── __init__.py   # Router aggregation
│   │   ├── home.py       # Dashboard summary
│   │   ├── agents.py     # Agent CRUD
│   │   ├── suites.py     # Suite queries
│   │   ├── tests.py      # Test queries
│   │   ├── comparison.py # Agent comparison
│   │   ├── trends.py     # Historical trends
│   │   ├── timeline.py   # Event timeline
│   │   ├── leaderboard.py# Performance matrix
│   │   ├── definitions.py# Suite definitions
│   │   ├── templates.py  # Template discovery
│   │   ├── websocket.py  # WebSocket real-time updates
│   │   ├── marketplace.py# Test suite marketplace
│   │   ├── public_leaderboard.py # Public benchmark leaderboard
│   │   ├── saml.py       # SAML SSO authentication
│   │   └── audit.py      # Audit logging
│   ├── services/         # Business logic layer
│   │   ├── test_service.py
│   │   ├── agent_service.py
│   │   ├── comparison_service.py
│   │   └── export_service.py
│   └── templates/        # Jinja2 templates
│       ├── base.html
│       ├── home.html
│       ├── test_results.html
│       ├── comparison.html
│       └── components/   # Reusable components
└── ...                   # v1 files (deprecated)
```

---

## Migration Steps

### Step 1: Enable v2 Feature Flag

Before migrating, verify v2 works with your existing setup:

```bash
# Enable v2
export ATP_DASHBOARD_V2=true

# Run tests
uv run pytest tests/unit/dashboard/ -v

# Start dashboard and verify
uv run atp dashboard
```

### Step 2: Migrate Custom Routes

#### v1 Route (app.py style)

```python
# v1: Route defined inline in app.py
from fastapi import APIRouter
from sqlalchemy import select
from atp.dashboard.api import get_session
from atp.dashboard.models import CustomModel

router = APIRouter()

@router.get("/api/custom/endpoint")
async def my_custom_endpoint():
    async with get_session() as session:
        result = await session.execute(select(CustomModel))
        return result.scalars().all()
```

#### v2 Route (modular style)

```python
# v2: atp/dashboard/v2/routes/custom.py
from fastapi import APIRouter
from atp.dashboard.models import CustomModel
from atp.dashboard.v2.dependencies import DBSession, CurrentUser
from sqlalchemy import select

router = APIRouter(prefix="/custom", tags=["custom"])

@router.get("/endpoint")
async def my_custom_endpoint(
    session: DBSession,
    user: CurrentUser,
) -> list[dict]:
    """Get custom data.

    Args:
        session: Database session (injected).
        user: Current user (optional auth).

    Returns:
        List of custom data.
    """
    result = await session.execute(select(CustomModel))
    return [item.to_dict() for item in result.scalars().all()]
```

Register the route in `routes/__init__.py`:

```python
# atp/dashboard/v2/routes/__init__.py
from atp.dashboard.v2.routes.custom import router as custom_router

# Add to main router
router.include_router(custom_router)
```

### Step 3: Migrate Business Logic to Services

#### v1 Style (logic in routes)

```python
# v1: Business logic mixed with route handling
@router.get("/api/reports/summary")
async def get_report_summary():
    async with get_session() as session:
        # Complex business logic inline
        agents = await session.execute(select(Agent))
        suites = await session.execute(select(SuiteExecution))

        # Calculations
        total_tests = sum(s.total_tests for s in suites.scalars())
        avg_score = calculate_average_score(suites)
        trends = calculate_trends(suites)

        return {
            "total_agents": len(list(agents.scalars())),
            "total_tests": total_tests,
            "avg_score": avg_score,
            "trends": trends,
        }
```

#### v2 Style (service layer)

```python
# v2: atp/dashboard/v2/services/report_service.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import Agent, SuiteExecution


class ReportService:
    """Service for generating reports."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_summary(self) -> dict:
        """Get report summary.

        Returns:
            Summary dictionary with agents, tests, scores, trends.
        """
        agents = await self.session.execute(select(Agent))
        suites = await self.session.execute(select(SuiteExecution))

        suite_list = list(suites.scalars())
        total_tests = sum(s.total_tests for s in suite_list)
        avg_score = self._calculate_average_score(suite_list)
        trends = self._calculate_trends(suite_list)

        return {
            "total_agents": len(list(agents.scalars())),
            "total_tests": total_tests,
            "avg_score": avg_score,
            "trends": trends,
        }

    def _calculate_average_score(self, suites: list) -> float:
        """Calculate average score across suites."""
        scores = [s.avg_score for s in suites if s.avg_score is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def _calculate_trends(self, suites: list) -> list:
        """Calculate trend data."""
        # Implementation
        return []
```

Register the dependency:

```python
# v2: atp/dashboard/v2/dependencies.py
from typing import Annotated
from fastapi import Depends
from atp.dashboard.v2.services.report_service import ReportService

async def get_report_service(session: DBSession) -> ReportService:
    """Get a ReportService instance."""
    return ReportService(session)

ReportServiceDep = Annotated[ReportService, Depends(get_report_service)]
```

Use in routes:

```python
# v2: atp/dashboard/v2/routes/reports.py
from atp.dashboard.v2.dependencies import ReportServiceDep, CurrentUser

router = APIRouter(prefix="/reports", tags=["reports"])

@router.get("/summary")
async def get_report_summary(
    service: ReportServiceDep,
    user: CurrentUser,
) -> dict:
    """Get report summary."""
    return await service.get_summary()
```

### Step 4: Migrate Templates

#### v1 Style (inline HTML)

```python
# v1: HTML embedded in Python
def create_index_html() -> str:
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Dashboard</title></head>
    <body>
        <div id="root"></div>
        <script>
            // Inline JavaScript
        </script>
    </body>
    </html>
    """
```

#### v2 Style (Jinja2 templates)

```html
<!-- v2: atp/dashboard/v2/templates/custom_page.html -->
{% extends "base.html" %}

{% block title %}Custom Page - ATP Dashboard{% endblock %}

{% block content %}
<div class="container mx-auto px-4 py-8">
    <h1 class="text-2xl font-bold mb-4">Custom Page</h1>
    <div id="custom-root"></div>
</div>
{% endblock %}

{% block scripts %}
<script type="text/babel">
    function CustomPage() {
        const [data, setData] = React.useState(null);

        React.useEffect(() => {
            fetch('/api/custom/endpoint')
                .then(r => r.json())
                .then(setData);
        }, []);

        if (!data) return <SkeletonBox />;

        return (
            <div>
                {/* Render data */}
            </div>
        );
    }

    const root = ReactDOM.createRoot(document.getElementById('custom-root'));
    root.render(<CustomPage />);
</script>
{% endblock %}
```

Register route for template:

```python
# v2: In factory.py or routes
@app.get("/custom-page", response_class=HTMLResponse)
async def custom_page(request: Request) -> HTMLResponse:
    """Render the custom page template."""
    return templates.TemplateResponse(
        request=request,
        name="custom_page.html",
    )
```

---

## Dependency Injection Reference

v2 provides pre-configured dependency types for clean route signatures:

```python
from atp.dashboard.v2.dependencies import (
    # Database
    DBSession,           # AsyncSession for DB operations

    # Configuration
    Config,              # DashboardConfig instance

    # Authentication
    CurrentUser,         # Optional user (may be None)
    RequiredUser,        # Required authenticated user
    AdminUser,           # Required admin user

    # Pagination
    Pagination,          # PaginationParams with offset/limit

    # Services
    TestServiceDep,      # TestService instance
    AgentServiceDep,     # AgentService instance
    ComparisonServiceDep,# ComparisonService instance
    ExportServiceDep,    # ExportService instance
)
```

Example usage:

```python
@router.get("/items")
async def list_items(
    session: DBSession,
    user: RequiredUser,      # Requires authentication
    pagination: Pagination,
    config: Config,
) -> list[dict]:
    # Use session, user, pagination, config
    ...
```

---

## Configuration

### Environment Variables

```bash
# Core
ATP_DASHBOARD_V2=true          # Enable v2 architecture
ATP_DATABASE_URL=...           # Database connection URL
ATP_DEBUG=true                 # Enable debug mode

# Authentication
ATP_SECRET_KEY=...             # JWT secret key
ATP_TOKEN_EXPIRE_MINUTES=60    # Token expiration

# CORS
ATP_CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Server
ATP_HOST=0.0.0.0
ATP_PORT=8080
```

### Programmatic Configuration

```python
from atp.dashboard.v2 import create_app, DashboardConfig

config = DashboardConfig(
    database_url="postgresql+asyncpg://localhost/atp",
    database_echo=True,
    debug=True,
    secret_key="my-secret-key",
    token_expire_minutes=120,
    cors_origins="http://localhost:3000",
    host="0.0.0.0",
    port=8080,
    title="My ATP Dashboard",
    version="1.0.0",
)

app = create_app(config=config)
```

---

## Testing

### v1 Test Style

```python
# v1: Manual app setup
from atp.dashboard.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_endpoint():
    response = client.get("/api/endpoint")
    assert response.status_code == 200
```

### v2 Test Style

```python
# v2: Using test factory
import pytest
from httpx import AsyncClient, ASGITransport
from atp.dashboard.v2 import create_test_app

@pytest.fixture
def test_app():
    """Create test app with in-memory database."""
    return create_test_app(use_v2_routes=True)

@pytest.fixture
async def client(test_app):
    """Create async test client."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

@pytest.mark.anyio
async def test_endpoint(client):
    """Test custom endpoint."""
    response = await client.get("/api/custom/endpoint")
    assert response.status_code == 200
```

---

## Common Migration Issues

### Issue 1: Session Management

**v1 Problem**:
```python
# v1: Manual session context
async with get_session() as session:
    ...
```

**v2 Solution**:
```python
# v2: Session injected via dependency
async def my_route(session: DBSession):
    # Session is already opened, will be committed/rolled back automatically
    ...
```

### Issue 2: CORS Configuration

**v1 Problem**:
```python
# v1: CORS configured inline
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

**v2 Solution**:
```python
# v2: CORS configured via DashboardConfig
config = DashboardConfig(cors_origins="http://localhost:3000")
app = create_app(config=config)
```

### Issue 3: Static Files

**v1 Problem**:
```python
# v1: Static files served from app.py
```

**v2 Solution**:
```python
# v2: Static files mounted at /static/v2
# Place files in atp/dashboard/v2/static/
# Access via /static/v2/css/dashboard.css
```

---

## Backward Compatibility

During migration, both versions can coexist:

```python
from atp.dashboard import create_app

# Returns v1 or v2 based on ATP_DASHBOARD_V2 env var
app = create_app()
```

Or explicitly choose:

```python
# Explicit v1
from atp.dashboard.app import create_app as create_v1_app
app = create_v1_app()

# Explicit v2
from atp.dashboard.v2 import create_app as create_v2_app
app = create_v2_app()
```

---

## Deprecation Timeline

| Version | Status | Notes |
|---------|--------|-------|
| v1 | Deprecated | Still works, but no new features |
| v2 | Recommended | Active development |

**Planned removal**: v1 will be removed in ATP 1.0.0. Migrate to v2 before then.

---

## Checklist

Before completing migration:

- [ ] Enable `ATP_DASHBOARD_V2=true`
- [ ] All tests pass with v2 enabled
- [ ] Custom routes migrated to `v2/routes/`
- [ ] Business logic extracted to services
- [ ] Templates extracted to `v2/templates/`
- [ ] Dependencies registered in `v2/dependencies.py`
- [ ] Configuration uses `DashboardConfig`
- [ ] Documentation updated

---

## See Also

- [Dashboard API Reference](dashboard-api.md) - API endpoints
- [Architecture](../03-architecture.md) - System architecture
- [Testing Guide](../guides/testing-guide-en.md) - Testing patterns
