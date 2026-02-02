# Dashboard API Reference

Complete reference for ATP Dashboard REST API endpoints.

## Overview

The ATP Dashboard provides a web interface and REST API for viewing test results, comparing agents, and analyzing execution patterns. The API is built with FastAPI and supports optional JWT authentication.

**Base URL**: `http://localhost:8000/api`

**Starting the Dashboard**:
```bash
uv run atp dashboard
# or
uv run python -m atp.dashboard
```

---

## Dashboard Architecture Versions

The dashboard supports two architecture versions:

### v1 (Default - Deprecated)

The original monolithic implementation with all routes in a single `app.py` file. This version is **deprecated** and will be removed in a future release.

### v2 (Recommended)

The new modular architecture with:
- **App Factory Pattern**: Configurable app creation via `create_app()`
- **Modular Routes**: Routes organized by domain (`routes/agents.py`, `routes/comparison.py`, etc.)
- **Service Layer**: Business logic separated into services (`services/test_service.py`, etc.)
- **Jinja2 Templates**: Extracted HTML templates with reusable components
- **Dependency Injection**: Clean dependency management via FastAPI's `Depends()`

**Enable v2**:
```bash
# Set environment variable
export ATP_DASHBOARD_V2=true

# Start dashboard
uv run atp dashboard
```

**Programmatic Usage**:
```python
from atp.dashboard.v2 import create_app, DashboardConfig

# Default configuration
app = create_app()

# Custom configuration
config = DashboardConfig(
    debug=True,
    database_url="postgresql+asyncpg://localhost/atp",
    cors_origins="http://localhost:3000,http://localhost:8080"
)
app = create_app(config=config)

# For testing
from atp.dashboard.v2 import create_test_app
test_app = create_test_app(use_v2_routes=True)
```

See [Dashboard Migration Guide](dashboard-migration.md) for migrating custom extensions to v2.

---

## Authentication

Authentication is optional by default. When enabled, use JWT Bearer tokens.

### Login

```http
POST /api/auth/token
Content-Type: application/x-www-form-urlencoded

username=user&password=pass
```

**Response**:
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}
```

**Usage**:
```bash
# Get token
TOKEN=$(curl -s -X POST "http://localhost:8000/api/auth/token" \
  -d "username=admin&password=secret" | jq -r '.access_token')

# Use token
curl -H "Authorization: Bearer $TOKEN" "http://localhost:8000/api/agents"
```

---

## Role-Based Access Control (RBAC)

The dashboard uses RBAC to control access to API endpoints. Each endpoint requires specific permissions that are granted through roles.

### Default Roles

| Role | Description | Key Permissions |
|------|-------------|-----------------|
| `admin` | Full system access | All permissions |
| `developer` | Read/write access for testing | Suites, agents, results read/write |
| `analyst` | Analytics and reporting access | Analytics, results read, export |
| `viewer` | Read-only access | All read permissions |

### Permissions Matrix

| Permission | admin | developer | analyst | viewer |
|------------|:-----:|:---------:|:-------:|:------:|
| `suites:read` | ✓ | ✓ | ✓ | ✓ |
| `suites:write` | ✓ | ✓ | - | - |
| `suites:delete` | ✓ | - | - | - |
| `agents:read` | ✓ | ✓ | ✓ | ✓ |
| `agents:write` | ✓ | ✓ | - | - |
| `agents:delete` | ✓ | - | - | - |
| `results:read` | ✓ | ✓ | ✓ | ✓ |
| `analytics:read` | ✓ | ✓ | ✓ | ✓ |
| `analytics:write` | ✓ | ✓ | ✓ | - |
| `analytics:delete` | ✓ | ✓ | - | - |
| `analytics:export` | ✓ | ✓ | ✓ | - |
| `budgets:read` | ✓ | ✓ | ✓ | ✓ |
| `budgets:write` | ✓ | ✓ | - | - |
| `budgets:delete` | ✓ | - | - | - |
| `marketplace:read` | ✓ | ✓ | ✓ | ✓ |
| `marketplace:write` | ✓ | ✓ | - | - |
| `marketplace:admin` | ✓ | - | - | - |
| `leaderboard:read` | ✓ | ✓ | ✓ | ✓ |
| `leaderboard:write` | ✓ | ✓ | - | - |
| `leaderboard:admin` | ✓ | - | - | - |
| `users:read` | ✓ | - | - | - |
| `users:write` | ✓ | - | - | - |
| `roles:read` | ✓ | - | - | - |
| `roles:write` | ✓ | - | - | - |
| `roles:delete` | ✓ | - | - | - |

### Endpoint Permissions

| Endpoint | Method | Required Permission |
|----------|--------|---------------------|
| `/api/agents` | GET | `agents:read` |
| `/api/agents` | POST | `agents:write` |
| `/api/agents/{id}` | GET | `agents:read` |
| `/api/agents/{id}` | PATCH | `agents:write` |
| `/api/agents/{id}` | DELETE | `agents:delete` |
| `/api/suites` | GET | `suites:read` |
| `/api/suites/{id}` | GET | `suites:read` |
| `/api/tests` | GET | `results:read` |
| `/api/tests/{id}` | GET | `results:read` |
| `/api/compare/agents` | GET | `results:read` |
| `/api/compare/side-by-side` | GET | `results:read` |
| `/api/leaderboard/matrix` | GET | `results:read` |
| `/api/timeline/events` | GET | `results:read` |
| `/api/timeline/compare` | GET | `results:read` |
| `/api/trends/suite` | GET | `results:read` |
| `/api/trends/test` | GET | `results:read` |
| `/api/dashboard/summary` | GET | `suites:read` |
| `/api/suite-definitions` | GET | `suites:read` |
| `/api/suite-definitions` | POST | `suites:write` |
| `/api/suite-definitions/{id}` | GET | `suites:read` |
| `/api/suite-definitions/{id}` | DELETE | `suites:delete` |
| `/api/templates` | GET | `suites:read` |
| `/api/costs` | GET | `analytics:read` |
| `/api/costs/*` | GET | `analytics:read` |
| `/api/analytics/trends` | GET | `analytics:read` |
| `/api/analytics/anomalies` | GET | `analytics:read` |
| `/api/analytics/correlations` | GET | `analytics:read` |
| `/api/analytics/export/*` | GET | `analytics:export` |
| `/api/analytics/reports` | GET | `analytics:read` |
| `/api/analytics/reports` | POST | `analytics:write` |
| `/api/analytics/reports/{id}` | PUT | `analytics:write` |
| `/api/analytics/reports/{id}` | DELETE | `analytics:delete` |
| `/api/budgets` | GET | `budgets:read` |
| `/api/budgets` | POST | `budgets:write` |
| `/api/budgets/{id}` | GET | `budgets:read` |
| `/api/budgets/{id}` | PUT | `budgets:write` |
| `/api/budgets/{id}` | DELETE | `budgets:delete` |
| `/api/roles` | GET | `roles:read` |
| `/api/roles` | POST | `roles:write` |
| `/api/roles/{id}` | GET | `roles:read` |
| `/api/roles/{id}` | PATCH | `roles:write` |
| `/api/roles/{id}` | DELETE | `roles:delete` |
| `/api/roles/users/{id}/roles` | GET | `users:read` |
| `/api/roles/users/assign` | POST | Admin only |
| `/api/roles/users/{id}/roles/{id}` | DELETE | Admin only |
| `/api/tenants/*` | ALL | Admin only |

### Role Management API

#### List Roles

```http
GET /api/roles
Authorization: Bearer <token>
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `include_inactive` | bool | No | false | Include inactive roles |
| `limit` | int | No | 50 | Max results |
| `offset` | int | No | 0 | Pagination offset |

**Response**:
```json
[
  {
    "id": 1,
    "name": "admin",
    "description": "Administrator with full access",
    "is_system": true,
    "is_active": true,
    "permissions": ["suites:read", "suites:write", "agents:read", ...],
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z"
  }
]
```

#### Create Custom Role

```http
POST /api/roles
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "test-lead",
  "description": "Test team lead with extended permissions",
  "permissions": ["suites:read", "suites:write", "agents:read", "results:read"]
}
```

#### Assign Role to User

```http
POST /api/roles/users/assign
Authorization: Bearer <token>
Content-Type: application/json

{
  "user_id": 5,
  "role_id": 2
}
```

#### Get User Permissions

```http
GET /api/roles/users/{user_id}/permissions
Authorization: Bearer <token>
```

**Response**:
```json
{
  "user_id": 5,
  "username": "developer1",
  "is_admin": false,
  "roles": [
    {
      "id": 2,
      "name": "developer",
      "permissions": ["suites:read", "suites:write", ...]
    }
  ],
  "permissions": ["suites:read", "suites:write", "agents:read", ...]
}
```

#### Get Current User Permissions

```http
GET /api/roles/me/permissions
Authorization: Bearer <token>
```

Returns the authenticated user's roles and effective permissions.

---

## Agent Comparison

### Side-by-Side Comparison

Compare 2-3 agents on a specific test with detailed execution events.

```http
GET /api/compare/side-by-side
```

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `suite_name` | string | Yes | Test suite name |
| `test_id` | string | Yes | Test identifier |
| `agents` | string[] | Yes | Agent names (2-3 agents) |

**Example**:
```bash
curl "http://localhost:8000/api/compare/side-by-side?\
suite_name=demo_suite&\
test_id=test-001&\
agents=gpt-4&\
agents=claude-3"
```

**Response** (`SideBySideComparisonResponse`):
```json
{
  "suite_name": "demo_suite",
  "test_id": "test-001",
  "test_name": "Basic file creation",
  "agents": [
    {
      "agent_name": "gpt-4",
      "test_execution_id": 42,
      "score": 0.85,
      "success": true,
      "duration_seconds": 12.5,
      "total_tokens": 1500,
      "total_steps": 5,
      "tool_calls": 3,
      "llm_calls": 2,
      "cost_usd": 0.045,
      "events": [
        {
          "sequence": 1,
          "timestamp": "2025-01-26T10:00:00Z",
          "event_type": "llm_request",
          "summary": "LLM request: gpt-4 (500 tokens)",
          "data": {
            "model": "gpt-4",
            "input_tokens": 200,
            "output_tokens": 300
          }
        },
        {
          "sequence": 2,
          "timestamp": "2025-01-26T10:00:05Z",
          "event_type": "tool_call",
          "summary": "Tool call: file_write (success)",
          "data": {
            "tool": "file_write",
            "status": "success",
            "args": {"path": "output.txt"}
          }
        }
      ]
    },
    {
      "agent_name": "claude-3",
      "test_execution_id": 43,
      "score": 0.92,
      "success": true,
      "duration_seconds": 8.2,
      "total_tokens": 1200,
      "total_steps": 4,
      "tool_calls": 2,
      "llm_calls": 2,
      "cost_usd": 0.036,
      "events": [...]
    }
  ]
}
```

**Event Types**:
- `tool_call` — Tool invocation with name, args, result
- `llm_request` — LLM API call with model, tokens
- `reasoning` — Agent reasoning/planning step
- `error` — Error occurrence
- `progress` — Progress update

---

### General Agent Comparison

Compare multiple agents across all tests in a suite.

```http
GET /api/compare/agents
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `suite_name` | string | Yes | - | Test suite name |
| `agents` | string[] | Yes | - | Agent names to compare |
| `limit_per_agent` | int | No | 10 | Max executions per agent |

**Example**:
```bash
curl "http://localhost:8000/api/compare/agents?\
suite_name=demo_suite&\
agents=gpt-4&\
agents=claude-3&\
limit_per_agent=20"
```

**Response** (`AgentComparisonResponse`):
```json
{
  "suite_name": "demo_suite",
  "agents": [
    {
      "agent_name": "gpt-4",
      "total_executions": 15,
      "avg_success_rate": 0.87,
      "avg_score": 0.82,
      "avg_duration_seconds": 14.5,
      "latest_success_rate": 0.90,
      "latest_score": 0.85
    },
    {
      "agent_name": "claude-3",
      "total_executions": 15,
      "avg_success_rate": 0.93,
      "avg_score": 0.88,
      "avg_duration_seconds": 10.2,
      "latest_success_rate": 1.0,
      "latest_score": 0.92
    }
  ],
  "tests": [
    {
      "test_id": "test-001",
      "test_name": "Basic file creation",
      "metrics_by_agent": {
        "gpt-4": {
          "agent_name": "gpt-4",
          "total_executions": 5,
          "avg_success_rate": 1.0,
          "avg_score": 0.90,
          "avg_duration_seconds": 12.0
        },
        "claude-3": {...}
      }
    }
  ]
}
```

---

## Leaderboard Matrix

### Get Leaderboard Matrix

Returns a matrix of tests (rows) × agents (columns) with scores, rankings, and patterns.

```http
GET /api/leaderboard/matrix
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `suite_name` | string | Yes | - | Test suite name |
| `agents` | string[] | No | all | Filter by agent names |
| `limit_executions` | int | No | 5 | Recent executions to consider |
| `limit` | int | No | 50 | Max tests (pagination) |
| `offset` | int | No | 0 | Pagination offset |

**Example**:
```bash
curl "http://localhost:8000/api/leaderboard/matrix?\
suite_name=demo_suite&\
limit=20&\
offset=0"
```

**Response** (`LeaderboardMatrixResponse`):
```json
{
  "suite_name": "demo_suite",
  "tests": [
    {
      "test_id": "test-001",
      "test_name": "Basic file creation",
      "tags": ["smoke", "basic"],
      "scores_by_agent": {
        "gpt-4": {
          "score": 0.85,
          "success": true,
          "execution_count": 5
        },
        "claude-3": {
          "score": 0.92,
          "success": true,
          "execution_count": 5
        }
      },
      "avg_score": 0.885,
      "difficulty": "easy",
      "pattern": null
    },
    {
      "test_id": "test-005",
      "test_name": "Complex reasoning task",
      "tags": ["advanced"],
      "scores_by_agent": {
        "gpt-4": {"score": 0.45, "success": false, "execution_count": 5},
        "claude-3": {"score": 0.38, "success": false, "execution_count": 5}
      },
      "avg_score": 0.415,
      "difficulty": "hard",
      "pattern": "hard_for_all"
    }
  ],
  "agents": [
    {
      "agent_name": "claude-3",
      "avg_score": 0.88,
      "pass_rate": 0.93,
      "total_tokens": 15000,
      "total_cost": 0.45,
      "rank": 1
    },
    {
      "agent_name": "gpt-4",
      "avg_score": 0.82,
      "pass_rate": 0.87,
      "total_tokens": 18000,
      "total_cost": 0.54,
      "rank": 2
    }
  ],
  "total_tests": 25,
  "total_agents": 2,
  "limit": 20,
  "offset": 0
}
```

**Difficulty Levels**:
| Level | Avg Score |
|-------|-----------|
| `easy` | ≥ 80% |
| `medium` | 60-79% |
| `hard` | 40-59% |
| `very_hard` | < 40% |

**Patterns**:
- `hard_for_all` — All agents score < 40% or pass rate ≤ 20%
- `easy` — All agents pass rate ≥ 80% and avg score ≥ 80%
- `high_variance` — Score range ≥ 40% between agents

---

## Timeline

### Get Timeline Events

Get execution events for a single agent with relative timing.

```http
GET /api/timeline/events
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `suite_name` | string | Yes | - | Test suite name |
| `test_id` | string | Yes | - | Test identifier |
| `agent_name` | string | Yes | - | Agent name |
| `event_types` | string[] | No | all | Filter by event types |
| `limit` | int | No | 1000 | Max events (max 1000) |
| `offset` | int | No | 0 | Pagination offset |

**Example**:
```bash
# All events
curl "http://localhost:8000/api/timeline/events?\
suite_name=demo_suite&\
test_id=test-001&\
agent_name=gpt-4"

# Only tool calls and LLM requests
curl "http://localhost:8000/api/timeline/events?\
suite_name=demo_suite&\
test_id=test-001&\
agent_name=gpt-4&\
event_types=tool_call&\
event_types=llm_request"
```

**Response** (`TimelineEventsResponse`):
```json
{
  "suite_name": "demo_suite",
  "test_id": "test-001",
  "test_name": "Basic file creation",
  "agent_name": "gpt-4",
  "total_events": 15,
  "events": [
    {
      "sequence": 1,
      "timestamp": "2025-01-26T10:00:00.000Z",
      "event_type": "llm_request",
      "summary": "LLM request: gpt-4 (500 tokens)",
      "data": {
        "model": "gpt-4",
        "input_tokens": 200,
        "output_tokens": 300,
        "duration_ms": 1500
      },
      "relative_time_ms": 0.0,
      "duration_ms": 1500.0
    },
    {
      "sequence": 2,
      "timestamp": "2025-01-26T10:00:01.500Z",
      "event_type": "reasoning",
      "summary": "Planning file creation...",
      "data": {
        "thought": "I need to create a file named output.txt",
        "step": "planning"
      },
      "relative_time_ms": 1500.0,
      "duration_ms": null
    },
    {
      "sequence": 3,
      "timestamp": "2025-01-26T10:00:02.000Z",
      "event_type": "tool_call",
      "summary": "Tool call: file_write (success)",
      "data": {
        "tool": "file_write",
        "status": "success",
        "args": {"path": "output.txt", "content": "Hello"},
        "result": {"bytes_written": 5},
        "duration_ms": 50
      },
      "relative_time_ms": 2000.0,
      "duration_ms": 50.0
    }
  ],
  "total_duration_ms": 12500.0,
  "execution_id": 42
}
```

---

### Compare Agent Timelines

Compare timelines of 2-3 agents on the same test, aligned by start time.

```http
GET /api/timeline/compare
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `suite_name` | string | Yes | - | Test suite name |
| `test_id` | string | Yes | - | Test identifier |
| `agents` | string[] | Yes | - | Agent names (2-3) |
| `event_types` | string[] | No | all | Filter by event types |

**Example**:
```bash
curl "http://localhost:8000/api/timeline/compare?\
suite_name=demo_suite&\
test_id=test-001&\
agents=gpt-4&\
agents=claude-3"
```

**Response** (`MultiTimelineResponse`):
```json
{
  "suite_name": "demo_suite",
  "test_id": "test-001",
  "test_name": "Basic file creation",
  "timelines": [
    {
      "agent_name": "gpt-4",
      "test_execution_id": 42,
      "start_time": "2025-01-26T10:00:00.000Z",
      "total_duration_ms": 12500.0,
      "events": [
        {
          "sequence": 1,
          "timestamp": "2025-01-26T10:00:00.000Z",
          "event_type": "llm_request",
          "summary": "LLM request: gpt-4 (500 tokens)",
          "data": {...},
          "relative_time_ms": 0.0,
          "duration_ms": 1500.0
        },
        ...
      ]
    },
    {
      "agent_name": "claude-3",
      "test_execution_id": 43,
      "start_time": "2025-01-26T10:00:00.000Z",
      "total_duration_ms": 8200.0,
      "events": [...]
    }
  ]
}
```

---

## Trends

### Suite Trends

Get historical trends for a test suite.

```http
GET /api/trends/suite
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `suite_name` | string | Yes | - | Test suite name |
| `agent_name` | string | No | all | Filter by agent |
| `metric` | string | No | success_rate | Metric: `success_rate`, `score`, `duration` |
| `limit` | int | No | 30 | Max data points |

**Example**:
```bash
curl "http://localhost:8000/api/trends/suite?\
suite_name=demo_suite&\
agent_name=gpt-4&\
metric=score&\
limit=30"
```

### Test Trends

Get historical trends for a specific test.

```http
GET /api/trends/test
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `suite_name` | string | Yes | - | Test suite name |
| `test_id` | string | Yes | - | Test identifier |
| `agent_name` | string | No | all | Filter by agent |
| `metric` | string | No | score | Metric: `score`, `duration`, `success_rate` |
| `limit` | int | No | 30 | Max data points |

---

## Agents

### List Agents

```http
GET /api/agents
```

**Response**:
```json
[
  {
    "id": 1,
    "name": "gpt-4",
    "agent_type": "http",
    "config": {"endpoint": "https://api.openai.com"},
    "description": "OpenAI GPT-4 agent",
    "created_at": "2025-01-20T10:00:00Z",
    "updated_at": "2025-01-20T10:00:00Z"
  }
]
```

### Create Agent

```http
POST /api/agents
Content-Type: application/json

{
  "name": "my-agent",
  "agent_type": "http",
  "config": {"endpoint": "http://localhost:9000"},
  "description": "My custom agent"
}
```

### Get Agent

```http
GET /api/agents/{agent_id}
```

### Update Agent

```http
PATCH /api/agents/{agent_id}
Content-Type: application/json

{
  "description": "Updated description"
}
```

### Delete Agent (Admin)

```http
DELETE /api/agents/{agent_id}
```

---

## Suite Executions

### List Suite Executions

```http
GET /api/suites
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `suite_name` | string | No | - | Filter by suite name |
| `agent_id` | int | No | - | Filter by agent ID |
| `limit` | int | No | 50 | Max results (max 100) |
| `offset` | int | No | 0 | Pagination offset |

### Get Suite Execution

```http
GET /api/suites/{execution_id}
```

### List Suite Names

```http
GET /api/suites/names/list
```

---

## Test Executions

### List Test Executions

```http
GET /api/tests
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `suite_execution_id` | int | No | - | Filter by suite execution |
| `test_id` | string | No | - | Filter by test ID |
| `success` | bool | No | - | Filter by success status |
| `limit` | int | No | 50 | Max results (max 100) |
| `offset` | int | No | 0 | Pagination offset |

### Get Test Execution

```http
GET /api/tests/{execution_id}
```

Returns detailed execution with runs, evaluations, and score components.

---

## Dashboard Summary

### Get Summary

```http
GET /api/dashboard/summary
```

**Response**:
```json
{
  "total_agents": 5,
  "total_suites": 12,
  "total_executions": 150,
  "recent_success_rate": 0.87,
  "recent_avg_score": 0.82,
  "recent_executions": [...]
}
```

---

## Cost Analytics

### Get Cost Summary

Get aggregated cost data with breakdowns by provider, model, and agent.

```http
GET /api/costs
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `start_date` | datetime | No | - | Filter by start date |
| `end_date` | datetime | No | - | Filter by end date |
| `provider` | string | No | - | Filter by provider |
| `model` | string | No | - | Filter by model |
| `agent_name` | string | No | - | Filter by agent name |

**Example**:
```bash
curl "http://localhost:8000/api/costs?\
start_date=2025-01-01T00:00:00&\
end_date=2025-01-31T23:59:59"
```

**Response** (`CostSummaryResponse`):
```json
{
  "total_cost": 125.45,
  "total_input_tokens": 2500000,
  "total_output_tokens": 1200000,
  "total_records": 1500,
  "by_provider": [
    {
      "name": "openai",
      "total_cost": 85.20,
      "total_input_tokens": 1800000,
      "total_output_tokens": 800000,
      "record_count": 1000,
      "percentage": 67.9
    },
    {
      "name": "anthropic",
      "total_cost": 40.25,
      "total_input_tokens": 700000,
      "total_output_tokens": 400000,
      "record_count": 500,
      "percentage": 32.1
    }
  ],
  "by_model": [
    {
      "name": "gpt-4",
      "total_cost": 60.00,
      "total_input_tokens": 1000000,
      "total_output_tokens": 500000,
      "record_count": 600,
      "percentage": 47.8
    }
  ],
  "by_agent": [
    {
      "name": "agent-one",
      "total_cost": 75.00,
      "total_input_tokens": 1500000,
      "total_output_tokens": 700000,
      "record_count": 900,
      "percentage": 59.8
    }
  ],
  "daily_trend": [
    {
      "date": "2025-01-15",
      "total_cost": 8.50,
      "total_tokens": 250000,
      "record_count": 100
    }
  ]
}
```

---

### List Cost Records

Get detailed cost records with filtering and pagination.

```http
GET /api/costs/records
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `start_date` | datetime | No | - | Filter by start date |
| `end_date` | datetime | No | - | Filter by end date |
| `provider` | string | No | - | Filter by provider |
| `model` | string | No | - | Filter by model |
| `agent_name` | string | No | - | Filter by agent name |
| `suite_id` | string | No | - | Filter by suite ID |
| `test_id` | string | No | - | Filter by test ID |
| `limit` | int | No | 50 | Max records (max 100) |
| `offset` | int | No | 0 | Pagination offset |

**Response** (`CostRecordList`):
```json
{
  "total": 150,
  "items": [
    {
      "id": 1,
      "timestamp": "2025-01-26T10:00:00Z",
      "provider": "openai",
      "model": "gpt-4",
      "input_tokens": 1000,
      "output_tokens": 500,
      "cost_usd": 0.045,
      "test_id": "test-001",
      "suite_id": "suite-001",
      "agent_name": "agent-one",
      "metadata": null
    }
  ],
  "limit": 50,
  "offset": 0
}
```

---

### Cost Breakdown Endpoints

#### By Provider
```http
GET /api/costs/by-provider
```

#### By Model
```http
GET /api/costs/by-model
```

#### By Agent
```http
GET /api/costs/by-agent
```

All return `list[CostBreakdownItem]` with the same structure.

---

### Get Cost Trend

Get daily cost trend over time.

```http
GET /api/costs/trend
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `start_date` | datetime | No | - | Filter by start date |
| `end_date` | datetime | No | - | Filter by end date |
| `provider` | string | No | - | Filter by provider |
| `model` | string | No | - | Filter by model |
| `agent_name` | string | No | - | Filter by agent name |

**Response** (`list[CostTrendPoint]`):
```json
[
  {
    "date": "2025-01-25",
    "total_cost": 8.50,
    "total_tokens": 250000,
    "record_count": 100
  },
  {
    "date": "2025-01-26",
    "total_cost": 12.30,
    "total_tokens": 350000,
    "record_count": 150
  }
]
```

---

## Budget Management

### List Budgets

Get all budgets with optional filtering and usage information.

```http
GET /api/budgets
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `period` | string | No | - | Filter by period: `daily`, `weekly`, `monthly` |
| `is_active` | bool | No | - | Filter by active status |
| `include_usage` | bool | No | true | Include current usage data |

**Response** (`BudgetList`):
```json
{
  "items": [
    {
      "id": 1,
      "name": "daily-budget",
      "period": "daily",
      "limit_usd": 100.00,
      "alert_threshold": 0.8,
      "scope": null,
      "alert_channels": ["slack"],
      "description": "Daily cost limit",
      "is_active": true,
      "created_at": "2025-01-01T00:00:00Z",
      "updated_at": "2025-01-01T00:00:00Z",
      "usage": {
        "budget_id": 1,
        "budget_name": "daily-budget",
        "period": "daily",
        "period_start": "2025-01-26T00:00:00Z",
        "spent": 45.50,
        "limit": 100.00,
        "remaining": 54.50,
        "percentage": 0.455,
        "is_over_threshold": false,
        "is_over_limit": false
      }
    }
  ],
  "total": 1
}
```

---

### Get Budget

```http
GET /api/budgets/{budget_id}
```

Returns `BudgetWithUsageResponse` with usage data.

---

### Create Budget (Auth Required)

```http
POST /api/budgets
Content-Type: application/json

{
  "name": "monthly-openai",
  "period": "monthly",
  "limit_usd": 500.00,
  "alert_threshold": 0.9,
  "scope": {"provider": "openai"},
  "alert_channels": ["slack", "email"],
  "description": "Monthly budget for OpenAI costs"
}
```

**Response**: `BudgetResponse` (201 Created)

---

### Update Budget (Auth Required)

```http
PUT /api/budgets/{budget_id}
Content-Type: application/json

{
  "limit_usd": 750.00,
  "is_active": true
}
```

All fields are optional. Only provided fields are updated.

---

### Delete Budget (Auth Required)

```http
DELETE /api/budgets/{budget_id}
```

Returns 204 No Content on success.

---

### Get Budget Usage

Get current usage for a specific budget.

```http
GET /api/budgets/{budget_id}/usage
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `reference_date` | datetime | No | now | Reference date for calculation |

**Response** (`BudgetUsageResponse`):
```json
{
  "budget_id": 1,
  "budget_name": "daily-budget",
  "period": "daily",
  "period_start": "2025-01-26T00:00:00Z",
  "spent": 45.50,
  "limit": 100.00,
  "remaining": 54.50,
  "percentage": 0.455,
  "is_over_threshold": false,
  "is_over_limit": false
}
```

---

### Check All Budgets

Check usage for all active budgets.

```http
GET /api/budgets/status/all
```

**Response**: `list[BudgetUsageResponse]`

---

## Error Responses

All endpoints return standard error responses:

```json
{
  "detail": "Error message description"
}
```

**HTTP Status Codes**:

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 204 | No Content (deleted) |
| 400 | Bad Request (validation error) |
| 401 | Unauthorized (auth required) |
| 403 | Forbidden (insufficient permissions) |
| 404 | Not Found |
| 422 | Unprocessable Entity (validation error) |
| 500 | Internal Server Error |

---

## Python Client Example

```python
import httpx

BASE_URL = "http://localhost:8000/api"

async def compare_agents():
    """Compare two agents on a specific test."""
    async with httpx.AsyncClient() as client:
        # Side-by-side comparison
        response = await client.get(
            f"{BASE_URL}/compare/side-by-side",
            params={
                "suite_name": "demo_suite",
                "test_id": "test-001",
                "agents": ["gpt-4", "claude-3"],
            },
        )
        response.raise_for_status()
        data = response.json()

        for agent in data["agents"]:
            print(f"\n{agent['agent_name']}:")
            print(f"  Score: {agent['score']}")
            print(f"  Duration: {agent['duration_seconds']}s")
            print(f"  Tokens: {agent['total_tokens']}")
            print(f"  Events: {len(agent['events'])}")


async def get_leaderboard():
    """Get leaderboard matrix."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/leaderboard/matrix",
            params={"suite_name": "demo_suite"},
        )
        response.raise_for_status()
        data = response.json()

        print("Agent Rankings:")
        for agent in data["agents"]:
            print(f"  #{agent['rank']} {agent['agent_name']}: "
                  f"score={agent['avg_score']:.2f}, "
                  f"pass_rate={agent['pass_rate']:.0%}")


async def get_timeline():
    """Get execution timeline for an agent."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/timeline/events",
            params={
                "suite_name": "demo_suite",
                "test_id": "test-001",
                "agent_name": "gpt-4",
            },
        )
        response.raise_for_status()
        data = response.json()

        print(f"Timeline for {data['agent_name']} on {data['test_name']}:")
        print(f"Total duration: {data['total_duration_ms']:.0f}ms")

        for event in data["events"]:
            print(f"  [{event['relative_time_ms']:6.0f}ms] "
                  f"{event['event_type']}: {event['summary']}")
```

---

## OpenAPI Schema

The complete OpenAPI schema is available at:

```
http://localhost:8000/openapi.json
http://localhost:8000/docs      # Swagger UI
http://localhost:8000/redoc     # ReDoc
```

---

## Performance Notes

- **Caching**: Leaderboard queries are cached for improved performance
- **Pagination**: All list endpoints support pagination via `limit` and `offset`
- **Event Limits**: Timeline endpoints limit to 1000 events per request
- **Bulk Queries**: Leaderboard uses optimized bulk queries instead of N+1

---

## Additional API Documentation

The following features have dedicated API reference documentation:

- **[WebSocket Real-Time Updates](../guides/websocket-guide.md)** — Real-time test progress, events, and logs via WebSocket
- **[Test Suite Marketplace](marketplace-api.md)** — Publishing, discovering, and installing test suites
- **[Public Leaderboard](public-leaderboard-api.md)** — Agent rankings, benchmark categories, and result publishing

---

## See Also

- [API Reference](api-reference.md) — Python API reference
- [RBAC Guide](../guides/rbac-guide.md) — Role-based access control
- [Architecture](../03-architecture.md) — System architecture
- [Usage Guide](../guides/usage.md) — Common workflows
