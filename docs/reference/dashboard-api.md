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

## See Also

- [API Reference](api-reference.md) — Python API reference
- [Architecture](../03-architecture.md) — System architecture
- [Usage Guide](../guides/usage.md) — Common workflows
