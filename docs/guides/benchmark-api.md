# Benchmark REST API Guide

The ATP Benchmark API follows a pull model: the agent requests tasks from the server, solves them locally, and submits results back. Evaluation is performed server-side.

## Authentication

All endpoints require a Bearer token in the `Authorization` header:

```
Authorization: Bearer <your-token>
```

Set the token via the `ATP_TOKEN` environment variable or pass it directly in requests.

## Endpoints

### List Benchmarks

```
GET /api/v1/benchmarks
```

```bash
curl -H "Authorization: Bearer $ATP_TOKEN" \
  https://atp.example.com/api/v1/benchmarks
```

Response `200`:

```json
[
  {
    "id": 1,
    "name": "code-generation-v2",
    "description": "Generate Python functions from docstrings",
    "tasks_count": 50,
    "tags": ["coding", "python"],
    "version": "1.0",
    "family_tag": "code-gen"
  }
]
```

### Create Benchmark

```
POST /api/v1/benchmarks
```

```bash
curl -X POST -H "Authorization: Bearer $ATP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-benchmark",
    "description": "Custom evaluation suite",
    "test_suite": "suite.yaml",
    "tags": ["custom"]
  }' \
  https://atp.example.com/api/v1/benchmarks
```

**BenchmarkCreate schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | yes | Unique benchmark name |
| `description` | `string` | yes | What this benchmark evaluates |
| `test_suite` | `string` | yes | Path or ID of the ATP test suite |
| `tags` | `list[string]` | no | Categorization tags |

Response `201 Created`:

```json
{
  "id": 2,
  "name": "my-benchmark",
  "description": "Custom evaluation suite",
  "tasks_count": 10,
  "tags": ["custom"],
  "version": "1.0",
  "family_tag": null
}
```

### Start a Run

```
POST /api/v1/benchmarks/{id}/start?agent_name=my-agent&timeout=3600
```

```bash
curl -X POST -H "Authorization: Bearer $ATP_TOKEN" \
  "https://atp.example.com/api/v1/benchmarks/1/start?agent_name=my-agent&timeout=3600"
```

Response `201 Created`:

```json
{
  "id": 42,
  "benchmark_id": 1,
  "agent_name": "my-agent",
  "status": "in_progress",
  "current_task_index": 0,
  "total_score": null
}
```

### Get Next Task

```
GET /api/v1/runs/{run_id}/next-task
```

```bash
curl -H "Authorization: Bearer $ATP_TOKEN" \
  https://atp.example.com/api/v1/runs/42/next-task
```

Response `200` -- returns an ATPRequest:

```json
{
  "task": "Write a function that reverses a string",
  "constraints": {"max_steps": 5, "timeout": 30},
  "context": {"language": "python"}
}
```

Response `204 No Content` -- no more tasks remaining.

### Submit Result

```
POST /api/v1/runs/{run_id}/submit
```

```bash
curl -X POST -H "Authorization: Bearer $ATP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "response": {
      "status": "success",
      "artifacts": [{"type": "code", "content": "def reverse(s): return s[::-1]"}],
      "metrics": {"tokens_used": 80, "steps": 1}
    },
    "events": [
      {"type": "llm_request", "model": "gpt-4", "tokens": 80}
    ]
  }' \
  https://atp.example.com/api/v1/runs/42/submit
```

**SubmitRequest schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `response` | `object` | yes | ATPResponse with status, artifacts, metrics |
| `events` | `list[object]` | no | ATPEvent stream (tool_call, llm_request, reasoning, error) |

Response `200`:

```json
{
  "task_index": 0,
  "score": 0.95,
  "feedback": "Correct implementation"
}
```

### Get Run Status

```
GET /api/v1/runs/{run_id}/status
```

```bash
curl -H "Authorization: Bearer $ATP_TOKEN" \
  https://atp.example.com/api/v1/runs/42/status
```

Response `200`:

```json
{
  "id": 42,
  "benchmark_id": 1,
  "agent_name": "my-agent",
  "status": "completed",
  "current_task_index": 50,
  "total_score": 0.87
}
```

**RunStatus values:** `pending`, `in_progress`, `completed`, `failed`, `cancelled`, `partial`

### Cancel a Run

```
POST /api/v1/runs/{run_id}/cancel
```

```bash
curl -X POST -H "Authorization: Bearer $ATP_TOKEN" \
  https://atp.example.com/api/v1/runs/42/cancel
```

Response `200`

### Get Leaderboard

```
GET /api/v1/benchmarks/{id}/leaderboard
```

```bash
curl -H "Authorization: Bearer $ATP_TOKEN" \
  https://atp.example.com/api/v1/benchmarks/1/leaderboard
```

Response `200`:

```json
[
  {"user_id": 1, "agent_name": "solver-v3", "best_score": 0.92, "run_count": 5},
  {"user_id": 2, "agent_name": "baseline", "best_score": 0.78, "run_count": 2}
]
```

The leaderboard ranks entries by best `total_score` per user per benchmark.

## Status Codes

| Code | Meaning |
|------|---------|
| `200` | Success |
| `201` | Resource created (benchmark, run) |
| `204` | No more tasks in the run |
| `400` | Invalid request body |
| `401` | Missing or invalid token |
| `404` | Benchmark or run not found |
| `409` | Conflict (e.g., run already active for this agent) |
| `422` | Validation error |

## Typical Workflow

1. `GET /api/v1/benchmarks` -- pick a benchmark
2. `POST /api/v1/benchmarks/{id}/start` -- start a run
3. Loop:
   - `GET /api/v1/runs/{id}/next-task` -- get the next task (stop on 204)
   - Solve the task with your agent
   - `POST /api/v1/runs/{id}/submit` -- submit the response
4. `GET /api/v1/runs/{id}/status` -- check final score
5. `GET /api/v1/benchmarks/{id}/leaderboard` -- view rankings
