# ATP SDK Guide

The `atp-sdk` package provides a Python client for interacting with the ATP benchmark platform. It wraps the REST API into a convenient iterator-based interface for pulling tasks, submitting results, and viewing leaderboards.

## Installation

```bash
uv add atp-sdk
# or
pip install atp-sdk
```

## Quick Start

```python
from atp_sdk import ATPClient

client = ATPClient(platform_url="https://atp.example.com", token="my-token")

# List available benchmarks
for bm in client.list_benchmarks():
    print(f"{bm.name}: {bm.tasks_count} tasks (v{bm.version})")

# Run a benchmark
run = client.start_run("benchmark-42", agent_name="my-agent")
for task in run:
    response = solve_task(task)  # your agent logic
    run.submit(response)

# Check results
print(run.status())
print(run.leaderboard())

client.close()
```

## Authentication

The client reads credentials in this order:

1. `token` parameter passed to `ATPClient(...)`
2. `ATP_TOKEN` environment variable

```bash
export ATP_TOKEN="your-bearer-token"
```

```python
# Token is picked up automatically
client = ATPClient(platform_url="https://atp.example.com")
```

## ATPClient API

### Constructor

```python
ATPClient(
    platform_url: str = "http://localhost:8000",
    token: str | None = None,
)
```

The client supports context-manager usage:

```python
with ATPClient(platform_url="https://atp.example.com") as client:
    benchmarks = client.list_benchmarks()
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `list_benchmarks()` | `list[BenchmarkInfo]` | List all available benchmarks |
| `get_benchmark(id)` | `BenchmarkInfo` | Get details of a specific benchmark |
| `start_run(benchmark_id, agent_name, timeout)` | `BenchmarkRun` | Start a new benchmark run |
| `get_leaderboard(benchmark_id)` | `list[dict]` | Get the leaderboard for a benchmark |
| `close()` | `None` | Close the underlying HTTP connection |

### BenchmarkInfo Model

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Benchmark identifier |
| `name` | `str` | Human-readable name |
| `description` | `str` | What the benchmark tests |
| `tasks_count` | `int` | Number of tasks in the benchmark |
| `tags` | `list[str]` | Categorization tags |
| `version` | `str` | Benchmark version (default `"1.0"`) |
| `family_tag` | `str \| None` | Optional family grouping |

## BenchmarkRun API

`BenchmarkRun` is an iterator that pulls tasks from the platform one at a time. It stops when the server returns HTTP 204 (no more tasks).

### Iterator Pattern

```python
run = client.start_run("benchmark-42", agent_name="solver-v3")

for task in run:
    # task is an ATPRequest dict with keys like:
    #   "task", "constraints", "context"
    result = my_agent(task)
    run.submit(result)
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `submit(response, events=None)` | `dict` | Submit a task response with optional events |
| `status()` | `dict` | Get current run status (pending, in_progress, completed, failed, cancelled, partial) |
| `cancel()` | `None` | Cancel the run |
| `leaderboard()` | `list[dict]` | Get the leaderboard for this benchmark |

### Submitting Events

You can attach ATP events (tool calls, LLM requests, reasoning steps) alongside each response:

```python
events = [
    {"type": "tool_call", "name": "search", "input": {"q": "ATP protocol"}},
    {"type": "llm_request", "model": "gpt-4", "tokens": 1200},
]
run.submit(response, events=events)
```

## Full Example: Agent That Solves Benchmarks

```python
"""Minimal agent that runs all tasks in a benchmark."""

import os

from atp_sdk import ATPClient


def my_agent_logic(task: dict) -> dict:
    """Replace with your actual agent implementation."""
    return {
        "status": "success",
        "artifacts": [{"type": "text", "content": f"Answer for: {task['task']}"}],
        "metrics": {"tokens_used": 100, "steps": 1},
    }


def main() -> None:
    with ATPClient(
        platform_url=os.getenv("ATP_URL", "http://localhost:8000"),
    ) as client:
        benchmarks = client.list_benchmarks()
        print(f"Found {len(benchmarks)} benchmarks")

        target = benchmarks[0]
        print(f"Running: {target.name} ({target.tasks_count} tasks)")

        run = client.start_run(target.id, agent_name="demo-agent")

        for task in run:
            response = my_agent_logic(task)
            result = run.submit(response)
            print(f"  Score: {result.get('score', 'N/A')}")

        status = run.status()
        print(f"Final status: {status['status']}")
        print(f"Total score: {status.get('total_score', 'N/A')}")

        for entry in run.leaderboard():
            print(f"  {entry['agent_name']}: {entry['best_score']}")


if __name__ == "__main__":
    main()
```

## Error Handling

The SDK raises `httpx.HTTPStatusError` on non-2xx responses. Wrap calls in try/except for production use:

```python
import httpx

try:
    run = client.start_run("nonexistent")
except httpx.HTTPStatusError as e:
    if e.response.status_code == 404:
        print("Benchmark not found")
    elif e.response.status_code == 409:
        print("Run already in progress")
    else:
        raise
```
