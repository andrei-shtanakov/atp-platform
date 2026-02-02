# Web Search Agent

A CLI agent that searches websites and extracts product data. Designed for testing with the ATP test site.

## Features

- Searches product catalogs by category and price
- Extracts company and contact information
- Supports both HTML scraping and JSON API
- Follows ATP Protocol (stdin JSON → stdout JSON, events on stderr)

## Quick Start

```bash
# Ensure test site is running (port 9876)
podman run -d -p 9876:9876 --name atp-test-site atp-test-site

# Run agent directly
echo '{"task_id": "t1", "task": {"description": "Find laptops under $1000"}}' \
  | TEST_SITE_URL=http://localhost:9876 uv run python examples/search_agent/agent.py
```

## Building Docker Image

```bash
# Build with Podman
podman build -t atp-search-agent examples/search_agent/

# Run in container
echo '{"task_id": "t1", "task": {"description": "Find laptops under $1000"}}' \
  | podman run -i --network=host atp-search-agent
```

## Supported Tasks

| Task Type | Example Description |
|-----------|---------------------|
| Product search | `"Find all laptops"`, `"Find laptops under $1000"` |
| Category filter | `"Find all accessories"` |
| Price filter | `"Find laptops over $1000"` |
| Company info | `"Find when the company was founded"` |
| Contact info | `"Get contact email"` |

## Running with ATP

```bash
# Run test suite
uv run atp test examples/test_suites/web_search.yaml \
  --adapter=cli \
  --adapter-config='command=uv' \
  --adapter-config='args=["run", "python", "examples/search_agent/agent.py"]' \
  --adapter-config='environment={"TEST_SITE_URL": "http://localhost:9876"}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_SITE_URL` | `http://localhost:9876` | Base URL of the test site |

## ATP Protocol

### Input (stdin)
```json
{
  "task_id": "search-001",
  "task": {
    "description": "Find all laptops with price under $1000",
    "expected_artifacts": ["laptops.json"]
  }
}
```

### Output (stdout)
```json
{
  "version": "1.0",
  "task_id": "search-001",
  "status": "completed",
  "artifacts": [
    {
      "type": "file",
      "path": "laptops.json",
      "content": "[{\"name\": \"Budget Laptop\", \"price\": 499.99}]",
      "content_type": "application/json"
    }
  ],
  "metrics": {
    "total_steps": 1,
    "wall_time_seconds": 0.05
  }
}
```

### Events (stderr)
```json
{"event_type": "progress", "payload": {"message": "Starting search", "percentage": 0}}
{"event_type": "tool_call", "payload": {"tool": "http_get", "status": "completed"}}
{"event_type": "progress", "payload": {"message": "Found 3 products", "percentage": 100}}
```

## Files

- `agent.py` - Main agent code
- `Dockerfile` - Container build file
- `requirements.txt` - Python dependencies (httpx, beautifulsoup4)
