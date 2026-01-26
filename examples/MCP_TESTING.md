# MCP Agent Testing Example

This example demonstrates how to test AI agents that connect to MCP (Model Context Protocol) servers.

## Files

| File | Description |
|------|-------------|
| `mock_mcp_server.py` | Mock MCP server for testing |
| `mcp_agent.py` | OpenAI-powered MCP agent |
| `mcp_simple_agent.py` | Simple MCP agent (no LLM required) |
| `test_suites/mcp_connection_test.yaml` | Test suite definition |
| `run_mcp_tests.sh` | Automation script |

## Quick Start

### 1. Test without OpenAI (Simple Agent)

```bash
# Terminal 1: Start mock MCP server
python examples/mock_mcp_server.py --port 9876

# Terminal 2: Run tests with simple agent
uv run atp test examples/test_suites/mcp_connection_test.yaml \
  --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["examples/mcp_simple_agent.py"]' \
  --adapter-config='inherit_environment=true' \
  --adapter-config='allowed_env_vars=["MCP_SERVER_URL"]' \
  -v
```

### 2. Test with OpenAI Agent

```bash
# Set API key
export OPENAI_API_KEY='sk-...'

# Run automated tests
./examples/run_mcp_tests.sh --runs 3 --output html
```

### 3. Generate Reports

```bash
# JSON report
./examples/run_mcp_tests.sh --output json
# Creates mcp_test_results.json

# JUnit XML (for CI/CD)
./examples/run_mcp_tests.sh --output junit
# Creates mcp_test_results.xml
```

## Test Cases

The test suite includes:

| Test ID | Description | Tags |
|---------|-------------|------|
| mcp-001 | List MCP tools | smoke, connection |
| mcp-002 | Get weather data | tool_call, weather |
| mcp-003 | Search and summarize | tool_call, search |
| mcp-004 | Multi-tool workflow | workflow, advanced |

## Mock MCP Server

The mock server provides:

- `get_weather` - Returns mock weather for cities
- `search_web` - Returns mock search results
- `read_file` - Returns mock file content
- `calculate` - Evaluates math expressions

### API Endpoints

```bash
# List tools
curl -X POST http://localhost:9876/tools/list -H "Content-Type: application/json" -d '{}'

# Call tool
curl -X POST http://localhost:9876/tools/call -H "Content-Type: application/json" \
  -d '{"tool": "get_weather", "arguments": {"location": "Moscow"}}'

# Health check
curl http://localhost:9876/health
```

## Creating Your Own Agent

To create an ATP-compatible agent:

1. **Read JSON from stdin** (ATP Request):
```json
{
  "task": {
    "description": "Your task description",
    "input_data": {...},
    "expected_artifacts": ["output.json"]
  },
  "constraints": {
    "max_steps": 20,
    "timeout_seconds": 120
  }
}
```

2. **Write JSON to stdout** (ATP Response):
```json
{
  "status": "completed",
  "message": "Task completed",
  "artifacts": ["output.json"],
  "events": [
    {
      "sequence": 1,
      "timestamp": "2025-01-26T10:00:00Z",
      "event_type": "tool_call",
      "payload": {"tool": "...", "status": "success"}
    }
  ]
}
```

## Run Script Options

```bash
./examples/run_mcp_tests.sh [options]

Options:
  --runs N         Number of runs per test (default: 3)
  --output FORMAT  Output: console, json, junit (default: console)
  --tags TAGS      Filter tests by tags (comma-separated)
  --verbose        Verbose output
  --keep-server    Don't stop MCP server after tests
  --port PORT      MCP server port (default: 9876)
```

## Example: Custom Test

Add to `test_suites/mcp_connection_test.yaml`:

```yaml
  - id: "mcp-custom"
    name: "My custom MCP test"
    tags: ["custom"]
    task:
      description: |
        Connect to MCP server and perform custom operation.
        Save result to custom_result.json.
      expected_artifacts:
        - "custom_result.json"
    constraints:
      max_steps: 10
      timeout_seconds: 60
    assertions:
      - type: "artifact_exists"
        config:
          path: "custom_result.json"
```

## Troubleshooting

### Port already in use
```bash
# Find and kill process on port
lsof -i :9876
kill -9 <PID>
```

### Agent not receiving environment variables
```bash
# Ensure inherit_environment is set
--adapter-config='inherit_environment=true'
--adapter-config='allowed_env_vars=["MCP_SERVER_URL","OPENAI_API_KEY"]'
```

### MCP server connection refused
```bash
# Check server is running
curl http://localhost:9876/health
```
