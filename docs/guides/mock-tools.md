# Mock Tools Guide

This guide explains how to use ATP's mock tools feature to test AI agents with controlled tool responses.

## Overview

The mock tools module provides:
- **MockToolServer**: A FastAPI-based server that simulates tool endpoints
- **YAML-based mock definitions**: Define tool responses declaratively in YAML
- **Pattern matching**: Match tool inputs to specific responses
- **Call recording**: Track all tool calls for assertions and debugging

## Quick Start

### 1. Create a Mock Definition

Create a YAML file defining your mock tools:

```yaml
# mocks/search_tools.yaml
name: search_tools
description: Mock tools for search functionality

default_delay_ms: 50  # Default delay for all tools

tools:
  - name: web_search
    description: Search the web for information
    responses:
      - when:
          type: contains
          field: query
          pattern: "python"
        then:
          output:
            results:
              - title: "Python Documentation"
                url: "https://docs.python.org"
              - title: "Learn Python"
                url: "https://learnpython.org"
          status: success

      - when:
          type: regex
          field: query
          pattern: "error|fail"
        then:
          error: "Search service unavailable"
          status: error
          delay_ms: 100

    default:
      output:
        results: []
      status: success

  - name: file_read
    description: Read file contents
    responses:
      - when:
          type: exact
          field: path
          pattern: "/etc/passwd"
        then:
          error: "Permission denied"
          status: error

    default:
      output:
        content: "mock file content"
      status: success
```

### 2. Start the Mock Server

```python
from atp.mock_tools import MockToolServer

# Create and configure server
server = MockToolServer()
server.load_definition("mocks/search_tools.yaml")

# Get the FastAPI app
app = server.get_app()

# Run with uvicorn
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8080)
```

### 3. Use in Tests

```python
import pytest
from atp.mock_tools import MockToolServer, MockDefinition, MockTool, MockResponse

@pytest.fixture
def mock_server():
    server = MockToolServer(record_calls=True)

    # Add tools programmatically
    server.add_tool(MockTool(
        name="calculator",
        default_response=MockResponse(output={"result": 42}),
    ))

    return server

async def test_agent_uses_calculator(mock_server):
    # Your test logic here
    from atp.mock_tools import ToolCall

    call = ToolCall(tool="calculator", input={"a": 10, "b": 32})
    response = await mock_server.call_tool(call)

    assert response.status == "success"
    assert response.output["result"] == 42

    # Check recorded calls
    records = mock_server.recorder.get_records(tool="calculator")
    assert len(records) == 1
```

## API Reference

### MockToolServer

The main server class that handles tool calls.

```python
server = MockToolServer(
    definition=None,      # Optional MockDefinition
    record_calls=True,    # Whether to record calls
)

# Load from YAML
server.load_definition("path/to/mocks.yaml")
server.load_definition_string(yaml_content)

# Add tools dynamically
server.add_tool(mock_tool)

# Execute calls
response = await server.call_tool(tool_call)

# Access recorder
records = server.recorder.get_records()
```

### Pattern Matching

Pattern matchers determine which response to return based on tool input.

#### Match Types

| Type | Description | Example |
|------|-------------|---------|
| `any` | Always matches | Default when no pattern specified |
| `exact` | Exact string match | `pattern: "specific value"` |
| `contains` | Substring match | `pattern: "search term"` |
| `regex` | Regular expression | `pattern: "^test-\\d+$"` |

#### Using Field Matching

```yaml
responses:
  - when:
      type: contains
      field: query       # Match specific field in input dict
      pattern: "python"
    then:
      output: {results: [...]}
```

### Call Recorder

Track and query tool calls for testing.

```python
recorder = server.recorder

# Get all records
all_records = recorder.get_records()

# Filter by tool
calculator_calls = recorder.get_records(tool="calculator")

# Filter by task
task_calls = recorder.get_records(task_id="task-001")

# Get count
total = recorder.get_call_count()
tool_count = recorder.get_call_count(tool="web_search")

# Clear records
cleared = recorder.clear()
```

### HTTP Endpoints

When running the server, these endpoints are available:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tools/call` | Execute a tool call |
| GET | `/tools` | List available tools |
| GET | `/tools/{name}` | Get tool details |
| GET | `/records` | List call records |
| DELETE | `/records` | Clear all records |
| GET | `/health` | Health check |

#### Tool Call Request

```json
POST /tools/call
{
  "tool": "web_search",
  "input": {"query": "python tutorials"},
  "task_id": "task-001"
}
```

#### Tool Call Response

```json
{
  "tool": "web_search",
  "status": "success",
  "output": {"results": [...]},
  "error": null,
  "duration_ms": 52.3
}
```

## Integration with ATP Protocol

Use mock tools with ATP Request context:

```python
from atp.protocol import ATPRequest, Task, Context

request = ATPRequest(
    task_id="test-001",
    task=Task(description="Search for Python tutorials"),
    context=Context(
        tools_endpoint="http://localhost:8080",  # Mock server URL
        workspace_path="/tmp/workspace",
    ),
)
```

## YAML Schema Reference

```yaml
name: string (required)
description: string (optional)
default_delay_ms: integer (optional, default: 0)

tools:
  - name: string (required)
    description: string (optional)

    responses:  # List of conditional responses
      - when:
          type: exact | contains | regex | any
          field: string (optional, for dict inputs)
          pattern: string (required except for 'any')
        then:
          output: any (optional)
          error: string (optional)
          status: success | error (default: success)
          delay_ms: integer (optional)

    default:  # Fallback response
      output: any
      error: string
      status: success | error
      delay_ms: integer
```

## Best Practices

1. **Use descriptive tool names** that match your agent's expected tools
2. **Test error cases** by adding error responses for edge cases
3. **Use delays** to simulate real-world latency
4. **Check recorded calls** to verify agent behavior
5. **Clear records** between tests to avoid interference
6. **Version your mock definitions** alongside your tests
