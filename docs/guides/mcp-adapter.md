# MCP (Model Context Protocol) Adapter Guide

This guide explains how to use the MCP adapter to test agents that communicate via the Model Context Protocol.

## Overview

The MCP (Model Context Protocol) is a standard protocol for AI tool integration that enables:

- **Tool Discovery**: Dynamically discover available tools from MCP servers
- **Tool Invocation**: Call tools with structured arguments
- **Resource Access**: Read files, data, and other resources
- **Prompt Templates**: Use predefined prompt templates
- **Event Streaming**: Receive real-time progress updates

The ATP MCP adapter supports both **stdio** (subprocess) and **SSE** (HTTP) transports.

## Quick Start

### Basic CLI Usage

```bash
# Run tests with MCP adapter using stdio transport
atp test suite.yaml --adapter=mcp \
  --adapter-config transport=stdio \
  --adapter-config command=npx \
  --adapter-config 'args=["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]'

# Run tests with MCP adapter using SSE transport
atp test suite.yaml --adapter=mcp \
  --adapter-config transport=sse \
  --adapter-config url=http://localhost:8080/mcp
```

### Python Usage

```python
from atp.adapters import MCPAdapter, MCPAdapterConfig
from atp.protocol import ATPRequest, Task

# Configure stdio transport
config = MCPAdapterConfig(
    transport="stdio",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
    timeout_seconds=60.0,
    startup_timeout=30.0,
)

# Create and initialize adapter
adapter = MCPAdapter(config)
await adapter.initialize()

# Check discovered tools
print(f"Discovered tools: {list(adapter.tools.keys())}")

# Execute a request
request = ATPRequest(
    task_id="test-001",
    task=Task(
        description="Read the configuration file",
        input_data={
            "tool": "read_file",
            "arguments": {"path": "/workspace/config.json"},
        },
    ),
)

response = await adapter.execute(request)
print(f"Status: {response.status}")

# Cleanup
await adapter.cleanup()
```

## Configuration

### Stdio Transport

Use stdio transport for MCP servers that run as subprocesses:

```yaml
agents:
  - name: filesystem-mcp
    type: mcp
    config:
      transport: "stdio"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
      working_dir: "/path/to/workspace"
      environment:
        MCP_ROOT: "/workspace"
        DEBUG: "true"
      inherit_environment: true
      allowed_env_vars: ["HOME", "PATH", "NODE_PATH"]
      startup_timeout: 30
      timeout_seconds: 120
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `transport` | string | `"stdio"` | Transport type |
| `command` | string | **required** | Command to run MCP server |
| `args` | list | `[]` | Command arguments |
| `working_dir` | string | `null` | Working directory for subprocess |
| `environment` | dict | `{}` | Environment variables to set |
| `inherit_environment` | bool | `true` | Inherit filtered parent environment |
| `allowed_env_vars` | list | `[]` | Additional env vars to allow |

### SSE Transport

Use SSE transport for remote MCP servers over HTTP:

```yaml
agents:
  - name: remote-mcp
    type: mcp
    config:
      transport: "sse"
      url: "https://mcp.example.com/sse"
      headers:
        Authorization: "Bearer ${MCP_API_KEY}"
        X-Client-ID: "atp-client"
      verify_ssl: true
      post_endpoint: "https://mcp.example.com/messages"
      startup_timeout: 10
      timeout_seconds: 90
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `transport` | string | **required** | Must be `"sse"` |
| `url` | string | **required** | SSE endpoint URL |
| `headers` | dict | `{}` | HTTP headers for requests |
| `verify_ssl` | bool | `true` | Verify SSL certificates |
| `post_endpoint` | string | `null` | Separate POST endpoint (if different from URL) |

### Common Options

These options apply to both transports:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `timeout_seconds` | float | `300.0` | Execution timeout |
| `startup_timeout` | float | `30.0` | Server startup timeout |
| `retry_count` | int | `0` | Number of retries on failure |
| `tools_filter` | list | `null` | Only expose these tools (null = all) |
| `resources_filter` | list | `null` | Only expose these resources (null = all) |
| `client_name` | string | `"atp"` | Client name for MCP handshake |
| `client_version` | string | `"1.0.0"` | Client version for MCP handshake |

## Test Suite Configuration

### Full Example

```yaml
test_suite: "mcp_tests"
version: "1.0"
description: "Tests for MCP-based agent"

defaults:
  timeout_seconds: 60
  runs_per_test: 1

agents:
  - name: filesystem-agent
    type: mcp
    config:
      transport: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
      startup_timeout: 30
      tools_filter:
        - read_file
        - write_file
        - list_directory

tests:
  - id: mcp-read-file
    name: "Read configuration file"
    tags: [mcp, file, read]
    task:
      description: "Read the application configuration"
      input_data:
        tool: read_file
        arguments:
          path: /workspace/config.json
    constraints:
      max_steps: 5
      timeout_seconds: 30
    assertions:
      - type: artifact_exists
        config:
          path: output

  - id: mcp-list-dir
    name: "List directory contents"
    tags: [mcp, directory]
    task:
      description: "List files in workspace"
      input_data:
        tool: list_directory
        arguments:
          path: /workspace
    constraints:
      timeout_seconds: 30

  - id: mcp-write-file
    name: "Write output file"
    tags: [mcp, file, write]
    task:
      description: "Write results to output file"
      input_data:
        tool: write_file
        arguments:
          path: /workspace/output.txt
          content: "Test results: PASSED"
    constraints:
      timeout_seconds: 30
```

### Input Data Format

When specifying `input_data` in task definitions, use these fields:

```yaml
input_data:
  tool: "tool_name"        # Specific tool to call
  arguments:               # Tool arguments
    arg1: value1
    arg2: value2
  resource: "uri://..."    # OR resource URI to read
```

If no `tool` or `resource` is specified, the adapter will:
1. Use the first available tool
2. Pass the task description as the `input` argument

## Tool Filtering

Filter which tools are exposed from the MCP server:

```yaml
config:
  tools_filter:
    - read_file
    - write_file
    # Only these tools will be available
```

This is useful for:
- Security: Limiting which tools tests can access
- Focus: Testing specific tool functionality
- Consistency: Ensuring tests don't depend on optional tools

## Event Streaming

The MCP adapter emits events during execution:

```python
async for event in adapter.stream_events(request):
    if isinstance(event, ATPEvent):
        if event.event_type == EventType.TOOL_CALL:
            print(f"Tool: {event.payload['tool']}")
            print(f"Status: {event.payload['status']}")
        elif event.event_type == EventType.PROGRESS:
            print(f"Progress: {event.payload['message']}")
    else:
        # Final response
        print(f"Result: {event.status}")
```

Event types emitted:
- `PROGRESS`: Connection established, operation started
- `TOOL_CALL`: Tool invocation start and completion
- `ERROR`: Error occurred (with recovery info)

## Health Checks

The adapter supports health checks:

```python
# Check if MCP server is healthy
is_healthy = await adapter.health_check()

if not is_healthy:
    # Attempt reconnection
    success = await adapter.reconnect()
```

## Error Handling

The adapter raises specific exceptions:

```python
from atp.adapters.exceptions import (
    AdapterConnectionError,
    AdapterTimeoutError,
    AdapterError,
)

try:
    response = await adapter.execute(request)
except AdapterConnectionError as e:
    print(f"Connection failed: {e}")
    # Server might not be running
except AdapterTimeoutError as e:
    print(f"Timeout after {e.timeout_seconds}s")
    # Operation took too long
except AdapterError as e:
    print(f"Adapter error: {e}")
    # General adapter issue
```

## Troubleshooting

### Common Issues

#### MCP Server Not Starting

**Symptoms**: `AdapterConnectionError` during initialization

**Possible causes**:
1. Command not found
2. Missing dependencies (e.g., npx not installed)
3. Working directory doesn't exist
4. Environment variables not set correctly

**Solutions**:
```yaml
config:
  # Verify command is correct
  command: "npx"  # Use full path if needed: "/usr/local/bin/npx"

  # Ensure working directory exists
  working_dir: "/existing/path"

  # Set required environment
  environment:
    PATH: "/usr/local/bin:/usr/bin"
    NODE_PATH: "/usr/local/lib/node_modules"
```

#### Tool Not Found

**Symptoms**: `AdapterError: Tool 'xyz' not found`

**Possible causes**:
1. Tool not provided by MCP server
2. Tool filtered out by `tools_filter`
3. Server didn't complete tool discovery

**Solutions**:
```python
# Check available tools after initialization
await adapter.initialize()
print("Available tools:", list(adapter.tools.keys()))

# Verify tool filter includes the tool
config = MCPAdapterConfig(
    tools_filter=["read_file", "xyz"],  # Include the tool
    ...
)
```

#### Timeout During Startup

**Symptoms**: `AdapterTimeoutError` during `initialize()`

**Possible causes**:
1. MCP server takes too long to start
2. Server is waiting for input
3. Network issues (SSE transport)

**Solutions**:
```yaml
config:
  # Increase startup timeout
  startup_timeout: 60  # Default is 30

  # For SSE, check network connectivity
  transport: sse
  url: "http://localhost:8080/mcp"
  verify_ssl: false  # For local development
```

#### Connection Refused (SSE)

**Symptoms**: `AdapterConnectionError` with "Connection refused"

**Possible causes**:
1. MCP server not running
2. Wrong URL or port
3. SSL certificate issues

**Solutions**:
```yaml
config:
  transport: sse
  url: "http://localhost:8080/mcp"  # Check URL
  verify_ssl: false  # Disable for self-signed certs
  headers:
    Authorization: "Bearer ${API_KEY}"  # Check auth
```

#### Empty Tool Discovery

**Symptoms**: `adapter.tools` is empty after initialization

**Possible causes**:
1. Server doesn't support `tools/list`
2. All tools filtered out
3. Server capability not enabled

**Solutions**:
```python
# Check server capabilities
await adapter.initialize()
print("Server info:", adapter.server_info)
print("Capabilities:", adapter.server_info.capabilities)

# Try without filters
config = MCPAdapterConfig(
    tools_filter=None,  # Don't filter
    ...
)
```

### Debug Mode

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Or just for MCP adapter
logging.getLogger("atp.adapters.mcp").setLevel(logging.DEBUG)
```

### Testing MCP Server Manually

Test your MCP server before using with ATP:

```bash
# Test stdio server
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"test","version":"1.0"},"capabilities":{}}}' | \
  npx -y @modelcontextprotocol/server-filesystem /workspace

# Test SSE server
curl -N -H "Accept: text/event-stream" http://localhost:8080/mcp
```

## Examples

### Testing a Filesystem MCP Server

```yaml
test_suite: "filesystem_mcp_tests"
version: "1.0"

agents:
  - name: fs-server
    type: mcp
    config:
      transport: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/test-workspace"]

tests:
  - id: create-and-read
    name: "Create and read file"
    task:
      description: |
        Create a file named 'test.txt' with content 'Hello, World!'
        Then read it back to verify.
      input_data:
        tool: write_file
        arguments:
          path: /tmp/test-workspace/test.txt
          content: "Hello, World!"
    assertions:
      - type: behavior
        config:
          check: no_errors
```

### Testing a Python MCP Server

```yaml
test_suite: "python_mcp_tests"
version: "1.0"

agents:
  - name: python-mcp
    type: mcp
    config:
      transport: stdio
      command: python
      args: ["-m", "my_mcp_server"]
      working_dir: "."
      environment:
        PYTHONPATH: "${PYTHONPATH}"
      inherit_environment: true

tests:
  - id: custom-tool
    name: "Call custom tool"
    task:
      description: "Execute custom business logic"
      input_data:
        tool: my_custom_tool
        arguments:
          input: "test data"
```

### Testing with Remote MCP Server

```yaml
test_suite: "remote_mcp_tests"
version: "1.0"

agents:
  - name: remote-mcp
    type: mcp
    config:
      transport: sse
      url: "${MCP_SERVER_URL}"
      headers:
        Authorization: "Bearer ${MCP_API_KEY}"
      verify_ssl: true
      startup_timeout: 10

tests:
  - id: api-call
    name: "Call remote API"
    task:
      description: "Execute remote operation"
      input_data:
        tool: api_operation
        arguments:
          endpoint: "/users"
```

## See Also

- [Adapter Development Guide](adapter-development.md)
- [Test Format Reference](../reference/test-format.md)
- [Configuration Reference](../reference/configuration.md)
- [MCP Protocol Specification](https://spec.modelcontextprotocol.io/)
