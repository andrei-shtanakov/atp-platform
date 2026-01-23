# Adapter Configuration Reference

This document describes how to configure agent adapters in ATP test suites.

## Overview

Adapters are the bridge between ATP Protocol and your agent implementation. They translate ATP requests into agent-specific formats and normalize responses back to ATP Protocol.

**Current Status**: MVP phase - adapters are defined in test suites but not yet implemented. This document describes the planned configuration format.

## Adapter Types

ATP supports multiple adapter types:

- **HTTP** - REST API agents
- **Docker** - Containerized agents
- **CLI** - Command-line agents
- **LangGraph** - LangGraph-based agents
- **CrewAI** - CrewAI-based agents
- **Custom** - Custom adapter implementations

## Agent Configuration Structure

```yaml
agents:
  - name: string           # Unique agent identifier
    type: string           # Adapter type
    config: dict           # Adapter-specific configuration
```

## HTTP Adapter

For agents exposed via HTTP REST API.

### Configuration

```yaml
agents:
  - name: "http-agent"
    type: "http"
    config:
      endpoint: string              # Required: API endpoint URL
      api_key: string               # Optional: API key for authentication
      timeout: int                  # Optional: Request timeout (seconds)
      headers: dict                 # Optional: Custom HTTP headers
      method: string                # Optional: HTTP method (default: POST)
      retry_count: int              # Optional: Number of retries (default: 3)
      retry_delay: int              # Optional: Delay between retries (seconds)
```

### Examples

#### Basic HTTP Agent

```yaml
agents:
  - name: "simple-agent"
    type: "http"
    config:
      endpoint: "http://localhost:8000/agent"
```

#### HTTP Agent with Authentication

```yaml
agents:
  - name: "authenticated-agent"
    type: "http"
    config:
      endpoint: "https://api.example.com/v1/agent"
      api_key: "${API_KEY}"
      timeout: 60
      headers:
        Content-Type: "application/json"
        X-Client-Version: "1.0"
```

#### HTTP Agent with Retry Logic

```yaml
agents:
  - name: "resilient-agent"
    type: "http"
    config:
      endpoint: "https://api.example.com/agent"
      api_key: "${API_KEY}"
      timeout: 120
      retry_count: 5
      retry_delay: 2
```

### Request Format

The HTTP adapter sends ATP requests as JSON:

```json
{
  "task": {
    "description": "Create a file named output.txt",
    "input_data": {},
    "expected_artifacts": ["output.txt"]
  },
  "constraints": {
    "max_steps": 10,
    "timeout_seconds": 60,
    "allowed_tools": ["file_write"]
  }
}
```

### Expected Response Format

The agent should return ATP response format:

```json
{
  "status": "success",
  "artifacts": ["output.txt"],
  "metrics": {
    "steps_taken": 3,
    "tokens_used": 1250,
    "cost_usd": 0.025
  },
  "events": [
    {
      "type": "tool_call",
      "timestamp": "2024-01-15T10:30:00Z",
      "data": {
        "tool": "file_write",
        "args": {"path": "output.txt", "content": "Hello"}
      }
    }
  ]
}
```

## Docker Adapter

For agents running in Docker containers.

### Configuration

```yaml
agents:
  - name: "docker-agent"
    type: "docker"
    config:
      image: string                 # Required: Docker image name
      tag: string                   # Optional: Image tag (default: latest)
      command: string               # Optional: Override container command
      environment: dict             # Optional: Environment variables
      volumes: list[string]         # Optional: Volume mounts
      network: string               # Optional: Docker network
      memory_limit: string          # Optional: Memory limit (e.g., "512m")
      cpu_limit: string             # Optional: CPU limit (e.g., "1.0")
      cleanup: bool                 # Optional: Remove container after run
```

### Examples

#### Basic Docker Agent

```yaml
agents:
  - name: "docker-agent"
    type: "docker"
    config:
      image: "my-agent"
      tag: "latest"
```

#### Docker Agent with Environment

```yaml
agents:
  - name: "configured-agent"
    type: "docker"
    config:
      image: "my-agent"
      tag: "v1.2.3"
      environment:
        API_KEY: "${API_KEY}"
        LOG_LEVEL: "debug"
        MODEL: "gpt-4"
      volumes:
        - "./data:/app/data"
        - "./output:/app/output"
```

#### Docker Agent with Resource Limits

```yaml
agents:
  - name: "constrained-agent"
    type: "docker"
    config:
      image: "my-agent"
      memory_limit: "1g"
      cpu_limit: "2.0"
      network: "agent-network"
      cleanup: true
```

## CLI Adapter

For command-line based agents.

### Configuration

```yaml
agents:
  - name: "cli-agent"
    type: "cli"
    config:
      command: string               # Required: Command to execute
      args: list[string]            # Optional: Command arguments
      cwd: string                   # Optional: Working directory
      environment: dict             # Optional: Environment variables
      timeout: int                  # Optional: Execution timeout
      shell: bool                   # Optional: Use shell (default: false)
```

### Examples

#### Basic CLI Agent

```yaml
agents:
  - name: "python-agent"
    type: "cli"
    config:
      command: "python"
      args: ["agent.py"]
      cwd: "./agents"
```

#### CLI Agent with Environment

```yaml
agents:
  - name: "configured-cli-agent"
    type: "cli"
    config:
      command: "./run_agent.sh"
      environment:
        API_KEY: "${API_KEY}"
        MODEL: "gpt-4"
        TIMEOUT: "300"
      timeout: 600
```

## LangGraph Adapter

For LangGraph-based agents.

### Configuration

```yaml
agents:
  - name: "langgraph-agent"
    type: "langgraph"
    config:
      graph_path: string            # Required: Path to graph module
      graph_name: string            # Optional: Graph variable name
      model: string                 # Optional: LLM model to use
      temperature: float            # Optional: Model temperature
      api_key: string               # Optional: LLM API key
      config: dict                  # Optional: Additional graph config
```

### Examples

#### Basic LangGraph Agent

```yaml
agents:
  - name: "langgraph-agent"
    type: "langgraph"
    config:
      graph_path: "./agents/my_graph.py"
      model: "gpt-4"
```

#### Configured LangGraph Agent

```yaml
agents:
  - name: "research-agent"
    type: "langgraph"
    config:
      graph_path: "./agents/research_graph.py"
      graph_name: "research_graph"
      model: "gpt-4-turbo"
      temperature: 0.7
      api_key: "${OPENAI_API_KEY}"
      config:
        max_iterations: 10
        tools: ["web_search", "file_write"]
```

## CrewAI Adapter

For CrewAI-based agents.

### Configuration

```yaml
agents:
  - name: "crewai-agent"
    type: "crewai"
    config:
      crew_path: string             # Required: Path to crew module
      crew_name: string             # Optional: Crew variable name
      model: string                 # Optional: LLM model to use
      api_key: string               # Optional: LLM API key
      config: dict                  # Optional: Additional crew config
```

### Examples

#### Basic CrewAI Agent

```yaml
agents:
  - name: "crew-agent"
    type: "crewai"
    config:
      crew_path: "./agents/my_crew.py"
      model: "gpt-4"
```

#### Multi-Agent Crew

```yaml
agents:
  - name: "research-crew"
    type: "crewai"
    config:
      crew_path: "./agents/research_crew.py"
      crew_name: "research_crew"
      model: "gpt-4"
      api_key: "${OPENAI_API_KEY}"
      config:
        verbose: true
        max_rpm: 10
```

## Custom Adapter

For custom adapter implementations.

### Configuration

```yaml
agents:
  - name: "custom-agent"
    type: "custom"
    config:
      adapter_class: string         # Required: Python class path
      config: dict                  # Optional: Custom configuration
```

### Examples

```yaml
agents:
  - name: "my-custom-agent"
    type: "custom"
    config:
      adapter_class: "my_package.adapters.CustomAdapter"
      config:
        param1: "value1"
        param2: 42
```

## Multiple Agents

Test suites can define multiple agents for comparison:

```yaml
test_suite: "agent_comparison"
version: "1.0"

agents:
  # Production agent
  - name: "prod-agent"
    type: "http"
    config:
      endpoint: "https://api.production.com/agent"
      api_key: "${PROD_API_KEY}"

  # Baseline agent for comparison
  - name: "baseline-agent"
    type: "http"
    config:
      endpoint: "https://api.baseline.com/agent"
      api_key: "${BASELINE_API_KEY}"

  # Development agent
  - name: "dev-agent"
    type: "docker"
    config:
      image: "agent-dev"
      tag: "latest"

tests:
  - id: "test-001"
    name: "Performance comparison test"
    # Test will run against all configured agents
```

## Environment Variables

All adapter configurations support environment variable substitution:

```yaml
agents:
  - name: "agent"
    type: "http"
    config:
      endpoint: "${API_ENDPOINT:http://localhost:8000}"
      api_key: "${API_KEY}"
      timeout: "${TIMEOUT:60}"
```

Load with variables:

```python
from atp.loader import TestLoader

loader = TestLoader(env={
    "API_ENDPOINT": "https://api.production.com",
    "API_KEY": "secret-key-123",
    "TIMEOUT": "120"
})
suite = loader.load_file("suite.yaml")
```

## Adapter Development

To create a custom adapter, implement the `AgentAdapter` interface:

```python
from atp.adapters.base import AgentAdapter
from atp.protocol import ATPRequest, ATPResponse

class MyCustomAdapter(AgentAdapter):
    """Custom adapter implementation."""

    def __init__(self, config: dict) -> None:
        """Initialize adapter with configuration."""
        self.config = config

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """Execute agent with ATP request."""
        # 1. Translate ATP request to agent format
        agent_request = self._translate_request(request)

        # 2. Execute agent
        agent_response = await self._run_agent(agent_request)

        # 3. Translate response to ATP format
        atp_response = self._translate_response(agent_response)

        return atp_response

    def _translate_request(self, request: ATPRequest) -> dict:
        """Convert ATP request to agent format."""
        # Implementation specific
        pass

    async def _run_agent(self, request: dict) -> dict:
        """Execute the agent."""
        # Implementation specific
        pass

    def _translate_response(self, response: dict) -> ATPResponse:
        """Convert agent response to ATP format."""
        # Implementation specific
        pass
```

Register custom adapter:

```python
from atp.core.registry import adapter_registry
from my_package.adapters import MyCustomAdapter

# Register adapter
adapter_registry.register("my_custom", MyCustomAdapter)

# Now can use in test suites
# type: "my_custom"
```

## Best Practices

### Security

1. **Never hardcode secrets** - Always use environment variables
```yaml
# Bad
config:
  api_key: "sk-abc123"

# Good
config:
  api_key: "${API_KEY}"
```

2. **Use HTTPS for production** - Encrypt API communications
```yaml
config:
  endpoint: "https://api.example.com"  # Not http://
```

3. **Limit resource access** - Use minimal permissions
```yaml
config:
  volumes:
    - "./data:/data:ro"  # Read-only when possible
```

### Performance

1. **Set appropriate timeouts** - Prevent hanging tests
```yaml
config:
  timeout: 300  # 5 minutes max
```

2. **Configure retries** - Handle transient failures
```yaml
config:
  retry_count: 3
  retry_delay: 2
```

3. **Resource limits** - Prevent resource exhaustion
```yaml
config:
  memory_limit: "2g"
  cpu_limit: "2.0"
```

### Configuration Management

1. **Use environment-specific configs** - Separate dev/staging/prod
```python
env_configs = {
    "dev": {"API_ENDPOINT": "http://localhost:8000"},
    "prod": {"API_ENDPOINT": "https://api.production.com"}
}
```

2. **Document required variables** - List env vars in README
```yaml
# Required environment variables:
# - API_KEY: Agent API key
# - API_ENDPOINT: Agent API endpoint
# - TIMEOUT: Request timeout in seconds
```

3. **Provide sensible defaults** - Make configuration optional
```yaml
config:
  endpoint: "${API_ENDPOINT:http://localhost:8000}"
  timeout: "${TIMEOUT:60}"
```

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common adapter issues.

## See Also

- [Test Format Reference](test-format.md) - Complete YAML format
- [Usage Guide](../guides/usage.md) - Loading and using test suites
- [Integration Guide](../06-integration.md) - Detailed integration patterns
