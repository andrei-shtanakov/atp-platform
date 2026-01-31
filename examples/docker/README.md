# ATP Container Examples

This directory contains examples for running ATP tests with containerized agents.
Supports both **Docker** and **Podman** runtimes.

## Quick Start

```bash
# Build and run tests (auto-detects Docker or Podman)
./examples/docker/run_tests.sh --build

# Force specific runtime
./examples/docker/run_tests.sh --build --runtime podman
./examples/docker/run_tests.sh --build --runtime docker

# Or step by step with Docker:
docker build -t atp-demo-agent:latest -f examples/docker/Dockerfile.agent examples/docker/

# Or with Podman:
podman build -t atp-demo-agent:latest -f examples/docker/Dockerfile.agent examples/docker/

# Run tests
uv run atp test examples/test_suites/docker_agent_test.yaml \
  --adapter=container \
  --adapter-config='image=atp-demo-agent:latest'

# Force Podman runtime
uv run atp test examples/test_suites/docker_agent_test.yaml \
  --adapter=container \
  --adapter-config='image=atp-demo-agent:latest' \
  --adapter-config='runtime=podman'
```

## Files Overview

| File | Description |
|------|-------------|
| `docker-compose.yml` | Full setup with MCP server, agents, and dashboard |
| `Dockerfile.agent` | Simple demo agent (no API keys needed) |
| `Dockerfile.openai-agent` | OpenAI-powered agent with MCP support |
| `Dockerfile.mcp-server` | Mock MCP server for testing |
| `Dockerfile.atp` | ATP test runner |
| `agent.py` | Demo agent source code |
| `openai_agent.py` | OpenAI agent source code |
| `run_tests.sh` | Helper script to build and run tests |

## Using Compose (Docker/Podman)

### Start all services

```bash
cd examples/docker

# With Docker Compose (v2)
docker compose up -d mcp-server dashboard
docker compose run --rm atp-test
docker compose down

# With Podman Compose
podman compose up -d mcp-server dashboard
podman compose run --rm atp-test
podman compose down

# With legacy docker-compose
docker-compose up -d mcp-server dashboard
docker-compose run --rm atp-test
docker-compose down

# View dashboard
open http://localhost:8080
```

### Run with OpenAI agent

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Build images (Docker)
docker compose build

# Or with Podman
podman compose build

# Run tests with OpenAI agent
docker compose run --rm atp-test \
  atp test /tests/mcp_connection_test.yaml \
  --adapter=container \
  --adapter-config='image=atp-openai-agent:latest' \
  --adapter-config='environment={"OPENAI_API_KEY": "'$OPENAI_API_KEY'", "MCP_SERVER_URL": "http://mcp-server:9876"}'
```

## Container Adapter Configuration

The `container` adapter supports these options:

```yaml
adapter: container
adapter_config:
  # Required
  image: "my-agent:latest"         # Container image name

  # Runtime (optional)
  runtime: "auto"                  # "auto", "docker", or "podman"
                                   # auto = detect (prefers podman if both available)

  # Resources
  resources:
    memory: "2g"                   # Memory limit
    cpu: "1"                       # CPU limit

  # Network
  network: "none"                  # none, host, bridge, or network name

  # Environment
  environment:
    API_KEY: "${API_KEY}"
    DEBUG: "true"

  # Volumes (host:container)
  volumes:
    "/data/input": "/app/input"
    "/data/output": "/app/output"

  # Working directory
  working_dir: "/app"

  # Security
  auto_remove: true                # Remove container after run
  read_only_root: false           # Read-only filesystem
  no_new_privileges: true         # No privilege escalation
  cap_drop: ["ALL"]               # Drop all capabilities
```

## Creating Your Own Agent

Your agent must follow the ATP Protocol:

1. **Read** ATP Request from `stdin` (JSON)
2. **Write** ATP Events to `stderr` (JSONL, optional)
3. **Write** ATP Response to `stdout` (JSON)

### Minimal Agent Example

```python
#!/usr/bin/env python3
import json
import sys

# Read request from stdin
request = json.loads(sys.stdin.read())
task = request.get("task", {})

# Do your work here...
result = process_task(task["description"])

# Write response to stdout
response = {
    "version": "1.0",
    "task_id": request.get("task_id", "unknown"),
    "status": "completed",
    "artifacts": [
        {"type": "text", "content": result}
    ]
}
print(json.dumps(response))
```

### Minimal Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY agent.py .
ENTRYPOINT ["python", "agent.py"]
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Host Machine                             │
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │  ATP Runner  │────>│ Docker API   │────>│   Agent      │ │
│  │  (atp test)  │     │              │     │  Container   │ │
│  └──────────────┘     └──────────────┘     └──────────────┘ │
│         │                                         │          │
│         │              stdin (JSON)               │          │
│         │ ───────────────────────────────────────>│          │
│         │                                         │          │
│         │              stdout (JSON)              │          │
│         │ <───────────────────────────────────────│          │
│         │                                         │          │
│         │              stderr (JSONL)             │          │
│         │ <───────────────────────────────────────│          │
│         │                                         │          │
│  ┌──────────────┐                         ┌──────────────┐  │
│  │  Dashboard   │                         │  MCP Server  │  │
│  │   :8080      │                         │    :9876     │  │
│  └──────────────┘                         └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Container runtime not found

```bash
# Check Docker is installed
docker --version
docker info

# Check Podman is installed
podman --version
podman info

# On macOS with Podman, you may need to start the machine
podman machine start
```

### Podman socket not running (Linux)

```bash
# Enable and start podman socket for rootless containers
systemctl --user enable podman.socket
systemctl --user start podman.socket

# Verify
podman info
```

### Image not found

```bash
# Build the image first
docker build -t atp-demo-agent:latest -f examples/docker/Dockerfile.agent examples/docker/

# Verify image exists
docker images | grep atp-demo-agent
```

### Container timeout

Increase the timeout in test suite or via CLI:

```bash
uv run atp test suite.yaml \
  --adapter=container \
  --adapter-config='image=my-agent:latest' \
  --adapter-config='timeout_seconds=120'
```

### Network issues

If your agent needs network access:

```bash
# Use bridge network
--adapter-config='network=bridge'

# Or connect to specific network
--adapter-config='network=my-network'
```

### Permission denied

Make sure your agent script is executable:

```dockerfile
RUN chmod +x agent.py
```
