# Container Setup Guide

This guide covers running ATP components in Docker and Podman containers.

## Prerequisites

You need one of the following container runtimes:

- **Docker** (with Docker Compose v2)
- **Podman** (with podman-compose)

### Installing Docker

```bash
# macOS
brew install docker docker-compose

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose-v2

# Fedora/RHEL
sudo dnf install docker docker-compose
```

### Installing Podman

```bash
# macOS
brew install podman podman-compose

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install podman podman-compose

# Fedora/RHEL (Podman is pre-installed)
sudo dnf install podman-compose
```

## Test Site Container

The ATP test site is a mock e-commerce application for testing web search agents.

### Starting the Test Site

From the project root:

```bash
# Using Docker Compose (v2)
docker compose -f tests/fixtures/test_site/docker-compose.yml up -d

# Using Podman Compose
podman-compose -f tests/fixtures/test_site/docker-compose.yml up -d

# Using legacy docker-compose
docker-compose -f tests/fixtures/test_site/docker-compose.yml up -d
```

### Verifying the Test Site

```bash
curl http://localhost:9876/health
# Expected: {"status":"ok","products_count":10}
```

### Stopping the Test Site

```bash
# Docker
docker compose -f tests/fixtures/test_site/docker-compose.yml down

# Podman
podman-compose -f tests/fixtures/test_site/docker-compose.yml down
```

### Building Manually

If you prefer to build and run manually:

```bash
cd tests/fixtures/test_site

# Docker
docker build -t atp-test-site .
docker run -d -p 9876:9876 --name atp-test-site atp-test-site

# Podman
podman build -t atp-test-site .
podman run -d -p 9876:9876 --name atp-test-site atp-test-site
```

## Running Agents in Containers

ATP supports running agents in containers via the Docker adapter.

### Docker Adapter Configuration

In your test suite YAML:

```yaml
agents:
  - name: "my-agent"
    type: "docker"
    config:
      image: "my-agent:latest"
      # Optional settings
      timeout: 300
      memory_limit: "1g"
      cpu_limit: 1.0
      environment:
        API_KEY: "${API_KEY}"
```

### Building an Agent Image

Your agent Dockerfile should:

1. Accept input via stdin (JSON)
2. Output response to stdout (JSON)
3. Optionally stream events to stderr (JSONL)

Example Dockerfile:

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agent.py .

# ATP Protocol: stdin → stdout
CMD ["python", "agent.py"]
```

### Running Tests with Docker Adapter

```bash
# Ensure your agent image exists
docker images | grep my-agent

# Run ATP tests
uv run atp test my_suite.yaml --adapter=docker
```

## CLI Adapter with Container Runtime

You can also use the CLI adapter to invoke containerized agents:

```bash
uv run atp test examples/test_suites/demo_file_agent.yaml \
  --adapter=cli \
  --adapter-config='command=docker' \
  --adapter-config='args=["run", "--rm", "-i", "my-agent:latest"]'
```

## Container Runtime Selection

ATP automatically detects the available container runtime. You can also
explicitly specify it:

```yaml
agents:
  - name: "my-agent"
    type: "docker"
    config:
      image: "my-agent:latest"
      runtime: "docker"  # or "podman"
```

## Networking

### Connecting Agent to Test Site

If your agent needs to access the test site from within a container:

```bash
# Create a shared network
docker network create atp-network

# Start test site on the network
docker compose -f tests/fixtures/test_site/docker-compose.yml up -d
docker network connect atp-network atp-test-site

# Run agent on the same network
docker run --rm -i --network atp-network my-agent:latest
```

Inside the agent container, the test site is available at:
- `http://atp-test-site:9876` (container name)
- `http://host.docker.internal:9876` (host machine, Docker Desktop only)

### Podman Networking

For Podman, use pods for networking:

```bash
# Create a pod
podman pod create --name atp-pod -p 9876:9876

# Run test site in the pod
podman run -d --pod atp-pod --name atp-test-site atp-test-site

# Run agent in the same pod (localhost networking)
podman run --rm -i --pod atp-pod my-agent:latest
```

## Resource Limits

### Docker

```yaml
agents:
  - name: "my-agent"
    type: "docker"
    config:
      image: "my-agent:latest"
      memory_limit: "2g"
      cpu_limit: 2.0
```

### Manual Limits

```bash
# Docker
docker run --rm -i --memory=2g --cpus=2 my-agent:latest

# Podman
podman run --rm -i --memory=2g --cpus=2 my-agent:latest
```

## Troubleshooting

### Permission Denied (Podman)

If you see permission errors with Podman:

```bash
# Run in rootless mode (default on most systems)
podman run --userns=keep-id ...

# Or use sudo for rootful mode
sudo podman run ...
```

### Container Not Starting

```bash
# Check container logs
docker logs atp-test-site

# Check if port is already in use
lsof -i :9876
```

### Network Connectivity Issues

```bash
# Verify containers can communicate
docker exec -it atp-test-site curl http://localhost:9876/health

# Check network configuration
docker network inspect atp-network
```

### Image Not Found

```bash
# List available images
docker images

# Pull or build the image
docker build -t my-agent:latest .
```

## Complete Example: Web Search Test

```bash
# 1. Start test site
docker compose -f tests/fixtures/test_site/docker-compose.yml up -d

# 2. Verify test site is running
curl http://localhost:9876/health

# 3. Run web search tests with CLI adapter
uv run atp test examples/test_suites/web_search.yaml \
  --adapter=cli \
  --adapter-config='command=uv' \
  --adapter-config='args=["run", "python", "examples/search_agent/agent.py"]' \
  --adapter-config='environment={"TEST_SITE_URL": "http://localhost:9876"}'

# 4. View results in dashboard
uv run atp dashboard --port 8080
# Open http://localhost:8080

# 5. Clean up
docker compose -f tests/fixtures/test_site/docker-compose.yml down
```

## See Also

- [Quick Start Guide](quickstart.md) - Getting started with ATP
- [Adapter Configuration](../reference/adapters.md) - Detailed adapter options
- [Test Site README](../../tests/fixtures/test_site/README.md) - Test site details
