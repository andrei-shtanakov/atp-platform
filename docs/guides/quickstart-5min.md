# 5-Minute Quickstart

Get your first ATP test running in under 5 minutes.

## Installation

```bash
pip install atp-platform
```

Or with uv (recommended):

```bash
uv add atp-platform
```

## Option A: Quickstart Command

The fastest path — ATP scaffolds everything for you:

```bash
atp quickstart
```

This creates a sample test suite and runs it against a built-in demo agent. No configuration required.

## Option B: Manual Setup

### 1. Create a test suite

Save as `smoke.yaml`:

```yaml
test_suite: "smoke"
version: "1.0"

agents:
  - name: "my-agent"
    type: "cli"
    config:
      command: "python"
      args: ["examples/demo_agent.py"]

tests:
  - id: "test-001"
    name: "Agent responds"
    task:
      description: "Create a file named hello.txt containing 'Hello, World!'"
      expected_artifacts: ["hello.txt"]
    constraints:
      max_steps: 3
      timeout_seconds: 30
    assertions:
      - type: "artifact_exists"
        config:
          path: "hello.txt"
```

### 2. Run with the CLI adapter

```bash
atp test smoke.yaml --adapter=cli
```

### 3. Run with an HTTP adapter

Start your agent on `localhost:8000`, then:

```bash
atp test smoke.yaml --adapter=http \
  --adapter-config='endpoint=http://localhost:8000'
```

## Test Catalog

Browse and run curated test suites from the catalog:

```bash
atp catalog list                    # browse categories
atp catalog list coding             # browse suites in a category
atp catalog info coding/file-operations  # suite details
atp catalog run coding/file-operations --adapter=http --adapter-config endpoint=http://localhost:8000
```

## Dashboard

View results and history in the web UI:

```bash
atp dashboard
# Open http://localhost:8080
```

## Docker

Build and run ATP in a container:

```bash
# Show version
docker compose run atp version

# Run a test suite
docker compose run atp test examples/test_suites/01_smoke_tests.yaml --adapter=cli

# Start the dashboard
docker compose up dashboard
# Open http://localhost:8080
```

Results are written to `./atp-results/` on the host.

## Next Steps

- [Full Quickstart Guide](quickstart.md) — deeper walkthrough with more examples
- [Test Format Reference](../reference/test-format.md) — all YAML fields explained
- [Adapter Configuration](../reference/adapters.md) — HTTP, CLI, Container, LangGraph, CrewAI, and more
- [Installation Guide](installation.md) — extras, workspace setup, CI integration
- [Usage Guide](usage.md) — CLI flags, filtering, baseline comparisons
