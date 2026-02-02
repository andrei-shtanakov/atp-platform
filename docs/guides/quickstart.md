# Quick Start Guide

This guide will walk you through running your first ATP tests in 5 minutes.

## Prerequisites

- ATP Platform installed (see [Installation Guide](installation.md))
- Basic understanding of YAML syntax
- Python 3.12+

## Quick Demo (No API Keys Required)

The fastest way to see ATP in action:

```bash
# Run the demo file agent
uv run atp test examples/test_suites/demo_file_agent.yaml \
  --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["examples/demo_agent.py"]' \
  -v
```

Expected output:
```
Suite: demo_file_agent
Agent: test-agent
Tests: 5, Runs per test: 1

  ✓ Create text file               [0.0s] PASSED
  ✓ Read existing file             [0.0s] PASSED
  ✓ List files in directory        [0.0s] PASSED
  ✓ Handle file not found          [0.0s] PASSED
  ✓ Multi-step: create and verify  [0.0s] PASSED

Result: PASSED
Passed: 5/5 (100.0%)
```

## With OpenAI (Requires API Key)

```bash
export OPENAI_API_KEY='sk-...'

uv run atp test examples/test_suites/openai_agent.yaml \
  --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["examples/openai_agent.py"]' \
  --adapter-config='inherit_environment=true' \
  --adapter-config='allowed_env_vars=["OPENAI_API_KEY","OPENAI_MODEL"]' \
  -v
```

## Step 1: Understanding Test Suites

An ATP test suite is a YAML file that defines:
- **Tests**: What tasks the agent should perform
- **Constraints**: Boundaries for execution (time, steps, tools)
- **Assertions**: How to validate results
- **Agents**: Which agents to test

## Step 2: Create Your First Test Suite

Create a file named `my_first_suite.yaml`:

```yaml
test_suite: "my_first_suite"
version: "1.0"
description: "My first ATP test suite"

defaults:
  runs_per_test: 1
  timeout_seconds: 60
  scoring:
    quality_weight: 0.4
    completeness_weight: 0.3
    efficiency_weight: 0.2
    cost_weight: 0.1

agents:
  - name: "my-agent"
    type: "http"
    config:
      endpoint: "http://localhost:8000"

tests:
  - id: "test-001"
    name: "Simple file creation"
    tags: ["smoke"]
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

## Step 3: Load the Test Suite

Create a Python script `run_suite.py`:

```python
from atp.loader import TestLoader

# Create loader
loader = TestLoader()

# Load test suite
suite = loader.load_file("my_first_suite.yaml")

# Display suite information
print(f"Suite: {suite.test_suite}")
print(f"Description: {suite.description}")
print(f"Version: {suite.version}")
print(f"Number of tests: {len(suite.tests)}")
print(f"Number of agents: {len(suite.agents)}")

# Display test details
for test in suite.tests:
    print(f"\nTest ID: {test.id}")
    print(f"  Name: {test.name}")
    print(f"  Tags: {', '.join(test.tags)}")
    print(f"  Description: {test.task.description}")
    print(f"  Max steps: {test.constraints.max_steps}")
    print(f"  Timeout: {test.constraints.timeout_seconds}s")
    print(f"  Assertions: {len(test.assertions)}")
```

Run the script:

```bash
uv run python run_suite.py
```

Expected output:
```
Suite: my_first_suite
Description: My first ATP test suite
Version: 1.0
Number of tests: 1
Number of agents: 1

Test ID: test-001
  Name: Simple file creation
  Tags: smoke
  Description: Create a file named hello.txt containing 'Hello, World!'
  Max steps: 3
  Timeout: 30s
  Assertions: 1
```

## Step 4: Understanding Test Components

### Task Definition

```yaml
task:
  description: "What the agent should do"
  input_data:
    key: "value"
  expected_artifacts: ["file1.txt", "file2.json"]
```

- **description**: Clear task instruction
- **input_data**: Optional data to provide to agent
- **expected_artifacts**: Files/outputs the agent should create

### Constraints

```yaml
constraints:
  max_steps: 10              # Maximum reasoning/action steps
  max_tokens: 50000          # Token budget
  timeout_seconds: 300       # Time limit
  allowed_tools: ["file_write", "web_search"]
  budget_usd: 1.0           # Cost limit
```

### Assertions

```yaml
assertions:
  - type: "artifact_exists"
    config:
      path: "output.txt"

  - type: "behavior"
    config:
      check: "no_repeated_actions"

  - type: "llm_eval"
    config:
      criteria: "factual_accuracy"
      threshold: 0.8
```

## Step 5: Using Variables

ATP supports environment variable substitution:

```yaml
agents:
  - name: "my-agent"
    type: "http"
    config:
      endpoint: "${API_ENDPOINT:http://localhost:8000}"
      api_key: "${API_KEY}"
```

Set variables before loading:

```python
from atp.loader import TestLoader

# Option 1: Pass env dict
loader = TestLoader(env={
    "API_ENDPOINT": "https://api.example.com",
    "API_KEY": "secret-key-123"
})

# Option 2: Use system environment variables
import os
os.environ["API_KEY"] = "secret-key-123"
loader = TestLoader()  # Will use os.environ

suite = loader.load_file("my_first_suite.yaml")
```

## Step 6: Add Multiple Tests

Expand your suite with more tests:

```yaml
tests:
  - id: "test-001"
    name: "Simple file creation"
    tags: ["smoke", "basic"]
    task:
      description: "Create a file named hello.txt"
    constraints:
      max_steps: 3
    assertions:
      - type: "artifact_exists"
        config:
          path: "hello.txt"

  - id: "test-002"
    name: "Data processing"
    tags: ["regression", "data"]
    task:
      description: "Read data.csv and create summary.json"
      input_data:
        csv_path: "data.csv"
      expected_artifacts: ["summary.json"]
    constraints:
      max_steps: 10
      timeout_seconds: 120
    assertions:
      - type: "artifact_exists"
        config:
          path: "summary.json"
      - type: "llm_eval"
        config:
          criteria: "completeness"
          threshold: 0.9

  - id: "test-003"
    name: "Web research task"
    tags: ["integration", "web"]
    task:
      description: "Search for 'Python testing' and save top 3 results"
      expected_artifacts: ["results.txt"]
    constraints:
      max_steps: 15
      allowed_tools: ["web_search", "file_write"]
      timeout_seconds: 180
    assertions:
      - type: "artifact_exists"
        config:
          path: "results.txt"
      - type: "behavior"
        config:
          check: "efficient_tool_use"
```

## Step 7: Test Suite Best Practices

### Naming Conventions

- **Test IDs**: Use `test-XXX` format (e.g., `test-001`, `test-042`)
- **Suite names**: Use `snake_case` or `kebab-case`
- **Tags**: Use descriptive tags like `smoke`, `regression`, `integration`

### Organizing Tests

```yaml
# Group tests by tags
tests:
  # Smoke tests - quick validation
  - id: "test-001"
    tags: ["smoke", "critical"]

  - id: "test-002"
    tags: ["smoke"]

  # Regression tests - detailed validation
  - id: "test-010"
    tags: ["regression", "data-processing"]

  - id: "test-011"
    tags: ["regression", "api"]

  # Integration tests - full workflows
  - id: "test-100"
    tags: ["integration", "e2e"]
```

### Using Defaults

Define common settings once:

```yaml
defaults:
  runs_per_test: 3          # Run each test 3 times
  timeout_seconds: 300      # 5 minute default timeout

  constraints:
    max_steps: 20
    allowed_tools: ["file_read", "file_write", "web_search"]

  scoring:
    quality_weight: 0.5
    completeness_weight: 0.3
    efficiency_weight: 0.1
    cost_weight: 0.1

tests:
  # This test inherits all defaults
  - id: "test-001"
    name: "Inherits defaults"
    task:
      description: "Do something"

  # This test overrides timeout
  - id: "test-002"
    name: "Custom timeout"
    task:
      description: "Long running task"
    constraints:
      timeout_seconds: 600  # Override default
```

## Step 8: Validation and Error Handling

ATP validates your test suite:

```python
from atp.loader import TestLoader
from atp.core.exceptions import ValidationError

loader = TestLoader()

try:
    suite = loader.load_file("test_suite.yaml")
    print("✓ Suite loaded successfully")
except ValidationError as e:
    print(f"✗ Validation error: {e}")
    # Fix the YAML and try again
```

Common validation errors:
- Duplicate test IDs
- Scoring weights don't sum to ~1.0
- Missing required fields
- Invalid YAML syntax
- Unresolved variables

## Next Steps

Now that you've run your first tests:

1. **Run the demo agents**: Try `examples/demo_agent.py` and `examples/openai_agent.py`
2. **Learn more about test format**: [Test Format Reference](../reference/test-format.md)
3. **Explore examples**: Check `examples/test_suites/` directory
4. **Read usage guide**: [Basic Usage](usage.md)
5. **Configure adapters**: [Adapter Configuration](../reference/adapters.md)

## Common Patterns

### Smoke Test Suite

Quick validation for basic functionality:

```yaml
test_suite: "smoke_tests"
defaults:
  runs_per_test: 1
  timeout_seconds: 60

tests:
  - id: "smoke-001"
    name: "Agent responds"
    tags: ["smoke"]
    task:
      description: "Echo 'hello'"
    constraints:
      max_steps: 1
```

### Regression Test Suite

Comprehensive validation:

```yaml
test_suite: "regression_tests"
defaults:
  runs_per_test: 3
  timeout_seconds: 300

tests:
  - id: "reg-001"
    name: "Data processing accuracy"
    tags: ["regression", "data"]
    # ... detailed test

  - id: "reg-002"
    name: "API integration"
    tags: ["regression", "api"]
    # ... detailed test
```

### Cost Analysis Suite

Track token usage and costs:

```yaml
test_suite: "cost_analysis"
defaults:
  scoring:
    cost_weight: 0.5      # Prioritize cost efficiency
    quality_weight: 0.3
    completeness_weight: 0.2
    efficiency_weight: 0.0

tests:
  - id: "cost-001"
    name: "Expensive task"
    constraints:
      budget_usd: 0.10    # Max 10 cents
      max_tokens: 10000
```

## Web Search Agent Demo

Test a web search agent against a mock e-commerce site:

```bash
# 1. Start the test site (port 9876)
# Using Docker Compose (v2):
docker compose -f tests/fixtures/test_site/docker-compose.yml up -d

# Or using Podman Compose:
podman-compose -f tests/fixtures/test_site/docker-compose.yml up -d

# Or using legacy docker-compose:
docker-compose -f tests/fixtures/test_site/docker-compose.yml up -d

# 2. Verify it's running
curl http://localhost:9876/health
# {"status":"ok","products_count":10}

# 3. Run the web search tests
uv run atp test examples/test_suites/web_search.yaml \
  --adapter=cli \
  --adapter-config='command=uv' \
  --adapter-config='args=["run", "python", "examples/search_agent/agent.py"]' \
  --adapter-config='environment={"TEST_SITE_URL": "http://localhost:9876"}'

# 4. View results in dashboard
uv run atp dashboard --port 8080
# Open http://localhost:8080

# 5. Stop the test site when done
docker compose -f tests/fixtures/test_site/docker-compose.yml down
# Or: podman-compose -f tests/fixtures/test_site/docker-compose.yml down
```

The test suite includes 6 tests:
- Find laptops under $1000
- Find all accessories
- Get company founding year
- Get contact email
- Find expensive laptops
- Multi-page information extraction

## Troubleshooting

See [Troubleshooting Guide](../reference/troubleshooting.md) for common issues and solutions.
