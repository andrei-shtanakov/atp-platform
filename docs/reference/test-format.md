# Test Format Reference

Complete reference for ATP test suite YAML format.

## Table of Contents

1. [Test Suite Structure](#test-suite-structure)
2. [Test Suite Fields](#test-suite-fields)
3. [Agent Configuration](#agent-configuration)
4. [Test Definition](#test-definition)
5. [Task Definition](#task-definition)
6. [Constraints](#constraints)
7. [Assertions](#assertions)
8. [Scoring Weights](#scoring-weights)
9. [Variable Substitution](#variable-substitution)
10. [Complete Examples](#complete-examples)

## Test Suite Structure

A test suite YAML file has the following top-level structure:

```yaml
test_suite: string            # Required: Suite name
version: string               # Required: Version (default: "1.0")
description: string           # Optional: Suite description
defaults: TestDefaults        # Optional: Default settings
agents: list[AgentConfig]     # Required: Agent configurations
tests: list[TestDefinition]   # Required: Test definitions (min 1)
```

## Test Suite Fields

### test_suite

**Type:** `string`
**Required:** Yes
**Description:** Unique identifier for the test suite.

**Examples:**
```yaml
test_suite: "smoke_tests"
test_suite: "regression_suite_v2"
test_suite: "api_integration_tests"
```

### version

**Type:** `string`
**Required:** No
**Default:** `"1.0"`
**Description:** Version of the test suite format.

**Examples:**
```yaml
version: "1.0"
version: "2.3"
```

### description

**Type:** `string`
**Required:** No
**Description:** Human-readable description of the test suite purpose.

**Examples:**
```yaml
description: "Smoke tests for basic agent functionality"
description: "Comprehensive regression suite for data processing agents"
```

### defaults

**Type:** `TestDefaults`
**Required:** No
**Description:** Default settings inherited by all tests.

**Structure:**
```yaml
defaults:
  runs_per_test: int          # Number of times to run each test
  timeout_seconds: int        # Default timeout
  constraints: Constraints    # Default constraint values
  scoring: ScoringWeights     # Default scoring weights
```

**Example:**
```yaml
defaults:
  runs_per_test: 3
  timeout_seconds: 300
  constraints:
    max_steps: 20
    allowed_tools: ["file_read", "file_write", "web_search"]
  scoring:
    quality_weight: 0.4
    completeness_weight: 0.3
    efficiency_weight: 0.2
    cost_weight: 0.1
```

## Agent Configuration

**Type:** `AgentConfig`
**Required:** Yes (minimum 1 agent)
**Description:** Defines agents to test.

### Fields

#### name

**Type:** `string`
**Required:** Yes
**Description:** Unique agent identifier.

#### type

**Type:** `string`
**Required:** Yes
**Values:** `"http"`, `"docker"`, `"cli"`, `"langgraph"`, `"crewai"`, `"custom"`
**Description:** Adapter type for the agent.

#### config

**Type:** `dict`
**Required:** Yes
**Description:** Agent-specific configuration. Structure depends on adapter type.

### Examples

#### HTTP Agent

```yaml
agents:
  - name: "api-agent"
    type: "http"
    config:
      endpoint: "https://api.example.com/agent"
      api_key: "${API_KEY}"
      timeout: 30
      headers:
        Content-Type: "application/json"
```

#### Docker Agent

```yaml
agents:
  - name: "docker-agent"
    type: "docker"
    config:
      image: "my-agent:latest"
      environment:
        API_KEY: "${API_KEY}"
      volumes:
        - "./data:/data"
```

#### LangGraph Agent

```yaml
agents:
  - name: "langgraph-agent"
    type: "langgraph"
    config:
      graph_path: "./agents/my_graph.py"
      model: "gpt-4"
      temperature: 0.7
```

## Test Definition

**Type:** `TestDefinition`
**Required:** Yes (minimum 1 test)
**Description:** Defines a single test case.

### Fields

#### id

**Type:** `string`
**Required:** Yes
**Description:** Unique test identifier within the suite.
**Format:** Recommended format is `test-XXX` or `<prefix>-XXX`.

**Examples:**
```yaml
id: "test-001"
id: "smoke-042"
id: "regression-data-processing-001"
```

#### name

**Type:** `string`
**Required:** Yes
**Description:** Human-readable test name.

**Examples:**
```yaml
name: "Basic file creation"
name: "API endpoint integration test"
name: "Multi-step data processing workflow"
```

#### description

**Type:** `string`
**Required:** No
**Description:** Detailed test description.

**Example:**
```yaml
description: >
  This test validates that the agent can successfully create
  a file with specific content and verify its existence.
```

#### tags

**Type:** `list[string]`
**Required:** No
**Default:** `[]`
**Description:** Tags for categorizing and filtering tests.

**Common tags:**
- `smoke` - Quick validation tests
- `regression` - Detailed validation tests
- `integration` - Integration/E2E tests
- `slow` - Long-running tests
- `critical` - Critical functionality tests

**Example:**
```yaml
tags: ["smoke", "critical", "file-operations"]
```

#### task

**Type:** `TaskDefinition`
**Required:** Yes
**Description:** Defines what the agent should do. See [Task Definition](#task-definition).

#### constraints

**Type:** `Constraints`
**Required:** No
**Description:** Execution boundaries. See [Constraints](#constraints).

#### assertions

**Type:** `list[Assertion]`
**Required:** No
**Default:** `[]`
**Description:** Validation checks. See [Assertions](#assertions).

#### scoring

**Type:** `ScoringWeights`
**Required:** No
**Description:** Custom scoring weights for this test. See [Scoring Weights](#scoring-weights).

## Task Definition

Defines what the agent should accomplish.

### Fields

#### description

**Type:** `string`
**Required:** Yes
**Description:** Clear instruction for the agent.

**Best practices:**
- Be specific and actionable
- Include expected outcomes
- Avoid ambiguity

**Examples:**
```yaml
task:
  description: "Create a file named output.txt containing 'Hello, World!'"

task:
  description: >
    Analyze the provided CSV file and generate a JSON summary
    containing row count, column names, and basic statistics.

task:
  description: |
    Search for 'Python testing best practices' and save the
    top 3 results to a file named results.txt. Each result
    should include title, URL, and a brief summary.
```

#### input_data

**Type:** `dict`
**Required:** No
**Description:** Additional data to provide to the agent.

**Examples:**
```yaml
task:
  description: "Process the data"
  input_data:
    csv_path: "data/input.csv"
    output_format: "json"
    columns: ["id", "name", "value"]
```

#### expected_artifacts

**Type:** `list[string]`
**Required:** No
**Description:** Files or outputs the agent should create.

**Examples:**
```yaml
task:
  description: "Generate report"
  expected_artifacts:
    - "report.json"
    - "summary.txt"
    - "charts/revenue.png"
```

## Constraints

Execution boundaries for agent behavior.

### Fields

#### max_steps

**Type:** `int`
**Required:** No
**Description:** Maximum number of reasoning/action steps.

**Example:**
```yaml
constraints:
  max_steps: 10
```

#### max_tokens

**Type:** `int`
**Required:** No
**Description:** Maximum token budget for LLM calls.

**Example:**
```yaml
constraints:
  max_tokens: 50000
```

#### timeout_seconds

**Type:** `int`
**Required:** No
**Default:** `300`
**Description:** Maximum execution time in seconds.

**Example:**
```yaml
constraints:
  timeout_seconds: 600  # 10 minutes
```

#### allowed_tools

**Type:** `list[string]`
**Required:** No
**Description:** Whitelist of tools the agent can use.

**Common tools:**
- `file_read` - Read files
- `file_write` - Write files
- `web_search` - Search the web
- `api_call` - Make API requests
- `python_repl` - Execute Python code
- `shell` - Execute shell commands

**Example:**
```yaml
constraints:
  allowed_tools:
    - "file_read"
    - "file_write"
    - "web_search"
```

#### budget_usd

**Type:** `float`
**Required:** No
**Description:** Maximum cost in USD for API calls.

**Example:**
```yaml
constraints:
  budget_usd: 0.50  # 50 cents max
```

### Complete Constraints Example

```yaml
constraints:
  max_steps: 15
  max_tokens: 100000
  timeout_seconds: 300
  allowed_tools:
    - "file_read"
    - "file_write"
    - "web_search"
  budget_usd: 1.0
```

## Assertions

Validation checks for test results.

### Assertion Structure

```yaml
assertions:
  - type: string      # Assertion type
    config: dict      # Type-specific configuration
```

### Assertion Types

#### artifact_exists

**Description:** Check if a file or artifact was created.

**Config:**
- `path` (string, required): Path to expected artifact

**Example:**
```yaml
assertions:
  - type: "artifact_exists"
    config:
      path: "output.txt"

  - type: "artifact_exists"
    config:
      path: "reports/summary.json"
```

#### behavior

**Description:** Validate agent behavior patterns.

**Config:**
- `check` (string, required): Behavior to validate

**Common checks:**
- `no_repeated_actions` - Agent doesn't repeat same action
- `efficient_tool_use` - Minimal tool calls
- `no_errors` - No error events emitted

**Example:**
```yaml
assertions:
  - type: "behavior"
    config:
      check: "no_repeated_actions"

  - type: "behavior"
    config:
      check: "efficient_tool_use"
```

#### llm_eval

**Description:** Use LLM to evaluate output quality.

**Config:**
- `criteria` (string, required): What to evaluate
- `threshold` (float, required): Minimum score (0.0-1.0)

**Common criteria:**
- `factual_accuracy` - Information is correct
- `completeness` - All requirements met
- `coherence` - Output is well-structured
- `relevance` - Output addresses the task

**Example:**
```yaml
assertions:
  - type: "llm_eval"
    config:
      criteria: "factual_accuracy"
      threshold: 0.8

  - type: "llm_eval"
    config:
      criteria: "completeness"
      threshold: 0.9
```

### Multiple Assertions

```yaml
assertions:
  - type: "artifact_exists"
    config:
      path: "output.json"

  - type: "behavior"
    config:
      check: "no_errors"

  - type: "llm_eval"
    config:
      criteria: "completeness"
      threshold: 0.85

  - type: "llm_eval"
    config:
      criteria: "factual_accuracy"
      threshold: 0.90
```

## Scoring Weights

Define relative importance of evaluation dimensions.

### Fields

All weights must be between 0.0 and 1.0, and should sum to approximately 1.0.

#### quality_weight

**Type:** `float`
**Range:** `0.0` - `1.0`
**Default:** `0.4`
**Description:** Weight for output quality assessment.

#### completeness_weight

**Type:** `float`
**Range:** `0.0` - `1.0`
**Default:** `0.3`
**Description:** Weight for task completion assessment.

#### efficiency_weight

**Type:** `float`
**Range:** `0.0` - `1.0`
**Default:** `0.2`
**Description:** Weight for resource efficiency (steps, tokens).

#### cost_weight

**Type:** `float`
**Range:** `0.0` - `1.0`
**Default:** `0.1`
**Description:** Weight for API cost efficiency.

### Examples

#### Balanced Scoring

```yaml
scoring:
  quality_weight: 0.4
  completeness_weight: 0.3
  efficiency_weight: 0.2
  cost_weight: 0.1
```

#### Quality-Focused

```yaml
scoring:
  quality_weight: 0.6
  completeness_weight: 0.3
  efficiency_weight: 0.1
  cost_weight: 0.0
```

#### Cost-Optimized

```yaml
scoring:
  quality_weight: 0.3
  completeness_weight: 0.2
  efficiency_weight: 0.2
  cost_weight: 0.3
```

## Variable Substitution

Reference environment variables in YAML files.

### Syntax

```yaml
${VARIABLE_NAME}              # Required variable
${VARIABLE_NAME:default}      # Optional with default
```

### Examples

#### Basic Substitution

```yaml
agents:
  - name: "agent"
    config:
      endpoint: "${API_ENDPOINT}"
      api_key: "${API_KEY}"
```

#### With Defaults

```yaml
agents:
  - name: "agent"
    config:
      endpoint: "${API_ENDPOINT:http://localhost:8000}"
      timeout: "${TIMEOUT:30}"
      debug: "${DEBUG:false}"
```

#### Nested Substitution

```yaml
agents:
  - name: "agent"
    config:
      base_url: "${BASE_URL:https://api.example.com}"
      endpoints:
        chat: "${BASE_URL}/chat"
        search: "${BASE_URL}/search"
```

#### In Task Data

```yaml
tests:
  - id: "test-001"
    name: "API test"
    task:
      description: "Test API endpoint"
      input_data:
        endpoint: "${TEST_ENDPOINT}"
        credentials:
          username: "${TEST_USER}"
          password: "${TEST_PASSWORD}"
```

### Loading with Variables

```python
from atp.loader import TestLoader

# Set variables
loader = TestLoader(env={
    "API_ENDPOINT": "https://api.production.com",
    "API_KEY": "secret-key-123",
    "TIMEOUT": "60"
})

# Load suite
suite = loader.load_file("suite.yaml")
```

## Complete Examples

### Minimal Test Suite

```yaml
test_suite: "minimal_suite"
version: "1.0"

agents:
  - name: "test-agent"
    type: "http"
    config:
      endpoint: "http://localhost:8000"

tests:
  - id: "test-001"
    name: "Basic test"
    task:
      description: "Echo 'hello'"
```

### Smoke Test Suite

```yaml
test_suite: "smoke_tests"
version: "1.0"
description: "Quick validation of basic functionality"

defaults:
  runs_per_test: 1
  timeout_seconds: 60
  constraints:
    max_steps: 5

agents:
  - name: "smoke-agent"
    type: "http"
    config:
      endpoint: "${API_ENDPOINT:http://localhost:8000}"
      timeout: 30

tests:
  - id: "smoke-001"
    name: "Agent responds"
    tags: ["smoke", "critical"]
    task:
      description: "Echo 'hello world'"
    assertions:
      - type: "behavior"
        config:
          check: "no_errors"

  - id: "smoke-002"
    name: "Basic file creation"
    tags: ["smoke", "file-ops"]
    task:
      description: "Create file output.txt with content 'test'"
      expected_artifacts: ["output.txt"]
    assertions:
      - type: "artifact_exists"
        config:
          path: "output.txt"

  - id: "smoke-003"
    name: "Simple calculation"
    tags: ["smoke", "math"]
    task:
      description: "Calculate 42 * 58 and save to result.txt"
    assertions:
      - type: "artifact_exists"
        config:
          path: "result.txt"
      - type: "llm_eval"
        config:
          criteria: "factual_accuracy"
          threshold: 1.0
```

### Comprehensive Test Suite

```yaml
test_suite: "comprehensive_suite"
version: "1.0"
description: "Full regression test suite for data processing agent"

defaults:
  runs_per_test: 3
  timeout_seconds: 300
  constraints:
    max_steps: 20
    max_tokens: 100000
    allowed_tools:
      - "file_read"
      - "file_write"
      - "python_repl"
      - "web_search"
    budget_usd: 0.50
  scoring:
    quality_weight: 0.5
    completeness_weight: 0.3
    efficiency_weight: 0.15
    cost_weight: 0.05

agents:
  - name: "data-agent"
    type: "http"
    config:
      endpoint: "${API_ENDPOINT}"
      api_key: "${API_KEY}"
      timeout: 60

  - name: "baseline-agent"
    type: "http"
    config:
      endpoint: "${BASELINE_ENDPOINT}"
      api_key: "${API_KEY}"

tests:
  - id: "test-001"
    name: "CSV data analysis"
    tags: ["regression", "data", "csv"]
    description: "Analyze CSV file and generate summary statistics"
    task:
      description: >
        Read the file data.csv and generate a JSON summary with:
        - Total row count
        - Column names and types
        - Basic statistics (mean, median, std) for numeric columns
        Save results to summary.json
      input_data:
        csv_path: "fixtures/data.csv"
      expected_artifacts:
        - "summary.json"
    constraints:
      max_steps: 10
      timeout_seconds: 120
    assertions:
      - type: "artifact_exists"
        config:
          path: "summary.json"
      - type: "behavior"
        config:
          check: "efficient_tool_use"
      - type: "llm_eval"
        config:
          criteria: "completeness"
          threshold: 0.9
      - type: "llm_eval"
        config:
          criteria: "factual_accuracy"
          threshold: 0.95
    scoring:
      quality_weight: 0.6
      completeness_weight: 0.3
      efficiency_weight: 0.1
      cost_weight: 0.0

  - id: "test-002"
    name: "Multi-file data merge"
    tags: ["regression", "data", "merge"]
    task:
      description: >
        Merge data from users.csv and transactions.csv based on user_id.
        Create merged_data.csv with all columns.
      input_data:
        users_file: "fixtures/users.csv"
        transactions_file: "fixtures/transactions.csv"
      expected_artifacts:
        - "merged_data.csv"
    constraints:
      max_steps: 15
    assertions:
      - type: "artifact_exists"
        config:
          path: "merged_data.csv"
      - type: "llm_eval"
        config:
          criteria: "completeness"
          threshold: 0.85

  - id: "test-003"
    name: "Web research and synthesis"
    tags: ["integration", "web", "research"]
    task:
      description: >
        Research 'Python async programming best practices' and create
        a markdown document with top 5 practices, including sources.
      expected_artifacts:
        - "async_best_practices.md"
    constraints:
      max_steps: 25
      timeout_seconds: 600
      allowed_tools:
        - "web_search"
        - "file_write"
    assertions:
      - type: "artifact_exists"
        config:
          path: "async_best_practices.md"
      - type: "behavior"
        config:
          check: "no_repeated_actions"
      - type: "llm_eval"
        config:
          criteria: "completeness"
          threshold: 0.8
      - type: "llm_eval"
        config:
          criteria: "factual_accuracy"
          threshold: 0.85
```

## Validation Rules

ATP validates test suites with the following rules:

1. **Required fields**: `test_suite`, `version`, `agents`, `tests`
2. **Unique test IDs**: No duplicate `id` values in `tests`
3. **Scoring weights**: Must sum to approximately 1.0 (within 0.01)
4. **Value ranges**: Weights must be 0.0-1.0
5. **Minimum tests**: At least one test required
6. **Minimum agents**: At least one agent required
7. **Variable resolution**: All `${VAR}` without defaults must be provided

## See Also

- [Quick Start Guide](../guides/quickstart.md) - Create your first test suite
- [Usage Guide](../guides/usage.md) - Common workflows
- [Troubleshooting](troubleshooting.md) - Common issues
- [Examples](../../examples/test_suites/) - Complete examples
